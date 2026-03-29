"""Three-agent harness orchestrator with OpenViking-backed persistent memory.

Implements the Anthropic harness pattern (planner -> generator -> evaluator)
on top of OpenViking's tiered context filesystem. Every orchestration action
emits ObservableEvent so the visualization layer can render the full
information processing pipeline.

Architecture::

    HarnessOrchestrator
        |
        +-- InstrumentedContextStore (viking:// memory)
        |
        +-- PlannerAgent    -> sprint specs
        +-- GeneratorAgent  -> code/artifacts
        +-- EvaluatorAgent  -> grades + critiques
        |
        +-- SprintContract  -> negotiated between generator & evaluator

The orchestrator owns the lifecycle:
  1. PLANNING       — planner decomposes goal into sprint specs
  2. CONTRACT       — generator + evaluator negotiate success criteria
  3. EXECUTING      — generator works against the contract
  4. VERIFYING      — evaluator grades the output
  5. ITERATING      — if grade < threshold, loop back to EXECUTING
  6. RETROSPECTIVE  — crystallize skills from confirmed outcomes

All phases emit PHASE_BOUNDARY events. All memory operations emit
MEMORY_LOAD / MEMORY_STORE / MEMORY_TIER_ESCALATION. All evaluator
grades emit EVALUATION_RESULT. Contract negotiations emit CONTRACT_EVENT.

The orchestrator is agent-SDK-agnostic: the actual agent implementations
are injected as callables. This module provides the orchestration skeleton
and the event emission — the caller provides the brains.

Usage::

    from agentdiag.harness import HarnessOrchestrator, SprintContract

    orch = HarnessOrchestrator(
        context_store=instrumented_store,
        planner=my_planner_fn,
        generator=my_generator_fn,
        evaluator=my_evaluator_fn,
        on_event=event_sink,
    )
    result = orch.run(goal="Build a login page", max_sprints=3)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Optional, Protocol

from agentdiag.observable import (
    ObservableEvent,
    EventType,
    MemoryTier,
    HarnessPhase,
    AgentRole,
    memory_load_event,
    memory_store_event,
    tier_escalation_event,
    phase_boundary_event,
    evaluation_event,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sprint Contract — the negotiated agreement between generator and evaluator
# ---------------------------------------------------------------------------

@dataclass
class SprintContract:
    """Negotiated success criteria for a single sprint.

    The generator proposes what it will build. The evaluator amends
    the criteria to be testable. Both agree before execution starts.

    Stored in viking://resources/current_project/contracts/sprint_N_{status}
    """
    sprint_number: int
    goal: str
    deliverables: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)
    generator_notes: str = ""
    evaluator_amendments: str = ""
    status: str = "proposed"  # proposed | amended | accepted | rejected
    max_iterations: int = 3

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvaluationGrade:
    """Evaluator's assessment of a sprint's output."""
    sprint_number: int
    overall_score: float          # 0.0 - 1.0
    criteria_scores: dict[str, float] = field(default_factory=dict)
    passed: bool = False
    critique: str = ""
    suggestions: list[str] = field(default_factory=list)
    iteration: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SprintResult:
    """Complete outcome of a single sprint."""
    sprint_number: int
    contract: SprintContract
    grades: list[EvaluationGrade] = field(default_factory=list)
    iterations: int = 0
    artifacts: dict[str, Any] = field(default_factory=dict)
    final_passed: bool = False


@dataclass
class HarnessResult:
    """Complete outcome of a full harness run."""
    goal: str
    sprints: list[SprintResult] = field(default_factory=list)
    total_iterations: int = 0
    overall_passed: bool = False
    duration_sec: float = 0.0


# ---------------------------------------------------------------------------
# Agent protocols — the caller injects these
# ---------------------------------------------------------------------------

class PlannerFn(Protocol):
    """Planner: goal + context -> list of sprint specs."""
    def __call__(
        self,
        goal: str,
        context: dict,
    ) -> list[dict]:
        """Returns list of sprint spec dicts with 'goal' and 'deliverables' keys."""
        ...


class GeneratorFn(Protocol):
    """Generator: contract + context -> artifacts dict."""
    def __call__(
        self,
        contract: SprintContract,
        context: dict,
        feedback: Optional[EvaluationGrade] = None,
    ) -> dict:
        """Returns artifacts dict. Keys are up to the implementation."""
        ...


class EvaluatorFn(Protocol):
    """Evaluator: contract + artifacts + context -> grade."""
    def __call__(
        self,
        contract: SprintContract,
        artifacts: dict,
        context: dict,
    ) -> EvaluationGrade:
        """Returns an evaluation grade."""
        ...


class ContractNegotiatorFn(Protocol):
    """Optional: evaluator reviews a proposed contract and amends it."""
    def __call__(
        self,
        contract: SprintContract,
        evaluator_context: dict,
    ) -> SprintContract:
        """Returns amended contract (may change success_criteria, status)."""
        ...


# ---------------------------------------------------------------------------
# HarnessOrchestrator
# ---------------------------------------------------------------------------

class HarnessOrchestrator:
    """Three-agent orchestrator emitting ObservableEvent at every stage.

    The orchestrator coordinates the planner, generator, and evaluator
    through sprint cycles. It doesn't know what the agents do internally —
    it only manages the lifecycle and emits events for observability.

    The InstrumentedContextStore handles memory-level events. This class
    handles harness-level events: phase boundaries, contract negotiations,
    evaluation results, and agent handoffs.
    """

    DEFAULT_PASS_THRESHOLD = 0.7

    def __init__(
        self,
        context_store: "InstrumentedContextStore",
        planner: PlannerFn,
        generator: GeneratorFn,
        evaluator: EvaluatorFn,
        contract_negotiator: Optional[ContractNegotiatorFn] = None,
        on_event: Optional[Callable[[ObservableEvent], None]] = None,
        pass_threshold: float = DEFAULT_PASS_THRESHOLD,
    ):
        self._ctx = context_store
        self._planner = planner
        self._generator = generator
        self._evaluator = evaluator
        self._negotiator = contract_negotiator
        self.on_event = on_event
        self._pass_threshold = pass_threshold
        self._step = 0

        # Track step ranges per sprint for evaluation metadata
        self._sprint_start_step: int = 0

    def _next_step(self) -> int:
        self._step += 1
        return self._step

    def _emit(self, event: ObservableEvent) -> None:
        if self.on_event is not None:
            try:
                self.on_event(event)
            except Exception:
                logger.debug("Event sink raised", exc_info=True)

    def _emit_phase(
        self,
        phase: HarnessPhase,
        previous: Optional[HarnessPhase] = None,
        sprint: Optional[int] = None,
        role: Optional[AgentRole] = None,
    ) -> None:
        """Emit PHASE_BOUNDARY. Must be called BEFORE the first event in the new phase."""
        self._emit(phase_boundary_event(
            step=self._next_step(),
            timestamp=time.time(),
            phase=phase,
            previous_phase=previous,
            sprint_number=sprint,
            agent_role=role,
        ))

    def _emit_handoff(self, to_role: AgentRole, sprint: Optional[int] = None) -> None:
        self._emit(ObservableEvent(
            step=self._next_step(),
            timestamp=time.time(),
            event_type=EventType.AGENT_HANDOFF,
            agent_role=to_role,
            sprint_number=sprint,
        ))

    def _emit_contract(
        self,
        contract: SprintContract,
        agent_role: AgentRole = AgentRole.ORCHESTRATOR,
    ) -> None:
        """Emit CONTRACT_EVENT with agent_role indicating who produced this state."""
        self._emit(ObservableEvent(
            step=self._next_step(),
            timestamp=time.time(),
            event_type=EventType.CONTRACT_EVENT,
            contract_status=contract.status,
            sprint_number=contract.sprint_number,
            agent_role=agent_role,
            metadata={
                "deliverables": len(contract.deliverables),
                "criteria": len(contract.success_criteria),
                "goal": contract.goal,
            },
        ))

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(
        self,
        goal: str,
        max_sprints: int = 5,
        max_iterations_per_sprint: int = 3,
    ) -> HarnessResult:
        """Execute the full harness lifecycle.

        1. Start session + load baseline context
        2. Planner decomposes goal into sprints
        3. For each sprint: negotiate -> execute -> evaluate -> iterate
        4. Retrospective + end session
        """
        start_time = time.time()
        result = HarnessResult(goal=goal)
        current_phase: Optional[HarnessPhase] = None

        # --- Session start ---
        session_id = self._ctx.start_session(goal=goal, source="harness")

        # --- Planning phase (boundary emitted BEFORE planner runs) ---
        self._emit_phase(
            HarnessPhase.PLANNING,
            previous=current_phase,
            role=AgentRole.PLANNER,
        )
        current_phase = HarnessPhase.PLANNING
        self._emit_handoff(AgentRole.PLANNER)

        planner_context = self._ctx.load_context_for_agent(AgentRole.PLANNER)
        sprint_specs = self._planner(goal, planner_context)

        # Cap sprints
        sprint_specs = sprint_specs[:max_sprints]

        # --- Sprint loop ---
        for i, spec in enumerate(sprint_specs, start=1):
            sprint_result = self._run_sprint(
                sprint_number=i,
                spec=spec,
                max_iterations=max_iterations_per_sprint,
                previous_phase=current_phase,
            )
            result.sprints.append(sprint_result)
            result.total_iterations += sprint_result.iterations
            current_phase = HarnessPhase.VERIFYING

        # --- Retrospective phase ---
        self._emit_phase(
            HarnessPhase.RETROSPECTIVE,
            previous=current_phase,
            role=AgentRole.ORCHESTRATOR,
        )

        # Run harness-level retrospective (distill sprint outcomes)
        self._harness_retrospective(result)

        # End session triggers context-level retrospective in InstrumentedContextStore
        try:
            from agentdiag.monitor import DashboardState
            dashboard = DashboardState(
                total_events=self._step,
                trust_score=self._compute_overall_trust(result),
            )
            self._ctx.end_session(dashboard)
        except Exception:
            logger.debug("Could not build DashboardState for end_session", exc_info=True)

        result.overall_passed = all(s.final_passed for s in result.sprints)
        result.duration_sec = time.time() - start_time
        return result

    # ------------------------------------------------------------------
    # Single sprint lifecycle
    # ------------------------------------------------------------------

    def _run_sprint(
        self,
        sprint_number: int,
        spec: dict,
        max_iterations: int,
        previous_phase: Optional[HarnessPhase],
    ) -> SprintResult:
        """Run one sprint: negotiate -> execute -> evaluate -> iterate."""

        # Build initial contract from planner spec
        contract = SprintContract(
            sprint_number=sprint_number,
            goal=spec.get("goal", ""),
            deliverables=spec.get("deliverables", []),
            success_criteria=spec.get("success_criteria", []),
            status="proposed",
            max_iterations=max_iterations,
        )

        # --- Contract negotiation (boundary BEFORE first negotiation event) ---
        self._emit_phase(
            HarnessPhase.CONTRACT_NEGOTIATION,
            previous=previous_phase,
            sprint=sprint_number,
            role=AgentRole.ORCHESTRATOR,
        )
        contract = self._negotiate_contract(contract)

        sprint_result = SprintResult(
            sprint_number=sprint_number,
            contract=contract,
        )

        feedback: Optional[EvaluationGrade] = None

        for iteration in range(1, max_iterations + 1):
            sprint_result.iterations = iteration

            # --- Execution (boundary BEFORE generator runs) ---
            self._emit_phase(
                HarnessPhase.EXECUTING,
                previous=HarnessPhase.CONTRACT_NEGOTIATION if iteration == 1 else HarnessPhase.ITERATING,
                sprint=sprint_number,
                role=AgentRole.GENERATOR,
            )
            self._emit_handoff(AgentRole.GENERATOR, sprint=sprint_number)

            # Record step range start for this sprint iteration
            self._sprint_start_step = self._step

            gen_context = self._ctx.load_context_for_agent(
                AgentRole.GENERATOR, sprint_number=sprint_number,
            )
            artifacts = self._generator(contract, gen_context, feedback)
            sprint_result.artifacts = artifacts

            # --- Verification (boundary BEFORE evaluator runs) ---
            self._emit_phase(
                HarnessPhase.VERIFYING,
                previous=HarnessPhase.EXECUTING,
                sprint=sprint_number,
                role=AgentRole.EVALUATOR,
            )
            self._emit_handoff(AgentRole.EVALUATOR, sprint=sprint_number)

            eval_context = self._ctx.load_context_for_agent(
                AgentRole.EVALUATOR, sprint_number=sprint_number,
            )
            grade = self._evaluator(contract, artifacts, eval_context)
            grade.iteration = iteration
            sprint_result.grades.append(grade)

            # Step range for this sprint iteration
            step_range_end = self._step
            eval_metadata = {
                "step_range": (self._sprint_start_step, step_range_end),
                "iteration": iteration,
            }

            # Emit evaluation result for each criterion with agent_role + step_range
            for criterion, score in grade.criteria_scores.items():
                self._emit(ObservableEvent(
                    step=self._next_step(),
                    timestamp=time.time(),
                    event_type=EventType.EVALUATION_RESULT,
                    evaluation_score=score,
                    evaluation_criterion=criterion,
                    sprint_number=sprint_number,
                    agent_role=AgentRole.EVALUATOR,
                    metadata=eval_metadata,
                ))

            # Emit overall evaluation with step_range
            self._emit(ObservableEvent(
                step=self._next_step(),
                timestamp=time.time(),
                event_type=EventType.EVALUATION_RESULT,
                evaluation_score=grade.overall_score,
                evaluation_criterion="overall",
                sprint_number=sprint_number,
                agent_role=AgentRole.EVALUATOR,
                metadata={
                    **eval_metadata,
                    "passed": grade.overall_score >= self._pass_threshold,
                    "critique": grade.critique,
                },
            ))

            # Store QA report in context
            self._store_qa_report(sprint_number, grade)

            # --- Pass or iterate ---
            if grade.overall_score >= self._pass_threshold:
                sprint_result.final_passed = True
                break

            if iteration < max_iterations:
                # GAN-style iteration: feed critique back to generator
                self._emit_phase(
                    HarnessPhase.ITERATING,
                    previous=HarnessPhase.VERIFYING,
                    sprint=sprint_number,
                    role=AgentRole.ORCHESTRATOR,
                )
                feedback = grade
                logger.debug(
                    "Sprint %d iteration %d: score %.2f < %.2f, iterating",
                    sprint_number, iteration, grade.overall_score, self._pass_threshold,
                )

        return sprint_result

    # ------------------------------------------------------------------
    # Contract negotiation — stores full trail
    # ------------------------------------------------------------------

    def _negotiate_contract(self, contract: SprintContract) -> SprintContract:
        """Generator proposes, evaluator amends, both agree.

        Emits CONTRACT_EVENT at each stage and stores each version in
        viking://resources/current_project/contracts/ for retrospective
        analysis and future negotiation reference.
        """
        n = contract.sprint_number

        # 1. Generator's proposal
        self._emit_contract(contract, agent_role=AgentRole.GENERATOR)
        self._store_contract_version(contract, "proposal")

        # 2. Evaluator amends (if negotiator provided)
        if self._negotiator is not None:
            eval_context = self._ctx.load_context_for_agent(
                AgentRole.EVALUATOR,
                sprint_number=n,
            )
            contract = self._negotiator(contract, eval_context)

            if contract.status == "amended":
                self._emit_contract(contract, agent_role=AgentRole.EVALUATOR)
                self._store_contract_version(contract, "amended")

        # 3. Accept the contract
        contract.status = "accepted"
        self._emit_contract(contract, agent_role=AgentRole.ORCHESTRATOR)
        self._store_contract_version(contract, "accepted")

        return contract

    def _store_contract_version(self, contract: SprintContract, version: str) -> None:
        """Store a contract version in viking://resources/current_project/contracts/."""
        n = contract.sprint_number
        step = self._ctx._next_step()
        self._ctx._emit(memory_store_event(
            step=step,
            timestamp=time.time(),
            uri=f"viking://resources/current_project/contracts/sprint_{n}_{version}",
            token_count=len(str(contract.to_dict())) // 4,
            namespace="resources/current_project/contracts",
        ))

    # ------------------------------------------------------------------
    # QA report storage
    # ------------------------------------------------------------------

    def _store_qa_report(self, sprint_number: int, grade: EvaluationGrade) -> None:
        """Store evaluation grade as a QA report in context."""
        step = self._ctx._next_step()
        self._ctx._emit(memory_store_event(
            step=step,
            timestamp=time.time(),
            uri=f"viking://resources/current_project/qa_reports/sprint_{sprint_number}",
            token_count=len(str(grade.to_dict())) // 4,
            namespace="resources/current_project/qa_reports",
        ))

    # ------------------------------------------------------------------
    # Harness-level retrospective — distill sprint outcomes into skills
    # ------------------------------------------------------------------

    def _harness_retrospective(self, result: HarnessResult) -> None:
        """Distill sprint outcomes into agent-specific skills and memories.

        This is the self-iteration loop that makes run N+1 better than run N.
        Writes to three paths:
        - agent/generator/memories/bug_patterns/ — what went wrong
        - agent/evaluator/skills/effective_tests/ — what evaluation criteria caught bugs
        - agent/generator/skills/design_patterns/ — what approaches passed QA
        """
        for sprint in result.sprints:
            n = sprint.sprint_number

            # Find grades that failed (bug patterns for the generator to learn from)
            failed_grades = [g for g in sprint.grades if g.overall_score < self._pass_threshold]
            for grade in failed_grades:
                # Bug patterns: what the generator got wrong
                if grade.critique:
                    step = self._ctx._next_step()
                    self._ctx._emit(memory_store_event(
                        step=step,
                        timestamp=time.time(),
                        uri=f"viking://agent/generator/memories/bug_patterns/sprint_{n}_iter_{grade.iteration}",
                        token_count=len(grade.critique) // 4,
                        namespace="agent/generator/memories",
                    ))

                # Effective tests: which criteria caught the issue
                failed_criteria = [c for c, s in grade.criteria_scores.items() if s < self._pass_threshold]
                for criterion in failed_criteria:
                    step = self._ctx._next_step()
                    self._ctx._emit(memory_store_event(
                        step=step,
                        timestamp=time.time(),
                        uri=f"viking://agent/evaluator/skills/effective_tests/{criterion}",
                        token_count=50,
                        namespace="agent/evaluator/skills",
                    ))

            # Design patterns: if the sprint ultimately passed, the final approach worked
            if sprint.final_passed and sprint.grades:
                step = self._ctx._next_step()
                self._ctx._emit(memory_store_event(
                    step=step,
                    timestamp=time.time(),
                    uri=f"viking://agent/generator/skills/design_patterns/sprint_{n}",
                    token_count=len(str(sprint.artifacts)) // 4 if sprint.artifacts else 50,
                    namespace="agent/generator/skills",
                ))

    # ------------------------------------------------------------------
    # Trust computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_overall_trust(result: HarnessResult) -> float:
        """Derive a trust score from sprint outcomes."""
        if not result.sprints:
            return 1.0
        passed = sum(1 for s in result.sprints if s.final_passed)
        return round(passed / len(result.sprints), 3)
