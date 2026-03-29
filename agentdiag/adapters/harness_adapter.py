"""Harness log replay adapter — reconstructs ObservableEvent from harness output.

The HarnessOrchestrator emits events in real-time during live runs.
This adapter replays serialized harness results (HarnessResult.to_dict())
for offline analysis, producing the same ObservableEvent stream that
would have been emitted during the original run.

Useful for:
- Comparing harness runs across time (did accumulated skills help?)
- Feeding past runs through updated visualization/analysis code
- Unit testing the full pipeline without running actual agents

Usage::

    from agentdiag.adapters.harness_adapter import HarnessLogAdapter

    adapter = HarnessLogAdapter()
    events = adapter.replay(harness_result_dict)
    for event in events:
        monitor.push(event)
"""

from __future__ import annotations

import json
import logging
from typing import Iterator, Optional

from agentdiag.observable import (
    ObservableEvent,
    EventType,
    MemoryTier,
    HarnessPhase,
    AgentRole,
    memory_store_event,
    phase_boundary_event,
    evaluation_event,
)

logger = logging.getLogger(__name__)


class HarnessLogAdapter:
    """Replays a serialized HarnessResult as an ObservableEvent stream.

    Reconstructs the phase boundaries, contract events, evaluation results,
    handoffs, and retrospective stores from a HarnessResult dict (as
    produced by HarnessResult.to_dict() or loaded from JSON).
    """

    def replay(
        self,
        result: dict,
        base_timestamp: float = 0.0,
    ) -> list[ObservableEvent]:
        """Replay a full harness result as events.

        Args:
            result: Serialized HarnessResult dict with 'goal', 'sprints', etc.
            base_timestamp: Starting timestamp for the reconstructed events.
                Defaults to 0.0 (relative time). Set to a real timestamp
                to align with other event sources.

        Returns:
            Chronologically ordered list of ObservableEvent.
        """
        events: list[ObservableEvent] = []
        step = 0
        t = base_timestamp

        goal = result.get("goal", "")

        # Session start
        step += 1
        t += 0.1
        events.append(ObservableEvent(
            step=step,
            timestamp=t,
            event_type=EventType.SESSION_START,
            metadata={"goal": goal, "source": "harness_replay"},
        ))

        # Planning phase
        step += 1
        t += 0.1
        events.append(phase_boundary_event(
            step=step,
            timestamp=t,
            phase=HarnessPhase.PLANNING,
            previous_phase=None,
            agent_role=AgentRole.PLANNER,
        ))

        # Planner handoff
        step += 1
        t += 0.1
        events.append(ObservableEvent(
            step=step,
            timestamp=t,
            event_type=EventType.AGENT_HANDOFF,
            agent_role=AgentRole.PLANNER,
        ))

        # Sprint loop
        sprints = result.get("sprints", [])
        previous_phase: Optional[HarnessPhase] = HarnessPhase.PLANNING

        for sprint in sprints:
            sprint_n = sprint.get("sprint_number", 0)
            contract = sprint.get("contract", {})
            grades = sprint.get("grades", [])
            iterations = sprint.get("iterations", 0)

            # Contract negotiation phase
            step += 1
            t += 0.2
            events.append(phase_boundary_event(
                step=step,
                timestamp=t,
                phase=HarnessPhase.CONTRACT_NEGOTIATION,
                previous_phase=previous_phase,
                sprint_number=sprint_n,
                agent_role=AgentRole.ORCHESTRATOR,
            ))

            # Contract proposal
            step += 1
            t += 0.1
            events.append(ObservableEvent(
                step=step,
                timestamp=t,
                event_type=EventType.CONTRACT_EVENT,
                contract_status="proposed",
                sprint_number=sprint_n,
                agent_role=AgentRole.GENERATOR,
                metadata={
                    "deliverables": len(contract.get("deliverables", [])),
                    "criteria": len(contract.get("success_criteria", [])),
                    "goal": contract.get("goal", ""),
                },
            ))

            # Contract accepted
            step += 1
            t += 0.1
            events.append(ObservableEvent(
                step=step,
                timestamp=t,
                event_type=EventType.CONTRACT_EVENT,
                contract_status="accepted",
                sprint_number=sprint_n,
                agent_role=AgentRole.ORCHESTRATOR,
                metadata={
                    "deliverables": len(contract.get("deliverables", [])),
                    "criteria": len(contract.get("success_criteria", [])),
                },
            ))

            # Contract stored
            step += 1
            t += 0.05
            events.append(memory_store_event(
                step=step,
                timestamp=t,
                uri=f"viking://resources/current_project/contracts/sprint_{sprint_n}_accepted",
                token_count=len(str(contract)) // 4,
                namespace="resources/current_project/contracts",
            ))

            # Iteration loop (execution + verification per grade)
            exec_start_step = step
            for i, grade in enumerate(grades, start=1):
                iteration_phase = HarnessPhase.CONTRACT_NEGOTIATION if i == 1 else HarnessPhase.ITERATING

                # Execution phase
                step += 1
                t += 0.3
                events.append(phase_boundary_event(
                    step=step,
                    timestamp=t,
                    phase=HarnessPhase.EXECUTING,
                    previous_phase=iteration_phase,
                    sprint_number=sprint_n,
                    agent_role=AgentRole.GENERATOR,
                ))

                step += 1
                t += 0.1
                events.append(ObservableEvent(
                    step=step,
                    timestamp=t,
                    event_type=EventType.AGENT_HANDOFF,
                    agent_role=AgentRole.GENERATOR,
                    sprint_number=sprint_n,
                ))

                # Verification phase
                step += 1
                t += 0.5
                events.append(phase_boundary_event(
                    step=step,
                    timestamp=t,
                    phase=HarnessPhase.VERIFYING,
                    previous_phase=HarnessPhase.EXECUTING,
                    sprint_number=sprint_n,
                    agent_role=AgentRole.EVALUATOR,
                ))

                step += 1
                t += 0.1
                events.append(ObservableEvent(
                    step=step,
                    timestamp=t,
                    event_type=EventType.AGENT_HANDOFF,
                    agent_role=AgentRole.EVALUATOR,
                    sprint_number=sprint_n,
                ))

                # Evaluation results with step_range
                eval_metadata = {
                    "step_range": (exec_start_step, step),
                    "iteration": grade.get("iteration", i),
                }

                criteria_scores = grade.get("criteria_scores", {})
                for criterion, score in criteria_scores.items():
                    step += 1
                    t += 0.05
                    events.append(ObservableEvent(
                        step=step,
                        timestamp=t,
                        event_type=EventType.EVALUATION_RESULT,
                        evaluation_score=score,
                        evaluation_criterion=criterion,
                        sprint_number=sprint_n,
                        agent_role=AgentRole.EVALUATOR,
                        metadata=eval_metadata,
                    ))

                # Overall evaluation
                overall = grade.get("overall_score", 0.0)
                step += 1
                t += 0.05
                events.append(ObservableEvent(
                    step=step,
                    timestamp=t,
                    event_type=EventType.EVALUATION_RESULT,
                    evaluation_score=overall,
                    evaluation_criterion="overall",
                    sprint_number=sprint_n,
                    agent_role=AgentRole.EVALUATOR,
                    metadata={
                        **eval_metadata,
                        "passed": overall >= 0.7,
                        "critique": grade.get("critique", ""),
                    },
                ))

                # QA report stored
                step += 1
                t += 0.05
                events.append(memory_store_event(
                    step=step,
                    timestamp=t,
                    uri=f"viking://resources/current_project/qa_reports/sprint_{sprint_n}",
                    token_count=len(str(grade)) // 4,
                    namespace="resources/current_project/qa_reports",
                ))

                # If failed and not last iteration, emit ITERATING
                if overall < 0.7 and i < len(grades):
                    step += 1
                    t += 0.1
                    events.append(phase_boundary_event(
                        step=step,
                        timestamp=t,
                        phase=HarnessPhase.ITERATING,
                        previous_phase=HarnessPhase.VERIFYING,
                        sprint_number=sprint_n,
                        agent_role=AgentRole.ORCHESTRATOR,
                    ))

                exec_start_step = step

            previous_phase = HarnessPhase.VERIFYING

        # Retrospective phase
        step += 1
        t += 0.2
        events.append(phase_boundary_event(
            step=step,
            timestamp=t,
            phase=HarnessPhase.RETROSPECTIVE,
            previous_phase=previous_phase,
            agent_role=AgentRole.ORCHESTRATOR,
        ))

        # Reconstruct retrospective stores from sprint outcomes
        for sprint in sprints:
            sprint_n = sprint.get("sprint_number", 0)
            grades = sprint.get("grades", [])
            final_passed = sprint.get("final_passed", False)

            for grade in grades:
                overall = grade.get("overall_score", 0.0)
                if overall < 0.7 and grade.get("critique"):
                    step += 1
                    t += 0.05
                    events.append(memory_store_event(
                        step=step,
                        timestamp=t,
                        uri=f"viking://agent/generator/memories/bug_patterns/sprint_{sprint_n}",
                        token_count=len(grade.get("critique", "")) // 4,
                        namespace="agent/generator/memories",
                    ))

                    for criterion, score in grade.get("criteria_scores", {}).items():
                        if score < 0.7:
                            step += 1
                            t += 0.02
                            events.append(memory_store_event(
                                step=step,
                                timestamp=t,
                                uri=f"viking://agent/evaluator/skills/effective_tests/{criterion}",
                                token_count=50,
                                namespace="agent/evaluator/skills",
                            ))

            if final_passed:
                step += 1
                t += 0.05
                events.append(memory_store_event(
                    step=step,
                    timestamp=t,
                    uri=f"viking://agent/generator/skills/design_patterns/sprint_{sprint_n}",
                    token_count=50,
                    namespace="agent/generator/skills",
                ))

        # Session end
        step += 1
        t += 0.1
        events.append(ObservableEvent(
            step=step,
            timestamp=t,
            event_type=EventType.SESSION_END,
            metadata={
                "total_sprints": len(sprints),
                "overall_passed": result.get("overall_passed", False),
                "duration_sec": result.get("duration_sec", 0.0),
            },
        ))

        return events

    def replay_iter(
        self,
        result: dict,
        base_timestamp: float = 0.0,
    ) -> Iterator[ObservableEvent]:
        """Iterator version."""
        yield from self.replay(result, base_timestamp)

    @staticmethod
    def from_json_file(path: str) -> dict:
        """Load a HarnessResult dict from a JSON file."""
        with open(path, encoding="utf-8") as f:
            return json.load(f)
