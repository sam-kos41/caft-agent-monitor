"""Instrumented OpenViking client — every viking:// operation emits ObservableEvent.

Wraps ContextStore so that memory loads, stores, tier escalations, searches,
and session lifecycle transitions are all visible to the visualization layer
through the ObservableEvent contract.

The wrapper is thin on purpose: ContextStore does the real work, this layer
just emits events. If ContextStore is unavailable (no OpenViking installed),
the instrumented store still emits events for operations that don't need
persistence (session start/end, memory load attempts that degrade to no-ops).

Usage::

    from agentdiag.context.instrumented import InstrumentedContextStore

    store = InstrumentedContextStore(db_path="./ctx")
    store.on_event = my_event_sink  # receives ObservableEvent

    sid = store.start_session(goal="Fix login bug")
    # emits SESSION_START + MEMORY_LOAD (for baseline priors)

    store.record_event(event, phase, diagnoses, trust_score=0.9)
    # emits MEMORY_STORE when a case is promoted

    store.end_session(dashboard_state)
    # emits MEMORY_STORE (summary) + SESSION_END + retrospective stores
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional

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
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token estimation (cheap heuristic — 1 token ~ 4 chars)
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Rough token count for event metadata. Not exact, not the point."""
    return max(1, len(text) // 4)


def _estimate_tokens_from_cases(cases: list) -> int:
    """Estimate tokens from a list of DiagnosticCase or dict objects."""
    total = 0
    for c in cases:
        if hasattr(c, "to_json"):
            total += _estimate_tokens(c.to_json())
        elif isinstance(c, dict):
            import json
            total += _estimate_tokens(json.dumps(c, default=str))
        else:
            total += 50  # fallback estimate
    return total


# ---------------------------------------------------------------------------
# InstrumentedContextStore
# ---------------------------------------------------------------------------

class InstrumentedContextStore:
    """ContextStore wrapper that emits ObservableEvent for every operation.

    Delegates all real work to the underlying ContextStore. The event sink
    (``on_event``) is a simple callback — the harness orchestrator or a
    test harness sets it. If no sink is set, events are silently dropped.

    The step counter auto-increments so events have monotonic ordering
    even when the caller doesn't track steps (e.g., session-level ops).
    Callers can also pass explicit steps for operations tied to trace events.
    """

    def __init__(
        self,
        db_path: str = "./agentdiag_context",
        on_event: Optional[Callable[[ObservableEvent], None]] = None,
    ):
        self.on_event = on_event
        self._step = 0

        # Try to create underlying ContextStore
        self._store: Optional["ContextStore"] = None
        try:
            from agentdiag.context.openviking import ContextStore
            self._store = ContextStore(db_path=db_path)
        except Exception:
            logger.debug("ContextStore unavailable, instrumented store runs event-only")

    def _next_step(self) -> int:
        self._step += 1
        return self._step

    def _emit(self, event: ObservableEvent) -> None:
        """Send event to sink. Never raises."""
        if self.on_event is not None:
            try:
                self.on_event(event)
            except Exception:
                logger.debug("Event sink raised", exc_info=True)

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def start_session(self, goal: str = "", source: str = "") -> str:
        """Start session, emit SESSION_START, load baseline priors at L0."""
        now = time.time()
        step = self._next_step()

        # Emit session start
        self._emit(ObservableEvent(
            step=step,
            timestamp=now,
            event_type=EventType.SESSION_START,
            metadata={"goal": goal, "source": source},
        ))

        session_id = ""
        if self._store is not None:
            session_id = self._store.start_session(goal=goal, source=source)

        # Load baseline priors — L0 skim of past failure patterns
        fp_rates = self.get_detector_fp_rates()
        if fp_rates:
            step = self._next_step()
            self._emit(memory_load_event(
                step=step,
                timestamp=time.time(),
                uri="viking://agent/evaluator/memories/fp_rates",
                tier=MemoryTier.L0,
                token_count=_estimate_tokens(str(fp_rates)),
                namespace="agent/evaluator/memories",
            ))

        # Load past failure patterns at L0
        patterns = self.get_failure_patterns()
        if patterns:
            step = self._next_step()
            self._emit(memory_load_event(
                step=step,
                timestamp=time.time(),
                uri="viking://agent/evaluator/skills/failure_patterns",
                tier=MemoryTier.L0,
                token_count=_estimate_tokens_from_cases(patterns),
                namespace="agent/evaluator/skills",
            ))

        return session_id

    def record_event(
        self,
        event: "TraceEvent",
        phase: "Phase",
        diagnoses: list,
        trust_score: float = 1.0,
    ) -> None:
        """Buffer event in memory. Emit MEMORY_STORE when cases are promoted."""
        if self._store is not None:
            # Snapshot promoted count before
            before = len(self._store._promoted_cases)
            self._store.record_event(event, phase, diagnoses, trust_score)
            after = len(self._store._promoted_cases)

            # Emit store events for newly promoted cases
            for case in self._store._promoted_cases[before:]:
                step = self._next_step()
                self._emit(memory_store_event(
                    step=step,
                    timestamp=time.time(),
                    uri=f"viking://resources/current_project/cases/{case.failure_name}",
                    token_count=_estimate_tokens(case.to_json()),
                    namespace="resources/current_project/cases",
                ))

    def end_session(self, dashboard_state: "DashboardState") -> dict:
        """Commit session, emit summary store + SESSION_END."""
        now = time.time()
        result: dict = {}

        if self._store is not None:
            # Emit the L1 summary store
            promoted = list(self._store._promoted_cases)
            step = self._next_step()
            self._emit(memory_store_event(
                step=step,
                timestamp=now,
                uri="viking://resources/current_project/session_summary",
                token_count=_estimate_tokens_from_cases(promoted) + 200,
                namespace="resources/current_project",
            ))

            result = self._store.end_session(dashboard_state)

        # Run retrospective — crystallize skills from confirmed cases
        self._retrospective()

        # Session end
        step = self._next_step()
        self._emit(ObservableEvent(
            step=step,
            timestamp=time.time(),
            event_type=EventType.SESSION_END,
            metadata={"committed": bool(result)},
        ))

        return result

    # ------------------------------------------------------------------
    # Retrospective — the self-iteration loop
    # ------------------------------------------------------------------

    def _retrospective(self) -> None:
        """Post-session skill crystallization.

        Reads confirmed and false-positive cases from the ledger, then
        stores distilled patterns as evaluator skills/anti-skills.
        Each operation emits observable events so the visualization layer
        can show the retrospective as a distinct phase.
        """
        if self._store is None:
            return

        confirmed = self._store.load_cases(status_filter="confirmed")
        false_positives = self._store.load_cases(status_filter="false_positive")

        if not confirmed and not false_positives:
            return

        # Store confirmed patterns as evaluator skills
        for case in confirmed:
            name = case.get("failure_name", "unknown")
            step = self._next_step()
            self._emit(memory_store_event(
                step=step,
                timestamp=time.time(),
                uri=f"viking://agent/evaluator/skills/{name}",
                token_count=_estimate_tokens(str(case)),
                namespace="agent/evaluator/skills",
            ))

        # Store FP patterns as anti-skills
        for case in false_positives:
            name = case.get("failure_name", "unknown")
            step = self._next_step()
            self._emit(memory_store_event(
                step=step,
                timestamp=time.time(),
                uri=f"viking://agent/evaluator/memories/false_positives/{name}",
                token_count=_estimate_tokens(str(case)),
                namespace="agent/evaluator/memories",
            ))

    # ------------------------------------------------------------------
    # Search — tier-aware loading with escalation events
    # ------------------------------------------------------------------

    def find_similar_failures(
        self,
        diagnosis: "CaftDiagnosis",
        limit: int = 5,
    ) -> list[dict]:
        """Search for similar past failures. Emits L0 load, escalates to L1 on hit."""
        step = self._next_step()
        uri = f"viking://agent/evaluator/skills/{diagnosis.failure_name}"

        # L0 probe
        self._emit(memory_load_event(
            step=step,
            timestamp=time.time(),
            uri=uri,
            tier=MemoryTier.L0,
            token_count=_estimate_tokens(diagnosis.description),
            namespace="agent/evaluator/skills",
        ))

        results: list[dict] = []
        if self._store is not None:
            results = self._store.find_similar_failures(diagnosis, limit=limit)

        # If we got results, that's an L0 -> L1 escalation
        if results:
            step = self._next_step()
            self._emit(tier_escalation_event(
                step=step,
                timestamp=time.time(),
                uri=uri,
                from_tier=MemoryTier.L0,
                to_tier=MemoryTier.L1,
                token_count=_estimate_tokens_from_cases(results),
            ))

        return results

    def search(self, query: str, limit: int = 10) -> list[dict]:
        """General search. Emits L0 load, escalates on results."""
        step = self._next_step()
        self._emit(memory_load_event(
            step=step,
            timestamp=time.time(),
            uri="viking://search",
            tier=MemoryTier.L0,
            token_count=_estimate_tokens(query),
            namespace="search",
        ))

        results: list[dict] = []
        if self._store is not None:
            results = self._store.search(query, limit=limit)

        if results:
            step = self._next_step()
            self._emit(tier_escalation_event(
                step=step,
                timestamp=time.time(),
                uri="viking://search",
                from_tier=MemoryTier.L0,
                to_tier=MemoryTier.L1,
                token_count=_estimate_tokens_from_cases(results),
            ))

        return results

    def load_context_for_agent(
        self,
        agent_role: AgentRole,
        sprint_number: int = 0,
        full_detail: bool = False,
    ) -> dict:
        """Tiered context loading for a harness agent.

        Loads the right context at the right tier for the requesting agent:
        - Planner: L0 of past sessions + L1 of user preferences
        - Generator: L2 of current sprint contract + L1 of past QA reports
        - Evaluator: L2 of calibration examples + L1 of past evaluations

        Returns a dict of namespace -> content mappings. The actual content
        is opaque to this layer — the harness interprets it.
        """
        now = time.time()
        context: dict = {}
        role_name = agent_role.value

        # Define what each role needs and at what tier
        load_specs: list[tuple[str, MemoryTier]] = []

        if agent_role == AgentRole.PLANNER:
            load_specs = [
                (f"viking://agent/planner/memories", MemoryTier.L1),
                (f"viking://user/memories/preferences", MemoryTier.L1),
                (f"viking://resources/reference_projects", MemoryTier.L0),
            ]
        elif agent_role == AgentRole.GENERATOR:
            load_specs = [
                (f"viking://agent/generator/skills", MemoryTier.L1),
                (f"viking://resources/current_project/contracts/sprint_{sprint_number}", MemoryTier.L2),
            ]
            if sprint_number > 1:
                load_specs.append(
                    (f"viking://resources/current_project/qa_reports/sprint_{sprint_number - 1}", MemoryTier.L1)
                )
        elif agent_role == AgentRole.EVALUATOR:
            tier = MemoryTier.L2 if full_detail else MemoryTier.L1
            load_specs = [
                (f"viking://agent/evaluator/skills", tier),
                (f"viking://agent/evaluator/memories/calibration", MemoryTier.L2),
                (f"viking://resources/current_project/contracts/sprint_{sprint_number}", MemoryTier.L2),
            ]

        for uri, tier in load_specs:
            step = self._next_step()
            namespace = uri.replace("viking://", "").strip("/")
            namespace = "/".join(namespace.split("/")[:3])

            # Search OpenViking for this namespace's content
            search_results: list[dict] = []
            if self._store is not None:
                search_results = self._store.search(
                    query=f"namespace:{namespace}", limit=5,
                )

            token_count = _estimate_tokens_from_cases(search_results) if search_results else 0
            self._emit(memory_load_event(
                step=step,
                timestamp=time.time(),
                uri=uri,
                tier=tier,
                token_count=token_count,
                namespace=namespace,
            ))

            context[uri] = search_results

        return context

    # ------------------------------------------------------------------
    # Passthrough — feedback loop methods
    # ------------------------------------------------------------------

    def get_detector_fp_rates(self) -> dict[str, float]:
        if self._store is not None:
            return self._store.get_detector_fp_rates()
        return {}

    def get_failure_patterns(self) -> list[dict]:
        if self._store is not None:
            return self._store.get_failure_patterns()
        return []

    def adjust_diagnosis_confidence(self, diagnosis: "CaftDiagnosis") -> "CaftDiagnosis":
        if self._store is not None:
            return self._store.adjust_diagnosis_confidence(diagnosis)
        return diagnosis

    def record_confirmation(self, candidate: "CaftDiagnosis", result: "ConfirmationResult") -> None:
        if self._store is not None:
            self._store.record_confirmation(candidate, result)

            # Emit store event for the confirmation record
            step = self._next_step()
            self._emit(memory_store_event(
                step=step,
                timestamp=time.time(),
                uri=f"viking://resources/current_project/confirmations/{candidate.failure_name}",
                token_count=_estimate_tokens(result.reasoning if hasattr(result, 'reasoning') else ""),
                namespace="resources/current_project/confirmations",
            ))

    def load_cases(self, status_filter: Optional[str] = None) -> list[dict]:
        if self._store is not None:
            return self._store.load_cases(status_filter=status_filter)
        return []

    def update_case_status(self, case_id: str, new_status: str, reviewer: str = "human", notes: str = "") -> bool:
        if self._store is not None:
            return self._store.update_case_status(case_id, new_status, reviewer, notes)
        return False

    def get_feedback_summary(self) -> dict:
        if self._store is not None:
            return self._store.get_feedback_summary()
        return {}

    def get_stats(self) -> dict:
        if self._store is not None:
            return self._store.get_stats()
        return {"db_path": "", "healthy": False, "session_count": 0}

    def close(self) -> None:
        if self._store is not None:
            self._store.close()
