"""Real-time monitor engine.

Wires together the full pipeline:
  stdin JSONL → adapter auto-parse → HTA state machine → CAFT detectors
    → (optional) LLM confirmation → DashboardState

The MonitorEngine is the single integration point consumed by the terminal UI.
It is framework-agnostic: any JSONL line that an adapter can parse becomes
a TraceEvent, gets classified into an HTA phase, and is checked by all
CAFT detectors.

V4: Added semantic confirmation layer. Detectors are now "candidate generators"
whose output can be confirmed/rejected by an LLM before becoming diagnoses.
Set confirm=True to enable (requires AGENTDIAG_LLM_PROVIDER + API key).
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, IO, Optional

from agentdiag.models import TraceEvent
from agentdiag.adapters import auto_parse
from agentdiag.hta import HTAStateMachine, HTAState, Phase
from agentdiag.caft.base import CaftDiagnosis, CaftSeverity
from agentdiag.caft.detectors import (
    ALL_CAFT_DETECTORS,
    ALL_CAFT_DETECTORS_FULL,
    run_caft_detectors,
    run_caft_detectors_traced,
)
from agentdiag.decision_trace import (
    DecisionTrace,
    make_step_record,
    make_detector_snapshot,
)
from agentdiag.cognitive import CognitiveStateTracker, CognitiveState

logger = logging.getLogger(__name__)


@dataclass
class ActionEntry:
    """A single action shown in the live action stream."""
    step: int
    tool: str
    event_type: str
    phase: Phase
    success: bool
    latency_ms: float
    timestamp: float  # wall-clock time when received


@dataclass
class DashboardState:
    """Complete snapshot of dashboard state at any point in time.

    The terminal UI reads this to render all panels.
    """
    # HTA
    hta_state: Optional[HTAState] = None
    progress_pct: float = 0.0

    # Action stream (most recent N)
    actions: list[ActionEntry] = field(default_factory=list)

    # CAFT diagnostics (all detected so far)
    diagnoses: list[CaftDiagnosis] = field(default_factory=list)

    # Aggregate stats
    total_events: int = 0
    total_errors: int = 0
    events_per_minute: float = 0.0
    start_time: Optional[float] = None
    last_event_time: Optional[float] = None

    # Trust score: starts at 1.0, decremented by CAFT findings (deprecated, kept for compat)
    trust_score: float = 1.0

    # Session health metrics (V5 — grounded in observable behavior)
    completion_rate: float = 0.0        # 0-100 (phase score - regressions*5)
    failure_density: float = 0.0        # diagnoses per 100 events
    active_diagnosis_count: int = 0     # len(diagnoses)
    completion_history: list[float] = field(default_factory=list)  # last 50 snapshots for sparkline

    # Confirmation stats (V4)
    candidates_total: int = 0
    candidates_confirmed: int = 0
    candidates_rejected: int = 0
    candidates_uncertain: int = 0
    candidates_autoconfirmed: int = 0

    @property
    def health(self) -> str:
        worst = self.worst_severity
        if worst == CaftSeverity.CRITICAL:
            return "failing"
        if worst == CaftSeverity.WARNING:
            return "degraded"
        if self.total_events > 0 and self.completion_rate < 30:
            return "degraded"
        return "healthy"

    @property
    def worst_severity(self) -> Optional[CaftSeverity]:
        if not self.diagnoses:
            return None
        return max(self.diagnoses, key=lambda d: _SEVERITY_RANK[d.severity]).severity


_SEVERITY_RANK = {
    CaftSeverity.INFO: 0,
    CaftSeverity.WARNING: 1,
    CaftSeverity.CRITICAL: 2,
}

_TRUST_PENALTIES = {
    CaftSeverity.INFO: 0.02,
    CaftSeverity.WARNING: 0.08,
    CaftSeverity.CRITICAL: 0.20,
}


class MonitorEngine:
    """Real-time monitor engine.

    Accepts raw events (dicts or TraceEvent), pushes them through the
    HTA state machine and CAFT detectors, and maintains a DashboardState
    snapshot that the UI can poll.

    V4: When confirm=True, detector candidates are sent to the LLM
    confirmation layer before becoming diagnoses. High-confidence
    candidates (>= AUTOCONFIRM_THRESHOLD) skip LLM to save latency.

    Usage::

        engine = MonitorEngine(goal="Fix the login bug")
        for line in sys.stdin:
            new_diagnoses = engine.push_raw(json.loads(line))
            state = engine.state  # render this

        # With confirmation:
        engine = MonitorEngine(goal="Fix the login bug", confirm=True)
    """

    MAX_ACTIONS = 100  # keep last N actions in the stream

    def __init__(
        self,
        goal: str = "",
        detectors: list | None = None,
        on_diagnosis: Callable[[CaftDiagnosis], None] | None = None,
        on_event: Callable[[ActionEntry], None] | None = None,
        context_store: "ContextStore | None" = None,
        confirm: bool = False,
        autoconfirm_threshold: float = 0.9,
        calibration_path: str | None = None,
        decision_trace: bool = False,
        cognitive: bool = False,
    ):
        self._hta = HTAStateMachine(goal=goal)
        self._detectors = self._resolve_detectors(
            detectors, confirm, calibration_path,
        )
        self._on_diagnosis = on_diagnosis
        self._on_event = on_event
        self._context_store = context_store
        self._confirm = confirm
        self._autoconfirm_threshold = autoconfirm_threshold

        self._events: list[TraceEvent] = []
        self._actions: list[ActionEntry] = []
        self._diagnoses: list[CaftDiagnosis] = []
        self._seen_failures: dict[str, int] = {}

        self._total_errors = 0
        self._start_time: Optional[float] = None
        self._last_event_time: Optional[float] = None
        self._trust_score = 1.0

        # V5: Session health tracking
        self._highest_phase: Phase = Phase.IDLE
        self._completion_history: list[float] = []

        # V6: Decision trace (optional, off by default)
        self._decision_trace_enabled = decision_trace
        self._decision_trace: Optional[DecisionTrace] = (
            DecisionTrace() if decision_trace else None
        )

        # V7: Cognitive Load Monitor (optional, off by default)
        self._cognitive_enabled = cognitive
        self._cognitive_tracker: Optional[CognitiveStateTracker] = (
            CognitiveStateTracker() if cognitive else None
        )

        # Confirmation stats (V4)
        self._candidates_total = 0
        self._candidates_confirmed = 0
        self._candidates_rejected = 0
        self._candidates_uncertain = 0
        self._candidates_autoconfirmed = 0

    @staticmethod
    def _resolve_detectors(
        explicit: list | None,
        confirm: bool,
        calibration_path: str | None,
    ) -> list:
        """Resolve which detectors to use.

        Priority:
        1. Explicit list (caller override)
        2. confirm=True → all detectors (LLM handles precision)
        3. calibration_path → calibrated from that file
        4. Default → registry's enabled detectors (auto-calibrated if baselines found)
        """
        if explicit is not None:
            return explicit
        if confirm:
            return list(ALL_CAFT_DETECTORS_FULL)
        if calibration_path:
            return MonitorEngine._load_calibrated(calibration_path)
        # Default: use registry (which auto-loads calibration if available)
        from agentdiag.caft.registry import detector_registry
        return detector_registry.get_enabled()

    @staticmethod
    def _load_calibrated(path: str) -> list:
        """Load calibrated detectors from a profile file.

        Falls back to registry's enabled detectors on failure.
        """
        try:
            from agentdiag.baselines import CalibrationProfile
            from agentdiag.caft.calibrated import make_calibrated_detectors

            profile = CalibrationProfile.load(path)
            disabled = {
                "missing_verification", "goal_drift",
                "tool_thrashing", "reasoning_action_mismatch",
            }
            all_det = make_calibrated_detectors(profile)
            return [d for d in all_det if d.name not in disabled]
        except Exception as e:
            logger.warning("Failed to load calibration from %s: %s", path, e)
            from agentdiag.caft.registry import detector_registry
            return detector_registry.get_enabled()

    _PHASE_SCORES = {
        Phase.IDLE: 0, Phase.GATHERING: 20, Phase.PLANNING: 40,
        Phase.EXECUTING: 60, Phase.VERIFYING: 80, Phase.DELIVERING: 100,
    }

    @property
    def state(self) -> DashboardState:
        """Current dashboard state snapshot."""
        hta = self._hta.state
        now = time.time()
        elapsed = (now - self._start_time) if self._start_time else 0.0
        epm = (len(self._events) / elapsed * 60) if elapsed > 1.0 else 0.0

        # V5: Completion rate — phase score minus regression penalty
        completion_rate = max(
            0,
            self._PHASE_SCORES.get(self._highest_phase, 0)
            - hta.regression_count * 5,
        )

        # V5: Failure density — diagnoses per 100 events
        n = len(self._events)
        failure_density = round((len(self._diagnoses) / n) * 100, 2) if n > 0 else 0.0

        # V5: Sparkline history
        self._completion_history.append(completion_rate)
        if len(self._completion_history) > 50:
            self._completion_history = self._completion_history[-50:]

        return DashboardState(
            hta_state=hta,
            progress_pct=hta.progress_pct,
            actions=list(self._actions[-self.MAX_ACTIONS:]),
            diagnoses=list(self._diagnoses),
            total_events=len(self._events),
            total_errors=self._total_errors,
            events_per_minute=round(epm, 1),
            start_time=self._start_time,
            last_event_time=self._last_event_time,
            trust_score=self._trust_score,
            completion_rate=completion_rate,
            failure_density=failure_density,
            active_diagnosis_count=len(self._diagnoses),
            completion_history=list(self._completion_history),
            candidates_total=self._candidates_total,
            candidates_confirmed=self._candidates_confirmed,
            candidates_rejected=self._candidates_rejected,
            candidates_uncertain=self._candidates_uncertain,
            candidates_autoconfirmed=self._candidates_autoconfirmed,
        )

    def _apply_confirmation(
        self,
        candidate: CaftDiagnosis,
        hta_state: HTAState,
    ) -> Optional[CaftDiagnosis]:
        """Apply semantic confirmation to a candidate diagnosis.

        Returns:
            The diagnosis (possibly with adjusted confidence) if confirmed/uncertain.
            None if rejected by the LLM.
        """
        from agentdiag.caft.confirm import (
            confirm_diagnosis_sync,
            is_llm_available,
            AUTOCONFIRM_THRESHOLD,
        )

        self._candidates_total += 1
        threshold = self._autoconfirm_threshold

        # Apply FP rate discount from historical feedback (before auto-confirm check)
        if self._context_store is not None:
            try:
                self._context_store.adjust_diagnosis_confidence(candidate)
            except Exception:
                pass

        # High-confidence candidates skip LLM confirmation
        if candidate.confidence >= threshold:
            self._candidates_autoconfirmed += 1
            logger.debug(
                "Auto-confirmed %s (confidence=%.2f >= %.2f)",
                candidate.failure_name,
                candidate.confidence,
                threshold,
            )
            return candidate

        # Check if LLM is available
        if not is_llm_available():
            # Fall back to rule-only mode
            self._candidates_uncertain += 1
            logger.debug(
                "No LLM available; passing %s as uncertain",
                candidate.failure_name,
            )
            return candidate

        # Query OpenViking for similar past cases
        context_cases = []
        if self._context_store is not None:
            try:
                context_cases = self._context_store.find_similar_failures(
                    diagnosis=candidate,
                    limit=3,
                )
            except Exception:
                pass

        # Call LLM confirmation
        result = confirm_diagnosis_sync(
            candidate=candidate,
            events=self._events,
            hta_state=hta_state,
            context_cases=context_cases,
        )

        # Record confirmation result in OpenViking
        if self._context_store is not None:
            try:
                self._context_store.record_confirmation(
                    candidate=candidate,
                    result=result,
                )
            except (AttributeError, Exception):
                # record_confirmation may not exist yet or may fail
                pass

        if result.status == "confirmed":
            self._candidates_confirmed += 1
            # Boost confidence to LLM's confidence if higher
            candidate.confidence = max(candidate.confidence, result.confidence)
            return candidate
        elif result.status == "rejected":
            self._candidates_rejected += 1
            logger.info(
                "Rejected %s at step %d: %s",
                candidate.failure_name,
                candidate.at_step,
                result.reasoning,
            )
            return None
        else:
            # Uncertain — include with reduced confidence
            self._candidates_uncertain += 1
            candidate.confidence = result.confidence
            return candidate

    @property
    def decision_trace(self) -> Optional[DecisionTrace]:
        """The decision trace, or None if tracing is disabled."""
        return self._decision_trace

    @property
    def cognitive_state(self) -> Optional[CognitiveStateTracker]:
        """The cognitive state tracker, or None if cognitive monitoring is disabled."""
        return self._cognitive_tracker

    def push(self, event: TraceEvent) -> list[CaftDiagnosis]:
        """Push a TraceEvent through the full pipeline.

        Returns any new CAFT diagnoses detected on this event.
        """
        now = time.time()
        if self._start_time is None:
            self._start_time = now
        self._last_event_time = now

        self._events.append(event)

        if not event.success:
            self._total_errors += 1

        # HTA classification
        prev_transition_count = len(self._hta._transitions)
        hta_state = self._hta.push(event)

        # Track highest phase ever reached (V5)
        if hta_state.current_phase > self._highest_phase:
            self._highest_phase = hta_state.current_phase

        # Action stream entry
        action = ActionEntry(
            step=event.step,
            tool=event.tool or event.type,
            event_type=event.type,
            phase=hta_state.current_phase,
            success=event.success,
            latency_ms=event.latency_ms,
            timestamp=now,
        )
        self._actions.append(action)
        if len(self._actions) > self.MAX_ACTIONS * 2:
            self._actions = self._actions[-self.MAX_ACTIONS:]

        if self._on_event:
            self._on_event(action)

        # CAFT detection (every event)
        # V6: Use traced variant when decision trace is enabled
        detector_snapshots: list[dict] = []
        if self._decision_trace_enabled:
            candidates, detector_snapshots = run_caft_detectors_traced(
                events=self._events,
                hta_state=hta_state,
                detectors=self._detectors,
                seen=self._seen_failures,
            )
        else:
            candidates = run_caft_detectors(
                events=self._events,
                hta_state=hta_state,
                detectors=self._detectors,
                seen=self._seen_failures,
            )

        # V6: Record to decision trace
        if self._decision_trace is not None:
            # Build HTA transition info (if a transition happened this step)
            hta_transition = None
            if len(hta_state.transitions) > prev_transition_count:
                t = hta_state.transitions[-1]
                hta_transition = {
                    "from": t.from_phase.label,
                    "to": t.to_phase.label,
                    "is_regression": t.is_regression,
                    "trigger": "strong_signal" if len(hta_state.transitions) > prev_transition_count else "hysteresis",
                }

            step_record = make_step_record(
                step=event.step,
                event_type=event.type,
                tool=event.tool or event.type,
                hta_phase=hta_state.current_phase.label,
                hta_transition=hta_transition,
                latency_ms=event.latency_ms,
                success=event.success,
            )

            # Update profile with output_hash for reread tracking
            self._decision_trace.profile.update(
                tool=event.tool or event.type,
                event_type=event.type,
                latency_ms=event.latency_ms,
                success=event.success,
                phase=hta_state.current_phase.label,
                output_hash=event.output_hash,
            )

            self._decision_trace.record(
                step_record=step_record,
                detector_snapshots=detector_snapshots,
                hta_regression_count=hta_state.regression_count,
            )

        # V7: Update cognitive state tracker
        if self._cognitive_tracker is not None:
            self._cognitive_tracker.update(
                event=event,
                phase=hta_state.current_phase,
                num_detectors=len(self._detectors),
                num_diagnoses=len(self._diagnoses),
                num_confirmed=self._candidates_confirmed,
                num_rejected=self._candidates_rejected,
                has_context_store=self._context_store is not None,
            )

        # V4: Apply semantic confirmation if enabled
        new_diagnoses = []
        for d in candidates:
            if self._confirm:
                confirmed = self._apply_confirmation(d, hta_state)
                if confirmed is None:
                    continue  # Rejected by LLM
                d = confirmed

            # Apply feedback adjustment: discount confidence based on
            # historical FP rate from reviewed cases in the context store
            if self._context_store is not None:
                self._context_store.adjust_diagnosis_confidence(d)

            new_diagnoses.append(d)
            self._diagnoses.append(d)
            penalty = _TRUST_PENALTIES.get(d.severity, 0.05)
            self._trust_score = max(0.0, self._trust_score - penalty)
            if self._on_diagnosis:
                self._on_diagnosis(d)

        # Record to persistent context if available
        if self._context_store is not None:
            self._context_store.record_event(
                event=event,
                phase=hta_state.current_phase,
                diagnoses=new_diagnoses,
                trust_score=self._trust_score,
            )

        return new_diagnoses

    def push_raw(self, data: dict | list) -> list[CaftDiagnosis]:
        """Parse raw data through adapter and push resulting events.

        Accepts a single event dict or a list of events (e.g., a full
        Claude API message with multiple content blocks).
        """
        # Wrap single dict in list for adapter auto_parse
        if isinstance(data, dict):
            data = [data]

        try:
            events = auto_parse(data)
        except ValueError:
            # Fall back to direct TraceEvent construction
            events = [TraceEvent.from_dict(d) if isinstance(d, dict) else d
                      for d in data]

        all_diagnoses = []
        for event in events:
            all_diagnoses.extend(self.push(event))
        return all_diagnoses

    def start_context_session(self, goal: str = "", source: str = "") -> str:
        """Start a context session if a context store is available.

        Returns the session ID, or empty string if no store.
        """
        if self._context_store is None:
            return ""
        return self._context_store.start_session(goal=goal, source=source)

    def end_context_session(self) -> dict:
        """Commit the context session with the current dashboard state.

        Returns the commit result dict, or empty dict if no store.
        """
        if self._context_store is None:
            return {}
        return self._context_store.end_session(self.state)

    def set_goal(self, goal: str) -> None:
        """Update the monitored goal."""
        self._hta.set_goal(goal)

    def reset(self) -> None:
        """Reset all state for a new monitoring session."""
        self._hta = HTAStateMachine(goal=self._hta._goal)
        self._events.clear()
        self._actions.clear()
        self._diagnoses.clear()
        self._seen_failures.clear()
        self._total_errors = 0
        self._start_time = None
        self._last_event_time = None
        self._trust_score = 1.0
        self._highest_phase = Phase.IDLE
        self._completion_history.clear()
        if self._decision_trace is not None:
            self._decision_trace.reset()
        if self._cognitive_tracker is not None:
            self._cognitive_tracker = CognitiveStateTracker()
        self._candidates_total = 0
        self._candidates_confirmed = 0
        self._candidates_rejected = 0
        self._candidates_uncertain = 0
        self._candidates_autoconfirmed = 0


def run_stdin_monitor(
    goal: str = "",
    stream: IO[str] | None = None,
    on_state: Callable[[DashboardState], None] | None = None,
    on_diagnosis: Callable[[CaftDiagnosis], None] | None = None,
    context_store: "ContextStore | None" = None,
    confirm: bool = False,
) -> DashboardState:
    """Read JSONL from stdin (or stream) and run the monitor pipeline.

    This is the non-UI entry point for `agentdiag monitor --input stdin`.
    The terminal UI wraps this with a rich Live display.

    Args:
        confirm: Enable LLM confirmation layer (default: False = rule-only).

    Returns the final DashboardState when input is exhausted.
    """
    engine = MonitorEngine(
        goal=goal,
        on_diagnosis=on_diagnosis,
        context_store=context_store,
        confirm=confirm,
    )
    source = stream or sys.stdin

    if context_store is not None:
        engine.start_context_session(goal=goal, source="stdin_monitor")

    for line in source:
        line = line.strip()
        if not line:
            continue

        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        engine.push_raw(data)

        if on_state:
            on_state(engine.state)

    if context_store is not None:
        engine.end_context_session()

    return engine.state
