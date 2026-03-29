"""Decision trace data layer.

Captures what every detector computed at every step, so the frontend can
render a decision timeline showing WHY detectors fired (or didn't).

The trace is optional (off by default) and accumulates three record types
per event pushed to MonitorEngine:

  1. StepRecord — event metadata + HTA state at that moment
  2. DetectorSnapshot — per-detector intermediate values + gate info
  3. SessionProfile — running aggregates updated incrementally

Usage::

    engine = MonitorEngine(goal="Fix bug", decision_trace=True)
    for event in events:
        engine.push(event)
    trace = engine.decision_trace
    trace.to_dict()  # full JSON-serializable trace
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from agentdiag.hta import Phase


# ── Lightweight record types (plain dicts, not dataclasses) ────────────


def make_step_record(
    step: int,
    event_type: str,
    tool: str,
    hta_phase: str,
    hta_transition: Optional[dict],
    latency_ms: float,
    success: bool,
) -> dict:
    """Build a step record dict."""
    return {
        "step": step,
        "event_type": event_type,
        "tool": tool,
        "hta_phase": hta_phase,
        "hta_transition": hta_transition,
        "latency_ms": round(latency_ms, 1),
        "success": success,
    }


def make_detector_snapshot(
    detector: str,
    step: int,
    fired: bool,
    confidence: float = 0.0,
    gate_failed: Optional[str] = None,
    evidence_preview: Optional[dict] = None,
    auto_confirmed: Optional[bool] = None,
    force_llm_review: Optional[bool] = None,
    llm_decision: Optional[str] = None,
    llm_reasoning: Optional[str] = None,
) -> dict:
    """Build a detector snapshot dict.

    When fired=False, gate_failed explains which condition prevented firing.
    When fired=True, confidence/evidence come from the diagnosis.
    """
    snap: dict[str, Any] = {
        "detector": detector,
        "step": step,
        "fired": fired,
        "confidence": round(confidence, 4),
    }
    if gate_failed is not None:
        snap["gate_failed"] = gate_failed
    if evidence_preview is not None:
        snap["evidence_preview"] = evidence_preview
    if fired:
        if auto_confirmed is not None:
            snap["auto_confirmed"] = auto_confirmed
        if force_llm_review is not None:
            snap["force_llm_review"] = force_llm_review
        if llm_decision is not None:
            snap["llm_decision"] = llm_decision
        if llm_reasoning is not None:
            snap["llm_reasoning"] = llm_reasoning
    return snap


# ── Session profile (incrementally maintained) ─────────────────────────


@dataclass
class SessionProfile:
    """Running session-level aggregates, updated on each step.

    All fields are designed for O(1) incremental update — no re-scanning
    of the full event list.
    """

    # Running sums for averages
    _total_latency_ms: float = 0.0
    _total_events: int = 0
    _total_errors: int = 0

    # Phase distribution (counts)
    _phase_counts: dict[str, int] = field(default_factory=dict)

    # Tool tracking
    _tool_set: set[str] = field(default_factory=set)
    _read_targets: dict[str, int] = field(default_factory=dict)  # hash → count

    # Streak tracking
    _current_streak_tool: str = ""
    _current_streak_count: int = 0
    _longest_streak_tool: str = ""
    _longest_streak_count: int = 0

    # Regression count (from HTA)
    regression_count: int = 0

    def update(
        self,
        tool: str,
        event_type: str,
        latency_ms: float,
        success: bool,
        phase: str,
        output_hash: Optional[str] = None,
    ) -> None:
        """O(1) incremental update from a single event."""
        self._total_events += 1
        self._total_latency_ms += latency_ms
        if not success:
            self._total_errors += 1

        # Phase distribution
        self._phase_counts[phase] = self._phase_counts.get(phase, 0) + 1

        # Tool tracking
        if tool:
            self._tool_set.add(tool)

        # Re-read tracking (for read-like tools)
        _read_tools = {"read", "read_file", "cat", "head", "grep", "glob"}
        if tool and tool.lower() in _read_tools and output_hash:
            self._read_targets[output_hash] = (
                self._read_targets.get(output_hash, 0) + 1
            )

        # Streak tracking
        if tool == self._current_streak_tool:
            self._current_streak_count += 1
        else:
            self._current_streak_tool = tool or ""
            self._current_streak_count = 1

        if self._current_streak_count > self._longest_streak_count:
            self._longest_streak_tool = self._current_streak_tool
            self._longest_streak_count = self._current_streak_count

    def to_dict(self) -> dict:
        """Snapshot the current profile as a plain dict."""
        n = self._total_events or 1
        total_phase = sum(self._phase_counts.values()) or 1

        return {
            "avg_latency_ms": round(self._total_latency_ms / n, 1),
            "error_rate": round(self._total_errors / n, 3),
            "phase_distribution": {
                k: round(v / total_phase, 3)
                for k, v in self._phase_counts.items()
            },
            "regression_count": self.regression_count,
            "unique_tools": len(self._tool_set),
            "reread_count": sum(
                1 for c in self._read_targets.values() if c > 1
            ),
            "longest_streak": {
                "tool": self._longest_streak_tool,
                "count": self._longest_streak_count,
            },
        }


# ── Decision trace accumulator ─────────────────────────────────────────


class DecisionTrace:
    """Accumulates the full decision trace for a monitoring session.

    Thread-safe for single-writer (MonitorEngine.push) because Python's
    GIL protects list.append and dict assignment. The reader
    (API endpoint / WebSocket) only calls to_dict() which snapshots.
    """

    def __init__(self) -> None:
        self._steps: list[dict] = []          # StepRecord per event
        self._snapshots: list[dict] = []      # DetectorSnapshot per (event, detector)
        self._profiles: list[dict] = []       # SessionProfile snapshot per event
        self._profile = SessionProfile()
        self._prev_phase: Optional[str] = None
        self._prev_transition_count: int = 0

    @property
    def profile(self) -> SessionProfile:
        return self._profile

    def record(
        self,
        step_record: dict,
        detector_snapshots: list[dict],
        hta_regression_count: int = 0,
    ) -> None:
        """Record one step's worth of trace data.

        Called once per event in MonitorEngine.push().
        """
        self._steps.append(step_record)
        self._snapshots.extend(detector_snapshots)

        # Update session profile from step record
        self._profile.update(
            tool=step_record.get("tool", ""),
            event_type=step_record.get("event_type", ""),
            latency_ms=step_record.get("latency_ms", 0.0),
            success=step_record.get("success", True),
            phase=step_record.get("hta_phase", "idle"),
        )
        self._profile.regression_count = hta_regression_count

        # Store profile snapshot
        self._profiles.append(self._profile.to_dict())

    def to_dict(self) -> dict:
        """Full trace as a JSON-serializable dict."""
        return {
            "steps": list(self._steps),
            "detector_snapshots": list(self._snapshots),
            "session_profiles": list(self._profiles),
            "total_steps": len(self._steps),
            "current_profile": self._profile.to_dict(),
        }

    def get_step(self, step: int) -> Optional[dict]:
        """Get a single step record by step number."""
        for s in self._steps:
            if s.get("step") == step:
                return s
        return None

    def get_detector_timeline(self, detector_name: str) -> list[dict]:
        """Get all snapshots for a given detector, in step order."""
        return [
            s for s in self._snapshots
            if s.get("detector") == detector_name
        ]

    def get_snapshots_at_step(self, step: int) -> list[dict]:
        """Get all detector snapshots for a given step."""
        return [
            s for s in self._snapshots
            if s.get("step") == step
        ]

    def latest_snapshots(self, n: int = 1) -> list[dict]:
        """Get detector snapshots from the last N steps."""
        if not self._steps:
            return []
        last_steps = {s["step"] for s in self._steps[-n:]}
        return [
            s for s in self._snapshots
            if s.get("step") in last_steps
        ]

    def reset(self) -> None:
        """Clear all trace data for a new session."""
        self._steps.clear()
        self._snapshots.clear()
        self._profiles.clear()
        self._profile = SessionProfile()
        self._prev_phase = None
        self._prev_transition_count = 0
