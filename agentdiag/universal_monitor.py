"""Universal Monitor — the bridge between event sources and visualization.

Consumes an ObservableEvent stream from any adapter (Claude Code, harness,
OpenViking, or mixed) and routes events through the IT analysis pipeline:

  ObservableEvent → EventRouter (per-signal SymbolStreams)
                  → SelfCalibratingBaseline (z-scores)
                  → CompositionalAnomalyDetector (signature matching)
                  → get_state() → WebSocket payload

Phase markers are excluded from SymbolStreams and stored for sparkline
vertical lines.  Memory operations go to both the SymbolStream AND a
dedicated memory-ops track.  Evaluation signals are stored for
retrospective markers.

Usage::

    monitor = UniversalMonitor()
    for event in event_stream:
        result = monitor.process(event)
        # result has type, metrics, anomalies, etc.

    # For WebSocket:
    ws_payload = monitor.get_state()
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Any, Optional

from agentdiag.observable import ObservableEvent, EventType
from agentdiag.cognitive import EventRouter, SymbolStream
from agentdiag.baseline import SelfCalibratingBaseline
from agentdiag.compositor import CompositionalAnomalyDetector, AnomalySignature


# ── IP stage classification ────────────────────────────────────────────────

_TOOL_TO_STAGE = {
    "read_file": "perception", "read": "perception", "grep": "perception",
    "glob": "perception", "find": "perception", "search": "perception",
    "cat": "perception", "head": "perception", "tail": "perception",
    "ls": "perception", "list_files": "perception",
    "reasoning": "decision_making", "planning": "decision_making",
    "thinking": "decision_making",
    "edit": "action", "write": "action", "write_file": "action",
    "bash": "action", "shell": "action",
    "pytest": "feedback", "test": "feedback", "lint": "feedback",
    "commit": "feedback", "push": "feedback",
}

_EVENT_TYPE_TO_STAGE = {
    EventType.FILE_READ: "perception",
    EventType.FILE_WRITE: "action",
    EventType.SHELL_COMMAND: "action",
    EventType.TOOL_CALL: "action",
    EventType.MEMORY_LOAD: "perception",
    EventType.MEMORY_STORE: "action",
    EventType.MEMORY_EVICT: "working_memory",
    EventType.MEMORY_TIER_ESCALATION: "working_memory",
    EventType.EVALUATION_RESULT: "feedback",
    EventType.CONTRACT_EVENT: "decision_making",
}

_HARNESS_ROLE_TO_STAGE = {
    "planner": "decision_making",
    "generator": "action",
    "evaluator": "feedback",
    "orchestrator": "decision_making",
}

_ADJACENT_STAGES = [
    ("perception", "attention"),
    ("attention", "working_memory"),
    ("working_memory", "decision_making"),
    ("decision_making", "action"),
    ("action", "feedback"),
]


def classify_stage(event: ObservableEvent) -> str:
    """Classify an event into an IP stage.

    Priority: agent_role (harness) > tool name > event type > default.
    """
    if event.agent_role is not None:
        role_val = event.agent_role.value if hasattr(event.agent_role, "value") else str(event.agent_role)
        if role_val in _HARNESS_ROLE_TO_STAGE:
            return _HARNESS_ROLE_TO_STAGE[role_val]

    tool = (event.tool_name or "").lower()
    if tool in _TOOL_TO_STAGE:
        return _TOOL_TO_STAGE[tool]

    if event.event_type in _EVENT_TYPE_TO_STAGE:
        return _EVENT_TYPE_TO_STAGE[event.event_type]

    return "action"


# ── Cross-stage MI tracker ─────────────────────────────────────────────────

class CrossStageMI:
    """Tracks IP stage transitions and computes MI between adjacent stages."""

    def __init__(self, window: int = 50) -> None:
        self._bigrams: deque[tuple[str, str]] = deque(maxlen=window)
        self._joint: dict[tuple[str, str], int] = {}
        self._margin_x: dict[str, int] = {}
        self._margin_y: dict[str, int] = {}
        self._total: int = 0
        self._prev_stage: Optional[str] = None

    def push(self, stage: str) -> None:
        if self._prev_stage is not None:
            bigram = (self._prev_stage, stage)
            if len(self._bigrams) == self._bigrams.maxlen:
                old = self._bigrams[0]
                self._joint[old] -= 1
                if self._joint[old] == 0:
                    del self._joint[old]
                self._margin_x[old[0]] -= 1
                if self._margin_x[old[0]] == 0:
                    del self._margin_x[old[0]]
                self._margin_y[old[1]] -= 1
                if self._margin_y[old[1]] == 0:
                    del self._margin_y[old[1]]
                self._total -= 1

            self._bigrams.append(bigram)
            self._joint[bigram] = self._joint.get(bigram, 0) + 1
            self._margin_x[bigram[0]] = self._margin_x.get(bigram[0], 0) + 1
            self._margin_y[bigram[1]] = self._margin_y.get(bigram[1], 0) + 1
            self._total += 1
        self._prev_stage = stage

    def flow_rates(self) -> dict[str, float]:
        """MI between adjacent IP stages for pipeline strip arrow thickness."""
        rates = {}
        for a, b in _ADJACENT_STAGES:
            key = f"{a}\u2192{b}"
            if self._total == 0:
                rates[key] = 0.0
                continue
            jc = self._joint.get((a, b), 0)
            if jc == 0:
                rates[key] = 0.0
                continue
            p_xy = jc / self._total
            p_x = self._margin_x.get(a, 0) / self._total
            p_y = self._margin_y.get(b, 0) / self._total
            if p_x > 0 and p_y > 0:
                mi = p_xy * math.log2(p_xy / (p_x * p_y))
                rates[key] = round(max(mi, 0.0), 4)
            else:
                rates[key] = 0.0
        return rates


# ── Working memory tracker (inferred + explicit) ──────────────────────────

class InferredWorkingMemory:
    """Tracks working memory from file events (inferred) and memory events (explicit).

    When explicit items (from OpenViking MEMORY_LOAD/EVICT) are present,
    they take priority.  When absent, inferred items from FILE_READ/WRITE
    fill in.  This is the graceful degradation path.
    """

    def __init__(self, max_items: int = 100, decay_steps: int = 50) -> None:
        self._items: dict[str, dict] = {}
        self._max_items = max_items
        self._decay_steps = decay_steps
        self._current_step = 0
        self._has_explicit = False

    def record_event(self, event: ObservableEvent) -> None:
        self._current_step = event.step
        if event.is_memory_operation():
            self._has_explicit = True
            self._handle_memory_event(event)
        elif event.event_type in (EventType.FILE_READ, EventType.FILE_WRITE):
            self._handle_file_event(event)

    def _handle_memory_event(self, event: ObservableEvent) -> None:
        key = event.viking_uri or event.namespace or event.to_symbol()
        tier = event.memory_tier.value if event.memory_tier else "l0"

        if event.event_type == EventType.MEMORY_EVICT:
            if key in self._items:
                self._items[key]["evicted"] = True
            return

        if key in self._items:
            self._items[key]["last_step"] = event.step
            self._items[key]["accesses"] = self._items[key].get("accesses", 0) + 1
            self._items[key]["tier"] = tier
            self._items[key]["evicted"] = False
        else:
            self._items[key] = {
                "name": event.namespace or key,
                "tier": tier,
                "first_step": event.step, "last_step": event.step,
                "tokens": event.token_count or 0,
                "accesses": 1, "source": "explicit", "evicted": False,
            }
        self._evict_oldest()

    def _handle_file_event(self, event: ObservableEvent) -> None:
        path = event.target_path or event.to_symbol()
        key = f"file:{path}"
        if key in self._items:
            self._items[key]["last_step"] = event.step
            self._items[key]["accesses"] = self._items[key].get("accesses", 0) + 1
        else:
            self._items[key] = {
                "name": path.split("/")[-1] if "/" in path else path,
                "tier": None,
                "first_step": event.step, "last_step": event.step,
                "tokens": event.output_tokens or event.input_tokens or 0,
                "accesses": 1, "source": "inferred", "evicted": False,
            }
        self._evict_oldest()

    def _evict_oldest(self) -> None:
        while len(self._items) > self._max_items:
            oldest = min(
                (k for k, v in self._items.items() if not v.get("evicted")),
                key=lambda k: self._items[k]["last_step"],
                default=None,
            )
            if oldest:
                self._items[oldest]["evicted"] = True
            else:
                break

    def _retention(self, item: dict) -> float:
        if item.get("evicted"):
            return 0.0
        steps_since = self._current_step - item.get("last_step", 0)
        recency = max(0.0, 1.0 - steps_since / self._decay_steps)
        consolidation = min(item.get("accesses", 1) / 5.0, 1.0)
        return min(1.0, 0.6 * recency + 0.4 * consolidation)

    def to_dict(self) -> dict:
        active, at_risk, evicted = [], [], []
        for key, item in self._items.items():
            ret = self._retention(item)
            entry = {
                "name": item["name"], "hash": key,
                "tier": item.get("tier"), "accesses": item.get("accesses", 1),
                "step": item.get("last_step", 0), "retention": round(ret, 3),
                "source": item.get("source", "inferred"),
                "tokens": item.get("tokens", 0),
            }
            if item.get("evicted"):
                evicted.append(entry)
            elif ret > 0.3:
                active.append(entry)
            else:
                at_risk.append(entry)

        active.sort(key=lambda x: x["retention"], reverse=True)
        at_risk.sort(key=lambda x: x["retention"])
        evicted.sort(key=lambda x: x["step"], reverse=True)

        return {
            "active_items": active[:20],
            "at_risk_items": at_risk[:10],
            "evicted_items": evicted[:5],
            "total_items": len(self._items),
            "has_explicit": self._has_explicit,
            "utilization": min(
                sum(i.get("tokens", 0) for i in self._items.values()) / 200_000, 1.0
            ),
        }


# ── Memory operations tracker (Agent 1's domain) ─────────────────────────

class _MemoryOpsTracker:
    """Lightweight tracker for memory operation metrics.

    Feeds namespace_entropy and memory_escalation_rate into the baseline
    so the compositor can detect context_thrashing signatures.
    """

    def __init__(self, window: int = 50) -> None:
        self._window: deque[ObservableEvent] = deque(maxlen=window)
        self._ns_counts: dict[str, int] = {}
        self._esc_count: int = 0

    def push(self, event: ObservableEvent) -> None:
        if len(self._window) == self._window.maxlen:
            self._remove(self._window[0])
        self._window.append(event)
        self._add(event)

    def _add(self, event: ObservableEvent) -> None:
        ns = event.namespace or "unknown"
        self._ns_counts[ns] = self._ns_counts.get(ns, 0) + 1
        if event.event_type == EventType.MEMORY_TIER_ESCALATION:
            self._esc_count += 1

    def _remove(self, event: ObservableEvent) -> None:
        ns = event.namespace or "unknown"
        c = self._ns_counts.get(ns, 0) - 1
        if c <= 0:
            self._ns_counts.pop(ns, None)
        else:
            self._ns_counts[ns] = c
        if event.event_type == EventType.MEMORY_TIER_ESCALATION:
            self._esc_count = max(0, self._esc_count - 1)

    def get_metrics(self) -> dict[str, float]:
        n = len(self._window)
        if n == 0:
            return {"namespace_entropy": 0.0, "memory_escalation_rate": 0.0}
        total = sum(self._ns_counts.values())
        ns_h = 0.0
        if total > 0:
            for count in self._ns_counts.values():
                p = count / total
                if p > 0:
                    ns_h -= p * math.log2(p)
        return {
            "namespace_entropy": round(ns_h, 4),
            "memory_escalation_rate": round(self._esc_count / n, 4),
        }


# ── UniversalMonitor ───────────────────────────────────────────────────────

class UniversalMonitor:
    """Consumes ObservableEvent stream, produces visualization state.

    Routes events through:
      EventRouter → per-signal SymbolStreams (IT computation)
      SelfCalibratingBaseline → z-scores (anomaly detection)
      CompositionalAnomalyDetector → signature matching
      CrossStageMI → pipeline strip arrow thickness
      InferredWorkingMemory → left panel items

    All state available via get_state() for WebSocket serialization.
    """

    def __init__(
        self,
        calibration_window: int = 100,
        sensitivity: float = 2.0,
    ) -> None:
        self.router = EventRouter(window=50)
        self.baseline = SelfCalibratingBaseline(
            calibration_window=calibration_window,
            sensitivity=sensitivity,
        )
        self.compositor = CompositionalAnomalyDetector()
        self.cross_stage = CrossStageMI(window=50)
        self.working_memory = InferredWorkingMemory()
        self._memory_tracker = _MemoryOpsTracker()

        self._event_count = 0
        self._error_count = 0
        self._start_time: Optional[float] = None
        self._last_event_time: Optional[float] = None
        self._anomaly_timeline: deque[dict] = deque(maxlen=200)
        self._current_phase: Optional[str] = None

    def process(self, event: ObservableEvent) -> dict:
        """Process a single event through the full pipeline.

        Returns a result dict with type, metrics, anomalies, etc.
        """
        self._event_count += 1
        now = time.time()
        if self._start_time is None:
            self._start_time = now
        self._last_event_time = now

        # Track working memory
        self.working_memory.record_event(event)

        # Track memory operations for namespace_entropy / escalation_rate
        if event.is_memory_operation():
            self._memory_tracker.push(event)

        # Phase markers: update baseline phase, store for sparklines
        if event.is_phase_marker():
            phase_val = event.phase.value if event.phase else None
            self._current_phase = phase_val
            self.baseline.set_phase(phase_val)
            self.router.process_event(event)  # stores in phase_markers list
            return {"type": "phase_marker", "event": event.to_dict(), "step": event.step}

        # Evaluation signals: store for retrospective markers
        if event.is_evaluation_signal():
            self.router.process_event(event)  # stores in evaluation_events list
            return {"type": "evaluation", "event": event.to_dict(), "step": event.step}

        # Normal event: full pipeline
        symbol = event.to_symbol()
        stage = classify_stage(event)

        self.router.process_event(event)
        self.cross_stage.push(stage)

        # Extract metrics for baseline
        act = self.router.action_stream
        metrics = {
            "action_entropy": act.entropy(),
            "action_mi": act.mi(),
            "compression_ratio": act.compression(),
            "last_surprisal": act.surprisal(),
            "tool_entropy": self.router.tool_stream.entropy(),
            "read_entropy": self.router.read_stream.entropy(),
            "memory_entropy": self.router.memory_stream.entropy(),
            "kl_divergence": act.kl_divergence(),
        }

        # Add Agent 1's memory operation metrics (namespace entropy,
        # escalation rate) so the compositor can detect context_thrashing.
        # Only include when the tracker has seen enough events to form a
        # meaningful baseline — avoids false positives from zero-variance
        # calibration when early events have no memory operations.
        if hasattr(self, '_memory_tracker') and len(self._memory_tracker._window) >= 10:
            mem = self._memory_tracker.get_metrics()
            metrics.update(mem)

        # Baseline z-scores → compositor
        anomalies_raw = self.baseline.observe(metrics, phase=self._current_phase)
        result = {
            "type": "observation",
            "event": event.to_dict(),
            "step": event.step,
            "stage": stage,
            "metrics": metrics,
            "anomalies": None,
        }

        if anomalies_raw:
            sig = self.compositor.analyze(anomalies_raw, step=event.step)
            if sig is not None:
                result["anomalies"] = sig.to_dict()
                self._anomaly_timeline.append(sig.to_dict())

        return result

    def get_state(self) -> dict:
        """Full state for WebSocket payload — everything the UI needs."""
        elapsed = (
            (self._last_event_time - self._start_time)
            if self._start_time and self._last_event_time else 0.0
        )
        return {
            "info_theoretic": self.router.to_dict(),
            "flow_rates": self.cross_stage.flow_rates(),
            "working_memory": self.working_memory.to_dict(),
            "anomalies": list(self._anomaly_timeline),
            "baseline": self.baseline.get_baseline_summary(),
            "compositor": self.compositor.get_summary(),
            "total_events": self._event_count,
            "total_errors": self._error_count,
            "events_per_minute": (
                (self._event_count / elapsed * 60) if elapsed > 1.0 else 0.0
            ),
            "current_phase": self._current_phase,
        }

    def get_memory_state(self) -> dict:
        """Working memory state for the left panel."""
        return self.working_memory.to_dict()

    @property
    def total_events(self) -> int:
        return self._event_count

    @property
    def is_calibrating(self) -> bool:
        return self.baseline.is_calibrating

    def manual_baseline_lock(self) -> None:
        """Escape hatch: freeze baseline at current state."""
        self.baseline.manual_baseline_lock()
