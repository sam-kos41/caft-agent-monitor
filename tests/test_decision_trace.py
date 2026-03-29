"""Tests for the decision trace data layer."""

import pytest

from agentdiag.models import TraceEvent
from agentdiag.hta import Phase
from agentdiag.caft.base import CaftDiagnosis, CaftSeverity
from agentdiag.monitor import MonitorEngine
from agentdiag.decision_trace import (
    DecisionTrace,
    SessionProfile,
    make_step_record,
    make_detector_snapshot,
)
from agentdiag.caft.detectors import (
    StallDetector,
    ErrorCascadeDetector,
    StepRepetitionDetector,
    ContextLossDetector,
    run_caft_detectors_traced,
)


# ── Helper factories ──────────────────────────────────────────────────


def _event(step: int, tool: str = "read_file", success: bool = True,
           latency_ms: float = 100.0, **kw) -> TraceEvent:
    return TraceEvent(
        step=step, type="tool_call", tool=tool,
        success=success, latency_ms=latency_ms, **kw,
    )


# ── Unit tests: record builders ───────────────────────────────────────


class TestMakeStepRecord:
    def test_basic(self):
        rec = make_step_record(
            step=1, event_type="tool_call", tool="Read",
            hta_phase="gathering", hta_transition=None,
            latency_ms=245.123, success=True,
        )
        assert rec["step"] == 1
        assert rec["tool"] == "Read"
        assert rec["hta_phase"] == "gathering"
        assert rec["hta_transition"] is None
        assert rec["latency_ms"] == 245.1
        assert rec["success"] is True

    def test_with_transition(self):
        t = {"from": "planning", "to": "executing", "is_regression": False, "trigger": "strong_signal"}
        rec = make_step_record(
            step=5, event_type="tool_call", tool="edit",
            hta_phase="executing", hta_transition=t,
            latency_ms=50.0, success=True,
        )
        assert rec["hta_transition"]["from"] == "planning"
        assert rec["hta_transition"]["is_regression"] is False


class TestMakeDetectorSnapshot:
    def test_not_fired(self):
        snap = make_detector_snapshot(
            detector="stall", step=10, fired=False,
            confidence=0.35, gate_failed="min_stall_count",
            evidence_preview={"stall_count": 1, "threshold": 4500},
        )
        assert snap["detector"] == "stall"
        assert snap["fired"] is False
        assert snap["gate_failed"] == "min_stall_count"
        assert "auto_confirmed" not in snap  # only on fired

    def test_fired(self):
        snap = make_detector_snapshot(
            detector="stall", step=42, fired=True,
            confidence=0.85, auto_confirmed=False,
            force_llm_review=True, llm_decision="rejected",
            llm_reasoning="Single slow grep...",
        )
        assert snap["fired"] is True
        assert snap["confidence"] == 0.85
        assert snap["auto_confirmed"] is False
        assert snap["llm_decision"] == "rejected"
        assert "gate_failed" not in snap


# ── Unit tests: SessionProfile ────────────────────────────────────────


class TestSessionProfile:
    def test_incremental_update(self):
        p = SessionProfile()
        p.update("read_file", "tool_call", 100.0, True, "gathering")
        p.update("read_file", "tool_call", 200.0, True, "gathering")
        p.update("edit", "tool_call", 50.0, False, "executing")

        d = p.to_dict()
        assert d["avg_latency_ms"] == pytest.approx(116.7, abs=0.1)
        assert d["error_rate"] == pytest.approx(0.333, abs=0.001)
        assert d["unique_tools"] == 2
        assert d["phase_distribution"]["gathering"] == pytest.approx(0.667, abs=0.001)
        assert d["phase_distribution"]["executing"] == pytest.approx(0.333, abs=0.001)

    def test_streak_tracking(self):
        p = SessionProfile()
        for i in range(5):
            p.update("read_file", "tool_call", 100.0, True, "gathering")
        p.update("edit", "tool_call", 50.0, True, "executing")
        p.update("edit", "tool_call", 50.0, True, "executing")

        d = p.to_dict()
        assert d["longest_streak"]["tool"] == "read_file"
        assert d["longest_streak"]["count"] == 5

    def test_reread_tracking(self):
        p = SessionProfile()
        p.update("read", "tool_call", 100.0, True, "gathering", output_hash="abc123")
        p.update("grep", "tool_call", 50.0, True, "gathering")
        p.update("read", "tool_call", 100.0, True, "gathering", output_hash="abc123")

        d = p.to_dict()
        assert d["reread_count"] == 1  # abc123 was read twice


# ── Unit tests: DecisionTrace ─────────────────────────────────────────


class TestDecisionTrace:
    def test_record_and_retrieve(self):
        trace = DecisionTrace()
        rec = make_step_record(1, "tool_call", "Read", "gathering", None, 100.0, True)
        snaps = [
            make_detector_snapshot("stall", 1, False, gate_failed="min_events"),
            make_detector_snapshot("error_cascade", 1, False, gate_failed="min_events"),
        ]
        trace.record(rec, snaps, hta_regression_count=0)

        assert len(trace._steps) == 1
        assert len(trace._snapshots) == 2
        assert len(trace._profiles) == 1

    def test_get_step(self):
        trace = DecisionTrace()
        for s in [1, 2, 3]:
            rec = make_step_record(s, "tool_call", "Read", "gathering", None, 100.0, True)
            trace.record(rec, [], hta_regression_count=0)

        found = trace.get_step(2)
        assert found is not None
        assert found["step"] == 2
        assert trace.get_step(99) is None

    def test_get_detector_timeline(self):
        trace = DecisionTrace()
        for s in [1, 2, 3]:
            rec = make_step_record(s, "tool_call", "Read", "gathering", None, 100.0, True)
            snaps = [
                make_detector_snapshot("stall", s, False, gate_failed="min_events"),
                make_detector_snapshot("error_cascade", s, False, gate_failed="min_events"),
            ]
            trace.record(rec, snaps, hta_regression_count=0)

        timeline = trace.get_detector_timeline("stall")
        assert len(timeline) == 3
        assert all(s["detector"] == "stall" for s in timeline)

    def test_get_snapshots_at_step(self):
        trace = DecisionTrace()
        rec = make_step_record(5, "tool_call", "Read", "gathering", None, 100.0, True)
        snaps = [
            make_detector_snapshot("stall", 5, False),
            make_detector_snapshot("error_cascade", 5, True, confidence=0.7),
        ]
        trace.record(rec, snaps)

        at_5 = trace.get_snapshots_at_step(5)
        assert len(at_5) == 2

    def test_to_dict(self):
        trace = DecisionTrace()
        rec = make_step_record(1, "tool_call", "Read", "gathering", None, 100.0, True)
        trace.record(rec, [])
        d = trace.to_dict()
        assert d["total_steps"] == 1
        assert "steps" in d
        assert "detector_snapshots" in d
        assert "session_profiles" in d
        assert "current_profile" in d

    def test_reset(self):
        trace = DecisionTrace()
        rec = make_step_record(1, "tool_call", "Read", "gathering", None, 100.0, True)
        trace.record(rec, [make_detector_snapshot("stall", 1, False)])
        assert trace.to_dict()["total_steps"] == 1

        trace.reset()
        assert trace.to_dict()["total_steps"] == 0
        assert len(trace._snapshots) == 0

    def test_latest_snapshots(self):
        trace = DecisionTrace()
        for s in [1, 2, 3]:
            rec = make_step_record(s, "tool_call", "Read", "gathering", None, 100.0, True)
            snaps = [make_detector_snapshot("stall", s, False)]
            trace.record(rec, snaps)

        latest = trace.latest_snapshots(1)
        assert len(latest) == 1
        assert latest[0]["step"] == 3

        latest2 = trace.latest_snapshots(2)
        assert len(latest2) == 2


# ── Integration: detector _last_snapshot ──────────────────────────────


class TestDetectorSnapshots:
    def test_stall_snapshot_min_events(self):
        """StallDetector should populate _last_snapshot even when not firing."""
        det = StallDetector()
        events = [_event(i, "read_file", latency_ms=100.0) for i in range(3)]
        from agentdiag.hta import HTAStateMachine
        hta = HTAStateMachine()
        for e in events:
            hta.push(e)
        result = det.check(events, hta.state)
        assert result is None
        assert det._last_snapshot is not None
        assert det._last_snapshot["fired"] is False
        assert det._last_snapshot["gate_failed"] == "min_events"

    def test_stall_snapshot_no_outliers(self):
        """StallDetector with uniform latencies should report no_outliers."""
        det = StallDetector()
        events = [_event(i, "read_file", latency_ms=100.0) for i in range(15)]
        from agentdiag.hta import HTAStateMachine
        hta = HTAStateMachine()
        for e in events:
            hta.push(e)
        result = det.check(events, hta.state)
        assert result is None
        assert det._last_snapshot["gate_failed"] == "no_outliers"
        assert "threshold_ms" in det._last_snapshot["evidence_preview"]

    def test_error_cascade_snapshot_no_chains(self):
        """ErrorCascadeDetector reports no_chains_above_min when no cascade."""
        det = ErrorCascadeDetector()
        events = [
            _event(1, "read_file", success=True),
            _event(2, "bash", success=False),
            _event(3, "read_file", success=True),
            _event(4, "bash", success=False),
        ]
        from agentdiag.hta import HTAStateMachine
        hta = HTAStateMachine()
        for e in events:
            hta.push(e)
        result = det.check(events, hta.state)
        assert result is None
        assert det._last_snapshot["gate_failed"] == "no_chains_above_min"
        assert det._last_snapshot["evidence_preview"]["longest_chain"] == 1

    def test_step_repetition_snapshot_below_threshold(self):
        """StepRepetitionDetector should report max_run_below_threshold."""
        det = StepRepetitionDetector()
        events = [_event(i, "read_file") for i in range(5)]
        from agentdiag.hta import HTAStateMachine
        hta = HTAStateMachine()
        for e in events:
            hta.push(e)
        result = det.check(events, hta.state)
        assert result is None
        # With 5 events, max_run=5 but threshold=9, so should be below threshold
        snap = det._last_snapshot
        assert snap is not None
        assert snap["fired"] is False

    def test_context_loss_snapshot_no_candidates(self):
        """ContextLossDetector should report no_reread_candidates."""
        det = ContextLossDetector()
        events = [_event(i, "read_file", output_hash=f"hash_{i}") for i in range(10)]
        from agentdiag.hta import HTAStateMachine
        hta = HTAStateMachine()
        for e in events:
            hta.push(e)
        result = det.check(events, hta.state)
        assert result is None
        snap = det._last_snapshot
        assert snap is not None
        assert snap["fired"] is False


# ── Integration: run_caft_detectors_traced ────────────────────────────


class TestRunCaftDetectorsTraced:
    def test_returns_snapshots_for_all_detectors(self):
        """Should return one snapshot per detector."""
        events = [_event(i, "read_file") for i in range(3)]
        from agentdiag.hta import HTAStateMachine
        hta = HTAStateMachine()
        for e in events:
            hta.push(e)

        detectors = [StallDetector(), ErrorCascadeDetector(), StepRepetitionDetector()]
        diagnoses, snapshots = run_caft_detectors_traced(
            events, hta.state, detectors=detectors,
        )
        assert len(snapshots) == 3
        detector_names = {s["detector"] for s in snapshots}
        assert detector_names == {"stall", "error_cascade", "step_repetition"}
        assert all(s["fired"] is False for s in snapshots)

    def test_fired_detector_snapshot(self):
        """When a detector fires, snapshot should have fired=True."""
        # Create an error cascade: 4 consecutive failures
        events = [_event(i, "bash", success=False) for i in range(5)]
        from agentdiag.hta import HTAStateMachine
        hta = HTAStateMachine()
        for e in events:
            hta.push(e)

        detectors = [ErrorCascadeDetector()]
        diagnoses, snapshots = run_caft_detectors_traced(
            events, hta.state, detectors=detectors,
        )
        assert len(diagnoses) == 1
        assert len(snapshots) == 1
        assert snapshots[0]["fired"] is True
        assert snapshots[0]["confidence"] > 0


# ── Integration: MonitorEngine with decision_trace ────────────────────


class TestMonitorEngineDecisionTrace:
    def test_trace_disabled_by_default(self):
        engine = MonitorEngine(goal="test")
        assert engine.decision_trace is None

    def test_trace_enabled(self):
        engine = MonitorEngine(goal="test", decision_trace=True)
        assert engine.decision_trace is not None
        assert engine.decision_trace.to_dict()["total_steps"] == 0

    def test_trace_accumulates_on_push(self):
        engine = MonitorEngine(goal="test", decision_trace=True)
        engine.push(TraceEvent(step=1, type="tool_call", tool="read_file"))
        engine.push(TraceEvent(step=2, type="tool_call", tool="grep"))
        engine.push(TraceEvent(step=3, type="tool_call", tool="edit"))

        trace = engine.decision_trace
        d = trace.to_dict()
        assert d["total_steps"] == 3
        assert len(d["steps"]) == 3
        # Should have snapshots for each detector at each step
        assert len(d["detector_snapshots"]) > 0
        # Session profiles should track running state
        assert len(d["session_profiles"]) == 3

    def test_trace_records_hta_transition(self):
        engine = MonitorEngine(goal="test", decision_trace=True)
        # Force a transition: gathering → executing (write is strong signal)
        engine.push(TraceEvent(step=1, type="tool_call", tool="read_file"))
        engine.push(TraceEvent(step=2, type="tool_call", tool="write"))

        trace = engine.decision_trace
        steps = trace.to_dict()["steps"]
        # Step 2 should have an HTA transition
        step2 = next(s for s in steps if s["step"] == 2)
        assert step2["hta_transition"] is not None
        assert step2["hta_transition"]["to"] == "executing"

    def test_trace_session_profile_updates(self):
        engine = MonitorEngine(goal="test", decision_trace=True)
        engine.push(TraceEvent(step=1, type="tool_call", tool="read_file", latency_ms=100.0))
        engine.push(TraceEvent(step=2, type="tool_call", tool="read_file", latency_ms=200.0))

        profile = engine.decision_trace.to_dict()["current_profile"]
        assert profile["unique_tools"] == 1
        assert profile["avg_latency_ms"] == pytest.approx(150.0)

    def test_trace_reset(self):
        engine = MonitorEngine(goal="test", decision_trace=True)
        engine.push(TraceEvent(step=1, type="tool_call", tool="read_file"))
        assert engine.decision_trace.to_dict()["total_steps"] == 1

        engine.reset()
        assert engine.decision_trace.to_dict()["total_steps"] == 0

    def test_no_overhead_when_disabled(self):
        """When decision_trace=False, push() should not call traced variant."""
        engine = MonitorEngine(goal="test", decision_trace=False)
        # Push events — this should work normally without trace
        for i in range(10):
            engine.push(TraceEvent(step=i, type="tool_call", tool="read_file"))
        assert engine.state.total_events == 10
        assert engine.decision_trace is None

    def test_detector_timeline_through_engine(self):
        """End-to-end: push events, get detector timeline."""
        engine = MonitorEngine(goal="test", decision_trace=True)
        for i in range(15):
            engine.push(TraceEvent(
                step=i, type="tool_call", tool="read_file",
                latency_ms=100.0,
            ))

        timeline = engine.decision_trace.get_detector_timeline("stall")
        assert len(timeline) == 15
        # All should show stall not firing (uniform latencies)
        assert all(s["fired"] is False for s in timeline)
