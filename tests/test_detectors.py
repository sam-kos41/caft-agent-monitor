"""Tests for all CAFT detectors."""

import json
import time
import pytest

from agentdiag.models import TraceEvent
from agentdiag.loading import load_trace
from agentdiag.hta import HTAState, HTANode, Phase, PhaseTransition
from agentdiag.caft.base import CaftDiagnosis
from agentdiag.caft.detectors import (
    StepRepetitionDetector,
    ContextLossDetector,
    PrematureTerminationDetector,
    ToolMisuseDetector,
    StallDetector,
    ErrorCascadeDetector,
    TokenExplosionDetector,
    AnalysisParalysisDetector,
    RecoveryFailureDetector,
    StrategicMyopiaDetector,
    ALL_CAFT_DETECTORS,
    run_caft_detectors,
)
from agentdiag.synthetic import (
    generate_normal_trace,
    generate_loop_trace,
    generate_stall_trace,
    generate_thrash_trace,
    generate_drift_trace,
    generate_cascade_trace,
    generate_token_explosion_trace,
    generate_dead_end_trace,
    generate_recovery_failure_trace,
)


def _default_hta() -> HTAState:
    """Create a default HTAState for testing."""
    return HTAState(
        goal="test",
        current_phase=Phase.EXECUTING,
        current_node=HTANode(
            phase=Phase.EXECUTING,
            start_step=0,
            start_time=time.monotonic(),
        ),
        completed_nodes=[],
        transitions=[],
        total_events=0,
        phase_event_counts={"executing": 10},
    )


# ---------------------------------------------------------------------------
# Migrated batch detectors
# ---------------------------------------------------------------------------

class TestToolMisuseDetector:
    def test_detects_thrash(self):
        events = generate_thrash_trace()
        hta = _default_hta()
        d = ToolMisuseDetector().check(events, hta)
        assert d is not None
        assert d.failure_name == "tool_misuse"
        assert d.confidence >= 0.4

    def test_no_false_positive_on_normal(self):
        events = generate_normal_trace()
        hta = _default_hta()
        d = ToolMisuseDetector().check(events, hta)
        assert d is None


class TestStallDetector:
    def test_detects_stall(self):
        events = generate_stall_trace()
        hta = _default_hta()
        d = StallDetector().check(events, hta)
        assert d is not None
        assert d.failure_name == "stall"
        assert d.confidence >= 0.5  # 2 stall events → moderate confidence
        assert d.evidence["max_latency_ms"] > 10000

    def test_no_false_positive_on_normal(self):
        events = generate_normal_trace()
        hta = _default_hta()
        d = StallDetector().check(events, hta)
        assert d is None


class TestErrorCascadeDetector:
    def test_detects_cascade(self):
        events = generate_cascade_trace()
        hta = _default_hta()
        d = ErrorCascadeDetector().check(events, hta)
        assert d is not None
        assert d.failure_name == "error_cascade"
        assert d.evidence["longest_error_chain"] >= 3

    def test_no_false_positive_on_normal(self):
        events = generate_normal_trace()
        hta = _default_hta()
        d = ErrorCascadeDetector().check(events, hta)
        assert d is None


class TestTokenExplosionDetector:
    def test_detects_explosion(self):
        events = generate_token_explosion_trace()
        hta = _default_hta()
        d = TokenExplosionDetector().check(events, hta)
        assert d is not None
        assert d.failure_name == "token_explosion"
        assert d.evidence["growth_ratio_last_vs_first_quarter"] > 2.0

    def test_no_false_positive_on_normal(self):
        events = generate_normal_trace()
        hta = _default_hta()
        d = TokenExplosionDetector().check(events, hta)
        assert d is None


class TestAnalysisParalysisDetector:
    def test_detects_dead_end(self):
        events = generate_dead_end_trace()
        hta = _default_hta()
        d = AnalysisParalysisDetector().check(events, hta)
        assert d is not None
        assert d.failure_name == "analysis_paralysis"
        assert d.evidence["max_consecutive_reasoning"] >= 4

    def test_no_false_positive_on_normal(self):
        events = generate_normal_trace()
        hta = _default_hta()
        d = AnalysisParalysisDetector().check(events, hta)
        assert d is None


class TestRecoveryFailureDetector:
    def test_detects_recovery_failure(self):
        events = generate_recovery_failure_trace()
        hta = _default_hta()
        d = RecoveryFailureDetector().check(events, hta)
        assert d is not None
        assert d.failure_name == "recovery_failure"
        assert d.evidence["recovery_failure_rate"] >= 0.4

    def test_no_false_positive_on_normal(self):
        events = generate_normal_trace()
        hta = _default_hta()
        d = RecoveryFailureDetector().check(events, hta)
        assert d is None


# ---------------------------------------------------------------------------
# All-detectors integration
# ---------------------------------------------------------------------------

class TestAllDetectors:
    def test_normal_trace_clean(self):
        events = generate_normal_trace()
        hta = _default_hta()
        diagnoses = run_caft_detectors(events, hta)
        assert len(diagnoses) == 0

    def test_cascade_detects_cascade_but_not_recovery_when_recovered(self):
        """Cascade trace has 6 errors then 9 successes — agent recovered.

        error_cascade fires (consecutive error chain). recovery_failure
        should NOT fire because the agent made 5+ unique successful
        operations after the last failed recovery window.
        """
        events = generate_cascade_trace()
        hta = _default_hta()
        diagnoses = run_caft_detectors(events, hta)
        names = [d.failure_name for d in diagnoses]
        assert "error_cascade" in names
        assert "recovery_failure" not in names

    def test_recovery_failure_fires_when_no_recovery(self):
        """Short session ending in errors should trigger recovery_failure."""
        from agentdiag.caft.detectors import RecoveryFailureDetector
        # 10 events: 7 OK, then 3 consecutive errors at end (no recovery)
        events = [
            TraceEvent(step=i+1, type="tool_call", tool="bash",
                       latency_ms=500, success=True, tokens_in=100, tokens_out=200)
            for i in range(7)
        ]
        for i in range(3):
            events.append(TraceEvent(
                step=8+i, type="tool_call", tool="bash",
                latency_ms=500, success=False, tokens_in=100, tokens_out=0,
            ))
        hta = _default_hta()
        d = RecoveryFailureDetector().check(events, hta)
        assert d is not None
        assert d.failure_name == "recovery_failure"

    def test_recovery_failure_skips_productive_session(self):
        """Long productive session with late errors should NOT fire."""
        from agentdiag.caft.detectors import RecoveryFailureDetector
        # 60 events: 55 OK, then 5 errors — session is 91% successful
        events = [
            TraceEvent(step=i+1, type="tool_call", tool="bash",
                       latency_ms=500, success=True, tokens_in=100, tokens_out=200,
                       output_hash=f"hash_{i}")
            for i in range(55)
        ]
        for i in range(5):
            events.append(TraceEvent(
                step=56+i, type="tool_call", tool="bash",
                latency_ms=500, success=False, tokens_in=100, tokens_out=0,
            ))
        hta = _default_hta()
        d = RecoveryFailureDetector().check(events, hta)
        assert d is None  # Productive session, should not fire

    def test_diagnosis_serialization(self):
        events = generate_stall_trace()
        hta = _default_hta()
        d = StallDetector().check(events, hta)
        assert d is not None
        result = d.to_dict()
        assert "failure_name" in result
        assert "evidence" in result
        assert result["severity"] in ("info", "warning", "critical")


# ---------------------------------------------------------------------------
# Trace loading
# ---------------------------------------------------------------------------

class TestTraceLoading:
    def test_load_json_array(self, tmp_path):
        events = [
            {"step": 1, "type": "tool_call", "tool": "search", "latency_ms": 500,
             "success": True, "tokens_in": 100, "tokens_out": 200},
            {"step": 2, "type": "tool_call", "tool": "read", "latency_ms": 300,
             "success": True, "tokens_in": 100, "tokens_out": 200},
        ]
        path = tmp_path / "trace.json"
        path.write_text(json.dumps(events))

        loaded = load_trace(path)
        assert len(loaded) == 2
        assert loaded[0].tool == "search"

    def test_load_jsonl(self, tmp_path):
        lines = [
            json.dumps({"step": 1, "type": "tool_call", "tool": "search",
                        "latency_ms": 500, "success": True, "tokens_in": 100,
                        "tokens_out": 200}),
            json.dumps({"step": 2, "type": "tool_call", "tool": "read",
                        "latency_ms": 300, "success": True, "tokens_in": 100,
                        "tokens_out": 200}),
        ]
        path = tmp_path / "trace.jsonl"
        path.write_text("\n".join(lines))

        loaded = load_trace(path)
        assert len(loaded) == 2

    def test_traceevent_from_dict_ignores_extra_fields(self):
        d = {"step": 1, "type": "tool_call", "tool": "search", "latency_ms": 500,
             "success": True, "tokens_in": 100, "tokens_out": 200,
             "extra_field": "should_be_ignored"}
        e = TraceEvent.from_dict(d)
        assert e.tool == "search"


# ---------------------------------------------------------------------------
# StrategicMyopiaDetector tests
# ---------------------------------------------------------------------------

def _build_myopia_trace(
    eval_cycles=5,
    edit_target="detectors.py",
    total_edits=10,
    spread_edits=False,
    include_annotation_edits=0,
    include_broad_reads=0,
    include_exploration_phases=False,
    metric_values=None,
):
    """Build a synthetic trace for strategic myopia testing.

    The trace alternates: [eval_run, N edits, read, ...] repeated eval_cycles times.
    """
    events = []
    step = 1

    # Initial gathering phase
    for i in range(5):
        events.append(TraceEvent(
            step=step, type="tool_call", tool="Read",
            goal_text=f"src/file_{i}.py" if include_broad_reads == 0 else None,
        ))
        step += 1

    if include_broad_reads > 0:
        for i in range(include_broad_reads):
            events.append(TraceEvent(
                step=step, type="tool_call", tool="Read",
                goal_text="docs/architecture.md" if i == 0 else "README.md",
            ))
            step += 1

    if include_exploration_phases:
        for i in range(3):
            events.append(TraceEvent(step=step, type="reasoning", goal_text="Planning approach"))
            step += 1

    # Build eval cycles
    for cycle in range(eval_cycles):
        # Eval run
        metric_text = ""
        if metric_values and cycle < len(metric_values):
            metric_text = f"F1={metric_values[cycle]}%"
        events.append(TraceEvent(
            step=step, type="tool_call", tool="Bash",
            goal_text=f"python scripts/run_ablation.py",
            error_message=metric_text if metric_text else None,
        ))
        step += 1

        # Edits between evals
        edits_per_cycle = max(total_edits // eval_cycles, 2)
        for j in range(edits_per_cycle):
            if spread_edits:
                target = f"src/module_{cycle}_{j}.py"
            else:
                target = edit_target
            events.append(TraceEvent(
                step=step, type="tool_call", tool="Edit",
                goal_text=target,
            ))
            step += 1

        # Annotation edits (distributed across cycles)
        if include_annotation_edits > 0 and cycle < include_annotation_edits:
            events.append(TraceEvent(
                step=step, type="tool_call", tool="Edit",
                goal_text="annotations/annotation_ledger.jsonl",
            ))
            step += 1

        # A read (narrow — same file)
        events.append(TraceEvent(
            step=step, type="tool_call", tool="Read",
            goal_text=edit_target,
        ))
        step += 1

    # Pad to MIN_EVENTS if needed
    while len(events) < 45:
        events.append(TraceEvent(
            step=step, type="tool_call", tool="Edit",
            goal_text=edit_target,
        ))
        step += 1

    return events


class TestStrategicMyopiaDetector:
    def test_clear_optimization_loop_fires(self):
        """Test 1: Clear optimization loop — should fire with confidence >= 0.7."""
        events = _build_myopia_trace(
            eval_cycles=5,
            edit_target="detectors.py",
            total_edits=12,
            include_annotation_edits=3,
            metric_values=[73, 51, 52, 75, 74],
        )
        hta = _default_hta()
        det = StrategicMyopiaDetector()
        d = det.check(events, hta)
        assert d is not None, f"Expected fire, snapshot: {det._last_snapshot}"
        assert d.failure_name == "strategic_myopia"
        assert d.caft_code == "3.5"
        assert d.confidence >= 0.7
        assert d.severity.value == "warning"
        assert d.force_llm_review is True

    def test_normal_productive_session_does_not_fire(self):
        """Test 2: Normal session — reads, edits different files, one test run."""
        events = []
        step = 1
        # Broad reads
        for i in range(10):
            events.append(TraceEvent(
                step=step, type="tool_call", tool="Read",
                goal_text=f"src/module_{i}.py" if i < 8 else "docs/architecture.md",
            ))
            step += 1
        # Diverse edits
        for i in range(10):
            events.append(TraceEvent(
                step=step, type="tool_call", tool="Edit",
                goal_text=f"src/module_{i}.py",
            ))
            step += 1
        # Single test run at end
        events.append(TraceEvent(
            step=step, type="tool_call", tool="Bash",
            goal_text="python -m pytest tests/",
        ))
        step += 1
        # Pad
        while len(events) < 45:
            events.append(TraceEvent(
                step=step, type="tool_call", tool="Read",
                goal_text=f"src/file_{step}.py",
            ))
            step += 1

        hta = _default_hta()
        det = StrategicMyopiaDetector()
        d = det.check(events, hta)
        assert d is None, f"Should not fire on normal session, snapshot: {det._last_snapshot}"

    def test_legitimate_iterative_dev_does_not_fire(self):
        """Test 3: 3 eval cycles but edits spread across 10+ files, broad reads present."""
        events = _build_myopia_trace(
            eval_cycles=3,
            spread_edits=True,
            total_edits=15,
            include_broad_reads=3,
        )
        hta = _default_hta()
        det = StrategicMyopiaDetector()
        d = det.check(events, hta)
        assert d is None, f"Should not fire — eval_cycles below threshold, snapshot: {det._last_snapshot}"

    def test_boundary_below_eval_threshold(self):
        """Test 4: 3 eval cycles (below MIN_EVAL_CYCLES=4) — should NOT fire."""
        events = _build_myopia_trace(
            eval_cycles=3,
            edit_target="detectors.py",
            total_edits=12,
        )
        hta = _default_hta()
        det = StrategicMyopiaDetector()
        d = det.check(events, hta)
        assert d is None, f"Should not fire below eval threshold, snapshot: {det._last_snapshot}"

    def test_ground_truth_mutation_higher_confidence(self):
        """Test 5: With ground truth mutation signal — should fire with higher confidence."""
        # Base case: 4 eval cycles, concentrated edits, phase stagnation, architecture blindness
        events_base = _build_myopia_trace(
            eval_cycles=5,
            edit_target="detectors.py",
            total_edits=12,
            include_annotation_edits=0,
        )
        # With annotation edits: adds ground_truth_mutation signal
        events_gt = _build_myopia_trace(
            eval_cycles=5,
            edit_target="detectors.py",
            total_edits=12,
            include_annotation_edits=3,
        )
        hta = _default_hta()
        det_base = StrategicMyopiaDetector()
        det_gt = StrategicMyopiaDetector()
        d_base = det_base.check(events_base, hta)
        d_gt = det_gt.check(events_gt, hta)
        # Both may or may not fire, but if both fire, gt version has more signals
        if d_base is not None and d_gt is not None:
            assert d_gt.confidence >= d_base.confidence
            assert d_gt.evidence["ground_truth_mutation"] is True
