"""Tests for calibrated CAFT detectors."""

import pytest

from agentdiag.models import TraceEvent
from agentdiag.hta import HTAStateMachine, HTAState, Phase
from agentdiag.baselines import (
    NormativePhaseModel,
    TransitionModel,
    ActionBaselineModel,
    CalibrationProfile,
    CalibrationPipeline,
    PhaseStats,
)
from agentdiag.caft.calibrated import (
    CalibratedStepRepetition,
    CalibratedGoalDrift,
    CalibratedMissingVerification,
    CalibratedContextLoss,
    make_calibrated_detectors,
)


# =========================================================================
# Helpers
# =========================================================================

def _build_events(specs):
    """Build TraceEvent list from compact specs.

    Each spec is (tool, event_type) or just a tool name (assumes tool_call).
    """
    events = []
    for i, spec in enumerate(specs):
        if isinstance(spec, str):
            tool, etype = spec, "tool_call"
        elif spec is None:
            tool, etype = None, "planning"
        else:
            tool, etype = spec
        events.append(TraceEvent(
            step=i + 1,
            type=etype,
            tool=tool if etype == "tool_call" else None,
            success=True,
            timestamp=float(i),
            tokens_in=10,
            tokens_out=5,
            latency_ms=100.0,
        ))
    return events


def _run_hta(events):
    """Run events through HTA and return final state."""
    sm = HTAStateMachine()
    state = None
    for e in events:
        state = sm.push(e)
    return state


def _fit_baseline_models():
    """Fit baseline models with normal data."""
    # Gathering phase: Read and Grep dominate, some repetition is normal
    gathering_stats = [
        PhaseStats(
            phase="gathering", step_count=5 + i, tool_calls=4 + i,
            unique_tools=2 + (i % 3), tool_diversity=0.5 + (i % 3) * 0.1,
            repetition_rate=0.15 + (i % 5) * 0.05,
            planning_events=0, error_count=0, error_rate=0.0,
            total_tokens=100, avg_latency_ms=200.0, duration_sec=10.0,
            action_entropy=1.0 + (i % 3) * 0.3,
        )
        for i in range(10)
    ]

    # Executing phase: Edit and Write, lower repetition
    executing_stats = [
        PhaseStats(
            phase="executing", step_count=4 + i, tool_calls=3 + i,
            unique_tools=2 + (i % 2), tool_diversity=0.5 + (i % 3) * 0.1,
            repetition_rate=0.05 + (i % 4) * 0.03,
            planning_events=0, error_count=0, error_rate=0.0,
            total_tokens=200, avg_latency_ms=300.0, duration_sec=15.0,
            action_entropy=1.0 + (i % 2) * 0.5,
        )
        for i in range(10)
    ]

    phase_model = NormativePhaseModel()
    phase_model.fit(gathering_stats + executing_stats)

    transitions = [
        ("gathering", "executing"),
        ("gathering", "executing"),
        ("gathering", "planning"),
        ("executing", "gathering"),
        ("executing", "verifying"),
        ("planning", "executing"),
    ]
    transition_model = TransitionModel(min_probability=0.05)
    transition_model.fit(transitions)

    action_model = ActionBaselineModel()
    action_model.fit({
        "gathering": [
            ["Read", "Read", "Grep"],
            ["Read", "Glob", "Read"],
            ["Grep", "Read", "Grep", "Read"],
        ],
        "executing": [
            ["Edit", "Write", "Bash"],
            ["Edit", "Edit", "Write"],
            ["Write", "Bash", "Edit"],
        ],
    })

    return phase_model, transition_model, action_model


# =========================================================================
# CalibratedStepRepetition
# =========================================================================

class TestCalibratedStepRepetition:
    def test_normal_repetition_does_not_fire(self):
        """Normal Read repetition in gathering should NOT fire."""
        phase_model, _, _ = _fit_baseline_models()
        detector = CalibratedStepRepetition(phase_model)

        # 3 consecutive Reads — normal for gathering
        events = _build_events([
            "Read", "Read", "Read", "Grep", "Read", "Grep",
        ])
        hta_state = _run_hta(events)
        result = detector.check(events, hta_state)
        assert result is None

    def test_extreme_repetition_fires(self):
        """30+ consecutive identical calls should fire."""
        phase_model, _, _ = _fit_baseline_models()
        detector = CalibratedStepRepetition(phase_model)

        events = _build_events(["Read"] * 35)
        hta_state = _run_hta(events)
        result = detector.check(events, hta_state)
        assert result is not None
        assert result.failure_name == "step_repetition"

    def test_too_few_events_returns_none(self):
        phase_model, _, _ = _fit_baseline_models()
        detector = CalibratedStepRepetition(phase_model)

        events = _build_events(["Read", "Read"])
        hta_state = _run_hta(events)
        assert detector.check(events, hta_state) is None


# =========================================================================
# CalibratedGoalDrift
# =========================================================================

class TestCalibratedGoalDrift:
    def test_normal_lifecycle_does_not_fire(self):
        """Normal Read→Edit transition should NOT fire as goal drift."""
        _, transition_model, action_model = _fit_baseline_models()
        detector = CalibratedGoalDrift(transition_model, action_model)

        events = _build_events([
            "Read", "Read", "Grep", "Read",
            None, None,  # planning
            "Edit", "Write", "Edit", "Write",
            "Bash", "Bash",
        ])
        hta_state = _run_hta(events)
        result = detector.check(events, hta_state)
        assert result is None

    def test_too_few_events_returns_none(self):
        _, transition_model, action_model = _fit_baseline_models()
        detector = CalibratedGoalDrift(transition_model, action_model)

        events = _build_events(["Read", "Read"])
        hta_state = _run_hta(events)
        assert detector.check(events, hta_state) is None


# =========================================================================
# CalibratedMissingVerification
# =========================================================================

class TestCalibratedMissingVerification:
    def test_few_executions_does_not_fire(self):
        """Small number of executions should NOT fire."""
        phase_model, _, _ = _fit_baseline_models()
        detector = CalibratedMissingVerification(phase_model)

        events = _build_events(["Edit", "Write"])
        hta_state = _run_hta(events)
        assert detector.check(events, hta_state) is None

    def test_normal_execution_count_does_not_fire(self):
        """Normal execution count (within P95) should NOT fire."""
        phase_model, _, _ = _fit_baseline_models()
        detector = CalibratedMissingVerification(phase_model)

        # Build a session within normal range
        events = _build_events([
            "Read", "Read",  # gathering
            "Edit", "Write", "Edit",  # executing — normal count
        ])
        hta_state = _run_hta(events)
        result = detector.check(events, hta_state)
        # Should not fire because exec count is within P95
        assert result is None


# =========================================================================
# CalibratedContextLoss
# =========================================================================

class TestCalibratedContextLoss:
    def test_normal_re_read_does_not_fire(self):
        """Reading same file twice with 1-2 operations between is normal."""
        _, _, action_model = _fit_baseline_models()
        detector = CalibratedContextLoss(action_model)

        events = _build_events(["Read", "Edit", "Read"])
        # Give them different output hashes (different files)
        events[0].output_hash = "abc123"
        events[2].output_hash = "def456"
        hta_state = _run_hta(events)
        assert detector.check(events, hta_state) is None

    def test_too_few_events_returns_none(self):
        _, _, action_model = _fit_baseline_models()
        detector = CalibratedContextLoss(action_model)
        events = _build_events(["Read", "Read"])
        hta_state = _run_hta(events)
        assert detector.check(events, hta_state) is None


# =========================================================================
# make_calibrated_detectors
# =========================================================================

class TestMakeCalibratedDetectors:
    def test_returns_all_detectors(self):
        phase_model, transition_model, action_model = _fit_baseline_models()
        profile = CalibrationProfile(
            phase_model=phase_model,
            transition_model=transition_model,
            action_model=action_model,
        )
        detectors = make_calibrated_detectors(profile)
        assert len(detectors) == 13

    def test_detector_names(self):
        phase_model, transition_model, action_model = _fit_baseline_models()
        profile = CalibrationProfile(
            phase_model=phase_model,
            transition_model=transition_model,
            action_model=action_model,
        )
        detectors = make_calibrated_detectors(profile)
        names = {d.name for d in detectors}
        expected = {
            "step_repetition", "context_loss",
            "premature_termination", "missing_verification",
            "reasoning_action_mismatch", "goal_drift",
            "tool_misuse", "stall", "error_cascade",
            "token_explosion", "analysis_paralysis",
            "recovery_failure", "tool_thrashing",
        }
        assert names == expected

    def test_detectors_have_check_method(self):
        phase_model, transition_model, action_model = _fit_baseline_models()
        profile = CalibrationProfile(
            phase_model=phase_model,
            transition_model=transition_model,
            action_model=action_model,
        )
        detectors = make_calibrated_detectors(profile)
        for d in detectors:
            assert hasattr(d, "check")
            assert callable(d.check)

    def test_calibrated_reduces_false_positives(self):
        """The whole point: calibrated detectors fire less on normal traces."""
        from agentdiag.caft.detectors import ALL_CAFT_DETECTORS, run_caft_detectors

        phase_model, transition_model, action_model = _fit_baseline_models()
        profile = CalibrationProfile(
            phase_model=phase_model,
            transition_model=transition_model,
            action_model=action_model,
        )
        calibrated = make_calibrated_detectors(profile)

        # Build a normal session
        events = _build_events([
            "Read", "Read", "Grep", "Read",
            None, None,
            "Edit", "Write", "Edit",
            "Bash",
        ])
        hta_state = _run_hta(events)

        # Raw detectors
        raw_results = run_caft_detectors(events, hta_state, ALL_CAFT_DETECTORS)

        # Calibrated detectors
        cal_results = run_caft_detectors(events, hta_state, calibrated)

        # Calibrated should fire <= raw (ideally strictly less)
        assert len(cal_results) <= len(raw_results)


# =========================================================================
# Integration: CalibrationPipeline → CalibratedDetectors
# =========================================================================

class TestCalibrationIntegration:
    def test_full_pipeline(self, tmp_path):
        """End-to-end: fit from sessions → save → load → make detectors."""
        # Create sessions
        sessions = [
            _build_events([
                "Read", "Read", "Grep",
                None, None,
                "Edit", "Write",
                "Bash",
            ])
            for _ in range(5)
        ]

        # Fit
        pipeline = CalibrationPipeline()
        profile = pipeline.fit_from_sessions(sessions)

        # Save
        path = tmp_path / "baselines.json"
        profile.save(path)

        # Load
        loaded = CalibrationProfile.load(path)

        # Make detectors
        detectors = make_calibrated_detectors(loaded)
        assert len(detectors) == 13

        # Run on a normal session
        test_events = _build_events([
            "Read", "Read", "Grep",
            None, None,
            "Edit", "Write",
        ])
        hta_state = _run_hta(test_events)

        from agentdiag.caft.detectors import run_caft_detectors
        results = run_caft_detectors(test_events, hta_state, detectors)
        # Normal session should have few/no detections with calibrated detectors
        # (at most 1-2 from the uncalibrated pass-through detectors)
        assert len(results) <= 3
