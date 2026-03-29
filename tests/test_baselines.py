"""Tests for normative baseline models and calibration pipeline."""

import json
import math

import pytest

from agentdiag.models import TraceEvent
from agentdiag.hta import Phase
from agentdiag.baselines import (
    PhaseStats,
    PhaseDistribution,
    NormativePhaseModel,
    TransitionModel,
    ActionBaselineModel,
    CalibrationProfile,
    CalibrationPipeline,
    extract_phase_segments,
    compute_phase_stats,
    _shannon_entropy,
    PHASE_METRICS,
)


# =========================================================================
# Helpers
# =========================================================================

def _make_events(specs):
    """Build TraceEvent list from compact specs.

    Each spec is (tool, event_type) or (tool, event_type, success).
    """
    events = []
    for i, spec in enumerate(specs):
        if len(spec) == 2:
            tool, etype = spec
            success = True
        else:
            tool, etype, success = spec
        events.append(TraceEvent(
            step=i + 1,
            type=etype,
            tool=tool if etype == "tool_call" else None,
            success=success,
            timestamp=float(i),
            tokens_in=10,
            tokens_out=5,
            latency_ms=100.0,
        ))
    return events


def _make_session_events():
    """Build a realistic session: gather → plan → execute → verify."""
    return _make_events([
        ("Read", "tool_call"),
        ("Read", "tool_call"),
        ("Grep", "tool_call"),
        (None, "planning"),
        (None, "planning"),
        ("Edit", "tool_call"),
        ("Write", "tool_call"),
        ("Edit", "tool_call"),
        ("Bash", "tool_call"),
        ("Bash", "tool_call"),    # test run
    ])


def _make_normal_session():
    """Build a normal session for baseline fitting."""
    return _make_events([
        ("Read", "tool_call"),
        ("Grep", "tool_call"),
        ("Read", "tool_call"),
        (None, "planning"),
        (None, "reasoning"),
        ("Edit", "tool_call"),
        ("Write", "tool_call"),
        ("Edit", "tool_call"),
        ("Bash", "tool_call"),
    ])


def _make_diverse_session():
    """Build a session with more tool variety."""
    return _make_events([
        ("Read", "tool_call"),
        ("Glob", "tool_call"),
        ("Grep", "tool_call"),
        ("Read", "tool_call"),
        (None, "planning"),
        ("Write", "tool_call"),
        ("Edit", "tool_call"),
        ("Bash", "tool_call"),
        ("Write", "tool_call"),
    ])


def _make_repetitive_session():
    """Build a session with extreme repetition (anomalous)."""
    return _make_events([
        ("Read", "tool_call"),
        ("Read", "tool_call"),
        ("Read", "tool_call"),
        ("Read", "tool_call"),
        ("Read", "tool_call"),
        ("Read", "tool_call"),
        ("Read", "tool_call"),
        ("Read", "tool_call"),
        ("Read", "tool_call"),
        ("Read", "tool_call"),
    ])


# =========================================================================
# Shannon entropy
# =========================================================================

class TestShannonEntropy:
    def test_empty(self):
        assert _shannon_entropy({}) == 0.0

    def test_single_item(self):
        assert _shannon_entropy({"a": 10}) == 0.0

    def test_uniform_two(self):
        e = _shannon_entropy({"a": 1, "b": 1})
        assert abs(e - 1.0) < 0.01  # log2(2) = 1.0

    def test_uniform_four(self):
        e = _shannon_entropy({"a": 1, "b": 1, "c": 1, "d": 1})
        assert abs(e - 2.0) < 0.01  # log2(4) = 2.0

    def test_skewed(self):
        e = _shannon_entropy({"a": 99, "b": 1})
        assert 0.0 < e < 0.5  # very skewed → low entropy


# =========================================================================
# Phase segment extraction
# =========================================================================

class TestPhaseSegments:
    def test_basic_segments(self):
        events = _make_session_events()
        segments = extract_phase_segments(events)
        assert len(segments) >= 2  # at least gathering + executing

        phases = [s[0] for s in segments]
        # Should see gathering, then executing (Write is strong signal)
        assert "gathering" in phases

    def test_empty_events(self):
        assert extract_phase_segments([]) == []

    def test_segments_cover_all_events(self):
        events = _make_session_events()
        segments = extract_phase_segments(events)
        total_events = sum(len(evts) for _, evts in segments)
        assert total_events == len(events)


class TestComputePhaseStats:
    def test_basic_stats(self):
        events = _make_events([
            ("Read", "tool_call"),
            ("Read", "tool_call"),
            ("Grep", "tool_call"),
        ])
        stats = compute_phase_stats("gathering", events)
        assert stats.phase == "gathering"
        assert stats.step_count == 3
        assert stats.tool_calls == 3
        assert stats.unique_tools == 2  # Read, Grep
        assert 0.0 < stats.tool_diversity <= 1.0

    def test_repetition_rate(self):
        events = _make_events([
            ("Read", "tool_call"),
            ("Read", "tool_call"),
            ("Read", "tool_call"),
        ])
        stats = compute_phase_stats("gathering", events)
        # 2 consecutive repeats / 3 tools ≈ 0.667
        assert stats.repetition_rate > 0.5

    def test_no_repetition(self):
        events = _make_events([
            ("Read", "tool_call"),
            ("Grep", "tool_call"),
            ("Glob", "tool_call"),
        ])
        stats = compute_phase_stats("gathering", events)
        assert stats.repetition_rate == 0.0

    def test_entropy(self):
        events = _make_events([
            ("Read", "tool_call"),
            ("Grep", "tool_call"),
            ("Glob", "tool_call"),
        ])
        stats = compute_phase_stats("gathering", events)
        # 3 unique tools, uniform → entropy ≈ log2(3) ≈ 1.58
        assert stats.action_entropy > 1.0

    def test_error_rate(self):
        events = _make_events([
            ("Read", "tool_call", True),
            ("Read", "tool_call", False),
            ("Read", "tool_call", True),
        ])
        stats = compute_phase_stats("gathering", events)
        assert abs(stats.error_rate - 1 / 3) < 0.01


# =========================================================================
# PhaseDistribution
# =========================================================================

class TestPhaseDistribution:
    def test_fit(self):
        dist = PhaseDistribution(
            phase="gathering",
            metric="step_count",
            values=[5, 10, 15, 20, 25],
        )
        dist.fit()
        assert dist.n == 5
        assert dist.mean == 15.0
        assert dist.p50 == 15.0  # median
        assert dist.p5 < dist.p50 < dist.p95

    def test_is_anomalous(self):
        dist = PhaseDistribution(
            phase="gathering",
            metric="step_count",
            values=[5, 10, 15, 20, 25],
        )
        dist.fit()
        assert not dist.is_anomalous(15)  # median
        assert dist.is_anomalous(100)     # way beyond p95

    def test_roundtrip(self):
        dist = PhaseDistribution(
            phase="gathering",
            metric="step_count",
            values=[1, 2, 3, 4, 5],
        )
        dist.fit()
        d = dist.to_dict()
        dist2 = PhaseDistribution.from_dict(d)
        assert dist2.phase == "gathering"
        assert dist2.p50 == dist.p50
        assert dist2.n == 5

    def test_empty_values(self):
        dist = PhaseDistribution(
            phase="x", metric="y", values=[],
        )
        dist.fit()
        assert dist.n == 0
        assert dist.mean == 0.0


# =========================================================================
# NormativePhaseModel
# =========================================================================

class TestNormativePhaseModel:
    def _make_phase_stats(self, n=10):
        """Generate n phase stats for fitting."""
        stats = []
        for i in range(n):
            stats.append(PhaseStats(
                phase="gathering",
                step_count=5 + i,
                tool_calls=4 + i,
                unique_tools=2 + (i % 3),
                tool_diversity=0.5 + (i % 5) * 0.1,
                repetition_rate=0.1 + (i % 4) * 0.05,
                planning_events=0,
                error_count=0,
                error_rate=0.0,
                total_tokens=100,
                avg_latency_ms=200.0,
                duration_sec=10.0,
                action_entropy=1.0 + (i % 3) * 0.5,
            ))
        for i in range(n):
            stats.append(PhaseStats(
                phase="executing",
                step_count=3 + i,
                tool_calls=3 + i,
                unique_tools=2 + (i % 2),
                tool_diversity=0.4 + (i % 3) * 0.15,
                repetition_rate=0.05 + (i % 5) * 0.03,
                planning_events=0,
                error_count=i % 3,
                error_rate=(i % 3) / max(3 + i, 1),
                total_tokens=200,
                avg_latency_ms=300.0,
                duration_sec=20.0,
                action_entropy=0.8 + (i % 4) * 0.3,
            ))
        return stats

    def test_fit(self):
        model = NormativePhaseModel()
        stats = self._make_phase_stats()
        model.fit(stats)
        assert model.is_fitted
        assert "gathering" in model.get_phases()
        assert "executing" in model.get_phases()

    def test_not_fitted_returns_false(self):
        model = NormativePhaseModel()
        assert not model.is_fitted
        assert not model.is_anomalous("gathering", "step_count", 100)

    def test_anomalous_value(self):
        model = NormativePhaseModel()
        stats = self._make_phase_stats()
        model.fit(stats)
        # Very high step count should be anomalous
        assert model.is_anomalous("gathering", "step_count", 1000)
        # Normal step count should not be anomalous
        assert not model.is_anomalous("gathering", "step_count", 7)

    def test_score_phase(self):
        model = NormativePhaseModel()
        stats = self._make_phase_stats()
        model.fit(stats)

        normal = PhaseStats(
            phase="gathering", step_count=7, tool_calls=6,
            unique_tools=3, tool_diversity=0.5, repetition_rate=0.15,
            planning_events=0, error_count=0, error_rate=0.0,
            total_tokens=100, avg_latency_ms=200.0, duration_sec=10.0,
            action_entropy=1.5,
        )
        result = model.score_phase(normal)
        # Normal stats should have few anomalies
        assert isinstance(result, dict)
        assert "step_count" in result

    def test_roundtrip(self):
        model = NormativePhaseModel()
        model.fit(self._make_phase_stats())
        d = model.to_dict()
        model2 = NormativePhaseModel.from_dict(d)
        assert model2.is_fitted
        assert model2.get_phases() == model.get_phases()

    def test_get_distribution(self):
        model = NormativePhaseModel()
        model.fit(self._make_phase_stats())
        dist = model.get_distribution("gathering", "step_count")
        assert dist is not None
        assert dist.n == 10
        assert dist.mean > 0


# =========================================================================
# TransitionModel
# =========================================================================

class TestTransitionModel:
    def test_fit(self):
        transitions = [
            ("gathering", "executing"),
            ("gathering", "executing"),
            ("gathering", "planning"),
            ("executing", "verifying"),
            ("executing", "gathering"),
        ]
        model = TransitionModel()
        model.fit(transitions)
        assert model.is_fitted

    def test_probability(self):
        transitions = [
            ("gathering", "executing"),
            ("gathering", "executing"),
            ("gathering", "planning"),
        ]
        model = TransitionModel()
        model.fit(transitions)
        # 2/3 transitions from gathering go to executing
        p = model.probability("gathering", "executing")
        assert abs(p - 2 / 3) < 0.01
        # 1/3 go to planning
        p = model.probability("gathering", "planning")
        assert abs(p - 1 / 3) < 0.01

    def test_anomalous_transition(self):
        transitions = [
            ("gathering", "executing"),
            ("gathering", "executing"),
            ("gathering", "executing"),
            ("gathering", "executing"),
        ]
        model = TransitionModel(min_probability=0.05)
        model.fit(transitions)
        # Never seen gathering → delivering
        assert model.is_anomalous("gathering", "delivering")
        # Common transition
        assert not model.is_anomalous("gathering", "executing")

    def test_unknown_phase_not_anomalous(self):
        model = TransitionModel()
        model.fit([("a", "b")])
        # Unknown source phase → can't judge
        assert not model.is_anomalous("unknown", "b")

    def test_expected_next(self):
        transitions = [
            ("gathering", "executing"),
            ("gathering", "executing"),
            ("gathering", "planning"),
        ]
        model = TransitionModel()
        model.fit(transitions)
        expected = model.get_expected_next("gathering")
        assert expected[0][0] == "executing"  # most likely

    def test_roundtrip(self):
        model = TransitionModel(min_probability=0.1)
        model.fit([("a", "b"), ("a", "c"), ("b", "c")])
        d = model.to_dict()
        model2 = TransitionModel.from_dict(d)
        assert model2.is_fitted
        assert model2.probability("a", "b") == model.probability("a", "b")
        assert model2._min_probability == 0.1

    def test_not_fitted(self):
        model = TransitionModel()
        assert not model.is_fitted
        assert not model.is_anomalous("a", "b")


# =========================================================================
# ActionBaselineModel
# =========================================================================

class TestActionBaselineModel:
    def test_fit(self):
        phase_events = {
            "gathering": [["Read", "Read", "Grep"], ["Read", "Glob"]],
            "executing": [["Edit", "Write", "Bash"], ["Edit", "Edit"]],
        }
        model = ActionBaselineModel()
        model.fit(phase_events)
        assert model.is_fitted

    def test_tool_frequency(self):
        phase_events = {
            "gathering": [["Read", "Read", "Grep"]],
        }
        model = ActionBaselineModel()
        model.fit(phase_events)
        # Read appears 2/3 times
        freq = model.get_tool_frequency("gathering", "Read")
        assert abs(freq - 2 / 3) < 0.01

    def test_novel_tool(self):
        phase_events = {
            "gathering": [["Read", "Grep"]],
        }
        model = ActionBaselineModel()
        model.fit(phase_events)
        assert model.is_novel_tool("gathering", "CustomTool")
        assert not model.is_novel_tool("gathering", "Read")

    def test_get_novel_tools(self):
        phase_events = {
            "gathering": [["Read", "Grep"]],
        }
        model = ActionBaselineModel()
        model.fit(phase_events)
        novel = model.get_novel_tools("gathering", ["Read", "Write", "Custom"])
        assert "Write" in novel
        assert "Custom" in novel
        assert "Read" not in novel

    def test_tool_divergence_identical(self):
        phase_events = {
            "gathering": [["Read", "Grep", "Read"]],
        }
        model = ActionBaselineModel()
        model.fit(phase_events)
        # Same distribution → divergence ≈ 0
        div = model.compute_tool_divergence("gathering", ["Read", "Grep", "Read"])
        assert div < 0.01

    def test_tool_divergence_different(self):
        phase_events = {
            "gathering": [["Read", "Read", "Read"]],
        }
        model = ActionBaselineModel()
        model.fit(phase_events)
        # Completely different tools → high divergence
        div = model.compute_tool_divergence("gathering", ["Write", "Write", "Write"])
        assert div > 0.5

    def test_roundtrip(self):
        phase_events = {
            "gathering": [["Read", "Grep"]],
            "executing": [["Edit", "Write"]],
        }
        model = ActionBaselineModel()
        model.fit(phase_events)
        d = model.to_dict()
        model2 = ActionBaselineModel.from_dict(d)
        assert model2.is_fitted
        assert model2.get_tool_frequency("gathering", "Read") == model.get_tool_frequency("gathering", "Read")

    def test_unknown_phase(self):
        model = ActionBaselineModel()
        model.fit({"a": [["x"]]})
        # Unknown phase → no novel tools (can't judge)
        assert model.get_novel_tools("unknown", ["x"]) == []
        assert model.compute_tool_divergence("unknown", ["x"]) == 0.0


# =========================================================================
# CalibrationProfile (save/load)
# =========================================================================

class TestCalibrationProfile:
    def test_save_and_load(self, tmp_path):
        # Create a profile
        phase_model = NormativePhaseModel()
        phase_model.fit([
            PhaseStats(
                phase="gathering", step_count=10, tool_calls=8,
                unique_tools=3, tool_diversity=0.375,
                repetition_rate=0.25, planning_events=0,
                error_count=0, error_rate=0.0, total_tokens=100,
                avg_latency_ms=200.0, duration_sec=10.0,
                action_entropy=1.5,
            ),
        ])

        transition_model = TransitionModel()
        transition_model.fit([("gathering", "executing")])

        action_model = ActionBaselineModel()
        action_model.fit({"gathering": [["Read", "Grep"]]})

        profile = CalibrationProfile(
            phase_model=phase_model,
            transition_model=transition_model,
            action_model=action_model,
            n_sessions=5,
            n_phase_segments=10,
            n_transitions=8,
        )

        path = tmp_path / "baselines.json"
        profile.save(path)
        assert path.exists()

        # Load and verify
        loaded = CalibrationProfile.load(path)
        assert loaded.n_sessions == 5
        assert loaded.phase_model.is_fitted
        assert loaded.transition_model.is_fitted
        assert loaded.action_model.is_fitted


# =========================================================================
# CalibrationPipeline
# =========================================================================

class TestCalibrationPipeline:
    def test_fit_from_sessions(self):
        sessions = [
            _make_normal_session(),
            _make_diverse_session(),
            _make_normal_session(),
        ]
        pipeline = CalibrationPipeline()
        profile = pipeline.fit_from_sessions(sessions)

        assert profile.n_sessions == 3
        assert profile.n_phase_segments > 0
        assert profile.phase_model.is_fitted
        assert profile.transition_model.is_fitted
        assert profile.action_model.is_fitted

    def test_fit_empty_sessions(self):
        pipeline = CalibrationPipeline()
        profile = pipeline.fit_from_sessions([])
        assert profile.n_sessions == 0
        assert not profile.phase_model.is_fitted

    def test_fit_and_query(self):
        sessions = [
            _make_normal_session(),
            _make_normal_session(),
            _make_diverse_session(),
        ]
        pipeline = CalibrationPipeline()
        profile = pipeline.fit_from_sessions(sessions)

        # Normal values should not be anomalous
        phases = profile.phase_model.get_phases()
        assert len(phases) > 0

    def test_save_and_reload(self, tmp_path):
        sessions = [_make_normal_session() for _ in range(5)]
        pipeline = CalibrationPipeline()
        profile = pipeline.fit_from_sessions(sessions)

        path = tmp_path / "test_baselines.json"
        profile.save(path)

        loaded = CalibrationProfile.load(path)
        assert loaded.n_sessions == 5
        assert loaded.phase_model.get_phases() == profile.phase_model.get_phases()
