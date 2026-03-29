"""Tests for the end-to-end demo script (scripts/demo_e2e.py)."""

import sys
from pathlib import Path

import pytest

# Add scripts to path for import
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from demo_e2e import (
    DEMO_FAILURES,
    MockLLMConfirmer,
    build_demo_trace,
    run_demo,
)

from agentdiag.caft.base import CaftDiagnosis, CaftSeverity


class TestBuildDemoTrace:
    def test_event_count(self):
        events = build_demo_trace()
        assert len(events) == 18

    def test_events_have_sequential_steps(self):
        events = build_demo_trace()
        steps = [e.step for e in events]
        assert steps == list(range(1, 19))

    def test_planted_failures_defined(self):
        assert "premature_termination" in DEMO_FAILURES
        assert "context_loss" in DEMO_FAILURES
        assert len(DEMO_FAILURES) == 2


class TestMockLLMConfirmer:
    def test_confirms_known_failures(self):
        mock = MockLLMConfirmer()
        for name in ["premature_termination", "context_loss"]:
            diag = CaftDiagnosis(
                caft_code="test",
                caft_category="test",
                failure_name=name,
                severity=CaftSeverity.WARNING,
                confidence=0.5,
                description="test",
                evidence={},
                at_step=1,
                remediation="test",
            )
            result = mock.confirm(diag)
            assert result["confirmed"] is True
            assert result["confidence"] > 0.5

    def test_rejects_unknown_failures(self):
        mock = MockLLMConfirmer()
        diag = CaftDiagnosis(
            caft_code="test",
            caft_category="test",
            failure_name="unknown_failure",
            severity=CaftSeverity.WARNING,
            confidence=0.5,
            description="test",
            evidence={},
            at_step=1,
            remediation="test",
        )
        result = mock.confirm(diag)
        assert result["confirmed"] is False
        assert result["confidence"] < 0.5


class TestRunDemo:
    def test_detects_all_planted_failures(self):
        result = run_demo(json_output=True)
        detected_names = {d["failure_name"] for d in result["detections"]}
        for name in DEMO_FAILURES:
            assert name in detected_names, f"Expected {name} to be detected"

    def test_json_output_structure(self):
        result = run_demo(json_output=True)
        assert "trace_events" in result
        assert result["trace_events"] == 18
        assert "planted_failures" in result
        assert "phases_seen" in result
        assert "detections" in result
        assert "hta_transitions" in result
        assert "trust_score" in result
        assert "health" in result

    def test_trust_score_degraded(self):
        """Detections should degrade trust score below 1.0."""
        result = run_demo(json_output=True)
        assert result["trust_score"] < 1.0

    def test_phases_seen(self):
        """Demo trace should visit gathering, planning, executing, delivering."""
        result = run_demo(json_output=True)
        phases = set(result["phases_seen"])
        assert "gathering" in phases
        assert "executing" in phases
        assert "delivering" in phases

    def test_with_confirm_mode(self):
        """Mock LLM confirmation should confirm all planted failures."""
        result = run_demo(json_output=True, confirm=True)
        assert result["confirm_mode"] is True
        confirmed = [d for d in result["detections"] if d.get("confirmed") is True]
        assert len(confirmed) >= 2

    def test_hta_transitions_present(self):
        """Should have at least 3 HTA transitions (gathering→planning→executing→delivering)."""
        result = run_demo(json_output=True)
        assert len(result["hta_transitions"]) >= 3

    def test_custom_trace(self):
        """Custom trace should work without planted failure comparison."""
        from agentdiag.models import TraceEvent
        custom = [
            TraceEvent(step=1, type="tool_call", tool="read_file"),
            TraceEvent(step=2, type="tool_call", tool="edit_file"),
            TraceEvent(step=3, type="tool_call", tool="git_commit"),
        ]
        result = run_demo(trace=custom, json_output=True)
        assert result["trace_events"] == 3
        assert result["planted_failures"] == []

    def test_detection_fields(self):
        """Each detection should have required fields."""
        result = run_demo(json_output=True)
        for det in result["detections"]:
            assert "failure_name" in det
            assert "caft_code" in det
            assert "at_step" in det
            assert "confidence" in det
            assert isinstance(det["confidence"], float)
