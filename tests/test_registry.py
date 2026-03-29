"""Tests for the CAFT detector registry."""

import time
from typing import Optional

from agentdiag.models import TraceEvent
from agentdiag.hta import HTAState, HTANode, Phase
from agentdiag.caft.base import CaftDetector, CaftDiagnosis, CaftSeverity
from agentdiag.caft.registry import detector_registry, DetectorRegistry


def _default_hta() -> HTAState:
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


class TestRegistry:
    def test_builtin_detectors_registered(self):
        names = set(detector_registry.names())
        # Should include both live and migrated detectors
        expected = {
            "step_repetition", "context_loss", "premature_termination",
            "tool_misuse", "stall", "error_cascade", "token_explosion",
            "analysis_paralysis", "recovery_failure",
            # Disabled but registered
            "missing_verification", "reasoning_action_mismatch",
            "goal_drift", "tool_thrashing",
        }
        assert expected == names

    def test_get_by_name(self):
        det = detector_registry.get("step_repetition")
        assert det.caft_code == "2.2"

    def test_get_unknown_raises(self):
        import pytest
        with pytest.raises(KeyError):
            detector_registry.get("nonexistent")

    def test_enabled_excludes_disabled(self):
        enabled_names = {d.name for d in detector_registry.get_enabled()}
        assert "missing_verification" not in enabled_names
        assert "goal_drift" not in enabled_names
        assert "step_repetition" in enabled_names

    def test_register_custom_detector(self):
        registry = DetectorRegistry()

        class CustomDetector:
            name = "custom"
            caft_code = "99.1"

            def check(self, events, hta_state):
                if events:
                    return CaftDiagnosis(
                        caft_code="99.1",
                        caft_category="custom",
                        failure_name="custom",
                        severity=CaftSeverity.INFO,
                        confidence=0.99,
                        description="Custom detection.",
                        evidence={"test": True},
                        at_step=1,
                        remediation="Fix it.",
                    )
                return None

        registry.register(CustomDetector())
        assert "custom" in registry.names()
        det = registry.get("custom")
        hta = _default_hta()
        result = det.check([TraceEvent(step=1, type="tool_call")], hta)
        assert result is not None
        assert result.failure_name == "custom"

    def test_unregister(self):
        registry = DetectorRegistry()

        class DummyDet:
            name = "dummy"
            caft_code = "0.0"
            def check(self, events, hta_state):
                return None

        registry.register(DummyDet())
        assert "dummy" in registry.names()
        registry.unregister("dummy")
        assert "dummy" not in registry.names()
