"""Tests for multi-agent TraceEvent support and synthetic trace."""

import pytest

from agentdiag.models import TraceEvent
from agentdiag.caft.synthetic import (
    CAFT_GENERATORS,
    generate_multi_agent_trace,
)
from agentdiag.monitor import MonitorEngine


class TestTraceEventAgentId:
    def test_default_agent_id_is_none(self):
        event = TraceEvent(step=1, type="tool_call")
        assert event.agent_id is None

    def test_agent_id_set(self):
        event = TraceEvent(step=1, type="tool_call", agent_id="S1")
        assert event.agent_id == "S1"

    def test_from_dict_with_agent_id(self):
        event = TraceEvent.from_dict({
            "step": 1, "type": "tool_call", "agent_id": "S2",
        })
        assert event.agent_id == "S2"

    def test_from_dict_without_agent_id(self):
        event = TraceEvent.from_dict({"step": 1, "type": "tool_call"})
        assert event.agent_id is None

    def test_from_dict_ignores_unknown_fields(self):
        event = TraceEvent.from_dict({
            "step": 1, "type": "tool_call", "agent_id": "S1",
            "unknown_field": "should be ignored",
        })
        assert event.agent_id == "S1"
        assert not hasattr(event, "unknown_field")


class TestMultiAgentTrace:
    def test_registered_in_generators(self):
        assert "multi_agent" in CAFT_GENERATORS

    def test_returns_events_and_empty_expected(self):
        events, expected = generate_multi_agent_trace()
        assert len(events) > 0
        assert expected == set()

    def test_has_sub_agent_events(self):
        events, _ = generate_multi_agent_trace()
        sub_events = [e for e in events if e.agent_id == "S1"]
        assert len(sub_events) >= 2

    def test_has_main_agent_events(self):
        events, _ = generate_multi_agent_trace()
        main_events = [e for e in events if e.agent_id is None]
        assert len(main_events) >= 5

    def test_has_task_spawn(self):
        events, _ = generate_multi_agent_trace()
        task_events = [e for e in events if e.tool == "Task"]
        assert len(task_events) >= 1

    def test_has_task_result(self):
        events, _ = generate_multi_agent_trace()
        result_events = [
            e for e in events
            if e.type == "tool_result" and e.agent_id == "S1"
        ]
        assert len(result_events) >= 1

    def test_no_caft_detections(self):
        """Multi-agent trace should be clean — no CAFT failures."""
        events, _ = generate_multi_agent_trace()
        engine = MonitorEngine(goal="Multi-agent test")
        all_dx = []
        for event in events:
            dx = engine.push(event)
            all_dx.extend(dx)
        # Allow only false positives from known noisy detectors
        real_dx = [
            d for d in all_dx
            if d.failure_name not in ("stall", "analysis_paralysis", "token_explosion")
        ]
        assert len(real_dx) == 0, f"Unexpected detections: {[d.failure_name for d in real_dx]}"

    def test_steps_are_sequential(self):
        events, _ = generate_multi_agent_trace()
        steps = [e.step for e in events]
        assert steps == list(range(1, len(events) + 1))


class TestMultiAgentStateDict:
    def test_dashboard_includes_agents(self):
        """Verify _dashboard_to_dict includes agents dict."""
        from agentdiag.visualize import _dashboard_to_dict, _agent_phases, _agent_active
        from agentdiag.monitor import DashboardState

        # Set up agent tracking state
        _agent_phases.clear()
        _agent_active.clear()
        _agent_phases["M"] = "executing"
        _agent_phases["S1"] = "gathering"
        _agent_active["M"] = True
        _agent_active["S1"] = True

        state = DashboardState()
        result = _dashboard_to_dict(state)

        assert "agents" in result
        assert "M" in result["agents"]
        assert "S1" in result["agents"]
        assert result["agents"]["M"]["phase"] == "executing"
        assert result["agents"]["S1"]["phase"] == "gathering"
        assert result["agents"]["M"]["active"] is True
        assert result["agents"]["S1"]["active"] is True

        # Clean up
        _agent_phases.clear()
        _agent_active.clear()
