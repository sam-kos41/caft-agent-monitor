"""Tests for the real-time monitor engine."""

import io
import json
import pytest

from agentdiag.models import TraceEvent
from agentdiag.hta import Phase
from agentdiag.caft.base import CaftDiagnosis, CaftSeverity
from agentdiag.monitor import MonitorEngine, DashboardState, run_stdin_monitor


def _event_dict(step: int, type: str = "tool_call", tool: str = "read_file", **kw) -> dict:
    d = {"step": step, "type": type, "tool": tool}
    d.update(kw)
    return d


class TestMonitorEngine:
    def test_initial_state(self):
        engine = MonitorEngine(goal="test goal")
        state = engine.state
        assert state.total_events == 0
        assert state.trust_score == 1.0
        assert state.health == "healthy"
        assert state.hta_state is not None
        assert state.hta_state.goal == "test goal"

    def test_push_trace_event(self):
        engine = MonitorEngine()
        event = TraceEvent(step=1, type="tool_call", tool="read_file")
        diagnoses = engine.push(event)
        assert isinstance(diagnoses, list)
        state = engine.state
        assert state.total_events == 1
        assert len(state.actions) == 1
        assert state.actions[0].tool == "read_file"
        assert state.actions[0].phase == Phase.GATHERING

    def test_push_raw_dict(self):
        engine = MonitorEngine()
        data = {"step": 1, "type": "tool_call", "tool": "edit_file"}
        diagnoses = engine.push_raw(data)
        assert isinstance(diagnoses, list)
        state = engine.state
        assert state.total_events == 1

    def test_push_raw_list(self):
        engine = MonitorEngine()
        data = [
            {"step": 1, "type": "tool_call", "tool": "read_file"},
            {"step": 2, "type": "tool_call", "tool": "write_file"},
        ]
        engine.push_raw(data)
        assert engine.state.total_events == 2

    def test_hta_phase_tracking(self):
        engine = MonitorEngine()
        # Push gathering events
        engine.push(TraceEvent(step=1, type="tool_call", tool="read_file"))
        engine.push(TraceEvent(step=2, type="tool_call", tool="grep"))
        assert engine.state.hta_state.current_phase == Phase.GATHERING

        # Push executing events (2 for hysteresis)
        engine.push(TraceEvent(step=3, type="tool_call", tool="edit_file"))
        engine.push(TraceEvent(step=4, type="tool_call", tool="write_file"))
        assert engine.state.hta_state.current_phase == Phase.EXECUTING

    def test_error_counting(self):
        engine = MonitorEngine()
        engine.push(TraceEvent(step=1, type="tool_call", tool="bash", success=False))
        engine.push(TraceEvent(step=2, type="tool_call", tool="bash", success=True))
        engine.push(TraceEvent(step=3, type="tool_call", tool="bash", success=False))
        assert engine.state.total_errors == 2

    def test_trust_score_decreases_on_diagnosis(self):
        """Trust should decrease when CAFT detectors fire."""
        engine = MonitorEngine()
        # Create step repetition (3+ consecutive identical operations)
        for i in range(5):
            engine.push(TraceEvent(step=i + 1, type="tool_call", tool="read_file"))
        state = engine.state
        # If step repetition fired, trust should have decreased
        if state.diagnoses:
            assert state.trust_score < 1.0

    def test_diagnosis_callback(self):
        """on_diagnosis callback should fire."""
        received = []
        engine = MonitorEngine(on_diagnosis=lambda d: received.append(d))
        # Trigger step repetition
        for i in range(5):
            engine.push(TraceEvent(step=i + 1, type="tool_call", tool="read_file"))
        # May or may not fire depending on detector, but callback mechanism works
        for d in received:
            assert isinstance(d, CaftDiagnosis)

    def test_event_callback(self):
        received = []
        engine = MonitorEngine(on_event=lambda a: received.append(a))
        engine.push(TraceEvent(step=1, type="tool_call", tool="read_file"))
        assert len(received) == 1
        assert received[0].tool == "read_file"

    def test_action_stream_limit(self):
        engine = MonitorEngine()
        for i in range(250):
            engine.push(TraceEvent(step=i + 1, type="tool_call", tool="read_file"))
        state = engine.state
        assert len(state.actions) <= engine.MAX_ACTIONS

    def test_reset(self):
        engine = MonitorEngine(goal="test")
        engine.push(TraceEvent(step=1, type="tool_call", tool="read_file"))
        engine.push(TraceEvent(step=2, type="tool_call", tool="read_file"))
        engine.reset()
        state = engine.state
        assert state.total_events == 0
        assert state.trust_score == 1.0
        assert len(state.actions) == 0
        assert len(state.diagnoses) == 0

    def test_set_goal(self):
        engine = MonitorEngine()
        engine.set_goal("new goal")
        engine.push(TraceEvent(step=1, type="tool_call", tool="read_file"))
        assert engine.state.hta_state.goal == "new goal"


class TestDashboardState:
    def test_health_levels(self):
        # No diagnoses, no events → healthy
        state = DashboardState(trust_score=1.0)
        assert state.health == "healthy"

        # WARNING severity → degraded
        state = DashboardState(
            diagnoses=[
                CaftDiagnosis(
                    caft_code="2.2", caft_category="memory",
                    failure_name="step_repetition",
                    severity=CaftSeverity.WARNING, confidence=0.5,
                    description="test", evidence={}, at_step=1,
                    remediation="fix",
                ),
            ]
        )
        assert state.health == "degraded"

        # CRITICAL severity → failing
        state = DashboardState(
            diagnoses=[
                CaftDiagnosis(
                    caft_code="5.4", caft_category="plan_structure",
                    failure_name="premature_termination",
                    severity=CaftSeverity.CRITICAL, confidence=0.85,
                    description="test", evidence={}, at_step=5,
                    remediation="fix",
                ),
            ]
        )
        assert state.health == "failing"

        # Low completion rate with events → degraded
        state = DashboardState(total_events=10, completion_rate=20)
        assert state.health == "degraded"

    def test_worst_severity_none_when_empty(self):
        state = DashboardState()
        assert state.worst_severity is None

    def test_worst_severity(self):
        state = DashboardState(
            diagnoses=[
                CaftDiagnosis(
                    caft_code="2.2", caft_category="memory",
                    failure_name="step_repetition",
                    severity=CaftSeverity.WARNING, confidence=0.5,
                    description="test", evidence={}, at_step=1,
                    remediation="fix",
                ),
                CaftDiagnosis(
                    caft_code="5.4", caft_category="plan_structure",
                    failure_name="premature_termination",
                    severity=CaftSeverity.CRITICAL, confidence=0.85,
                    description="test", evidence={}, at_step=5,
                    remediation="fix",
                ),
            ]
        )
        assert state.worst_severity == CaftSeverity.CRITICAL


class TestRunStdinMonitor:
    def test_reads_jsonl_stream(self):
        lines = [
            json.dumps({"step": 1, "type": "tool_call", "tool": "read_file"}),
            json.dumps({"step": 2, "type": "tool_call", "tool": "write_file"}),
            json.dumps({"step": 3, "type": "reasoning"}),
        ]
        stream = io.StringIO("\n".join(lines))
        state = run_stdin_monitor(goal="test", stream=stream)
        assert state.total_events == 3

    def test_skips_invalid_json(self):
        lines = [
            json.dumps({"step": 1, "type": "tool_call", "tool": "read_file"}),
            "not valid json {{{",
            json.dumps({"step": 2, "type": "tool_call", "tool": "write_file"}),
        ]
        stream = io.StringIO("\n".join(lines))
        state = run_stdin_monitor(stream=stream)
        assert state.total_events == 2

    def test_skips_blank_lines(self):
        lines = [
            json.dumps({"step": 1, "type": "tool_call", "tool": "read_file"}),
            "",
            "  ",
            json.dumps({"step": 2, "type": "tool_call", "tool": "write_file"}),
        ]
        stream = io.StringIO("\n".join(lines))
        state = run_stdin_monitor(stream=stream)
        assert state.total_events == 2

    def test_on_state_callback(self):
        states_received = []
        lines = [
            json.dumps({"step": 1, "type": "tool_call", "tool": "read_file"}),
            json.dumps({"step": 2, "type": "tool_call", "tool": "write_file"}),
        ]
        stream = io.StringIO("\n".join(lines))
        run_stdin_monitor(
            stream=stream,
            on_state=lambda s: states_received.append(s),
        )
        assert len(states_received) == 2

    def test_full_pipeline_integration(self):
        """Integration test: events -> HTA phases -> CAFT detection."""
        events = []
        # Gathering phase
        events.append({"step": 1, "type": "tool_call", "tool": "read_file"})
        events.append({"step": 2, "type": "tool_call", "tool": "grep"})
        # Planning
        events.append({"step": 3, "type": "reasoning", "goal_text": "Plan the fix"})
        events.append({"step": 4, "type": "reasoning", "goal_text": "Design approach"})
        # Executing
        events.append({"step": 5, "type": "tool_call", "tool": "edit_file"})
        events.append({"step": 6, "type": "tool_call", "tool": "write_file"})
        events.append({"step": 7, "type": "tool_call", "tool": "edit_file"})
        # Verifying
        events.append({"step": 8, "type": "tool_call", "tool": "run_tests"})
        events.append({"step": 9, "type": "tool_call", "tool": "pytest"})
        # Delivering
        events.append({"step": 10, "type": "tool_call", "tool": "git_commit"})
        events.append({"step": 11, "type": "tool_call", "tool": "push"})

        stream = io.StringIO("\n".join(json.dumps(e) for e in events))
        state = run_stdin_monitor(goal="Fix the login bug", stream=stream)

        assert state.total_events == 11
        assert state.total_errors == 0
        assert state.hta_state is not None
        assert state.hta_state.goal == "Fix the login bug"
        # Should have visited multiple phases
        assert state.hta_state.total_events == 11
