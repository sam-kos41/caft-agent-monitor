"""Integration tests: MonitorEngine with cognitive=True."""

import pytest

from agentdiag.models import TraceEvent
from agentdiag.monitor import MonitorEngine
from agentdiag.cognitive import CognitiveStateTracker


def _event(step: int, tool: str = "read_file", success: bool = True,
           latency_ms: float = 100.0, event_type: str = "tool_call",
           output_hash: str | None = None,
           tokens_in: int = 0, tokens_out: int = 0) -> TraceEvent:
    return TraceEvent(
        step=step, type=event_type, tool=tool,
        success=success, latency_ms=latency_ms,
        output_hash=output_hash,
        tokens_in=tokens_in, tokens_out=tokens_out,
    )


class TestMonitorEngineCognitive:
    def test_cognitive_disabled_by_default(self):
        engine = MonitorEngine(goal="test")
        assert engine.cognitive_state is None

    def test_cognitive_enabled(self):
        engine = MonitorEngine(goal="test", cognitive=True)
        assert engine.cognitive_state is not None
        assert isinstance(engine.cognitive_state, CognitiveStateTracker)

    def test_cognitive_state_updates_on_push(self):
        engine = MonitorEngine(goal="test", cognitive=True)
        engine.push(TraceEvent(step=1, type="tool_call", tool="read_file"))
        engine.push(TraceEvent(step=2, type="tool_call", tool="grep"))
        engine.push(TraceEvent(step=3, type="tool_call", tool="edit"))

        state = engine.cognitive_state.state
        assert state.last_updated_step == 3
        assert state.active_ip_stage in {
            "perception", "attention", "working_memory",
            "decision_making", "action", "feedback",
        }

    def test_cognitive_state_perception_after_reads(self):
        engine = MonitorEngine(goal="test", cognitive=True)
        for i in range(10):
            engine.push(TraceEvent(
                step=i, type="tool_call", tool="read_file",
                output_hash=f"file_{i}", tokens_in=500,
            ))
        state = engine.cognitive_state.state
        assert state.perception_breadth > 0
        assert state.perception_recency > 0

    def test_cognitive_state_memory_grows_with_tokens(self):
        engine = MonitorEngine(goal="test", cognitive=True)
        engine.push(TraceEvent(
            step=1, type="tool_call", tool="read_file",
            tokens_in=100000, tokens_out=100000,
        ))
        state = engine.cognitive_state.state
        assert state.memory_utilization == pytest.approx(1.0, abs=0.01)

    def test_cognitive_state_action_phase(self):
        engine = MonitorEngine(goal="test", cognitive=True)
        # Push enough to transition to executing
        engine.push(TraceEvent(step=1, type="tool_call", tool="read_file"))
        engine.push(TraceEvent(step=2, type="tool_call", tool="write"))
        state = engine.cognitive_state.state
        assert state.active_ip_stage == "action"

    def test_cognitive_state_decision_deliberation(self):
        engine = MonitorEngine(goal="test", cognitive=True)
        for i in range(4):
            engine.push(TraceEvent(step=i, type="reasoning"))
        state = engine.cognitive_state.state
        assert state.decision_deliberation == 1.0

    def test_cognitive_reset(self):
        engine = MonitorEngine(goal="test", cognitive=True)
        engine.push(TraceEvent(step=1, type="tool_call", tool="read_file"))
        assert engine.cognitive_state.state.last_updated_step == 1

        engine.reset()
        assert engine.cognitive_state is not None
        assert engine.cognitive_state.state.last_updated_step == 0

    def test_no_overhead_when_disabled(self):
        engine = MonitorEngine(goal="test", cognitive=False)
        for i in range(50):
            engine.push(TraceEvent(step=i, type="tool_call", tool="read_file"))
        assert engine.state.total_events == 50
        assert engine.cognitive_state is None

    def test_cognitive_to_dict_serializable(self):
        """Full round-trip: push events, get cognitive state as dict."""
        engine = MonitorEngine(goal="test", cognitive=True)
        for i in range(20):
            tool = ["read_file", "grep", "edit", "bash", "pytest"][i % 5]
            engine.push(TraceEvent(
                step=i, type="tool_call", tool=tool,
                latency_ms=100.0 + i * 10,
                success=(i % 7 != 0),
                output_hash=f"h{i}",
                tokens_in=500,
            ))

        d = engine.cognitive_state.to_dict()
        # Verify all top-level sections exist
        assert "perception" in d
        assert "attention" in d
        assert "working_memory" in d
        assert "decision_making" in d
        assert "action" in d
        assert "feedback" in d
        assert "overall_cognitive_load" in d
        assert "metacognitive" in d
        assert d["metacognitive"]["monitors_active"] > 0

    def test_working_memory_model_via_engine(self):
        engine = MonitorEngine(goal="test", cognitive=True)
        for i in range(10):
            engine.push(TraceEvent(
                step=i, type="tool_call", tool="read_file",
                output_hash=f"file_{i}", tokens_in=1000,
            ))
        mem = engine.cognitive_state.working_memory.to_dict()
        assert mem["total_items"] == 10
        assert mem["active_count"] == 10
        assert mem["at_risk_count"] == 0

    def test_decision_log_via_engine(self):
        engine = MonitorEngine(goal="test", cognitive=True)
        # Reasoning → action pattern
        engine.push(TraceEvent(step=1, type="reasoning"))
        engine.push(TraceEvent(step=2, type="reasoning"))
        engine.push(TraceEvent(step=3, type="tool_call", tool="edit"))
        engine.push(TraceEvent(step=4, type="reasoning"))  # closes decision

        log = engine.cognitive_state.decision_log
        assert len(log.get_recent()) >= 1

    def test_cognitive_and_decision_trace_coexist(self):
        """Both optional subsystems can be enabled simultaneously."""
        engine = MonitorEngine(
            goal="test", cognitive=True, decision_trace=True,
        )
        for i in range(10):
            engine.push(TraceEvent(
                step=i, type="tool_call", tool="read_file",
                latency_ms=100.0,
            ))
        assert engine.cognitive_state is not None
        assert engine.decision_trace is not None
        assert engine.cognitive_state.state.last_updated_step == 9
        assert engine.decision_trace.to_dict()["total_steps"] == 10
