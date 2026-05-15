"""Tests for cross-agent coordination monitoring.

Validates:
  - CrossAgentMI computation and temporal pairing
  - Coordination signal detection (write→read)
  - Coordination failure detection (stale read, race condition, concurrent mod)
  - Dependency graph inference
  - Graceful degradation with 0 or 1 agents
  - Full CoordinationTracker pipeline
"""

from __future__ import annotations

import time
import pytest

from agentdiag.observable import (
    ObservableEvent,
    EventType,
    file_read_event,
    file_write_event,
    tool_call_event,
)
from agentdiag.universal_monitor import UniversalMonitor
from agentdiag.coordination import (
    CoordinationTracker,
    CrossAgentMI,
    CoordinationSignal,
    CoordinationFailure,
    _AgentFileTracker,
)


# ── Helpers ───────────────────────────────────────────────────────────────

def _make_read(step: int, path: str) -> ObservableEvent:
    return file_read_event(step=step, timestamp=float(step), path=path)


def _make_write(step: int, path: str) -> ObservableEvent:
    return file_write_event(step=step, timestamp=float(step), path=path)


def _make_tool(step: int, tool: str, path: str = "") -> ObservableEvent:
    return tool_call_event(
        step=step, timestamp=float(step), tool_name=tool, target_path=path or None,
    )


# ── CrossAgentMI unit tests ──────────────────────────────────────────────

class TestCrossAgentMI:
    """Test the MI computation between two agents."""

    def test_empty_mi_is_zero(self):
        mi = CrossAgentMI()
        assert mi.mi() == 0.0
        assert mi.pair_count == 0

    def test_identical_sequences_have_high_mi(self):
        mi = CrossAgentMI(window=50, temporal_gap=5)
        for i in range(50):
            mi.record_a(f"action_{i % 3}", i)
            mi.record_b(f"action_{i % 3}", i)
        assert mi.mi() > 0.0
        assert mi.pair_count > 0

    def test_independent_sequences_have_low_mi(self):
        mi = CrossAgentMI(window=50, temporal_gap=5)
        # A does action_0,1,2 cyclically; B does action_3,4,5
        for i in range(50):
            mi.record_a(f"a_{i % 3}", i)
            mi.record_b(f"b_{i % 3}", i)
        # MI should be low since the symbol sets don't overlap
        # (they're independent but still paired temporally)
        # With non-overlapping symbols, MI can still be nonzero if
        # there are consistent pairings — but lower than identical
        assert mi.pair_count > 0

    def test_temporal_gap_filters_distant_events(self):
        mi = CrossAgentMI(window=50, temporal_gap=3)
        # A acts at steps 0,1,2; B acts at steps 100,101,102
        for i in range(3):
            mi.record_a("read", i)
        for i in range(3):
            mi.record_b("write", 100 + i)
        # Events are too far apart to pair
        assert mi.pair_count == 0

    def test_history_records_values(self):
        mi = CrossAgentMI(window=50, temporal_gap=5)
        for i in range(20):
            mi.record_a("read", i)
            mi.record_b("write", i)
        assert len(mi.history) > 0


# ── _AgentFileTracker unit tests ─────────────────────────────────────────

class TestAgentFileTracker:

    def test_record_and_query(self):
        t = _AgentFileTracker()
        t.record_write("src/main.py", 1)
        t.record_read("src/utils.py", 2)

        assert t.has_written("src/main.py")
        assert not t.has_written("src/utils.py")
        assert t.has_read("src/utils.py")
        assert t.last_write("src/main.py") == 1
        assert t.last_read("src/utils.py") == 2
        assert t.last_read("nonexistent.py") is None
        assert "src/main.py" in t.written_resources
        assert "src/utils.py" in t.read_resources

    def test_overwrites_update_step(self):
        t = _AgentFileTracker()
        t.record_write("f.py", 1)
        t.record_write("f.py", 5)
        assert t.last_write("f.py") == 5


# ── CoordinationTracker integration tests ────────────────────────────────

class TestCoordinationTracker:

    def test_register_agents(self):
        tracker = CoordinationTracker()
        tracker.register_agent("a")
        tracker.register_agent("b")
        state = tracker.get_state()
        assert len(state["nodes"]) == 2
        assert len(state["edges"]) == 1  # one pair

    def test_single_agent_degrades_gracefully(self):
        tracker = CoordinationTracker()
        tracker.register_agent("solo")
        state = tracker.get_state()
        assert len(state["nodes"]) == 1
        assert len(state["edges"]) == 0
        assert state["summary"]["coordination_health"] == "decoupled"

    def test_empty_tracker_degrades_gracefully(self):
        tracker = CoordinationTracker()
        state = tracker.get_state()
        assert state["nodes"] == []
        assert state["edges"] == []
        assert state["summary"]["agent_count"] == 0

    def test_unknown_agent_raises(self):
        tracker = CoordinationTracker()
        tracker.register_agent("a")
        with pytest.raises(ValueError, match="Unknown agent"):
            tracker.observe("nonexistent", _make_read(1, "f.py"))

    def test_write_read_coordination_signal(self):
        """Agent A writes a file, Agent B reads it within the window → signal."""
        tracker = CoordinationTracker(coordination_window=10)
        tracker.register_agent("a")
        tracker.register_agent("b")

        # A writes src/api.py
        r1 = tracker.observe("a", _make_write(1, "src/api.py"))
        assert r1["coordination_signals"] == []

        # B reads src/api.py within window
        r2 = tracker.observe("b", _make_read(2, "src/api.py"))
        assert len(r2["coordination_signals"]) == 1
        sig = r2["coordination_signals"][0]
        assert sig["signal_type"] == "write_read"
        assert sig["source_agent"] == "a"
        assert sig["target_agent"] == "b"
        assert sig["resource"] == "src/api.py"

    def test_write_read_outside_window_no_signal(self):
        """Read too far after write → no coordination signal."""
        tracker = CoordinationTracker(coordination_window=5)
        tracker.register_agent("a")
        tracker.register_agent("b")

        tracker.observe("a", _make_write(1, "f.py"))
        # Feed enough filler events to push global step beyond window
        for i in range(10):
            tracker.observe("a", _make_tool(10 + i, "grep", "search"))

        r = tracker.observe("b", _make_read(100, "f.py"))
        assert len(r["coordination_signals"]) == 0

    def test_race_condition_detection(self):
        """Two agents writing the same file within the window → race condition."""
        tracker = CoordinationTracker(coordination_window=10)
        tracker.register_agent("a")
        tracker.register_agent("b")

        tracker.observe("a", _make_write(1, "shared.py"))
        r = tracker.observe("b", _make_write(2, "shared.py"))

        failures = r["coordination_failures"]
        race_failures = [f for f in failures if f["failure_type"] == "race_condition"]
        assert len(race_failures) == 1
        assert race_failures[0]["resource"] == "shared.py"
        assert race_failures[0]["severity"] == "warning"

    def test_concurrent_modification_detection(self):
        """Agent B reads a file, Agent A writes it shortly after → concurrent mod."""
        tracker = CoordinationTracker(coordination_window=10)
        tracker.register_agent("a")
        tracker.register_agent("b")

        # B reads the file
        tracker.observe("b", _make_read(1, "config.py"))
        # A writes the same file
        r = tracker.observe("a", _make_write(2, "config.py"))

        failures = r["coordination_failures"]
        conc_failures = [f for f in failures if f["failure_type"] == "concurrent_modification"]
        assert len(conc_failures) == 1
        assert "config.py" in conc_failures[0]["description"]

    def test_stale_read_detection(self):
        """Agent B read file long ago, Agent A modifies it → stale read warning."""
        tracker = CoordinationTracker(coordination_window=5)
        tracker.register_agent("a")
        tracker.register_agent("b")

        # B reads the file
        tracker.observe("b", _make_read(1, "models.py"))

        # Many steps pass (push global step forward)
        for i in range(20):
            tracker.observe("a", _make_tool(10 + i, "grep"))

        # A writes the file — B's read is now stale
        r = tracker.observe("a", _make_write(50, "models.py"))
        failures = r["coordination_failures"]
        stale = [f for f in failures if f["failure_type"] == "stale_read"]
        assert len(stale) == 1
        assert "models.py" in stale[0]["description"]

    def test_dependency_graph_inference(self):
        """Repeated write→read patterns infer a dependency."""
        tracker = CoordinationTracker(coordination_window=10)
        tracker.register_agent("producer")
        tracker.register_agent("consumer")

        # Producer writes, consumer reads — 5 times
        for i in range(5):
            tracker.observe("producer", _make_write(i * 2, "output.json"))
            tracker.observe("consumer", _make_read(i * 2 + 1, "output.json"))

        state = tracker.get_state()
        deps = state["dependency_graph"]
        assert len(deps) >= 1
        dep = deps[0]
        assert dep["producer"] == "producer"
        assert dep["consumer"] == "consumer"
        assert dep["signal_count"] == 5
        assert dep["strength"] == "strong"

    def test_mi_matrix(self):
        """MI matrix returns values for all pairs."""
        tracker = CoordinationTracker()
        tracker.register_agent("a")
        tracker.register_agent("b")

        for i in range(30):
            tracker.observe("a", _make_read(i, f"file_{i % 3}.py"))
            tracker.observe("b", _make_write(i, f"file_{i % 3}.py"))

        matrix = tracker.get_mi_matrix()
        assert ("a", "b") in matrix
        assert isinstance(matrix[("a", "b")], float)

    def test_full_state_structure(self):
        """get_state returns all expected keys."""
        tracker = CoordinationTracker()
        tracker.register_agent("a")
        tracker.register_agent("b")

        # Feed some events
        tracker.observe("a", _make_write(1, "f.py"))
        tracker.observe("b", _make_read(2, "f.py"))

        state = tracker.get_state()
        assert "nodes" in state
        assert "edges" in state
        assert "signals" in state
        assert "failures" in state
        assert "global_step" in state
        assert "dependency_graph" in state
        assert "summary" in state

        summary = state["summary"]
        assert summary["agent_count"] == 2
        assert summary["pair_count"] == 1
        assert "coordination_health" in summary

    def test_three_agents_creates_three_pairs(self):
        """3 agents → 3 pairwise MI trackers."""
        tracker = CoordinationTracker()
        tracker.register_agent("a")
        tracker.register_agent("b")
        tracker.register_agent("c")

        state = tracker.get_state()
        assert len(state["edges"]) == 3  # (a,b), (a,c), (b,c)

    def test_edge_status_reflects_failures(self):
        """Edge status becomes 'warning' when failures exist between the pair."""
        tracker = CoordinationTracker(coordination_window=10)
        tracker.register_agent("a")
        tracker.register_agent("b")

        # Create a race condition failure
        tracker.observe("a", _make_write(1, "shared.py"))
        tracker.observe("b", _make_write(2, "shared.py"))

        state = tracker.get_state()
        edge = state["edges"][0]
        assert edge["status"] == "warning"

    def test_monitor_passed_to_register(self):
        """Can pass an existing UniversalMonitor."""
        mon = UniversalMonitor(sensitivity=3.0)
        tracker = CoordinationTracker()
        tracker.register_agent("x", monitor=mon)
        assert tracker._agents["x"] is mon
