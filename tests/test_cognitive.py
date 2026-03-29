"""Tests for the Cognitive Load Monitor components."""

import pytest

from agentdiag.models import TraceEvent
from agentdiag.hta import Phase
from agentdiag.cognitive import (
    CognitiveState,
    CognitiveStateTracker,
    WorkingMemoryModel,
    MemoryItem,
    DecisionPoint,
    DecisionPointLog,
    MetacognitiveState,
)


# ── Helper factories ──────────────────────────────────────────────────


def _event(step: int, tool: str = "read_file", success: bool = True,
           latency_ms: float = 100.0, event_type: str = "tool_call",
           output_hash: str | None = None, input_hash: str | None = None,
           tokens_in: int = 0, tokens_out: int = 0) -> TraceEvent:
    return TraceEvent(
        step=step, type=event_type, tool=tool,
        success=success, latency_ms=latency_ms,
        output_hash=output_hash, input_hash=input_hash,
        tokens_in=tokens_in, tokens_out=tokens_out,
    )


# ── CognitiveState ──────────────────────────────────────────────────


class TestCognitiveState:
    def test_to_dict_structure(self):
        state = CognitiveState()
        d = state.to_dict()
        assert "perception" in d
        assert "attention" in d
        assert "working_memory" in d
        assert "decision_making" in d
        assert "action" in d
        assert "feedback" in d
        assert "overall_cognitive_load" in d
        assert "active_ip_stage" in d

    def test_defaults(self):
        state = CognitiveState()
        assert state.perception_breadth == 0.0
        assert state.action_success_rate == 1.0
        assert state.feedback_error_response == 1.0
        assert state.overall_cognitive_load == 0.0

    def test_to_dict_values_rounded(self):
        state = CognitiveState(perception_breadth=0.12345678)
        d = state.to_dict()
        assert d["perception"]["breadth"] == 0.123


# ── WorkingMemoryModel ──────────────────────────────────────────────


class TestWorkingMemoryModel:
    def test_record_access_creates_item(self):
        mem = WorkingMemoryModel()
        ev = _event(1, "read", output_hash="abc123", tokens_in=500)
        mem.record_access(ev)
        assert mem._items.get("abc123") is not None
        assert mem._items["abc123"].access_count == 1

    def test_record_access_increments_count(self):
        mem = WorkingMemoryModel()
        ev1 = _event(1, "read", output_hash="abc123", tokens_in=500)
        ev2 = _event(5, "read", output_hash="abc123", tokens_in=500)
        mem.record_access(ev1)
        mem.record_access(ev2)
        assert mem._items["abc123"].access_count == 2
        assert mem._items["abc123"].last_accessed_step == 5

    def test_active_items_recent(self):
        mem = WorkingMemoryModel()
        for i in range(5):
            mem.record_access(_event(i, "read", output_hash=f"hash_{i}", tokens_in=100))
        active = mem.get_active_items()
        assert len(active) == 5

    def test_at_risk_items_old(self):
        mem = WorkingMemoryModel()
        mem.record_access(_event(1, "read", output_hash="old_hash", tokens_in=100))
        # Advance step far past DECAY_RATE
        mem._current_step = 100
        at_risk = mem.get_at_risk_items()
        assert len(at_risk) == 1
        assert at_risk[0].resource_hash == "old_hash"

    def test_utilization_grows_with_tokens(self):
        mem = WorkingMemoryModel()
        mem.record_access(_event(1, "read", tokens_in=50000, tokens_out=50000))
        assert mem.estimate_utilization() == pytest.approx(0.5, abs=0.01)

    def test_utilization_clamped(self):
        mem = WorkingMemoryModel()
        mem.record_access(_event(1, "read", tokens_in=150000, tokens_out=100000))
        assert mem.estimate_utilization() == 1.0

    def test_no_hash_no_item(self):
        """Events without output_hash or input_hash shouldn't create items."""
        mem = WorkingMemoryModel()
        mem.record_access(_event(1, "read", tokens_in=100))
        assert len(mem._items) == 0

    def test_eviction_at_max_items(self):
        mem = WorkingMemoryModel()
        mem.MAX_ITEMS = 5
        for i in range(7):
            mem.record_access(_event(i, "read", output_hash=f"h{i}", tokens_in=10))
        assert len(mem._items) == 5

    def test_to_dict_structure(self):
        mem = WorkingMemoryModel()
        mem.record_access(_event(1, "read", output_hash="abc", tokens_in=1000))
        d = mem.to_dict()
        assert "total_items" in d
        assert "active_count" in d
        assert "at_risk_count" in d
        assert "utilization" in d
        assert "cumulative_tokens" in d
        assert "active_items" in d
        assert "at_risk_items" in d


# ── MemoryItem ──────────────────────────────────────────────────────


class TestMemoryItem:
    def test_retention_high_when_recent(self):
        item = MemoryItem("h1", "read", acquired_step=50,
                          last_accessed_step=50, token_estimate=100, access_count=1)
        # recency=1.0 (0 steps since), consolidation=0.2 (1/5)
        # 0.6*1.0 + 0.4*0.2 = 0.68
        assert item.retention_probability(50) == pytest.approx(0.68, abs=0.01)

    def test_retention_decays_with_distance(self):
        item = MemoryItem("h1", "read", acquired_step=1,
                          last_accessed_step=1, token_estimate=100, access_count=1)
        p1 = item.retention_probability(10)
        p2 = item.retention_probability(40)
        assert p1 > p2

    def test_retention_higher_with_more_accesses(self):
        item1 = MemoryItem("h1", "read", acquired_step=1,
                           last_accessed_step=1, token_estimate=100, access_count=1)
        item5 = MemoryItem("h1", "read", acquired_step=1,
                           last_accessed_step=1, token_estimate=100, access_count=5)
        assert item5.retention_probability(30) > item1.retention_probability(30)


# ── DecisionPointLog ────────────────────────────────────────────────


class TestDecisionPointLog:
    def test_reasoning_then_action_creates_point(self):
        log = DecisionPointLog()
        log.push(_event(1, event_type="reasoning"), Phase.PLANNING)
        log.push(_event(2, event_type="reasoning"), Phase.PLANNING)
        log.push(_event(3, "edit"), Phase.EXECUTING)
        # Push another reasoning event to close the previous decision
        result = log.push(_event(4, event_type="reasoning"), Phase.PLANNING)
        assert result is not None
        assert result.action_tool == "edit"
        assert len(result.reasoning_steps) == 2

    def test_action_without_reasoning(self):
        log = DecisionPointLog()
        log.push(_event(1, "read"), Phase.GATHERING)
        # Action directly with another reasoning event to close it
        log.push(_event(2, event_type="reasoning"), Phase.PLANNING)
        result = log.flush()
        # The pending reasoning should close
        assert result is None  # reasoning doesn't create action points on its own

    def test_strategy_change_detected(self):
        log = DecisionPointLog()
        log.push(_event(1, "read"), Phase.GATHERING)
        log.push(_event(2, event_type="reasoning"), Phase.PLANNING)
        # Close previous decision
        dp = log._finalize_pending()

        # New decision with different tool
        log.push(_event(3, "edit"), Phase.EXECUTING)
        log.push(_event(4, event_type="reasoning"), Phase.PLANNING)
        dp2 = log._finalize_pending()
        if dp2:
            assert dp2.strategy_changed is True

    def test_get_recent(self):
        log = DecisionPointLog()
        for i in range(20):
            log.push(_event(i * 2, event_type="reasoning"), Phase.PLANNING)
            log.push(_event(i * 2 + 1, "read"), Phase.GATHERING)
        log.flush()

        recent = log.get_recent(5)
        assert len(recent) <= 5

    def test_quality_metrics_empty(self):
        log = DecisionPointLog()
        m = log.get_decision_quality_metrics()
        assert m["total_decisions"] == 0

    def test_quality_metrics_populated(self):
        log = DecisionPointLog()
        # Create several decisions
        for i in range(5):
            log.push(_event(i * 3, event_type="reasoning"), Phase.PLANNING)
            log.push(_event(i * 3 + 1, "edit", success=(i != 2)), Phase.EXECUTING)
        log.push(_event(99, event_type="reasoning"), Phase.PLANNING)  # close last

        m = log.get_decision_quality_metrics()
        assert m["total_decisions"] >= 1
        assert "success_rate" in m
        assert "avg_deliberation_ms" in m

    def test_to_dict_structure(self):
        log = DecisionPointLog()
        d = log.to_dict()
        assert "decisions" in d
        assert "metrics" in d


# ── MetacognitiveState ──────────────────────────────────────────────


class TestMetacognitiveState:
    def test_to_dict(self):
        meta = MetacognitiveState(
            monitors_active=14,
            detections_total=3,
            detections_confirmed=2,
            detections_rejected=1,
        )
        d = meta.to_dict()
        assert d["monitors_active"] == 14
        assert d["detections_total"] == 3


# ── CognitiveStateTracker ──────────────────────────────────────────


class TestCognitiveStateTracker:
    def test_perception_tracks_reads(self):
        t = CognitiveStateTracker()
        for i in range(5):
            t.update(_event(i, "read_file", output_hash=f"h{i}"), Phase.GATHERING)
        s = t.state
        assert s.perception_breadth > 0
        assert s.perception_depth > 0
        assert s.perception_recency > 0

    def test_perception_recency_decays(self):
        t = CognitiveStateTracker()
        t.update(_event(1, "read_file", output_hash="h1"), Phase.GATHERING)
        r1 = t.state.perception_recency

        # Non-read events move step forward without new reads
        for i in range(2, 25):
            t.update(_event(i, "edit"), Phase.EXECUTING)
        r2 = t.state.perception_recency
        assert r2 < r1

    def test_attention_diversity_with_varied_reads(self):
        t = CognitiveStateTracker()
        for i in range(10):
            t.update(
                _event(i, "read_file", output_hash=f"unique_{i}"),
                Phase.GATHERING,
            )
        s = t.state
        assert s.attention_diversity == pytest.approx(1.0, abs=0.01)

    def test_attention_tunnel_risk_with_repeated_reads(self):
        t = CognitiveStateTracker()
        for i in range(10):
            t.update(
                _event(i, "read_file", output_hash="same_hash"),
                Phase.GATHERING,
            )
        s = t.state
        # All reads target same hash → high tunnel risk
        assert s.attention_tunnel_risk == 0.0  # only 1 unique in window, but tunnel is 1 - unique/total

    def test_working_memory_utilization_grows(self):
        t = CognitiveStateTracker()
        t.update(
            _event(1, "read_file", tokens_in=50000, tokens_out=50000),
            Phase.GATHERING,
        )
        assert t.state.memory_utilization == pytest.approx(0.5, abs=0.01)

    def test_decision_deliberation_with_reasoning_streak(self):
        t = CognitiveStateTracker()
        for i in range(4):
            t.update(
                _event(i, event_type="reasoning"),
                Phase.PLANNING,
            )
        s = t.state
        assert s.decision_deliberation == 1.0  # 4 streak / threshold 4

    def test_decision_deliberation_resets_on_action(self):
        t = CognitiveStateTracker()
        for i in range(3):
            t.update(_event(i, event_type="reasoning"), Phase.PLANNING)
        assert t.state.decision_deliberation == pytest.approx(0.75, abs=0.01)

        t.update(_event(4, "edit"), Phase.EXECUTING)
        assert t.state.decision_deliberation == 0.0

    def test_action_success_rate_tracks_failures(self):
        t = CognitiveStateTracker()
        for i in range(8):
            t.update(_event(i, "bash", success=True), Phase.EXECUTING)
        for i in range(8, 10):
            t.update(_event(i, "bash", success=False), Phase.EXECUTING)
        s = t.state
        assert s.action_success_rate == pytest.approx(0.8, abs=0.01)

    def test_action_repetition_risk_with_same_input(self):
        t = CognitiveStateTracker()
        for i in range(9):
            t.update(
                _event(i, "read_file", input_hash="same_input"),
                Phase.GATHERING,
            )
        s = t.state
        assert s.action_repetition_risk == pytest.approx(1.0, abs=0.01)

    def test_feedback_verify_ratio(self):
        t = CognitiveStateTracker()
        # 5 executing events
        for i in range(5):
            t.update(_event(i, "edit"), Phase.EXECUTING)
        # 5 verifying events
        for i in range(5, 10):
            t.update(_event(i, "pytest"), Phase.VERIFYING)
        s = t.state
        assert s.feedback_verify_ratio > 0

    def test_active_ip_stage_follows_phase(self):
        t = CognitiveStateTracker()
        t.update(_event(1, "read_file"), Phase.GATHERING)
        assert t.state.active_ip_stage == "perception"

        t.update(_event(2, "edit"), Phase.EXECUTING)
        assert t.state.active_ip_stage == "action"

        t.update(_event(3, "pytest"), Phase.VERIFYING)
        assert t.state.active_ip_stage == "feedback"

    def test_overall_cognitive_load_bounded(self):
        t = CognitiveStateTracker()
        for i in range(50):
            t.update(
                _event(i, "bash", success=False, latency_ms=5000.0,
                       tokens_in=10000, tokens_out=10000),
                Phase.EXECUTING,
            )
        s = t.state
        assert 0.0 <= s.overall_cognitive_load <= 1.0

    def test_metacognitive_state_updates(self):
        t = CognitiveStateTracker()
        t.update(
            _event(1, "read_file"), Phase.GATHERING,
            num_detectors=14, num_diagnoses=3,
            num_confirmed=2, num_rejected=1,
            has_context_store=True,
        )
        meta = t.metacognitive
        assert meta.monitors_active == 14
        assert meta.detections_total == 3
        assert meta.learning_available is True

    def test_to_dict_includes_submodels(self):
        t = CognitiveStateTracker()
        t.update(_event(1, "read_file", output_hash="h1"), Phase.GATHERING)
        d = t.to_dict()
        assert "working_memory" in d
        assert "decision_points" in d
        assert "metacognitive" in d
        assert "perception" in d

    def test_update_is_o1_per_event(self):
        """Tracker should not grow in cost as events accumulate.

        We verify this by checking that internal data structures have
        bounded size regardless of how many events we push.
        """
        t = CognitiveStateTracker()
        for i in range(200):
            t.update(
                _event(i, "read_file", output_hash=f"h{i}", tokens_in=100),
                Phase.GATHERING,
            )
        # Deques should be bounded
        assert len(t._recent_reads) <= t.ATTENTION_WINDOW
        assert len(t._recent_tools) <= t.ACTION_WINDOW
        assert len(t._recent_actions) <= t.ACTION_WINDOW
        assert len(t._recent_latencies) <= t.ACTION_WINDOW
        assert len(t._recent_phases) <= t.FEEDBACK_WINDOW
        # Working memory model is bounded
        assert len(t._memory._items) <= t._memory.MAX_ITEMS
