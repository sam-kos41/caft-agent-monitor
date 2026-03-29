"""Tests for the CAFT (Cognitive Agent Failure Taxonomy) detectors."""

import pytest
from agentdiag.models import TraceEvent
from agentdiag.hta import HTAStateMachine, HTAState, Phase, PhaseTransition, HTANode
from agentdiag.caft.base import CaftDiagnosis, CaftSeverity
from agentdiag.caft.detectors import (
    StepRepetitionDetector,
    ContextLossDetector,
    PrematureTerminationDetector,
    MissingVerificationDetector,
    ReasoningActionMismatchDetector,
    GoalDriftDetector,
    ToolThrashingDetector,
    StallDetector,
    ErrorCascadeDetector,
    run_caft_detectors,
    ALL_CAFT_DETECTORS,
    ALL_CAFT_DETECTORS_FULL,
    TIER_2_FAILURE_TYPES,
    _segment_agent_blocks,
)


def _make_hta_state(
    phase: Phase = Phase.EXECUTING,
    phase_counts: dict | None = None,
    transitions: list | None = None,
    total_events: int = 10,
) -> HTAState:
    """Helper to create an HTAState for testing."""
    return HTAState(
        goal="test",
        current_phase=phase,
        current_node=HTANode(phase=phase, start_step=0, start_time=0.0),
        completed_nodes=[],
        transitions=transitions or [],
        total_events=total_events,
        phase_event_counts=phase_counts or {},
    )


# --- StepRepetitionDetector ---

class TestStepRepetition:
    def test_no_fire_below_threshold(self):
        det = StepRepetitionDetector()
        events = [
            TraceEvent(step=1, type="tool_call", tool="read_file"),
            TraceEvent(step=2, type="tool_call", tool="read_file"),
        ]
        result = det.check(events, _make_hta_state())
        assert result is None

    def test_fires_on_ten_identical_operations(self):
        """V3: needs 9+ consecutive identical (tool, input_hash) operations."""
        det = StepRepetitionDetector()
        events = [
            TraceEvent(step=i, type="tool_call", tool="read_file", input_hash="same123")
            for i in range(1, 11)  # 10 consecutive
        ]
        result = det.check(events, _make_hta_state())
        assert result is not None
        assert result.caft_code == "2.2"
        assert result.failure_name == "step_repetition"
        assert result.evidence["consecutive_count"] >= 9

    def test_no_fire_eight_identical(self):
        """V3: 8 consecutive identical ops is below threshold=9."""
        det = StepRepetitionDetector()
        events = [
            TraceEvent(step=i, type="tool_call", tool="read_file", input_hash="same123")
            for i in range(1, 9)  # 8 consecutive
        ]
        # Add a different event to ensure enough events
        events.append(TraceEvent(step=9, type="tool_call", tool="edit_file"))
        result = det.check(events, _make_hta_state())
        assert result is None

    def test_no_fire_three_consecutive_different_inputs(self):
        """V3: 3 reads of DIFFERENT files is exploration, not repetition."""
        det = StepRepetitionDetector()
        events = [
            TraceEvent(step=1, type="tool_call", tool="read_file", input_hash="file_a"),
            TraceEvent(step=2, type="tool_call", tool="read_file", input_hash="file_b"),
            TraceEvent(step=3, type="tool_call", tool="read_file", input_hash="file_c"),
        ]
        result = det.check(events, _make_hta_state())
        assert result is None

    def test_no_fire_with_variation(self):
        det = StepRepetitionDetector()
        events = [
            TraceEvent(step=1, type="tool_call", tool="read_file"),
            TraceEvent(step=2, type="tool_call", tool="write_file"),
            TraceEvent(step=3, type="tool_call", tool="read_file"),
        ]
        result = det.check(events, _make_hta_state())
        assert result is None

    def test_no_fire_with_output_diversity(self):
        """V3: 10+ identical ops with >80% unique output_hashes = progressive work."""
        det = StepRepetitionDetector()
        events = [
            TraceEvent(
                step=i, type="tool_call", tool="read_file",
                input_hash="same_file",
                output_hash=f"unique_output_{i}",  # all unique
            )
            for i in range(1, 11)  # 10 consecutive
        ]
        result = det.check(events, _make_hta_state())
        assert result is None

    def test_fires_with_identical_outputs(self):
        """V3: 10+ identical ops with same output_hash = true repetition."""
        det = StepRepetitionDetector()
        events = [
            TraceEvent(
                step=i, type="tool_call", tool="read_file",
                input_hash="same_file",
                output_hash="same_output",  # all identical
            )
            for i in range(1, 11)  # 10 consecutive
        ]
        result = det.check(events, _make_hta_state())
        assert result is not None
        assert result.failure_name == "step_repetition"

    def test_no_fire_gathering_phase_below_10(self):
        """V3: In GATHERING phase, threshold raised to 10."""
        det = StepRepetitionDetector()
        events = [
            TraceEvent(step=i, type="tool_call", tool="read_file")
            for i in range(1, 10)  # 9 consecutive = meets base threshold but below 10
        ]
        hta = _make_hta_state(phase=Phase.GATHERING)
        result = det.check(events, hta)
        assert result is None


# --- ContextLossDetector ---

class TestContextLoss:
    def test_no_fire_few_events(self):
        det = ContextLossDetector()
        events = [TraceEvent(step=1, type="tool_call", tool="read_file")]
        result = det.check(events, _make_hta_state())
        assert result is None

    def test_fires_on_reread_with_substantial_intervening_work(self):
        """V3: needs max(5, 8% of events) intervening non-read operations."""
        det = ContextLossDetector()
        events = [
            TraceEvent(step=1, type="tool_call", tool="read_file", output_hash="abc123"),
            TraceEvent(step=2, type="tool_call", tool="edit_file"),
            TraceEvent(step=3, type="tool_call", tool="write_file"),
            TraceEvent(step=4, type="tool_call", tool="bash"),
            TraceEvent(step=5, type="tool_call", tool="edit_file"),
            TraceEvent(step=6, type="tool_call", tool="write_file"),
            TraceEvent(step=7, type="tool_call", tool="read_file", output_hash="abc123"),
        ]
        result = det.check(events, _make_hta_state())
        assert result is not None
        assert result.caft_code == "2.1"
        assert result.failure_name == "context_loss"

    def test_no_fire_with_only_two_intervening_ops(self):
        """V3: 2 intervening ops is too few -- normal re-read after edit."""
        det = ContextLossDetector()
        events = [
            TraceEvent(step=1, type="tool_call", tool="read_file", output_hash="abc123"),
            TraceEvent(step=2, type="tool_call", tool="edit_file"),
            TraceEvent(step=3, type="tool_call", tool="write_file"),
            TraceEvent(step=4, type="tool_call", tool="read_file", output_hash="abc123"),
        ]
        result = det.check(events, _make_hta_state())
        assert result is None

    def test_no_fire_without_intervening_work(self):
        """Adjacent re-reads are normal (e.g., checking file after edit)."""
        det = ContextLossDetector()
        events = [
            TraceEvent(step=1, type="tool_call", tool="read_file", output_hash="abc123"),
            TraceEvent(step=2, type="tool_call", tool="read_file", output_hash="abc123"),
            TraceEvent(step=3, type="tool_call", tool="edit_file"),
            TraceEvent(step=4, type="tool_call", tool="write_file"),
        ]
        result = det.check(events, _make_hta_state())
        assert result is None

    def test_scans_all_pairs_not_just_first(self):
        """V3: Scans all re-read pairs and fires on strongest signal."""
        det = ContextLossDetector()
        events = [
            # First resource: re-read blocked by continuation
            TraceEvent(step=1, type="tool_call", tool="read_file", output_hash="blocked_res"),
            TraceEvent(step=2, type="tool_call", tool="edit_file"),
            TraceEvent(step=3, type="tool_call", tool="write_file"),
            TraceEvent(step=4, type="user_input", goal_text="context was compressed"),
            TraceEvent(step=5, type="tool_call", tool="read_file", output_hash="blocked_res"),
            # Second resource: valid re-read with substantial gap
            TraceEvent(step=6, type="tool_call", tool="read_file", output_hash="valid_res"),
            TraceEvent(step=7, type="tool_call", tool="edit_file"),
            TraceEvent(step=8, type="tool_call", tool="write_file"),
            TraceEvent(step=9, type="tool_call", tool="bash"),
            TraceEvent(step=10, type="tool_call", tool="edit_file"),
            TraceEvent(step=11, type="tool_call", tool="write_file"),
            TraceEvent(step=12, type="tool_call", tool="read_file", output_hash="valid_res"),
        ]
        result = det.check(events, _make_hta_state())
        assert result is not None
        assert result.evidence["resource_hash"] == "valid_res"

    def test_relative_staleness_for_long_sessions(self):
        """V3: Gap must be >= max(5, 8% of total events)."""
        det = ContextLossDetector()
        # 100 events total, so 8% = 8 intervening needed (not just 5)
        events = [
            TraceEvent(step=1, type="tool_call", tool="read_file", output_hash="long_res"),
        ]
        # Add 6 intervening writes (enough for MIN=5 but not enough for 8% of 100)
        for i in range(2, 8):
            events.append(TraceEvent(step=i, type="tool_call", tool="edit_file"))
        events.append(TraceEvent(step=8, type="tool_call", tool="read_file", output_hash="long_res"))
        # Pad to 100 events
        for i in range(9, 101):
            events.append(TraceEvent(step=i, type="tool_call", tool="bash"))

        result = det.check(events, _make_hta_state(total_events=100))
        # 6 intervening < 8 (8% of 100), so should NOT fire
        assert result is None


# --- PrematureTerminationDetector ---

class TestPrematureTermination:
    def test_no_fire_when_not_delivering(self):
        det = PrematureTerminationDetector()
        events = [TraceEvent(step=i, type="tool_call", tool="edit_file") for i in range(5)]
        hta = _make_hta_state(
            phase=Phase.EXECUTING,
            phase_counts={"executing": 5},
        )
        result = det.check(events, hta)
        assert result is None

    def test_fires_when_delivering_without_verification(self):
        det = PrematureTerminationDetector()
        events = [TraceEvent(step=i, type="tool_call", tool="git_commit") for i in range(5)]
        hta = _make_hta_state(
            phase=Phase.DELIVERING,
            phase_counts={"executing": 5, "delivering": 5},
            transitions=[],
        )
        result = det.check(events, hta)
        assert result is not None
        assert result.caft_code == "5.4"
        assert result.severity == CaftSeverity.CRITICAL

    def test_no_fire_brief_delivering(self):
        """V2: Brief delivery (1-2 events) doesn't trigger."""
        det = PrematureTerminationDetector()
        events = [TraceEvent(step=i, type="tool_call", tool="git_commit") for i in range(5)]
        hta = _make_hta_state(
            phase=Phase.DELIVERING,
            phase_counts={"executing": 5, "delivering": 2},
            transitions=[],
        )
        result = det.check(events, hta)
        assert result is None

    def test_no_fire_when_verified(self):
        det = PrematureTerminationDetector()
        events = [TraceEvent(step=i, type="tool_call", tool="git_commit") for i in range(5)]
        hta = _make_hta_state(
            phase=Phase.DELIVERING,
            phase_counts={"executing": 5, "verifying": 2, "delivering": 5},
            transitions=[
                PhaseTransition(
                    from_phase=Phase.EXECUTING,
                    to_phase=Phase.VERIFYING,
                    at_step=4,
                    at_time=0.0,
                    is_regression=False,
                ),
            ],
        )
        result = det.check(events, hta)
        assert result is None

    # Mode 2+3 tests removed: V13 moved semantic PT detection to Tier 2.

    def test_mode1_has_force_llm_review(self):
        """V13: PT Mode 1 output must have force_llm_review=True."""
        det = PrematureTerminationDetector()
        events = [TraceEvent(step=i, type="tool_call", tool="git_commit") for i in range(5)]
        hta = _make_hta_state(
            phase=Phase.DELIVERING,
            phase_counts={"executing": 5, "delivering": 5},
            transitions=[],
        )
        result = det.check(events, hta)
        assert result is not None
        assert result.force_llm_review is True


class TestStallForceReview:
    """Stall: force_llm_review is always False (IQR detector is reliable)."""

    def test_stall_always_force_llm_review(self):
        det = StallDetector()
        # Build events with outlier latencies — only 1 stall event
        events = [
            TraceEvent(step=i, type="tool_call", tool="read_file", latency_ms=100)
            for i in range(1, 15)
        ]
        events.append(TraceEvent(
            step=15, type="tool_call", tool="read_file", latency_ms=50000,
        ))
        hta = _make_hta_state(total_events=len(events))
        result = det.check(events, hta)
        if result is not None:
            assert result.force_llm_review is False

    def test_stall_many_events_no_force_review(self):
        det = StallDetector()
        events = [
            TraceEvent(step=i, type="tool_call", tool="read_file", latency_ms=100)
            for i in range(1, 15)
        ]
        # Add 3 stall events (> 2 threshold)
        for i in range(15, 18):
            events.append(TraceEvent(
                step=i, type="tool_call", tool="read_file", latency_ms=50000,
            ))
        hta = _make_hta_state(total_events=len(events))
        result = det.check(events, hta)
        if result is not None:
            assert result.force_llm_review is False


class TestErrorCascadeForceReview:
    """ErrorCascade: force_llm_review depends on chain length (<5 → True)."""

    def test_error_cascade_short_chain_forces_review(self):
        from agentdiag.caft.detectors import ErrorCascadeDetector
        det = ErrorCascadeDetector()
        # Short chain (3 errors < 5 threshold)
        events = [
            TraceEvent(step=1, type="tool_call", tool="read_file", success=True),
        ]
        for i in range(2, 5):
            events.append(TraceEvent(
                step=i, type="tool_call", tool="read_file", success=False,
            ))
        hta = _make_hta_state(total_events=len(events))
        result = det.check(events, hta)
        if result is not None:
            assert result.force_llm_review is True

    def test_error_cascade_long_chain_no_force_review(self):
        from agentdiag.caft.detectors import ErrorCascadeDetector
        det = ErrorCascadeDetector()
        # Long chain (8 errors >= 5 threshold)
        events = [
            TraceEvent(step=1, type="tool_call", tool="read_file", success=True),
        ]
        for i in range(2, 10):
            events.append(TraceEvent(
                step=i, type="tool_call", tool="read_file", success=False,
            ))
        hta = _make_hta_state(total_events=len(events))
        result = det.check(events, hta)
        assert result is not None
        assert result.force_llm_review is False


# --- MissingVerificationDetector ---

class TestMissingVerification:
    def test_no_fire_below_threshold(self):
        det = MissingVerificationDetector()
        events = [
            TraceEvent(step=i, type="tool_call", tool="write_file")
            for i in range(5)
        ]
        hta = _make_hta_state(phase_counts={"executing": 5})
        result = det.check(events, hta)
        assert result is None

    def test_fires_above_threshold_with_writes(self):
        """V3: threshold 25 executing events, multi-pattern detection."""
        det = MissingVerificationDetector()
        events = [
            TraceEvent(step=i, type="tool_call", tool="write_file")
            for i in range(30)
        ]
        hta = _make_hta_state(phase_counts={"executing": 30})
        result = det.check(events, hta)
        assert result is not None
        assert result.caft_code == "5.3"
        assert result.failure_name == "missing_verification"

    def test_no_fire_if_verified(self):
        det = MissingVerificationDetector()
        events = [
            TraceEvent(step=i, type="tool_call", tool="write_file")
            for i in range(20)
        ]
        hta = _make_hta_state(phase_counts={"executing": 20, "verifying": 2})
        result = det.check(events, hta)
        assert result is None

    def test_no_fire_without_writes(self):
        det = MissingVerificationDetector()
        events = [
            TraceEvent(step=i, type="tool_call", tool="read_file")
            for i in range(20)
        ]
        hta = _make_hta_state(phase_counts={"executing": 20})
        result = det.check(events, hta)
        assert result is None

    def test_no_fire_with_delegated_verification(self):
        """V3: user pasting test output counts as verification (pattern 4)."""
        det = MissingVerificationDetector()
        events = [
            TraceEvent(step=i, type="tool_call", tool="write_file")
            for i in range(20)
        ]
        events.append(TraceEvent(
            step=21, type="user_input",
            goal_text="I ran the tests and they pass. Here's the output from pytest.",
        ))
        hta = _make_hta_state(phase_counts={"executing": 20})
        result = det.check(events, hta)
        assert result is None

    def test_no_fire_with_bash_test_command(self):
        """V3 Pattern 2: Bash command with test intent counts as verification."""
        det = MissingVerificationDetector()
        events = [
            TraceEvent(step=i, type="tool_call", tool="write_file")
            for i in range(20)
        ]
        events.append(TraceEvent(
            step=21, type="tool_call", tool="bash",
            goal_text="python -m pytest tests/ -v",
        ))
        hta = _make_hta_state(phase_counts={"executing": 20})
        result = det.check(events, hta)
        assert result is None

    def test_no_fire_with_task_subagent_testing(self):
        """V3 Pattern 3: Task subagent with testing intent."""
        det = MissingVerificationDetector()
        events = [
            TraceEvent(step=i, type="tool_call", tool="write_file")
            for i in range(20)
        ]
        events.append(TraceEvent(
            step=21, type="tool_call", tool="task",
            goal_text="Run the test suite and check results",
        ))
        hta = _make_hta_state(phase_counts={"executing": 20})
        result = det.check(events, hta)
        assert result is None

    def test_no_fire_with_reasoning_acknowledgment(self):
        """V3 Pattern 5: Agent acknowledges test results in reasoning."""
        det = MissingVerificationDetector()
        events = [
            TraceEvent(step=i, type="tool_call", tool="write_file")
            for i in range(20)
        ]
        events.append(TraceEvent(
            step=21, type="reasoning",
            goal_text="All tests pass. The implementation is verified and working.",
        ))
        hta = _make_hta_state(phase_counts={"executing": 20})
        result = det.check(events, hta)
        assert result is None


# --- ReasoningActionMismatchDetector ---

class TestReasoningActionMismatch:
    def test_fires_when_plans_read_but_does_write(self):
        det = ReasoningActionMismatchDetector()
        events = [
            TraceEvent(step=1, type="reasoning", goal_text="I should read the config file"),
            TraceEvent(step=2, type="tool_call", tool="write_file"),
        ]
        result = det.check(events, _make_hta_state())
        assert result is not None
        assert result.caft_code == "6.4"
        assert result.evidence["planned_intent"] == "read/review"
        assert result.evidence["actual_tool"] == "write_file"

    def test_no_fire_when_plans_read_and_does_read(self):
        det = ReasoningActionMismatchDetector()
        events = [
            TraceEvent(step=1, type="reasoning", goal_text="Let me read the file"),
            TraceEvent(step=2, type="tool_call", tool="read_file"),
        ]
        result = det.check(events, _make_hta_state())
        assert result is None

    def test_fires_when_plans_test_but_does_edit(self):
        det = ReasoningActionMismatchDetector()
        events = [
            TraceEvent(step=1, type="reasoning", goal_text="Now I should run the tests"),
            TraceEvent(step=2, type="tool_call", tool="edit_file"),
        ]
        result = det.check(events, _make_hta_state())
        assert result is not None
        assert result.evidence["planned_intent"] == "test/verify"

    def test_no_fire_without_goal_text(self):
        det = ReasoningActionMismatchDetector()
        events = [
            TraceEvent(step=1, type="reasoning"),
            TraceEvent(step=2, type="tool_call", tool="write_file"),
        ]
        result = det.check(events, _make_hta_state())
        assert result is None


# --- GoalDriftDetector ---

class TestGoalDrift:
    def test_no_fire_below_min_events(self):
        det = GoalDriftDetector()
        events = [TraceEvent(step=i, type="tool_call", tool="read_file") for i in range(15)]
        result = det.check(events, _make_hta_state())
        assert result is None

    def test_fires_on_block_based_drift(self):
        """V3: agent-block segmentation detects sustained off-topic behavior."""
        det = GoalDriftDetector()
        events = []
        step = 1

        # Block 1 (baseline): read_file, grep (4 events)
        for i in range(4):
            tool = "read_file" if i % 2 == 0 else "grep"
            events.append(TraceEvent(step=step, type="tool_call", tool=tool))
            step += 1

        # user_input at step 5
        events.append(TraceEvent(step=step, type="user_input", goal_text="ok"))
        step += 1

        # Block 2 (baseline): read_file, grep (4 events)
        for i in range(4):
            tool = "read_file" if i % 2 == 0 else "grep"
            events.append(TraceEvent(step=step, type="tool_call", tool=tool))
            step += 1

        # user_input at step 10
        events.append(TraceEvent(step=step, type="user_input", goal_text="ok"))
        step += 1

        # 3 drift blocks of 7 events each, separated by user_input
        # Novel tools {deploy, compile, generate} NOT in baseline {read_file, grep}
        for block_num in range(3):
            for tool in ["deploy", "compile", "generate", "deploy",
                         "compile", "generate", "deploy"]:
                events.append(TraceEvent(step=step, type="tool_call", tool=tool))
                step += 1
            if block_num < 2:
                events.append(TraceEvent(step=step, type="user_input", goal_text="ok"))
                step += 1

        # Regressions at mid-block positions (>3 steps from any user_input)
        # Block 3 starts at step 11, user_input at 10 and 18 -> regression at 15
        # Block 4 starts at step 19, user_input at 18 and 26 -> regression at 23
        # Block 5 starts at step 27, user_input at 26 -> regression at 31
        regressions = [
            PhaseTransition(Phase.EXECUTING, Phase.GATHERING, 15, 0.0, True),
            PhaseTransition(Phase.EXECUTING, Phase.GATHERING, 23, 0.0, True),
            PhaseTransition(Phase.EXECUTING, Phase.GATHERING, 31, 0.0, True),
        ]
        hta = _make_hta_state(
            phase=Phase.EXECUTING,
            transitions=regressions,
            total_events=len(events),
        )
        result = det.check(events, hta)
        assert result is not None
        assert result.caft_code == "2.4"
        assert result.failure_name == "goal_drift"
        assert result.evidence["drift_blocks"] >= 3

    def test_no_fire_with_consistent_tools(self):
        det = GoalDriftDetector()
        events = [
            TraceEvent(step=i, type="tool_call", tool="read_file")
            for i in range(25)
        ]
        result = det.check(events, _make_hta_state())
        assert result is None

    def test_no_fire_insufficient_blocks(self):
        """V3: Need at least 3 agent blocks."""
        det = GoalDriftDetector()
        # All in one block (no user messages)
        events = [
            TraceEvent(step=i, type="tool_call", tool="read_file" if i < 10 else "deploy")
            for i in range(25)
        ]
        hta = _make_hta_state(
            transitions=[
                PhaseTransition(Phase.EXECUTING, Phase.GATHERING, i, 0.0, True)
                for i in range(5, 15)
            ],
            total_events=25,
        )
        result = det.check(events, hta)
        # Only 1 block (no user_input separators) -> should not fire
        assert result is None

    def test_no_fire_without_regressions(self):
        """V3: Need unprompted regressions in addition to drift blocks."""
        det = GoalDriftDetector()
        events = []
        step = 1

        # 2 baseline blocks
        for _ in range(2):
            for i in range(3):
                events.append(TraceEvent(step=step, type="tool_call", tool="read_file"))
                step += 1
            events.append(TraceEvent(step=step, type="user_input", goal_text="ok"))
            step += 1

        # 4 drift blocks
        for _ in range(4):
            for tool in ["deploy", "publish", "compile"]:
                events.append(TraceEvent(step=step, type="tool_call", tool=tool))
                step += 1

        # No regressions
        hta = _make_hta_state(
            transitions=[],  # no regressions
            total_events=len(events),
        )
        result = det.check(events, hta)
        assert result is None


# --- ToolThrashingDetector ---

class TestToolThrashing:
    def test_no_fire_below_threshold(self):
        det = ToolThrashingDetector()
        events = [
            TraceEvent(step=i, type="tool_call", tool="read_file")
            for i in range(10)
        ]
        hta = _make_hta_state(phase=Phase.EXECUTING)
        result = det.check(events, hta)
        assert result is None

    def test_fires_on_15_consecutive_reads_in_executing(self):
        """V3: 15+ consecutive read-only in EXECUTING phase."""
        det = ToolThrashingDetector()
        events = [
            TraceEvent(step=i, type="tool_call", tool="read_file")
            for i in range(1, 17)  # 16 consecutive
        ]
        hta = _make_hta_state(phase=Phase.EXECUTING)
        result = det.check(events, hta)
        assert result is not None
        assert result.caft_code == "3.1"
        assert result.failure_name == "tool_thrashing"
        assert result.evidence["consecutive_read_only"] >= 15

    def test_no_fire_with_writes_interspersed(self):
        """Reads broken by writes should not trigger."""
        det = ToolThrashingDetector()
        events = []
        for i in range(20):
            if i % 5 == 4:
                events.append(TraceEvent(step=i + 1, type="tool_call", tool="edit_file"))
            else:
                events.append(TraceEvent(step=i + 1, type="tool_call", tool="read_file"))
        hta = _make_hta_state(phase=Phase.EXECUTING)
        result = det.check(events, hta)
        assert result is None

    def test_higher_threshold_in_gathering(self):
        """V3: In non-EXECUTING phases, threshold is 25."""
        det = ToolThrashingDetector()
        events = [
            TraceEvent(step=i, type="tool_call", tool="read_file")
            for i in range(1, 22)  # 21 consecutive -- below 25
        ]
        hta = _make_hta_state(phase=Phase.GATHERING)
        result = det.check(events, hta)
        assert result is None

    def test_fires_at_25_in_gathering(self):
        """V3: 25+ reads in GATHERING phase triggers."""
        det = ToolThrashingDetector()
        events = [
            TraceEvent(step=i, type="tool_call", tool="read_file")
            for i in range(1, 27)  # 26 consecutive
        ]
        hta = _make_hta_state(phase=Phase.GATHERING)
        result = det.check(events, hta)
        assert result is not None
        assert result.evidence["consecutive_read_only"] >= 25

    def test_mixed_read_tools_count(self):
        """V3: Different read-only tools (grep, glob, search) all count."""
        det = ToolThrashingDetector()
        tools = ["read_file", "grep", "glob", "search", "read_file",
                 "grep", "glob", "search", "read_file", "grep",
                 "glob", "search", "read_file", "grep", "glob", "search"]
        events = [
            TraceEvent(step=i + 1, type="tool_call", tool=tool)
            for i, tool in enumerate(tools)
        ]
        hta = _make_hta_state(phase=Phase.EXECUTING)
        result = det.check(events, hta)
        assert result is not None


# --- _segment_agent_blocks ---

class TestSegmentAgentBlocks:
    def test_single_block_no_user_messages(self):
        events = [
            TraceEvent(step=1, type="tool_call", tool="read_file"),
            TraceEvent(step=2, type="tool_call", tool="edit_file"),
        ]
        blocks = _segment_agent_blocks(events)
        assert len(blocks) == 1
        assert len(blocks[0]) == 2

    def test_multiple_blocks(self):
        events = [
            TraceEvent(step=1, type="tool_call", tool="read_file"),
            TraceEvent(step=2, type="user_input", goal_text="now fix it"),
            TraceEvent(step=3, type="tool_call", tool="edit_file"),
            TraceEvent(step=4, type="tool_call", tool="write_file"),
        ]
        blocks = _segment_agent_blocks(events)
        assert len(blocks) == 2
        assert len(blocks[0]) == 1
        assert len(blocks[1]) == 2

    def test_leading_user_message(self):
        events = [
            TraceEvent(step=1, type="user_input", goal_text="fix the bug"),
            TraceEvent(step=2, type="tool_call", tool="read_file"),
        ]
        blocks = _segment_agent_blocks(events)
        assert len(blocks) == 1


# --- run_caft_detectors ---

class TestRunCaftDetectors:
    def test_returns_empty_for_clean_trace(self):
        events = [
            TraceEvent(step=1, type="tool_call", tool="read_file"),
            TraceEvent(step=2, type="reasoning"),
        ]
        hta = _make_hta_state()
        results = run_caft_detectors(events, hta)
        assert results == []

    def test_deduplication(self):
        """Same detector at same step shouldn't fire twice."""
        events = [
            TraceEvent(step=i, type="tool_call", tool="read_file")
            for i in range(6)
        ]
        hta = _make_hta_state()
        seen = {}
        r1 = run_caft_detectors(events, hta, seen=seen)
        r2 = run_caft_detectors(events, hta, seen=seen)
        if r1:
            # Same events → same at_step → deduplicated
            assert len(r2) == 0

    def test_all_detectors_registered(self):
        """6 active detectors (V12b pruning: removed 7 zero-GT + 1 low-signal detectors)."""
        assert len(ALL_CAFT_DETECTORS) == 6
        names = {d.name for d in ALL_CAFT_DETECTORS}
        expected = {
            "context_loss", "premature_termination",
            "stall", "error_cascade",
            "analysis_paralysis", "recovery_failure",
        }
        assert names == expected

    def test_full_equals_caft_v13(self):
        """V13: FULL == CAFT (GoalDrift moved to Tier 2)."""
        assert len(ALL_CAFT_DETECTORS_FULL) == len(ALL_CAFT_DETECTORS)
        full_names = {d.name for d in ALL_CAFT_DETECTORS_FULL}
        caft_names = {d.name for d in ALL_CAFT_DETECTORS}
        assert full_names == caft_names

    def test_tier2_failure_types(self):
        """V13: Tier 2 handles PT and GoalDrift."""
        assert TIER_2_FAILURE_TYPES == {"premature_termination", "goal_drift"}


# --- CaftDiagnosis ---

class TestCaftDiagnosis:
    def test_to_dict(self):
        d = CaftDiagnosis(
            caft_code="2.2",
            caft_category="memory",
            failure_name="step_repetition",
            severity=CaftSeverity.WARNING,
            confidence=0.75,
            description="Test",
            evidence={"key": "value"},
            at_step=5,
            remediation="Fix it.",
        )
        out = d.to_dict()
        assert out["severity"] == "warning"
        assert out["caft_code"] == "2.2"
        assert out["confidence"] == 0.75

    def test_severity_values(self):
        assert CaftSeverity.INFO.value == "info"
        assert CaftSeverity.WARNING.value == "warning"
        assert CaftSeverity.CRITICAL.value == "critical"
