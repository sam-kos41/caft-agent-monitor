"""Tests for the HTA (Hierarchical Task Analysis) state machine."""

import pytest
from agentdiag.models import TraceEvent
from agentdiag.hta import (
    Phase,
    classify_event,
    HTAStateMachine,
    HTAState,
    PhaseTransition,
)


# --- classify_event ---

def test_classify_read_tool_as_gathering():
    e = TraceEvent(step=1, type="tool_call", tool="read_file")
    assert classify_event(e)[0] == Phase.GATHERING


def test_classify_search_tool_as_gathering():
    e = TraceEvent(step=1, type="tool_call", tool="grep")
    assert classify_event(e)[0] == Phase.GATHERING


def test_classify_reasoning_as_planning():
    e = TraceEvent(step=1, type="reasoning")
    assert classify_event(e)[0] == Phase.PLANNING


def test_classify_thinking_as_planning():
    e = TraceEvent(step=1, type="thinking")
    assert classify_event(e)[0] == Phase.PLANNING


def test_classify_write_tool_as_executing():
    e = TraceEvent(step=1, type="tool_call", tool="write_file")
    assert classify_event(e)[0] == Phase.EXECUTING


def test_classify_edit_tool_as_executing():
    e = TraceEvent(step=1, type="tool_call", tool="edit_file")
    assert classify_event(e)[0] == Phase.EXECUTING


def test_classify_bash_as_executing():
    e = TraceEvent(step=1, type="tool_call", tool="bash")
    assert classify_event(e)[0] == Phase.EXECUTING


def test_classify_test_tool_as_verifying():
    e = TraceEvent(step=1, type="tool_call", tool="run_tests")
    assert classify_event(e)[0] == Phase.VERIFYING


def test_classify_pytest_as_verifying():
    e = TraceEvent(step=1, type="tool_call", tool="pytest")
    assert classify_event(e)[0] == Phase.VERIFYING


def test_classify_commit_as_delivering():
    e = TraceEvent(step=1, type="tool_call", tool="git_commit")
    assert classify_event(e)[0] == Phase.DELIVERING


def test_classify_output_as_delivering():
    e = TraceEvent(step=1, type="output")
    assert classify_event(e)[0] == Phase.DELIVERING


def test_classify_unknown_tool_call_as_executing():
    e = TraceEvent(step=1, type="tool_call", tool="custom_unknown_tool")
    assert classify_event(e)[0] == Phase.EXECUTING


def test_classify_unknown_type_as_planning():
    e = TraceEvent(step=1, type="some_unknown_type")
    assert classify_event(e)[0] == Phase.PLANNING


# --- HTAStateMachine ---

def test_initial_state_is_idle():
    sm = HTAStateMachine(goal="test")
    state = sm.state
    assert state.current_phase == Phase.IDLE
    assert state.total_events == 0
    assert state.goal == "test"


def test_first_event_transitions_from_idle():
    sm = HTAStateMachine()
    e = TraceEvent(step=1, type="tool_call", tool="read_file")
    state = sm.push(e)
    assert state.current_phase == Phase.GATHERING
    assert state.total_events == 1


def test_hysteresis_requires_two_events():
    """Phase doesn't change on a single event in a new phase."""
    sm = HTAStateMachine()
    # Start in gathering
    sm.push(TraceEvent(step=1, type="tool_call", tool="read_file"))
    sm.push(TraceEvent(step=2, type="tool_call", tool="grep"))

    # Single planning event should NOT transition yet
    state = sm.push(TraceEvent(step=3, type="reasoning"))
    assert state.current_phase == Phase.GATHERING

    # Second consecutive planning event should transition
    state = sm.push(TraceEvent(step=4, type="reasoning"))
    assert state.current_phase == Phase.PLANNING


def test_hysteresis_resets_on_current_phase():
    """If we see current-phase event, pending count resets."""
    sm = HTAStateMachine()
    sm.push(TraceEvent(step=1, type="tool_call", tool="read_file"))
    sm.push(TraceEvent(step=2, type="tool_call", tool="grep"))

    # One planning event
    sm.push(TraceEvent(step=3, type="reasoning"))
    # Back to gathering — resets pending
    sm.push(TraceEvent(step=4, type="tool_call", tool="read_file"))
    # One more planning — still shouldn't transition (count reset)
    state = sm.push(TraceEvent(step=5, type="reasoning"))
    assert state.current_phase == Phase.GATHERING


def test_full_lifecycle_forward():
    """Walk through all phases in order."""
    sm = HTAStateMachine(goal="Fix bug")

    # Gathering
    sm.push(TraceEvent(step=1, type="tool_call", tool="read_file"))
    sm.push(TraceEvent(step=2, type="tool_call", tool="grep"))

    # Planning (need 2 for hysteresis)
    sm.push(TraceEvent(step=3, type="reasoning"))
    state = sm.push(TraceEvent(step=4, type="reasoning"))
    assert state.current_phase == Phase.PLANNING

    # Executing
    sm.push(TraceEvent(step=5, type="tool_call", tool="edit_file"))
    state = sm.push(TraceEvent(step=6, type="tool_call", tool="write_file"))
    assert state.current_phase == Phase.EXECUTING

    # Verifying
    sm.push(TraceEvent(step=7, type="tool_call", tool="run_tests"))
    state = sm.push(TraceEvent(step=8, type="tool_call", tool="pytest"))
    assert state.current_phase == Phase.VERIFYING

    # Delivering
    sm.push(TraceEvent(step=9, type="tool_call", tool="git_commit"))
    state = sm.push(TraceEvent(step=10, type="tool_call", tool="push"))
    assert state.current_phase == Phase.DELIVERING


def test_regression_detected():
    """Going from EXECUTING back to GATHERING is a regression."""
    sm = HTAStateMachine()
    # Get to executing
    sm.push(TraceEvent(step=1, type="tool_call", tool="read_file"))
    sm.push(TraceEvent(step=2, type="tool_call", tool="edit_file"))
    sm.push(TraceEvent(step=3, type="tool_call", tool="write_file"))

    # Go back to gathering
    sm.push(TraceEvent(step=4, type="tool_call", tool="read_file"))
    state = sm.push(TraceEvent(step=5, type="tool_call", tool="grep"))
    assert state.current_phase == Phase.GATHERING
    assert state.regression_count >= 1


def test_phase_event_counts():
    sm = HTAStateMachine()
    sm.push(TraceEvent(step=1, type="tool_call", tool="read_file"))
    sm.push(TraceEvent(step=2, type="tool_call", tool="grep"))
    sm.push(TraceEvent(step=3, type="tool_call", tool="read_file"))
    state = sm.push(TraceEvent(step=4, type="reasoning"))

    assert state.phase_event_counts.get("gathering", 0) == 3
    assert state.phase_event_counts.get("planning", 0) == 1


def test_progress_pct_increases():
    sm = HTAStateMachine()
    # Gathering
    sm.push(TraceEvent(step=1, type="tool_call", tool="read_file"))
    p1 = sm.state.progress_pct

    # Move to executing
    sm.push(TraceEvent(step=2, type="tool_call", tool="edit_file"))
    sm.push(TraceEvent(step=3, type="tool_call", tool="write_file"))
    p2 = sm.state.progress_pct

    assert p2 > p1


def test_set_goal():
    sm = HTAStateMachine(goal="initial")
    assert sm.state.goal == "initial"
    sm.set_goal("updated")
    assert sm.state.goal == "updated"


def test_completed_nodes_tracked():
    sm = HTAStateMachine()
    # Gathering
    sm.push(TraceEvent(step=1, type="tool_call", tool="read_file"))
    sm.push(TraceEvent(step=2, type="tool_call", tool="grep"))
    # Transition to executing (skipping planning for simplicity)
    sm.push(TraceEvent(step=3, type="tool_call", tool="edit_file"))
    sm.push(TraceEvent(step=4, type="tool_call", tool="write_file"))

    state = sm.state
    # Should have at least one completed node (gathering)
    assert len(state.completed_nodes) >= 1
    assert any(n.phase == Phase.GATHERING for n in state.completed_nodes)


# --- Strong signal (hysteresis bypass) ---

def test_strong_signal_write_bypasses_hysteresis():
    """A single Write call should transition from GATHERING to EXECUTING."""
    sm = HTAStateMachine()
    sm.push(TraceEvent(step=1, type="tool_call", tool="read_file"))
    sm.push(TraceEvent(step=2, type="tool_call", tool="grep"))
    assert sm.state.current_phase == Phase.GATHERING

    # Single Write → strong signal → immediate transition
    state = sm.push(TraceEvent(step=3, type="tool_call", tool="Write"))
    assert state.current_phase == Phase.EXECUTING


def test_strong_signal_edit_bypasses_hysteresis():
    """A single Edit call should transition from GATHERING to EXECUTING."""
    sm = HTAStateMachine()
    sm.push(TraceEvent(step=1, type="tool_call", tool="read_file"))
    sm.push(TraceEvent(step=2, type="tool_call", tool="grep"))

    state = sm.push(TraceEvent(step=3, type="tool_call", tool="Edit"))
    assert state.current_phase == Phase.EXECUTING


def test_interleaved_read_write_reaches_executing():
    """The original bug: Read→Write→Read→Write stuck in GATHERING.

    With strong signals, every Write immediately transitions to EXECUTING.
    """
    sm = HTAStateMachine()
    sm.push(TraceEvent(step=1, type="tool_call", tool="Read"))
    sm.push(TraceEvent(step=2, type="tool_call", tool="Read"))
    assert sm.state.current_phase == Phase.GATHERING

    # Write is strong → EXECUTING
    state = sm.push(TraceEvent(step=3, type="tool_call", tool="Write"))
    assert state.current_phase == Phase.EXECUTING

    # Read → back to pending GATHERING, but only 1 event
    state = sm.push(TraceEvent(step=4, type="tool_call", tool="Read"))
    # Still EXECUTING (1 gathering not enough for hysteresis)
    assert state.current_phase == Phase.EXECUTING

    # Another Write → strong → EXECUTING (already there, stays)
    state = sm.push(TraceEvent(step=5, type="tool_call", tool="Write"))
    assert state.current_phase == Phase.EXECUTING


def test_strong_signal_bash_bypasses_hysteresis():
    """Bash is a strong executing signal."""
    sm = HTAStateMachine()
    sm.push(TraceEvent(step=1, type="tool_call", tool="read_file"))
    sm.push(TraceEvent(step=2, type="tool_call", tool="grep"))

    state = sm.push(TraceEvent(step=3, type="tool_call", tool="Bash"))
    assert state.current_phase == Phase.EXECUTING


def test_strong_signal_pytest_bypasses_hysteresis():
    """pytest is a strong verifying signal."""
    sm = HTAStateMachine()
    sm.push(TraceEvent(step=1, type="tool_call", tool="edit_file"))
    sm.push(TraceEvent(step=2, type="tool_call", tool="write_file"))

    state = sm.push(TraceEvent(step=3, type="tool_call", tool="pytest"))
    assert state.current_phase == Phase.VERIFYING


def test_strong_signal_commit_bypasses_hysteresis():
    """commit is a strong delivering signal."""
    sm = HTAStateMachine()
    sm.push(TraceEvent(step=1, type="tool_call", tool="edit_file"))
    sm.push(TraceEvent(step=2, type="tool_call", tool="write_file"))

    state = sm.push(TraceEvent(step=3, type="tool_call", tool="commit"))
    assert state.current_phase == Phase.DELIVERING


def test_classify_event_returns_strong_flag():
    """classify_event returns (phase, is_strong) tuple."""
    # Strong tools
    phase, strong = classify_event(TraceEvent(step=1, type="tool_call", tool="Write"))
    assert phase == Phase.EXECUTING
    assert strong is True

    # Non-strong tools
    phase, strong = classify_event(TraceEvent(step=1, type="tool_call", tool="read_file"))
    assert phase == Phase.GATHERING
    assert strong is False

    # Reasoning is not strong
    phase, strong = classify_event(TraceEvent(step=1, type="reasoning"))
    assert phase == Phase.PLANNING
    assert strong is False


# --- Phase properties ---

def test_phase_label():
    assert Phase.GATHERING.label == "gathering"
    assert Phase.DELIVERING.label == "delivering"


def test_phase_color():
    assert Phase.GATHERING.color == "cyan"
    assert Phase.EXECUTING.color == "green"
