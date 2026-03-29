"""Synthetic trace generators for CAFT failure modes.

Each generator returns a labeled tuple: (events, expected_failures) where
expected_failures is the set of CAFT failure_names that SHOULD fire.

This is the ground truth for the validation benchmark.

V3 (2026-03-17): Updated to match V3 detector thresholds:
  - step_repetition: 9+ identical (tool, input_hash) operations, output diversity check
  - context_loss: 5+ intervening non-read ops (relative staleness), scans all pairs
  - missing_verification: 15+ executing events, multi-pattern verification detection
  - goal_drift: agent-block segmentation, 3+ drift blocks + 3+ unprompted regressions
  - tool_thrashing: NEW -- 25+ consecutive read-only in any phase
"""

from __future__ import annotations

from agentdiag.models import TraceEvent


def _e(step: int, type: str = "tool_call", tool: str | None = None, **kw) -> TraceEvent:
    """Shorthand for building test events."""
    return TraceEvent(step=step, type=type, tool=tool, **kw)


# ---------------------------------------------------------------------------
# Clean trace (should trigger ZERO CAFT detectors)
# ---------------------------------------------------------------------------

def generate_clean_trace() -> tuple[list[TraceEvent], set[str]]:
    """A healthy agent lifecycle: gather, plan, execute, verify, deliver.

    Kept to 9 events (below GoalDrift MIN_EVENTS=20) to avoid false positives.
    Uses unique output_hashes to avoid ContextLoss false positives.
    No consecutive identical tools to avoid StepRepetition.
    Expected: no CAFT failures.
    """
    events = [
        # Gathering
        _e(1, tool="read_file", latency_ms=300, output_hash="h1"),
        _e(2, tool="grep", latency_ms=200, output_hash="h2"),
        # Planning
        _e(3, type="reasoning", goal_text="Plan the fix"),
        # Executing (with writes)
        _e(4, tool="edit_file", latency_ms=400),
        _e(5, tool="write_file", latency_ms=350),
        # Verifying
        _e(6, tool="run_tests", latency_ms=2000),
        _e(7, tool="pytest", latency_ms=1500),
        # Delivering
        _e(8, tool="git_commit", latency_ms=500),
        _e(9, tool="push", latency_ms=600),
    ]
    return events, set()


# ---------------------------------------------------------------------------
# CAFT 2.2 -- Step Repetition
# ---------------------------------------------------------------------------

def generate_step_repetition_trace() -> tuple[list[TraceEvent], set[str]]:
    """Agent repeats the same operation 10 consecutive times.

    V3: Threshold=9. We generate 10 consecutive identical ops
    (no input_hash = same identity, no output_hash = no diversity exemption).
    Unique output_hashes on non-repeated events prevent ContextLoss.
    Expected: step_repetition fires.
    """
    events = [
        _e(1, tool="grep", latency_ms=200, output_hash="u1"),
        # 10 consecutive identical tool+type pairs (no input_hash = same identity)
        _e(2, tool="read_file", latency_ms=300),
        _e(3, tool="read_file", latency_ms=310),
        _e(4, tool="read_file", latency_ms=320),
        _e(5, tool="read_file", latency_ms=330),
        _e(6, tool="read_file", latency_ms=340),
        _e(7, tool="read_file", latency_ms=350),
        _e(8, tool="read_file", latency_ms=360),
        _e(9, tool="read_file", latency_ms=370),
        _e(10, tool="read_file", latency_ms=380),
        _e(11, tool="read_file", latency_ms=390),
        # Break the run
        _e(12, tool="edit_file", latency_ms=400, output_hash="u12"),
    ]
    return events, {"step_repetition"}


# ---------------------------------------------------------------------------
# CAFT 2.1 -- Context Loss
# ---------------------------------------------------------------------------

def generate_context_loss_trace() -> tuple[list[TraceEvent], set[str]]:
    """Agent reads a file, does substantial work, then re-reads the same file.

    V3: Needs max(5, 8% of total events) intervening non-read operations.
    Same output_hash on steps 1 and 8 signals re-read of same resource.
    Expected: context_loss fires.
    """
    events = [
        # First read
        _e(1, tool="read_file", latency_ms=300, output_hash="fileABC123"),
        # Substantial intervening work (6 non-read tool calls)
        _e(2, tool="edit_file", latency_ms=500),
        _e(3, tool="write_file", latency_ms=400),
        _e(4, tool="bash", latency_ms=600),
        _e(5, tool="edit_file", latency_ms=350),
        _e(6, tool="write_file", latency_ms=400),
        _e(7, tool="bash", latency_ms=500),
        # Re-read same resource (same output_hash)
        _e(8, tool="read_file", latency_ms=300, output_hash="fileABC123"),
        # More work
        _e(9, tool="edit_file", latency_ms=400),
    ]
    return events, {"context_loss"}


# ---------------------------------------------------------------------------
# CAFT 5.4 -- Premature Termination
# ---------------------------------------------------------------------------

def generate_premature_termination_trace() -> tuple[list[TraceEvent], set[str]]:
    """Agent goes from EXECUTING directly to DELIVERING, never VERIFYING.

    V3: Needs 3+ delivering events (MIN_DELIVERING_EVENTS=3) to fire.
    Brief delivery transitions (1-2 events) are ignored.
    Expected: premature_termination fires.
    """
    events = [
        # Gathering (2 events for hysteresis)
        _e(1, tool="read_file", latency_ms=300, output_hash="p1"),
        _e(2, tool="grep", latency_ms=200, output_hash="p2"),
        # Executing (4 events -- surpasses MIN_EXECUTE_EVENTS=3)
        _e(3, tool="edit_file", latency_ms=400),
        _e(4, tool="write_file", latency_ms=350),
        _e(5, tool="edit_file", latency_ms=300),
        _e(6, tool="write_file", latency_ms=400),
        # Delivering (4 events -- surpasses MIN_DELIVERING_EVENTS=3)
        _e(7, tool="git_commit", latency_ms=500),
        _e(8, tool="push", latency_ms=600),
        _e(9, tool="git_commit", latency_ms=500),
        _e(10, tool="push", latency_ms=600),
    ]
    return events, {"premature_termination"}


# ---------------------------------------------------------------------------
# CAFT 5.3 -- Missing Verification
# ---------------------------------------------------------------------------

def generate_missing_verification_trace() -> tuple[list[TraceEvent], set[str]]:
    """Agent does 16+ execution events with file writes but zero verification.

    V3: Multi-pattern detection means we need NO verification of any kind:
    no pytest in Bash, no Task subagent, no user verification, no reasoning
    acknowledgment. Unique hashes prevent ContextLoss.
    Expected: missing_verification fires.
    """
    events = [
        _e(1, tool="read_file", latency_ms=300, output_hash="mv1"),
        _e(2, tool="edit_file", latency_ms=400),
        _e(3, tool="write_file", latency_ms=350),
        _e(4, tool="bash", latency_ms=200),
        _e(5, tool="read_file", latency_ms=250, output_hash="mv2"),
        _e(6, tool="edit_file", latency_ms=300),
        _e(7, tool="write_file", latency_ms=400),
        _e(8, tool="bash", latency_ms=200),
        _e(9, tool="read_file", latency_ms=200, output_hash="mv3"),
        _e(10, tool="edit_file", latency_ms=350),
        _e(11, tool="write_file", latency_ms=300),
        _e(12, tool="bash", latency_ms=200),
        _e(13, tool="read_file", latency_ms=250, output_hash="mv4"),
        _e(14, tool="edit_file", latency_ms=400),
        _e(15, tool="write_file", latency_ms=350),
        _e(16, tool="bash", latency_ms=200),
        _e(17, tool="read_file", latency_ms=300, output_hash="mv5"),
        _e(18, tool="edit_file", latency_ms=300),
        _e(19, tool="write_file", latency_ms=400),
        _e(20, tool="edit_file", latency_ms=350),
    ]
    return events, {"missing_verification"}


# ---------------------------------------------------------------------------
# CAFT 6.4 -- Reasoning-Action Mismatch
# ---------------------------------------------------------------------------

def generate_reasoning_action_mismatch_trace() -> tuple[list[TraceEvent], set[str]]:
    """Agent says 'I should read the file' then immediately writes.

    Short trace to avoid GoalDrift. Unique hashes to avoid ContextLoss.
    Expected: reasoning_action_mismatch fires.
    """
    events = [
        _e(1, tool="read_file", latency_ms=300, output_hash="rm1"),
        # Reasoning says "read/review" but next action is write
        _e(2, type="reasoning", goal_text="Let me examine the configuration file"),
        _e(3, tool="write_file", latency_ms=400),
        # Continue normally
        _e(4, tool="read_file", latency_ms=300, output_hash="rm2"),
    ]
    return events, {"reasoning_action_mismatch"}


# ---------------------------------------------------------------------------
# CAFT 2.4 -- Goal Drift
# ---------------------------------------------------------------------------

def generate_goal_drift_trace() -> tuple[list[TraceEvent], set[str]]:
    """Agent has 3+ drifted agent blocks + 3+ unprompted phase regressions.

    V3: Agent-block segmentation. Events are segmented by user_input events.
    Baseline tools from first 2 blocks. Later blocks with >50% novel tools
    count as "drift blocks".

    Structure:
    - Block 1 (baseline): read_file, grep (GATHERING)
    - Block 2 (baseline): read_file, grep (GATHERING)
    - Blocks 3-5 (drift): bash+fetch oscillation (EXECUTING/GATHERING)
      Each separated by user_input, 10 events per block.
      bash=EXECUTING (strong), fetch=GATHERING (needs hysteresis).
      Pairs of fetch cause EXECUTING->GATHERING regressions mid-block
      (away from user_input, so unprompted).

    All output_hashes unique to avoid ContextLoss false positive.
    Expected: goal_drift fires.
    """
    events = []
    step = 1

    # Block 1 (baseline): focused gathering
    for i in range(4):
        tool = "read_file" if i % 2 == 0 else "grep"
        events.append(_e(step, tool=tool, latency_ms=300, output_hash=f"gd_{step}"))
        step += 1

    # User message (separates block 1 from block 2)
    events.append(_e(step, type="user_input", goal_text="ok"))
    step += 1

    # Block 2 (baseline): more focused gathering
    for i in range(4):
        tool = "read_file" if i % 2 == 0 else "grep"
        events.append(_e(step, tool=tool, latency_ms=300, output_hash=f"gd_{step}"))
        step += 1

    # User message (separates block 2 from drift blocks)
    events.append(_e(step, type="user_input", goal_text="ok"))
    step += 1

    # Blocks 3-5: drifted (novel tools {bash, fetch} NOT in baseline {read_file, grep})
    # Each block: bash*4, fetch*2, bash, fetch*2, bash (10 events)
    # fetch pairs cause GATHERING transition via hysteresis after EXECUTING.
    # Regressions at mid-block positions are >3 steps from user_input = unprompted.
    drift_block = ["bash", "bash", "bash", "bash", "fetch", "fetch",
                    "bash", "fetch", "fetch", "bash"]
    for block_num in range(3):
        for tool in drift_block:
            events.append(_e(step, tool=tool, latency_ms=400, output_hash=f"gd_{step}"))
            step += 1
        if block_num < 2:
            events.append(_e(step, type="user_input", goal_text="ok"))
            step += 1

    return events, {"goal_drift"}


# ---------------------------------------------------------------------------
# CAFT 3.1 -- Tool Thrashing
# ---------------------------------------------------------------------------

def generate_tool_thrashing_trace() -> tuple[list[TraceEvent], set[str]]:
    """Agent performs 26+ consecutive read-only operations.

    V3: NEW detector. The agent is stuck in analysis paralysis -- reading and
    searching without taking action. After initial writes, HTA transitions to
    GATHERING after enough reads, so threshold becomes ANY_PHASE_THRESHOLD=25.
    We generate 26 consecutive reads to exceed that.

    Uses cycling read tools (read_file, grep, glob, search) to avoid
    StepRepetition FP (max consecutive identical = 1).
    Unique output_hashes to avoid ContextLoss.
    Expected: tool_thrashing fires.
    """
    events = [
        # Initial writes to establish EXECUTING phase
        _e(1, tool="edit_file", latency_ms=400),
        _e(2, tool="write_file", latency_ms=350),
        _e(3, tool="edit_file", latency_ms=300),
    ]
    step = 4
    # 26 consecutive read-only operations (cycling to avoid step_repetition)
    read_cycle = ["read_file", "grep", "glob", "search"]
    for i in range(26):
        tool = read_cycle[i % len(read_cycle)]
        events.append(_e(step, tool=tool, latency_ms=200, output_hash=f"tt_{step}"))
        step += 1

    return events, {"tool_thrashing"}


# ---------------------------------------------------------------------------
# Multi-failure trace (triggers multiple CAFT detectors)
# ---------------------------------------------------------------------------

def generate_multi_failure_trace() -> tuple[list[TraceEvent], set[str]]:
    """A pathological trace that triggers step_repetition + premature_termination.

    V3: This trace now triggers:
    - step_repetition: 10 consecutive edit_file (no input_hash = same identity, threshold=9)
    - premature_termination: goes to DELIVERING without VERIFYING (3+ delivering events)
    """
    events = [
        # Gathering
        _e(1, tool="read_file", latency_ms=300, output_hash="mf1"),
        _e(2, tool="grep", latency_ms=200, output_hash="mf2"),
        # Executing (4+ events for MIN_EXECUTE_EVENTS)
        _e(3, tool="edit_file", latency_ms=400),
        _e(4, tool="write_file", latency_ms=350),
        _e(5, tool="edit_file", latency_ms=300),
        _e(6, tool="write_file", latency_ms=400),
        # 10 consecutive edit_file -> step_repetition fires (threshold=9)
        _e(7, tool="edit_file", latency_ms=400),
        _e(8, tool="edit_file", latency_ms=410),
        _e(9, tool="edit_file", latency_ms=420),
        _e(10, tool="edit_file", latency_ms=430),
        _e(11, tool="edit_file", latency_ms=440),
        _e(12, tool="edit_file", latency_ms=450),
        _e(13, tool="edit_file", latency_ms=460),
        _e(14, tool="edit_file", latency_ms=470),
        _e(15, tool="edit_file", latency_ms=480),
        _e(16, tool="edit_file", latency_ms=490),
        # Skip verification -> DELIVERING (4 events for MIN_DELIVERING_EVENTS=3)
        _e(17, tool="git_commit", latency_ms=500),
        _e(18, tool="push", latency_ms=600),
        _e(19, tool="git_commit", latency_ms=500),
        _e(20, tool="push", latency_ms=600),
    ]
    return events, {"step_repetition", "premature_termination"}


# ---------------------------------------------------------------------------
# Multi-agent trace (main agent spawns sub-agent via Task)
# ---------------------------------------------------------------------------

def generate_multi_agent_trace() -> tuple[list[TraceEvent], set[str]]:
    """Main agent spawns a sub-agent via Task tool; both work in parallel.

    Demonstrates multi-agent visualization without triggering CAFT detectors.
    The trace is short (< 20 events) to avoid GoalDrift/MissingVerification.
    All output_hashes unique to avoid ContextLoss.

    Expected: no CAFT failures (clean multi-agent workflow).
    """
    events = [
        # Main agent: gathering
        _e(1, tool="read_file", latency_ms=250, output_hash="ma_1"),
        _e(2, tool="grep", latency_ms=200, output_hash="ma_2"),
        # Main agent spawns sub-agent
        _e(3, tool="Task", latency_ms=100, output_hash="ma_3"),
        # Sub-agent works (agent_id="S1")
        TraceEvent(step=4, type="tool_call", tool="read_file",
                   latency_ms=300, output_hash="sa_4", agent_id="S1"),
        TraceEvent(step=5, type="tool_call", tool="grep",
                   latency_ms=200, output_hash="sa_5", agent_id="S1"),
        TraceEvent(step=6, type="tool_call", tool="edit_file",
                   latency_ms=400, agent_id="S1"),
        # Main agent continues in parallel
        _e(7, tool="edit_file", latency_ms=350),
        _e(8, tool="write_file", latency_ms=300),
        # Sub-agent returns result
        TraceEvent(step=9, type="tool_result", tool="Task",
                   latency_ms=50, agent_id="S1"),
        # Main agent verifies and delivers
        _e(10, tool="run_tests", latency_ms=2000),
        _e(11, tool="git_commit", latency_ms=500),
        _e(12, tool="push", latency_ms=600),
    ]
    return events, set()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

CAFT_GENERATORS: dict[str, callable] = {
    "clean": generate_clean_trace,
    "step_repetition": generate_step_repetition_trace,
    "context_loss": generate_context_loss_trace,
    "premature_termination": generate_premature_termination_trace,
    "missing_verification": generate_missing_verification_trace,
    "reasoning_action_mismatch": generate_reasoning_action_mismatch_trace,
    "goal_drift": generate_goal_drift_trace,
    "tool_thrashing": generate_tool_thrashing_trace,
    "multi_failure": generate_multi_failure_trace,
    "multi_agent": generate_multi_agent_trace,
}

ALL_CAFT_SCENARIO_NAMES = list(CAFT_GENERATORS.keys())
