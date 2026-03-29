"""Hierarchical Task Analysis state machine.

Infers WHERE an agent is in a task hierarchy by classifying raw events
into task phases. This is the core innovation of the real-time dashboard:
rather than predefined HTA trees, the system infers the task structure
from observed behavior.

Phases follow the universal agent task lifecycle:
  GATHERING → PLANNING → EXECUTING → VERIFYING → DELIVERING

The state machine tracks transitions, detects phase regressions (going
backward), and computes time-in-state for progress estimation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

from agentdiag.models import TraceEvent


class Phase(IntEnum):
    """Task phases in expected forward order."""
    IDLE = 0
    GATHERING = 1    # reading files, searching, fetching context
    PLANNING = 2     # reasoning, outlining, designing approach
    EXECUTING = 3    # writing code, editing files, running commands
    VERIFYING = 4    # running tests, reviewing output, checking results
    DELIVERING = 5   # committing, summarizing, presenting final output

    @property
    def label(self) -> str:
        return self.name.lower()

    @property
    def color(self) -> str:
        """Rich color for terminal display."""
        return {
            Phase.IDLE: "dim",
            Phase.GATHERING: "cyan",
            Phase.PLANNING: "yellow",
            Phase.EXECUTING: "green",
            Phase.VERIFYING: "magenta",
            Phase.DELIVERING: "blue",
        }[self]


# Tool/event classification rules.
# Each set maps tool names or event types to their most likely phase.
_GATHERING_TOOLS = {
    "read_file", "search_docs", "web_search", "search_codebase",
    "list_files", "glob", "grep", "find", "cat", "head", "tail",
    "read", "fetch", "get", "list", "ls", "describe",
}
_PLANNING_TYPES = {"reasoning", "planning", "thinking"}
_EXECUTING_TOOLS = {
    "write_file", "edit_file", "edit", "write", "create_file",
    "run_code", "execute", "bash", "shell", "npm", "pip",
    "install", "build", "compile", "generate",
}
_VERIFYING_TOOLS = {
    "run_tests", "test", "pytest", "jest", "check", "lint",
    "validate", "verify", "review", "diff", "compare",
}
_DELIVERING_TOOLS = {
    "commit", "git_commit", "push", "deploy", "publish",
    "submit", "send", "deliver", "output", "summarize",
}

# Strong-signal tools bypass hysteresis (1 event is sufficient).
# These are unambiguous phase indicators that should never be suppressed
# by the 2-event hysteresis window. Without this, interleaved patterns
# like Read → Write → Read → Write never transition out of GATHERING.
_STRONG_EXECUTING = {"write", "edit", "write_file", "edit_file", "create_file", "bash"}
_STRONG_VERIFYING = {"pytest", "jest", "run_tests"}
_STRONG_DELIVERING = {"commit", "git_commit", "push", "deploy"}


def classify_event(event: TraceEvent) -> tuple[Phase, bool]:
    """Classify a single event into an HTA phase.

    Returns:
        (phase, is_strong) — is_strong means this tool is an unambiguous
        phase indicator that should bypass hysteresis.
    """
    # Reasoning/planning events
    if event.type in _PLANNING_TYPES:
        return Phase.PLANNING, False

    # Tool calls: classify by tool name
    if event.tool:
        tool_lower = event.tool.lower().replace("-", "_").replace(" ", "_")
        # Check each phase's tool set (most specific first)
        if any(t in tool_lower for t in _DELIVERING_TOOLS):
            strong = any(t in tool_lower for t in _STRONG_DELIVERING)
            return Phase.DELIVERING, strong
        if any(t in tool_lower for t in _VERIFYING_TOOLS):
            strong = any(t in tool_lower for t in _STRONG_VERIFYING)
            return Phase.VERIFYING, strong
        if any(t in tool_lower for t in _EXECUTING_TOOLS):
            strong = any(t in tool_lower for t in _STRONG_EXECUTING)
            return Phase.EXECUTING, strong
        if any(t in tool_lower for t in _GATHERING_TOOLS):
            return Phase.GATHERING, False

    # Output events
    if event.type == "output":
        return Phase.DELIVERING, False

    # Default: if it's a tool_call we can't classify, treat as executing
    if event.type == "tool_call":
        return Phase.EXECUTING, False

    return Phase.PLANNING, False


@dataclass
class PhaseTransition:
    """Record of a phase change."""
    from_phase: Phase
    to_phase: Phase
    at_step: int
    at_time: float
    is_regression: bool   # went backward in the lifecycle


@dataclass
class HTANode:
    """A node in the inferred HTA tree."""
    phase: Phase
    start_step: int
    start_time: float
    end_step: Optional[int] = None
    end_time: Optional[float] = None
    event_count: int = 0
    tool_counts: dict[str, int] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        end = self.end_time if self.end_time is not None else time.monotonic()
        return end - self.start_time

    @property
    def is_active(self) -> bool:
        return self.end_step is None


@dataclass
class HTAState:
    """Complete HTA state at any point in time."""
    goal: str
    current_phase: Phase
    current_node: HTANode
    completed_nodes: list[HTANode]
    transitions: list[PhaseTransition]
    total_events: int
    phase_event_counts: dict[str, int]

    @property
    def progress_pct(self) -> float:
        """Estimated completion percentage based on phase progression."""
        if self.current_phase == Phase.IDLE:
            return 0.0
        # Weight phases: gathering=20%, planning=15%, executing=40%, verifying=20%, delivering=5%
        weights = {
            Phase.GATHERING: 0.20,
            Phase.PLANNING: 0.35,
            Phase.EXECUTING: 0.75,
            Phase.VERIFYING: 0.95,
            Phase.DELIVERING: 1.00,
        }
        # Base progress from reaching current phase
        base = weights.get(self.current_phase, 0.0)
        # Subtract a bit since we're IN the phase, not past it
        prev_phases = [p for p in Phase if p < self.current_phase and p != Phase.IDLE]
        if prev_phases:
            prev_max = weights.get(max(prev_phases), 0.0)
        else:
            prev_max = 0.0
        # Interpolate within current phase based on event count
        # (more events in phase = closer to transitioning out)
        phase_events = self.phase_event_counts.get(self.current_phase.label, 0)
        # Assume ~10 events per phase as a rough normalization
        phase_progress = min(phase_events / 10.0, 1.0)
        return prev_max + (base - prev_max) * phase_progress

    @property
    def regression_count(self) -> int:
        return sum(1 for t in self.transitions if t.is_regression)


class HTAStateMachine:
    """Infers task structure from a stream of TraceEvents.

    The machine classifies each event into a phase, tracks transitions,
    and builds an HTA tree of completed and active phase nodes.

    Key design decisions:
    - Rule-based classification (fast, deterministic, no API dependency)
    - Hysteresis: requires 2+ consecutive events in a new phase before
      transitioning (prevents single-event noise from causing flicker)
    - Supports phase regression (going backward) which is normal behavior
      (e.g., executing → gathering to read another file)
    """

    HYSTERESIS = 2  # events in new phase before transition commits

    def __init__(self, goal: str = ""):
        self._goal = goal
        self._current_phase = Phase.IDLE
        self._current_node = HTANode(
            phase=Phase.IDLE,
            start_step=0,
            start_time=time.monotonic(),
        )
        self._completed_nodes: list[HTANode] = []
        self._transitions: list[PhaseTransition] = []
        self._total_events = 0
        self._phase_event_counts: dict[str, int] = {}

        # Hysteresis state
        self._pending_phase: Optional[Phase] = None
        self._pending_count = 0

    @property
    def state(self) -> HTAState:
        return HTAState(
            goal=self._goal,
            current_phase=self._current_phase,
            current_node=self._current_node,
            completed_nodes=list(self._completed_nodes),
            transitions=list(self._transitions),
            total_events=self._total_events,
            phase_event_counts=dict(self._phase_event_counts),
        )

    def push(self, event: TraceEvent) -> HTAState:
        """Process an event and return updated HTA state."""
        self._total_events += 1
        now = time.monotonic()

        classified, is_strong = classify_event(event)

        # Update phase event counts
        key = classified.label
        self._phase_event_counts[key] = self._phase_event_counts.get(key, 0) + 1

        # Update current node tool counts
        tool_name = event.tool or event.type
        self._current_node.tool_counts[tool_name] = (
            self._current_node.tool_counts.get(tool_name, 0) + 1
        )
        self._current_node.event_count += 1

        # Hysteresis: only transition if we see N consecutive events in new phase.
        # Strong signals (Write, Edit, Bash, pytest, commit) bypass hysteresis
        # because they are unambiguous phase indicators.
        if classified != self._current_phase:
            if is_strong:
                # Strong signal: transition immediately
                self._transition_to(classified, event.step, now)
                self._pending_phase = None
                self._pending_count = 0
            elif classified == self._pending_phase:
                self._pending_count += 1
                if self._pending_count >= self.HYSTERESIS:
                    self._transition_to(classified, event.step, now)
                    self._pending_phase = None
                    self._pending_count = 0
            else:
                self._pending_phase = classified
                self._pending_count = 1
        else:
            # Reset pending if we see current phase again
            self._pending_phase = None
            self._pending_count = 0

        # Special case: first real event transitions from IDLE
        if self._current_phase == Phase.IDLE and self._total_events == 1:
            self._transition_to(classified, event.step, now)

        return self.state

    def _transition_to(self, new_phase: Phase, step: int, now: float) -> None:
        """Commit a phase transition."""
        if new_phase == self._current_phase:
            return

        is_regression = new_phase < self._current_phase

        # Close current node
        self._current_node.end_step = step
        self._current_node.end_time = now
        if self._current_node.event_count > 0:
            self._completed_nodes.append(self._current_node)

        # Record transition
        self._transitions.append(PhaseTransition(
            from_phase=self._current_phase,
            to_phase=new_phase,
            at_step=step,
            at_time=now,
            is_regression=is_regression,
        ))

        # Open new node
        self._current_phase = new_phase
        self._current_node = HTANode(
            phase=new_phase,
            start_step=step,
            start_time=now,
        )

    def set_goal(self, goal: str) -> None:
        """Update the tracked goal text."""
        self._goal = goal
