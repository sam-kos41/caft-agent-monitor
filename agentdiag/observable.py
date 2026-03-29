"""
Shared Contract: ObservableEvent
================================
Both agents build against this interface. Agent 1 (OpenViking + harness)
produces these events. Agent 2 (visualization) consumes them.

Neither agent needs to understand the other's domain -- the contract
is the bridge.

Usage:
  - Agent 1: Emit ObservableEvent with memory/harness fields populated
  - Agent 2: Call event.to_symbol() for IT computation, check event_type
    for rendering decisions (e.g., PHASE_BOUNDARY -> vertical line, not data point)
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EventType(Enum):
    """All observable event types across both agent domains."""

    # Tool-level observation (Agent 2: visualization domain)
    TOOL_CALL = "tool_call"
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    SHELL_COMMAND = "shell_command"

    # Harness-level observation (Agent 1: orchestration domain)
    PHASE_BOUNDARY = "phase_boundary"
    CONTRACT_EVENT = "contract_event"
    EVALUATION_RESULT = "evaluation_result"
    AGENT_HANDOFF = "agent_handoff"

    # Memory-level observation (Agent 1: OpenViking domain)
    MEMORY_LOAD = "memory_load"
    MEMORY_STORE = "memory_store"
    MEMORY_EVICT = "memory_evict"
    MEMORY_TIER_ESCALATION = "memory_tier_escalation"

    # Meta (both domains)
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    ERROR = "error"


class MemoryTier(Enum):
    """OpenViking's three-tier context loading levels."""
    L0 = "l0"  # ~100 tokens  -- abstract/title
    L1 = "l1"  # ~2k tokens   -- overview/summary
    L2 = "l2"  # full content  -- detailed document


class HarnessPhase(Enum):
    """Phases declared by the three-agent harness orchestrator."""
    PLANNING = "planning"
    CONTRACT_NEGOTIATION = "contract_negotiation"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    ITERATING = "iterating"
    RETROSPECTIVE = "retrospective"


class AgentRole(Enum):
    """Which agent in the harness produced this event."""
    PLANNER = "planner"
    GENERATOR = "generator"
    EVALUATOR = "evaluator"
    ORCHESTRATOR = "orchestrator"


# ---------------------------------------------------------------------------
# Core Event
# ---------------------------------------------------------------------------

@dataclass
class ObservableEvent:
    """
    The universal event that both systems produce and consume.

    Design principles:
      - All fields except step, timestamp, event_type are Optional
      - Each agent populates only the fields relevant to its domain
      - to_symbol() produces a string for information-theoretic computation
      - to_dict() produces a JSON-serializable dict for WebSocket transport

    Agent 2 (visualization) rules:
      - Push to_symbol() into SymbolStream for entropy/MI/compression
      - PHASE_BOUNDARY events -> vertical markers on sparklines, NOT data points
      - MEMORY_LOAD events -> memory operation track on timeline
      - EVALUATION_RESULT events -> calibration signal (retrospective)

    Agent 1 (harness + OpenViking) rules:
      - Emit one event per observable action
      - Populate memory fields for any viking:// operation
      - Populate harness fields for phase transitions and evaluator grades
      - Let to_symbol() handle the abstraction
    """

    # --- Required fields (both agents must provide) ---
    step: int
    timestamp: float
    event_type: EventType

    # --- Tool-level fields (Agent 2's primary domain) ---
    tool_name: Optional[str] = None
    target_path: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    duration_ms: Optional[float] = None

    # --- Memory-level fields (Agent 1's OpenViking domain) ---
    viking_uri: Optional[str] = None
    memory_tier: Optional[MemoryTier] = None
    token_count: Optional[int] = None
    namespace: Optional[str] = None
    previous_tier: Optional[MemoryTier] = None

    # --- Harness-level fields (Agent 1's orchestration domain) ---
    phase: Optional[HarnessPhase] = None
    previous_phase: Optional[HarnessPhase] = None
    sprint_number: Optional[int] = None
    agent_role: Optional[AgentRole] = None
    evaluation_score: Optional[float] = None
    evaluation_criterion: Optional[str] = None
    contract_status: Optional[str] = None

    # --- Derived / override ---
    symbol: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_symbol(self) -> str:
        """
        Produce a string symbol for information-theoretic computation.

        The SymbolStream doesn't care what these mean -- it just computes
        entropy over their distribution. The symbol encoding determines
        what the IT measures actually capture:

          tool:Read         -> tool usage diversity
          mem:l2:agent/gen  -> memory access patterns
          phase:executing   -> phase transition patterns
        """
        if self.symbol:
            return self.symbol

        if self.event_type == EventType.TOOL_CALL:
            return f"tool:{self.tool_name or 'unknown'}"

        if self.event_type == EventType.FILE_READ:
            return f"read:{_truncate_path(self.target_path)}"

        if self.event_type == EventType.FILE_WRITE:
            return f"write:{_truncate_path(self.target_path)}"

        if self.event_type == EventType.SHELL_COMMAND:
            return f"shell:{self.tool_name or 'cmd'}"

        if self.event_type == EventType.MEMORY_LOAD:
            tier = self.memory_tier.value if self.memory_tier else "?"
            return f"mem_load:{tier}:{self.namespace or 'unknown'}"

        if self.event_type == EventType.MEMORY_STORE:
            return f"mem_store:{self.namespace or 'unknown'}"

        if self.event_type == EventType.MEMORY_EVICT:
            return f"mem_evict:{self.namespace or 'unknown'}"

        if self.event_type == EventType.MEMORY_TIER_ESCALATION:
            prev = self.previous_tier.value if self.previous_tier else "?"
            curr = self.memory_tier.value if self.memory_tier else "?"
            return f"mem_esc:{prev}>{curr}"

        if self.event_type == EventType.PHASE_BOUNDARY:
            return f"phase:{self.phase.value if self.phase else 'unknown'}"

        if self.event_type == EventType.AGENT_HANDOFF:
            return f"handoff:{self.agent_role.value if self.agent_role else 'unknown'}"

        if self.event_type == EventType.EVALUATION_RESULT:
            return f"eval:{self.evaluation_criterion or 'general'}"

        if self.event_type == EventType.CONTRACT_EVENT:
            return f"contract:{self.contract_status or 'unknown'}"

        return f"{self.event_type.value}"

    def is_phase_marker(self) -> bool:
        """Agent 2: render as vertical line on sparklines, not a data point."""
        return self.event_type in (
            EventType.PHASE_BOUNDARY,
            EventType.SESSION_START,
            EventType.SESSION_END,
        )

    def is_memory_operation(self) -> bool:
        """Agent 2: render on memory operations track."""
        return self.event_type in (
            EventType.MEMORY_LOAD,
            EventType.MEMORY_STORE,
            EventType.MEMORY_EVICT,
            EventType.MEMORY_TIER_ESCALATION,
        )

    def is_evaluation_signal(self) -> bool:
        """Agent 2: use for retrospective calibration, not real-time display."""
        return self.event_type == EventType.EVALUATION_RESULT

    def to_dict(self) -> dict:
        """JSON-serializable dict for WebSocket transport to frontend."""
        d = {}
        for k, v in asdict(self).items():
            if v is None:
                continue
            if isinstance(v, Enum):
                d[k] = v.value
            else:
                d[k] = v
        d["symbol"] = self.to_symbol()
        return d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _truncate_path(path: Optional[str], max_depth: int = 2) -> str:
    """Truncate file paths to last N components for readable symbols."""
    if not path:
        return "unknown"
    parts = path.replace("\\", "/").rstrip("/").split("/")
    return "/".join(parts[-max_depth:])


def _parse_namespace(uri: str) -> str:
    """Extract namespace from viking:// URI.

    viking://agent/generator/skills/design_patterns -> agent/generator/skills
    viking://resources/current_project/spec -> resources/current_project
    """
    path = uri.replace("viking://", "").strip("/")
    parts = path.split("/")
    return "/".join(parts[:min(3, len(parts))])


# ---------------------------------------------------------------------------
# Convenience constructors (Agent 1 uses these)
# ---------------------------------------------------------------------------

def memory_load_event(
    step: int,
    timestamp: float,
    uri: str,
    tier: MemoryTier,
    token_count: int,
    namespace: Optional[str] = None,
) -> ObservableEvent:
    """Create a MEMORY_LOAD event from an OpenViking access."""
    return ObservableEvent(
        step=step,
        timestamp=timestamp,
        event_type=EventType.MEMORY_LOAD,
        viking_uri=uri,
        memory_tier=tier,
        token_count=token_count,
        namespace=namespace or _parse_namespace(uri),
    )


def memory_store_event(
    step: int,
    timestamp: float,
    uri: str,
    token_count: int,
    namespace: Optional[str] = None,
) -> ObservableEvent:
    """Create a MEMORY_STORE event (e.g., skill crystallization)."""
    return ObservableEvent(
        step=step,
        timestamp=timestamp,
        event_type=EventType.MEMORY_STORE,
        viking_uri=uri,
        token_count=token_count,
        namespace=namespace or _parse_namespace(uri),
    )


def tier_escalation_event(
    step: int,
    timestamp: float,
    uri: str,
    from_tier: MemoryTier,
    to_tier: MemoryTier,
    token_count: int,
) -> ObservableEvent:
    """Create a MEMORY_TIER_ESCALATION event (L0->L1 or L1->L2)."""
    return ObservableEvent(
        step=step,
        timestamp=timestamp,
        event_type=EventType.MEMORY_TIER_ESCALATION,
        viking_uri=uri,
        memory_tier=to_tier,
        previous_tier=from_tier,
        token_count=token_count,
        namespace=_parse_namespace(uri),
    )


def phase_boundary_event(
    step: int,
    timestamp: float,
    phase: HarnessPhase,
    previous_phase: Optional[HarnessPhase] = None,
    sprint_number: Optional[int] = None,
    agent_role: Optional[AgentRole] = None,
) -> ObservableEvent:
    """Create a PHASE_BOUNDARY event from the harness orchestrator."""
    return ObservableEvent(
        step=step,
        timestamp=timestamp,
        event_type=EventType.PHASE_BOUNDARY,
        phase=phase,
        previous_phase=previous_phase,
        sprint_number=sprint_number,
        agent_role=agent_role,
    )


def evaluation_event(
    step: int,
    timestamp: float,
    score: float,
    criterion: str,
    sprint_number: Optional[int] = None,
) -> ObservableEvent:
    """Create an EVALUATION_RESULT event from the evaluator agent."""
    return ObservableEvent(
        step=step,
        timestamp=timestamp,
        event_type=EventType.EVALUATION_RESULT,
        evaluation_score=score,
        evaluation_criterion=criterion,
        sprint_number=sprint_number,
        agent_role=AgentRole.EVALUATOR,
    )


# Convenience constructors (Agent 2 uses these -- wrapping existing trace data)

def tool_call_event(
    step: int,
    timestamp: float,
    tool_name: str,
    target_path: Optional[str] = None,
    duration_ms: Optional[float] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
) -> ObservableEvent:
    """Create a TOOL_CALL event from a Claude Code trace entry."""
    return ObservableEvent(
        step=step,
        timestamp=timestamp,
        event_type=EventType.TOOL_CALL,
        tool_name=tool_name,
        target_path=target_path,
        duration_ms=duration_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def file_read_event(
    step: int,
    timestamp: float,
    path: str,
    output_tokens: Optional[int] = None,
) -> ObservableEvent:
    """Create a FILE_READ event."""
    return ObservableEvent(
        step=step,
        timestamp=timestamp,
        event_type=EventType.FILE_READ,
        target_path=path,
        output_tokens=output_tokens,
    )


def file_write_event(
    step: int,
    timestamp: float,
    path: str,
    input_tokens: Optional[int] = None,
) -> ObservableEvent:
    """Create a FILE_WRITE event."""
    return ObservableEvent(
        step=step,
        timestamp=timestamp,
        event_type=EventType.FILE_WRITE,
        target_path=path,
        input_tokens=input_tokens,
    )
