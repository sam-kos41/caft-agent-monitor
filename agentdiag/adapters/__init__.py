"""Trace adapter system for multi-framework trace ingestion.

Two adapter layers:

1. **TraceAdapter** (existing) — converts framework-specific formats to TraceEvent.
   Used by the legacy MonitorEngine and evaluation pipeline.

2. **AgentAdapter** (new) — converts data sources to ObservableEvent streams.
   Used by the UniversalMonitor and visualization layer.

The ``get_adapter()`` factory returns AgentAdapter instances by name.
The ``MixedAdapter`` merges multiple event streams by timestamp — this is
the default for harness runs where events come from Claude Code traces,
OpenViking access logs, and harness phase transitions simultaneously.
"""

from __future__ import annotations

from typing import Optional

from agentdiag.models import TraceEvent
from agentdiag.adapters.base import TraceAdapter, AdapterMeta
from agentdiag.adapters.generic import GenericAdapter
from agentdiag.adapters.claude import ClaudeAdapter
from agentdiag.adapters.openai import OpenAIAdapter
from agentdiag.adapters.langchain import LangChainAdapter

# Ordered by specificity: most specific first, generic last
_ADAPTERS: list = [
    ClaudeAdapter(),
    LangChainAdapter(),
    OpenAIAdapter(),
    GenericAdapter(),
]


def auto_parse(data: dict | list | str) -> list[TraceEvent]:
    """Auto-detect the trace format and parse it into TraceEvents."""
    for adapter in _ADAPTERS:
        if adapter.can_parse(data):
            return adapter.parse(data)
    raise ValueError(
        "No adapter matched this trace format. "
        "Ensure data is a list of TraceEvent dicts, Claude messages, "
        "OpenAI messages, or LangChain runs."
    )


# ---------------------------------------------------------------------------
# AgentAdapter layer (ObservableEvent-based)
# ---------------------------------------------------------------------------

from agentdiag.observable import ObservableEvent  # noqa: E402


class AgentAdapter:
    """Base class for adapters that produce ObservableEvent streams."""

    name: str = "base"

    def replay(self, source: str, **kwargs) -> list[ObservableEvent]:
        """Replay events from a source path or identifier."""
        raise NotImplementedError


class MixedAdapter(AgentAdapter):
    """Merges multiple ObservableEvent streams by timestamp.

    The default adapter for harness runs where events come from
    multiple sources simultaneously.

    Usage::

        mixed = MixedAdapter()
        events = mixed.merge(
            viking_events,
            harness_events,
            tool_events,
        )
    """

    name: str = "mixed"

    def merge(self, *streams: list[ObservableEvent]) -> list[ObservableEvent]:
        """Merge multiple event streams into one chronological stream.

        Re-numbers steps sequentially after sorting by timestamp.
        """
        all_events: list[ObservableEvent] = []
        for stream in streams:
            all_events.extend(stream)

        all_events.sort(key=lambda e: (e.timestamp, e.step))

        for i, event in enumerate(all_events, start=1):
            event.step = i

        return all_events

    def replay(self, source: str, **kwargs) -> list[ObservableEvent]:
        """Not meaningful for mixed adapter — use merge() directly."""
        return []


def get_adapter(source: str) -> AgentAdapter:
    """Factory: get an AgentAdapter by source name.

    Registered adapters:
      - "viking" — replays OpenViking access logs
      - "harness" — replays serialized HarnessResult
      - "mixed" — merges multiple event sources by timestamp
      - "claude_code" — converts Claude Code JSONL to ObservableEvent

    Args:
        source: Adapter name.

    Returns:
        An AgentAdapter instance.

    Raises:
        ValueError: If the source name is not recognized.
    """
    source = source.lower().replace("-", "_")

    if source == "viking":
        from agentdiag.adapters.viking_adapter import VikingLogAdapter
        return VikingLogAdapter()
    elif source == "harness":
        from agentdiag.adapters.harness_adapter import HarnessLogAdapter
        return HarnessLogAdapter()
    elif source == "mixed":
        return MixedAdapter()
    elif source == "claude_code":
        from agentdiag.adapters.claude_adapter import ClaudeCodeAdapter
        return ClaudeCodeAdapter()
    else:
        raise ValueError(
            f"Unknown adapter '{source}'. "
            f"Available: viking, harness, mixed, claude_code"
        )


__all__ = [
    "TraceAdapter",
    "AdapterMeta",
    "GenericAdapter",
    "ClaudeAdapter",
    "OpenAIAdapter",
    "LangChainAdapter",
    "auto_parse",
    "AgentAdapter",
    "MixedAdapter",
    "get_adapter",
]
