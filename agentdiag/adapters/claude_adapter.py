"""Claude Code adapter — maps JSONL trace entries to ObservableEvent.

This is Agent 2's adapter.  It converts Claude Code session JSONL
(the format produced by ~/.claude/projects/*/sessions/*.jsonl) into
ObservableEvent instances using the shared contract's convenience
constructors.

For backward compatibility, also supports converting the older
TraceEvent dataclass (from agentdiag.models) into ObservableEvent.

Usage::

    adapter = ClaudeCodeAdapter()
    for line in open("session.jsonl"):
        data = json.loads(line)
        events = adapter.parse(data)
        for event in events:
            monitor.process(event)  # event is an ObservableEvent
"""

from __future__ import annotations

import json
from typing import Any, Optional

from agentdiag.observable import (
    ObservableEvent,
    EventType,
    tool_call_event,
    file_read_event,
    file_write_event,
)


# Tools that indicate file reading
_READ_TOOLS = {
    "read_file", "read", "cat", "head", "tail",
    "search_docs", "web_search", "search_codebase",
    "list_files", "glob", "grep", "find", "ls",
    "fetch", "get", "list", "describe",
}

# Tools that indicate file writing
_WRITE_TOOLS = {
    "write_file", "write", "edit", "edit_file",
}

# Tools that indicate shell execution
_SHELL_TOOLS = {
    "bash", "shell", "run_code", "terminal",
}

# Event types that are reasoning/planning (not tool calls)
_REASONING_TYPES = {"reasoning", "planning", "thinking"}


class ClaudeCodeAdapter:
    """Converts Claude Code JSONL trace data to ObservableEvent instances.

    Handles two input formats:
    1. Raw Claude Code session JSONL (dict with 'type', 'tool', etc.)
    2. TraceEvent dataclass instances (from agentdiag.models)
    """

    def __init__(self) -> None:
        self._step_counter = 0

    def parse(self, data: Any) -> list[ObservableEvent]:
        """Convert a single trace entry to ObservableEvent(s).

        Args:
            data: A dict from JSONL, or a TraceEvent instance.

        Returns:
            List of ObservableEvent (usually 1, sometimes 0 for skipped entries).
        """
        # Handle TraceEvent instances
        from agentdiag.models import TraceEvent
        if isinstance(data, TraceEvent):
            return [self._from_trace_event(data)]

        # Handle raw dicts from JSONL
        if isinstance(data, dict):
            return self._from_dict(data)

        return []

    def reset(self) -> None:
        """Reset step counter for a new session."""
        self._step_counter = 0

    def _from_trace_event(self, event: Any) -> ObservableEvent:
        """Convert a TraceEvent to ObservableEvent."""
        tool = (event.tool or event.type or "unknown").lower()

        if event.type in _REASONING_TYPES:
            return ObservableEvent(
                step=event.step,
                timestamp=event.timestamp or 0.0,
                event_type=EventType.TOOL_CALL,
                tool_name=event.type,
                duration_ms=event.latency_ms,
                input_tokens=event.tokens_in,
                output_tokens=event.tokens_out,
            )

        if tool in _READ_TOOLS:
            return ObservableEvent(
                step=event.step,
                timestamp=event.timestamp or 0.0,
                event_type=EventType.FILE_READ,
                tool_name=event.tool,
                target_path=None,
                output_tokens=event.tokens_out,
                duration_ms=event.latency_ms,
            )

        if tool in _WRITE_TOOLS:
            return ObservableEvent(
                step=event.step,
                timestamp=event.timestamp or 0.0,
                event_type=EventType.FILE_WRITE,
                tool_name=event.tool,
                target_path=None,
                input_tokens=event.tokens_in,
                duration_ms=event.latency_ms,
            )

        if tool in _SHELL_TOOLS:
            return ObservableEvent(
                step=event.step,
                timestamp=event.timestamp or 0.0,
                event_type=EventType.SHELL_COMMAND,
                tool_name=event.tool,
                duration_ms=event.latency_ms,
                input_tokens=event.tokens_in,
                output_tokens=event.tokens_out,
            )

        return ObservableEvent(
            step=event.step,
            timestamp=event.timestamp or 0.0,
            event_type=EventType.TOOL_CALL,
            tool_name=event.tool or event.type,
            duration_ms=event.latency_ms,
            input_tokens=event.tokens_in,
            output_tokens=event.tokens_out,
        )

    def _from_dict(self, data: dict) -> list[ObservableEvent]:
        """Convert a raw JSONL dict to ObservableEvent."""
        step = data.get("step", self._step_counter)
        self._step_counter = max(self._step_counter, step + 1)
        timestamp = data.get("timestamp", 0.0)

        event_type = data.get("type", "")
        tool = (data.get("tool") or event_type or "unknown").lower()

        # Skip empty/non-event entries
        if not event_type and not tool:
            return []

        if event_type in _REASONING_TYPES:
            return [ObservableEvent(
                step=step,
                timestamp=timestamp,
                event_type=EventType.TOOL_CALL,
                tool_name=event_type,
                duration_ms=data.get("latency_ms"),
                input_tokens=data.get("tokens_in"),
                output_tokens=data.get("tokens_out"),
            )]

        if tool in _READ_TOOLS:
            return [ObservableEvent(
                step=step,
                timestamp=timestamp,
                event_type=EventType.FILE_READ,
                tool_name=data.get("tool"),
                target_path=data.get("target_path") or data.get("path"),
                output_tokens=data.get("tokens_out"),
                duration_ms=data.get("latency_ms"),
            )]

        if tool in _WRITE_TOOLS:
            return [ObservableEvent(
                step=step,
                timestamp=timestamp,
                event_type=EventType.FILE_WRITE,
                tool_name=data.get("tool"),
                target_path=data.get("target_path") or data.get("path"),
                input_tokens=data.get("tokens_in"),
                duration_ms=data.get("latency_ms"),
            )]

        if tool in _SHELL_TOOLS:
            return [ObservableEvent(
                step=step,
                timestamp=timestamp,
                event_type=EventType.SHELL_COMMAND,
                tool_name=data.get("tool"),
                duration_ms=data.get("latency_ms"),
                input_tokens=data.get("tokens_in"),
                output_tokens=data.get("tokens_out"),
            )]

        return [ObservableEvent(
            step=step,
            timestamp=timestamp,
            event_type=EventType.TOOL_CALL,
            tool_name=data.get("tool") or event_type,
            duration_ms=data.get("latency_ms"),
            input_tokens=data.get("tokens_in"),
            output_tokens=data.get("tokens_out"),
        )]
