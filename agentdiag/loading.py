"""Trace loading utilities."""

from __future__ import annotations

import json
from pathlib import Path

from agentdiag.models import TraceEvent


def load_trace(path: str | Path) -> list[TraceEvent]:
    """Load a trace from a JSON file (array of events or newline-delimited).

    Supports auto-detection of trace format via the adapter system.
    Falls back to newline-delimited JSON if JSON array parsing fails.
    """
    path = Path(path)
    text = path.read_text()

    # Try JSON array first
    try:
        data = json.loads(text)
        if isinstance(data, list):
            # Try auto_parse for multi-framework support
            from agentdiag.adapters import auto_parse
            try:
                return auto_parse(data)
            except ValueError:
                pass
            # Fallback: treat as raw TraceEvent dicts
            return [TraceEvent.from_dict(d) for d in data]
    except json.JSONDecodeError:
        pass

    # Try newline-delimited JSON
    events = []
    for line in text.strip().splitlines():
        line = line.strip()
        if line:
            events.append(TraceEvent.from_dict(json.loads(line)))
    return events
