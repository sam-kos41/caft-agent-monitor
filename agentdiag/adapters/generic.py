"""Generic adapter — handles already-formatted TraceEvent dicts."""

from __future__ import annotations

from agentdiag.models import TraceEvent
from agentdiag.adapters.base import AdapterMeta


class GenericAdapter:
    """Parses traces that are already in TraceEvent dict format."""

    meta = AdapterMeta(
        name="generic",
        version="1.0.0",
        supported_formats=["json", "jsonl"],
    )

    def can_parse(self, data: dict | list | str) -> bool:
        if isinstance(data, list) and len(data) > 0:
            first = data[0]
            if isinstance(first, dict):
                return "step" in first and "type" in first
        return False

    def parse(self, data: dict | list | str) -> list[TraceEvent]:
        if not isinstance(data, list):
            raise ValueError("GenericAdapter expects a list of event dicts.")
        return [TraceEvent.from_dict(d) for d in data]
