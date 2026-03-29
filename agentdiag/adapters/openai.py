"""OpenAI adapter — parses OpenAI Chat Completions API trace format."""

from __future__ import annotations

import hashlib

from agentdiag.models import TraceEvent
from agentdiag.adapters.base import AdapterMeta


class OpenAIAdapter:
    """Parses traces from the OpenAI Chat Completions API format.

    Expects a list of message objects with "role", "content", and
    optionally "tool_calls" arrays.
    """

    meta = AdapterMeta(
        name="openai",
        version="1.0.0",
        supported_formats=["json"],
    )

    def can_parse(self, data: dict | list | str) -> bool:
        if isinstance(data, list) and len(data) > 0:
            first = data[0]
            if isinstance(first, dict) and "role" in first:
                # OpenAI format: has "role" + either "tool_calls" or
                # content is a string (not a list of blocks like Claude)
                content = first.get("content")
                has_tool_calls = any(
                    "tool_calls" in msg for msg in data if isinstance(msg, dict)
                )
                content_is_string = content is None or isinstance(content, str)
                return content_is_string and (has_tool_calls or first.get("role") in ("system", "user", "assistant", "tool"))
        return False

    def parse(self, data: dict | list | str) -> list[TraceEvent]:
        if not isinstance(data, list):
            raise ValueError("OpenAIAdapter expects a list of message objects.")

        events = []
        step = 1

        for msg in data:
            role = msg.get("role", "")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls", [])

            # Assistant reasoning (text content without tool calls)
            if role == "assistant" and content and not tool_calls:
                events.append(TraceEvent(
                    step=step,
                    type="reasoning",
                    tokens_out=len(str(content).split()),
                ))
                step += 1

            # Tool calls from assistant
            if tool_calls:
                for tc in tool_calls:
                    func = tc.get("function", {})
                    tool_name = func.get("name", "unknown_tool")
                    arguments = func.get("arguments", "")
                    events.append(TraceEvent(
                        step=step,
                        type="tool_call",
                        tool=tool_name,
                        tokens_in=len(str(arguments).split()),
                        output_hash=hashlib.md5(str(arguments).encode()).hexdigest()[:8],
                    ))
                    step += 1

            # Tool responses
            if role == "tool":
                events.append(TraceEvent(
                    step=step,
                    type="tool_call",
                    tool=msg.get("tool_call_id", "tool_response"),
                    tokens_out=len(str(content).split()),
                    success=True,
                ))
                step += 1

        return events
