"""Claude adapter — parses Anthropic Messages API trace format."""

from __future__ import annotations

import hashlib
from typing import Any

from agentdiag.models import TraceEvent
from agentdiag.adapters.base import AdapterMeta


class ClaudeAdapter:
    """Parses traces from the Anthropic Messages API format.

    Expects a list of message objects with content blocks, including
    tool_use and tool_result blocks.
    """

    meta = AdapterMeta(
        name="claude",
        version="1.0.0",
        supported_formats=["json"],
    )

    def can_parse(self, data: dict | list | str) -> bool:
        if isinstance(data, list) and len(data) > 0:
            first = data[0]
            if isinstance(first, dict):
                # Claude messages have "role" and "content" (list of blocks)
                if "role" in first and "content" in first:
                    content = first["content"]
                    if isinstance(content, list) and len(content) > 0:
                        block = content[0]
                        return isinstance(block, dict) and "type" in block
        return False

    def parse(self, data: dict | list | str) -> list[TraceEvent]:
        if not isinstance(data, list):
            raise ValueError("ClaudeAdapter expects a list of message objects.")

        events = []
        step = 1

        for msg in data:
            role = msg.get("role", "")
            content = msg.get("content", [])
            if isinstance(content, str):
                # Simple text content
                events.append(TraceEvent(
                    step=step,
                    type="reasoning",
                    tokens_out=len(content.split()),
                ))
                step += 1
                continue

            for block in content:
                block_type = block.get("type", "")

                if block_type == "tool_use":
                    tool_name = block.get("name", "unknown_tool")
                    input_data = block.get("input", {})
                    input_str = str(input_data)
                    events.append(TraceEvent(
                        step=step,
                        type="tool_call",
                        tool=tool_name,
                        tokens_in=len(input_str.split()),
                        output_hash=hashlib.md5(input_str.encode()).hexdigest()[:8],
                    ))
                    step += 1

                elif block_type == "tool_result":
                    content_text = block.get("content", "")
                    if isinstance(content_text, list):
                        content_text = " ".join(
                            b.get("text", "") for b in content_text if isinstance(b, dict)
                        )
                    is_error = block.get("is_error", False)
                    events.append(TraceEvent(
                        step=step,
                        type="tool_call",
                        tool=block.get("tool_use_id", "tool_result"),
                        success=not is_error,
                        tokens_out=len(str(content_text).split()),
                        error_message=str(content_text) if is_error else None,
                    ))
                    step += 1

                elif block_type == "text":
                    text = block.get("text", "")
                    if role == "assistant":
                        events.append(TraceEvent(
                            step=step,
                            type="reasoning",
                            tokens_out=len(text.split()),
                        ))
                        step += 1

                elif block_type == "thinking":
                    text = block.get("thinking", "")
                    events.append(TraceEvent(
                        step=step,
                        type="planning",
                        tokens_out=len(text.split()),
                    ))
                    step += 1

        return events
