"""LangChain adapter — parses LangSmith / callback trace format."""

from __future__ import annotations

import hashlib

from agentdiag.models import TraceEvent
from agentdiag.adapters.base import AdapterMeta


class LangChainAdapter:
    """Parses traces from LangSmith or LangChain callback format.

    Expects a list of run objects with "run_type", "name", "inputs",
    "outputs", "error", etc.
    """

    meta = AdapterMeta(
        name="langchain",
        version="1.0.0",
        supported_formats=["json"],
    )

    def can_parse(self, data: dict | list | str) -> bool:
        if isinstance(data, list) and len(data) > 0:
            first = data[0]
            if isinstance(first, dict):
                return "run_type" in first and "name" in first
        return False

    def parse(self, data: dict | list | str) -> list[TraceEvent]:
        if not isinstance(data, list):
            raise ValueError("LangChainAdapter expects a list of run objects.")

        events = []
        step = 1

        for run in data:
            run_type = run.get("run_type", "")
            name = run.get("name", "unknown")
            inputs = run.get("inputs", {})
            outputs = run.get("outputs", {})
            error = run.get("error")
            latency_ms = run.get("latency_ms", 0.0)

            # Compute tokens from inputs/outputs if available
            input_str = str(inputs)
            output_str = str(outputs)
            tokens_in = run.get("total_tokens", len(input_str.split()))
            tokens_out = len(output_str.split())

            if run_type == "tool":
                events.append(TraceEvent(
                    step=step,
                    type="tool_call",
                    tool=name,
                    latency_ms=latency_ms,
                    success=error is None,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    output_hash=hashlib.md5(output_str.encode()).hexdigest()[:8],
                    error_message=error,
                ))
            elif run_type == "llm":
                events.append(TraceEvent(
                    step=step,
                    type="reasoning",
                    latency_ms=latency_ms,
                    success=error is None,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    error_message=error,
                ))
            elif run_type == "chain":
                events.append(TraceEvent(
                    step=step,
                    type="planning",
                    tool=name,
                    latency_ms=latency_ms,
                    success=error is None,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    error_message=error,
                ))
            else:
                events.append(TraceEvent(
                    step=step,
                    type="tool_call",
                    tool=name,
                    latency_ms=latency_ms,
                    success=error is None,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    error_message=error,
                ))

            step += 1

        return events
