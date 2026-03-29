"""Adapter protocol for multi-framework trace ingestion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from agentdiag.models import TraceEvent


@dataclass
class AdapterMeta:
    """Metadata for a trace adapter."""
    name: str               # "claude", "openai", "langchain"
    version: str
    supported_formats: list[str] = field(default_factory=list)


@runtime_checkable
class TraceAdapter(Protocol):
    """Protocol that all trace adapters must satisfy."""
    meta: AdapterMeta

    def can_parse(self, data: dict | list | str) -> bool:
        """Auto-detect if this adapter handles this trace format."""
        ...

    def parse(self, data: dict | list | str) -> list[TraceEvent]:
        """Convert framework-specific format to TraceEvent list."""
        ...
