"""Data models for agent trace diagnostics."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class TraceEvent:
    """A single event in an agent execution trace."""
    step: int
    type: str                         # tool_call, reasoning, planning, output, user_input
    tool: Optional[str] = None
    latency_ms: float = 0.0
    success: bool = True
    tokens_in: int = 0
    tokens_out: int = 0
    timestamp: Optional[float] = None # seconds from trace start
    output_hash: Optional[str] = None # for state change detection
    goal_text: Optional[str] = None   # for drift detection
    error_message: Optional[str] = None
    input_hash: Optional[str] = None  # hash of tool input (same input = same operation)
    agent_id: Optional[str] = None    # sub-agent identifier (None = main agent "M")
    target_path: Optional[str] = None # file path or resource acted upon

    @classmethod
    def from_dict(cls, d: dict) -> "TraceEvent":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class TraceFeatures:
    """Aggregate features computed from a full trace."""
    event_rate: float          # events per minute
    tool_diversity: float      # unique tools / total tool calls
    repetition_score: float    # consecutive identical tool calls / total
    latency_trend: float       # slope of latency over steps (ms/step)
    token_growth_rate: float   # slope of cumulative tokens over steps
    error_rate: float          # failures / total
    state_change_rate: float   # proportion of steps that changed output
    plan_depth: float          # max consecutive reasoning steps before action
    total_events: int
    total_tokens: int
    duration_sec: float


@dataclass
class Diagnosis:
    """A single diagnosed failure mode with structured evidence."""
    failure_type: str
    confidence: float
    evidence: dict
    explanation: str

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


@dataclass
class DiagnosticReport:
    """Complete diagnostic report for an agent trace."""
    trace_id: str
    features: TraceFeatures
    diagnoses: list[Diagnosis]
    summary: str
    overall_health: str        # healthy, degraded, failing

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "features": asdict(self.features),
            "diagnoses": [d.to_dict() for d in self.diagnoses],
            "summary": self.summary,
            "overall_health": self.overall_health,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)
