"""Typed evidence dataclasses for each failure mode.

Each detector can optionally return typed evidence instead of raw dicts.
All evidence types support .to_dict() for backward-compatible serialization.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class EvidenceBase:
    """Base class for all typed evidence."""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LoopEvidence(EvidenceBase):
    repeated_pattern: list[str]
    pattern_count: int
    pattern_length: int
    total_tool_calls: int
    state_changes_during_loop: int


@dataclass
class ThrashEvidence(EvidenceBase):
    worst_window_tools: list[str]
    unique_tools_in_window: list[str]
    switch_rate: float
    window_start_step: int


@dataclass
class StallEvidence(EvidenceBase):
    stall_steps: list[int]
    max_latency_ms: float
    median_latency_ms: float
    threshold_ms: float
    stall_count: int
    worst_step: int
    stall_fraction: float = 0.0
    active_events: int = 0
    idle_events_excluded: int = 0


@dataclass
class DriftEvidence(EvidenceBase):
    tool_distribution_shift: float
    error_rate_first_half: float
    error_rate_second_half: float
    latency_ratio_2nd_vs_1st: float
    new_tools_in_second_half: list[str]
    dropped_tools_in_second_half: list[str]


@dataclass
class CascadeEvidence(EvidenceBase):
    longest_error_chain: int
    chain_start_step: int
    chain_end_step: int
    total_error_chains: int
    total_errors: int
    tools_in_chain: list[str | None]


@dataclass
class TokenExplosionEvidence(EvidenceBase):
    growth_ratio_last_vs_first_quarter: float
    tokens_per_step_slope: float
    acceleration: float
    first_quarter_avg_tokens: float
    last_quarter_avg_tokens: float
    total_tokens: int


@dataclass
class DeadEndEvidence(EvidenceBase):
    max_consecutive_reasoning: int
    dead_end_start_step: int
    dead_end_end_step: int
    total_reasoning_events: int
    total_events: int


@dataclass
class RecoveryFailureEvidence(EvidenceBase):
    total_errors: int
    failed_recoveries: int
    recovery_failure_rate: float
    same_tool_retries: int
    worst_error_step: int | None
    worst_consecutive_failures_after: int


# Mapping from failure_type to evidence class for deserialization
EVIDENCE_TYPES: dict[str, type[EvidenceBase]] = {
    "LOOP": LoopEvidence,
    "TOOL_THRASH": ThrashEvidence,
    "STALL": StallEvidence,
    "DRIFT": DriftEvidence,
    "CASCADE": CascadeEvidence,
    "TOKEN_EXPLOSION": TokenExplosionEvidence,
    "DEAD_END": DeadEndEvidence,
    "RECOVERY_FAILURE": RecoveryFailureEvidence,
}


def parse_evidence(failure_type: str, data: dict) -> EvidenceBase:
    """Parse a raw evidence dict into a typed evidence object."""
    cls = EVIDENCE_TYPES.get(failure_type)
    if cls is None:
        raise ValueError(f"Unknown failure type: {failure_type}")
    return cls(**data)
