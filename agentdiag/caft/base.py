"""CAFT detector base classes and diagnosis model.

All detectors implement the CaftDetector protocol:
    name: str
    caft_code: str
    check(events, hta_state) -> Optional[CaftDiagnosis]
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Protocol, Optional, runtime_checkable

from agentdiag.models import TraceEvent
from agentdiag.hta import HTAState


class CaftSeverity(Enum):
    """Severity levels for CAFT diagnoses."""
    INFO = "info"          # noteworthy but not problematic
    WARNING = "warning"    # potential issue, monitor closely
    CRITICAL = "critical"  # active failure, intervention recommended


@dataclass
class CaftDiagnosis:
    """A CAFT failure diagnosis with taxonomy codes and evidence."""
    caft_code: str           # e.g., "2.2" (Memory: Repetition)
    caft_category: str       # e.g., "memory"
    failure_name: str        # e.g., "step_repetition"
    severity: CaftSeverity
    confidence: float        # 0.0 to 1.0
    description: str         # human-readable explanation
    evidence: dict           # structured evidence
    at_step: int             # step where detected
    remediation: str         # short actionable suggestion
    force_llm_review: bool = False  # bypass auto-confirm, always send to LLM

    def to_dict(self) -> dict:
        d = asdict(self)
        d["severity"] = self.severity.value
        return d


@runtime_checkable
class CaftDetector(Protocol):
    """Protocol for CAFT detectors.

    All detectors implement this protocol. They receive the full event list
    and current HTA state, and return a CaftDiagnosis if a failure is detected.
    """
    name: str
    caft_code: str

    def check(
        self,
        events: list[TraceEvent],
        hta_state: HTAState,
    ) -> Optional[CaftDiagnosis]:
        """Check for a CAFT failure. Returns diagnosis or None."""
        ...


# Alias for backward compatibility
Detector = CaftDetector
