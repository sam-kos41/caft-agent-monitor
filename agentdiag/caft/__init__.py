"""CAFT (Cognitive Agent Failure Taxonomy) detector pipeline.

All detectors produce CaftDiagnosis objects via check(events, hta_state).

V4: Added semantic confirmation layer (caft.confirm) for LLM-based
disambiguation of detector candidates.
"""

from agentdiag.caft.base import (
    CaftDetector,
    CaftDiagnosis,
    CaftSeverity,
    Detector,
)
from agentdiag.caft.detectors import (
    ALL_CAFT_DETECTORS,
    ALL_CAFT_DETECTORS_FULL,
    run_caft_detectors,
)
from agentdiag.caft.registry import detector_registry, DetectorRegistry

__all__ = [
    "CaftDetector",
    "CaftDiagnosis",
    "CaftSeverity",
    "Detector",
    "DetectorRegistry",
    "detector_registry",
    "ALL_CAFT_DETECTORS",
    "ALL_CAFT_DETECTORS_FULL",
    "run_caft_detectors",
]
