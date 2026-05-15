"""Inter-rater validation harness for CAFT.

Compares ratings from three sources on the same Claude Code sessions:
  - Human (gold standard) via web UI
  - Local LLM (Ollama) via HTTP API
  - CAFT itself (rule-based mapping from existing metrics)

Computes pairwise Cohen's kappa, Krippendorff's alpha, and Spearman
correlation to assess construct validity. Surfaces top disagreements
for manual review so the constructs themselves can be refined.

Public API:

    from agentdiag.validation import (
        build_digest,           # session jsonl path -> SessionDigest
        Ledger,                 # JSONL-backed rating store
        rate_with_ollama,       # call local Ollama
        rate_with_caft,         # rule-based mapping from CAFT metrics
        compute_agreement,      # pairwise kappa, alpha, correlations
        write_report,           # markdown report from a ledger
    )
"""

from agentdiag.validation.digest import (
    SessionDigest,
    build_digest,
    DIMENSIONS,
    LIKERT_DIMS,
    CATEGORICAL_DIMS,
    HEALTH_LABELS,
    SCALE_ANCHORS,
    SCALE_NOTES,
    HEALTH_ANCHORS,
    DIM_POLARITY,
)
from agentdiag.validation.ledger import Ledger, Rating
from agentdiag.validation.rate_ollama import rate_with_ollama, OllamaError
from agentdiag.validation.rate_caft import rate_with_caft
from agentdiag.validation.signals import (
    extract_signals, rate_with_signals, SessionSignals,
)
from agentdiag.validation.agreement import (
    cohens_kappa,
    krippendorff_alpha,
    compute_agreement,
)
from agentdiag.validation.report import write_report

__all__ = [
    "SessionDigest",
    "build_digest",
    "DIMENSIONS",
    "LIKERT_DIMS",
    "CATEGORICAL_DIMS",
    "HEALTH_LABELS",
    "SCALE_ANCHORS",
    "SCALE_NOTES",
    "HEALTH_ANCHORS",
    "DIM_POLARITY",
    "Ledger",
    "Rating",
    "rate_with_ollama",
    "OllamaError",
    "rate_with_caft",
    "extract_signals",
    "rate_with_signals",
    "SessionSignals",
    "cohens_kappa",
    "krippendorff_alpha",
    "compute_agreement",
    "write_report",
]
