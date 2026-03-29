"""Gold-standard annotation schema and tools for CAFT trace labeling.

Provides:
  - AnnotationSchema: the structured record each annotator fills out
  - AnnotationStore: persistent JSONL storage for annotations
  - compute_kappa: Cohen's kappa and multi-label agreement metrics
  - CLI annotation tool (run via `agentdiag annotate`)

Annotation Guidelines:
  For each trace, the annotator examines the event sequence and records:
  1. Whether any CAFT failure occurred (binary)
  2. The primary failure type (code + name)
  3. Where it started (step or window)
  4. Severity (1-5), confidence (1-5)
  5. Whether it's directly observable or inferred
  6. Supporting evidence steps
  7. Free-text reasoning
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
from collections import Counter

from agentdiag.caft.taxonomy import (
    CAFT_TAXONOMY,
    CaftType,
    Detectability,
    get_observable_types,
    get_latent_types,
    get_categories,
    get_category_types,
)


@dataclass
class Annotation:
    """A single annotator's label for one trace."""
    # Identity
    trace_id: str
    annotator_id: str
    timestamp: float = field(default_factory=time.time)

    # Primary failure (None if trace is clean)
    has_failure: bool = False
    primary_caft_code: Optional[str] = None       # e.g., "2.2"
    primary_caft_category: Optional[str] = None    # e.g., "memory"
    primary_caft_subtype: Optional[str] = None     # e.g., "step_repetition"

    # Onset localization
    failure_onset_step: Optional[int] = None
    failure_onset_window: Optional[tuple[int, int]] = None  # (start, end)

    # Secondary failures
    secondary_failures: list[str] = field(default_factory=list)  # list of CAFT codes

    # Severity & confidence
    severity: int = 0         # 1-5 (0 = not set / clean trace)
    annotator_confidence: int = 0  # 1-5

    # Observable vs latent
    observable_vs_latent: str = "observable"  # "observable" or "latent"

    # Evidence
    evidence_steps: list[int] = field(default_factory=list)
    free_text_explanation: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        # Convert tuple to list for JSON
        if d["failure_onset_window"]:
            d["failure_onset_window"] = list(d["failure_onset_window"])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Annotation":
        if d.get("failure_onset_window"):
            d["failure_onset_window"] = tuple(d["failure_onset_window"])
        # Filter to known fields
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})

    def validate(self) -> list[str]:
        """Validate annotation completeness. Returns list of issues."""
        issues = []
        if self.has_failure:
            if not self.primary_caft_code:
                issues.append("Missing primary_caft_code")
            elif self.primary_caft_code not in CAFT_TAXONOMY:
                issues.append(f"Unknown CAFT code: {self.primary_caft_code}")
            if self.failure_onset_step is None and self.failure_onset_window is None:
                issues.append("Missing failure onset (step or window)")
            if self.severity < 1 or self.severity > 5:
                issues.append(f"Severity must be 1-5, got {self.severity}")
            if self.annotator_confidence < 1 or self.annotator_confidence > 5:
                issues.append(f"Confidence must be 1-5, got {self.annotator_confidence}")
            if not self.evidence_steps:
                issues.append("No evidence steps provided")
            for code in self.secondary_failures:
                if code not in CAFT_TAXONOMY:
                    issues.append(f"Unknown secondary CAFT code: {code}")
        return issues


class AnnotationStore:
    """Persistent JSONL storage for annotations."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._annotations: list[Annotation] = []
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    self._annotations.append(Annotation.from_dict(json.loads(line)))
                except (json.JSONDecodeError, TypeError):
                    continue

    def add(self, annotation: Annotation) -> None:
        """Add an annotation and persist to disk."""
        issues = annotation.validate()
        if issues:
            raise ValueError(f"Invalid annotation: {'; '.join(issues)}")
        self._annotations.append(annotation)
        with open(self.path, "a") as f:
            f.write(json.dumps(annotation.to_dict()) + "\n")

    def get_all(self) -> list[Annotation]:
        return list(self._annotations)

    def get_for_trace(self, trace_id: str) -> list[Annotation]:
        return [a for a in self._annotations if a.trace_id == trace_id]

    def get_by_annotator(self, annotator_id: str) -> list[Annotation]:
        return [a for a in self._annotations if a.annotator_id == annotator_id]

    @property
    def trace_ids(self) -> set[str]:
        return {a.trace_id for a in self._annotations}

    @property
    def annotator_ids(self) -> set[str]:
        return {a.annotator_id for a in self._annotations}

    def __len__(self) -> int:
        return len(self._annotations)


# ──────────────────────────────────────────────────────────────────
# Inter-Annotator Agreement (Cohen's Kappa + Multi-Label)
# ──────────────────────────────────────────────────────────────────

@dataclass
class AgreementReport:
    """Inter-annotator agreement metrics."""
    annotator_a: str
    annotator_b: str
    n_traces: int

    # Binary agreement: is there a failure at all?
    binary_kappa: float
    binary_agreement: float

    # Category-level agreement (8 categories)
    category_kappa: float
    category_agreement: float

    # Subtype-level agreement (32 types)
    subtype_kappa: float
    subtype_agreement: float

    # Onset localization agreement
    onset_mean_abs_error: Optional[float]  # mean |step_a - step_b|
    onset_within_5_steps: float            # fraction within 5 steps

    # Severity correlation
    severity_correlation: Optional[float]  # Spearman rho

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        lines = [
            f"Agreement: {self.annotator_a} vs {self.annotator_b} (n={self.n_traces})",
            f"  Binary:   kappa={self.binary_kappa:.3f}  agree={self.binary_agreement:.0%}",
            f"  Category: kappa={self.category_kappa:.3f}  agree={self.category_agreement:.0%}",
            f"  Subtype:  kappa={self.subtype_kappa:.3f}  agree={self.subtype_agreement:.0%}",
        ]
        if self.onset_mean_abs_error is not None:
            lines.append(f"  Onset MAE: {self.onset_mean_abs_error:.1f} steps  "
                         f"within-5: {self.onset_within_5_steps:.0%}")
        if self.severity_correlation is not None:
            lines.append(f"  Severity rho: {self.severity_correlation:.3f}")
        return "\n".join(lines)


def _cohens_kappa(labels_a: list, labels_b: list) -> float:
    """Compute Cohen's kappa for two lists of labels."""
    assert len(labels_a) == len(labels_b)
    n = len(labels_a)
    if n == 0:
        return 0.0

    # All unique labels
    all_labels = sorted(set(labels_a) | set(labels_b))
    if len(all_labels) <= 1:
        # Perfect agreement trivially
        return 1.0

    # Confusion matrix
    label_to_idx = {l: i for i, l in enumerate(all_labels)}
    k = len(all_labels)
    matrix = [[0] * k for _ in range(k)]
    for a, b in zip(labels_a, labels_b):
        matrix[label_to_idx[a]][label_to_idx[b]] += 1

    # Observed agreement
    p_o = sum(matrix[i][i] for i in range(k)) / n

    # Expected agreement
    row_sums = [sum(matrix[i]) for i in range(k)]
    col_sums = [sum(matrix[i][j] for i in range(k)) for j in range(k)]
    p_e = sum(row_sums[i] * col_sums[i] for i in range(k)) / (n * n)

    if p_e >= 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


def compute_agreement(
    store: AnnotationStore,
    annotator_a: str,
    annotator_b: str,
) -> AgreementReport:
    """Compute inter-annotator agreement between two annotators.

    Only considers traces that BOTH annotators have labeled.
    """
    a_by_trace = {a.trace_id: a for a in store.get_by_annotator(annotator_a)}
    b_by_trace = {a.trace_id: a for a in store.get_by_annotator(annotator_b)}
    common_traces = sorted(set(a_by_trace) & set(b_by_trace))

    if not common_traces:
        return AgreementReport(
            annotator_a=annotator_a,
            annotator_b=annotator_b,
            n_traces=0,
            binary_kappa=0.0, binary_agreement=0.0,
            category_kappa=0.0, category_agreement=0.0,
            subtype_kappa=0.0, subtype_agreement=0.0,
            onset_mean_abs_error=None, onset_within_5_steps=0.0,
            severity_correlation=None,
        )

    # Collect paired labels
    binary_a, binary_b = [], []
    cat_a, cat_b = [], []
    sub_a, sub_b = [], []
    onset_diffs = []
    sev_a, sev_b = [], []

    for trace_id in common_traces:
        aa = a_by_trace[trace_id]
        bb = b_by_trace[trace_id]

        # Binary
        binary_a.append("failure" if aa.has_failure else "clean")
        binary_b.append("failure" if bb.has_failure else "clean")

        # Category
        cat_a.append(aa.primary_caft_category or "none")
        cat_b.append(bb.primary_caft_category or "none")

        # Subtype
        sub_a.append(aa.primary_caft_code or "none")
        sub_b.append(bb.primary_caft_code or "none")

        # Onset
        if aa.failure_onset_step is not None and bb.failure_onset_step is not None:
            onset_diffs.append(abs(aa.failure_onset_step - bb.failure_onset_step))

        # Severity
        if aa.has_failure and bb.has_failure:
            sev_a.append(aa.severity)
            sev_b.append(bb.severity)

    n = len(common_traces)

    # Binary kappa
    binary_kappa = _cohens_kappa(binary_a, binary_b)
    binary_agree = sum(1 for a, b in zip(binary_a, binary_b) if a == b) / n

    # Category kappa
    cat_kappa = _cohens_kappa(cat_a, cat_b)
    cat_agree = sum(1 for a, b in zip(cat_a, cat_b) if a == b) / n

    # Subtype kappa
    sub_kappa = _cohens_kappa(sub_a, sub_b)
    sub_agree = sum(1 for a, b in zip(sub_a, sub_b) if a == b) / n

    # Onset localization
    onset_mae = sum(onset_diffs) / len(onset_diffs) if onset_diffs else None
    onset_w5 = sum(1 for d in onset_diffs if d <= 5) / len(onset_diffs) if onset_diffs else 0.0

    # Severity correlation (Spearman rank)
    sev_rho = _spearman_rho(sev_a, sev_b) if len(sev_a) >= 3 else None

    return AgreementReport(
        annotator_a=annotator_a,
        annotator_b=annotator_b,
        n_traces=n,
        binary_kappa=round(binary_kappa, 4),
        binary_agreement=round(binary_agree, 4),
        category_kappa=round(cat_kappa, 4),
        category_agreement=round(cat_agree, 4),
        subtype_kappa=round(sub_kappa, 4),
        subtype_agreement=round(sub_agree, 4),
        onset_mean_abs_error=round(onset_mae, 2) if onset_mae is not None else None,
        onset_within_5_steps=round(onset_w5, 4),
        severity_correlation=round(sev_rho, 4) if sev_rho is not None else None,
    )


def _spearman_rho(x: list[float], y: list[float]) -> float:
    """Compute Spearman rank correlation (no numpy dependency)."""
    n = len(x)
    if n < 2:
        return 0.0

    def _rank(vals):
        indexed = sorted(enumerate(vals), key=lambda p: p[1])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and indexed[j + 1][1] == indexed[i][1]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[indexed[k][0]] = avg_rank
            i = j + 1
        return ranks

    rx = _rank(x)
    ry = _rank(y)
    d_sq = sum((a - b) ** 2 for a, b in zip(rx, ry))
    return 1.0 - (6.0 * d_sq) / (n * (n * n - 1))
