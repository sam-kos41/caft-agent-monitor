"""Disagreement computation and annotation triage utilities.

Computes structured disagreement between annotation layers (detector vs auto,
auto vs human, human vs adjudicated) and scores sessions for annotation
priority to support active learning and quality control.

Disagreement is valuable signal:
  - detector vs auto: calibrates detector thresholds
  - auto vs human: measures LLM annotation quality
  - human vs adjudicated: identifies hard cases
  - any disagreement: drives annotation queue priority
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

from agentdiag.annotation_models import (
    AnnotationRecord,
    AnnotatorType,
    LabelStatus,
)


# ──────────────────────────────────────────────────────────────────
# Pairwise disagreement
# ──────────────────────────────────────────────────────────────────

@dataclass
class DisagreementSummary:
    """Structured comparison between two AnnotationRecords."""
    source_a: str  # annotator_type of first record
    source_b: str  # annotator_type of second record
    session_id: str

    # Binary: do they agree on has_failure?
    binary_agree: bool = True

    # Code: do they agree on primary CAFT code?
    code_agree: bool = True
    code_a: str = ""
    code_b: str = ""

    # Severity delta (abs)
    severity_delta: int = 0

    # Onset delta (abs steps)
    onset_delta: int = 0

    # Confidence delta
    confidence_delta: int = 0

    # Human-readable summary
    description: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def has_disagreement(self) -> bool:
        return not self.binary_agree or not self.code_agree


def compare_annotations(a: AnnotationRecord, b: AnnotationRecord) -> DisagreementSummary:
    """Compare two AnnotationRecords and produce a structured disagreement summary.

    Either record may be from any layer (detector, auto, human, adjudicated).
    """
    binary_agree = a.has_failure == b.has_failure
    code_agree = a.primary_caft_code == b.primary_caft_code
    severity_delta = abs(a.severity - b.severity)
    onset_delta = abs(a.onset_step - b.onset_step) if a.onset_step and b.onset_step else 0
    confidence_delta = abs(a.confidence - b.confidence)

    # Build description
    parts = []
    if not binary_agree:
        label_a = "failure" if a.has_failure else "clean"
        label_b = "failure" if b.has_failure else "clean"
        parts.append(f"binary: {a.annotator_type}={label_a} vs {b.annotator_type}={label_b}")
    elif a.has_failure and b.has_failure and not code_agree:
        parts.append(f"code: {a.annotator_type}={a.primary_caft_code} vs "
                     f"{b.annotator_type}={b.primary_caft_code}")
    if severity_delta > 1:
        parts.append(f"severity delta={severity_delta}")
    if onset_delta > 10:
        parts.append(f"onset delta={onset_delta} steps")

    return DisagreementSummary(
        source_a=a.annotator_type,
        source_b=b.annotator_type,
        session_id=a.effective_session_id,
        binary_agree=binary_agree,
        code_agree=code_agree,
        code_a=a.primary_caft_code,
        code_b=b.primary_caft_code,
        severity_delta=severity_delta,
        onset_delta=onset_delta,
        confidence_delta=confidence_delta,
        description="; ".join(parts) if parts else "agree",
    )


# ──────────────────────────────────────────────────────────────────
# Session-level disagreement bundle
# ──────────────────────────────────────────────────────────────────

@dataclass
class SessionDisagreementBundle:
    """All pairwise disagreements for a single session."""
    session_id: str
    detector_vs_auto: Optional[DisagreementSummary] = None
    auto_vs_human: Optional[DisagreementSummary] = None
    human_vs_adjudicated: Optional[DisagreementSummary] = None
    detector_vs_human: Optional[DisagreementSummary] = None

    @property
    def total_disagreements(self) -> int:
        count = 0
        for d in [self.detector_vs_auto, self.auto_vs_human,
                  self.human_vs_adjudicated, self.detector_vs_human]:
            if d and d.has_disagreement:
                count += 1
        return count

    @property
    def any_disagreement(self) -> bool:
        return self.total_disagreements > 0

    def to_dict(self) -> dict:
        d = {"session_id": self.session_id, "total_disagreements": self.total_disagreements}
        for name in ["detector_vs_auto", "auto_vs_human",
                     "human_vs_adjudicated", "detector_vs_human"]:
            val = getattr(self, name)
            d[name] = val.to_dict() if val else None
        return d


def compute_session_disagreement_bundle(
    session_id: str,
    records: list[AnnotationRecord],
) -> SessionDisagreementBundle:
    """Compute all pairwise disagreements for a session's annotation records.

    Groups records by annotator_type, picks the best (most recent) of each
    type, and compares adjacent layers.
    """
    by_type: dict[str, AnnotationRecord] = {}
    for r in records:
        existing = by_type.get(r.annotator_type)
        if existing is None or r.updated_at > existing.updated_at:
            by_type[r.annotator_type] = r

    bundle = SessionDisagreementBundle(session_id=session_id)

    det = by_type.get(AnnotatorType.DETECTOR.value)
    auto = by_type.get(AnnotatorType.AUTO.value)
    human = by_type.get(AnnotatorType.HUMAN.value)
    adj = by_type.get(AnnotatorType.ADJUDICATED.value)

    if det and auto:
        bundle.detector_vs_auto = compare_annotations(det, auto)
    if auto and human:
        bundle.auto_vs_human = compare_annotations(auto, human)
    if human and adj:
        bundle.human_vs_adjudicated = compare_annotations(human, adj)
    if det and human:
        bundle.detector_vs_human = compare_annotations(det, human)

    return bundle


# ──────────────────────────────────────────────────────────────────
# Annotation priority scoring (for triage queue)
# ──────────────────────────────────────────────────────────────────

@dataclass
class AnnotationPriority:
    """Priority score for a session needing annotation."""
    session_id: str
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)

    # Component scores
    severity_score: float = 0.0
    uncertainty_score: float = 0.0
    disagreement_score: float = 0.0
    novelty_score: float = 0.0
    unlabeled_score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def compute_annotation_priority(
    session_id: str,
    records: list[AnnotationRecord],
    all_failure_counts: dict[str, int] | None = None,
) -> AnnotationPriority:
    """Score how urgently a session needs annotation.

    Higher score = more urgent. Factors:
    - severity: high-severity findings need review first
    - uncertainty: low confidence = needs human judgment
    - disagreement: conflicting layers = hard case
    - novelty: rare failure types need more examples
    - unlabeled: completely unlabeled sessions need any label

    Args:
        session_id: The session to score.
        records: All AnnotationRecords for this session.
        all_failure_counts: Global failure type frequencies (for novelty).
    """
    priority = AnnotationPriority(session_id=session_id)

    if not records:
        priority.unlabeled_score = 10.0
        priority.reasons.append("completely unlabeled")
        priority.score = 10.0
        return priority

    # Check if already adjudicated
    has_adjudicated = any(
        r.label_status == LabelStatus.ADJUDICATED.value for r in records
    )
    if has_adjudicated:
        priority.score = 0.0
        return priority

    # Severity score: max severity across all layers
    max_severity = max((r.severity for r in records if r.has_failure), default=0)
    priority.severity_score = max_severity * 2.0  # 0-10 scale
    if max_severity >= 4:
        priority.reasons.append(f"high severity ({max_severity})")

    # Uncertainty score: inverse of min confidence
    confidences = [r.confidence for r in records if r.confidence > 0]
    if confidences:
        min_conf = min(confidences)
        priority.uncertainty_score = (5 - min_conf) * 2.0  # 0-8 scale
        if min_conf <= 2:
            priority.reasons.append(f"low confidence ({min_conf})")
    else:
        priority.uncertainty_score = 6.0
        priority.reasons.append("no confidence scores")

    # Disagreement score
    bundle = compute_session_disagreement_bundle(session_id, records)
    priority.disagreement_score = bundle.total_disagreements * 3.0
    if bundle.any_disagreement:
        priority.reasons.append(f"{bundle.total_disagreements} layer disagreements")

    # Novelty score: rare failure types
    if all_failure_counts:
        for r in records:
            if r.has_failure and r.primary_caft_code:
                name = r.primary_caft_name or r.primary_caft_code
                count = all_failure_counts.get(name, 0)
                if count <= 2:
                    priority.novelty_score = max(priority.novelty_score, 5.0)
                    priority.reasons.append(f"rare type: {name} (n={count})")
                elif count <= 5:
                    priority.novelty_score = max(priority.novelty_score, 2.0)

    # Unlabeled bonus
    has_human = any(r.annotator_type in {AnnotatorType.HUMAN.value, AnnotatorType.ADJUDICATED.value}
                    for r in records)
    if not has_human:
        priority.unlabeled_score = 4.0
        priority.reasons.append("no human review")

    priority.score = (
        priority.severity_score
        + priority.uncertainty_score
        + priority.disagreement_score
        + priority.novelty_score
        + priority.unlabeled_score
    )

    return priority


def rank_annotation_queue(
    records_by_session: dict[str, list[AnnotationRecord]],
    all_failure_counts: dict[str, int] | None = None,
    limit: int = 50,
) -> list[AnnotationPriority]:
    """Rank all sessions by annotation priority, highest first.

    Args:
        records_by_session: {session_id: [AnnotationRecord, ...]}.
        all_failure_counts: Global failure type frequencies.
        limit: Max results to return.

    Returns:
        Sorted list of AnnotationPriority, highest score first.
    """
    priorities = []
    for session_id, records in records_by_session.items():
        p = compute_annotation_priority(session_id, records, all_failure_counts)
        if p.score > 0:
            priorities.append(p)

    priorities.sort(key=lambda p: p.score, reverse=True)
    return priorities[:limit]
