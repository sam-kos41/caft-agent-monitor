"""First-class annotation models for the CAFT labeling pipeline.

Defines AnnotationRecord — the universal annotation artifact that captures
detector predictions, auto-annotations, human labels, and adjudicated gold
labels as distinct, versionable, non-destructive layers.

Design principles:
  1. Never overwrite raw detector output with annotation.
  2. Every annotation tracks its source (detector/auto/human/adjudicated).
  3. Version pinning: CAFT version, codebook version, prompt version.
  4. Label lifecycle: unlabeled → auto_labeled → human_reviewed → adjudicated.
  5. Disagreement fields are populated by comparison utilities, not by annotators.

Relationship to existing types:
  - CaftDiagnosis (caft/base.py): immutable detector output → preserved as-is
  - Annotation (annotation.py): existing schema → still valid for inter-annotator
  - AnnotationRecord (this module): the unified annotation artifact
  - DiagnosticCase (context/openviking.py): OpenViking case → links to records
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional

from agentdiag.caft.taxonomy import CAFT_TAXONOMY, Detectability


# ──────────────────────────────────────────────────────────────────
# Version constants — bump when definitions change
# ──────────────────────────────────────────────────────────────────

CAFT_VERSION = "1.0"           # CAFT taxonomy version (32 types, 8 categories)
CODEBOOK_VERSION = "1.0"       # Annotation codebook/criteria version
ANNOTATION_PROMPT_VERSION = "1.0"  # auto_annotate_prompt.py version


# ──────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────

class LabelStatus(Enum):
    """Lifecycle status of an annotation label."""
    UNLABELED = "unlabeled"             # No annotation exists yet
    AUTO_LABELED = "auto_labeled"       # LLM-generated annotation
    HUMAN_REVIEWED = "human_reviewed"   # Human annotator has reviewed
    ADJUDICATED = "adjudicated"         # Final gold label after disagreement resolution
    HELD_OUT_TEST = "held_out_test"     # Reserved for evaluation, excluded from tuning


class AnnotatorType(Enum):
    """Source of an annotation."""
    DETECTOR = "detector"       # Raw detector prediction (immutable)
    AUTO = "auto"               # LLM-generated annotation
    HUMAN = "human"             # Human annotator
    ADJUDICATED = "adjudicated" # Final resolved label


# ──────────────────────────────────────────────────────────────────
# AnnotationRecord
# ──────────────────────────────────────────────────────────────────

@dataclass
class AnnotationRecord:
    """A single annotation for one trace/session.

    Multiple AnnotationRecords can exist for the same session_id,
    one per annotator_type layer. This preserves all layers of truth:
      1. detector → raw prediction
      2. auto     → LLM confirmation
      3. human    → human review
      4. adjudicated → final gold label
    """
    # Identity
    annotation_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    session_id: str = ""
    trace_id: str = ""  # alias for session_id if different

    # Source
    annotator_type: str = AnnotatorType.DETECTOR.value
    annotator_id: str = ""  # model name, username, or detector name

    # Versioning
    caft_version: str = CAFT_VERSION
    codebook_version: str = CODEBOOK_VERSION
    annotation_prompt_version: str = ANNOTATION_PROMPT_VERSION

    # Label lifecycle
    label_status: str = LabelStatus.UNLABELED.value

    # Classification
    has_failure: bool = False
    primary_caft_code: str = ""           # e.g., "2.2"
    secondary_caft_codes: list[str] = field(default_factory=list)

    # Localization
    onset_step: int = 0
    evidence_steps: list[int] = field(default_factory=list)

    # Severity & confidence
    severity: int = 0             # 1-5 (0 = clean / not set)
    confidence: int = 0           # 1-5 (integer scale)

    # Observability
    observable_vs_latent: str = "observable"

    # Rationale
    free_text_rationale: str = ""

    # Disagreement tracking (populated by comparison utilities, not annotators)
    disagreement_with_detector: Optional[str] = None
    disagreement_with_auto: Optional[str] = None
    disagreement_with_human: Optional[str] = None

    # Timestamps
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # ── Serialization ───────────────────────────────────────────

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AnnotationRecord":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

    # ── Derived properties ──────────────────────────────────────

    @property
    def primary_caft_name(self) -> str:
        """Resolve primary CAFT code to failure name."""
        t = CAFT_TAXONOMY.get(self.primary_caft_code)
        return t.name if t else ""

    @property
    def primary_caft_category(self) -> str:
        """Resolve primary CAFT code to category."""
        t = CAFT_TAXONOMY.get(self.primary_caft_code)
        return t.category if t else ""

    @property
    def is_observable(self) -> bool:
        t = CAFT_TAXONOMY.get(self.primary_caft_code)
        if t:
            return t.detectability == Detectability.OBSERVABLE
        return self.observable_vs_latent == "observable"

    @property
    def effective_session_id(self) -> str:
        """Return trace_id if set, else session_id."""
        return self.trace_id or self.session_id

    # ── Validation ──────────────────────────────────────────────

    def validate(self) -> list[str]:
        """Validate the annotation. Returns list of issues."""
        issues = []
        if not self.session_id and not self.trace_id:
            issues.append("Missing session_id or trace_id")
        if self.annotator_type not in {e.value for e in AnnotatorType}:
            issues.append(f"Unknown annotator_type: {self.annotator_type}")
        if self.label_status not in {e.value for e in LabelStatus}:
            issues.append(f"Unknown label_status: {self.label_status}")
        if self.has_failure:
            if not self.primary_caft_code:
                issues.append("has_failure=True but no primary_caft_code")
            elif self.primary_caft_code not in CAFT_TAXONOMY:
                issues.append(f"Unknown CAFT code: {self.primary_caft_code}")
            if self.severity < 1 or self.severity > 5:
                issues.append(f"severity must be 1-5, got {self.severity}")
            if self.confidence < 1 or self.confidence > 5:
                issues.append(f"confidence must be 1-5, got {self.confidence}")
            for code in self.secondary_caft_codes:
                if code not in CAFT_TAXONOMY:
                    issues.append(f"Unknown secondary CAFT code: {code}")
        return issues

    # ── Dedup key ───────────────────────────────────────────────

    @property
    def dedup_key(self) -> tuple:
        """Unique key for deduplication."""
        return (
            self.effective_session_id,
            self.annotator_type,
            self.annotator_id,
            self.caft_version,
        )


# ──────────────────────────────────────────────────────────────────
# Builder functions — one per annotation layer
# ──────────────────────────────────────────────────────────────────

def build_detector_annotation(
    session_id: str,
    diagnosis: "CaftDiagnosis",
    detector_name: str = "",
) -> AnnotationRecord:
    """Build an AnnotationRecord from a raw detector CaftDiagnosis.

    Detector annotations are immutable predictions. They represent what the
    heuristic detectors found, before any LLM or human review.
    """
    from agentdiag.caft.base import CaftDiagnosis as _Diag
    return AnnotationRecord(
        session_id=session_id,
        trace_id=session_id,
        annotator_type=AnnotatorType.DETECTOR.value,
        annotator_id=detector_name or diagnosis.failure_name,
        label_status=LabelStatus.UNLABELED.value,
        has_failure=True,
        primary_caft_code=diagnosis.caft_code,
        onset_step=diagnosis.at_step,
        severity=_severity_to_int(diagnosis.severity.value),
        confidence=max(1, min(5, int(diagnosis.confidence * 5))),
        observable_vs_latent="observable",
        free_text_rationale=diagnosis.description,
        evidence_steps=[diagnosis.at_step],
    )


def build_auto_annotation(
    session_id: str,
    has_failure: bool,
    primary_caft_code: str = "",
    secondary_caft_codes: list[str] | None = None,
    onset_step: int = 0,
    severity: int = 0,
    confidence: int = 0,
    rationale: str = "",
    annotator_id: str = "claude-sonnet-4-5",
    prompt_version: str = ANNOTATION_PROMPT_VERSION,
) -> AnnotationRecord:
    """Build an AnnotationRecord from LLM auto-annotation output.

    Auto annotations can be replaced by newer prompt versions. They sit
    between detector predictions and human labels in the trust hierarchy.
    """
    return AnnotationRecord(
        session_id=session_id,
        trace_id=session_id,
        annotator_type=AnnotatorType.AUTO.value,
        annotator_id=annotator_id,
        annotation_prompt_version=prompt_version,
        label_status=LabelStatus.AUTO_LABELED.value,
        has_failure=has_failure,
        primary_caft_code=primary_caft_code,
        secondary_caft_codes=secondary_caft_codes or [],
        onset_step=onset_step,
        severity=severity,
        confidence=confidence,
        free_text_rationale=rationale,
    )


def build_human_annotation(
    session_id: str,
    annotator_id: str,
    has_failure: bool,
    primary_caft_code: str = "",
    secondary_caft_codes: list[str] | None = None,
    onset_step: int = 0,
    evidence_steps: list[int] | None = None,
    severity: int = 0,
    confidence: int = 0,
    rationale: str = "",
) -> AnnotationRecord:
    """Build an AnnotationRecord from human annotation.

    Human annotations are append-only. Multiple humans can annotate
    the same session. Human labels outrank auto labels in trust.
    """
    return AnnotationRecord(
        session_id=session_id,
        trace_id=session_id,
        annotator_type=AnnotatorType.HUMAN.value,
        annotator_id=annotator_id,
        label_status=LabelStatus.HUMAN_REVIEWED.value,
        has_failure=has_failure,
        primary_caft_code=primary_caft_code,
        secondary_caft_codes=secondary_caft_codes or [],
        onset_step=onset_step,
        evidence_steps=evidence_steps or [],
        severity=severity,
        confidence=confidence,
        free_text_rationale=rationale,
    )


def build_adjudicated_annotation(
    session_id: str,
    adjudicator_id: str,
    has_failure: bool,
    primary_caft_code: str = "",
    secondary_caft_codes: list[str] | None = None,
    onset_step: int = 0,
    evidence_steps: list[int] | None = None,
    severity: int = 0,
    confidence: int = 5,
    rationale: str = "",
) -> AnnotationRecord:
    """Build the final adjudicated gold label for a session.

    Adjudicated labels are the only trusted source for evaluation
    and calibration. They represent the resolved ground truth after
    considering all other annotation layers.
    """
    return AnnotationRecord(
        session_id=session_id,
        trace_id=session_id,
        annotator_type=AnnotatorType.ADJUDICATED.value,
        annotator_id=adjudicator_id,
        label_status=LabelStatus.ADJUDICATED.value,
        has_failure=has_failure,
        primary_caft_code=primary_caft_code,
        secondary_caft_codes=secondary_caft_codes or [],
        onset_step=onset_step,
        evidence_steps=evidence_steps or [],
        severity=severity,
        confidence=confidence,
        free_text_rationale=rationale,
    )


# ──────────────────────────────────────────────────────────────────
# Conversion from legacy ground_truth_50.json format
# ──────────────────────────────────────────────────────────────────

def from_ground_truth_trace(trace: dict, annotator_id: str = "manual") -> list[AnnotationRecord]:
    """Convert a single trace entry from ground_truth_*.json into AnnotationRecords.

    Returns one or more records:
    - One "clean" record if no failures
    - One record per failure_detail entry, or one from actual_failures if no details
    """
    session_id = trace.get("session_id", "")
    actual_failures = trace.get("actual_failures", [])
    failure_details = trace.get("failure_details", [])
    completed = trace.get("agent_completed", True)

    if not actual_failures:
        # Clean trace — single record
        return [build_human_annotation(
            session_id=session_id,
            annotator_id=annotator_id,
            has_failure=False,
            confidence=5,
            rationale=f"Clean trace. Completed={completed}",
        )]

    records = []
    if failure_details:
        for detail in failure_details:
            records.append(build_human_annotation(
                session_id=session_id,
                annotator_id=annotator_id,
                has_failure=True,
                primary_caft_code=detail.get("caft_code", ""),
                onset_step=detail.get("onset_step", 0),
                severity=detail.get("severity", 3),
                confidence=detail.get("confidence", 3),
                rationale=detail.get("rationale", ""),
            ))
    else:
        # Legacy format: actual_failures without details
        for name in actual_failures:
            code = _name_to_code(name)
            records.append(build_human_annotation(
                session_id=session_id,
                annotator_id=annotator_id,
                has_failure=True,
                primary_caft_code=code,
                rationale=f"From ground truth actual_failures: {name}",
            ))

    return records


def from_ground_truth_file(gt: dict) -> list[AnnotationRecord]:
    """Convert an entire ground_truth_*.json file into AnnotationRecords."""
    annotator_id = gt.get("annotator", "manual")
    records = []
    for trace in gt.get("traces", []):
        records.extend(from_ground_truth_trace(trace, annotator_id))
    return records


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def _severity_to_int(severity_str: str) -> int:
    """Convert CaftSeverity string to 1-5 integer."""
    mapping = {"info": 2, "warning": 3, "critical": 5}
    return mapping.get(severity_str, 3)


def _name_to_code(failure_name: str) -> str:
    """Look up CAFT code from failure name."""
    for code, t in CAFT_TAXONOMY.items():
        if t.name == failure_name:
            return code
    return ""
