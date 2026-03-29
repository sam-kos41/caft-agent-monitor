"""Evaluation metrics for CAFT detector assessment.

Defines the matching logic (detection ↔ annotation), per-detector and
aggregate precision/recall/F1, bootstrap confidence intervals, and
McNemar's test for pairwise mode comparison.

Shared by both ``scripts/run_ablation.py`` and the existing
``evaluate.py`` pipeline.  Import only numpy (no scipy).

Matching rules
--------------
- A detection matches an annotation if:
      same failure_name AND |onset_step - annotated_onset| <= match_window
- If CAFT code matches but step outside window: partial match (0.5 TP)
- Wrong CAFT code at right step: FP
- Annotation with NO candidate at all: FN
- Annotation matched by candidate but LLM rejects: FN (LLM error)
- Latent CAFT types (require semantic understanding): tracked separately

Note on onset_step
------------------
Ground-truth onset_step values are JSONL line numbers in the original trace file,
NOT TraceEvent indices. Since these are incomparable, all annotations are loaded
with onset_step=0, which disables step-window matching. Matching is purely by
(trace_id, failure_name). This is equivalent to session-level detection evaluation.

Ablation modes
--------------
1. ``strict``   — production thresholds, no LLM
2. ``loose``    — candidate-generator thresholds, no LLM
3. ``loose+llm`` — candidate generator + LLM confirmation
4. ``oracle``   — candidate generator + perfect LLM (ground truth filter)
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

from agentdiag.caft.base import CaftDiagnosis


# ── Tier classification ──────────────────────────────────────────────
# Tier A: structural/rule-based detectors
# Tier B: semantic/LLM-assessed detectors
_TIER_A_DETECTORS = {"context_loss", "stall", "error_cascade", "analysis_paralysis", "recovery_failure",
                     "step_repetition", "tool_thrashing", "token_explosion", "tool_misuse", "strategic_myopia",
                     "missing_verification", "reasoning_action_mismatch"}
_TIER_B_DETECTORS = {"premature_termination", "goal_drift"}


# ── Core dataclasses ─────────────────────────────────────────────────

@dataclass
class Detection:
    """A single detector firing, used for evaluation matching."""
    trace_id: str
    failure_name: str
    caft_code: str
    onset_step: int
    confidence: float = 0.0
    confirmed: Optional[bool] = None   # None = no LLM, True/False = LLM result
    latency_ms: float = 0.0            # rule detection time
    llm_latency_ms: float = 0.0        # LLM confirmation time
    llm_tokens: int = 0                # tokens used for LLM call
    evidence: Optional[dict] = None    # detector evidence for LLM prompt
    force_llm_review: bool = False     # bypass auto-confirm, always send to LLM


@dataclass
class Annotation:
    """A single ground-truth annotation for matching."""
    trace_id: str
    failure_name: str
    caft_code: str
    onset_step: int = 0
    is_latent: bool = False            # latent types tracked separately


@dataclass
class MatchResult:
    """Result of matching one detection to one annotation."""
    detection: Detection
    annotation: Optional[Annotation]
    match_type: str  # "full", "partial", "fp", "fn"
    tp_weight: float = 0.0  # 1.0 for full, 0.5 for partial, 0.0 for fp/fn


@dataclass
class DetectorResult:
    """Per-detector evaluation metrics."""
    detector: str
    caft_code: str
    tp: float = 0.0        # can be fractional (partial matches = 0.5)
    fp: int = 0
    fn: int = 0
    candidates: int = 0    # total candidates generated
    confirmed: int = 0     # confirmed by LLM
    rejected: int = 0      # rejected by LLM
    uncertain: int = 0     # LLM returned uncertain
    avg_latency_ms: float = 0.0
    avg_llm_latency_ms: float = 0.0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 1.0 if self.fn == 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 1.0 if self.fp == 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["precision"] = round(self.precision, 4)
        d["recall"] = round(self.recall, 4)
        d["f1"] = round(self.f1, 4)
        return d


@dataclass
class BootstrapCI:
    """Bootstrap confidence interval for a metric."""
    metric: str
    point_estimate: float
    ci_lower: float
    ci_upper: float
    n_iterations: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ModeComparison:
    """Pairwise comparison between two evaluation modes."""
    mode_a: str
    mode_b: str
    mcnemar_statistic: float
    p_value: float
    significant: bool  # at alpha=0.05
    a_better: bool     # True if mode_a has higher F1

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ComparisonTable:
    """Full comparison across all modes."""
    pairwise: list[ModeComparison]
    per_detector_winners: dict[str, str]  # detector → best mode name

    def to_dict(self) -> dict:
        return {
            "pairwise": [p.to_dict() for p in self.pairwise],
            "per_detector_winners": self.per_detector_winners,
        }


@dataclass
class EvalReport:
    """Complete evaluation report for one ablation mode.

    Named ``EvalReport`` to avoid collision with ``evaluate.py``'s
    ``EvaluationReport`` which has a different shape (per-session).
    """
    mode: str  # strict / loose / loose+llm / oracle
    per_detector: list[DetectorResult]
    macro_p: float = 0.0
    macro_r: float = 0.0
    macro_f1: float = 0.0
    micro_p: float = 0.0
    micro_r: float = 0.0
    micro_f1: float = 0.0
    candidates_per_trace: float = 0.0
    llm_confirmation_rate: Optional[float] = None
    llm_agreement_with_gt: Optional[float] = None
    latency_ms: dict[str, float] = field(default_factory=dict)
    bootstrap_ci: dict[str, BootstrapCI] = field(default_factory=dict)
    n_traces: int = 0
    n_annotations: int = 0
    n_candidates: int = 0
    # Session-level binary metrics (any failure detected per session?)
    session_p: float = 0.0
    session_r: float = 0.0
    session_f1: float = 0.0
    # Latent type tracking
    latent_fn: int = 0  # latent annotations with no candidate

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "per_detector": [d.to_dict() for d in self.per_detector],
            "macro_p": round(self.macro_p, 4),
            "macro_r": round(self.macro_r, 4),
            "macro_f1": round(self.macro_f1, 4),
            "micro_p": round(self.micro_p, 4),
            "micro_r": round(self.micro_r, 4),
            "micro_f1": round(self.micro_f1, 4),
            "candidates_per_trace": round(self.candidates_per_trace, 2),
            "llm_confirmation_rate": (
                round(self.llm_confirmation_rate, 4)
                if self.llm_confirmation_rate is not None else None
            ),
            "llm_agreement_with_gt": (
                round(self.llm_agreement_with_gt, 4)
                if self.llm_agreement_with_gt is not None else None
            ),
            "latency_ms": {k: round(v, 1) for k, v in self.latency_ms.items()},
            "bootstrap_ci": {k: v.to_dict() for k, v in self.bootstrap_ci.items()},
            "session_p": round(self.session_p, 4),
            "session_r": round(self.session_r, 4),
            "session_f1": round(self.session_f1, 4),
            "n_traces": self.n_traces,
            "n_annotations": self.n_annotations,
            "n_candidates": self.n_candidates,
            "latent_fn": self.latent_fn,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


# ── Annotation validation ────────────────────────────────────────────

@dataclass
class AnnotationQualityReport:
    """Pre-flight quality report for annotation data."""
    total_annotations: int = 0
    total_sessions: int = 0
    n_human_reviewed: int = 0
    n_adjudicated: int = 0
    n_draft: int = 0
    n_detector_only: int = 0
    n_clean_sessions: int = 0
    n_failure_sessions: int = 0
    failure_type_counts: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    is_valid: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def validate_annotations_jsonl(path: str) -> AnnotationQualityReport:
    """Validate annotation JSONL file for ground-truth quality.

    Checks for:
        1. Circular ground truth (detector annotations posing as GT)
        2. Human/detector conflicts (human says CLEAN, detector says failure)
        3. DRAFT annotations that need review
        4. Class imbalance warnings
        5. Minimum sample sizes per failure type
        6. Session ID consistency (short/long mismatches)

    Returns:
        AnnotationQualityReport with warnings/errors.
    """
    report = AnnotationQualityReport()

    # Load all records grouped by session prefix
    session_groups: dict[str, list[dict]] = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            sid = d.get("session_id", d.get("trace_id", ""))
            prefix = sid[:8]
            session_groups[prefix].append(d)
            report.total_annotations += 1

    report.total_sessions = len(session_groups)

    # Analyze each session
    detector_only_sessions: list[str] = []
    human_detector_conflicts: list[str] = []

    for prefix, records in session_groups.items():
        has_human = any(
            r.get("annotator_type") in ("human", "adjudicated")
            and r.get("label_status") in ("human_reviewed", "adjudicated")
            for r in records
        )
        has_detector = any(
            r.get("annotator_type") == "detector"
            and r.get("label_status") == "unlabeled"
            and r.get("has_failure", False)
            for r in records
        )
        has_draft = any(
            r.get("label_status") == "auto_labeled"
            for r in records
        )

        # Count statuses
        for r in records:
            status = r.get("label_status", "unlabeled")
            atype = r.get("annotator_type", "detector")
            if status == "human_reviewed" and atype == "human":
                report.n_human_reviewed += 1
            elif status == "adjudicated":
                report.n_adjudicated += 1
            elif status == "auto_labeled":
                report.n_draft += 1
            elif status == "unlabeled" and atype == "detector":
                report.n_detector_only += 1

        # Check for human CLEAN vs detector failure conflicts
        human_clean = any(
            r.get("annotator_type") in ("human", "adjudicated")
            and r.get("label_status") in ("human_reviewed", "adjudicated")
            and not r.get("has_failure", False)
            for r in records
        )
        if human_clean:
            report.n_clean_sessions += 1
            if has_detector:
                human_detector_conflicts.append(prefix)
        else:
            has_any_failure = any(r.get("has_failure", False) for r in records)
            if has_any_failure:
                report.n_failure_sessions += 1
                for r in records:
                    if r.get("has_failure", False):
                        code = r.get("primary_caft_code", "unknown")
                        report.failure_type_counts[code] = \
                            report.failure_type_counts.get(code, 0) + 1

        # Detector-only session (no human annotation at all)
        if has_detector and not has_human and not has_draft:
            detector_only_sessions.append(prefix)

    # Generate warnings and errors
    if detector_only_sessions:
        report.errors.append(
            f"CIRCULAR GROUND TRUTH: {len(detector_only_sessions)} session(s) have "
            f"only unlabeled detector annotations — these would evaluate detectors "
            f"against their own output. Sessions: "
            f"{detector_only_sessions[:5]}{'...' if len(detector_only_sessions) > 5 else ''}"
        )
        report.is_valid = False

    if human_detector_conflicts:
        report.warnings.append(
            f"HUMAN/DETECTOR CONFLICT: {len(human_detector_conflicts)} session(s) "
            f"have human=CLEAN but detector fired. The new loader correctly suppresses "
            f"these, but they indicate noisy detectors. Sessions: "
            f"{human_detector_conflicts[:5]}{'...' if len(human_detector_conflicts) > 5 else ''}"
        )

    if report.n_draft > 0:
        report.warnings.append(
            f"DRAFT ANNOTATIONS: {report.n_draft} auto_labeled annotation(s) have "
            f"not been human-reviewed. These are included in ground truth but may "
            f"be incorrect. Review them before publishing results."
        )

    # Class imbalance check
    if report.failure_type_counts:
        total_failures = sum(report.failure_type_counts.values())
        for code, count in report.failure_type_counts.items():
            if count / total_failures > 0.5:
                report.warnings.append(
                    f"CLASS IMBALANCE: {code} accounts for {count}/{total_failures} "
                    f"({count/total_failures:.0%}) of failure annotations. "
                    f"Per-detector metrics may be unstable."
                )

    # Sample size check
    if report.n_human_reviewed + report.n_adjudicated < 10:
        report.warnings.append(
            f"LOW SAMPLE SIZE: Only {report.n_human_reviewed + report.n_adjudicated} "
            f"human-reviewed/adjudicated annotations. Recommend at least 30 for "
            f"stable evaluation metrics."
        )

    # Session ID format check
    n_short = sum(1 for prefix, recs in session_groups.items()
                  if any(len(r.get("session_id", "")) == 8 for r in recs))
    n_long = sum(1 for prefix, recs in session_groups.items()
                 if any(len(r.get("session_id", "")) > 8 for r in recs))
    n_mixed = sum(1 for prefix, recs in session_groups.items()
                  if (any(len(r.get("session_id", "")) == 8 for r in recs)
                      and any(len(r.get("session_id", "")) > 8 for r in recs)))
    if n_mixed > 0:
        report.warnings.append(
            f"MIXED SESSION IDS: {n_mixed} session(s) have both short (8-char) "
            f"and full UUID annotations. The loader handles this via prefix "
            f"matching, but consider normalizing to a single format."
        )

    return report


# ── Matching engine ──────────────────────────────────────────────────

def match_detections(
    detections: list[Detection],
    annotations: list[Annotation],
    match_window: int = 5,
) -> tuple[list[MatchResult], list[Annotation]]:
    """Match detections to annotations and classify as TP/FP/FN.

    Returns:
        (match_results, unmatched_annotations)
        match_results contains one entry per detection.
        unmatched_annotations are FNs (ground truth with no match).
    """
    results: list[MatchResult] = []
    matched_ann_ids: set[int] = set()  # indices into annotations list

    # Sort detections by confidence descending (greedy best-match-first)
    sorted_dets = sorted(
        enumerate(detections),
        key=lambda x: x[1].confidence,
        reverse=True,
    )

    for _det_idx, det in sorted_dets:
        best_match: Optional[tuple[int, str, float]] = None  # (ann_idx, match_type, weight)

        for ann_idx, ann in enumerate(annotations):
            if ann_idx in matched_ann_ids:
                continue
            if ann.is_latent:
                continue  # latent types handled separately
            if ann.trace_id != det.trace_id:
                continue

            step_diff = abs(det.onset_step - ann.onset_step) if ann.onset_step > 0 else 0

            if det.failure_name == ann.failure_name:
                if ann.onset_step == 0 or step_diff <= match_window:
                    # Full match: correct code, within step window
                    best_match = (ann_idx, "full", 1.0)
                    break  # perfect match, stop looking
                else:
                    # Partial: correct code, wrong step
                    if best_match is None or best_match[2] < 0.5:
                        best_match = (ann_idx, "partial", 0.5)
            elif det.caft_code == ann.caft_code:
                # Same CAFT code but different failure_name (shouldn't happen
                # in practice, but handle gracefully)
                if ann.onset_step == 0 or step_diff <= match_window:
                    if best_match is None or best_match[2] < 0.5:
                        best_match = (ann_idx, "partial", 0.5)

        if best_match is not None:
            ann_idx, match_type, weight = best_match
            matched_ann_ids.add(ann_idx)
            results.append(MatchResult(
                detection=det,
                annotation=annotations[ann_idx],
                match_type=match_type,
                tp_weight=weight,
            ))
        else:
            # No matching annotation → FP
            results.append(MatchResult(
                detection=det,
                annotation=None,
                match_type="fp",
                tp_weight=0.0,
            ))

    # Unmatched annotations → FN
    unmatched = [
        ann for i, ann in enumerate(annotations)
        if i not in matched_ann_ids and not ann.is_latent
    ]

    return results, unmatched


# ── Core evaluation ──────────────────────────────────────────────────

def compute_evaluation(
    annotations: list[Annotation],
    detections: list[Detection],
    mode: str,
    match_window: int = 5,
    bootstrap_n: int = 1000,
    skip_bootstrap: bool = False,
) -> EvalReport:
    """Compute full evaluation metrics for one ablation mode.

    Args:
        annotations: Ground-truth annotations.
        detections: Detector outputs (filtered by mode).
        mode: One of strict/loose/loose+llm/oracle.
        match_window: Step-window tolerance for matching.
        bootstrap_n: Number of bootstrap iterations.
        skip_bootstrap: Skip CI computation (for fast iteration).

    Returns:
        EvalReport with per-detector and aggregate metrics.
    """
    trace_ids = set(a.trace_id for a in annotations) | set(d.trace_id for d in detections)
    n_traces = len(trace_ids)

    # Match
    matches, unmatched_anns = match_detections(detections, annotations, match_window)

    # Count latent FNs separately
    latent_anns = [a for a in annotations if a.is_latent]
    latent_fn = len(latent_anns)

    # Per-detector accumulation
    detector_stats: dict[str, DetectorResult] = {}
    all_detector_names = set()
    for det in detections:
        all_detector_names.add(det.failure_name)
    for ann in annotations:
        if not ann.is_latent:
            all_detector_names.add(ann.failure_name)

    for name in all_detector_names:
        # Find caft_code from any detection or annotation
        code = ""
        for d in detections:
            if d.failure_name == name:
                code = d.caft_code
                break
        if not code:
            for a in annotations:
                if a.failure_name == name:
                    code = a.caft_code
                    break
        detector_stats[name] = DetectorResult(detector=name, caft_code=code)

    # Score matches
    for m in matches:
        det_name = m.detection.failure_name
        if det_name not in detector_stats:
            detector_stats[det_name] = DetectorResult(
                detector=det_name, caft_code=m.detection.caft_code,
            )
        ds = detector_stats[det_name]
        ds.candidates += 1

        if m.match_type in ("full", "partial"):
            ds.tp += m.tp_weight
        else:
            ds.fp += 1

        # LLM stats
        if m.detection.confirmed is True:
            ds.confirmed += 1
        elif m.detection.confirmed is False:
            ds.rejected += 1

    # Score unmatched annotations as FN
    for ann in unmatched_anns:
        name = ann.failure_name
        if name not in detector_stats:
            detector_stats[name] = DetectorResult(detector=name, caft_code=ann.caft_code)
        detector_stats[name].fn += 1

    # Latency stats
    latency_buckets: dict[str, list[float]] = defaultdict(list)
    llm_latency_buckets: dict[str, list[float]] = defaultdict(list)
    for det in detections:
        if det.latency_ms > 0:
            latency_buckets[det.failure_name].append(det.latency_ms)
        if det.llm_latency_ms > 0:
            llm_latency_buckets[det.failure_name].append(det.llm_latency_ms)

    for name, ds in detector_stats.items():
        lats = latency_buckets.get(name, [])
        ds.avg_latency_ms = sum(lats) / len(lats) if lats else 0.0
        llm_lats = llm_latency_buckets.get(name, [])
        ds.avg_llm_latency_ms = sum(llm_lats) / len(llm_lats) if llm_lats else 0.0

    per_detector = sorted(detector_stats.values(), key=lambda d: d.detector)

    # Macro averages (each detector weighted equally)
    active = [d for d in per_detector if d.tp + d.fp + d.fn > 0]
    macro_p = sum(d.precision for d in active) / len(active) if active else 0.0
    macro_r = sum(d.recall for d in active) / len(active) if active else 0.0
    macro_f1 = sum(d.f1 for d in active) / len(active) if active else 0.0

    # Micro averages (each detection weighted equally)
    total_tp = sum(d.tp for d in per_detector)
    total_fp = sum(d.fp for d in per_detector)
    total_fn = sum(d.fn for d in per_detector)
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (
        2 * micro_p * micro_r / (micro_p + micro_r)
        if (micro_p + micro_r) > 0 else 0.0
    )

    # Session-level binary: does this session have ANY detection/annotation?
    sessions_with_det = {d.trace_id for d in detections}
    sessions_with_ann = {a.trace_id for a in annotations if not a.is_latent}
    session_tp = len(sessions_with_det & sessions_with_ann)
    session_fp = len(sessions_with_det - sessions_with_ann)
    session_fn = len(sessions_with_ann - sessions_with_det)
    session_p = session_tp / (session_tp + session_fp) if (session_tp + session_fp) > 0 else 0.0
    session_r = session_tp / (session_tp + session_fn) if (session_tp + session_fn) > 0 else 0.0
    session_f1 = (
        2 * session_p * session_r / (session_p + session_r)
        if (session_p + session_r) > 0 else 0.0
    )

    # Candidates per trace
    candidates_per_trace = len(detections) / n_traces if n_traces > 0 else 0.0

    # LLM stats
    llm_confirmed = sum(1 for d in detections if d.confirmed is True)
    llm_rejected = sum(1 for d in detections if d.confirmed is False)
    llm_uncertain = sum(1 for d in detections if d.confirmed is None and mode in ("loose+llm",))
    llm_total = llm_confirmed + llm_rejected + llm_uncertain
    llm_confirmation_rate = (
        llm_confirmed / llm_total if llm_total > 0 else None
    ) if "llm" in mode else None

    # LLM agreement with ground truth
    llm_agreement = _compute_llm_agreement(
        detections, annotations, match_window
    ) if "llm" in mode else None

    # Latency per detector
    latency_dict = {}
    for name, ds in detector_stats.items():
        total = ds.avg_latency_ms + ds.avg_llm_latency_ms
        if total > 0:
            latency_dict[name] = round(total, 1)

    report = EvalReport(
        mode=mode,
        per_detector=per_detector,
        macro_p=macro_p,
        macro_r=macro_r,
        macro_f1=macro_f1,
        micro_p=micro_p,
        micro_r=micro_r,
        micro_f1=micro_f1,
        candidates_per_trace=candidates_per_trace,
        llm_confirmation_rate=llm_confirmation_rate,
        llm_agreement_with_gt=llm_agreement,
        latency_ms=latency_dict,
        session_p=session_p,
        session_r=session_r,
        session_f1=session_f1,
        n_traces=n_traces,
        n_annotations=len([a for a in annotations if not a.is_latent]),
        n_candidates=len(detections),
        latent_fn=latent_fn,
    )

    # Bootstrap CIs
    if not skip_bootstrap and n_traces >= 2:
        report.bootstrap_ci = bootstrap_ci(
            annotations, detections, match_window, n_iterations=bootstrap_n,
        )

    return report


def _compute_llm_agreement(
    detections: list[Detection],
    annotations: list[Annotation],
    match_window: int,
) -> float:
    """How often does the LLM decision agree with ground truth?

    For each detection that has an LLM decision (confirmed is not None):
    - If LLM confirmed and detection is TP → agree
    - If LLM rejected and detection is FP → agree
    - Otherwise → disagree
    """
    llm_dets = [d for d in detections if d.confirmed is not None]
    if not llm_dets:
        return 0.0

    # Run matching to know which are TP/FP
    matches, _ = match_detections(llm_dets, annotations, match_window)

    agreements = 0
    for m in matches:
        is_tp = m.match_type in ("full", "partial")
        if is_tp and m.detection.confirmed:
            agreements += 1
        elif not is_tp and not m.detection.confirmed:
            agreements += 1

    return agreements / len(matches) if matches else 0.0


# ── Bootstrap confidence intervals ──────────────────────────────────

def bootstrap_ci(
    annotations: list[Annotation],
    detections: list[Detection],
    match_window: int = 5,
    n_iterations: int = 1000,
    alpha: float = 0.05,
) -> dict[str, BootstrapCI]:
    """Compute bootstrap 95% CIs on macro P/R/F1.

    Resamples traces (not individual detections) with replacement.
    Uses numpy only (no scipy dependency).
    """
    rng = np.random.default_rng(42)

    trace_ids = sorted(set(a.trace_id for a in annotations) | set(d.trace_id for d in detections))
    n = len(trace_ids)
    if n < 2:
        return {}

    # Index annotations/detections by trace_id
    ann_by_trace: dict[str, list[Annotation]] = defaultdict(list)
    det_by_trace: dict[str, list[Detection]] = defaultdict(list)
    for a in annotations:
        ann_by_trace[a.trace_id].append(a)
    for d in detections:
        det_by_trace[d.trace_id].append(d)

    boot_p = np.zeros(n_iterations)
    boot_r = np.zeros(n_iterations)
    boot_f1 = np.zeros(n_iterations)
    boot_session_f1 = np.zeros(n_iterations)

    for i in range(n_iterations):
        sampled = rng.choice(trace_ids, size=n, replace=True)
        sample_anns: list[Annotation] = []
        sample_dets: list[Detection] = []
        for tid in sampled:
            sample_anns.extend(ann_by_trace.get(tid, []))
            sample_dets.extend(det_by_trace.get(tid, []))

        if not sample_anns and not sample_dets:
            continue

        report = compute_evaluation(
            sample_anns, sample_dets, mode="bootstrap",
            match_window=match_window, skip_bootstrap=True,
        )
        boot_p[i] = report.macro_p
        boot_r[i] = report.macro_r
        boot_f1[i] = report.macro_f1
        boot_session_f1[i] = report.session_f1

    lo = alpha / 2 * 100
    hi = (1 - alpha / 2) * 100

    # Point estimates from original data
    orig = compute_evaluation(
        annotations, detections, mode="point_est",
        match_window=match_window, skip_bootstrap=True,
    )

    return {
        "macro_precision": BootstrapCI(
            metric="macro_precision",
            point_estimate=orig.macro_p,
            ci_lower=float(np.percentile(boot_p, lo)),
            ci_upper=float(np.percentile(boot_p, hi)),
            n_iterations=n_iterations,
        ),
        "macro_recall": BootstrapCI(
            metric="macro_recall",
            point_estimate=orig.macro_r,
            ci_lower=float(np.percentile(boot_r, lo)),
            ci_upper=float(np.percentile(boot_r, hi)),
            n_iterations=n_iterations,
        ),
        "macro_f1": BootstrapCI(
            metric="macro_f1",
            point_estimate=orig.macro_f1,
            ci_lower=float(np.percentile(boot_f1, lo)),
            ci_upper=float(np.percentile(boot_f1, hi)),
            n_iterations=n_iterations,
        ),
        "session_f1": BootstrapCI(
            metric="session_f1",
            point_estimate=orig.session_f1,
            ci_lower=float(np.percentile(boot_session_f1, lo)),
            ci_upper=float(np.percentile(boot_session_f1, hi)),
            n_iterations=n_iterations,
        ),
    }


# ── McNemar's test ──────────────────────────────────────────────────

def mcnemar_test(
    annotations: list[Annotation],
    detections_a: list[Detection],
    detections_b: list[Detection],
    match_window: int = 5,
) -> ModeComparison:
    """McNemar's test comparing two modes on the same traces.

    Builds a 2x2 contingency table per annotation:
    - b01: mode_a wrong, mode_b right
    - b10: mode_a right, mode_b wrong
    Then chi2 = (|b01 - b10| - 1)^2 / (b01 + b10)  [with continuity correction]
    p-value from chi2(df=1) approximated via numpy.

    Returns ModeComparison with statistic and p-value.
    """
    # Get TP sets for each mode
    matches_a, _ = match_detections(detections_a, annotations, match_window)
    matches_b, _ = match_detections(detections_b, annotations, match_window)

    # For each non-latent annotation, check if each mode matched it
    ann_matched_a: set[tuple[str, str]] = set()  # (trace_id, failure_name)
    ann_matched_b: set[tuple[str, str]] = set()

    for m in matches_a:
        if m.annotation and m.match_type in ("full", "partial"):
            ann_matched_a.add((m.annotation.trace_id, m.annotation.failure_name))
    for m in matches_b:
        if m.annotation and m.match_type in ("full", "partial"):
            ann_matched_b.add((m.annotation.trace_id, m.annotation.failure_name))

    # Build contingency counts
    all_ann_keys = set()
    for a in annotations:
        if not a.is_latent:
            all_ann_keys.add((a.trace_id, a.failure_name))

    b01 = 0  # a wrong, b right
    b10 = 0  # a right, b wrong
    for key in all_ann_keys:
        a_right = key in ann_matched_a
        b_right = key in ann_matched_b
        if not a_right and b_right:
            b01 += 1
        elif a_right and not b_right:
            b10 += 1

    # McNemar statistic with continuity correction
    denom = b01 + b10
    if denom == 0:
        statistic = 0.0
        p_value = 1.0
    else:
        statistic = (abs(b01 - b10) - 1) ** 2 / denom
        # Approximate p-value from chi2(df=1) using normal CDF
        # chi2(df=1) ~ N(0,1)^2 → p = 2*(1 - Phi(sqrt(statistic)))
        z = np.sqrt(max(statistic, 0))
        # Normal CDF approximation (Abramowitz & Stegun 26.2.17)
        p_value = float(_normal_sf(z))

    # F1 comparison
    report_a = compute_evaluation(
        annotations, detections_a, "a", match_window, skip_bootstrap=True,
    )
    report_b = compute_evaluation(
        annotations, detections_b, "b", match_window, skip_bootstrap=True,
    )
    a_better = report_a.macro_f1 >= report_b.macro_f1

    return ModeComparison(
        mode_a="a",
        mode_b="b",
        mcnemar_statistic=round(statistic, 4),
        p_value=round(p_value, 4),
        significant=p_value < 0.05,
        a_better=a_better,
    )


def _normal_sf(z: float) -> float:
    """Survival function (1 - CDF) of standard normal, numpy only.

    Uses the complementary error function: sf(z) = 0.5 * erfc(z/sqrt(2))
    erfc approximated via Horner polynomial (Abramowitz & Stegun 7.1.26).
    """
    if z < 0:
        return 1.0 - _normal_sf(-z)

    # erfc(x) approximation for x >= 0
    x = z / np.sqrt(2)
    t = 1.0 / (1.0 + 0.3275911 * x)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 +
           t * (-1.453152027 + t * 1.061405429))))
    erfc_approx = poly * np.exp(-(x ** 2))
    return float(0.5 * erfc_approx)


# ── Mode comparison ──────────────────────────────────────────────────

def compare_modes(
    reports: dict[str, EvalReport],
    annotations: list[Annotation],
    detections_by_mode: dict[str, list[Detection]],
    match_window: int = 5,
) -> ComparisonTable:
    """Compare all modes pairwise and identify per-detector winners.

    Args:
        reports: {mode_name: EvalReport} for each mode.
        annotations: Ground-truth annotations.
        detections_by_mode: {mode_name: detections} for McNemar test.
        match_window: Step tolerance for matching.

    Returns:
        ComparisonTable with pairwise tests and per-detector winners.
    """
    mode_names = sorted(reports.keys())
    pairwise: list[ModeComparison] = []

    # Standard comparison pairs
    pairs = [
        ("strict", "loose+llm"),
        ("loose", "loose+llm"),
        ("loose+llm", "oracle"),
    ]

    for a, b in pairs:
        if a in detections_by_mode and b in detections_by_mode:
            comp = mcnemar_test(
                annotations,
                detections_by_mode[a],
                detections_by_mode[b],
                match_window,
            )
            comp.mode_a = a
            comp.mode_b = b
            pairwise.append(comp)

    # Per-detector winners
    per_detector_winners: dict[str, str] = {}
    all_detectors = set()
    for r in reports.values():
        for d in r.per_detector:
            if d.tp + d.fp + d.fn > 0:
                all_detectors.add(d.detector)

    for det_name in sorted(all_detectors):
        best_mode = ""
        best_f1 = -1.0
        for mode_name, r in reports.items():
            for d in r.per_detector:
                if d.detector == det_name and d.f1 > best_f1:
                    best_f1 = d.f1
                    best_mode = mode_name
        per_detector_winners[det_name] = best_mode

    return ComparisonTable(
        pairwise=pairwise,
        per_detector_winners=per_detector_winners,
    )


# ── Pretty-printing ─────────────────────────────────────────────────

def format_comparison_table(
    reports: dict[str, EvalReport],
    comparison: ComparisonTable,
    date_str: str = "",
) -> str:
    """Format a human-readable comparison table."""
    lines = []
    lines.append("AGENTDIAG ABLATION STUDY")
    lines.append("=" * 60)

    # Find any report for N stats
    any_report = next(iter(reports.values()))
    lines.append(
        f"Traces: {any_report.n_traces} ({any_report.mode} split)  "
        f"Annotations: {any_report.n_annotations}  "
        f"Date: {date_str}"
    )
    lines.append("")

    # Mode comparison table
    header = (
        f"{'Mode':<22} {'Precision':>9} {'Recall':>8} "
        f"{'F1':>6} {'micro-F1':>9} {'95% CI (F1)':>16}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    mode_order = ["strict", "loose", "loose+llm", "oracle"]
    for mode in mode_order:
        if mode not in reports:
            continue
        r = reports[mode]
        ci = r.bootstrap_ci.get("macro_f1")
        ci_str = (
            f"[{ci.ci_lower:.0%}, {ci.ci_upper:.0%}]"
            if ci else "N/A"
        )
        label = {
            "strict": "strict (rules)",
            "loose": "loose (candidates)",
            "loose+llm": "loose + LLM",
            "oracle": "oracle (ceiling)",
        }.get(mode, mode)
        lines.append(
            f"{label:<22} {r.macro_p:>8.0%} {r.macro_r:>7.0%} "
            f"{r.macro_f1:>5.0%} {r.micro_f1:>8.0%} {ci_str:>16}"
        )

    lines.append("")

    # Session-level binary metrics
    lines.append("Session-level (binary: any failure per session?):")
    for mode in mode_order:
        if mode not in reports:
            continue
        r = reports[mode]
        label = {
            "strict": "strict",
            "loose": "loose",
            "loose+llm": "loose+llm",
            "oracle": "oracle",
        }.get(mode, mode)
        lines.append(
            f"  {label:<16} P={r.session_p:.0%}  R={r.session_r:.0%}  F1={r.session_f1:.0%}"
        )
    lines.append("")

    # Pairwise comparisons
    if comparison.pairwise:
        lines.append("Pairwise comparisons (McNemar's test):")
        for c in comparison.pairwise:
            sig = "significant" if c.significant else "not significant"
            lines.append(f"  {c.mode_a} vs {c.mode_b}: p={c.p_value:.3f} ({sig})")
        lines.append("")

    # Per-detector winners
    if comparison.per_detector_winners:
        lines.append("Per-detector winners:")
        for det, mode in sorted(comparison.per_detector_winners.items()):
            r = reports.get(mode)
            if r:
                dr = next((d for d in r.per_detector if d.detector == det), None)
                if dr:
                    lines.append(
                        f"  {det}: {mode} (P={dr.precision:.0%} R={dr.recall:.0%})"
                    )
                else:
                    lines.append(f"  {det}: {mode}")
            else:
                lines.append(f"  {det}: {mode}")

    # Tier A (rules) vs Tier B (LLM) breakdown for the best LLM mode
    best_llm_mode = "loose+llm" if "loose+llm" in reports else None
    if not best_llm_mode:
        # Fall back to any mode with detections
        for m in mode_order:
            if m in reports:
                best_llm_mode = m
                break
    if best_llm_mode:
        r = reports[best_llm_mode]
        for tier_name, tier_set in [("Tier A (rules)", _TIER_A_DETECTORS),
                                     ("Tier B (LLM)", _TIER_B_DETECTORS)]:
            tier_dets = [d for d in r.per_detector if d.detector in tier_set and d.tp + d.fp + d.fn > 0]
            if tier_dets:
                tp = sum(d.precision for d in tier_dets) / len(tier_dets)
                tr = sum(d.recall for d in tier_dets) / len(tier_dets)
                tf = sum(d.f1 for d in tier_dets) / len(tier_dets)
                det_names = ", ".join(sorted(d.detector for d in tier_dets))
                lines.append(f"  {tier_name:<16} P={tp:.0%}  R={tr:.0%}  F1={tf:.0%}  [{det_names}]")
        lines.append("")

    return "\n".join(lines)
