#!/usr/bin/env python3
"""Run the CAFT detector ablation study across 4 modes.

Loads annotated traces, runs strict/loose/loose+llm/oracle modes,
computes P/R/F1 with bootstrap CIs, and outputs comparison tables.

Usage:
    python scripts/run_ablation.py \
        --annotations annotations/ground_truth_20.json \
        --split test \
        --output-dir results/ablation_$(date +%Y%m%d)

    # Fast iteration (no bootstrap, no LLM)
    python scripts/run_ablation.py \
        --annotations annotations/ground_truth_20.json \
        --no-bootstrap --modes strict loose oracle

    # Single detector
    python scripts/run_ablation.py \
        --annotations annotations/ground_truth_20.json \
        --detectors premature_termination

Also available via CLI:
    agentdiag evaluate --ablation --annotations labels.jsonl --split test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

# Add package to path for script invocation
sys.path.insert(0, str(Path(__file__).parent.parent))

from agentdiag.adapters.claude_code import ClaudeCodeExtractor
from agentdiag.caft.base import CaftDiagnosis
from agentdiag.caft.detectors import ALL_CAFT_DETECTORS, ALL_CAFT_DETECTORS_FULL, TIER_2_FAILURE_TYPES
from agentdiag.hta import HTAStateMachine
from agentdiag.metrics import (
    Annotation,
    ComparisonTable,
    Detection,
    EvalReport,
    bootstrap_ci,
    compare_modes,
    compute_evaluation,
    format_comparison_table,
    validate_annotations_jsonl,
)
from agentdiag.monitor import MonitorEngine


# ── Ground truth loading ─────────────────────────────────────────────

# Map failure_name → caft_code
_NAME_TO_CODE = {
    "step_repetition": "2.2",
    "context_loss": "2.1",
    "goal_drift": "2.4",
    "tool_thrashing": "3.1",
    "premature_termination": "5.4",
    "missing_verification": "5.3",
    "reasoning_action_mismatch": "6.4",
    "error_cascade": "4.2",
    "recovery_failure": "4.3",
    "analysis_paralysis": "3.4",
    "stall": "4.4",
    "token_explosion": "4.4",
    "tool_misuse": "4.1",
    "resource_exhaustion": "4.5",
}

# Reverse: caft_code → failure_name (first match wins)
_CODE_TO_NAME = {v: k for k, v in reversed(list(_NAME_TO_CODE.items()))}


def load_gt_session_ids(path: Path) -> set[str]:
    """Return ALL session IDs from a ground_truth_*.json, including clean traces.

    Clean traces (actual_failures=[]) are important for evaluation because
    any detector firing on a clean session is a false positive.  Previously
    these were invisible because load_annotations_from_gt skipped them.
    """
    with open(path) as f:
        gt = json.load(f)
    return {t.get("session_id", "") for t in gt.get("traces", []) if t.get("session_id")}


def load_annotations_from_gt(path: Path) -> list[Annotation]:
    """Load annotations from ground_truth_*.json format."""
    with open(path) as f:
        gt = json.load(f)

    annotations: list[Annotation] = []
    for trace in gt.get("traces", []):
        session_id = trace.get("session_id", "")
        actual = trace.get("actual_failures", [])
        details = trace.get("failure_details", [])

        if not actual:
            # Clean trace — no annotation needed (absence = no failure)
            continue

        if details:
            for d in details:
                # Derive failure_name: prefer explicit field, then
                # reverse-lookup from caft_code (always reliable), then
                # caft_name (may be Title Case or snake_case).
                fname = d.get("failure_name", "")
                if not fname:
                    fname = _CODE_TO_NAME.get(d.get("caft_code", ""), "")
                if not fname:
                    # Last resort: normalize caft_name to snake_case
                    fname = d.get("caft_name", "").lower().replace(" ", "_")
                # NOTE: onset_step in ground truth JSON is a JSONL line number,
                # NOT a TraceEvent index. Detectors report TraceEvent indices.
                # These are incomparable, so we set onset_step=0 to disable
                # step-window matching (same behavior as JSONL human annotations).
                annotations.append(Annotation(
                    trace_id=session_id,
                    failure_name=fname,
                    caft_code=d.get("caft_code", ""),
                    onset_step=0,
                ))
        else:
            for name in actual:
                annotations.append(Annotation(
                    trace_id=session_id,
                    failure_name=name,
                    caft_code=_NAME_TO_CODE.get(name, ""),
                    onset_step=0,
                ))

    return annotations


def _derive_failure_name(d: dict) -> str:
    """Derive failure_name from available fields in an annotation record.

    Priority: primary_caft_subtype > failure_name > annotator_id (for detector) > code lookup.
    """
    name = d.get("primary_caft_subtype") or d.get("failure_name")
    if name:
        return name
    # For detector annotations, annotator_id IS the detector/failure name
    if d.get("annotator_type") == "detector":
        return d.get("annotator_id", "")
    # Fall back to reverse-mapping from CAFT code
    code = d.get("primary_caft_code", d.get("caft_code", ""))
    if code:
        return _CODE_TO_NAME.get(code, code)
    return ""


def load_annotations_from_jsonl(path: Path) -> list[Annotation]:
    """Load ground-truth annotations from JSONL with authority rules.

    Authority hierarchy (higher overrides lower for the same session):
        adjudicated > human_reviewed > auto_labeled > unlabeled

    Rules:
        1. Only ``human_reviewed`` and ``adjudicated`` annotations are
           trusted as ground truth.
        2. ``auto_labeled`` (DRAFT) annotations are included BUT flagged
           with a warning — they should be reviewed.
        3. ``unlabeled`` detector-only annotations are NEVER used as
           ground truth.  Using them would be circular (evaluating
           detectors against their own predictions).
        4. A human ``has_failure=False`` (CLEAN) annotation **suppresses**
           any detector or auto annotation for the same session.  This
           prevents the detector-fires → becomes-ground-truth → matches-
           itself loop.
        5. Session IDs are canonicalized by 8-char prefix so short IDs
           (``e2eff792``) and full UUIDs (``e2eff792-fb22-...``) map to
           the same session.

    Returns:
        List of Annotation objects suitable for evaluation.  CLEAN
        sessions are represented by absence (no entry), which lets
        detector firings on them be counted as FP.
    """
    # Pass 1: read ALL records, grouped by canonical session prefix.
    _TRUSTED_STATUSES = {"human_reviewed", "adjudicated"}
    _DRAFT_STATUS = "auto_labeled"
    _AUTHORITY = {"adjudicated": 4, "human_reviewed": 3, "auto_labeled": 2, "unlabeled": 1}

    # prefix → list of raw dicts
    session_groups: dict[str, list[dict]] = defaultdict(list)
    n_skipped_detector = 0
    n_draft_warned = 0

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            sid = d.get("session_id", d.get("trace_id", ""))
            prefix = sid[:8]
            session_groups[prefix].append(d)

    # Pass 2: resolve each session's ground truth.
    annotations: list[Annotation] = []
    clean_sessions: set[str] = set()  # prefixes where human said CLEAN

    for prefix, records in session_groups.items():
        # Sort by authority (highest first)
        records.sort(
            key=lambda r: _AUTHORITY.get(r.get("label_status", "unlabeled"), 0),
            reverse=True,
        )

        # Find the highest-authority record for this session
        best = records[0]
        best_status = best.get("label_status", "unlabeled")
        best_type = best.get("annotator_type", "detector")

        # Rule 4: if the best human/adjudicated annotation says CLEAN,
        # the entire session is CLEAN — no failure annotations from any
        # lower layer.
        if (best_status in _TRUSTED_STATUSES
                and best_type in ("human", "adjudicated")
                and not best.get("has_failure", False)):
            clean_sessions.add(prefix)
            continue

        # Collect failure annotations from trusted sources only.
        seen_failures: set[str] = set()  # (failure_name,) dedup
        for d in records:
            status = d.get("label_status", "unlabeled")
            atype = d.get("annotator_type", "detector")

            if not d.get("has_failure", False):
                continue

            # Rule 3: skip unlabeled detector predictions entirely
            if status == "unlabeled" and atype == "detector":
                n_skipped_detector += 1
                continue

            # Rule 2: include DRAFTs but warn
            if status == _DRAFT_STATUS:
                n_draft_warned += 1

            # Rule 1: only trusted + DRAFT annotations survive
            if status not in (_TRUSTED_STATUSES | {_DRAFT_STATUS}):
                continue

            failure_name = _derive_failure_name(d)
            if not failure_name:
                continue

            # Dedup: one annotation per failure_name per session
            if failure_name in seen_failures:
                continue
            seen_failures.add(failure_name)

            # Use the canonical (longest) session ID from the group
            canonical_sid = max(
                (r.get("session_id", r.get("trace_id", "")) for r in records),
                key=len,
            )

            annotations.append(Annotation(
                trace_id=canonical_sid,
                failure_name=failure_name,
                caft_code=d.get("primary_caft_code", d.get("caft_code", "")),
                onset_step=d.get("onset_step", 0),
                is_latent=d.get("observable_vs_latent", "observable") == "latent",
            ))

    if n_skipped_detector > 0:
        print(f"  [GT-AUTHORITY] Excluded {n_skipped_detector} unlabeled detector "
              f"annotation(s) from ground truth (circular evaluation prevention)")
    if n_draft_warned > 0:
        print(f"  [GT-AUTHORITY] WARNING: {n_draft_warned} DRAFT (auto_labeled) "
              f"annotation(s) included — review before publishing results")
    if clean_sessions:
        print(f"  [GT-AUTHORITY] {len(clean_sessions)} session(s) marked CLEAN "
              f"by human reviewer (detector firings on these are FP)")

    return annotations


def load_annotations(path: Path) -> list[Annotation]:
    """Auto-detect format and load annotations."""
    if path.suffix == ".json":
        return load_annotations_from_gt(path)
    return load_annotations_from_jsonl(path)


def _load_clean_session_ids_from_jsonl(path: Path) -> set[str]:
    """Load canonical session IDs for CLEAN sessions from JSONL.

    A session is CLEAN if a human_reviewed or adjudicated annotation
    with has_failure=False exists.  These sessions must be included in
    the evaluation so detector firings on them count as FP.
    """
    _TRUSTED = {"human_reviewed", "adjudicated"}
    # prefix → longest session ID seen
    prefix_canonical: dict[str, str] = {}
    clean_prefixes: set[str] = set()

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            sid = d.get("session_id", d.get("trace_id", ""))
            prefix = sid[:8]
            if len(sid) > len(prefix_canonical.get(prefix, "")):
                prefix_canonical[prefix] = sid
            status = d.get("label_status", "unlabeled")
            atype = d.get("annotator_type", "detector")
            if (status in _TRUSTED
                    and atype in ("human", "adjudicated")
                    and not d.get("has_failure", False)):
                clean_prefixes.add(prefix)

    return {prefix_canonical.get(p, p) for p in clean_prefixes}


# ── LLM cache ───────────────────────────────────────────────────────

@dataclass
class LLMCacheEntry:
    trace_id: str
    detector: str
    candidate_step: int
    confirmed: bool
    confidence: float
    reasoning: str
    latency_ms: float
    tokens: int


class LLMCache:
    """File-backed LLM decision cache to avoid redundant API calls."""

    def __init__(self, path: Path):
        self._path = path
        self._cache: dict[tuple[str, str, int], LLMCacheEntry] = {}
        self._load()

    def _load(self):
        if self._path.exists():
            with open(self._path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    key = (d["trace_id"], d["detector"], d["candidate_step"])
                    self._cache[key] = LLMCacheEntry(**d)

    def get(self, trace_id: str, detector: str, step: int) -> LLMCacheEntry | None:
        return self._cache.get((trace_id, detector, step))

    def put(self, entry: LLMCacheEntry):
        key = (entry.trace_id, entry.detector, entry.candidate_step)
        self._cache[key] = entry
        with open(self._path, "a") as f:
            f.write(json.dumps(asdict(entry)) + "\n")

    def __len__(self):
        return len(self._cache)


# ── Trace runner ─────────────────────────────────────────────────────

def run_detectors_on_traces(
    session_ids: list[str],
    detectors: list,
    traces_root: Path,
    detector_filter: set[str] | None = None,
    verbose: bool = False,
    progress_callback=None,
) -> dict[str, list[tuple[CaftDiagnosis, float]]]:
    """Run detectors on traces and return {session_id: [(diagnosis, latency_ms)]}."""
    extractor = ClaudeCodeExtractor()
    if progress_callback:
        progress_callback({"type": "discovering", "traces_root": str(traces_root)})
    all_sessions = extractor.discover(traces_root, min_lines=5)
    session_map = {s.session_id: s for s in all_sessions}

    if verbose:
        print(f"  Discovered {len(all_sessions)} sessions in {traces_root}")
        det_names = [d.name for d in detectors]
        print(f"  Active detectors: {det_names}")

    if progress_callback:
        progress_callback({
            "type": "scanning_start",
            "discovered": len(all_sessions),
            "to_scan": len(session_ids),
        })

    results: dict[str, list[tuple[CaftDiagnosis, float]]] = {}
    _skipped: list[str] = []

    for sid_idx, sid in enumerate(session_ids):
        session = session_map.get(sid)
        if session is None:
            # Try prefix match
            matches = [s for s in all_sessions if s.session_id.startswith(sid)]
            if matches:
                session = matches[0]
                if verbose:
                    print(f"  {sid[:10]}: prefix-matched → {session.session_id[:16]}...")
            else:
                _skipped.append(sid)
                results[sid] = []
                if progress_callback:
                    progress_callback({
                        "type": "session_scan", "session_id": sid,
                        "session_idx": sid_idx + 1,
                        "total_sessions": len(session_ids),
                        "n_events": 0, "detections": [], "skipped": True,
                    })
                continue

        try:
            events = extractor.parse_session(session)
        except Exception as exc:
            if verbose:
                print(f"  {sid[:10]}: PARSE ERROR: {exc}")
            results[sid] = []
            if progress_callback:
                progress_callback({
                    "type": "session_scan", "session_id": sid,
                    "session_idx": sid_idx + 1,
                    "total_sessions": len(session_ids),
                    "n_events": 0, "detections": [], "skipped": True,
                })
            continue

        if not events:
            results[sid] = []
            if progress_callback:
                progress_callback({
                    "type": "session_scan", "session_id": sid,
                    "session_idx": sid_idx + 1,
                    "total_sessions": len(session_ids),
                    "n_events": 0, "detections": [], "skipped": True,
                })
            continue

        # Run through MonitorEngine
        t0 = time.time()
        engine = MonitorEngine(
            goal=f"Session {sid[:8]}",
            detectors=detectors,
        )
        for event in events:
            engine.push(event)

        # Post-processing: re-evaluate detectors against the complete trace.
        #
        # 1. RETRACTABLE detectors (recovery_failure): retract if the
        #    full-trace check returns None — their progress filters need
        #    the full trace to work correctly.
        # 2. RECONFIRMABLE detectors (stall): update confidence to the
        #    full-trace value. Early incremental firing computes IQR on
        #    few events → lower confidence → misses auto-confirm threshold.
        #    Full-trace IQR is more reliable. Never retract (a stall at
        #    step 10 is real even if the agent recovered by step 200).
        _RETRACTABLE = {"recovery_failure"}
        _RECONFIRMABLE = {"stall"}
        hta_state = engine._hta.state
        retract = set()
        for d in engine._diagnoses:
            det_obj = next(
                (det for det in detectors if det.name == d.failure_name), None,
            )
            if det_obj is None:
                continue
            if d.failure_name in _RETRACTABLE:
                full_diag = det_obj.check(events, hta_state)
                if full_diag is None:
                    retract.add(d.failure_name)
            elif d.failure_name in _RECONFIRMABLE:
                full_diag = det_obj.check(events, hta_state)
                if full_diag is not None and full_diag.confidence >= d.confidence:
                    d.confidence = full_diag.confidence
                    # Also update force_llm_review from full-trace result.
                    # Early firing may set the flag (sparse stalls) but the
                    # full trace shows enough stall events to auto-confirm.
                    # Uses >= so equal-confidence cases (both 1.0) still
                    # update the flag from the full-trace check.
                    if hasattr(full_diag, "force_llm_review"):
                        d.force_llm_review = full_diag.force_llm_review
        if retract:
            engine._diagnoses = [
                d for d in engine._diagnoses
                if d.failure_name not in retract
            ]

        elapsed_ms = (time.time() - t0) * 1000.0
        n_diag = max(len(engine.state.diagnoses), 1)

        diagnoses = []
        for d in engine.state.diagnoses:
            if detector_filter and d.failure_name not in detector_filter:
                continue
            diagnoses.append((d, elapsed_ms / n_diag))

        results[sid] = diagnoses

        if verbose:
            if diagnoses:
                diag_strs = [f"{d.failure_name}@{d.at_step}(c={d.confidence:.1f})"
                             for d, _ in diagnoses]
                print(f"  {sid[:10]}: {len(events)} events → {diag_strs}")
            else:
                print(f"  {sid[:10]}: {len(events)} events → (no detections)")

        # Emit per-session progress for web dashboard
        if progress_callback:
            det_names = [d.failure_name for d, _ in diagnoses]
            progress_callback({
                "type": "session_scan",
                "session_id": sid,
                "session_idx": sid_idx + 1,
                "total_sessions": len(session_ids),
                "n_events": len(events),
                "detections": det_names,
            })

    if _skipped and verbose:
        print(f"  SKIPPED (trace not found): {[s[:10] for s in _skipped]}")

    return results


def diagnoses_to_detections(
    results: dict[str, list[tuple[CaftDiagnosis, float]]],
) -> list[Detection]:
    """Convert MonitorEngine output to Detection objects."""
    detections = []
    for sid, diags in results.items():
        for diag, latency in diags:
            detections.append(Detection(
                trace_id=sid,
                failure_name=diag.failure_name,
                caft_code=diag.caft_code,
                onset_step=diag.at_step,
                confidence=diag.confidence,
                latency_ms=latency,
                evidence=diag.evidence,
                force_llm_review=getattr(diag, "force_llm_review", False),
            ))
    return detections


def dedup_detections(detections: list[Detection]) -> list[Detection]:
    """Keep only the highest-confidence detection per (trace_id, failure_name).

    Refireable detectors (stall, error_cascade) can fire at multiple steps
    during incremental push, but evaluation should count at most one detection
    per detector per session.
    """
    best: dict[tuple[str, str], Detection] = {}
    for d in detections:
        key = (d.trace_id, d.failure_name)
        if key not in best or d.confidence > best[key].confidence:
            best[key] = d
    return list(best.values())


# Detectors whose confirmation should suppress premature_termination
# for the same session, since PT is typically a symptom of these failures.
_PT_SUPPRESSORS = {"stall", "error_cascade"}


def suppress_symptomatic_pt(detections: list[Detection]) -> list[Detection]:
    """Suppress premature_termination when a root-cause detector co-fires.

    When stall or error_cascade is confirmed in a session, any co-occurring
    premature_termination is likely a symptom (session ended poorly because
    of the infrastructure failure, not because the agent chose to stop).
    Removing the symptomatic PT reduces false positives without losing signal.
    """
    # Find sessions that have a confirmed root-cause detector
    sessions_with_root_cause: set[str] = set()
    for d in detections:
        if d.failure_name in _PT_SUPPRESSORS:
            sessions_with_root_cause.add(d.trace_id)

    # Keep everything except PT in those sessions
    return [
        d for d in detections
        if not (d.failure_name == "premature_termination"
                and d.trace_id in sessions_with_root_cause)
    ]


# ── Oracle mode ──────────────────────────────────────────────────────

def apply_oracle_filter(
    detections: list[Detection],
    annotations: list[Annotation],
    match_window: int = 5,
) -> list[Detection]:
    """Filter detections using ground truth (perfect LLM).

    Keep a detection only if it matches an annotation on (trace_id,
    failure_name).  Step-window matching is intentionally NOT applied
    here — the oracle is the ceiling metric and should reflect what a
    perfect confirmation layer could achieve.  Step-window evaluation
    is handled separately by compute_evaluation / match_detections.
    """
    ann_keys = {(a.trace_id, a.failure_name) for a in annotations}

    filtered = []
    for det in detections:
        if (det.trace_id, det.failure_name) in ann_keys:
            filtered.append(Detection(
                trace_id=det.trace_id,
                failure_name=det.failure_name,
                caft_code=det.caft_code,
                onset_step=det.onset_step,
                confidence=det.confidence,
                confirmed=True,
                latency_ms=det.latency_ms,
            ))

    return filtered


# ── LLM confirmation mode ───────────────────────────────────────────

def apply_llm_confirmation(
    detections: list[Detection],
    session_events: dict[str, list],
    session_hta: dict[str, object],
    llm_cache: LLMCache | None = None,
    llm_log_path: Path | None = None,
    context_store=None,
    progress_callback=None,
    annotations: list | None = None,
) -> list[Detection]:
    """Run LLM confirmation on candidate detections.

    Uses cache to avoid redundant calls. Logs all decisions.
    If context_store is provided, queries OpenViking for similar past cases
    and records confirmation results (mirrors MonitorEngine pattern).
    """
    from agentdiag.caft.confirm import confirm_diagnosis_sync, is_llm_available, AUTOCONFIRM_THRESHOLD
    from agentdiag.caft.base import CaftSeverity

    # Build ground-truth lookup for tagging events with TP/FP/TN/FN
    _gt_keys: set[tuple[str, str]] = set()
    if annotations:
        for a in annotations:
            _gt_keys.add((a.trace_id, a.failure_name))
            _gt_keys.add((a.trace_id[:8], a.failure_name))

    def _gt_label(trace_id: str, failure_name: str, confirmed: bool) -> str:
        """Return TP/FP/TN/FN based on LLM decision vs ground truth."""
        has_ann = (trace_id, failure_name) in _gt_keys or (trace_id[:8], failure_name) in _gt_keys
        if confirmed and has_ann:
            return "TP"
        elif confirmed and not has_ann:
            return "FP"
        elif not confirmed and has_ann:
            return "FN"
        else:
            return "TN"

    confirmed_dets = []

    for det in detections:
        # Apply FP rate discount from historical feedback
        if context_store is not None:
            try:
                from agentdiag.caft.base import CaftDiagnosis as _CD, CaftSeverity as _CS
                _adj = _CD(
                    caft_code=det.caft_code, caft_category="",
                    failure_name=det.failure_name, severity=_CS.WARNING,
                    confidence=det.confidence, description="", evidence={},
                    at_step=det.onset_step, remediation="",
                )
                context_store.adjust_diagnosis_confidence(_adj)
                if _adj.confidence != det.confidence:
                    if progress_callback:
                        progress_callback({
                            "type": "fp_adjustment",
                            "session_id": det.trace_id,
                            "failure_name": det.failure_name,
                            "original": round(det.confidence, 3),
                            "adjusted": round(_adj.confidence, 3),
                        })
                    det = Detection(
                        trace_id=det.trace_id, failure_name=det.failure_name,
                        caft_code=det.caft_code, onset_step=det.onset_step,
                        confidence=_adj.confidence, confirmed=det.confirmed,
                        latency_ms=det.latency_ms,
                    )
            except Exception:
                pass

        # Auto-confirm high-confidence detections FIRST, before checking
        # cache.  Auto-confirm is based on rule confidence — the cache may
        # contain weaker LLM verdicts that should NOT override a strong rule.
        if det.confidence >= AUTOCONFIRM_THRESHOLD and not det.force_llm_review:
            print(f"    [{det.trace_id[:10]}] {det.failure_name}: AUTO-CONFIRM "
                  f"(confidence={det.confidence:.2f} >= {AUTOCONFIRM_THRESHOLD})")
            if progress_callback:
                progress_callback({
                    "type": "auto_confirm",
                    "session_id": det.trace_id,
                    "failure_name": det.failure_name,
                    "confidence": round(det.confidence, 3),
                    "gt_match": _gt_label(det.trace_id, det.failure_name, True),
                })
            new_det = Detection(
                trace_id=det.trace_id,
                failure_name=det.failure_name,
                caft_code=det.caft_code,
                onset_step=det.onset_step,
                confidence=det.confidence,
                confirmed=True,
                latency_ms=det.latency_ms,
            )
            confirmed_dets.append(new_det)
            if llm_cache:
                llm_cache.put(LLMCacheEntry(
                    trace_id=det.trace_id,
                    detector=det.failure_name,
                    candidate_step=det.onset_step,
                    confirmed=True,
                    confidence=det.confidence,
                    reasoning="Auto-confirmed (high confidence)",
                    latency_ms=0.0,
                    tokens=0,
                ))
            continue

        # Check cache for LLM verdict (only for candidates that didn't
        # auto-confirm above — these need LLM review).
        if llm_cache:
            cached = llm_cache.get(det.trace_id, det.failure_name, det.onset_step)
            if cached:
                verdict = "CONFIRMED" if cached.confirmed else "REJECTED"
                print(f"    [{det.trace_id[:10]}] {det.failure_name}: {verdict} "
                      f"(cached, confidence={cached.confidence:.2f}) "
                      f"reason: {(cached.reasoning or '')[:80]}")
                new_det = Detection(
                    trace_id=det.trace_id,
                    failure_name=det.failure_name,
                    caft_code=det.caft_code,
                    onset_step=det.onset_step,
                    confidence=cached.confidence,
                    confirmed=cached.confirmed,
                    latency_ms=det.latency_ms,
                    llm_latency_ms=cached.latency_ms,
                    llm_tokens=cached.tokens,
                )
                if cached.confirmed:
                    confirmed_dets.append(new_det)
                continue

        # Call LLM for medium/low confidence candidates
        if not is_llm_available():
            print(f"    [{det.trace_id[:10]}] {det.failure_name}: LLM UNAVAILABLE "
                  f"(confidence={det.confidence:.2f}, "
                  f"{'kept with discount' if det.confidence >= 0.5 else 'DROPPED'})")
            # Keep high-confidence candidates with discounted confidence
            # (matches MonitorEngine behavior which keeps uncertain candidates)
            if det.confidence >= 0.5:
                new_det = Detection(
                    trace_id=det.trace_id,
                    failure_name=det.failure_name,
                    caft_code=det.caft_code,
                    onset_step=det.onset_step,
                    confidence=det.confidence * 0.7,
                    confirmed=True,
                    latency_ms=det.latency_ms,
                )
                confirmed_dets.append(new_det)
                if llm_cache:
                    llm_cache.put(LLMCacheEntry(
                        trace_id=det.trace_id,
                        detector=det.failure_name,
                        candidate_step=det.onset_step,
                        confirmed=True,
                        confidence=det.confidence * 0.7,
                        reasoning="LLM unavailable; kept with discounted confidence",
                        latency_ms=0.0,
                        tokens=0,
                    ))
            continue

        # Build a CaftDiagnosis stub for the confirm call.
        # NOTE: evidence is mostly empty — passing real detector evidence
        # caused the LLM to over-rationalize and reject legitimate findings.
        stub = CaftDiagnosis(
            caft_code=det.caft_code,
            caft_category="",
            failure_name=det.failure_name,
            severity=CaftSeverity.WARNING,
            confidence=det.confidence,
            description="",
            evidence={},
            at_step=det.onset_step,
            remediation="",
        )

        events = session_events.get(det.trace_id, [])
        hta = session_hta.get(det.trace_id)

        if not events or hta is None:
            if det.confidence >= 0.5:
                new_det = Detection(
                    trace_id=det.trace_id,
                    failure_name=det.failure_name,
                    caft_code=det.caft_code,
                    onset_step=det.onset_step,
                    confidence=det.confidence * 0.7,
                    confirmed=True,
                    latency_ms=det.latency_ms,
                )
                confirmed_dets.append(new_det)
                if llm_cache:
                    llm_cache.put(LLMCacheEntry(
                        trace_id=det.trace_id,
                        detector=det.failure_name,
                        candidate_step=det.onset_step,
                        confirmed=True,
                        confidence=det.confidence * 0.7,
                        reasoning="Missing session data; kept with discounted confidence",
                        latency_ms=0.0,
                        tokens=0,
                    ))
            continue

        # Query OpenViking for similar past cases (mirrors MonitorEngine pattern)
        context_cases = []
        if context_store is not None:
            try:
                context_cases = context_store.find_similar_failures(
                    diagnosis=stub, limit=3,
                )
            except Exception:
                pass

        if progress_callback:
            progress_callback({
                "type": "llm_start",
                "session_id": det.trace_id,
                "failure_name": det.failure_name,
                "call_type": "confirm",
                "context_cases_count": len(context_cases),
            })

        print(f"    [{det.trace_id[:10]}] {det.failure_name}: calling LLM "
              f"(confidence={det.confidence:.2f}, step={det.onset_step})...")
        t0 = time.time()
        result = confirm_diagnosis_sync(stub, events, hta, context_cases=context_cases)
        llm_ms = (time.time() - t0) * 1000.0

        verdict = "CONFIRMED" if result.confirmed else "REJECTED"
        print(f"    [{det.trace_id[:10]}] {det.failure_name}: LLM → {verdict} "
              f"(confidence={result.confidence:.2f}, {llm_ms:.0f}ms) "
              f"reason: {(result.reasoning or '')[:100]}")

        if progress_callback:
            progress_callback({
                "type": "llm_result",
                "session_id": det.trace_id,
                "failure_name": det.failure_name,
                "call_type": "confirm",
                "confirmed": result.confirmed,
                "confidence": round(result.confidence, 3),
                "reasoning": (result.reasoning or "")[:500],
                "latency_ms": round(llm_ms, 1),
                "context_cases_count": len(context_cases),
                "gt_match": _gt_label(det.trace_id, det.failure_name, result.confirmed),
            })

        # Record confirmation result in OpenViking
        if context_store is not None:
            try:
                context_store.record_confirmation(candidate=stub, result=result)
            except Exception:
                pass

        new_det = Detection(
            trace_id=det.trace_id,
            failure_name=det.failure_name,
            caft_code=det.caft_code,
            onset_step=det.onset_step,
            confidence=result.confidence,
            confirmed=result.confirmed,
            latency_ms=det.latency_ms,
            llm_latency_ms=llm_ms,
        )

        if result.confirmed:
            confirmed_dets.append(new_det)

        if llm_cache:
            llm_cache.put(LLMCacheEntry(
                trace_id=det.trace_id,
                detector=det.failure_name,
                candidate_step=det.onset_step,
                confirmed=result.confirmed,
                confidence=result.confidence,
                reasoning=result.reasoning,
                latency_ms=llm_ms,
                tokens=0,
            ))

        if llm_log_path:
            with open(llm_log_path, "a") as f:
                f.write(json.dumps({
                    "trace_id": det.trace_id,
                    "detector": det.failure_name,
                    "step": det.onset_step,
                    "confirmed": result.confirmed,
                    "confidence": result.confidence,
                    "reasoning": result.reasoning,
                    "latency_ms": round(llm_ms, 1),
                    "status": result.status,
                }) + "\n")

    return confirmed_dets


# ── Tier 2: session-end assessment ────────────────────────────────────

def run_tier2_assessments(
    session_ids: list[str],
    session_events: dict[str, list],
    session_hta: dict[str, object],
    skip_pt_sessions: set[str] | None = None,
    llm_cache: LLMCache | None = None,
    llm_log: Path | None = None,
    verbose: bool = False,
    context_store=None,
    progress_callback=None,
    annotations: list | None = None,
) -> list[Detection]:
    """Run Tier 2 session-end assessment on all sessions.

    Makes TWO binary LLM calls per session (PT + GoalDrift) instead of
    one 5-way classification. Binary prompts are simpler for the LLM and
    give each question dedicated attention.

    Args:
        session_ids: All session IDs to assess.
        session_events: {sid: [TraceEvent, ...]} for each session.
        session_hta: {sid: HTAState} for each session.
        skip_pt_sessions: Sessions that already have confirmed PT from Tier 1.5.
        llm_cache: Cache for LLM decisions (keys: tier2_pt / tier2_gd).
        llm_log: Path to append LLM decision log.
        verbose: Print per-session details.
        context_store: Optional OpenViking ContextStore for similar-case retrieval.

    Returns:
        List of Detection objects for PT and GoalDrift findings.
    """
    from agentdiag.caft.confirm import assess_pt_sync, assess_gd_sync

    # Build ground-truth lookup for tagging events
    _gt_keys: set[tuple[str, str]] = set()
    if annotations:
        for a in annotations:
            _gt_keys.add((a.trace_id, a.failure_name))
            _gt_keys.add((a.trace_id[:8], a.failure_name))

    def _gt_label(trace_id: str, failure_name: str, confirmed: bool) -> str:
        has_ann = (trace_id, failure_name) in _gt_keys or (trace_id[:8], failure_name) in _gt_keys
        if confirmed and has_ann:
            return "TP"
        elif confirmed and not has_ann:
            return "FP"
        elif not confirmed and has_ann:
            return "FN"
        else:
            return "TN"

    skip_pt = skip_pt_sessions or set()
    detections: list[Detection] = []

    for sid in session_ids:
        events = session_events.get(sid)
        hta = session_hta.get(sid)
        if not events or hta is None:
            continue

        # ── PT assessment ──────────────────────────────────────────
        if sid not in skip_pt:
            pt_cached = llm_cache.get(sid, "tier2_pt", -1) if llm_cache else None
            if pt_cached:
                pt_v = "YES" if pt_cached.confirmed else "NO"
                print(f"    [{sid[:10]}] Tier2-PT: {pt_v} "
                      f"(cached, c={pt_cached.confidence:.2f})")
                if pt_cached.confirmed and pt_cached.confidence >= 0.60:
                    detections.append(Detection(
                        trace_id=sid, failure_name="premature_termination",
                        caft_code="5.4", onset_step=0,
                        confidence=pt_cached.confidence, confirmed=True,
                    ))
            else:
                # Query similar past PT cases from OpenViking
                pt_context = []
                if context_store is not None:
                    try:
                        from agentdiag.caft.base import CaftDiagnosis, CaftSeverity
                        pt_stub = CaftDiagnosis(
                            caft_code="5.4", caft_category="",
                            failure_name="premature_termination",
                            severity=CaftSeverity.WARNING, confidence=0.5,
                            description="Premature termination check",
                            evidence={}, at_step=0, remediation="",
                        )
                        pt_context = context_store.find_similar_failures(pt_stub, limit=3)
                    except Exception:
                        pass

                if progress_callback:
                    progress_callback({
                        "type": "llm_start",
                        "session_id": sid,
                        "failure_name": "premature_termination",
                        "call_type": "tier2_pt",
                        "context_cases_count": len(pt_context),
                    })

                print(f"    [{sid[:10]}] Tier2-PT: calling LLM...")
                t0 = time.time()
                pt_result = assess_pt_sync(events, hta, context_cases=pt_context)
                pt_ms = (time.time() - t0) * 1000.0

                pt_v = "YES" if pt_result.premature_termination else "NO"
                print(f"    [{sid[:10]}] Tier2-PT: LLM → {pt_v} "
                      f"(c={pt_result.confidence:.2f}, {pt_ms:.0f}ms) "
                      f"reason: {(pt_result.reasoning or '')[:100]}")

                if progress_callback:
                    progress_callback({
                        "type": "llm_result",
                        "session_id": sid,
                        "failure_name": "premature_termination",
                        "call_type": "tier2_pt",
                        "confirmed": pt_result.premature_termination,
                        "confidence": round(pt_result.confidence, 3),
                        "reasoning": (pt_result.reasoning or "")[:500],
                        "latency_ms": round(pt_ms, 1),
                        "context_cases_count": len(pt_context),
                        "gt_match": _gt_label(sid, "premature_termination", pt_result.premature_termination),
                    })

                if verbose:
                    print(f"    Tier2-PT {sid[:10]}: PT={pt_result.premature_termination} "
                          f"(c={pt_result.confidence:.2f})")

                if llm_cache:
                    llm_cache.put(LLMCacheEntry(
                        trace_id=sid, detector="tier2_pt", candidate_step=-1,
                        confirmed=pt_result.premature_termination,
                        confidence=pt_result.confidence,
                        reasoning=pt_result.reasoning,
                        latency_ms=pt_ms, tokens=0,
                    ))
                if llm_log:
                    with open(llm_log, "a") as f:
                        f.write(json.dumps({
                            "trace_id": sid, "detector": "tier2_pt", "step": -1,
                            "premature_termination": pt_result.premature_termination,
                            "confidence": pt_result.confidence,
                            "reasoning": pt_result.reasoning,
                            "latency_ms": round(pt_ms, 1),
                        }) + "\n")

                if pt_result.premature_termination and pt_result.confidence >= 0.60:
                    detections.append(Detection(
                        trace_id=sid, failure_name="premature_termination",
                        caft_code="5.4", onset_step=0,
                        confidence=pt_result.confidence, confirmed=True,
                    ))

        # ── GoalDrift assessment ───────────────────────────────────
        gd_cached = llm_cache.get(sid, "tier2_gd", -1) if llm_cache else None
        if gd_cached:
            gd_v = "YES" if gd_cached.confirmed else "NO"
            print(f"    [{sid[:10]}] Tier2-GD: {gd_v} "
                  f"(cached, c={gd_cached.confidence:.2f})")
            if gd_cached.confirmed and gd_cached.confidence >= 0.60:
                detections.append(Detection(
                    trace_id=sid, failure_name="goal_drift",
                    caft_code="2.4", onset_step=0,
                    confidence=gd_cached.confidence, confirmed=True,
                ))
        else:
            # Query similar past GoalDrift cases from OpenViking
            gd_context = []
            if context_store is not None:
                try:
                    from agentdiag.caft.base import CaftDiagnosis, CaftSeverity
                    gd_stub = CaftDiagnosis(
                        caft_code="2.4", caft_category="",
                        failure_name="goal_drift",
                        severity=CaftSeverity.WARNING, confidence=0.5,
                        description="Goal drift check",
                        evidence={}, at_step=0, remediation="",
                    )
                    gd_context = context_store.find_similar_failures(gd_stub, limit=3)
                except Exception:
                    pass

            if progress_callback:
                progress_callback({
                    "type": "llm_start",
                    "session_id": sid,
                    "failure_name": "goal_drift",
                    "call_type": "tier2_gd",
                    "context_cases_count": len(gd_context),
                })

            print(f"    [{sid[:10]}] Tier2-GD: calling LLM...")
            t0 = time.time()
            gd_result = assess_gd_sync(events, hta, context_cases=gd_context)
            gd_ms = (time.time() - t0) * 1000.0

            gd_v = "YES" if gd_result.goal_drift else "NO"
            print(f"    [{sid[:10]}] Tier2-GD: LLM → {gd_v} "
                  f"(c={gd_result.confidence:.2f}, {gd_ms:.0f}ms) "
                  f"reason: {(gd_result.reasoning or '')[:100]}")

            if progress_callback:
                progress_callback({
                    "type": "llm_result",
                    "session_id": sid,
                    "failure_name": "goal_drift",
                    "call_type": "tier2_gd",
                    "confirmed": gd_result.goal_drift,
                    "confidence": round(gd_result.confidence, 3),
                    "reasoning": (gd_result.reasoning or "")[:500],
                    "latency_ms": round(gd_ms, 1),
                    "context_cases_count": len(gd_context),
                    "gt_match": _gt_label(sid, "goal_drift", gd_result.goal_drift),
                })

            if verbose:
                print(f"    Tier2-GD {sid[:10]}: GD={gd_result.goal_drift} "
                      f"(c={gd_result.confidence:.2f})")

            if llm_cache:
                llm_cache.put(LLMCacheEntry(
                    trace_id=sid, detector="tier2_gd", candidate_step=-1,
                    confirmed=gd_result.goal_drift,
                    confidence=gd_result.confidence,
                    reasoning=gd_result.reasoning,
                    latency_ms=gd_ms, tokens=0,
                ))
            if llm_log:
                with open(llm_log, "a") as f:
                    f.write(json.dumps({
                        "trace_id": sid, "detector": "tier2_gd", "step": -1,
                        "goal_drift": gd_result.goal_drift,
                        "confidence": gd_result.confidence,
                        "reasoning": gd_result.reasoning,
                        "latency_ms": round(gd_ms, 1),
                    }) + "\n")

            if gd_result.goal_drift and gd_result.confidence >= 0.60:
                detections.append(Detection(
                    trace_id=sid, failure_name="goal_drift",
                    caft_code="2.4", onset_step=0,
                    confidence=gd_result.confidence, confirmed=True,
                ))

    if verbose:
        n_pt = sum(1 for d in detections if d.failure_name == "premature_termination")
        n_gd = sum(1 for d in detections if d.failure_name == "goal_drift")
        print(f"  Tier 2 results: {n_pt} PT, {n_gd} GoalDrift "
              f"from {len(session_ids)} sessions")

    return detections


# ── Main ablation runner ─────────────────────────────────────────────

def run_ablation(
    annotations_path: Path,
    traces_root: Path,
    output_dir: Path,
    split: str | None = None,
    splits_file: Path | None = None,
    llm_provider: str | None = None,
    match_window: int = 5,
    bootstrap_n: int = 1000,
    skip_bootstrap: bool = False,
    detector_filter: set[str] | None = None,
    modes: list[str] | None = None,
    dry_run: bool = False,
    verbose: bool = False,
    context_db: str | None = None,
    progress_callback=None,
) -> dict[str, EvalReport]:
    """Run all ablation modes and produce comparison.

    Args:
        annotations_path: Ground truth file (.json or .jsonl).
        traces_root: Root directory for trace JSONL files.
        output_dir: Directory for output files.
        split: Filter traces by split.
        splits_file: Path to splits.json.
        llm_provider: LLM provider for loose+llm mode.
        match_window: Step tolerance for matching.
        bootstrap_n: Bootstrap iterations.
        skip_bootstrap: Skip CI computation.
        detector_filter: Only run specific detectors.
        modes: Subset of modes to run.
        dry_run: Show what would be evaluated without running.

    Returns:
        {mode_name: EvalReport}
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if progress_callback:
        progress_callback({"type": "setup", "step": "initializing"})

    # Initialize OpenViking context store (optional)
    context_store = None
    if context_db:
        from agentdiag.context import get_context_store
        context_store = get_context_store(context_db)
        if context_store:
            print(f"  OpenViking context store: {context_db}")
        else:
            print("  Warning: Could not initialize context store", file=sys.stderr)

    if progress_callback:
        progress_callback({"type": "setup", "step": "loading_annotations"})

    # Pre-flight annotation quality check (JSONL only)
    if annotations_path.suffix == ".jsonl":
        qa = validate_annotations_jsonl(str(annotations_path))
        if qa.errors:
            print(f"\n{'=' * 60}")
            print("ANNOTATION QUALITY CHECK — ERRORS")
            print(f"{'=' * 60}")
            for err in qa.errors:
                print(f"  ERROR: {err}")
            print(f"\n  The new ground-truth loader will exclude circular/untrusted "
                  f"annotations automatically, but these errors indicate data "
                  f"quality issues that should be fixed at the source.")
            print(f"{'=' * 60}\n")
        if qa.warnings:
            print(f"\n  Annotation warnings:")
            for w in qa.warnings:
                print(f"    WARNING: {w}")
            print()
        if progress_callback:
            progress_callback({
                "type": "annotation_qa",
                "report": qa.to_dict(),
            })

    # Load annotations
    annotations = load_annotations(annotations_path)
    if not annotations and not annotations_path.suffix == ".json":
        print(f"No annotations found in {annotations_path}", file=sys.stderr)
        return {}

    # Get session IDs from annotations + clean sessions.
    # CLEAN sessions must be included so detector firings on them are counted as FP.
    annotation_sids = set(a.trace_id for a in annotations)
    if annotations_path.suffix == ".json":
        all_gt_sids = load_gt_session_ids(annotations_path)
        session_ids = sorted(annotation_sids | all_gt_sids)
    else:
        # For JSONL: also include clean sessions from the same file.
        clean_sids = _load_clean_session_ids_from_jsonl(annotations_path)
        session_ids = sorted(annotation_sids | clean_sids)
    if verbose:
        n_clean = len(set(session_ids) - annotation_sids)
        print(f"\nLoaded {len(annotations)} annotations across "
              f"{len(annotation_sids)} sessions from {annotations_path.name}"
              f" (+{n_clean} clean sessions for FP evaluation)")

    # Filter by split
    if split and splits_file and splits_file.exists():
        from agentdiag.splits import SplitManager
        sm = SplitManager(splits_file)
        split_ids = set(sm.get_traces(split))
        # Build prefix set for matching short IDs (e.g., "e21636e6") against
        # long UUIDs in the split (e.g., "e21636e6-7f54-4ec6-...").
        split_prefixes = {sid[:8] for sid in split_ids}
        pre_filter = len(session_ids)
        session_ids = [s for s in session_ids
                       if s in split_ids or s[:8] in split_prefixes]
        annotations = [a for a in annotations
                       if a.trace_id in split_ids or a.trace_id[:8] in split_prefixes]
        if verbose:
            print(f"Split filter ({split}): {pre_filter} → {len(session_ids)} sessions, "
                  f"{len(annotations)} annotations "
                  f"(split has {len(split_ids)} IDs)")

    # Deduplicate session IDs: short-prefix IDs (e.g. "e2eff792") and
    # their full UUID counterparts (e.g. "e2eff792-fb22-4f3c-...") refer
    # to the same underlying session.  Keep only the longest (canonical)
    # ID from each group to avoid double-processing and inflated FP counts.
    _prefix_groups: dict[str, list[str]] = defaultdict(list)
    for sid in session_ids:
        _prefix_groups[sid[:8]].append(sid)
    canonical_ids = sorted(max(group, key=len) for group in _prefix_groups.values())
    if len(canonical_ids) < len(session_ids):
        n_dupes = len(session_ids) - len(canonical_ids)
        print(f"Deduplicated {n_dupes} session ID(s) "
              f"({len(session_ids)} → {len(canonical_ids)} unique)")
        # Remap annotations to use canonical IDs
        _prefix_to_canonical = {cid[:8]: cid for cid in canonical_ids}
        seen_ann: set[tuple[str, str]] = set()
        deduped_anns: list[Annotation] = []
        for a in annotations:
            canon = _prefix_to_canonical.get(a.trace_id[:8], a.trace_id)
            key = (canon, a.failure_name)
            if key not in seen_ann:
                seen_ann.add(key)
                deduped_anns.append(Annotation(
                    trace_id=canon,
                    failure_name=a.failure_name,
                    caft_code=a.caft_code,
                    onset_step=a.onset_step,
                    is_latent=getattr(a, "is_latent", False),
                ))
        annotations = deduped_anns
        session_ids = canonical_ids

    if dry_run or verbose:
        print(f"\n{'DRY RUN — ' if dry_run else ''}Session/Annotation Summary "
              f"({len(session_ids)} sessions, {len(annotations)} annotations):")
        for sid in session_ids:
            anns = [a for a in annotations if a.trace_id == sid]
            names = [f"{a.failure_name}@{a.onset_step}" for a in anns]
            print(f"  {sid[:10]}: {names or ['(clean)']}")
        if dry_run:
            return {}

    all_modes = modes or ["strict", "loose", "loose+llm", "oracle"]

    # Set LLM provider if specified
    if llm_provider:
        os.environ["AGENTDIAG_LLM_PROVIDER"] = llm_provider

    if progress_callback:
        progress_callback({
            "type": "setup", "step": "ready",
            "total_sessions": len(session_ids),
            "total_annotations": len(annotations),
            "modes": all_modes,
        })

    reports: dict[str, EvalReport] = {}
    detections_by_mode: dict[str, list[Detection]] = {}

    def _log_matches(mode_name: str, dets: list[Detection]):
        """Verbose per-session match breakdown."""
        if not verbose:
            return
        from agentdiag.metrics import match_detections
        matches, unmatched = match_detections(dets, annotations, match_window)
        # Group by session
        all_sids = sorted(set(
            [d.trace_id for d in dets] + [a.trace_id for a in annotations]
        ))
        for sid in all_sids:
            anns = [a for a in annotations if a.trace_id == sid]
            ann_names = {a.failure_name for a in anns}
            det_for_sid = [d for d in dets if d.trace_id == sid]
            det_names = [d.failure_name for d in det_for_sid]
            tps = [m for m in matches
                   if m.detection.trace_id == sid and m.match_type in ("full", "partial")]
            fps = [m for m in matches
                   if m.detection.trace_id == sid and m.match_type == "fp"]
            fns = [a for a in unmatched if a.trace_id == sid]
            tp_names = [m.detection.failure_name for m in tps]
            fp_names = [m.detection.failure_name for m in fps]
            fn_names = [a.failure_name for a in fns]
            status_parts = []
            if tp_names:
                status_parts.append(f"TP:{tp_names}")
            if fp_names:
                status_parts.append(f"FP:{fp_names}")
            if fn_names:
                status_parts.append(f"FN:{fn_names}")
            if not status_parts:
                status_parts.append("(clean, no dets)")
            print(f"    {sid[:10]}: ann={list(ann_names)} det={det_names} → {' '.join(status_parts)}")

    # ── Mode 1: strict ─────────────────────────────
    if "strict" in all_modes:
        print("Running strict mode (production thresholds)...")
        if progress_callback:
            progress_callback({"type": "mode_start", "mode": "strict",
                               "total_sessions": len(session_ids)})
        results = run_detectors_on_traces(
            session_ids, list(ALL_CAFT_DETECTORS), traces_root, detector_filter,
            verbose=verbose, progress_callback=progress_callback,
        )
        dets = dedup_detections(diagnoses_to_detections(results))
        dets = suppress_symptomatic_pt(dets)
        report = compute_evaluation(
            annotations, dets, "strict", match_window, bootstrap_n, skip_bootstrap,
        )
        reports["strict"] = report
        detections_by_mode["strict"] = dets
        print(f"  → {len(dets)} detections, macro F1={report.macro_f1:.0%}")
        _log_matches("strict", dets)
        if progress_callback:
            progress_callback({"type": "mode_complete", "mode": "strict",
                               "report": report.to_dict(), "n_detections": len(dets)})

    # ── Mode 2: loose ──────────────────────────────
    if "loose" in all_modes:
        print("Running loose mode (candidate generator)...")
        if progress_callback:
            progress_callback({"type": "mode_start", "mode": "loose",
                               "total_sessions": len(session_ids)})
        results = run_detectors_on_traces(
            session_ids, list(ALL_CAFT_DETECTORS_FULL), traces_root, detector_filter,
            verbose=verbose, progress_callback=progress_callback,
        )
        dets_loose = dedup_detections(diagnoses_to_detections(results))
        report = compute_evaluation(
            annotations, dets_loose, "loose", match_window, bootstrap_n, skip_bootstrap,
        )
        reports["loose"] = report
        detections_by_mode["loose"] = dets_loose
        print(f"  → {len(dets_loose)} detections, macro F1={report.macro_f1:.0%}")
        _log_matches("loose", dets_loose)
        if progress_callback:
            progress_callback({"type": "mode_complete", "mode": "loose",
                               "report": report.to_dict(), "n_detections": len(dets_loose)})

    # ── Mode 3: loose+llm ─────────────────────────
    if "loose+llm" in all_modes:
        from agentdiag.caft.confirm import _get_provider, _get_model
        print(f"Running loose+llm mode (provider={_get_provider()}, model={_get_model()})...")
        if progress_callback:
            progress_callback({"type": "mode_start", "mode": "loose+llm",
                               "total_sessions": len(session_ids)})
        if "loose" not in detections_by_mode:
            results = run_detectors_on_traces(
                session_ids, list(ALL_CAFT_DETECTORS_FULL), traces_root, detector_filter,
                verbose=verbose, progress_callback=progress_callback,
            )
            dets_loose = dedup_detections(diagnoses_to_detections(results))
        else:
            dets_loose = detections_by_mode["loose"]

        llm_cache = LLMCache(output_dir / "llm_cache.jsonl")
        llm_log = output_dir / "llm_decisions.jsonl"

        # Enable full prompt+response tracing for debugging LLM decisions
        from agentdiag.caft.confirm import enable_llm_tracing
        enable_llm_tracing(output_dir / "llm_traces.jsonl")

        # V13: Parse events and HTA state for ALL sessions (Tier 2 needs
        # every session, not just those with rule candidates).
        parse_sids = set(session_ids)
        extractor = ClaudeCodeExtractor()
        all_sessions = extractor.discover(traces_root, min_lines=5)
        session_map = {s.session_id: s for s in all_sessions}

        session_events: dict[str, list] = {}
        session_hta: dict[str, object] = {}
        for sid in parse_sids:
            session = session_map.get(sid)
            if session is None:
                matches = [s for s in all_sessions if s.session_id.startswith(sid)]
                if matches:
                    session = matches[0]
            if session is None:
                continue
            try:
                events = extractor.parse_session(session)
                if not events:
                    continue
                hta_machine = HTAStateMachine()
                for event in events:
                    hta_state = hta_machine.push(event)
                session_events[sid] = events
                session_hta[sid] = hta_state
            except Exception:
                continue

        print(f"  Parsed {len(session_events)}/{len(parse_sids)} sessions for LLM context")

        # Tier 1 + 1.5: rule detections + LLM confirmation
        print(f"\n  ── Tier 1: LLM confirmation ({len(dets_loose)} candidates) ──")
        dets_llm = dedup_detections(apply_llm_confirmation(
            dets_loose, session_events, session_hta, llm_cache, llm_log,
            context_store=context_store,
            progress_callback=progress_callback,
            annotations=annotations,
        ))

        # Tier 2: binary PT + GoalDrift assessment per session
        print(f"\n  ── Tier 2: PT + GoalDrift assessment ({len(session_ids)} sessions) ──")
        tier1_pt_sessions = {d.trace_id for d in dets_llm
                             if d.failure_name == "premature_termination"}
        dets_tier2 = run_tier2_assessments(
            session_ids, session_events, session_hta,
            skip_pt_sessions=tier1_pt_sessions,
            llm_cache=llm_cache, llm_log=llm_log, verbose=verbose,
            context_store=context_store,
            progress_callback=progress_callback,
            annotations=annotations,
        )

        # Merge all tiers + suppress symptomatic PT
        dets_merged = dedup_detections(dets_llm + dets_tier2)
        dets_merged = suppress_symptomatic_pt(dets_merged)

        report = compute_evaluation(
            annotations, dets_merged, "loose+llm", match_window, bootstrap_n, skip_bootstrap,
        )
        reports["loose+llm"] = report
        detections_by_mode["loose+llm"] = dets_merged
        print(f"  → {len(dets_merged)} detections, macro F1={report.macro_f1:.0%}")
        _log_matches("loose+llm", dets_merged)
        if progress_callback:
            progress_callback({"type": "mode_complete", "mode": "loose+llm",
                               "report": report.to_dict(), "n_detections": len(dets_merged)})

    # ── Mode 4: oracle ─────────────────────────────
    if "oracle" in all_modes:
        print("Running oracle mode (perfect LLM)...")
        if progress_callback:
            progress_callback({"type": "mode_start", "mode": "oracle",
                               "total_sessions": len(session_ids)})
        if "loose" not in detections_by_mode:
            results = run_detectors_on_traces(
                session_ids, list(ALL_CAFT_DETECTORS_FULL), traces_root, detector_filter,
                verbose=verbose, progress_callback=progress_callback,
            )
            dets_loose = dedup_detections(diagnoses_to_detections(results))
        else:
            dets_loose = detections_by_mode["loose"]

        if verbose:
            # Show oracle matching details
            ann_keys = {(a.trace_id, a.failure_name) for a in annotations}
            det_keys = {(d.trace_id, d.failure_name) for d in dets_loose}
            print(f"  Oracle matching: {len(ann_keys)} annotation keys, "
                  f"{len(det_keys)} detection keys")
            matched_keys = ann_keys & det_keys
            missed_keys = ann_keys - det_keys
            extra_keys = det_keys - ann_keys
            if matched_keys:
                print(f"    MATCHED ({len(matched_keys)}): "
                      f"{[(k[0][:8], k[1]) for k in sorted(matched_keys)]}")
            if missed_keys:
                print(f"    MISSED ({len(missed_keys)}, annotated but not detected): "
                      f"{[(k[0][:8], k[1]) for k in sorted(missed_keys)]}")
            if extra_keys:
                print(f"    EXTRA ({len(extra_keys)}, detected but not annotated): "
                      f"{[(k[0][:8], k[1]) for k in sorted(extra_keys)]}")

        dets_oracle = apply_oracle_filter(dets_loose, annotations, match_window)

        # V13: Perfect Tier 2 — add any annotated PT/GoalDrift not caught by rules
        detected_keys = {(d.trace_id, d.failure_name) for d in dets_oracle}
        for a in annotations:
            if a.failure_name in TIER_2_FAILURE_TYPES:
                if (a.trace_id, a.failure_name) not in detected_keys:
                    dets_oracle.append(Detection(
                        trace_id=a.trace_id,
                        failure_name=a.failure_name,
                        caft_code=_NAME_TO_CODE.get(a.failure_name, ""),
                        onset_step=0,
                        confidence=1.0,
                        confirmed=True,
                    ))

        report = compute_evaluation(
            annotations, dets_oracle, "oracle", match_window, bootstrap_n, skip_bootstrap,
        )
        reports["oracle"] = report
        detections_by_mode["oracle"] = dets_oracle
        print(f"  → {len(dets_oracle)} detections, macro F1={report.macro_f1:.0%}")
        _log_matches("oracle", dets_oracle)
        if progress_callback:
            progress_callback({"type": "mode_complete", "mode": "oracle",
                               "report": report.to_dict(), "n_detections": len(dets_oracle)})

    # ── Comparison ─────────────────────────────────
    comparison = compare_modes(reports, annotations, detections_by_mode, match_window)

    # ── Write outputs ──────────────────────────────
    date_str = time.strftime("%Y-%m-%d")

    # comparison_table.json
    comp_data = {
        "date": date_str,
        "n_traces": len(session_ids),
        "n_annotations": len(annotations),
        "match_window": match_window,
        "modes": {name: r.to_dict() for name, r in reports.items()},
        "comparison": comparison.to_dict(),
    }
    with open(output_dir / "comparison_table.json", "w") as f:
        json.dump(comp_data, f, indent=2, default=str)

    # comparison_table.txt
    table_txt = format_comparison_table(reports, comparison, date_str)
    with open(output_dir / "comparison_table.txt", "w") as f:
        f.write(table_txt)
    print("\n" + table_txt)

    if progress_callback:
        progress_callback({
            "type": "ablation_complete",
            "reports": {n: r.to_dict() for n, r in reports.items()},
            "comparison_table": table_txt,
        })

    # per_detector_breakdown.json
    breakdown = {}
    for mode_name, r in reports.items():
        breakdown[mode_name] = {d.detector: d.to_dict() for d in r.per_detector}
    with open(output_dir / "per_detector_breakdown.json", "w") as f:
        json.dump(breakdown, f, indent=2)

    # bootstrap_distributions.json
    boot_data = {}
    for mode_name, r in reports.items():
        if r.bootstrap_ci:
            boot_data[mode_name] = {k: v.to_dict() for k, v in r.bootstrap_ci.items()}
    if boot_data:
        with open(output_dir / "bootstrap_distributions.json", "w") as f:
            json.dump(boot_data, f, indent=2)

    # Clean up context store
    if context_store is not None:
        try:
            context_store.close()
        except Exception:
            pass

    print(f"\nOutputs written to {output_dir}/")
    return reports


# ── Cross-validation ─────────────────────────────────────────────────

def run_cross_validation(
    annotations_path: Path,
    traces_root: Path,
    output_dir: Path,
    splits_file: Path,
    k: int = 5,
    match_window: int = 5,
    detector_filter: set[str] | None = None,
    modes: list[str] | None = None,
    verbose: bool = False,
) -> dict[str, dict]:
    """K-fold cross-validation on train+val sessions.

    Combines train+val, splits into K folds, runs all modes on each fold,
    reports mean/std F1 per mode.  Gives ~3x tighter CI than the small
    test set.

    Returns:
        {mode: {"mean_f1": float, "std_f1": float, "fold_f1s": list[float]}}
    """
    import random

    output_dir.mkdir(parents=True, exist_ok=True)
    all_modes = modes or ["strict", "loose", "oracle"]

    # Load annotations
    annotations = load_annotations(annotations_path)
    if not annotations:
        print("No annotations found.", file=sys.stderr)
        return {}

    # Get train+val session IDs
    sm = None
    if splits_file and splits_file.exists():
        from agentdiag.splits import SplitManager
        sm = SplitManager(splits_file)
        train_ids = set(sm.get_traces("train"))
        val_ids = set(sm.get_traces("val"))
        pool_ids = train_ids | val_ids
    else:
        # Fall back: use all annotated sessions
        pool_ids = set(a.trace_id for a in annotations)

    # Deduplicate prefixes
    _prefix_groups: dict[str, list[str]] = defaultdict(list)
    for sid in pool_ids:
        _prefix_groups[sid[:8]].append(sid)
    canonical_ids = sorted(max(group, key=len) for group in _prefix_groups.values())

    # Filter annotations to pool
    canonical_prefixes = {cid[:8] for cid in canonical_ids}
    _prefix_to_canonical = {cid[:8]: cid for cid in canonical_ids}
    pool_anns: list[Annotation] = []
    seen_ann: set[tuple[str, str]] = set()
    for a in annotations:
        if a.trace_id[:8] not in canonical_prefixes:
            continue
        canon = _prefix_to_canonical.get(a.trace_id[:8], a.trace_id)
        key = (canon, a.failure_name)
        if key not in seen_ann:
            seen_ann.add(key)
            pool_anns.append(Annotation(
                trace_id=canon,
                failure_name=a.failure_name,
                caft_code=a.caft_code,
                onset_step=a.onset_step,
                is_latent=getattr(a, "is_latent", False),
            ))

    # Get session IDs that have annotations (failures)
    failure_sids = sorted(set(a.trace_id for a in pool_anns))
    clean_sids = sorted(set(canonical_ids) - set(failure_sids))

    print(f"\nCross-validation: {len(canonical_ids)} sessions "
          f"({len(failure_sids)} with failures, {len(clean_sids)} clean), "
          f"{len(pool_anns)} annotations, K={k}")

    # Stratified K-fold: distribute failure sessions evenly across folds,
    # then fill with clean sessions.
    random.seed(42)
    random.shuffle(failure_sids)
    random.shuffle(clean_sids)

    folds: list[list[str]] = [[] for _ in range(k)]
    for i, sid in enumerate(failure_sids):
        folds[i % k].append(sid)
    for i, sid in enumerate(clean_sids):
        folds[(i + len(failure_sids)) % k].append(sid)

    print(f"Fold sizes: {[len(f) for f in folds]}")

    # Run each fold
    fold_results: dict[str, list[float]] = {m: [] for m in all_modes}

    for fold_idx in range(k):
        test_sids = set(folds[fold_idx])
        fold_anns = [a for a in pool_anns if a.trace_id in test_sids]

        if not fold_anns:
            print(f"  Fold {fold_idx+1}/{k}: no annotations in test set, skipping")
            continue

        print(f"\n  Fold {fold_idx+1}/{k}: {len(test_sids)} sessions, "
              f"{len(fold_anns)} annotations")

        fold_session_ids = sorted(test_sids)

        # Run modes
        if "strict" in all_modes or "loose" in all_modes or "oracle" in all_modes:
            # Run loose detectors (superset) once
            results_full = run_detectors_on_traces(
                fold_session_ids, list(ALL_CAFT_DETECTORS_FULL), traces_root,
                detector_filter, verbose=False,
            )
            dets_full = dedup_detections(diagnoses_to_detections(results_full))

            # Also run strict detectors
            results_strict = run_detectors_on_traces(
                fold_session_ids, list(ALL_CAFT_DETECTORS), traces_root,
                detector_filter, verbose=False,
            )
            dets_strict = dedup_detections(diagnoses_to_detections(results_strict))

        for mode in all_modes:
            if mode == "strict":
                dets = dets_strict
            elif mode == "loose":
                dets = dets_full
            elif mode == "oracle":
                dets = apply_oracle_filter(dets_full, fold_anns, match_window)
            else:
                continue  # skip loose+llm in CV (too slow / needs API key)

            report = compute_evaluation(
                fold_anns, dets, mode, match_window,
                bootstrap_n=0, skip_bootstrap=True,
            )
            fold_results[mode].append(report.macro_f1)
            if verbose:
                print(f"    {mode}: P={report.macro_precision:.0%} "
                      f"R={report.macro_recall:.0%} F1={report.macro_f1:.0%} "
                      f"({len(dets)} dets)")

    # Aggregate
    print(f"\n{'='*60}")
    print(f"Cross-validation results ({k}-fold on train+val)")
    print(f"{'='*60}")

    cv_results = {}
    for mode in all_modes:
        f1s = fold_results[mode]
        if not f1s:
            continue
        import numpy as _np
        mean_f1 = float(_np.mean(f1s))
        std_f1 = float(_np.std(f1s))
        cv_results[mode] = {
            "mean_f1": round(mean_f1, 3),
            "std_f1": round(std_f1, 3),
            "fold_f1s": [round(f, 3) for f in f1s],
        }
        print(f"  {mode:12s}: F1 = {mean_f1:.1%} +/- {std_f1:.1%}  "
              f"(folds: {[f'{f:.0%}' for f in f1s]})")

    # Write results
    with open(output_dir / "cross_validation.json", "w") as f:
        json.dump({
            "k": k,
            "n_sessions": len(canonical_ids),
            "n_annotations": len(pool_anns),
            "fold_sizes": [len(fold) for fold in folds],
            "results": cv_results,
        }, f, indent=2)

    print(f"\nCV results written to {output_dir}/cross_validation.json")
    return cv_results


# ── CLI entry point ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run CAFT detector ablation study",
    )
    parser.add_argument(
        "--annotations", required=True,
        help="Path to annotations file (.json or .jsonl)",
    )
    parser.add_argument(
        "--traces", default="~/.claude/projects",
        help="Root directory for trace JSONL files",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: results/ablation_YYYYMMDD)",
    )
    parser.add_argument("--split", default=None, help="Filter by split")
    parser.add_argument("--splits-file", default="splits.json", help="Splits file")
    parser.add_argument("--llm-provider", default=None, help="LLM provider for loose+llm")
    parser.add_argument("--match-window", type=int, default=5, help="Step tolerance")
    parser.add_argument("--bootstrap-n", type=int, default=1000, help="Bootstrap iterations")
    parser.add_argument("--no-bootstrap", action="store_true", help="Skip CI computation")
    parser.add_argument(
        "--detectors", default=None,
        help="Comma-separated detector names to run (subset)",
    )
    parser.add_argument(
        "--modes", default=None,
        help="Comma-separated modes to run (default: all)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show plan without running")
    parser.add_argument(
        "--cross-validate", type=int, default=0, metavar="K",
        help="Run K-fold cross-validation on train+val instead of test eval",
    )
    parser.add_argument(
        "--context-db", default=None,
        help="Path to OpenViking context database for case-based reasoning",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Detailed diagnostic output")
    args = parser.parse_args()

    annotations_path = Path(args.annotations)
    if not annotations_path.exists():
        print(f"Error: Annotations file not found: {annotations_path}", file=sys.stderr)
        sys.exit(1)

    traces_root = Path(args.traces).expanduser()
    output_dir = Path(args.output_dir or f"results/ablation_{time.strftime('%Y%m%d')}")

    detector_filter = set(args.detectors.split(",")) if args.detectors else None
    modes = args.modes.split(",") if args.modes else None
    splits_file = Path(args.splits_file) if args.splits_file else None

    if args.cross_validate > 0:
        run_cross_validation(
            annotations_path=annotations_path,
            traces_root=traces_root,
            output_dir=output_dir,
            splits_file=splits_file,
            k=args.cross_validate,
            match_window=args.match_window,
            detector_filter=detector_filter,
            modes=modes,
            verbose=args.verbose,
        )
    else:
        run_ablation(
            annotations_path=annotations_path,
            traces_root=traces_root,
            output_dir=output_dir,
            split=args.split,
            splits_file=splits_file,
            llm_provider=args.llm_provider,
            match_window=args.match_window,
            bootstrap_n=args.bootstrap_n,
            skip_bootstrap=args.no_bootstrap,
            detector_filter=detector_filter,
            modes=modes,
            dry_run=args.dry_run,
            verbose=args.verbose,
            context_db=args.context_db,
        )


if __name__ == "__main__":
    main()
