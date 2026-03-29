#!/usr/bin/env python3
"""Build evidence-based annotations for the ablation study.

This script:
1. Discovers real Claude Code traces
2. Runs MonitorEngine (ALL_CAFT_DETECTORS_FULL) on each trace
3. Enriches legacy ground_truth_50 annotations with onset_step from detectors
4. Selects 30 new unannotated traces (diverse sizes/projects)
5. Produces draft annotations for human review
6. Outputs a unified annotation JSONL + splits

Usage:
    python scripts/build_annotations.py \
        --legacy annotations/ground_truth_50.json \
        --output-dir annotations/ablation_ready \
        --n-new 30

    # Dry run (show what would be selected)
    python scripts/build_annotations.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agentdiag.adapters.claude_code import ClaudeCodeExtractor
from agentdiag.caft.detectors import ALL_CAFT_DETECTORS_FULL
from agentdiag.caft.base import CaftDiagnosis
from agentdiag.hta import Phase
from agentdiag.models import TraceEvent
from agentdiag.monitor import MonitorEngine
from agentdiag.annotation_models import (
    AnnotationRecord,
    AnnotatorType,
    LabelStatus,
    build_detector_annotation,
    build_human_annotation,
    from_ground_truth_file,
)
from agentdiag.annotation_store import AnnotationLedger


# ── Trace discovery ─────────────────────────────────────────────────

@dataclass
class TraceInfo:
    """Summary of a discovered trace file."""
    path: Path
    session_id: str  # full UUID
    prefix: str      # 8-char prefix
    size_bytes: int
    n_events: int = 0
    n_tool_calls: int = 0
    project: str = ""
    category: str = ""  # small/medium/large


def discover_traces(root: Path, min_size: int = 1000) -> list[TraceInfo]:
    """Find all UUID-named JSONL trace files."""
    uuid_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}')
    traces = []
    for p in root.rglob("*.jsonl"):
        if not uuid_pattern.match(p.name):
            continue
        size = p.stat().st_size
        if size < min_size:
            continue
        sid = p.stem
        # Extract project from path
        parts = p.parts
        project = ""
        try:
            idx = parts.index("projects")
            if idx + 1 < len(parts):
                project = parts[idx + 1]
        except ValueError:
            pass

        category = "small" if size < 50_000 else ("medium" if size < 500_000 else "large")
        traces.append(TraceInfo(
            path=p, session_id=sid, prefix=sid[:8],
            size_bytes=size, project=project, category=category,
        ))
    return traces


# ── Run detectors ───────────────────────────────────────────────────

@dataclass
class TraceEvidence:
    """Evidence collected from running detectors on a trace."""
    session_id: str
    path: str
    n_events: int
    n_tool_calls: int
    phases_seen: list[str]
    phase_counts: dict[str, int]
    n_regressions: int
    diagnoses: list[dict]  # serialized CaftDiagnosis dicts
    trust_score: float
    health: str
    # Tool usage summary
    top_tools: list[tuple[str, int]]
    # First/last events
    first_tool: str
    last_tool: str
    # Error info
    n_errors: int
    # Detection time
    detection_ms: float


def run_detectors_on_trace(trace: TraceInfo) -> TraceEvidence | None:
    """Run full MonitorEngine pipeline on a trace, collect evidence."""
    extractor = ClaudeCodeExtractor()
    try:
        # Discover sessions in this file's parent directory
        sessions = extractor.discover(trace.path.parent, min_lines=3)
        session = None
        for s in sessions:
            if s.session_id == trace.session_id:
                session = s
                break
        if session is None:
            # Try direct parse
            events = []
            with open(trace.path) as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        parsed = extractor._parse_message(data, len(events) + 1)
                        events.extend(parsed)
                    except (json.JSONDecodeError, Exception):
                        continue
        else:
            events = extractor.parse_session(session)
    except Exception as e:
        return None

    if not events or len(events) < 3:
        return None

    # Run through MonitorEngine with all detectors
    t0 = time.time()
    engine = MonitorEngine(
        goal=f"Session {trace.prefix}",
        detectors=list(ALL_CAFT_DETECTORS_FULL),
    )
    for event in events:
        engine.push(event)
    detection_ms = (time.time() - t0) * 1000.0

    state = engine.state
    hta = state.hta_state

    # Collect tool usage
    tool_counts = Counter()
    n_errors = 0
    for e in events:
        if e.tool:
            tool_counts[e.tool] += 1
        if not e.success:
            n_errors += 1

    return TraceEvidence(
        session_id=trace.session_id,
        path=str(trace.path),
        n_events=len(events),
        n_tool_calls=sum(1 for e in events if e.type == "tool_call"),
        phases_seen=sorted(set(
            hta.current_phase.label for _ in [1]
        ) if hta else []),
        phase_counts=dict(hta.phase_event_counts) if hta else {},
        n_regressions=hta.regression_count if hta else 0,
        diagnoses=[d.to_dict() for d in state.diagnoses],
        trust_score=round(state.trust_score, 3),
        health=state.health,
        top_tools=tool_counts.most_common(10),
        first_tool=events[0].tool or events[0].type if events else "",
        last_tool=events[-1].tool or events[-1].type if events else "",
        n_errors=n_errors,
        detection_ms=round(detection_ms, 1),
    )


# ── Annotation helpers ──────────────────────────────────────────────

def classify_trace_for_annotation(evidence: TraceEvidence) -> dict:
    """Produce draft annotation based on detector evidence.

    Returns a dict with fields for AnnotationRecord.
    This is a DRAFT that needs human review.
    """
    diagnoses = evidence.diagnoses

    if not diagnoses:
        return {
            "has_failure": False,
            "primary_caft_code": "",
            "onset_step": 0,
            "severity": 0,
            "confidence": 3,
            "rationale": (
                f"No detectors fired. {evidence.n_events} events, "
                f"{evidence.n_tool_calls} tool calls, "
                f"trust={evidence.trust_score}, health={evidence.health}. "
                f"Top tools: {[t for t, _ in evidence.top_tools[:5]]}."
            ),
        }

    # Use the highest-severity diagnosis
    best = max(diagnoses, key=lambda d: (
        {"critical": 3, "warning": 2, "info": 1}.get(d.get("severity", "info"), 0),
        d.get("confidence", 0),
    ))

    return {
        "has_failure": True,
        "primary_caft_code": best.get("caft_code", ""),
        "onset_step": best.get("at_step", 0),
        "severity": {"info": 2, "warning": 3, "critical": 4}.get(
            best.get("severity", "warning"), 3
        ),
        "confidence": 2,  # LOW confidence — needs human review
        "rationale": (
            f"DRAFT (detector output, needs review). "
            f"Primary: {best.get('failure_name', '')} at step {best.get('at_step', 0)}. "
            f"{best.get('description', '')} "
            f"Total detections: {len(diagnoses)}. "
            f"Other: {[d.get('failure_name') for d in diagnoses if d != best]}."
        ),
    }


# ── Enrich legacy annotations ──────────────────────────────────────

def enrich_legacy_with_onset(
    gt_traces: list[dict],
    all_traces: list[TraceInfo],
    evidence_cache: dict[str, TraceEvidence],
) -> list[dict]:
    """Add onset_step to legacy ground truth by matching detector output.

    For each failure in GT50, find the matching detector firing and
    copy its onset_step.
    """
    # Map 8-char prefix to full trace info
    prefix_to_trace = {}
    for t in all_traces:
        prefix_to_trace[t.prefix] = t

    enriched = []
    for gt_trace in gt_traces:
        sid = gt_trace["session_id"]
        failures = gt_trace.get("actual_failures", [])
        details = gt_trace.get("failure_details", [])

        if not failures:
            enriched.append(gt_trace)
            continue

        if details and all(d.get("onset_step", 0) > 0 for d in details):
            # Already has onset_step
            enriched.append(gt_trace)
            continue

        # Need to enrich — find detector evidence
        trace_info = prefix_to_trace.get(sid)
        if trace_info and trace_info.session_id in evidence_cache:
            evidence = evidence_cache[trace_info.session_id]
            new_details = []
            for name in failures:
                # Find matching detector firing
                matching = [
                    d for d in evidence.diagnoses
                    if d.get("failure_name") == name
                ]
                if matching:
                    best = max(matching, key=lambda d: d.get("confidence", 0))
                    new_details.append({
                        "failure_name": name,
                        "caft_code": best.get("caft_code", ""),
                        "onset_step": best.get("at_step", 0),
                        "severity": 3,
                        "confidence": 3,
                        "rationale": (
                            f"From GT50 verdict + detector onset. "
                            f"Detector: {best.get('description', '')}"
                        ),
                    })
                else:
                    # No detector matched — keep without onset
                    new_details.append({
                        "failure_name": name,
                        "caft_code": _name_to_code(name),
                        "onset_step": 0,
                        "severity": 3,
                        "confidence": 3,
                        "rationale": f"From GT50 verdict. No detector fired for {name}.",
                    })

            gt_trace = dict(gt_trace)
            gt_trace["failure_details"] = new_details
        enriched.append(gt_trace)

    return enriched


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
}


def _name_to_code(name: str) -> str:
    return _NAME_TO_CODE.get(name, "")


# ── Select new traces ──────────────────────────────────────────────

def select_new_traces(
    all_traces: list[TraceInfo],
    already_annotated: set[str],
    n: int = 30,
) -> list[TraceInfo]:
    """Select n diverse traces not already annotated.

    Targets: ~10 small, ~10 medium, ~10 large with project diversity.
    """
    candidates = [t for t in all_traces
                  if t.session_id not in already_annotated
                  and t.prefix not in already_annotated]

    # Sort by size within each category
    small = sorted([t for t in candidates if t.category == "small"],
                   key=lambda t: t.size_bytes, reverse=True)
    medium = sorted([t for t in candidates if t.category == "medium"],
                    key=lambda t: t.size_bytes, reverse=True)
    large = sorted([t for t in candidates if t.category == "large"],
                   key=lambda t: t.size_bytes, reverse=True)

    # Select with project diversity
    def select_diverse(pool, target):
        selected = []
        seen_projects = set()
        # First pass: one per project
        for t in pool:
            if t.project not in seen_projects and len(selected) < target:
                selected.append(t)
                seen_projects.add(t.project)
        # Second pass: fill remaining
        for t in pool:
            if t not in selected and len(selected) < target:
                selected.append(t)
        return selected

    n_small = min(len(small), n // 3)
    n_large = min(len(large), n // 3)
    n_medium = min(len(medium), n - n_small - n_large)

    selected = (
        select_diverse(small, n_small) +
        select_diverse(medium, n_medium) +
        select_diverse(large, n_large)
    )

    # If still short, fill from any category
    remaining = [t for t in candidates if t not in selected]
    while len(selected) < n and remaining:
        selected.append(remaining.pop(0))

    return selected[:n]


# ── Assign splits ───────────────────────────────────────────────────

def assign_splits(
    session_ids: list[str],
    n_train: int = 15,
    n_val: int = 8,
    n_test: int = 7,
) -> dict[str, list[str]]:
    """Deterministic split assignment.

    Uses sorted session IDs for reproducibility.
    """
    sorted_ids = sorted(session_ids)
    total = len(sorted_ids)

    # Scale if fewer traces
    if total < n_train + n_val + n_test:
        ratio_train = n_train / (n_train + n_val + n_test)
        ratio_val = n_val / (n_train + n_val + n_test)
        n_train = int(total * ratio_train)
        n_val = int(total * ratio_val)
        n_test = total - n_train - n_val

    return {
        "train": sorted_ids[:n_train],
        "val": sorted_ids[n_train:n_train + n_val],
        "test": sorted_ids[n_train + n_val:n_train + n_val + n_test],
    }


# ── Main pipeline ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build annotations for ablation study")
    parser.add_argument("--legacy", default="annotations/ground_truth_50.json",
                        help="Legacy ground truth file")
    parser.add_argument("--traces-root", default="~/.claude/projects",
                        help="Root directory for traces")
    parser.add_argument("--output-dir", default="annotations/ablation_ready",
                        help="Output directory")
    parser.add_argument("--n-new", type=int, default=30, help="Number of new traces to annotate")
    parser.add_argument("--dry-run", action="store_true", help="Show selection without processing")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    traces_root = Path(args.traces_root).expanduser()
    legacy_path = Path(args.legacy)

    print("=" * 60)
    print("AGENTDIAG ANNOTATION BUILDER")
    print("=" * 60)

    # ── Step 1: Discover all traces ──
    print("\n[1/7] Discovering traces...")
    all_traces = discover_traces(traces_root)
    print(f"  Found {len(all_traces)} UUID trace files")
    categories = Counter(t.category for t in all_traces)
    print(f"  Small: {categories['small']}, Medium: {categories['medium']}, Large: {categories['large']}")

    # ── Step 2: Load legacy annotations ──
    print("\n[2/7] Loading legacy annotations...")
    gt_data = None
    legacy_sids: set[str] = set()
    if legacy_path.exists():
        with open(legacy_path) as f:
            gt_data = json.load(f)
        legacy_sids = {t["session_id"] for t in gt_data["traces"]}
        n_fail = sum(1 for t in gt_data["traces"] if t.get("actual_failures"))
        print(f"  Loaded {len(legacy_sids)} legacy traces ({n_fail} with failures)")
    else:
        print(f"  Legacy file not found: {legacy_path}")

    # Also mark ledger_30 sessions as annotated
    ledger_path = Path("annotations/annotation_ledger_30.jsonl")
    ledger_sids: set[str] = set()
    if ledger_path.exists():
        with open(ledger_path) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                d = json.loads(line)
                sid = d.get("session_id", "")
                if sid: ledger_sids.add(sid)
        print(f"  Loaded {len(ledger_sids)} ledger-30 session IDs")

    already_annotated = legacy_sids | ledger_sids | {s[:8] for s in ledger_sids}

    # ── Step 3: Select new traces ──
    print(f"\n[3/7] Selecting {args.n_new} new traces...")
    new_traces = select_new_traces(all_traces, already_annotated, args.n_new)
    print(f"  Selected {len(new_traces)} new traces:")
    for cat in ["small", "medium", "large"]:
        n = sum(1 for t in new_traces if t.category == cat)
        print(f"    {cat}: {n}")
    projects = Counter(t.project for t in new_traces)
    print(f"  From {len(projects)} projects: {projects.most_common(5)}")

    if args.dry_run:
        print("\nDRY RUN — would process:")
        for t in new_traces:
            print(f"  {t.session_id[:12]}... {t.category:>6} {t.size_bytes:>10} bytes  {t.project[:40]}")
        return

    # ── Step 4: Run detectors on ALL traces ──
    print(f"\n[4/7] Running detectors on {len(new_traces)} new + legacy traces...")

    # Build lookup for legacy traces
    legacy_traces = []
    if gt_data:
        for gt_trace in gt_data["traces"]:
            sid = gt_trace["session_id"]
            matches = [t for t in all_traces if t.prefix == sid or t.session_id == sid]
            if matches:
                legacy_traces.append(matches[0])

    all_to_process = new_traces + legacy_traces
    evidence_cache: dict[str, TraceEvidence] = {}
    failed_traces: list[str] = []

    for i, trace in enumerate(all_to_process):
        label = "new" if trace in new_traces else "legacy"
        sys.stdout.write(f"\r  [{i+1}/{len(all_to_process)}] {trace.prefix} ({label})...")
        sys.stdout.flush()
        evidence = run_detectors_on_trace(trace)
        if evidence:
            evidence_cache[trace.session_id] = evidence
            trace.n_events = evidence.n_events
            trace.n_tool_calls = evidence.n_tool_calls
        else:
            failed_traces.append(trace.session_id)

    print(f"\n  Processed {len(evidence_cache)} traces, {len(failed_traces)} failures")

    # ── Step 5: Enrich legacy + create new annotations ──
    print("\n[5/7] Building annotations...")

    ledger = AnnotationLedger(output_dir / "annotation_ledger.jsonl")

    # 5a: Convert & enrich legacy
    if gt_data:
        enriched_traces = enrich_legacy_with_onset(
            gt_data["traces"], all_traces, evidence_cache,
        )
        enriched_gt = dict(gt_data)
        enriched_gt["traces"] = enriched_traces
        enriched_gt["date"] = time.strftime("%Y-%m-%d")
        enriched_gt["method"] += " + detector onset enrichment"

        # Save enriched GT
        with open(output_dir / "ground_truth_50_enriched.json", "w") as f:
            json.dump(enriched_gt, f, indent=2)

        # Convert to AnnotationRecords
        legacy_records = from_ground_truth_file(enriched_gt)
        n_added = ledger.add_many(legacy_records)
        print(f"  Legacy: {n_added} records added ({len(enriched_traces)} traces)")

        # Also add detector evidence for legacy traces
        for trace in legacy_traces:
            if trace.session_id in evidence_cache:
                ev = evidence_cache[trace.session_id]
                for d in ev.diagnoses:
                    from agentdiag.caft.base import CaftDiagnosis as _D, CaftSeverity as _S
                    diag = _D(
                        caft_code=d.get("caft_code", ""),
                        caft_category=d.get("caft_category", ""),
                        failure_name=d.get("failure_name", ""),
                        severity=_S(d.get("severity", "warning")),
                        confidence=d.get("confidence", 0.5),
                        description=d.get("description", ""),
                        evidence=d.get("evidence", {}),
                        at_step=d.get("at_step", 0),
                        remediation=d.get("remediation", ""),
                    )
                    det_rec = build_detector_annotation(
                        session_id=trace.session_id,
                        diagnosis=diag,
                    )
                    ledger.add(det_rec)

    # 5b: Create annotations for new traces
    n_new_failures = 0
    n_new_clean = 0
    for trace in new_traces:
        if trace.session_id not in evidence_cache:
            continue

        evidence = evidence_cache[trace.session_id]
        draft = classify_trace_for_annotation(evidence)

        # Add detector layer (all detections)
        for d in evidence.diagnoses:
            from agentdiag.caft.base import CaftDiagnosis as _D, CaftSeverity as _S
            diag = _D(
                caft_code=d.get("caft_code", ""),
                caft_category=d.get("caft_category", ""),
                failure_name=d.get("failure_name", ""),
                severity=_S(d.get("severity", "warning")),
                confidence=d.get("confidence", 0.5),
                description=d.get("description", ""),
                evidence=d.get("evidence", {}),
                at_step=d.get("at_step", 0),
                remediation=d.get("remediation", ""),
            )
            det_rec = build_detector_annotation(
                session_id=trace.session_id,
                diagnosis=diag,
            )
            ledger.add(det_rec)

        # Add draft human annotation
        human_rec = build_human_annotation(
            session_id=trace.session_id,
            annotator_id="build_annotations.py",
            has_failure=draft["has_failure"],
            primary_caft_code=draft["primary_caft_code"],
            onset_step=draft["onset_step"],
            severity=draft["severity"],
            confidence=draft["confidence"],
            rationale=draft["rationale"],
        )
        human_rec.label_status = LabelStatus.AUTO_LABELED.value  # DRAFT, not reviewed
        ledger.add(human_rec)

        if draft["has_failure"]:
            n_new_failures += 1
        else:
            n_new_clean += 1

    print(f"  New: {len(new_traces)} traces ({n_new_failures} with detections, {n_new_clean} clean)")

    # ── Step 6: Assign splits ──
    print("\n[6/7] Assigning splits...")

    # All session IDs in the ledger
    all_session_ids = sorted(ledger.get_sessions())
    splits = assign_splits(all_session_ids, n_train=15, n_val=8, n_test=7)

    # If we have more than 30, adjust split sizes proportionally
    total = len(all_session_ids)
    if total > 30:
        n_test = max(7, int(total * 0.23))
        n_val = max(8, int(total * 0.27))
        n_train = total - n_test - n_val
        splits = assign_splits(all_session_ids, n_train, n_val, n_test)

    splits_data = {
        "train": splits["train"],
        "val": splits["val"],
        "test": splits["test"],
        "metadata": {
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total": total,
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
        },
    }
    with open(output_dir / "splits.json", "w") as f:
        json.dump(splits_data, f, indent=2)
    print(f"  train: {len(splits['train'])}, val: {len(splits['val'])}, test: {len(splits['test'])}")

    # ── Step 7: Save evidence + summary ──
    print("\n[7/7] Saving outputs...")

    # Save evidence for all traces
    evidence_data = {}
    for sid, ev in evidence_cache.items():
        evidence_data[sid] = {
            "session_id": ev.session_id,
            "n_events": ev.n_events,
            "n_tool_calls": ev.n_tool_calls,
            "n_diagnoses": len(ev.diagnoses),
            "trust_score": ev.trust_score,
            "health": ev.health,
            "diagnoses": ev.diagnoses,
            "top_tools": ev.top_tools,
            "n_errors": ev.n_errors,
            "detection_ms": ev.detection_ms,
        }
    with open(output_dir / "trace_evidence.json", "w") as f:
        json.dump(evidence_data, f, indent=2, default=str)

    # Output combined annotation file for run_ablation.py.
    # Uses canonical (deduplicated) session IDs and writes ALL annotation
    # layers per session so the ground-truth loader can apply authority rules.
    #
    # Previously this wrote only get_best_label() per session, which missed
    # the human CLEAN annotations (has_failure=False) that the loader needs
    # to suppress detector-only ground truth.
    canonical_sessions = ledger.get_sessions()  # already deduplicated
    combined_anns = []
    seen_dedup: set[tuple[str, str, str]] = set()  # (prefix, annotator_type, failure_name)
    for sid in sorted(canonical_sessions):
        records = ledger.get_for_session(sid)
        for rec in records:
            # Normalize session_id to canonical form
            key = (sid[:8], rec.annotator_type, rec.primary_caft_code if rec.has_failure else "_CLEAN_")
            if key in seen_dedup:
                continue
            seen_dedup.add(key)
            d = rec.to_dict()
            # Ensure canonical session ID
            d["session_id"] = sid
            d["trace_id"] = sid
            combined_anns.append(d)

    with open(output_dir / "annotations.jsonl", "w") as f:
        for ann in combined_anns:
            f.write(json.dumps(ann, default=str) + "\n")
    print(f"  annotations.jsonl: {len(combined_anns)} records "
          f"({len(canonical_sessions)} canonical sessions)")

    # Summary
    stats = ledger.stats()
    print(f"\n{'=' * 60}")
    print("ANNOTATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total records: {stats['total_records']}")
    print(f"Unique sessions: {stats['unique_sessions']}")
    print(f"By annotator type: {stats['by_annotator_type']}")
    print(f"By label status: {stats['by_label_status']}")
    print(f"Failure types: {stats['by_failure_type']}")
    print(f"Gold count: {stats['gold_count']}")
    print(f"Trainable count: {stats['trainable_count']}")

    # Failure statistics
    human_recs = ledger.get_by_type("human")
    failure_sessions = set(r.effective_session_id for r in human_recs if r.has_failure)
    clean_sessions = set(r.effective_session_id for r in human_recs if not r.has_failure)
    print(f"\nFailure rate: {len(failure_sessions)}/{len(failure_sessions | clean_sessions)} "
          f"({len(failure_sessions) / max(len(failure_sessions | clean_sessions), 1):.0%})")

    # Per-detector coverage
    detector_recs = ledger.get_by_type("detector")
    det_coverage = Counter(r.primary_caft_name or r.primary_caft_code for r in detector_recs if r.has_failure)
    print(f"\nDetector candidate coverage:")
    for name, cnt in det_coverage.most_common():
        print(f"  {name}: {cnt}")

    print(f"\nOutputs written to {output_dir}/:")
    print(f"  annotation_ledger.jsonl  ({len(ledger)} records)")
    print(f"  annotations.jsonl        (best-label per session, for run_ablation.py)")
    print(f"  splits.json              ({len(splits_data['train'])} train / {len(splits_data['val'])} val / {len(splits_data['test'])} test)")
    print(f"  trace_evidence.json      ({len(evidence_cache)} traces)")
    if gt_data:
        print(f"  ground_truth_50_enriched.json (legacy + onset_step)")


if __name__ == "__main__":
    main()
