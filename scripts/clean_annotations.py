#!/usr/bin/env python3
"""Clean existing annotation data by applying ground-truth authority rules.

This migration script:
1. Reads the raw annotation_ledger.jsonl (all layers preserved)
2. Applies the authority hierarchy: adjudicated > human > auto > detector
3. Resolves session ID conflicts (short 8-char ↔ full UUID)
4. Writes a clean annotations.jsonl with:
   - Human CLEAN annotations preserved (so detector FPs are visible)
   - Unlabeled detector annotations excluded from ground truth
   - DRAFT annotations flagged for review
   - Canonical session IDs throughout
5. Outputs a validation report

Usage:
    python scripts/clean_annotations.py \
        --input annotations/ablation_ready/annotations.jsonl \
        --output annotations/ablation_ready/annotations_clean.jsonl

    # Preview only (don't write)
    python scripts/clean_annotations.py --input annotations.jsonl --dry-run

    # Also rebuild from ledger
    python scripts/clean_annotations.py \
        --ledger annotations/ablation_ready/annotation_ledger.jsonl \
        --output annotations/ablation_ready/annotations_clean.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agentdiag.metrics import validate_annotations_jsonl


_AUTHORITY = {"adjudicated": 4, "human_reviewed": 3, "auto_labeled": 2, "unlabeled": 1}
_TRUSTED_STATUSES = {"human_reviewed", "adjudicated"}


def clean_annotations(
    input_path: Path,
    output_path: Path | None = None,
    dry_run: bool = False,
) -> dict:
    """Clean annotation file and return stats."""

    # Load all records
    all_records: list[dict] = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            all_records.append(json.loads(line))

    print(f"Loaded {len(all_records)} records from {input_path}")

    # Group by 8-char session prefix
    groups: dict[str, list[dict]] = defaultdict(list)
    for d in all_records:
        sid = d.get("session_id", d.get("trace_id", ""))
        prefix = sid[:8]
        groups[prefix].append(d)

    # Process each session
    output_records: list[dict] = []
    stats = {
        "total_input": len(all_records),
        "total_sessions": len(groups),
        "clean_sessions": 0,
        "failure_sessions": 0,
        "detector_only_excluded": 0,
        "draft_included": 0,
        "human_reviewed_included": 0,
        "adjudicated_included": 0,
        "conflicts_resolved": 0,
        "output_records": 0,
    }

    for prefix, records in sorted(groups.items()):
        # Find canonical (longest) session ID
        canonical_sid = max(
            (r.get("session_id", r.get("trace_id", "")) for r in records),
            key=len,
        )

        # Sort by authority
        records.sort(
            key=lambda r: _AUTHORITY.get(r.get("label_status", "unlabeled"), 0),
            reverse=True,
        )

        # Find highest-authority annotation
        best = records[0]
        best_status = best.get("label_status", "unlabeled")
        best_type = best.get("annotator_type", "detector")

        # Human/adjudicated CLEAN → write CLEAN record, skip everything else
        if (best_status in _TRUSTED_STATUSES
                and best_type in ("human", "adjudicated")
                and not best.get("has_failure", False)):
            stats["clean_sessions"] += 1
            # Check if detectors fired on this CLEAN session
            detector_fires = [r for r in records
                              if r.get("annotator_type") == "detector"
                              and r.get("has_failure", False)]
            if detector_fires:
                stats["conflicts_resolved"] += 1
                det_names = [r.get("annotator_id", "?") for r in detector_fires]
                print(f"  {prefix}: CLEAN (human) — suppressed detector(s): {det_names}")

            # Write the CLEAN human annotation with canonical ID
            clean_rec = dict(best)
            clean_rec["session_id"] = canonical_sid
            clean_rec["trace_id"] = canonical_sid
            output_records.append(clean_rec)
            stats["human_reviewed_included"] += 1
            continue

        # Failure session: include only trusted annotations
        session_has_failure = False
        seen_failures: set[str] = set()

        for r in records:
            status = r.get("label_status", "unlabeled")
            atype = r.get("annotator_type", "detector")

            if not r.get("has_failure", False):
                continue

            # Skip unlabeled detector annotations
            if status == "unlabeled" and atype == "detector":
                stats["detector_only_excluded"] += 1
                continue

            # Derive failure name for dedup
            fname = (r.get("primary_caft_subtype")
                     or r.get("failure_name")
                     or (r.get("annotator_id") if atype == "detector" else ""))
            if not fname:
                code = r.get("primary_caft_code", "")
                fname = code or "unknown"

            if fname in seen_failures:
                continue
            seen_failures.add(fname)

            if status in _TRUSTED_STATUSES:
                stats["human_reviewed_included" if status == "human_reviewed"
                       else "adjudicated_included"] += 1
            elif status == "auto_labeled":
                stats["draft_included"] += 1

            # Write with canonical ID
            out = dict(r)
            out["session_id"] = canonical_sid
            out["trace_id"] = canonical_sid
            output_records.append(out)
            session_has_failure = True

        if session_has_failure:
            stats["failure_sessions"] += 1
        else:
            # All failure annotations were detector-only → effectively CLEAN
            stats["clean_sessions"] += 1

    stats["output_records"] = len(output_records)

    # Write output
    if not dry_run and output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for rec in output_records:
                f.write(json.dumps(rec, default=str) + "\n")
        print(f"\nWrote {len(output_records)} records to {output_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Clean annotation data")
    parser.add_argument("--input", required=True, help="Input annotations JSONL")
    parser.add_argument("--output", help="Output clean annotations JSONL")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else None

    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Validate BEFORE cleaning
    print("=" * 60)
    print("PRE-CLEAN VALIDATION")
    print("=" * 60)
    qa_before = validate_annotations_jsonl(str(input_path))
    print(f"  Sessions: {qa_before.total_sessions}")
    print(f"  Human reviewed: {qa_before.n_human_reviewed}")
    print(f"  Adjudicated: {qa_before.n_adjudicated}")
    print(f"  Drafts: {qa_before.n_draft}")
    print(f"  Detector-only: {qa_before.n_detector_only}")
    print(f"  Clean sessions: {qa_before.n_clean_sessions}")
    print(f"  Failure sessions: {qa_before.n_failure_sessions}")
    for err in qa_before.errors:
        print(f"  ERROR: {err}")
    for w in qa_before.warnings:
        print(f"  WARNING: {w}")

    # Clean
    print(f"\n{'=' * 60}")
    print("CLEANING" + (" (DRY RUN)" if args.dry_run else ""))
    print("=" * 60)
    stats = clean_annotations(input_path, output_path, dry_run=args.dry_run)

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Validate AFTER cleaning
    if output_path and output_path.exists() and not args.dry_run:
        print(f"\n{'=' * 60}")
        print("POST-CLEAN VALIDATION")
        print("=" * 60)
        qa_after = validate_annotations_jsonl(str(output_path))
        print(f"  Sessions: {qa_after.total_sessions}")
        print(f"  Errors: {len(qa_after.errors)}")
        print(f"  Warnings: {len(qa_after.warnings)}")
        for err in qa_after.errors:
            print(f"  ERROR: {err}")
        for w in qa_after.warnings:
            print(f"  WARNING: {w}")

        if not qa_after.errors:
            print("\n  CLEAN annotations are ready for evaluation.")
        else:
            print("\n  WARNING: Some issues remain. Review the errors above.")


if __name__ == "__main__":
    main()
