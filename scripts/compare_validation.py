#!/usr/bin/env python3
"""Compare human validation marks against system anomaly detections.

Reads a validation log JSONL file produced by ``agentdiag live --log-validation``
and computes agreement between human judgment and system detection.

Agreement is computed over a temporal window: a human "struggling" mark at
step N is considered a **match** if the system detected any named anomaly
within ±W steps of N (default W=15). A human "fine" mark is a match if the
system detected NO named anomaly within ±W steps.

Output:
  - Total human marks and system detections
  - Confusion matrix (true positives, false positives, false negatives, true negatives)
  - Agreement rate, precision, recall, F1
  - Per-signature breakdown (which signatures align with human judgment)
  - Timeline showing where human and system agree/disagree

Usage::

    python scripts/compare_validation.py validation_log.jsonl
    python scripts/compare_validation.py validation_log.jsonl --window 20
    python scripts/compare_validation.py validation_log.jsonl --json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load_log(path: str) -> list[dict]:
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def compute_agreement(
    records: list[dict],
    window: int = 15,
) -> dict:
    """Compute agreement between human marks and system detections.

    Returns a dict with confusion matrix, rates, and per-signature breakdown.
    """
    human_marks = [r for r in records if r["type"] == "human_mark"]
    detections = [r for r in records if r["type"] == "system_detection"]

    # Build step sets for fast lookup
    detection_steps: dict[int, list[dict]] = defaultdict(list)
    for d in detections:
        step = d.get("step", 0)
        detection_steps[step].append(d)

    named_detection_steps: set[int] = set()
    for d in detections:
        sig = d.get("signature", "")
        if sig and sig != "unclassified_anomaly":
            named_detection_steps.add(d.get("step", 0))

    # Classify each human mark
    tp = 0  # human=struggling, system=detected nearby
    fp = 0  # system detected, but human said fine nearby
    fn = 0  # human=struggling, system=nothing nearby
    tn = 0  # human=fine, system=nothing nearby

    per_mark_results: list[dict] = []
    matched_signatures: Counter = Counter()

    for mark in human_marks:
        mark_step = mark.get("step", 0)
        label = mark.get("label", "")

        # Check if any named detection is within ±window steps
        nearby_sigs: list[str] = []
        for s in range(mark_step - window, mark_step + window + 1):
            if s in named_detection_steps:
                for d in detection_steps.get(s, []):
                    sig = d.get("signature", "")
                    if sig and sig != "unclassified_anomaly":
                        nearby_sigs.append(sig)

        has_nearby_detection = len(nearby_sigs) > 0

        if label == "struggling":
            if has_nearby_detection:
                tp += 1
                for sig in nearby_sigs:
                    matched_signatures[sig] += 1
            else:
                fn += 1
        elif label == "fine":
            if has_nearby_detection:
                fp += 1
            else:
                tn += 1

        per_mark_results.append({
            "step": mark_step,
            "label": label,
            "nearby_detections": nearby_sigs,
            "agreement": (label == "struggling" and has_nearby_detection)
                         or (label == "fine" and not has_nearby_detection),
        })

    # Compute rates
    total_marks = len(human_marks)
    agreement_count = tp + tn
    agreement_rate = agreement_count / total_marks if total_marks > 0 else 0.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    struggling_marks = sum(1 for m in human_marks if m.get("label") == "struggling")
    fine_marks = sum(1 for m in human_marks if m.get("label") == "fine")

    # Per-signature hit rate
    sig_counts = Counter(d.get("signature", "") for d in detections if d.get("signature", "") != "unclassified_anomaly")

    return {
        "summary": {
            "human_marks": total_marks,
            "struggling_marks": struggling_marks,
            "fine_marks": fine_marks,
            "system_detections": len(detections),
            "named_detections": len([d for d in detections if d.get("signature", "") != "unclassified_anomaly"]),
            "window_steps": window,
        },
        "confusion_matrix": {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn,
        },
        "rates": {
            "agreement_rate": round(agreement_rate, 3),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        },
        "matched_signatures": dict(matched_signatures),
        "signature_counts": dict(sig_counts),
        "per_mark": per_mark_results,
    }


def format_report(result: dict) -> str:
    """Format the agreement result as a human-readable report."""
    lines: list[str] = []
    s = result["summary"]
    cm = result["confusion_matrix"]
    r = result["rates"]

    lines.append("=" * 60)
    lines.append("  VALIDATION AGREEMENT REPORT")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"  Human marks:        {s['human_marks']}")
    lines.append(f"    Struggling:       {s['struggling_marks']}")
    lines.append(f"    Fine:             {s['fine_marks']}")
    lines.append(f"  System detections:  {s['system_detections']}")
    lines.append(f"    Named:            {s['named_detections']}")
    lines.append(f"  Window:             +/- {s['window_steps']} steps")
    lines.append("")
    lines.append("  CONFUSION MATRIX")
    lines.append("  " + "-" * 40)
    lines.append(f"                    System: Yes   System: No")
    lines.append(f"  Human: Struggling   TP={cm['true_positive']:3d}        FN={cm['false_negative']:3d}")
    lines.append(f"  Human: Fine         FP={cm['false_positive']:3d}        TN={cm['true_negative']:3d}")
    lines.append("")
    lines.append(f"  Agreement rate:  {r['agreement_rate']:.1%}")
    lines.append(f"  Precision:       {r['precision']:.1%}")
    lines.append(f"  Recall:          {r['recall']:.1%}")
    lines.append(f"  F1:              {r['f1']:.1%}")
    lines.append("")

    if result["matched_signatures"]:
        lines.append("  MATCHED SIGNATURES (aligned with human 'struggling')")
        lines.append("  " + "-" * 40)
        for sig, count in sorted(result["matched_signatures"].items(), key=lambda x: -x[1]):
            lines.append(f"    {sig}: {count}")
        lines.append("")

    if result["signature_counts"]:
        lines.append("  ALL SYSTEM SIGNATURES")
        lines.append("  " + "-" * 40)
        for sig, count in sorted(result["signature_counts"].items(), key=lambda x: -x[1]):
            matched = result["matched_signatures"].get(sig, 0)
            lines.append(f"    {sig}: {count} total, {matched} matched human marks")
        lines.append("")

    # Timeline (abbreviated)
    per_mark = result["per_mark"]
    if per_mark:
        lines.append("  TIMELINE")
        lines.append("  " + "-" * 40)
        for m in per_mark:
            icon = "Y" if m["agreement"] else "N"
            nearby = ", ".join(m["nearby_detections"][:3]) or "none"
            lines.append(f"    step {m['step']:>5d}  {m['label']:<12s}  [{icon}]  nearby: {nearby}")
        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare human validation marks against system detections",
    )
    parser.add_argument("log_file", help="Path to validation log JSONL")
    parser.add_argument("--window", type=int, default=15,
                        help="Step window for temporal matching (default: 15)")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON instead of formatted report")
    parser.add_argument("--output", type=str, default=None,
                        help="Save report to file")

    args = parser.parse_args()

    if not Path(args.log_file).exists():
        print(f"Error: File not found: {args.log_file}", file=sys.stderr)
        sys.exit(1)

    records = load_log(args.log_file)
    result = compute_agreement(records, window=args.window)

    if args.json:
        output = json.dumps(result, indent=2, default=str)
    else:
        output = format_report(result)

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"Report saved to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
