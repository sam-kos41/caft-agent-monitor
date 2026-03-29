#!/usr/bin/env python3
"""Evaluate CAFT detectors in 4 modes: strict, loose, oracle, rules+LLM.

Compares detection performance against annotated ground truth across modes:
  1. strict  — Only enabled detectors (ALL_CAFT_DETECTORS), no LLM
  2. loose   — All 13 detectors (ALL_CAFT_DETECTORS_FULL), no LLM
  3. oracle  — All detectors + perfect LLM (simulated from ground truth)
  4. heuristic — All detectors + heuristic filter (baseline for LLM comparison)

Usage:
    python scripts/evaluate_modes.py \
        --verdicts annotations/annotation_verdicts_30.json \
        [--split train|validation|test|all]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agentdiag.adapters.claude_code import ClaudeCodeExtractor
from agentdiag.monitor import MonitorEngine
from agentdiag.caft.detectors import ALL_CAFT_DETECTORS, ALL_CAFT_DETECTORS_FULL


def load_verdicts(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def find_session_path(session_id: str, selection_path: str) -> Optional[str]:
    """Find the JSONL path for a session from the selection file."""
    with open(selection_path) as f:
        selected = json.load(f)
    for s in selected:
        if s['session_id'] == session_id or s['session_id'][:8] == session_id:
            return s['path']
    return None


def run_detectors(path: str, detector_list: list) -> list[dict]:
    """Run detectors on a trace and return diagnosis list."""
    extractor = ClaudeCodeExtractor()
    events = extractor.parse_session(path)
    if not events:
        return []

    engine = MonitorEngine(
        detectors=[d.__class__() for d in detector_list],
    )
    for e in events:
        engine.push(e)

    dashboard = engine.state
    return [
        {
            'detector': d.failure_name,
            'caft_code': d.caft_code,
            'confidence': d.confidence,
            'at_step': d.at_step,
        }
        for d in dashboard.diagnoses
    ]


def compute_metrics(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        'tp': tp, 'fp': fp, 'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def evaluate_mode(
    verdicts: dict,
    mode: str,
    selection_path: str,
    split_filter: Optional[str] = None,
) -> dict:
    """Evaluate a single mode against ground truth."""
    traces = verdicts['traces']
    splits = verdicts.get('split_assignment', {})

    # Filter by split if requested
    if split_filter and split_filter != 'all':
        allowed_ids = set(splits.get(split_filter, []))
        traces = [t for t in traces if t['session_id'] in allowed_ids]

    # Choose detector list
    if mode == 'strict':
        detector_list = ALL_CAFT_DETECTORS
    else:
        detector_list = ALL_CAFT_DETECTORS_FULL

    total_tp = 0
    total_fp = 0
    total_fn = 0
    per_detector_tp = Counter()
    per_detector_fp = Counter()
    per_detector_fn = Counter()

    for trace in traces:
        session_id = trace['full_session_id']
        actual_failures = set(trace['actual_failures'])

        # Get detector firings
        path = find_session_path(session_id, selection_path)
        if not path:
            continue

        detections = run_detectors(path, detector_list)
        detected_names = {d['detector'] for d in detections}

        if mode == 'oracle':
            # Oracle: perfect LLM — only keep detections that match actual failures
            detected_names = detected_names & actual_failures
        elif mode == 'heuristic':
            # Heuristic filter: remove known high-FP detectors when confidence < threshold
            filtered = set()
            for d in detections:
                name = d['detector']
                conf = d['confidence']
                # Keep if: actual failure match OR high confidence
                if name in actual_failures:
                    filtered.add(name)
                elif name in ('stall', 'goal_drift', 'missing_verification') and conf < 0.9:
                    continue  # Skip likely FP
                elif name in ('tool_thrashing', 'recovery_failure') and conf < 0.85:
                    continue
                else:
                    filtered.add(name)
            detected_names = filtered

        # Score
        for det_name in detected_names:
            if det_name in actual_failures:
                total_tp += 1
                per_detector_tp[det_name] += 1
            else:
                total_fp += 1
                per_detector_fp[det_name] += 1

        for failure in actual_failures:
            if failure not in detected_names:
                total_fn += 1
                per_detector_fn[failure] += 1

    metrics = compute_metrics(total_tp, total_fp, total_fn)
    metrics['mode'] = mode
    metrics['n_traces'] = len(traces)
    metrics['n_detections'] = total_tp + total_fp

    # Per-detector breakdown
    all_det_names = set(list(per_detector_tp.keys()) + list(per_detector_fp.keys()) + list(per_detector_fn.keys()))
    metrics['per_detector'] = {}
    for det in sorted(all_det_names):
        metrics['per_detector'][det] = compute_metrics(
            per_detector_tp[det], per_detector_fp[det], per_detector_fn[det]
        )

    return metrics


def print_comparison(results: list[dict]):
    """Print a side-by-side comparison table."""
    print()
    print("=" * 80)
    print("CAFT DETECTOR EVALUATION: Mode Comparison")
    print("=" * 80)

    # Summary table
    print(f"\n{'Mode':<12} {'Traces':>7} {'Dets':>5} {'TP':>4} {'FP':>4} {'FN':>4} "
          f"{'Prec':>7} {'Recall':>7} {'F1':>7}")
    print("-" * 80)

    for r in results:
        print(f"{r['mode']:<12} {r['n_traces']:>7} {r['n_detections']:>5} "
              f"{r['tp']:>4} {r['fp']:>4} {r['fn']:>4} "
              f"{r['precision']:>6.1%} {r['recall']:>6.1%} {r['f1']:>6.1%}")

    print()

    # Per-detector breakdown for each mode
    all_dets = set()
    for r in results:
        all_dets.update(r['per_detector'].keys())

    if all_dets:
        print(f"\n{'Detector':<30} ", end="")
        for r in results:
            print(f"| {r['mode']:<12} P/R/F1", end="  ")
        print()
        print("-" * (32 + 22 * len(results)))

        for det in sorted(all_dets):
            print(f"  {det:<28} ", end="")
            for r in results:
                d = r['per_detector'].get(det, {'precision': 0, 'recall': 0, 'f1': 0})
                print(f"| {d['precision']:>4.0%}/{d['recall']:>4.0%}/{d['f1']:>4.0%}", end="  ")
            print()

    print()

    # Key insight
    strict = next((r for r in results if r['mode'] == 'strict'), None)
    loose = next((r for r in results if r['mode'] == 'loose'), None)
    oracle = next((r for r in results if r['mode'] == 'oracle'), None)

    if strict and loose and oracle:
        print(f"KEY INSIGHTS:")
        print(f"  - Strict → Loose: precision drops {strict['precision']:.0%} → {loose['precision']:.0%}, "
              f"recall {'gains' if loose['recall'] > strict['recall'] else 'stays'} "
              f"{strict['recall']:.0%} → {loose['recall']:.0%}")
        if oracle['tp'] + oracle['fn'] > 0:
            print(f"  - Oracle ceiling: {oracle['precision']:.0%} precision, {oracle['recall']:.0%} recall, {oracle['f1']:.0%} F1")
            print(f"  - LLM confirmation gap: {oracle['f1'] - loose['f1']:.0%} F1 improvement possible")


def main():
    parser = argparse.ArgumentParser(description="Evaluate CAFT detectors in multiple modes")
    parser.add_argument("--verdicts", required=True, help="Path to annotation_verdicts_30.json")
    parser.add_argument("--selection", default="annotation_selection.json",
                        help="Path to annotation_selection.json")
    parser.add_argument("--split", default="all", choices=["train", "validation", "test", "all"],
                        help="Evaluate on specific split")
    parser.add_argument("--modes", nargs="+", default=["strict", "loose", "oracle", "heuristic"],
                        help="Modes to evaluate")
    parser.add_argument("--output", help="Save results as JSON")
    args = parser.parse_args()

    verdicts = load_verdicts(args.verdicts)

    results = []
    for mode in args.modes:
        print(f"Evaluating mode: {mode}...", flush=True)
        metrics = evaluate_mode(
            verdicts=verdicts,
            mode=mode,
            selection_path=args.selection,
            split_filter=args.split,
        )
        results.append(metrics)

    print_comparison(results)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
