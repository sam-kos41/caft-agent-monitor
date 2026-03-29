"""Evaluate detectors against ground truth annotations.

Runs all 20 annotated traces through the detector pipeline and computes
precision, recall, and F1 per detector against ground_truth_20.json.

Supports two modes:
  --mode rules-only      Default: conservative detector list (ALL_CAFT_DETECTORS)
  --mode rules-plus-llm  All detectors + LLM confirmation layer

Usage:
    python scripts/evaluate_ground_truth.py
    python scripts/evaluate_ground_truth.py --mode rules-plus-llm
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

# Add package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agentdiag.adapters.claude_code import ClaudeCodeExtractor
from agentdiag.monitor import MonitorEngine
from agentdiag.caft.detectors import ALL_CAFT_DETECTORS, ALL_CAFT_DETECTORS_FULL


def load_ground_truth():
    gt_path = Path(__file__).parent.parent / "annotations" / "ground_truth_20.json"
    with open(gt_path) as f:
        return json.load(f)


def find_session_path(session_id: str, traces_root: Path) -> Path | None:
    """Find the JSONL file for a session ID."""
    for p in traces_root.rglob(f"{session_id}*.jsonl"):
        return p
    return None


def run_evaluation(mode: str = "rules-only") -> dict:
    """Run the evaluation and return structured results.

    Args:
        mode: "rules-only" or "rules-plus-llm"

    Returns:
        Dict with per-trace results, per-detector metrics, and overall stats.
    """
    gt = load_ground_truth()
    traces = gt["traces"]
    extractor = ClaudeCodeExtractor()

    # Discover all sessions
    traces_root = Path("~/.claude/projects").expanduser()
    all_sessions = extractor.discover(traces_root, min_lines=5)

    # Select detector list and confirmation mode
    if mode == "rules-plus-llm":
        detectors = list(ALL_CAFT_DETECTORS_FULL)
        confirm = True
    else:
        detectors = list(ALL_CAFT_DETECTORS)
        confirm = False

    # All known detector names (superset for metrics)
    detector_names = [
        "step_repetition", "goal_drift", "missing_verification",
        "context_loss", "premature_termination", "reasoning_action_mismatch",
        "tool_thrashing",
    ]

    # Counters: TP, FP, FN per detector
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    total_fired = 0
    total_tp = 0

    results = []

    for trace in traces:
        session_id = trace["session_id"]
        trace_num = trace["trace_num"]

        # Find session
        matching = [s for s in all_sessions if s.session_id.startswith(session_id)]
        if not matching:
            print(f"  [SKIP] Trace {trace_num} ({session_id}): session not found")
            continue

        session = matching[0]

        # Parse and run detectors
        try:
            events = extractor.parse_session(session)
        except Exception as ex:
            print(f"  [ERROR] Trace {trace_num} ({session_id}): {ex}")
            continue

        engine = MonitorEngine(
            goal=f"Trace {trace_num}",
            detectors=detectors,
            confirm=confirm,
        )
        for event in events:
            engine.push(event)

        state = engine.state
        detected = {d.failure_name for d in state.diagnoses}

        # Ground truth: which detectors SHOULD fire (TP verdicts)
        actual_failures = set(trace.get("actual_failures", []))

        # For each detector, determine TP/FP/FN
        trace_result = {
            "trace_num": trace_num,
            "session_id": session_id,
            "detected": sorted(detected),
            "actual": sorted(actual_failures),
            "per_detector": {},
            "confirmation_stats": {
                "total": state.candidates_total,
                "confirmed": state.candidates_confirmed,
                "rejected": state.candidates_rejected,
                "uncertain": state.candidates_uncertain,
                "autoconfirmed": state.candidates_autoconfirmed,
            },
        }

        for det_name in detector_names:
            fired = det_name in detected
            is_real = det_name in actual_failures

            if fired and is_real:
                tp[det_name] += 1
                total_tp += 1
                trace_result["per_detector"][det_name] = "TP"
            elif fired and not is_real:
                fp[det_name] += 1
                trace_result["per_detector"][det_name] = "FP"
            elif not fired and is_real:
                fn[det_name] += 1
                trace_result["per_detector"][det_name] = "FN"
            else:
                trace_result["per_detector"][det_name] = "TN"

            if fired:
                total_fired += 1

        results.append(trace_result)

    # Compute overall metrics
    total_fp = sum(fp.values())
    total_fn = sum(fn.values())
    overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = (
        2 * overall_prec * overall_rec / (overall_prec + overall_rec)
        if (overall_prec + overall_rec) > 0 else 0
    )

    return {
        "mode": mode,
        "trace_count": len(results),
        "results": results,
        "per_detector": {
            det: {
                "tp": tp[det], "fp": fp[det], "fn": fn[det],
                "precision": tp[det] / (tp[det] + fp[det]) if (tp[det] + fp[det]) > 0 else None,
                "recall": tp[det] / (tp[det] + fn[det]) if (tp[det] + fn[det]) > 0 else None,
            }
            for det in detector_names
        },
        "overall": {
            "total_fired": total_fired,
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
            "precision": overall_prec,
            "recall": overall_rec,
            "f1": overall_f1,
        },
    }


def print_evaluation(eval_result: dict) -> None:
    """Print formatted evaluation results."""
    mode = eval_result["mode"]
    results = eval_result["results"]
    detector_names = [
        "step_repetition", "goal_drift", "missing_verification",
        "context_loss", "premature_termination", "reasoning_action_mismatch",
        "tool_thrashing",
    ]

    mode_label = "RULES ONLY" if mode == "rules-only" else "RULES + LLM CONFIRMATION"
    print(f"\n{'=' * 90}")
    print(f"  DETECTOR EVALUATION [{mode_label}] — {len(results)} traces")
    print(f"{'=' * 90}")

    # Per-trace table
    print(f"\n  {'#':>3} {'Session':<12} {'Detected':<40} {'Actual':<35}")
    print(f"  {'─' * 85}")
    for r in results:
        det_str = ", ".join(r["detected"]) if r["detected"] else "—"
        act_str = ", ".join(r["actual"]) if r["actual"] else "—"
        print(f"  {r['trace_num']:>3} {r['session_id'][:10]:<12} {det_str:<40} {act_str:<35}")

    # Per-detector metrics
    print(f"\n  {'─' * 85}")
    print(f"\n  PER-DETECTOR METRICS:")
    print(f"  {'Detector':<30} {'TP':>4} {'FP':>4} {'FN':>4} {'Prec':>7} {'Recall':>7} {'F1':>7}")
    print(f"  {'─' * 70}")

    per_det = eval_result["per_detector"]
    for det in detector_names:
        m = per_det[det]
        t, f, n = m["tp"], m["fp"], m["fn"]
        prec = m["precision"]
        rec = m["recall"]
        f1 = 2 * prec * rec / (prec + rec) if prec is not None and rec is not None and (prec + rec) > 0 else None

        prec_str = f"{prec:.0%}" if prec is not None else "N/A"
        rec_str = f"{rec:.0%}" if rec is not None else "N/A"
        f1_str = f"{f1:.0%}" if f1 is not None else "N/A"

        print(f"  {det:<30} {t:>4} {f:>4} {n:>4} {prec_str:>7} {rec_str:>7} {f1_str:>7}")

    o = eval_result["overall"]
    print(f"  {'─' * 70}")
    print(f"  {'OVERALL':<30} {o['total_tp']:>4} {o['total_fp']:>4} {o['total_fn']:>4} "
          f"{o['precision']:>6.0%} {o['recall']:>6.0%} {o['f1']:>6.0%}")

    print(f"\n  Total firings: {o['total_fired']}")
    print(f"  True positives: {o['total_tp']}")
    print(f"  Overall precision: {o['precision']:.1%}")
    print(f"  Overall recall: {o['recall']:.1%}")
    print(f"  Overall F1: {o['f1']:.1%}")

    # Confirmation stats (if using LLM mode)
    if mode == "rules-plus-llm":
        total_conf = sum(r["confirmation_stats"]["total"] for r in results)
        total_confirmed = sum(r["confirmation_stats"]["confirmed"] for r in results)
        total_rejected = sum(r["confirmation_stats"]["rejected"] for r in results)
        total_uncertain = sum(r["confirmation_stats"]["uncertain"] for r in results)
        total_auto = sum(r["confirmation_stats"]["autoconfirmed"] for r in results)

        print(f"\n  LLM CONFIRMATION STATS:")
        print(f"  Total candidates: {total_conf}")
        print(f"  Auto-confirmed (high confidence): {total_auto}")
        print(f"  LLM confirmed: {total_confirmed}")
        print(f"  LLM rejected: {total_rejected}")
        print(f"  Uncertain (LLM unavailable/failed): {total_uncertain}")

    print(f"{'=' * 90}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate detectors against ground truth")
    parser.add_argument(
        "--mode",
        choices=["rules-only", "rules-plus-llm"],
        default="rules-only",
        help="Evaluation mode (default: rules-only)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run both modes and compare side-by-side",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    args = parser.parse_args()

    if args.compare:
        # Run both modes
        print("\n  Running rules-only evaluation...")
        rules_result = run_evaluation("rules-only")
        print("\n  Running rules+LLM evaluation...")
        llm_result = run_evaluation("rules-plus-llm")

        if args.json:
            print(json.dumps({
                "rules_only": rules_result,
                "rules_plus_llm": llm_result,
            }, indent=2))
        else:
            print_evaluation(rules_result)
            print_evaluation(llm_result)

            # Comparison summary
            ro = rules_result["overall"]
            lo = llm_result["overall"]
            print(f"\n{'=' * 60}")
            print(f"  COMPARISON: Rules-Only vs Rules+LLM")
            print(f"{'=' * 60}")
            print(f"  {'Metric':<20} {'Rules-Only':>15} {'Rules+LLM':>15} {'Delta':>10}")
            print(f"  {'─' * 55}")
            for metric in ["precision", "recall", "f1"]:
                rv = ro[metric]
                lv = lo[metric]
                delta = lv - rv
                sign = "+" if delta > 0 else ""
                print(f"  {metric:<20} {rv:>14.1%} {lv:>14.1%} {sign}{delta:>9.1%}")
            print(f"  {'─' * 55}")
            print(f"  {'Total firings':<20} {ro['total_fired']:>15} {lo['total_fired']:>15}")
            print(f"  {'True positives':<20} {ro['total_tp']:>15} {lo['total_tp']:>15}")
            print(f"  {'False positives':<20} {ro['total_fp']:>15} {lo['total_fp']:>15}")
            print(f"{'=' * 60}")
    else:
        result = run_evaluation(args.mode)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print_evaluation(result)


if __name__ == "__main__":
    main()
