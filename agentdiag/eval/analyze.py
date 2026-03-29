"""Analysis script — computes detection metrics from runner output.

Takes runner result JSON files and computes:
  - True positive rate (TPR) per failure type
  - False positive rate (FPR) on clean traces
  - Detection latency (steps from injection to first detection)
  - Signature accuracy (correct compositor label)
  - Cross-task generalization
  - Phase-conditional vs global comparison

Usage::

    python -m agentdiag.eval.analyze --results results/ --output report.md
    python -m agentdiag.eval.analyze --results results/ --manifest traces/manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional

from agentdiag.eval.trace_generator import EXPECTED_SIGNATURES, FAILURE_VARIANTS


def load_manifest(manifest_path: str | Path) -> dict:
    """Load the trace generation manifest."""
    with open(manifest_path) as f:
        return json.load(f)


def load_results(
    results_dir: str | Path,
    z_threshold: float = 3.0,
) -> dict[str, dict]:
    """Load all result files for a given z-threshold.

    Returns {trace_name: result_dict}.
    """
    results_dir = Path(results_dir)
    results = {}

    # Try 2-decimal format first (sweep), then 1-decimal (legacy)
    for fmt in (f"_z{z_threshold:.2f}.json", f"_z{z_threshold:.1f}.json"):
        for path in sorted(results_dir.glob(f"*{fmt}")):
            name = path.stem.replace(fmt.replace(".json", ""), "")
            with open(path) as f:
                results[name] = json.load(f)
        if results:
            break

    return results


def _parse_trace_name(name: str) -> tuple[str, str]:
    """Parse 'rest_api_clean' into ('rest_api', 'clean')."""
    for variant in FAILURE_VARIANTS + ["clean"]:
        if name.endswith(f"_{variant}"):
            task = name[: -(len(variant) + 1)]
            return task, variant
    return name, "unknown"


def compute_detection_metrics(
    results: dict[str, dict],
    manifest: dict[str, dict],
    detection_window: int = 30,
) -> dict:
    """Compute all detection metrics.

    Args:
        results: {trace_name: runner_result}
        manifest: {filename: trace_info} from generate_all()
        detection_window: Max steps after injection to count as detected.

    Returns:
        Comprehensive metrics dict.
    """
    # Classify results
    clean_results = {}
    failure_results = defaultdict(list)  # variant → list of (task, result, manifest_info)

    for trace_name, result in results.items():
        task_name, variant = _parse_trace_name(trace_name)
        filename = f"{trace_name}.jsonl"
        info = manifest.get(filename, {})

        if variant == "clean":
            clean_results[task_name] = result
        elif variant in FAILURE_VARIANTS:
            failure_results[variant].append((task_name, result, info))

    # ── FPR: false positive rate on clean traces ──
    total_clean_steps = 0
    total_clean_anomalies = 0
    clean_anomaly_counts = {}

    for task_name, result in clean_results.items():
        n_events = result.get("total_events", 0)
        n_anomalies = result.get("total_anomalies", 0)
        total_clean_steps += n_events
        total_clean_anomalies += n_anomalies
        clean_anomaly_counts[task_name] = n_anomalies

    fpr = total_clean_anomalies / max(total_clean_steps, 1)

    # ── TPR, latency, signature accuracy per failure type ──
    per_variant = {}
    all_latencies = []
    all_detected = 0
    all_total = 0

    for variant in FAILURE_VARIANTS:
        entries = failure_results.get(variant, [])
        if not entries:
            continue

        detected = 0
        correct_sig = 0
        latencies = []
        task_results = {}

        expected_sig = EXPECTED_SIGNATURES.get(variant, "unclassified_anomaly")

        for task_name, result, info in entries:
            inject_step = info.get("inject_step", 0)
            inject_end = info.get("inject_end", inject_step + 30)
            anomalies = result.get("anomalies", [])

            # Check for detection within window of injection
            detection_step = None
            detected_sig = None
            has_correct_sig = False
            sigs_in_window = {}
            for a in anomalies:
                a_step = a.get("step", 0)
                sig = a.get("signature", "unknown")
                # Detection counts if it falls within injection range + window
                if inject_step <= a_step <= inject_end + detection_window:
                    sigs_in_window[sig] = sigs_in_window.get(sig, 0) + 1
                    if detection_step is None or a_step < detection_step:
                        detection_step = a_step
                        detected_sig = sig
                    if sig == expected_sig:
                        has_correct_sig = True

            is_detected = detection_step is not None
            if is_detected:
                detected += 1
                latency = detection_step - inject_step
                latencies.append(latency)
                all_latencies.append(latency)
                if has_correct_sig:
                    correct_sig += 1

            task_results[task_name] = {
                "detected": is_detected,
                "detection_step": detection_step,
                "latency": (detection_step - inject_step) if is_detected else None,
                "detected_signature": detected_sig,
                "expected_signature": expected_sig,
                "correct_signature": has_correct_sig,
                "signatures_in_window": sigs_in_window,
                "inject_step": inject_step,
                "inject_end": inject_end,
                "total_anomalies": len(anomalies),
            }

        n = len(entries)
        all_detected += detected
        all_total += n

        per_variant[variant] = {
            "tpr": detected / max(n, 1),
            "n_total": n,
            "n_detected": detected,
            "signature_accuracy": correct_sig / max(detected, 1),
            "latency_mean": _mean(latencies),
            "latency_median": _median(latencies),
            "latency_std": _std(latencies),
            "latency_values": latencies,
            "expected_signature": expected_sig,
            "task_results": task_results,
        }

    # ── Cross-task generalization ──
    domain_tpr = defaultdict(lambda: {"detected": 0, "total": 0})
    for variant, entries in failure_results.items():
        for task_name, result, info in entries:
            domain = info.get("domain", "unknown")
            inject_step = info.get("inject_step", 0)
            inject_end = info.get("inject_end", inject_step + 30)
            anomalies = result.get("anomalies", [])
            is_detected = any(
                inject_step <= a.get("step", 0) <= inject_end + detection_window
                for a in anomalies
            )
            domain_tpr[domain]["total"] += 1
            if is_detected:
                domain_tpr[domain]["detected"] += 1

    cross_task = {
        domain: {
            "tpr": counts["detected"] / max(counts["total"], 1),
            **counts,
        }
        for domain, counts in domain_tpr.items()
    }

    return {
        "overall": {
            "tpr": all_detected / max(all_total, 1),
            "fpr": fpr,
            "n_failure_traces": all_total,
            "n_detected": all_detected,
            "n_clean_traces": len(clean_results),
            "total_clean_anomalies": total_clean_anomalies,
            "latency_mean": _mean(all_latencies),
            "latency_median": _median(all_latencies),
            "latency_std": _std(all_latencies),
        },
        "per_variant": per_variant,
        "cross_task": cross_task,
        "clean_anomaly_counts": clean_anomaly_counts,
        "detection_window": detection_window,
    }


def generate_report(metrics: dict, output_path: str | Path) -> str:
    """Generate a markdown report from metrics."""
    lines = []
    lines.append("# CAFT Evaluation Report\n")

    overall = metrics["overall"]
    lines.append("## Overall Results\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| True Positive Rate | {overall['tpr']:.3f} |")
    lines.append(f"| False Positive Rate | {overall['fpr']:.4f} |")
    lines.append(f"| Detection Latency (median) | {overall['latency_median']:.1f} steps |")
    lines.append(f"| Detection Latency (mean) | {overall['latency_mean']:.1f} +/- {overall['latency_std']:.1f} steps |")
    lines.append(f"| Failure Traces | {overall['n_failure_traces']} |")
    lines.append(f"| Detected | {overall['n_detected']} |")
    lines.append(f"| Clean Traces | {overall['n_clean_traces']} |")
    lines.append(f"| Clean False Alarms | {overall['total_clean_anomalies']} |")
    lines.append("")

    # Per-variant breakdown
    lines.append("## Detection by Failure Type\n")
    lines.append("| Failure Type | TPR | Sig. Accuracy | Latency (med) | Expected Signature |")
    lines.append("|-------------|-----|---------------|---------------|-------------------|")
    for variant, data in metrics.get("per_variant", {}).items():
        lines.append(
            f"| {variant} | {data['tpr']:.3f} | {data['signature_accuracy']:.3f} "
            f"| {data['latency_median']:.1f} | {data['expected_signature']} |"
        )
    lines.append("")

    # Cross-task generalization
    lines.append("## Cross-Task Generalization\n")
    lines.append("| Domain | TPR | Detected/Total |")
    lines.append("|--------|-----|----------------|")
    for domain, data in metrics.get("cross_task", {}).items():
        lines.append(f"| {domain} | {data['tpr']:.3f} | {data['detected']}/{data['total']} |")
    lines.append("")

    # Task × variant heatmap data
    lines.append("## Task x Failure Detection Matrix\n")
    header = "| Task |"
    sep = "|------|"
    for variant in FAILURE_VARIANTS:
        header += f" {variant} |"
        sep += "--------|"
    lines.append(header)
    lines.append(sep)

    tasks_seen = set()
    for variant, data in metrics.get("per_variant", {}).items():
        for task_name in data.get("task_results", {}):
            tasks_seen.add(task_name)

    for task_name in sorted(tasks_seen):
        row = f"| {task_name} |"
        for variant in FAILURE_VARIANTS:
            vdata = metrics.get("per_variant", {}).get(variant, {})
            tr = vdata.get("task_results", {}).get(task_name)
            if tr is None:
                row += " - |"
            elif tr["detected"]:
                row += " Y |"
            else:
                row += " N |"
        lines.append(row)
    lines.append("")

    # Clean trace false alarms
    lines.append("## Clean Trace False Alarms\n")
    lines.append("| Task | Anomalies |")
    lines.append("|------|-----------|")
    for task, count in sorted(metrics.get("clean_anomaly_counts", {}).items()):
        lines.append(f"| {task} | {count} |")
    lines.append("")

    report = "\n".join(lines)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)

    return report


# ── Sweep analysis ────────────────────────────────────────────────────────

def load_sweep_summary(sweep_dir: str | Path) -> dict:
    """Load sweep_summary.json from a sweep output directory."""
    path = Path(sweep_dir) / "sweep_summary.json"
    with open(path) as f:
        return json.load(f)


def analyze_sweep(summary: dict) -> dict:
    """Analyze a sweep summary to find optimal operating points.

    For each failure type and overall, finds the z-threshold that
    maximizes Youden's J statistic (TPR - FPR).

    Returns a dict with optimal thresholds and comparisons to z=3.0.
    """
    thresholds = summary["thresholds"]
    ov = summary["overall"]

    def _find_optimal(tpr_list, fpr_list):
        best_j = -1.0
        best_idx = 0
        for i, (t, f) in enumerate(zip(tpr_list, fpr_list)):
            j = t - f
            if j > best_j:
                best_j = j
                best_idx = i
        return best_idx, best_j

    # Overall optimal
    ov_idx, ov_j = _find_optimal(ov["tpr"], ov["fpr"])

    # Find z=3.0 index
    z3_idx = None
    for i, z in enumerate(thresholds):
        if abs(z - 3.0) < 0.01:
            z3_idx = i
            break
    if z3_idx is None:
        # Closest to 3.0
        z3_idx = min(range(len(thresholds)), key=lambda i: abs(thresholds[i] - 3.0))

    result = {
        "thresholds": thresholds,
        "overall": {
            "optimal_z": thresholds[ov_idx],
            "optimal_j": round(ov_j, 4),
            "optimal_tpr": ov["tpr"][ov_idx],
            "optimal_fpr": ov["fpr"][ov_idx],
            "optimal_sig_accuracy": ov["sig_accuracy"][ov_idx],
            "optimal_latency": ov["latency_median"][ov_idx],
            "default_z": 3.0,
            "default_tpr": ov["tpr"][z3_idx],
            "default_fpr": ov["fpr"][z3_idx],
            "default_sig_accuracy": ov["sig_accuracy"][z3_idx],
            "default_latency": ov["latency_median"][z3_idx],
        },
        "by_failure": {},
    }

    for variant, vdata in summary["by_failure"].items():
        v_idx, v_j = _find_optimal(vdata["tpr"], vdata["fpr"])
        result["by_failure"][variant] = {
            "optimal_z": thresholds[v_idx],
            "optimal_j": round(v_j, 4),
            "optimal_tpr": vdata["tpr"][v_idx],
            "optimal_fpr": vdata["fpr"][v_idx],
            "optimal_sig_accuracy": vdata["sig_accuracy"][v_idx],
            "optimal_latency": vdata["latency_median"][v_idx],
            "default_tpr": vdata["tpr"][z3_idx],
            "default_fpr": vdata["fpr"][z3_idx],
        }

    return result


def generate_sweep_report(sweep_analysis: dict, output_path: str | Path) -> str:
    """Generate the sensitivity analysis section of the report."""
    lines = []
    lines.append("# Sensitivity Analysis\n")

    ov = sweep_analysis["overall"]
    lines.append("## Overall Optimal Operating Point\n")
    lines.append(f"| Metric | z=3.0 (default) | z={ov['optimal_z']:.2f} (optimal) |")
    lines.append(f"|--------|-----------------|{'--' * 10}|")
    lines.append(f"| TPR | {ov['default_tpr']:.3f} | {ov['optimal_tpr']:.3f} |")
    lines.append(f"| FPR | {ov['default_fpr']:.4f} | {ov['optimal_fpr']:.4f} |")
    lines.append(f"| Youden's J | {ov['default_tpr'] - ov['default_fpr']:.3f} | {ov['optimal_j']:.3f} |")
    lines.append(f"| Sig. Accuracy | {ov['default_sig_accuracy']:.3f} | {ov['optimal_sig_accuracy']:.3f} |")
    lines.append(f"| Latency (median) | {ov['default_latency']:.1f} steps | {ov['optimal_latency']:.1f} steps |")
    lines.append("")

    lines.append("## Optimal Threshold by Failure Type\n")
    lines.append("| Failure Type | Optimal z | J | TPR | FPR | TPR @z=3.0 |")
    lines.append("|-------------|-----------|-----|-----|-----|-----------|")
    for variant, vdata in sweep_analysis["by_failure"].items():
        lines.append(
            f"| {variant} | {vdata['optimal_z']:.2f} | {vdata['optimal_j']:.3f} "
            f"| {vdata['optimal_tpr']:.3f} | {vdata['optimal_fpr']:.4f} "
            f"| {vdata['default_tpr']:.3f} |"
        )
    lines.append("")

    # Check if failure types have divergent optimal thresholds
    opt_zs = [v["optimal_z"] for v in sweep_analysis["by_failure"].values()]
    if opt_zs:
        z_range = max(opt_zs) - min(opt_zs)
        lines.append("## Threshold Divergence\n")
        if z_range > 1.0:
            lines.append(
                f"Optimal thresholds diverge by {z_range:.2f} across failure types "
                f"(range: {min(opt_zs):.2f} to {max(opt_zs):.2f}). "
                f"Consider per-failure-type thresholds for production use."
            )
        else:
            lines.append(
                f"Optimal thresholds are consistent across failure types "
                f"(range: {min(opt_zs):.2f} to {max(opt_zs):.2f}, spread={z_range:.2f}). "
                f"A single threshold is sufficient."
            )
    lines.append("")

    report = "\n".join(lines)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)
    return report


# ── Helpers ───────────────────────────────────────────────────────────────

def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return (sum((x - m) ** 2 for x in values) / (len(values) - 1)) ** 0.5


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze CAFT evaluation results")
    parser.add_argument("--results", type=str, default="agentdiag/eval/results",
                        help="Results directory from runner")
    parser.add_argument("--manifest", type=str, default="agentdiag/eval/traces/manifest.json",
                        help="Trace generation manifest")
    parser.add_argument("--output", type=str, default="agentdiag/eval/results/report.md",
                        help="Output report path")
    parser.add_argument("--z-threshold", type=float, default=3.0,
                        help="Z-threshold to analyze")
    parser.add_argument("--detection-window", type=int, default=30,
                        help="Max steps after injection for detection")
    parser.add_argument("--json", action="store_true",
                        help="Also output raw metrics as JSON")
    parser.add_argument("--sweep", type=str, default=None,
                        help="Sweep results directory (contains sweep_summary.json)")

    args = parser.parse_args()

    if args.sweep:
        summary = load_sweep_summary(args.sweep)
        analysis = analyze_sweep(summary)
        report = generate_sweep_report(analysis, args.output)
        print(report)
        if args.json:
            json_path = args.output.replace(".md", ".json")
            with open(json_path, "w") as f:
                json.dump(analysis, f, indent=2, default=str)
            print(f"\nJSON written to {json_path}")
        return

    manifest = load_manifest(args.manifest)
    results = load_results(args.results, z_threshold=args.z_threshold)

    if not results:
        print(f"No results found in {args.results} for z={args.z_threshold}")
        return

    print(f"Loaded {len(results)} results for z={args.z_threshold}")

    metrics = compute_detection_metrics(
        results, manifest,
        detection_window=args.detection_window,
    )

    report = generate_report(metrics, args.output)
    print(report)

    if args.json:
        json_path = args.output.replace(".md", ".json")
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"\nJSON metrics written to {json_path}")


if __name__ == "__main__":
    main()
