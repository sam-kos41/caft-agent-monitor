"""Evaluation runner — feeds traces through the full CAFT pipeline.

Processes JSONL traces through ClaudeCodeAdapter → UniversalMonitor,
recording metric timelines and anomaly detections for analysis.

Usage::

    python -m agentdiag.eval.runner --trace traces/rest_api_clean.jsonl
    python -m agentdiag.eval.runner --trace-dir traces/ --output results/
    python -m agentdiag.eval.runner --all --z-threshold 3.0
    python -m agentdiag.eval.runner --all --sweep --z-range 1.5,5.0,0.5
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

from agentdiag.adapters.claude_adapter import ClaudeCodeAdapter
from agentdiag.universal_monitor import UniversalMonitor


def run_trace(
    trace_path: str | Path,
    z_threshold: float = 3.0,
    calibration_window: int = 100,
) -> dict:
    """Feed a single trace through the full pipeline and record results.

    Args:
        trace_path: Path to JSONL trace file.
        z_threshold: Sensitivity for SelfCalibratingBaseline.
        calibration_window: Number of events for calibration period.

    Returns:
        Result dict with events, anomalies, metrics timeline, and final state.
    """
    adapter = ClaudeCodeAdapter()
    monitor = UniversalMonitor(
        calibration_window=calibration_window,
        sensitivity=z_threshold,
    )

    total_parsed = 0
    anomalies = []
    metrics_timeline = []

    with open(trace_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            events = adapter.parse(raw)

            for event in events:
                result = monitor.process(event)
                total_parsed += 1

                # Record metrics for every observation (not phase markers)
                if result.get("type") == "observation":
                    metrics_entry = {
                        "step": result["step"],
                    }
                    if result.get("metrics"):
                        metrics_entry.update(result["metrics"])
                    metrics_timeline.append(metrics_entry)

                # Record anomalies
                if result.get("anomalies"):
                    anomalies.append({
                        "step": result["step"],
                        "signature": result["anomalies"].get("signature", "unknown"),
                        "severity": result["anomalies"].get("severity", "info"),
                        "metrics": result["anomalies"].get("metrics", {}),
                    })

    state = monitor.get_state()

    return {
        "trace": str(trace_path),
        "z_threshold": z_threshold,
        "calibration_window": calibration_window,
        "total_events": total_parsed,
        "total_anomalies": len(anomalies),
        "anomalies": anomalies,
        "metrics_timeline": metrics_timeline,
        "compositor_summary": state.get("compositor", {}),
        "baseline_summary": state.get("baseline", {}),
        "is_calibrating": state.get("baseline", {}).get("calibrating", True),
    }


def run_trace_dir(
    trace_dir: str | Path,
    output_dir: str | Path,
    z_threshold: float = 3.0,
    calibration_window: int = 100,
) -> dict[str, dict]:
    """Run all JSONL traces in a directory.

    Returns {filename: result_dict}.
    """
    trace_dir = Path(trace_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    traces = sorted(trace_dir.glob("*.jsonl"))

    for trace_path in traces:
        name = trace_path.stem
        print(f"  Running {name}... ", end="", flush=True)
        t0 = time.time()

        result = run_trace(
            trace_path,
            z_threshold=z_threshold,
            calibration_window=calibration_window,
        )

        elapsed = time.time() - t0
        print(f"{result['total_events']} events, "
              f"{result['total_anomalies']} anomalies ({elapsed:.1f}s)")

        # Write individual result
        result_path = output_dir / f"{name}_z{z_threshold:.2f}.json"
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)

        results[name] = result

    return results


def run_sweep(
    trace_dir: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path = "agentdiag/eval/traces/manifest.json",
    z_min: float = 1.5,
    z_max: float = 5.0,
    z_step: float = 0.25,
    calibration_window: int = 100,
) -> dict:
    """Run all traces at multiple z-thresholds and produce sweep_summary.json.

    For each threshold, runs every trace through the pipeline and computes
    detection metrics using ``compute_detection_metrics``.

    Returns the sweep summary dict (also written to ``sweep_summary.json``).
    """
    from agentdiag.eval.analyze import (
        load_manifest,
        load_results,
        compute_detection_metrics,
    )
    from agentdiag.eval.trace_generator import FAILURE_VARIANTS

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(manifest_path)

    thresholds: list[float] = []
    z = z_min
    while z <= z_max + 1e-9:
        thresholds.append(round(z, 2))
        z += z_step

    # Pre-structure for the summary
    overall_tpr: list[float] = []
    overall_fpr: list[float] = []
    overall_sig_acc: list[float] = []
    overall_latency: list[float] = []
    by_failure: dict[str, dict[str, list]] = {
        v: {"tpr": [], "fpr": [], "sig_accuracy": [], "latency_median": []}
        for v in FAILURE_VARIANTS
    }

    for z_val in thresholds:
        print(f"\n=== z-threshold = {z_val:.2f} ===")
        # Run all traces at this threshold
        results = run_trace_dir(
            trace_dir, output_dir,
            z_threshold=z_val,
            calibration_window=calibration_window,
        )

        # Compute detection metrics
        # load_results expects files named *_z{z:.1f}.json — match that format
        loaded = load_results(output_dir, z_threshold=z_val)
        metrics = compute_detection_metrics(loaded, manifest)

        ov = metrics["overall"]
        overall_tpr.append(ov["tpr"])
        overall_fpr.append(ov["fpr"])
        overall_latency.append(ov["latency_median"])

        # Overall signature accuracy across all failure types
        total_detected = 0
        total_correct = 0
        for v in FAILURE_VARIANTS:
            vdata = metrics.get("per_variant", {}).get(v, {})
            n_det = vdata.get("n_detected", 0)
            sa = vdata.get("signature_accuracy", 0)
            total_detected += n_det
            total_correct += int(round(sa * n_det))
            by_failure[v]["tpr"].append(vdata.get("tpr", 0))
            by_failure[v]["fpr"].append(ov["fpr"])  # FPR is global
            by_failure[v]["sig_accuracy"].append(sa)
            by_failure[v]["latency_median"].append(vdata.get("latency_median", 0))

        overall_sig_acc.append(total_correct / max(total_detected, 1))

    summary = {
        "thresholds": thresholds,
        "overall": {
            "tpr": overall_tpr,
            "fpr": overall_fpr,
            "sig_accuracy": overall_sig_acc,
            "latency_median": overall_latency,
        },
        "by_failure": by_failure,
    }

    with open(output_dir / "sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSweep complete: {len(thresholds)} thresholds, "
          f"summary at {output_dir / 'sweep_summary.json'}")
    return summary


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run evaluation traces through CAFT pipeline")
    parser.add_argument("--trace", type=str, help="Run a single trace file")
    parser.add_argument("--trace-dir", type=str, default="agentdiag/eval/traces",
                        help="Directory of JSONL traces")
    parser.add_argument("--output", type=str, default="agentdiag/eval/results",
                        help="Output directory for results")
    parser.add_argument("--all", action="store_true",
                        help="Run all traces in trace-dir")
    parser.add_argument("--sweep", action="store_true",
                        help="Run sensitivity sweep across z-thresholds")
    parser.add_argument("--z-threshold", type=float, default=3.0,
                        help="Baseline sensitivity z-threshold")
    parser.add_argument("--z-range", type=str, default="1.5,5.0,0.25",
                        help="Z sweep range: min,max,step")
    parser.add_argument("--calibration-window", type=int, default=100,
                        help="Number of calibration events")

    args = parser.parse_args()

    if args.trace:
        result = run_trace(args.trace, z_threshold=args.z_threshold,
                           calibration_window=args.calibration_window)
        out_path = Path(args.output)
        out_path.mkdir(parents=True, exist_ok=True)
        name = Path(args.trace).stem
        with open(out_path / f"{name}_z{args.z_threshold:.2f}.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"Events: {result['total_events']}, Anomalies: {result['total_anomalies']}")
        for a in result["anomalies"]:
            print(f"  Step {a['step']}: {a['signature']} ({a['severity']})")

    elif args.sweep:
        z_parts = args.z_range.split(",")
        z_min, z_max, z_step = float(z_parts[0]), float(z_parts[1]), float(z_parts[2])
        run_sweep(
            args.trace_dir, args.output,
            manifest_path=Path(args.trace_dir) / "manifest.json",
            z_min=z_min, z_max=z_max, z_step=z_step,
            calibration_window=args.calibration_window,
        )

    elif args.all:
        run_trace_dir(args.trace_dir, args.output,
                      z_threshold=args.z_threshold,
                      calibration_window=args.calibration_window)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
