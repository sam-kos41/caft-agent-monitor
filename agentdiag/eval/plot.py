"""Publication-quality figures for IEEE THMS.

Generates:
  1. ROC curve (double-column) — TPR vs FPR across z-thresholds
  2. Detection latency histogram (single-column) — by failure type
  3. Task x failure heatmap (double-column)
  4. Timeline example (double-column) — IT metrics with annotations
  5. Phase comparison bar chart (single-column)

Usage::

    python -m agentdiag.eval.plot --results results/ --output figures/
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from agentdiag.eval.trace_generator import FAILURE_VARIANTS, EXPECTED_SIGNATURES
from agentdiag.eval.analyze import (
    load_manifest,
    load_results,
    compute_detection_metrics,
    _parse_trace_name,
)


# ── IEEE THMS style ───────────────────────────────────────────────────────

SINGLE_COL = 3.5   # inches
DOUBLE_COL = 7.16   # inches
DPI = 300
FONT_SIZE = 9

_COLORS = {
    "loop": "#1f77b4",
    "drift": "#ff7f0e",
    "thrash": "#2ca02c",
    "stall": "#d62728",
}


def _apply_style():
    """Apply IEEE-friendly matplotlib style."""
    plt.rcParams.update({
        "font.size": FONT_SIZE,
        "font.family": "serif",
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE + 1,
        "xtick.labelsize": FONT_SIZE - 1,
        "ytick.labelsize": FONT_SIZE - 1,
        "legend.fontsize": FONT_SIZE - 1,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


# ── Figure 1: ROC Curve from sweep_summary.json ──────────────────────────

def plot_roc_curve(
    summary: dict,
    output_dir: Path,
):
    """ROC curve from sweep_summary.json.

    One curve per failure type + overall.  Each point is one z-threshold.
    z=3.0 operating point marked with a larger dot; Youden's J optimal
    marked with a diamond.
    """
    _apply_style()

    thresholds = summary["thresholds"]
    ov = summary["overall"]
    by_f = summary["by_failure"]

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, DOUBLE_COL * 0.6))

    def _find_j_optimal(tpr_list, fpr_list):
        best_j, best_idx = -1, 0
        for i, (t, f) in enumerate(zip(tpr_list, fpr_list)):
            j = t - f
            if j > best_j:
                best_j, best_idx = j, i
        return best_idx

    def _find_z_idx(target=3.0):
        return min(range(len(thresholds)), key=lambda i: abs(thresholds[i] - target))

    z3_idx = _find_z_idx(3.0)

    # Per-failure curves
    for variant in FAILURE_VARIANTS:
        vd = by_f.get(variant)
        if not vd:
            continue
        fprs, tprs = vd["fpr"], vd["tpr"]
        color = _COLORS.get(variant, "#888")
        opt_idx = _find_j_optimal(tprs, fprs)
        opt_z = thresholds[opt_idx]

        ax.plot(fprs, tprs, "o-", color=color, markersize=3, linewidth=1.3,
                label=f"{variant} (opt z={opt_z:.2f})", zorder=2)
        # z=3.0 marker (larger circle)
        ax.plot(fprs[z3_idx], tprs[z3_idx], "o", color=color,
                markersize=7, markeredgecolor="black", markeredgewidth=0.6, zorder=3)
        # Youden's J optimal (diamond)
        ax.plot(fprs[opt_idx], tprs[opt_idx], "D", color=color,
                markersize=7, markeredgecolor="black", markeredgewidth=0.8, zorder=4)

    # Overall curve
    opt_idx_ov = _find_j_optimal(ov["tpr"], ov["fpr"])
    ax.plot(ov["fpr"], ov["tpr"], "s-", color="black", markersize=3,
            linewidth=1.8, label=f"overall (opt z={thresholds[opt_idx_ov]:.2f})", zorder=2)
    ax.plot(ov["fpr"][z3_idx], ov["tpr"][z3_idx], "o", color="black",
            markersize=7, markeredgecolor="white", markeredgewidth=0.6, zorder=3)
    ax.plot(ov["fpr"][opt_idx_ov], ov["tpr"][opt_idx_ov], "D", color="black",
            markersize=7, markeredgecolor="white", markeredgewidth=0.8, zorder=4)

    # Chance diagonal
    ax.plot([0, 1], [0, 1], "k--", alpha=0.25, linewidth=0.7, label="chance")

    # Legend markers explanation
    ax.plot([], [], "o", color="gray", markersize=7, markeredgecolor="black",
            markeredgewidth=0.6, label="z = 3.0", linestyle="None")
    ax.plot([], [], "D", color="gray", markersize=7, markeredgecolor="black",
            markeredgewidth=0.8, label="optimal J", linestyle="None")

    all_fpr = ov["fpr"]
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Across Sensitivity Thresholds")
    ax.legend(loc="lower right", fontsize=FONT_SIZE - 2)
    ax.set_xlim(-0.01, max(0.20, max(all_fpr) + 0.02))
    ax.set_ylim(-0.02, 1.05)

    _save(fig, output_dir / "roc_curve")


def plot_tpr_fpr_vs_threshold(
    summary: dict,
    output_dir: Path,
):
    """Secondary plot: TPR and FPR vs z-threshold on the same axes."""
    _apply_style()

    thresholds = summary["thresholds"]
    ov = summary["overall"]

    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.8))

    ax.plot(thresholds, ov["tpr"], "o-", color="#1f77b4", markersize=3,
            linewidth=1.5, label="TPR")
    ax.plot(thresholds, ov["fpr"], "s-", color="#d62728", markersize=3,
            linewidth=1.5, label="FPR")

    # Mark z=3.0
    ax.axvline(3.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6,
               label="z = 3.0")

    ax.set_xlabel("Sensitivity Threshold (z)")
    ax.set_ylabel("Rate")
    ax.set_title("TPR and FPR vs Sensitivity Threshold")
    ax.legend(loc="center right")
    ax.set_xlim(thresholds[0] - 0.1, thresholds[-1] + 0.1)
    ax.set_ylim(-0.02, 1.05)

    _save(fig, output_dir / "tpr_fpr_vs_threshold")


# ── Figure 2: Detection Latency Histogram ────────────────────────────────

def plot_latency_histogram(
    metrics: dict,
    output_dir: Path,
):
    """Histogram of detection latency (steps) by failure type."""
    _apply_style()

    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.8))

    all_latencies = []
    labels = []
    colors = []

    for variant in FAILURE_VARIANTS:
        vdata = metrics.get("per_variant", {}).get(variant)
        if vdata and vdata.get("latency_values"):
            all_latencies.append(vdata["latency_values"])
            labels.append(variant)
            colors.append(_COLORS.get(variant, "#888"))

    if all_latencies:
        ax.hist(all_latencies, bins=15, label=labels, color=colors,
                alpha=0.7, edgecolor="white", linewidth=0.5, stacked=False)
        ax.set_xlabel("Detection Latency (steps)")
        ax.set_ylabel("Count")
        ax.set_title("Detection Latency Distribution")
        ax.legend()

    _save(fig, output_dir / "latency_histogram")


# ── Figure 3: Task x Failure Heatmap ─────────────────────────────────────

def plot_heatmap(
    metrics: dict,
    output_dir: Path,
):
    """Heatmap: rows=tasks, columns=failure types, colored by detection."""
    _apply_style()

    tasks = set()
    for variant in FAILURE_VARIANTS:
        vdata = metrics.get("per_variant", {}).get(variant, {})
        tasks.update(vdata.get("task_results", {}).keys())
    tasks = sorted(tasks)

    if not tasks:
        return

    matrix = np.full((len(tasks), len(FAILURE_VARIANTS)), np.nan)
    for j, variant in enumerate(FAILURE_VARIANTS):
        vdata = metrics.get("per_variant", {}).get(variant, {})
        for i, task in enumerate(tasks):
            tr = vdata.get("task_results", {}).get(task)
            if tr is not None:
                matrix[i, j] = 1.0 if tr["detected"] else 0.0

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, max(3, len(tasks) * 0.35)))

    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#d62728", "#2ca02c", "#cccccc"])  # red, green, gray
    # Map: 0=red (missed), 1=green (detected), NaN=gray
    display = np.where(np.isnan(matrix), 2, matrix)

    ax.imshow(display, aspect="auto", cmap=cmap, vmin=0, vmax=2)

    ax.set_xticks(range(len(FAILURE_VARIANTS)))
    ax.set_xticklabels(FAILURE_VARIANTS, rotation=45, ha="right")
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels(tasks)
    ax.set_title("Detection Success: Task x Failure Type")

    # Legend
    patches = [
        mpatches.Patch(color="#2ca02c", label="Detected"),
        mpatches.Patch(color="#d62728", label="Missed"),
        mpatches.Patch(color="#cccccc", label="N/A"),
    ]
    ax.legend(handles=patches, loc="upper right", bbox_to_anchor=(1.25, 1))

    _save(fig, output_dir / "detection_heatmap")


# ── Figure 4: Timeline Example ───────────────────────────────────────────

def plot_timeline(
    results_dir: Path,
    manifest: dict,
    output_dir: Path,
    trace_names: Optional[list[str]] = None,
    z_threshold: float = 3.0,
):
    """Timeline sparklines showing IT metrics with injection/detection annotations."""
    _apply_style()

    results = load_results(results_dir, z_threshold=z_threshold)

    # Pick up to 3 interesting traces if not specified
    if trace_names is None:
        trace_names = []
        for name in results:
            _, variant = _parse_trace_name(name)
            if variant in ("loop", "drift", "thrash") and len(trace_names) < 3:
                trace_names.append(name)

    if not trace_names:
        return

    n_traces = len(trace_names)
    fig, axes = plt.subplots(n_traces, 1, figsize=(DOUBLE_COL, 2.2 * n_traces),
                              sharex=False)
    if n_traces == 1:
        axes = [axes]

    for idx, trace_name in enumerate(trace_names):
        ax = axes[idx]
        result = results.get(trace_name)
        if not result:
            continue

        filename = f"{trace_name}.jsonl"
        info = manifest.get(filename, {})
        inject_step = info.get("inject_step")
        inject_end = info.get("inject_end")

        timeline = result.get("metrics_timeline", [])
        if not timeline:
            continue

        steps = [m["step"] for m in timeline]
        entropy = [m.get("action_entropy", 0) for m in timeline]
        mi = [m.get("action_mi", 0) for m in timeline]
        compression = [m.get("compression_ratio", 0) for m in timeline]

        ax.plot(steps, entropy, label="entropy", linewidth=0.8, alpha=0.8)
        ax.plot(steps, mi, label="MI", linewidth=0.8, alpha=0.8)
        ax.plot(steps, compression, label="compression", linewidth=0.8, alpha=0.8)

        # Injection region
        if inject_step is not None and inject_end is not None:
            ax.axvspan(inject_step, inject_end, alpha=0.15, color="red",
                       label="injection")

        # Detection points
        for a in result.get("anomalies", []):
            a_step = a.get("step", 0)
            if inject_step and inject_step - 5 <= a_step <= (inject_end or inject_step) + 40:
                ax.axvline(a_step, color="red", alpha=0.4, linewidth=0.5)

        _, variant = _parse_trace_name(trace_name)
        ax.set_title(f"{trace_name} ({variant})", fontsize=FONT_SIZE)
        ax.set_ylabel("Value")
        if idx == 0:
            ax.legend(loc="upper right", fontsize=FONT_SIZE - 2)

    axes[-1].set_xlabel("Step")
    fig.suptitle("IT Metric Timelines with Injected Failures", fontsize=FONT_SIZE + 1)
    plt.tight_layout()

    _save(fig, output_dir / "timeline_example")


# ── Figure 5: Phase Comparison Bar Chart ──────────────────────────────────

def plot_phase_comparison(
    metrics_with_phase: dict,
    metrics_without_phase: dict,
    output_dir: Path,
):
    """Bar chart: TPR with vs without phase labels, grouped by failure type."""
    _apply_style()

    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.8))

    variants = [v for v in FAILURE_VARIANTS
                if v in metrics_with_phase.get("per_variant", {})
                or v in metrics_without_phase.get("per_variant", {})]

    if not variants:
        return

    x = np.arange(len(variants))
    width = 0.35

    tpr_with = [metrics_with_phase.get("per_variant", {}).get(v, {}).get("tpr", 0)
                for v in variants]
    tpr_without = [metrics_without_phase.get("per_variant", {}).get(v, {}).get("tpr", 0)
                   for v in variants]

    ax.bar(x - width / 2, tpr_with, width, label="With phases", color="#1f77b4")
    ax.bar(x + width / 2, tpr_without, width, label="Without phases", color="#ff7f0e")

    ax.set_xlabel("Failure Type")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Phase-Conditional vs Global Detection")
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.1)

    _save(fig, output_dir / "phase_comparison")


# ── Helpers ───────────────────────────────────────────────────────────────

def _save(fig, path: Path):
    """Save figure as both PNG and PDF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path) + ".png", dpi=DPI, bbox_inches="tight")
    fig.savefig(str(path) + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}.png + .pdf")


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate publication figures")
    parser.add_argument("--results", type=str, default="agentdiag/eval/results",
                        help="Results directory")
    parser.add_argument("--manifest", type=str, default="agentdiag/eval/traces/manifest.json",
                        help="Trace manifest")
    parser.add_argument("--output", type=str, default="agentdiag/eval/figures",
                        help="Output directory for figures")
    parser.add_argument("--z-threshold", type=float, default=3.0,
                        help="Z-threshold for metrics")
    parser.add_argument("--sweep", type=str, default=None,
                        help="Sweep results dir (contains sweep_summary.json)")

    args = parser.parse_args()
    output_dir = Path(args.output)

    if args.sweep:
        from agentdiag.eval.analyze import load_sweep_summary
        summary = load_sweep_summary(args.sweep)
        print("Generating sweep figures...")
        plot_roc_curve(summary, output_dir)
        plot_tpr_fpr_vs_threshold(summary, output_dir)
        print("Done.")
        return

    results_dir = Path(args.results)
    manifest = load_manifest(args.manifest)

    results = load_results(results_dir, z_threshold=args.z_threshold)
    if not results:
        print("No results found.")
        return

    metrics = compute_detection_metrics(results, manifest)

    print("Generating figures...")
    plot_latency_histogram(metrics, output_dir)
    plot_heatmap(metrics, output_dir)
    plot_timeline(results_dir, manifest, output_dir, z_threshold=args.z_threshold)
    print("Done.")


if __name__ == "__main__":
    main()
