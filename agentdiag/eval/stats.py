"""Statistical significance tests for CAFT evaluation.

Tests:
  - Binomial test: Is TPR significantly above chance?
  - Wilcoxon signed-rank: Phase-conditional vs global detection
  - Bootstrap 95% CI: On TPR, FPR, detection latency
  - Cohen's d: Effect size for phase comparison
  - Kruskal-Wallis + Dunn's: Cross-domain detection differences

Usage::

    python -m agentdiag.eval.stats --results results/ --output stats_report.md
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats as sp_stats

from agentdiag.eval.trace_generator import FAILURE_VARIANTS
from agentdiag.eval.analyze import (
    load_manifest,
    load_results,
    compute_detection_metrics,
    _parse_trace_name,
)


# ── Binomial Test ─────────────────────────────────────────────────────────

def binomial_test(n_detected: int, n_total: int, chance_rate: float) -> dict:
    """Test whether TPR is significantly above chance (FPR).

    H0: detection rate = chance_rate
    H1: detection rate > chance_rate (one-sided)
    """
    if n_total == 0:
        return {"statistic": 0, "p_value": 1.0, "significant": False}

    result = sp_stats.binomtest(n_detected, n_total, chance_rate, alternative="greater")
    return {
        "n_detected": n_detected,
        "n_total": n_total,
        "chance_rate": chance_rate,
        "observed_rate": n_detected / n_total,
        "p_value": result.pvalue,
        "significant": result.pvalue < 0.05,
        "ci_95": list(result.proportion_ci(confidence_level=0.95)),
    }


# ── Bootstrap Confidence Intervals ────────────────────────────────────────

def bootstrap_ci(
    values: list[float],
    n_resamples: int = 10000,
    ci: float = 0.95,
    statistic: str = "mean",
    seed: int = 42,
) -> dict:
    """Bootstrap confidence interval for a statistic."""
    if not values:
        return {"estimate": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "n": 0}

    rng = np.random.RandomState(seed)
    arr = np.array(values)

    stat_fn = np.mean if statistic == "mean" else np.median
    observed = float(stat_fn(arr))

    boot_stats = []
    for _ in range(n_resamples):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_stats.append(float(stat_fn(sample)))

    boot_stats = sorted(boot_stats)
    alpha = 1 - ci
    lo = boot_stats[int(alpha / 2 * n_resamples)]
    hi = boot_stats[int((1 - alpha / 2) * n_resamples)]

    return {
        "estimate": round(observed, 4),
        "ci_lower": round(lo, 4),
        "ci_upper": round(hi, 4),
        "n": len(values),
        "n_resamples": n_resamples,
        "statistic": statistic,
    }


# ── Cohen's d ─────────────────────────────────────────────────────────────

def cohens_d(group1: list[float], group2: list[float]) -> float:
    """Compute Cohen's d effect size (pooled std)."""
    if not group1 or not group2:
        return 0.0
    n1, n2 = len(group1), len(group2)
    m1, m2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled = math.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled < 1e-10:
        return 0.0
    return float((m1 - m2) / pooled)


def interpret_d(d: float) -> str:
    """Interpret Cohen's d magnitude."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


# ── Wilcoxon Signed-Rank ─────────────────────────────────────────────────

def wilcoxon_test(
    paired_with: list[float],
    paired_without: list[float],
) -> dict:
    """Wilcoxon signed-rank test for paired phase vs no-phase comparison.

    Requires at least 6 paired observations.
    """
    if len(paired_with) < 6 or len(paired_with) != len(paired_without):
        return {
            "statistic": None, "p_value": None,
            "significant": False, "reason": "insufficient paired data",
        }

    diffs = [a - b for a, b in zip(paired_with, paired_without)]
    # Check if all diffs are zero
    if all(abs(d) < 1e-10 for d in diffs):
        return {
            "statistic": 0, "p_value": 1.0, "significant": False,
            "reason": "no difference",
        }

    try:
        stat, p = sp_stats.wilcoxon(paired_with, paired_without, alternative="greater")
        return {
            "statistic": float(stat),
            "p_value": float(p),
            "significant": p < 0.05,
            "n_pairs": len(paired_with),
        }
    except Exception as e:
        return {"statistic": None, "p_value": None, "significant": False, "reason": str(e)}


# ── Kruskal-Wallis + Dunn's ──────────────────────────────────────────────

def kruskal_wallis_test(
    domain_detections: dict[str, list[int]],
) -> dict:
    """Test whether detection rate differs across domains.

    Args:
        domain_detections: {domain: [0/1 per trace indicating detection]}
    """
    groups = [v for v in domain_detections.values() if len(v) >= 2]
    labels = [k for k, v in domain_detections.items() if len(v) >= 2]

    if len(groups) < 2:
        return {"statistic": None, "p_value": None, "significant": False,
                "reason": "fewer than 2 domains with data"}

    try:
        stat, p = sp_stats.kruskal(*groups)
        result = {
            "statistic": float(stat),
            "p_value": float(p),
            "significant": p < 0.05,
            "n_groups": len(groups),
            "group_labels": labels,
        }

        # Post-hoc Dunn's test if significant
        if p < 0.05:
            pairwise = _dunns_test(groups, labels)
            result["pairwise_dunn"] = pairwise

        return result
    except Exception as e:
        return {"statistic": None, "p_value": None, "significant": False, "reason": str(e)}


def _dunns_test(groups: list[list[int]], labels: list[str]) -> list[dict]:
    """Simplified Dunn's test using rank-sum comparisons with Bonferroni."""
    n_comparisons = len(groups) * (len(groups) - 1) // 2
    results = []

    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            try:
                stat, p = sp_stats.mannwhitneyu(
                    groups[i], groups[j], alternative="two-sided"
                )
                adjusted_p = min(p * n_comparisons, 1.0)  # Bonferroni
                results.append({
                    "group_a": labels[i],
                    "group_b": labels[j],
                    "statistic": float(stat),
                    "p_raw": float(p),
                    "p_adjusted": float(adjusted_p),
                    "significant": adjusted_p < 0.05,
                })
            except Exception:
                results.append({
                    "group_a": labels[i],
                    "group_b": labels[j],
                    "significant": False,
                })

    return results


# ── Full statistical report ──────────────────────────────────────────────

def run_all_tests(
    metrics: dict,
    n_bootstrap: int = 10000,
) -> dict:
    """Run all statistical tests on evaluation metrics.

    Returns structured results for each test.
    """
    overall = metrics["overall"]
    per_variant = metrics.get("per_variant", {})

    results = {}

    # 1. Binomial test: overall TPR > FPR
    results["binomial_overall"] = binomial_test(
        n_detected=overall["n_detected"],
        n_total=overall["n_failure_traces"],
        chance_rate=max(overall["fpr"], 0.01),  # floor at 1% to avoid degenerate test
    )

    # Per-variant binomial
    results["binomial_per_variant"] = {}
    for variant, vdata in per_variant.items():
        results["binomial_per_variant"][variant] = binomial_test(
            n_detected=vdata["n_detected"],
            n_total=vdata["n_total"],
            chance_rate=max(overall["fpr"], 0.01),
        )

    # 2. Bootstrap CIs
    all_latencies = []
    for vdata in per_variant.values():
        all_latencies.extend(vdata.get("latency_values", []))

    results["bootstrap_tpr"] = bootstrap_ci(
        [1.0] * overall["n_detected"] + [0.0] * (overall["n_failure_traces"] - overall["n_detected"]),
        n_resamples=n_bootstrap,
        statistic="mean",
    )

    results["bootstrap_latency"] = bootstrap_ci(
        all_latencies,
        n_resamples=n_bootstrap,
        statistic="median",
    )

    # Bootstrap per variant
    results["bootstrap_per_variant"] = {}
    for variant, vdata in per_variant.items():
        det = [1.0] * vdata["n_detected"] + [0.0] * (vdata["n_total"] - vdata["n_detected"])
        results["bootstrap_per_variant"][variant] = {
            "tpr": bootstrap_ci(det, n_resamples=n_bootstrap),
            "latency": bootstrap_ci(vdata.get("latency_values", []),
                                     n_resamples=n_bootstrap, statistic="median"),
        }

    # 3. Kruskal-Wallis across domains
    domain_detections = defaultdict(list)
    for variant, vdata in per_variant.items():
        for task_name, tr in vdata.get("task_results", {}).items():
            # Look up domain from manifest
            domain = metrics.get("cross_task", {})
            # Just use a simple approach: group by task prefix
            for d, ddata in metrics.get("cross_task", {}).items():
                domain_detections[d]  # ensure key exists
            domain_detections[_guess_domain(task_name)].append(
                1 if tr["detected"] else 0
            )

    results["kruskal_wallis"] = kruskal_wallis_test(domain_detections)

    return results


def _guess_domain(task_name: str) -> str:
    """Rough domain guess from task name (for grouping)."""
    _MAP = {
        "rest_api": "web_app", "react_dashboard": "web_app", "chat_app": "web_app",
        "fullstack_auth": "web_app",
        "markdown_docs": "docs",
        "bash_utility": "devops", "ci_cd_setup": "devops",
        "unit_test_suite": "testing", "api_test_harness": "testing",
        "cli_data_processor": "cli_tool",
        "etl_pipeline": "data_pipeline", "ml_pipeline": "data_pipeline",
        "game_physics": "game",
    }
    return _MAP.get(task_name, "unknown")


# ── AUC from sweep ────────────────────────────────────────────────────────

def compute_auc(fpr_list: list[float], tpr_list: list[float]) -> float:
    """Compute area under the ROC curve using the trapezoidal rule.

    Points are sorted by FPR ascending before integration.  Endpoints
    are anchored at (0, 0) and (1, 1) to span the full [0, 1] range,
    following the standard ROC convention.
    """
    if len(fpr_list) < 2:
        return 0.0
    # Sort by FPR
    pairs = sorted(set(zip(fpr_list, tpr_list)))
    fpr_sorted = [p[0] for p in pairs]
    tpr_sorted = [p[1] for p in pairs]
    # Anchor endpoints: (0, 0) at the strict end and (1, 1) at the lenient end
    if fpr_sorted[-1] < 1.0:
        fpr_sorted.append(1.0)
        tpr_sorted.append(1.0)
    if fpr_sorted[0] > 0.0:
        fpr_sorted.insert(0, 0.0)
        tpr_sorted.insert(0, 0.0)
    # Trapezoidal rule
    auc = 0.0
    for i in range(1, len(fpr_sorted)):
        dx = fpr_sorted[i] - fpr_sorted[i - 1]
        avg_y = (tpr_sorted[i] + tpr_sorted[i - 1]) / 2
        auc += dx * avg_y
    return round(auc, 4)


def compute_sweep_stats(summary: dict) -> dict:
    """Compute AUC and other sweep statistics.

    Args:
        summary: sweep_summary.json contents with thresholds, overall, by_failure.

    Returns:
        Dict with AUC per failure type and overall, plus interpretations.
    """
    from agentdiag.eval.trace_generator import FAILURE_VARIANTS

    ov = summary["overall"]
    by_f = summary["by_failure"]

    overall_auc = compute_auc(ov["fpr"], ov["tpr"])

    per_failure_auc = {}
    for v in FAILURE_VARIANTS:
        vd = by_f.get(v)
        if vd:
            per_failure_auc[v] = compute_auc(vd["fpr"], vd["tpr"])

    def _interpret(auc_val):
        if auc_val >= 0.95:
            return "excellent"
        elif auc_val >= 0.90:
            return "very good"
        elif auc_val >= 0.80:
            return "good"
        elif auc_val >= 0.70:
            return "fair"
        else:
            return "poor"

    return {
        "overall_auc": overall_auc,
        "overall_interpretation": _interpret(overall_auc),
        "per_failure_auc": per_failure_auc,
        "per_failure_interpretation": {
            v: _interpret(a) for v, a in per_failure_auc.items()
        },
    }


def generate_sweep_stats_report(
    sweep_stats: dict,
    output_path: str | Path,
) -> str:
    """Generate the AUC section of the statistical report."""
    lines = []
    lines.append("# Sweep Statistical Analysis\n")

    lines.append("## Area Under the ROC Curve (AUC)\n")
    lines.append(f"- **Overall AUC**: {sweep_stats['overall_auc']:.4f} "
                  f"({sweep_stats['overall_interpretation']})")
    lines.append("")
    lines.append("| Failure Type | AUC | Interpretation |")
    lines.append("|-------------|-----|----------------|")
    for v, auc in sweep_stats.get("per_failure_auc", {}).items():
        interp = sweep_stats.get("per_failure_interpretation", {}).get(v, "")
        lines.append(f"| {v} | {auc:.4f} | {interp} |")
    lines.append("")

    report = "\n".join(lines)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)
    return report


def generate_stats_report(test_results: dict, output_path: str | Path) -> str:
    """Generate markdown report of statistical test results."""
    lines = []
    lines.append("# Statistical Analysis Report\n")

    # Binomial test
    lines.append("## 1. Binomial Test: Detection Above Chance\n")
    bt = test_results.get("binomial_overall", {})
    lines.append(f"- H0: TPR = {bt.get('chance_rate', 'N/A'):.3f} (chance = FPR)")
    lines.append(f"- H1: TPR > chance")
    lines.append(f"- Observed TPR: {bt.get('observed_rate', 0):.3f}")
    lines.append(f"- p-value: {bt.get('p_value', 'N/A'):.4f}")
    lines.append(f"- **{'Significant' if bt.get('significant') else 'Not significant'}** at alpha=0.05")
    if bt.get("ci_95"):
        lines.append(f"- 95% CI: [{bt['ci_95'][0]:.3f}, {bt['ci_95'][1]:.3f}]")
    lines.append("")

    # Per-variant binomial
    lines.append("### Per-Variant Binomial Tests\n")
    lines.append("| Variant | TPR | p-value | Significant |")
    lines.append("|---------|-----|---------|-------------|")
    for variant, bt in test_results.get("binomial_per_variant", {}).items():
        sig = "Yes" if bt.get("significant") else "No"
        lines.append(f"| {variant} | {bt.get('observed_rate', 0):.3f} | "
                      f"{bt.get('p_value', 1):.4f} | {sig} |")
    lines.append("")

    # Bootstrap CIs
    lines.append("## 2. Bootstrap 95% Confidence Intervals\n")
    bt_tpr = test_results.get("bootstrap_tpr", {})
    bt_lat = test_results.get("bootstrap_latency", {})
    lines.append(f"- **TPR**: {bt_tpr.get('estimate', 0):.3f} "
                  f"[{bt_tpr.get('ci_lower', 0):.3f}, {bt_tpr.get('ci_upper', 0):.3f}]")
    lines.append(f"- **Latency (median)**: {bt_lat.get('estimate', 0):.1f} steps "
                  f"[{bt_lat.get('ci_lower', 0):.1f}, {bt_lat.get('ci_upper', 0):.1f}]")
    lines.append(f"- Resamples: {bt_tpr.get('n_resamples', 0)}")
    lines.append("")

    # Per-variant bootstrap
    lines.append("### Per-Variant Bootstrap CIs\n")
    lines.append("| Variant | TPR [95% CI] | Latency [95% CI] |")
    lines.append("|---------|-------------|------------------|")
    for variant, bv in test_results.get("bootstrap_per_variant", {}).items():
        tpr = bv.get("tpr", {})
        lat = bv.get("latency", {})
        lines.append(
            f"| {variant} | {tpr.get('estimate', 0):.3f} "
            f"[{tpr.get('ci_lower', 0):.3f}, {tpr.get('ci_upper', 0):.3f}] | "
            f"{lat.get('estimate', 0):.1f} "
            f"[{lat.get('ci_lower', 0):.1f}, {lat.get('ci_upper', 0):.1f}] |"
        )
    lines.append("")

    # Kruskal-Wallis
    lines.append("## 3. Kruskal-Wallis: Cross-Domain Comparison\n")
    kw = test_results.get("kruskal_wallis", {})
    if kw.get("statistic") is not None:
        lines.append(f"- H statistic: {kw['statistic']:.3f}")
        lines.append(f"- p-value: {kw['p_value']:.4f}")
        lines.append(f"- **{'Significant' if kw['significant'] else 'Not significant'}** "
                      f"difference across {kw.get('n_groups', 0)} domains")
        if kw.get("pairwise_dunn"):
            lines.append("\n### Pairwise Dunn's Test (Bonferroni-corrected)\n")
            lines.append("| Group A | Group B | p (adjusted) | Significant |")
            lines.append("|---------|---------|-------------|-------------|")
            for pw in kw["pairwise_dunn"]:
                sig = "Yes" if pw.get("significant") else "No"
                lines.append(f"| {pw['group_a']} | {pw['group_b']} | "
                              f"{pw.get('p_adjusted', 1):.4f} | {sig} |")
    else:
        lines.append(f"- Skipped: {kw.get('reason', 'insufficient data')}")
    lines.append("")

    report = "\n".join(lines)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)

    return report


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Statistical analysis of CAFT evaluation")
    parser.add_argument("--results", type=str, default="agentdiag/eval/results")
    parser.add_argument("--manifest", type=str, default="agentdiag/eval/traces/manifest.json")
    parser.add_argument("--output", type=str, default="agentdiag/eval/results/stats_report.md")
    parser.add_argument("--z-threshold", type=float, default=3.0)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--json", action="store_true", help="Also output raw JSON")
    parser.add_argument("--sweep", type=str, default=None,
                        help="Sweep results dir (contains sweep_summary.json)")

    args = parser.parse_args()

    if args.sweep:
        from agentdiag.eval.analyze import load_sweep_summary
        summary = load_sweep_summary(args.sweep)
        sweep_stats = compute_sweep_stats(summary)
        report = generate_sweep_stats_report(sweep_stats, args.output)
        print(report)
        if args.json:
            json_path = args.output.replace(".md", ".json")
            with open(json_path, "w") as f:
                json.dump(sweep_stats, f, indent=2, default=str)
            print(f"\nJSON written to {json_path}")
        return

    manifest = load_manifest(args.manifest)
    results = load_results(args.results, z_threshold=args.z_threshold)

    if not results:
        print("No results found.")
        return

    metrics = compute_detection_metrics(results, manifest)
    test_results = run_all_tests(metrics, n_bootstrap=args.n_bootstrap)

    report = generate_stats_report(test_results, args.output)
    print(report)

    if args.json:
        json_path = args.output.replace(".md", ".json")
        with open(json_path, "w") as f:
            json.dump(test_results, f, indent=2, default=str)
        print(f"\nJSON written to {json_path}")


if __name__ == "__main__":
    main()
