#!/usr/bin/env python3
"""
Compile results from all 4 experiment conditions:
  1. A-Normal:    Original trivia game build (vanilla, healthy prompts)
  2. B-Normal:    Condition B harness sessions (healthy prompts)
  3. A-BadPrompt: Condition A with injected thrash anomaly
  4. B-BadPrompt: Condition B harness with injected anomaly (if available)

Runs IP evaluation on each and produces a combined comparison report.
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict

# Add agentdiag to path
SCRIPT_DIR = Path(__file__).resolve().parent
for candidate in [SCRIPT_DIR, SCRIPT_DIR.parent, SCRIPT_DIR / "agentdiag"]:
    if (candidate / "agentdiag" / "cognitive.py").exists():
        sys.path.insert(0, str(candidate))
        break

from evaluate_sessions import replay_session, extract_ip_profile, compare_agents

RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

HOME = Path.home()
CLAUDE_PROJECTS = HOME / ".claude" / "projects"


# ---------------------------------------------------------------------------
# Session directories for each condition
# ---------------------------------------------------------------------------

EXPERIMENTS = {
    "A-Normal": {
        "label": "Condition A: Vanilla (healthy prompts)",
        "description": "Original trivia game build. Single Claude agent, no anomaly injection.",
        "session_dir": CLAUDE_PROJECTS / "-Users-samkoscelny-GazeVLM-local-agentdiag-agentdiag-trivia-game",
        "anomaly": None,
        # Exclude the current meta-session (building the experiment framework, not a real build)
        "exclude_sessions": ["505da599"],
    },
    "B-Normal": {
        "label": "Condition B: Harness (healthy prompts)",
        "description": "Harness-orchestrated sessions via condition_b_harness directory.",
        "session_dir": CLAUDE_PROJECTS / "-Users-samkoscelny-GazeVLM-local-agentdiag-agentdiag-trivia-game-condition-b-harness",
        "anomaly": None,
    },
    "A-BadPrompt": {
        "label": "Condition A: Vanilla (thrash anomaly injected)",
        "description": "3 agents with Agent B given bad prompt (build frontend blind).",
        "session_dir": CLAUDE_PROJECTS / "-Users-samkoscelny-GazeVLM-local-agentdiag-agentdiag-trivia-game-condition-a-vanilla",
        "anomaly": "thrash",
    },
    "B-BadPrompt": {
        "label": "Condition B: Harness (thrash anomaly injected)",
        "description": "Harness-orchestrated with generator given same bad prompt (build blind).",
        "session_dir": CLAUDE_PROJECTS / "-Users-samkoscelny-trivia-game-experiment-b",
        "anomaly": "thrash",
    },
}


def find_jsonl_files(session_dir: Path) -> list[str]:
    """Find all JSONL session files in a directory."""
    if session_dir is None or not session_dir.exists():
        return []
    files = sorted(session_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    return [str(f) for f in files]


def analyze_condition(name: str, config: dict) -> dict:
    """Run IP evaluation on a condition's sessions."""
    session_dir = config["session_dir"]
    jsonl_files = find_jsonl_files(session_dir)

    result = {
        "name": name,
        "label": config["label"],
        "description": config["description"],
        "anomaly": config["anomaly"],
        "session_count": len(jsonl_files),
        "total_size_kb": 0,
        "profiles": [],
        "errors": [],
    }

    # Filter out excluded sessions
    exclude = set(config.get("exclude_sessions", []))
    if exclude:
        jsonl_files = [f for f in jsonl_files if Path(f).stem[:8] not in exclude]

    if not jsonl_files:
        print(f"  {name}: No sessions found")
        return result

    total_size = sum(os.path.getsize(f) for f in jsonl_files)
    result["total_size_kb"] = total_size // 1024
    print(f"  {name}: {len(jsonl_files)} sessions, {total_size // 1024}KB total")

    for path in jsonl_files:
        fname = Path(path).stem[:8]
        size_kb = os.path.getsize(path) // 1024
        try:
            session_result = replay_session(path)
            profile = extract_ip_profile(session_result)
            result["profiles"].append(profile)
            print(f"    {fname} ({size_kb}KB): {session_result['events_processed']} events, "
                  f"{session_result['anomaly_count']} anomalies")
        except Exception as e:
            result["errors"].append(f"{fname}: {e}")
            print(f"    {fname} ({size_kb}KB): ERROR - {e}")

    return result


def compute_averages(profiles: list[dict]) -> dict:
    """Compute average metrics across profiles."""
    if not profiles:
        return {}
    n = len(profiles)
    return {
        "events": sum(p["events"] for p in profiles) / n,
        "sensory_entropy": sum(p["sensory_entropy"] for p in profiles) / n,
        "perceptual_entropy": sum(p["perceptual_entropy"] for p in profiles) / n,
        "action_mi": sum(p["action_mi"] for p in profiles) / n,
        "coherence": sum(p["coherence"] for p in profiles) / n,
        "compression": sum(p["compression"] for p in profiles) / n,
        "consolidation_rate": sum(p["consolidation_rate"] for p in profiles) / n,
        "feedback_mi": sum(p["feedback_mi"] for p in profiles) / n,
        "kl_divergence": sum(p["kl_divergence"] for p in profiles) / n,
        "total_anomalies": sum(p["total_anomalies"] for p in profiles) / n,
        "named_anomalies": sum(p["named_anomalies"] for p in profiles) / n,
    }


def generate_combined_report(conditions: dict[str, dict]) -> str:
    """Generate a combined comparison report."""
    lines = []
    lines.append("=" * 78)
    lines.append("COMBINED EXPERIMENT RESULTS: 4-Condition Comparison")
    lines.append("=" * 78)
    lines.append("")
    lines.append("Conditions:")
    lines.append("  A-Normal:    Vanilla Claude, healthy prompts (original build)")
    lines.append("  B-Normal:    Harness-orchestrated, healthy prompts")
    lines.append("  A-BadPrompt: Vanilla Claude, Agent B given thrash anomaly")
    lines.append("  B-BadPrompt: Harness-orchestrated, generator given thrash anomaly")
    lines.append("")

    # ---- Summary table ----
    lines.append("-" * 78)
    lines.append("SUMMARY TABLE")
    lines.append("-" * 78)
    header = f"  {'Metric':<28} {'A-Normal':>10} {'B-Normal':>10} {'A-Bad':>10} {'B-Bad':>10}"
    lines.append(header)
    lines.append(f"  {'-'*28} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    avgs = {}
    for name, cond in conditions.items():
        avgs[name] = compute_averages(cond["profiles"])

    metrics = [
        ("Sessions", "session_count", "{}", False),
        ("Total size (KB)", "total_size_kb", "{}", False),
        ("Avg events/session", "events", "{:.0f}", True),
        ("Avg sensory entropy (b)", "sensory_entropy", "{:.2f}", True),
        ("Avg perceptual entropy (b)", "perceptual_entropy", "{:.2f}", True),
        ("Avg action MI (b)", "action_mi", "{:.2f}", True),
        ("Avg coherence (%)", "coherence", "{:.1f}", True),
        ("Avg compression", "compression", "{:.2f}", True),
        ("Avg consolidation (%)", "consolidation_rate", "{:.1f}", True),
        ("Avg feedback MI (b)", "feedback_mi", "{:.2f}", True),
        ("Avg KL divergence", "kl_divergence", "{:.3f}", True),
        ("Avg anomalies/session", "total_anomalies", "{:.1f}", True),
        ("Avg named anomalies", "named_anomalies", "{:.1f}", True),
    ]

    for label, key, fmt, from_avg in metrics:
        vals = []
        for name in ["A-Normal", "B-Normal", "A-BadPrompt", "B-BadPrompt"]:
            if from_avg:
                v = avgs.get(name, {}).get(key, None)
            else:
                v = conditions.get(name, {}).get(key, None)
            if v is not None:
                vals.append(fmt.format(v))
            else:
                vals.append("--")
        lines.append(f"  {label:<28} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10} {vals[3]:>10}")

    lines.append("")

    # ---- Per-condition detail ----
    for name, cond in conditions.items():
        lines.append("=" * 78)
        lines.append(f"{cond['label']}")
        lines.append(f"  {cond['description']}")
        if cond["anomaly"]:
            lines.append(f"  Anomaly injected: {cond['anomaly']}")
        lines.append(f"  Sessions: {cond['session_count']}, Total: {cond['total_size_kb']}KB")
        lines.append("=" * 78)

        if not cond["profiles"]:
            lines.append("  No sessions analyzed (not yet run or no data)")
            lines.append("")
            continue

        for p in cond["profiles"]:
            lines.append(f"  Session {p['session_id']} ({p['events']} events, {p['file_size_kb']}KB)")
            lines.append(f"    Sensory: {p['sensory_entropy']:.2f}b  Perceptual: {p['perceptual_entropy']:.2f}b")
            lines.append(f"    Action MI: {p['action_mi']:.2f}b  Coherence: {p['coherence']:.1f}%")
            lines.append(f"    Compression: {p['compression']:.2f}  Consolidation: {p['consolidation_rate']:.1f}%")
            lines.append(f"    Feedback MI: {p['feedback_mi']:.2f}b  KL div: {p['kl_divergence']:.3f}")
            lines.append(f"    Anomalies: {p['total_anomalies']} total, {p['named_anomalies']} named")
            if p["signature_breakdown"]:
                sigs = ", ".join(f"{k}({v})" for k, v in
                               sorted(p["signature_breakdown"].items(), key=lambda x: -x[1]))
                lines.append(f"    Signatures: {sigs}")
        lines.append("")

    # ---- Cross-condition analysis ----
    lines.append("=" * 78)
    lines.append("CROSS-CONDITION ANALYSIS")
    lines.append("=" * 78)
    lines.append("")

    # Normal vs BadPrompt (same monitoring, different prompts)
    a_norm = avgs.get("A-Normal", {})
    a_bad = avgs.get("A-BadPrompt", {})
    if a_norm and a_bad:
        lines.append("1. Effect of bad prompt (A-Normal vs A-BadPrompt):")
        lines.append("   Same monitoring (vanilla), different prompts.")
        mi_diff = a_bad.get("action_mi", 0) - a_norm.get("action_mi", 0)
        kl_diff = a_bad.get("kl_divergence", 0) - a_norm.get("kl_divergence", 0)
        anom_diff = a_bad.get("total_anomalies", 0) - a_norm.get("total_anomalies", 0)
        lines.append(f"   Action MI change:   {mi_diff:+.2f}b")
        lines.append(f"   KL divergence change: {kl_diff:+.3f}")
        lines.append(f"   Anomaly count change: {anom_diff:+.1f}")
        if anom_diff > 0:
            lines.append("   -> Bad prompt produces MORE anomalies (expected)")
        elif anom_diff == 0:
            lines.append("   -> Same anomaly count (CAFT may need tuning)")
        lines.append("")

    # Vanilla vs Harness (same prompts, different monitoring)
    b_norm = avgs.get("B-Normal", {})
    if a_norm and b_norm:
        lines.append("2. Effect of harness (A-Normal vs B-Normal):")
        lines.append("   Same healthy prompts, different monitoring infrastructure.")
        mi_diff = b_norm.get("action_mi", 0) - a_norm.get("action_mi", 0)
        coh_diff = b_norm.get("coherence", 0) - a_norm.get("coherence", 0)
        lines.append(f"   Action MI change:   {mi_diff:+.2f}b")
        lines.append(f"   Coherence change:   {coh_diff:+.1f}%")
        lines.append("")

    # BadPrompt: Vanilla vs Harness (same bad prompt, different monitoring)
    b_bad = avgs.get("B-BadPrompt", {})
    if a_bad and b_bad:
        lines.append("3. Harness advantage on same failure (A-BadPrompt vs B-BadPrompt):")
        lines.append("   Same thrash anomaly, vanilla vs harness monitoring.")
        mi_diff = b_bad.get("action_mi", 0) - a_bad.get("action_mi", 0)
        kl_diff = b_bad.get("kl_divergence", 0) - a_bad.get("kl_divergence", 0)
        anom_diff = b_bad.get("total_anomalies", 0) - a_bad.get("total_anomalies", 0)
        named_diff = b_bad.get("named_anomalies", 0) - a_bad.get("named_anomalies", 0)
        lines.append(f"   Action MI change:     {mi_diff:+.2f}b")
        lines.append(f"   KL divergence change: {kl_diff:+.3f}")
        lines.append(f"   Anomaly count change: {anom_diff:+.1f}")
        lines.append(f"   Named anomaly change: {named_diff:+.1f}")
        if named_diff > 0:
            lines.append("   -> Harness detects MORE named anomalies (better classification)")
        lines.append("")
    elif a_bad and not b_bad:
        lines.append("3. B-BadPrompt not yet run. To complete the comparison:")
        lines.append("   python experiment.py run-b --inject-anomaly thrash --inject-both")
        lines.append("")

    # ---- Conclusion ----
    lines.append("=" * 78)
    lines.append("INTERPRETATION GUIDE")
    lines.append("=" * 78)
    lines.append("")
    lines.append("  Key metrics to compare:")
    lines.append("    - Action MI: Higher = more purposeful. Bad prompts should lower it.")
    lines.append("    - KL divergence: Higher = behavior shifted from baseline. Spikes = anomaly.")
    lines.append("    - Named anomalies: More = better classification (not just 'something weird').")
    lines.append("    - Coherence: Higher = sequential logic. Bad prompts may lower it.")
    lines.append("    - Compression: Lower = more repetitive. Loop anomaly should tank this.")
    lines.append("")
    lines.append("  What the harness should improve:")
    lines.append("    - Phase-conditional baselines -> fewer false positives on healthy agents")
    lines.append("    - Evaluation markers -> retrospective validation of anomalies")
    lines.append("    - Better anomaly classification (named vs unclassified)")
    lines.append("")

    return "\n".join(lines)


def main():
    print("=" * 60)
    print("Compiling results from all 4 experiment conditions...")
    print("=" * 60)
    print()

    # Analyze each condition
    conditions = {}
    for name, config in EXPERIMENTS.items():
        conditions[name] = analyze_condition(name, config)

    print()

    # Generate report
    report = generate_combined_report(conditions)
    print(report)

    # Save
    report_path = RESULTS_DIR / "combined_4way_report.txt"
    report_path.write_text(report)
    print(f"\nReport saved to: {report_path}")

    # Save raw data
    data_path = RESULTS_DIR / "combined_4way_data.json"
    data_path.write_text(json.dumps(conditions, indent=2, default=str))
    print(f"Raw data saved to: {data_path}")


if __name__ == "__main__":
    main()
