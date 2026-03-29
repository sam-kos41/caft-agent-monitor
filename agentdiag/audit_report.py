"""
Professional audit report generator for CAFT.

Produces a client-ready report from caft audit results, including:
- Executive summary (green/yellow/red count)
- Per-session health cards with plain-English explanations
- Dollar estimate of wasted time
- Recommendations

Usage:
    from agentdiag.audit_report import generate_audit_report
    report = generate_audit_report(audit_results, team_size=5, hourly_rate=75)
    print(report)

    # Or from CLI:
    caft audit /path/to/traces --report client_report.txt --team-size 5 --hourly-rate 75
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional


def _health_icon(health: str) -> str:
    if health == "green":
        return "[OK]"
    elif health == "yellow":
        return "[WARN]"
    elif health == "red":
        return "[FAIL]"
    return "[?]"


def _explain_anomaly(signature: str) -> str:
    """Plain-English explanation of an anomaly signature."""
    explanations = {
        "distributional_shift": (
            "The agent's behavior pattern changed significantly. It started "
            "doing very different things than it was doing before — often a sign "
            "of getting confused, losing context, or drifting off the original goal."
        ),
        "mechanical_repetition": (
            "The agent got stuck in a loop — repeating the same sequence of "
            "actions (read, edit, test, fail) without making meaningful progress. "
            "This is the most common failure mode in coding agents."
        ),
        "context_thrashing": (
            "The agent is rapidly switching between unrelated files or contexts, "
            "unable to maintain focus. Often happens when the task is too complex "
            "for the context window or when the agent lost its plan."
        ),
        "progress_stall": (
            "The agent is reading a lot but writing very little. It appears engaged "
            "but isn't producing output — often a sign of being stuck on a hard "
            "problem without admitting it."
        ),
        "premature_termination": (
            "The agent declared it was done without verifying its work. No tests "
            "were run, no output was checked. Code may have been delivered broken."
        ),
        "goal_drift": (
            "The agent started working on something different from what it was "
            "asked to do. The task distribution shifted mid-session."
        ),
    }
    return explanations.get(signature, f"Anomalous behavior detected ({signature}).")


def _estimate_wasted_time(session: dict) -> float:
    """Estimate minutes of wasted time for a problematic session.

    Heuristic: each anomaly represents ~20-30 seconds of unproductive work.
    The delay between first anomaly and when a human would notice adds
    another 5-15 minutes of unmonitored waste.
    """
    anomaly_count = session.get("anomaly_count", 0)
    events = session.get("events", 0)

    if anomaly_count == 0:
        return 0.0

    # Anomalies typically cluster in the latter half of a session
    # Estimate: anomaly_count * 0.5 min of direct waste
    # Plus: detection delay of 5-15 min (scales with session length)
    direct_waste = anomaly_count * 0.5
    detection_delay = min(15.0, events * 0.1)

    return direct_waste + detection_delay


def generate_audit_report(
    audit_results: dict,
    team_size: int = 1,
    sessions_per_week: int = 0,
    hourly_rate: float = 75.0,
    company_name: str = "Your Team",
) -> str:
    """Generate a professional audit report.

    Args:
        audit_results: Output from CAFT.audit()
        team_size: Number of developers using agents
        sessions_per_week: Estimated sessions per week (0 = auto-estimate)
        hourly_rate: Loaded engineering cost per hour
        company_name: Client name for the report header
    """
    results = audit_results.get("results", [])
    n_sessions = len(results)

    if n_sessions == 0:
        return "No sessions analyzed."

    healthy = [r for r in results if r["health"] == "green"]
    degraded = [r for r in results if r["health"] == "yellow"]
    problematic = [r for r in results if r["health"] == "red"]

    total_anomalies = sum(r["anomaly_count"] for r in results)
    anomaly_rate = len(degraded) + len(problematic)
    anomaly_pct = (anomaly_rate / n_sessions * 100) if n_sessions > 0 else 0

    # Estimate wasted time
    total_wasted_min = sum(_estimate_wasted_time(r) for r in results)
    avg_wasted_per_problem = total_wasted_min / max(anomaly_rate, 1)

    if sessions_per_week == 0:
        sessions_per_week = max(10, n_sessions * 2)  # Conservative estimate

    problem_sessions_per_week = int(sessions_per_week * anomaly_pct / 100)
    weekly_wasted_hours = problem_sessions_per_week * avg_wasted_per_problem / 60
    monthly_cost = weekly_wasted_hours * 4.3 * hourly_rate * team_size

    lines = []

    # Header
    lines.append("=" * 70)
    lines.append("CAFT AGENT MONITORING AUDIT REPORT")
    lines.append("=" * 70)
    lines.append(f"Prepared for: {company_name}")
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
    lines.append(f"Sessions analyzed: {n_sessions}")
    lines.append("")

    # Executive summary
    lines.append("EXECUTIVE SUMMARY")
    lines.append("-" * 40)
    lines.append(f"  Healthy sessions:     {len(healthy):3d}  ({len(healthy)/n_sessions*100:.0f}%)")
    lines.append(f"  Degraded sessions:    {len(degraded):3d}  ({len(degraded)/n_sessions*100:.0f}%)")
    lines.append(f"  Problematic sessions: {len(problematic):3d}  ({len(problematic)/n_sessions*100:.0f}%)")
    lines.append(f"  Total anomalies:      {total_anomalies}")
    lines.append("")

    if problematic or degraded:
        lines.append(f"  {anomaly_pct:.0f}% of sessions showed anomalous behavior.")
        lines.append(f"  CAFT detected these with zero configuration and zero training data.")
        lines.append("")
    else:
        lines.append("  All sessions appear healthy. No anomalies detected.")
        lines.append("")

    # Cost estimate
    if anomaly_rate > 0:
        lines.append("ESTIMATED IMPACT")
        lines.append("-" * 40)
        lines.append(f"  Based on {n_sessions} analyzed sessions:")
        lines.append(f"    Anomaly rate:           {anomaly_pct:.0f}%")
        lines.append(f"    Avg wasted time/issue:  {avg_wasted_per_problem:.0f} minutes")
        lines.append("")
        lines.append(f"  Projected at {sessions_per_week} sessions/week, {team_size} developers:")
        lines.append(f"    Problem sessions/week:  ~{problem_sessions_per_week}")
        lines.append(f"    Wasted hours/week:      ~{weekly_wasted_hours:.1f}")
        lines.append(f"    Monthly cost:           ~${monthly_cost:,.0f}")
        lines.append(f"    (at ${hourly_rate:.0f}/hr loaded engineering cost)")
        lines.append("")
        lines.append("  With CAFT monitoring, these issues are detected in real-time,")
        lines.append("  allowing developers to intervene before the agent wastes further time.")
        lines.append("")

    # Per-session details
    lines.append("SESSION DETAILS")
    lines.append("-" * 40)

    for r in sorted(results, key=lambda x: x["anomaly_count"], reverse=True):
        health = r["health"]
        icon = _health_icon(health)
        path = Path(r.get("path", "unknown"))
        name = path.stem[:12] if path.stem else "unknown"
        events = r.get("events", 0)
        anomalies = r.get("anomaly_count", 0)
        metrics = r.get("metrics", {})

        lines.append(f"  {icon} {name}")
        lines.append(f"      Events: {events}  |  Anomalies: {anomalies}  |  Health: {health.upper()}")

        if metrics:
            mi = metrics.get("action_mi", 0)
            kl = metrics.get("kl_divergence", 0)
            lines.append(f"      Action MI: {mi:.2f}b  |  KL divergence: {kl:.3f}")

        # Explain what went wrong
        if anomalies > 0:
            wasted = _estimate_wasted_time(r)
            lines.append(f"      Estimated wasted time: {wasted:.0f} minutes")

            # Get signature breakdown from anomalies
            sig_counts = {}
            for a in r.get("anomalies", []):
                if isinstance(a, dict):
                    sig = a.get("signature", "unclassified")
                    sig_counts[sig] = sig_counts.get(sig, 0) + 1

            if sig_counts:
                top_sig = max(sig_counts, key=sig_counts.get)
                lines.append(f"      Primary issue: {top_sig} ({sig_counts[top_sig]}x)")
                lines.append(f"      {_explain_anomaly(top_sig)}")

        lines.append("")

    # Recommendations
    lines.append("RECOMMENDATIONS")
    lines.append("-" * 40)

    if not problematic and not degraded:
        lines.append("  Your agent sessions look healthy. Consider running CAFT")
        lines.append("  in continuous monitoring mode to catch issues as they arise.")
    else:
        lines.append("  1. Set up continuous monitoring with CAFT to catch anomalies")
        lines.append("     in real-time instead of discovering them after the fact.")
        lines.append("")
        lines.append("  2. Review the problematic sessions above — the anomaly")
        lines.append("     signatures indicate specific failure modes that could be")
        lines.append("     addressed with better prompting or task decomposition.")
        lines.append("")

        if any("mechanical_repetition" in str(r.get("anomalies", "")) for r in results):
            lines.append("  3. Multiple loop/repetition anomalies detected. Consider adding")
            lines.append("     a retry limit or verification step to your agent workflows.")
            lines.append("")

        if any("distributional_shift" in str(r.get("anomalies", "")) for r in results):
            lines.append("  4. Distributional shifts detected — agents may be losing context")
            lines.append("     mid-task. Consider breaking large tasks into smaller sprints.")
            lines.append("")

    # Footer
    lines.append("=" * 70)
    lines.append("Generated by CAFT (Cognitive Agent Fault Taxonomy)")
    lines.append("Zero-training anomaly detection for AI agents")
    lines.append("https://github.com/sam-kos41/caft-agent-monitor")
    lines.append("=" * 70)

    return "\n".join(lines)
