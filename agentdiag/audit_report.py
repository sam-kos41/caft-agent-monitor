"""
Session behavioral-profile report generator for CAFT.

Produces a descriptive report from caft audit results:
- Summary by behavioral state (steady / phase_shifting / looping)
- Per-session IT-signature breakdown with plain-English explanations
- An explicit "how to read this" caveat

NOT included (deliberately, see docs/CONSTRUCT_REVISION.md): any
quality/health verdict or dollar-cost-of-waste estimate. Those were
not supported by the math and were removed. The team_size/hourly_rate
parameters are retained for signature compatibility but ignored.

Usage:
    from agentdiag.audit_report import generate_audit_report
    print(generate_audit_report(audit_results))
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

    # Behavioral state is DESCRIPTIVE, not a quality verdict — see
    # docs/CONSTRUCT_REVISION.md. The prior healthy/degraded/problematic
    # framing and dollar-cost estimate were not supported by the math
    # and have been removed.
    def _state(r):
        return r.get("behavioral_state", r.get("health", "unknown"))

    steady = [r for r in results if _state(r) == "steady"]
    phase_shifting = [r for r in results if _state(r) == "phase_shifting"]
    looping = [r for r in results if _state(r) == "looping"]
    total_anomalies = sum(r["anomaly_count"] for r in results)

    lines = []
    lines.append("=" * 70)
    lines.append("CAFT SESSION BEHAVIORAL PROFILE")
    lines.append("=" * 70)
    lines.append(f"Prepared for: {company_name}")
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
    lines.append(f"Sessions analyzed: {n_sessions}")
    lines.append("")
    lines.append("NOTE: 'Behavioral state' describes the information-theoretic")
    lines.append("shape of each session (how repetitive / how varied). It is")
    lines.append("NOT a quality or success verdict — a long productive research")
    lines.append("session is legitimately 'phase_shifting'; a focused search is")
    lines.append("legitimately 'looping'. Interpretation requires context.")
    lines.append("")

    lines.append("SUMMARY")
    lines.append("-" * 40)
    lines.append(f"  steady          {len(steady):3d}  ({len(steady)/n_sessions*100:.0f}%)  stable, varied flow")
    lines.append(f"  phase_shifting  {len(phase_shifting):3d}  ({len(phase_shifting)/n_sessions*100:.0f}%)  action mix changes over time")
    lines.append(f"  looping         {len(looping):3d}  ({len(looping)/n_sessions*100:.0f}%)  repetition dominates")
    lines.append(f"  IT-anomaly windows (within-session deviation): {total_anomalies}")
    lines.append("")

    # Per-session details
    lines.append("SESSION DETAILS")
    lines.append("-" * 40)

    for r in sorted(results, key=lambda x: x["anomaly_count"], reverse=True):
        state = r.get("behavioral_state", r.get("health", "unknown"))
        path = Path(r.get("path", "unknown"))
        name = path.stem[:12] if path.stem else "unknown"
        events = r.get("events", 0)
        anomalies = r.get("anomaly_count", 0)
        metrics = r.get("metrics", {})

        lines.append(f"  [{state}] {name}")
        lines.append(f"      Events: {events}  |  IT-anomaly windows: {anomalies}")

        if metrics:
            mi = metrics.get("action_mi", 0)
            kl = metrics.get("kl_divergence", 0)
            lines.append(f"      Action MI: {mi:.2f}b  |  KL divergence: {kl:.3f}")

        if anomalies > 0:
            # Signature breakdown — descriptive labels, no time/cost claim
            sig_counts = {}
            for a in r.get("anomalies", []):
                if isinstance(a, dict):
                    sig = a.get("signature", "unclassified")
                    sig_counts[sig] = sig_counts.get(sig, 0) + 1

            if sig_counts:
                top_sig = max(sig_counts, key=sig_counts.get)
                lines.append(f"      Dominant IT signature: {top_sig} ({sig_counts[top_sig]}x)")
                lines.append(f"      {_explain_anomaly(top_sig)}")

        lines.append("")

    # How to read this — deliberately not "recommendations" / not a sales
    # pitch. The honest framing per docs/CONSTRUCT_REVISION.md.
    lines.append("HOW TO READ THIS")
    lines.append("-" * 40)
    lines.append("  These are descriptive behavioral profiles, not quality")
    lines.append("  verdicts. 'looping' / high repetition can be a focused")
    lines.append("  search (fine) OR a stuck agent (not fine) — the IT")
    lines.append("  signature cannot tell which; that needs human or")
    lines.append("  task-outcome context. CAFT surfaces the shape; it does")
    lines.append("  not adjudicate success. Use the per-session signatures")
    lines.append("  as a place to LOOK, not as a conclusion.")
    lines.append("")

    lines.append("=" * 70)
    lines.append("Generated by CAFT — information-theoretic session profiling")
    lines.append("Behavioral descriptors, not validated quality verdicts")
    lines.append("https://github.com/sam-kos41/caft-agent-monitor")
    lines.append("=" * 70)

    return "\n".join(lines)
