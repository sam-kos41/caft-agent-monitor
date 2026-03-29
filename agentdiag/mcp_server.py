"""
CAFT MCP Server — AI agent monitoring via Model Context Protocol.

Makes CAFT monitoring available inside Claude Code, Claude Desktop, Cursor,
or any MCP-compatible client. The LLM becomes the dashboard.

Setup (Claude Code):
    Add to ~/.claude/settings.json or project .mcp.json:
    {
      "mcpServers": {
        "caft": {
          "command": "python",
          "args": ["-m", "agentdiag.mcp_server"]
        }
      }
    }

    Or if installed via pip/uvx:
    {
      "mcpServers": {
        "caft": {
          "command": "uvx",
          "args": ["caft-mcp"]
        }
      }
    }

Then ask Claude:
    "What's my agent health status?"
    "Audit my recent coding sessions"
    "Analyze the session at ~/.claude/projects/.../abc123.jsonl"
    "What happened at step 142?"
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

# Initialize the MCP server
mcp = FastMCP(
    "caft",
    instructions=(
        "CAFT: Zero-config anomaly detection for AI coding agents. "
        "Monitors agent behavior using information theory and detects "
        "failures without training data. Use caft_status for health, "
        "caft_detect to find traces, caft_analyze for single sessions, "
        "caft_audit for batch analysis, caft_explain for anomaly details."
    ),
)

# ---------------------------------------------------------------------------
# Lazy imports — avoid importing heavy modules until needed
# ---------------------------------------------------------------------------

_plugin = None
_monitor_state = {}


def _get_plugin():
    """Lazy-load the CAFT plugin to avoid import-time overhead."""
    global _plugin
    if _plugin is None:
        # Add agentdiag to path if needed
        import sys
        server_dir = Path(__file__).resolve().parent.parent
        if str(server_dir) not in sys.path:
            sys.path.insert(0, str(server_dir))

        from agentdiag.plugin import CAFT
        _plugin = CAFT(sensitivity=2.0)
    return _plugin


def _get_detect():
    """Lazy-load the detect function."""
    from agentdiag.plugin import detect_agents
    return detect_agents


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def caft_detect() -> str:
    """Scan the system for AI agent trace files.

    Discovers traces from Claude Code, Codex, Cursor, Aider, and custom
    locations (set CAFT_TRACES env var). Returns a list of discovered
    projects with session counts and sizes.
    """
    detect = _get_detect()
    sources = detect()

    if not sources:
        return (
            "No agent traces detected on this system.\n\n"
            "Supported locations:\n"
            "  - Claude Code: ~/.claude/projects/\n"
            "  - Codex: ~/.codex/\n"
            "  - Cursor: ~/.cursor/\n"
            "  - Custom: set CAFT_TRACES=/path/to/traces\n"
        )

    lines = [f"Found {len(sources)} agent trace source(s):\n"]
    for s in sources:
        lines.append(f"  {s.name}")
        lines.append(f"    Path: {s.path}")
        lines.append(f"    Sessions: {s.session_count}")
        lines.append(f"    Size: {s.total_size_kb}KB")
        lines.append("")

    return "\n".join(lines)


@mcp.tool()
def caft_analyze(session_path: str) -> str:
    """Analyze a single agent session file and return its health profile.

    Replays the session through the full CAFT pipeline and returns:
    - Health status (green/yellow/red)
    - Key metrics (action MI, entropy, KL divergence)
    - Anomaly count and types
    - Plain-English assessment

    Args:
        session_path: Absolute path to a JSONL session file.
    """
    caft = _get_plugin()

    path = Path(session_path).expanduser()
    if not path.exists():
        return f"File not found: {session_path}"

    result = caft._analyze_session(str(path))

    health = result["health"]
    events = result["events"]
    anomalies = result["anomaly_count"]
    metrics = result.get("metrics", {})

    health_icon = {"green": "OK", "yellow": "WARN", "red": "FAIL"}.get(health, "?")
    mi = metrics.get("action_mi", 0)
    kl = metrics.get("kl_divergence", 0)
    entropy = metrics.get("tool_entropy", 0)

    lines = [
        f"Health: {health_icon} {health.upper()}",
        f"Events: {events}",
        f"Anomalies: {anomalies}",
        "",
        "Metrics:",
        f"  Action MI: {mi:.2f} bits (higher = more purposeful)",
        f"  Tool entropy: {entropy:.2f} bits (diversity of actions)",
        f"  KL divergence: {kl:.3f} (behavior shift from baseline)",
        "",
    ]

    # Assessment
    if health == "green":
        lines.append("Assessment: This session looks healthy. The agent worked")
        lines.append("purposefully with consistent behavior throughout.")
    elif health == "yellow":
        lines.append("Assessment: Some minor irregularities detected. The agent")
        lines.append("may have had brief periods of confusion or context loss,")
        lines.append("but overall completed its task.")
    else:
        lines.append("Assessment: Significant problems detected. The agent's")
        lines.append("behavior pattern broke down — likely got stuck in a loop,")
        lines.append("lost context, or drifted off the original goal.")

        # Explain specific anomalies
        if anomalies > 0:
            lines.append("")
            sig_counts = {}
            for a in result.get("anomalies", []):
                if isinstance(a, dict):
                    sig = a.get("signature", "unclassified")
                    sig_counts[sig] = sig_counts.get(sig, 0) + 1

            if sig_counts:
                lines.append("Anomaly breakdown:")
                for sig, count in sorted(sig_counts.items(), key=lambda x: -x[1]):
                    lines.append(f"  {sig}: {count}x")

            if mi < 0.5:
                lines.append("")
                lines.append(f"Action MI is very low ({mi:.2f}b) — the agent's actions")
                lines.append("had almost no sequential logic. This typically means")
                lines.append("the agent was thrashing between unrelated operations.")

    return "\n".join(lines)


@mcp.tool()
def caft_audit(traces_dir: str = "") -> str:
    """Batch audit all agent sessions in a directory.

    Analyzes every JSONL file and returns a summary showing which sessions
    were healthy, degraded, or problematic. Includes a cost estimate of
    wasted developer time.

    Args:
        traces_dir: Path to directory with trace files. If empty, auto-detects
                    the most recent Claude Code project.
    """
    caft = _get_plugin()

    # Auto-detect if no path given
    if not traces_dir:
        detect = _get_detect()
        sources = detect()
        if not sources:
            return "No agent traces found. Specify a traces_dir path."
        # Pick the most recent non-empty source
        sources_sorted = sorted(
            [s for s in sources if s.session_count > 0],
            key=lambda s: s.path.stat().st_mtime if s.path.exists() else 0,
            reverse=True,
        )
        if not sources_sorted:
            return "No agent sessions found."
        traces_dir = str(sources_sorted[0].path)

    path = Path(traces_dir).expanduser()
    if not path.exists():
        return f"Directory not found: {traces_dir}"

    results = caft.audit(str(path), verbose=False)

    if "error" in results:
        return results["error"]

    # Generate professional report
    try:
        from agentdiag.audit_report import generate_audit_report
        return generate_audit_report(results, company_name="Agent Audit")
    except ImportError:
        pass

    # Fallback: simple summary
    n = results["sessions"]
    lines = [
        f"Audited {n} sessions from {path.name}",
        "",
        f"  Healthy:     {results['healthy']}",
        f"  Degraded:    {results['degraded']}",
        f"  Problematic: {results['problematic']}",
        f"  Total anomalies: {results['total_anomalies']}",
        "",
    ]

    for r in results.get("results", []):
        health = r["health"]
        icon = {"green": "OK", "yellow": "WARN", "red": "FAIL"}.get(health, "?")
        name = Path(r.get("path", "?")).stem[:12]
        lines.append(f"  [{icon}] {name}: {r['events']} events, "
                     f"{r['anomaly_count']} anomalies")

    return "\n".join(lines)


@mcp.tool()
def caft_explain(anomaly_signature: str) -> str:
    """Explain what a CAFT anomaly signature means in plain English.

    Args:
        anomaly_signature: The signature name (e.g., "mechanical_repetition",
                          "distributional_shift", "context_thrashing").
    """
    explanations = {
        "distributional_shift": {
            "what": (
                "The agent's behavior pattern changed significantly mid-session. "
                "It started doing very different things than before."
            ),
            "why": (
                "This usually happens when the agent loses context, hits an unexpected "
                "error, or drifts away from its original goal. The information-theoretic "
                "signature is a spike in KL divergence (the current action distribution "
                "diverges from the learned baseline)."
            ),
            "action": (
                "Check what changed at the flagged step. Did the agent encounter an "
                "error? Did it start reading files unrelated to the task? Consider "
                "restarting with a more specific prompt or breaking the task into "
                "smaller pieces."
            ),
        },
        "mechanical_repetition": {
            "what": (
                "The agent got stuck in a loop — repeating the same sequence of "
                "actions (typically: read file, edit file, run test, see failure, "
                "repeat) without making meaningful progress."
            ),
            "why": (
                "The compression ratio stays high while action mutual information "
                "drops — the agent is generating predictable sequences but not "
                "learning from feedback. Common in debugging loops where the agent "
                "keeps trying the same fix."
            ),
            "action": (
                "Intervene and redirect. The agent likely needs a different approach "
                "entirely. Consider: explaining the error differently, providing more "
                "context about the codebase, or breaking the problem into a simpler "
                "first step."
            ),
        },
        "context_thrashing": {
            "what": (
                "The agent is rapidly switching between unrelated files and contexts, "
                "unable to maintain focus on one area long enough to make progress."
            ),
            "why": (
                "High tool entropy combined with low action MI — lots of diverse "
                "actions but no sequential logic. Often happens when the context "
                "window is overloaded or the task requires information the agent "
                "can't hold in memory."
            ),
            "action": (
                "Reduce the scope. Tell the agent to focus on one file or one "
                "component at a time. If the task requires coordinating across "
                "many files, provide a summary or architecture document upfront."
            ),
        },
        "progress_stall": {
            "what": (
                "The agent is reading extensively but writing almost nothing. "
                "It appears busy but isn't producing output."
            ),
            "why": (
                "Low consolidation rate — lots of file reads, few writes. The "
                "agent may be stuck trying to understand a complex codebase, or "
                "it may be uncertain about what to do and stalling."
            ),
            "action": (
                "Give the agent a specific first step: 'Start by creating X file "
                "with Y function.' Breaking the analysis paralysis with a concrete "
                "action often unblocks progress."
            ),
        },
        "premature_termination": {
            "what": (
                "The agent declared it was done without verifying its work. "
                "No tests were run, no output was checked."
            ),
            "why": (
                "Low feedback MI — the agent never closed the loop between "
                "writing code and checking if it works. This is the most common "
                "source of silently broken deliverables."
            ),
            "action": (
                "Always include a verification step in your prompts: 'After "
                "writing the code, run the tests and fix any failures.' Consider "
                "adding automated verification to your agent workflow."
            ),
        },
    }

    sig = anomaly_signature.lower().strip()
    if sig in explanations:
        e = explanations[sig]
        return (
            f"Anomaly: {anomaly_signature}\n\n"
            f"What happened:\n{e['what']}\n\n"
            f"Why it happened:\n{e['why']}\n\n"
            f"What to do:\n{e['action']}"
        )

    return (
        f"Unknown anomaly signature: '{anomaly_signature}'\n\n"
        f"Known signatures: {', '.join(explanations.keys())}\n\n"
        f"This may be an unclassified anomaly — the IT metrics flagged "
        f"unusual behavior but it didn't match a known failure pattern."
    )


@mcp.tool()
def caft_status() -> str:
    """Get the current CAFT monitoring status.

    Returns health (green/yellow/red), key metrics, and recent activity.
    If no session is actively being monitored, reports on the most recent
    completed session.
    """
    caft = _get_plugin()

    # Check if we have an active monitor
    status = caft.status()
    if status.get("health") != "not_started" and status.get("event_count", 0) > 0:
        health = status["health"]
        icon = {"green": "OK", "yellow": "WARN", "red": "FAIL"}.get(health, "?")
        it = status.get("info_theoretic", {})
        return (
            f"Health: {icon} {health.upper()}\n"
            f"Events: {status['event_count']}\n"
            f"Anomalies: {status['anomaly_count']}\n"
            f"Uptime: {status.get('uptime_seconds', 0):.0f}s\n"
            f"\nMetrics:\n"
            f"  Action MI: {it.get('action_mi', 0):.2f}b\n"
            f"  Entropy: {it.get('tool_entropy', 0):.2f}b\n"
            f"  KL divergence: {it.get('kl_divergence', 0):.3f}\n"
        )

    # No active session — find and analyze the most recent one
    detect = _get_detect()
    sources = detect()
    if not sources:
        return "No active monitoring and no agent traces found."

    # Find most recent session file
    latest_file = None
    latest_mtime = 0
    for source in sources:
        for f in source.path.glob("*.jsonl"):
            mt = f.stat().st_mtime
            if mt > latest_mtime:
                latest_mtime = mt
                latest_file = f

    if latest_file is None:
        return "No active monitoring and no session files found."

    # Analyze the most recent session
    result = caft._analyze_session(str(latest_file))
    health = result["health"]
    icon = {"green": "OK", "yellow": "WARN", "red": "FAIL"}.get(health, "?")
    age_min = (time.time() - latest_mtime) / 60

    return (
        f"No active monitoring. Most recent session:\n\n"
        f"  File: {latest_file.name}\n"
        f"  Age: {age_min:.0f} minutes ago\n"
        f"  Health: {icon} {health.upper()}\n"
        f"  Events: {result['events']}\n"
        f"  Anomalies: {result['anomaly_count']}\n"
        f"\nRun `caft_analyze` with the full path for detailed analysis."
    )


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------

@mcp.resource("caft://health")
def get_health() -> str:
    """Current CAFT health status."""
    return caft_status()


@mcp.resource("caft://signatures")
def get_signatures() -> str:
    """List of anomaly signatures CAFT can detect."""
    return (
        "CAFT Anomaly Signatures:\n\n"
        "1. distributional_shift — Agent's behavior pattern changed significantly\n"
        "2. mechanical_repetition — Agent stuck in a read/edit/test/fail loop\n"
        "3. context_thrashing — Rapid switching between unrelated files\n"
        "4. progress_stall — Reading a lot, writing nothing\n"
        "5. premature_termination — Delivered without testing\n"
        "6. goal_drift — Working on something other than the requested task\n"
        "\nUse caft_explain(signature_name) for detailed explanations."
    )


@mcp.resource("caft://about")
def get_about() -> str:
    """About CAFT agent monitoring."""
    return (
        "CAFT (Cognitive Agent Fault Taxonomy)\n"
        "=====================================\n\n"
        "Zero-training anomaly detection for AI coding agents.\n\n"
        "CAFT monitors agent behavior using information theory — entropy,\n"
        "mutual information, KL divergence, and compression — to detect\n"
        "when an agent gets stuck, drifts off goal, or wastes time.\n\n"
        "No training data needed. No configuration needed. Works on any\n"
        "agent that produces tool call logs (Claude Code, Codex, Cursor, etc.).\n\n"
        "Tools available:\n"
        "  caft_status   — Current health status\n"
        "  caft_detect   — Find agent traces on this system\n"
        "  caft_analyze  — Analyze a single session\n"
        "  caft_audit    — Batch audit all sessions\n"
        "  caft_explain  — Explain an anomaly signature\n"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Run the CAFT MCP server."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
