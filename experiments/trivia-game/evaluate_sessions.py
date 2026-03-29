"""
IP Theory Evaluation of Multi-Agent Sessions
=============================================

Replays agent session traces through the full CAFT pipeline and produces
a comparative cognitive profile. No human judgment needed — the evaluation
is purely information-theoretic.

Usage:
    python evaluate_sessions.py /path/to/project-session-dir/
    python evaluate_sessions.py ~/.claude/projects/-Users-...-trivia-game/

What it measures per agent:
    - Wickens stage metrics (sensory, perceptual, attention, WM, decision, execution, feedback)
    - Anomaly count and signatures
    - Phase profile (what % of time in each cognitive state)
    - Efficiency indicators (feedback utilization, consolidation rate, coherence)
    - Cross-agent comparison (who was healthiest, who struggled)

What it tells you about harness/OpenViking integration:
    - Memory stream activity (>0 means OpenViking is connected)
    - Phase boundary count (>0 means harness is orchestrating)
    - Feedback-action MI (>0 means the agent adapts to environmental feedback)
    - Consolidation/retrieval rates (how effectively the agent uses long-term memory)
"""

import json
import glob
import sys
import os
from pathlib import Path
from collections import defaultdict

# Add the agentdiag parent to path
SCRIPT_DIR = Path(__file__).resolve().parent
AGENTDIAG_ROOT = SCRIPT_DIR
# Try common locations
for candidate in [SCRIPT_DIR, SCRIPT_DIR.parent, SCRIPT_DIR / "agentdiag"]:
    if (candidate / "agentdiag" / "cognitive.py").exists():
        AGENTDIAG_ROOT = candidate
        break
sys.path.insert(0, str(AGENTDIAG_ROOT))

from agentdiag.universal_monitor import UniversalMonitor
from agentdiag.observable import ObservableEvent, EventType


# ---------------------------------------------------------------------------
# Session replay
# ---------------------------------------------------------------------------

def replay_session(jsonl_path: str, sensitivity: float = 3.0) -> dict:
    """Replay a session JSONL through the full pipeline and capture the final state."""
    monitor = UniversalMonitor(sensitivity=sensitivity)

    # Try to import the live.py extraction function
    try:
        from agentdiag.live import _extract_trace_events_from_cc
        extractor = _extract_trace_events_from_cc
    except ImportError:
        extractor = None

    # Try the adapter
    try:
        from agentdiag.adapters.claude_adapter import ClaudeCodeAdapter
        adapter = ClaudeCodeAdapter()
    except ImportError:
        adapter = None

    events_processed = 0
    anomalies = []
    all_anomaly_steps = []

    with open(jsonl_path, "r") as f:
        raw_lines = f.readlines()

    # Method 1: Use live.py extractor (handles raw CC format)
    # _extract_trace_events_from_cc takes (raw: dict, step_counter: list[int])
    if extractor is not None:
        step_counter = [0]
        trace_events = []
        for line in raw_lines:
            try:
                raw = json.loads(line.strip())
                extracted = extractor(raw, step_counter)
                trace_events.extend(extracted)
            except (json.JSONDecodeError, KeyError, TypeError):
                continue

        if adapter is not None:
            for te in trace_events:
                try:
                    # Convert trace event dict to something the adapter can handle
                    from agentdiag.models import TraceEvent
                    if isinstance(te, dict):
                        tev = TraceEvent(**{k: te.get(k) for k in TraceEvent.__dataclass_fields__ if k in te})
                    else:
                        tev = te

                    # Try to convert to ObservableEvent
                    from agentdiag.cognitive import trace_event_to_observable
                    obs_event = trace_event_to_observable(tev)
                    if obs_event is not None:
                        result = monitor.process(obs_event)
                        events_processed += 1
                        if result and result.get("anomalies"):
                            anomalies.append(result["anomalies"])
                            all_anomaly_steps.append(events_processed)
                except Exception:
                    pass
        else:
            # Fallback: process trace events directly if possible
            for te in trace_events:
                try:
                    step = te.get("step", events_processed + 1)
                    tool = te.get("tool", "unknown")
                    target = te.get("target_path", te.get("output", ""))

                    if tool in ("Read", "Grep", "Glob"):
                        etype = EventType.FILE_READ
                    elif tool in ("Write", "Edit", "MultiEdit"):
                        etype = EventType.FILE_WRITE
                    elif tool in ("Bash",):
                        etype = EventType.SHELL_COMMAND
                    else:
                        etype = EventType.TOOL_CALL

                    obs = ObservableEvent(
                        step=step,
                        timestamp=te.get("timestamp", 0.0),
                        event_type=etype,
                        tool_name=tool,
                        target_path=str(target)[:200] if target else None,
                    )
                    result = monitor.process(obs)
                    events_processed += 1
                    if result and result.get("anomalies"):
                        anomalies.append(result["anomalies"])
                        all_anomaly_steps.append(events_processed)
                except Exception:
                    pass

    # Method 2: Direct JSONL parsing if no extractor or it found nothing
    if events_processed == 0:
        step = 0
        for line in raw_lines:
            try:
                raw = json.loads(line.strip())

                # Format A: Claude Code raw format (assistant messages with tool_use)
                if raw.get("type") == "assistant":
                    content = raw.get("message", {}).get("content", [])
                    for block in content:
                        if block.get("type") == "tool_use":
                            step += 1
                            tool = block.get("name", "unknown")
                            inp = block.get("input", {})
                            target = inp.get("file_path") or inp.get("path") or inp.get("command", "")

                            if tool in ("Read", "Grep", "Glob"):
                                etype = EventType.FILE_READ
                            elif tool in ("Write", "Edit", "MultiEdit"):
                                etype = EventType.FILE_WRITE
                            elif tool in ("Bash",):
                                etype = EventType.SHELL_COMMAND
                            else:
                                etype = EventType.TOOL_CALL

                            obs = ObservableEvent(
                                step=step,
                                timestamp=raw.get("timestamp", 0.0) if isinstance(raw.get("timestamp"), (int, float)) else 0.0,
                                event_type=etype,
                                tool_name=tool,
                                target_path=str(target)[:200] if target else None,
                            )
                            result = monitor.process(obs)
                            events_processed += 1
                            if result and result.get("anomalies"):
                                anomalies.append(result["anomalies"])
                                all_anomaly_steps.append(step)

                # Format B: eval trace format (step/type/tool at top level)
                elif "step" in raw and "tool" in raw:
                    step = raw["step"]
                    tool = raw.get("tool", "unknown")
                    target = raw.get("target_path", "")

                    if tool.lower() in ("read", "grep", "glob"):
                        etype = EventType.FILE_READ
                    elif tool.lower() in ("write", "edit", "multiedit"):
                        etype = EventType.FILE_WRITE
                    elif tool.lower() in ("bash",):
                        etype = EventType.SHELL_COMMAND
                    else:
                        etype = EventType.TOOL_CALL

                    obs = ObservableEvent(
                        step=step,
                        timestamp=raw.get("timestamp", 0.0) if isinstance(raw.get("timestamp"), (int, float)) else 0.0,
                        event_type=etype,
                        tool_name=tool,
                        target_path=str(target)[:200] if target else None,
                    )
                    result = monitor.process(obs)
                    events_processed += 1
                    if result and result.get("anomalies"):
                        anomalies.append(result["anomalies"])
                        all_anomaly_steps.append(step)

            except (json.JSONDecodeError, KeyError, TypeError):
                continue

    state = monitor.get_state()
    return {
        "path": jsonl_path,
        "session_id": Path(jsonl_path).stem[:8],
        "file_size_kb": os.path.getsize(jsonl_path) // 1024,
        "events_processed": events_processed,
        "anomaly_count": len(anomalies),
        "anomaly_steps": all_anomaly_steps,
        "anomalies": anomalies,
        "state": state,
    }


# ---------------------------------------------------------------------------
# IP profile extraction
# ---------------------------------------------------------------------------

def extract_ip_profile(result: dict) -> dict:
    """Extract a Wickens IP profile from a session result.

    The Wickens data lives inside EventRouter.to_dict()["wickens"],
    which UniversalMonitor.get_state() nests under "info_theoretic".
    Field names map to EventRouter's actual output (not percentages).
    """
    state = result.get("state", {})
    it = state.get("info_theoretic", {})

    # Wickens is nested inside info_theoretic (from EventRouter.to_dict())
    wickens = it.get("wickens", {})

    # Anomaly signature breakdown
    sig_counts = defaultdict(int)
    for a_group in result.get("anomalies", []):
        if isinstance(a_group, dict):
            sig = a_group.get("signature", "unclassified_anomaly")
            sig_counts[sig] += 1
        elif isinstance(a_group, list):
            for a in a_group:
                sig = a.get("signature", "unclassified_anomaly") if isinstance(a, dict) else "unknown"
                sig_counts[sig] += 1

    # Extract Wickens stage data (field names match EventRouter.to_dict())
    sensory = wickens.get("sensory", {})
    perceptual = wickens.get("perceptual", {})
    attention = wickens.get("attention", {})
    wm = wickens.get("working_memory", {})
    ltm = wickens.get("ltm", {})
    decision = wickens.get("response_selection", {})
    execution = wickens.get("response_execution", {})
    feedback = wickens.get("feedback", {})

    # Working memory: compute utilization from InferredWorkingMemory if available
    wm_state = state.get("working_memory", {})
    wm_active = len(wm_state.get("active_items", []))
    wm_utilization = wm_state.get("utilization", 0)

    return {
        "session_id": result["session_id"],
        "events": result["events_processed"],
        "file_size_kb": result["file_size_kb"],

        # Wickens stages — field names match EventRouter.to_dict() output
        "sensory_entropy": sensory.get("input_entropy", it.get("tool_entropy", 0)),
        "perceptual_entropy": perceptual.get("read_entropy", it.get("read_entropy", 0)),
        "perceptual_focus": round(perceptual.get("focus", 0) * 100, 1),  # 0-1 float -> percentage
        "attention_resource": round(attention.get("resource", 0) * 100, 1),  # 0-1 float -> percentage
        "wm_utilization": round(wm_utilization * 100, 1),  # from InferredWorkingMemory
        "wm_active_items": wm_active,  # from InferredWorkingMemory
        "ltm_items": ltm.get("stored_items", 0),
        "consolidation_rate": round(wm.get("consolidation_rate", 0) * 100, 1),
        "retrieval_rate": round(wm.get("retrieval_rate", ltm.get("retrieval_rate", 0)) * 100, 1),
        "action_mi": decision.get("action_mi", it.get("action_mi", 0)),
        "coherence": round(decision.get("coherence", 0) * 100, 1),  # 0-1 float -> percentage
        "compression": execution.get("compression", it.get("compression_ratio", 0)),
        "execution_efficiency": round(execution.get("efficiency", 0) * 100, 1),  # 0-1 float -> percentage
        "feedback_mi": feedback.get("feedback_action_mi", 0),
        "feedback_events": feedback.get("feedback_events", 0),

        # IT measures (top-level from EventRouter)
        "kl_divergence": it.get("kl_divergence", 0),
        "last_surprisal": it.get("last_surprisal", 0),

        # Anomalies
        "total_anomalies": result["anomaly_count"],
        "named_anomalies": sum(v for k, v in sig_counts.items() if k != "unclassified_anomaly"),
        "signature_breakdown": dict(sig_counts),

        # Infrastructure indicators
        # Phase markers and memory events live inside info_theoretic from EventRouter
        "has_memory_events": len(it.get("memory_events", [])) > 0,
        "has_phase_boundaries": len(it.get("phase_markers", [])) > 0,
        "has_evaluation_results": len(it.get("evaluation_events", [])) > 0,
        "memory_ops_count": len(it.get("memory_events", [])),
        "phase_boundary_count": len(it.get("phase_markers", [])),
    }


# ---------------------------------------------------------------------------
# Comparative analysis
# ---------------------------------------------------------------------------

def compare_agents(profiles: list[dict]) -> dict:
    """Compare IP profiles across agents and produce assessment."""

    if not profiles:
        return {"error": "No profiles to compare"}

    # Sort by coherence (best indicator of session health)
    by_coherence = sorted(profiles, key=lambda p: p["coherence"], reverse=True)
    by_anomalies = sorted(profiles, key=lambda p: p["total_anomalies"])
    by_mi = sorted(profiles, key=lambda p: p["action_mi"], reverse=True)

    # Find the healthiest and most troubled agents
    healthiest = by_coherence[0]
    most_troubled = by_coherence[-1]

    # Compute averages for baseline comparison
    n = len(profiles)
    avg = {
        "sensory_entropy": sum(p["sensory_entropy"] for p in profiles) / n,
        "action_mi": sum(p["action_mi"] for p in profiles) / n,
        "coherence": sum(p["coherence"] for p in profiles) / n,
        "compression": sum(p["compression"] for p in profiles) / n,
        "feedback_mi": sum(p["feedback_mi"] for p in profiles) / n,
        "total_anomalies": sum(p["total_anomalies"] for p in profiles) / n,
    }

    # Per-agent assessments
    assessments = []
    for p in profiles:
        issues = []
        strengths = []

        # Coherence assessment
        if p["coherence"] > 80:
            strengths.append("Highly coherent action sequences")
        elif p["coherence"] < 30:
            issues.append(f"Very low coherence ({p['coherence']:.0f}%) — actions are nearly independent")
        elif p["coherence"] < 50:
            issues.append(f"Low coherence ({p['coherence']:.0f}%) — weak sequential structure")

        # MI assessment
        if p["action_mi"] > 1.5:
            strengths.append(f"Strong mutual information ({p['action_mi']:.2f}b) — predictable, purposeful actions")
        elif p["action_mi"] < 0.5:
            issues.append(f"Very low MI ({p['action_mi']:.2f}b) — actions lack sequential logic")

        # Feedback assessment
        if p["feedback_mi"] > 0.3:
            strengths.append(f"Good feedback utilization (MI={p['feedback_mi']:.2f}b) — adapts to results")
        elif p["feedback_events"] > 0 and p["feedback_mi"] < 0.1:
            issues.append("Receives feedback but doesn't adapt (low feedback-action MI)")
        elif p["feedback_events"] == 0:
            issues.append("No feedback loop — never runs tests or checks results")

        # Consolidation assessment
        if p["consolidation_rate"] > 15:
            strengths.append(f"Active consolidation ({p['consolidation_rate']:.0f}%) — writing and committing work")
        elif p["consolidation_rate"] < 3 and p["events"] > 50:
            issues.append("Very low consolidation — reading a lot but barely writing")

        # Compression assessment
        if p["compression"] < 0.5:
            issues.append(f"Highly repetitive behavior (compression={p['compression']:.2f})")
        elif p["compression"] < 0.8 and p["compression"] > 0:
            issues.append(f"Some repetitive patterns detected (compression={p['compression']:.2f})")

        # Anomaly assessment
        if p["named_anomalies"] > 5:
            sigs = ", ".join(f"{k}({v})" for k, v in p["signature_breakdown"].items()
                           if k != "unclassified_anomaly")
            issues.append(f"Multiple named anomalies detected: {sigs}")
        elif p["total_anomalies"] > 10:
            issues.append(f"{p['total_anomalies']} anomalies detected (mostly unclassified)")

        # Attention assessment
        if p["attention_resource"] > 90:
            strengths.append("High attention allocation — broadly engaged")
        elif p["attention_resource"] < 40:
            issues.append("Low attention allocation — narrow focus or disengaged")

        # Overall health
        if not issues:
            health = "healthy"
            health_detail = "No issues detected. Agent operated within normal parameters."
        elif len(issues) <= 1 and all("low" not in i.lower() or "very" not in i.lower() for i in issues):
            health = "minor_concerns"
            health_detail = "Mostly healthy with minor observations."
        elif any("very low" in i.lower() or "stuck" in i.lower() or "repetitive" in i.lower() for i in issues):
            health = "problematic"
            health_detail = "Significant information processing issues detected."
        else:
            health = "degraded"
            health_detail = "Some issues present but agent was functional."

        assessments.append({
            "session_id": p["session_id"],
            "health": health,
            "health_detail": health_detail,
            "strengths": strengths,
            "issues": issues,
            "profile": p,
        })

    # Infrastructure assessment
    any_memory = any(p["has_memory_events"] for p in profiles)
    any_phases = any(p["has_phase_boundaries"] for p in profiles)
    any_evals = any(p["has_evaluation_results"] for p in profiles)

    infra = {
        "openviking_connected": any_memory,
        "harness_active": any_phases,
        "evaluator_running": any_evals,
        "explanation": [],
    }

    if not any_memory:
        infra["explanation"].append(
            "OpenViking is NOT connected — memory operations stream is empty. "
            "Working memory is inferred from file reads/writes only. "
            "With OpenViking: you'd see explicit L0/L1/L2 tier loading, "
            "namespace entropy, and the self-iteration loop (skills accumulating across runs)."
        )
    else:
        total_ops = sum(p["memory_ops_count"] for p in profiles)
        infra["explanation"].append(
            f"OpenViking IS connected — {total_ops} memory operations recorded. "
            "The system can track tier escalations, namespace access patterns, "
            "and skill crystallization across runs."
        )

    if not any_phases:
        infra["explanation"].append(
            "Harness is NOT active — no phase boundaries detected. "
            "Anomaly baselines are global (not phase-conditional). "
            "With harness: you'd see PLANNING→EXECUTING→VERIFYING phases, "
            "sprint contracts, and phase-conditional baselines that reduce false positives."
        )
    else:
        infra["explanation"].append(
            "Harness IS active — phase boundaries detected. "
            "Baselines are phase-conditional: what's normal during PLANNING "
            "is different from what's normal during EXECUTING."
        )

    if not any_evals:
        infra["explanation"].append(
            "Evaluator is NOT running — no QA grades in the event stream. "
            "With evaluator: you'd see retrospective markers on the timeline showing "
            "where bugs were introduced and whether IT anomalies preceded them."
        )

    return {
        "agent_count": n,
        "healthiest": healthiest["session_id"],
        "most_troubled": most_troubled["session_id"],
        "averages": avg,
        "assessments": assessments,
        "infrastructure": infra,
        "ranking_by_coherence": [p["session_id"] for p in by_coherence],
        "ranking_by_anomalies": [p["session_id"] for p in by_anomalies],
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(comparison: dict) -> str:
    """Generate a human-readable report."""
    lines = []
    lines.append("=" * 70)
    lines.append("CAFT: Information Processing Evaluation Report")
    lines.append("=" * 70)
    lines.append("")

    # Infrastructure status
    infra = comparison["infrastructure"]
    lines.append("INFRASTRUCTURE STATUS")
    lines.append("-" * 40)
    lines.append(f"  OpenViking connected:  {'YES' if infra['openviking_connected'] else 'NO'}")
    lines.append(f"  Harness active:        {'YES' if infra['harness_active'] else 'NO'}")
    lines.append(f"  Evaluator running:     {'YES' if infra['evaluator_running'] else 'NO'}")
    lines.append("")
    for exp in infra["explanation"]:
        lines.append(f"  → {exp}")
    lines.append("")

    # Overall summary
    lines.append("OVERALL SUMMARY")
    lines.append("-" * 40)
    lines.append(f"  Agents monitored:   {comparison['agent_count']}")
    lines.append(f"  Healthiest agent:   {comparison['healthiest']}")
    lines.append(f"  Most troubled:      {comparison['most_troubled']}")
    lines.append(f"  Avg coherence:      {comparison['averages']['coherence']:.1f}%")
    lines.append(f"  Avg action MI:      {comparison['averages']['action_mi']:.2f} bits")
    lines.append(f"  Avg anomalies:      {comparison['averages']['total_anomalies']:.1f}")
    lines.append("")

    # Per-agent details
    for assessment in comparison["assessments"]:
        p = assessment["profile"]
        lines.append("=" * 70)
        lines.append(f"AGENT: {p['session_id']}  ({p['events']} events, {p['file_size_kb']}KB)")
        lines.append(f"HEALTH: {assessment['health'].upper()} — {assessment['health_detail']}")
        lines.append("=" * 70)

        # Wickens profile
        lines.append("")
        lines.append("  Wickens IP Profile:")
        lines.append(f"    Sensory processing:    {p['sensory_entropy']:.2f}b entropy")
        lines.append(f"    Perceptual processing: {p['perceptual_entropy']:.2f}b entropy, {p['perceptual_focus']:.0f}% focus")
        lines.append(f"    Attention resources:    {p['attention_resource']:.0f}%")
        lines.append(f"    Working memory:         {p['wm_active_items']} active items, {p['wm_utilization']:.0f}% utilized")
        lines.append(f"    Long-term memory:       {p['ltm_items']} stored, {p['consolidation_rate']:.0f}% consolidation, {p['retrieval_rate']:.0f}% retrieval")
        lines.append(f"    Response selection:     MI={p['action_mi']:.2f}b, {p['coherence']:.0f}% coherent")
        lines.append(f"    Response execution:     compression={p['compression']:.2f}, {p['execution_efficiency']:.0f}% efficiency")
        lines.append(f"    Feedback:               MI={p['feedback_mi']:.2f}b, {p['feedback_events']} events")

        # IT measures
        lines.append("")
        lines.append("  Information-Theoretic Measures:")
        lines.append(f"    KL divergence:    {p['kl_divergence']:.3f}")
        lines.append(f"    Last surprisal:   {p['last_surprisal']:.3f}")

        # Anomalies
        lines.append("")
        lines.append(f"  Anomalies: {p['total_anomalies']} total, {p['named_anomalies']} named")
        if p["signature_breakdown"]:
            for sig, count in sorted(p["signature_breakdown"].items(), key=lambda x: -x[1]):
                lines.append(f"    {sig}: {count}")

        # Strengths and issues
        if assessment["strengths"]:
            lines.append("")
            lines.append("  Strengths:")
            for s in assessment["strengths"]:
                lines.append(f"    ✓ {s}")
        if assessment["issues"]:
            lines.append("")
            lines.append("  Issues:")
            for i in assessment["issues"]:
                lines.append(f"    ✗ {i}")

        lines.append("")

    # What harness/OpenViking would add
    lines.append("=" * 70)
    lines.append("WHAT HARNESS + OPENVIKING WOULD ADD")
    lines.append("=" * 70)
    lines.append("")

    if not infra["openviking_connected"]:
        lines.append("  Without OpenViking (current state):")
        lines.append("    - Working memory is INFERRED from file read/write recency")
        lines.append("    - No visibility into context tier loading (L0/L1/L2)")
        lines.append("    - No cross-run skill accumulation")
        lines.append("    - Memory namespace entropy unavailable → context_thrashing")
        lines.append("      signature relies on action_mi + kl_divergence only")
        lines.append("")
        lines.append("  With OpenViking:")
        lines.append("    - Working memory is DIRECTLY OBSERVABLE (L2 loads = active, L1 = background)")
        lines.append("    - Tier escalation rate becomes a first-class metric")
        lines.append("    - Cross-run learning: bug patterns, effective tests, and design")
        lines.append("      patterns accumulate in viking://agent/*/skills/")
        lines.append("    - Run N+1 starts with richer context than run N")
        lines.append("    - Memory namespace entropy enables precise context_thrashing detection")
        lines.append("")

    if not infra["harness_active"]:
        lines.append("  Without Harness (current state):")
        lines.append("    - Single global baseline for all metrics")
        lines.append("    - High entropy during exploration AND execution → more false positives")
        lines.append("    - No sprint contracts → no evaluator QA grades on timeline")
        lines.append("    - No retrospective correlation (which anomalies preceded real bugs?)")
        lines.append("")
        lines.append("  With Harness:")
        lines.append("    - Phase-conditional baselines: what's normal during PLANNING")
        lines.append("      differs from what's normal during EXECUTING")
        lines.append("    - Evaluator grades provide retrospective calibration")
        lines.append("    - Sprint contracts give the evaluator testable criteria")
        lines.append("    - The GAN-style feedback loop (evaluator critiques → generator improves)")
        lines.append("      should produce characteristic MI patterns visible in the dashboard")
        lines.append("")

    # Quantified impact estimate
    lines.append("  Estimated impact on detection quality:")
    lines.append("    - Phase-conditional baselines reduce FPR by ~30-50% (tighter per-phase std)")
    lines.append("    - OpenViking memory metrics enable context_thrashing detection on")
    lines.append("      tier escalation patterns (currently uses proxy metrics)")
    lines.append("    - Evaluator grades enable retrospective validation without human marking")
    lines.append("    - Cross-run learning means the system gets BETTER at detecting failures")
    lines.append("      specific to your codebase and workflow over time")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluate_sessions.py <session-dir-or-glob>")
        print("")
        print("Examples:")
        print("  python evaluate_sessions.py ~/.claude/projects/-Users-...-trivia-game/")
        print("  python evaluate_sessions.py ~/.claude/projects/-Users-...-trivia-game/*.jsonl")
        print("  python evaluate_sessions.py ../eval/traces/")
        sys.exit(1)

    target = sys.argv[1]

    # Find JSONL files
    if os.path.isdir(target):
        jsonl_files = sorted(glob.glob(os.path.join(target, "*.jsonl")))
    elif "*" in target:
        jsonl_files = sorted(glob.glob(target))
    else:
        jsonl_files = [target]

    if not jsonl_files:
        print(f"No JSONL files found at: {target}")
        sys.exit(1)

    print(f"Found {len(jsonl_files)} session(s) to analyze")
    print()

    # Replay each session
    results = []
    for path in jsonl_files:
        name = Path(path).stem[:8]
        size = os.path.getsize(path) // 1024
        print(f"  Replaying {name} ({size}KB)...", end=" ", flush=True)
        try:
            result = replay_session(path)
            print(f"{result['events_processed']} events, {result['anomaly_count']} anomalies")
            results.append(result)
        except Exception as e:
            print(f"ERROR: {e}")

    if not results:
        print("No sessions could be processed.")
        sys.exit(1)

    # Extract profiles
    profiles = [extract_ip_profile(r) for r in results]

    # Compare
    comparison = compare_agents(profiles)

    # Generate and print report
    report = generate_report(comparison)
    print()
    print(report)

    # Save report
    report_path = "ip_evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report saved to: {report_path}")

    # Save raw data as JSON
    json_path = "ip_evaluation_data.json"
    with open(json_path, "w") as f:
        json.dump({
            "profiles": profiles,
            "comparison": {k: v for k, v in comparison.items() if k != "assessments"},
            "assessments": [{k: v for k, v in a.items() if k != "profile"} for a in comparison["assessments"]],
        }, f, indent=2, default=str)
    print(f"Raw data saved to: {json_path}")


if __name__ == "__main__":
    main()
