#!/usr/bin/env python3
"""End-to-end demo of the agentdiag pipeline.

Demonstrates the ENTIRE pipeline on a single trace: event ingestion,
HTA phase classification, CAFT detection, and (optionally) LLM
confirmation — with step-by-step narrated output.

This is the SINGLE artifact that proves the system works.

Usage:
    # Built-in synthetic trace (no dependencies)
    python scripts/demo_e2e.py

    # With JSON output
    python scripts/demo_e2e.py --json

    # Real trace file
    python scripts/demo_e2e.py --trace ~/.claude/projects/session.jsonl

    # With mock LLM confirmation
    python scripts/demo_e2e.py --confirm

Also available via CLI:
    agentdiag monitor --demo e2e
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Add package to path for script invocation
sys.path.insert(0, str(Path(__file__).parent.parent))

from agentdiag.models import TraceEvent
from agentdiag.hta import HTAStateMachine, Phase
from agentdiag.caft.base import CaftDiagnosis, CaftSeverity
from agentdiag.caft.detectors import ALL_CAFT_DETECTORS_FULL, run_caft_detectors
from agentdiag.monitor import MonitorEngine


# ── Built-in synthetic trace ────────────────────────────────────────

def _e(step: int, type: str = "tool_call", tool: str | None = None, **kw) -> TraceEvent:
    """Shorthand for building trace events."""
    return TraceEvent(step=step, type=type, tool=tool, **kw)


# Ground truth: 2 planted failures
# 1. premature_termination: agent says 'done' at step 14 without verifying
# 2. context_loss: re-reads file from step 1 after 8+ intervening ops (step 12)
DEMO_FAILURES = {
    "premature_termination": {"onset_step": 14, "description": "Agent delivers without verification"},
    "context_loss": {"onset_step": 12, "description": "Agent re-reads file after 8+ intervening ops"},
}


def build_demo_trace() -> list[TraceEvent]:
    """Build the canonical demo trace with 2 planted failures.

    18 events, covering the full HTA lifecycle with clear failure modes:

    Steps  1-3:  GATHERING  — read files, search codebase
    Steps  4-6:  PLANNING   — reasoning about approach
    Steps  7-11: EXECUTING  — write code, edit files
    Step  12:    GATHERING  — context_loss (re-read file from step 1)
    Steps 13-14: EXECUTING  — more edits
    Steps 15-18: DELIVERING — premature_termination (no VERIFYING phase)
    """
    events = [
        # Phase 1: GATHERING (steps 1-3)
        _e(1, tool="read_file", latency_ms=250, output_hash="config_v1",
           goal_text="Read config.py"),
        _e(2, tool="grep", latency_ms=180, output_hash="grep_auth",
           goal_text="Search for auth handler"),
        _e(3, tool="read_file", latency_ms=300, output_hash="handler_v1",
           goal_text="Read auth_handler.py"),

        # Phase 2: PLANNING (steps 4-6)
        _e(4, type="reasoning", goal_text="Plan the authentication fix"),
        _e(5, type="reasoning", goal_text="Design token refresh logic"),
        _e(6, type="planning", goal_text="Outline implementation steps"),

        # Phase 3: EXECUTING (steps 7-11)
        _e(7, tool="edit_file", latency_ms=400, goal_text="Update auth_handler.py"),
        _e(8, tool="write_file", latency_ms=350, goal_text="Create token_refresh.py"),
        _e(9, tool="edit_file", latency_ms=300, goal_text="Update config.py"),
        _e(10, tool="bash", latency_ms=500, goal_text="Install dependency"),
        _e(11, tool="edit_file", latency_ms=280, goal_text="Fix import"),

        # Phase 4: CONTEXT LOSS — re-read config.py from step 1 (same output_hash)
        # 10 intervening ops between step 1 and step 12
        _e(12, tool="read_file", latency_ms=250, output_hash="config_v1",
           goal_text="Read config.py again"),

        # More executing
        _e(13, tool="edit_file", latency_ms=350, goal_text="Final edit to config"),
        _e(14, tool="write_file", latency_ms=300, goal_text="Write test stub"),

        # Phase 5: DELIVERING without VERIFYING = premature_termination
        # 4 delivering events (> MIN_DELIVERING_EVENTS=3)
        _e(15, tool="git_commit", latency_ms=500, goal_text="Commit changes"),
        _e(16, tool="push", latency_ms=600, goal_text="Push to remote"),
        _e(17, tool="git_commit", latency_ms=450, goal_text="Second commit"),
        _e(18, tool="push", latency_ms=550, goal_text="Final push"),
    ]
    return events


# ── Mock LLM confirmation ───────────────────────────────────────────

class MockLLMConfirmer:
    """Mock LLM that confirms planted failures and rejects FPs.

    For the demo, this simulates what a real LLM would do:
    - Confirms step_repetition (obvious repeated reads)
    - Confirms premature_termination (no verification before delivery)
    - Confirms context_loss (re-read same file after extensive work)
    - Rejects any spurious detections
    """

    KNOWN_FAILURES = {"premature_termination", "context_loss"}

    def confirm(self, diagnosis: CaftDiagnosis) -> dict:
        is_real = diagnosis.failure_name in self.KNOWN_FAILURES
        return {
            "confirmed": is_real,
            "confidence": 0.92 if is_real else 0.15,
            "reasoning": (
                f"Confirmed: {diagnosis.description}"
                if is_real
                else f"Rejected: Normal workflow pattern, not a real {diagnosis.failure_name}"
            ),
            "status": "confirmed" if is_real else "rejected",
        }


# ── Narration helpers ───────────────────────────────────────────────

def _phase_symbol(phase: Phase) -> str:
    """Simple ASCII symbol for a phase."""
    return {
        Phase.IDLE: ".",
        Phase.GATHERING: "?",
        Phase.PLANNING: "~",
        Phase.EXECUTING: ">",
        Phase.VERIFYING: "!",
        Phase.DELIVERING: "*",
    }.get(phase, ".")


def _severity_marker(sev: CaftSeverity) -> str:
    return {
        CaftSeverity.INFO: "[info]",
        CaftSeverity.WARNING: "[WARN]",
        CaftSeverity.CRITICAL: "[CRIT]",
    }.get(sev, "[????]")


# ── Try Rich, fall back to plain ────────────────────────────────────

def _try_rich():
    """Check if Rich is available and return Console or None."""
    try:
        from rich.console import Console
        return Console()
    except ImportError:
        return None


def _print_header(console, title: str, json_mode: bool = False):
    if json_mode:
        return
    if console:
        console.rule(f"[bold]{title}[/bold]")
    else:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")


def _print_section(console, title: str, json_mode: bool = False):
    if json_mode:
        return
    if console:
        console.print(f"\n[bold cyan]{title}[/bold cyan]")
    else:
        print(f"\n--- {title} ---")


def _print_event(console, step: int, event: TraceEvent, phase: Phase, json_mode: bool = False):
    if json_mode:
        return
    sym = _phase_symbol(phase)
    tool = event.tool or event.type
    if console:
        phase_color = phase.color if hasattr(phase, 'color') else "white"
        console.print(
            f"  [{phase_color}]{sym}[/{phase_color}] "
            f"Step {step:>2}: {tool:<15} "
            f"[dim]{phase.label:<12}[/dim] "
            f"[dim]{event.latency_ms:>5.0f}ms[/dim]"
        )
    else:
        print(f"  {sym} Step {step:>2}: {tool:<15} {phase.label:<12} {event.latency_ms:>5.0f}ms")


def _print_diagnosis(console, diag: CaftDiagnosis, json_mode: bool = False):
    if json_mode:
        return
    marker = _severity_marker(diag.severity)
    if console:
        color = "yellow" if diag.severity == CaftSeverity.WARNING else "red"
        console.print(
            f"\n  [{color}]{marker} CAFT {diag.caft_code}: "
            f"{diag.failure_name}[/{color}]"
            f"\n    Confidence: {diag.confidence:.0%}"
            f"\n    {diag.description}"
            f"\n    Step: {diag.at_step}"
        )
    else:
        print(f"\n  {marker} CAFT {diag.caft_code}: {diag.failure_name}")
        print(f"    Confidence: {diag.confidence:.0%}")
        print(f"    {diag.description}")
        print(f"    Step: {diag.at_step}")


# ── Core demo runner ────────────────────────────────────────────────

def run_demo(
    trace: list[TraceEvent] | None = None,
    json_output: bool = False,
    confirm: bool = False,
    delay: float = 0.0,
) -> dict:
    """Run the end-to-end demo pipeline.

    Args:
        trace: Custom trace events. None uses the built-in demo trace.
        json_output: Output JSON only (no narration).
        confirm: Enable mock LLM confirmation.
        delay: Seconds between events for animated output.

    Returns:
        Dict with full pipeline results (for --json mode and testing).
    """
    console = _try_rich() if not json_output else None
    events = trace or build_demo_trace()
    mock_llm = MockLLMConfirmer() if confirm else None

    # Result accumulator
    result = {
        "trace_events": len(events),
        "planted_failures": list(DEMO_FAILURES.keys()) if trace is None else [],
        "phases_seen": [],
        "detections": [],
        "hta_transitions": [],
        "trust_score": 1.0,
        "health": "healthy",
        "confirm_mode": confirm,
    }

    # ── Header ──
    _print_header(console, "AGENTDIAG END-TO-END DEMO", json_output)

    if not json_output:
        msg = f"Trace: {len(events)} events"
        if trace is None:
            msg += f" (built-in, {len(DEMO_FAILURES)} planted failures)"
        if confirm:
            msg += " + mock LLM confirmation"
        if console:
            console.print(f"[dim]{msg}[/dim]")
        else:
            print(msg)

    if trace is None and not json_output:
        _print_section(console, "Planted failures (ground truth)", json_output)
        for name, info in DEMO_FAILURES.items():
            if console:
                console.print(f"  [yellow]- {name}[/yellow] at step {info['onset_step']}: {info['description']}")
            else:
                print(f"  - {name} at step {info['onset_step']}: {info['description']}")

    # ── Build pipeline ──
    all_detectors = list(ALL_CAFT_DETECTORS_FULL)
    engine = MonitorEngine(
        goal="Fix authentication bug in login handler",
        detectors=all_detectors,
    )

    # ── Event-by-event processing ──
    _print_section(console, "Event stream (phase classification + detection)", json_output)

    phases_seen = set()
    all_diagnoses: list[CaftDiagnosis] = []

    for event in events:
        new_diagnoses = engine.push(event)
        hta_state = engine.state.hta_state
        current_phase = hta_state.current_phase if hta_state else Phase.IDLE
        phases_seen.add(current_phase.label)

        _print_event(console, event.step, event, current_phase, json_output)

        if delay > 0:
            time.sleep(delay)

        # Handle new detections
        for diag in new_diagnoses:
            if confirm and mock_llm:
                llm_result = mock_llm.confirm(diag)
                if not llm_result["confirmed"]:
                    if not json_output:
                        if console:
                            console.print(
                                f"\n  [dim]LLM rejected {diag.failure_name}: "
                                f"{llm_result['reasoning']}[/dim]"
                            )
                        else:
                            print(f"\n  LLM rejected {diag.failure_name}: {llm_result['reasoning']}")
                    result["detections"].append({
                        "failure_name": diag.failure_name,
                        "caft_code": diag.caft_code,
                        "at_step": diag.at_step,
                        "confidence": round(diag.confidence, 3),
                        "confirmed": False,
                        "reasoning": llm_result["reasoning"],
                    })
                    continue
                # Confirmed — adjust confidence
                diag.confidence = max(diag.confidence, llm_result["confidence"])

            all_diagnoses.append(diag)
            _print_diagnosis(console, diag, json_output)
            result["detections"].append({
                "failure_name": diag.failure_name,
                "caft_code": diag.caft_code,
                "at_step": diag.at_step,
                "confidence": round(diag.confidence, 3),
                "confirmed": True if confirm else None,
                "reasoning": "",
            })

    # ── HTA transitions ──
    state = engine.state
    if state.hta_state:
        for t in state.hta_state.transitions:
            result["hta_transitions"].append({
                "from": t.from_phase.label,
                "to": t.to_phase.label,
                "at_step": t.at_step,
                "regression": t.is_regression,
            })

    result["phases_seen"] = sorted(phases_seen)
    result["trust_score"] = round(state.trust_score, 3)
    result["health"] = state.health

    # ── Summary ──
    _print_section(console, "Pipeline Summary", json_output)

    if not json_output:
        lines = [
            f"Total events: {len(events)}",
            f"Phases seen: {', '.join(sorted(phases_seen))}",
            f"HTA transitions: {len(result['hta_transitions'])}",
            f"  Regressions: {sum(1 for t in result['hta_transitions'] if t['regression'])}",
            f"Detections: {len(all_diagnoses)}",
            f"Trust score: {state.trust_score:.2f}",
            f"Health: {state.health}",
        ]
        if confirm:
            confirmed_count = sum(
                1 for d in result["detections"] if d.get("confirmed") is True
            )
            rejected_count = sum(
                1 for d in result["detections"] if d.get("confirmed") is False
            )
            lines.append(f"LLM confirmed: {confirmed_count}, rejected: {rejected_count}")

        for line in lines:
            if console:
                console.print(f"  {line}")
            else:
                print(f"  {line}")

        # Ground truth comparison (built-in trace only)
        if trace is None:
            _print_section(console, "Ground Truth Comparison", json_output)
            detected_names = {d.failure_name for d in all_diagnoses}
            for name, info in DEMO_FAILURES.items():
                hit = name in detected_names
                if console:
                    marker = "[green]HIT[/green]" if hit else "[red]MISS[/red]"
                    console.print(f"  {name}: {marker}")
                else:
                    marker = "HIT" if hit else "MISS"
                    print(f"  {name}: {marker}")

            # Check for FPs (detections not in ground truth)
            fp_names = detected_names - set(DEMO_FAILURES.keys())
            if fp_names:
                if console:
                    console.print(f"\n  [yellow]False positives: {', '.join(fp_names)}[/yellow]")
                else:
                    print(f"\n  False positives: {', '.join(fp_names)}")

    # JSON output
    if json_output:
        print(json.dumps(result, indent=2))

    return result


# ── CLI entry point ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end demo of the agentdiag pipeline",
    )
    parser.add_argument(
        "--trace", default=None,
        help="Path to trace JSONL file (default: built-in synthetic)",
    )
    parser.add_argument("--json", action="store_true", help="JSON output only")
    parser.add_argument("--confirm", action="store_true", help="Enable mock LLM confirmation")
    parser.add_argument("--delay", type=float, default=0.0, help="Seconds between events")
    args = parser.parse_args()

    trace = None
    if args.trace:
        trace_path = Path(args.trace).expanduser()
        if not trace_path.exists():
            print(f"Error: Trace file not found: {trace_path}", file=sys.stderr)
            sys.exit(1)
        # Parse trace from JSONL
        from agentdiag.adapters.claude_code import ClaudeCodeExtractor
        extractor = ClaudeCodeExtractor()
        events = []
        with open(trace_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    parsed = extractor._parse_message(data, len(events) + 1)
                    events.extend(parsed)
                except (json.JSONDecodeError, Exception):
                    continue
        if not events:
            print(f"Error: No events parsed from {trace_path}", file=sys.stderr)
            sys.exit(1)
        trace = events

    run_demo(
        trace=trace,
        json_output=args.json,
        confirm=args.confirm,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
