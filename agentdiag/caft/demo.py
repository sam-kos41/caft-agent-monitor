"""Live demo for the CAFT monitor dashboard.

Generates synthetic events with realistic timing delays so you can
watch the dashboard update in real-time.

Usage:
    agentdiag monitor-demo                    # all scenarios
    agentdiag monitor-demo step_repetition    # single scenario
    agentdiag monitor-demo --delay 0.3        # slower playback
    agentdiag monitor-demo --plain            # text output
"""

from __future__ import annotations

import json
import signal
import sys
import time
from typing import IO

from agentdiag.models import TraceEvent

# Handle SIGPIPE gracefully (standard Unix behavior for piped commands)
try:
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
except AttributeError:
    pass  # Windows doesn't have SIGPIPE
from agentdiag.caft.synthetic import CAFT_GENERATORS, ALL_CAFT_SCENARIO_NAMES


SHOWCASE_SEQUENCE: list[tuple[str, float]] = [
    ("e2e", 3.0),
    ("multi_agent", 3.0),
    ("step_repetition", 2.0),
    ("context_loss", 2.0),
    ("premature_termination", 2.0),
    ("clean", 2.0),
]


def _event_to_jsonl(event: TraceEvent) -> str:
    """Serialize a TraceEvent to a JSONL line."""
    d = {
        "step": event.step,
        "type": event.type,
    }
    if event.tool:
        d["tool"] = event.tool
    if event.latency_ms:
        d["latency_ms"] = event.latency_ms
    if not event.success:
        d["success"] = event.success
    if event.output_hash:
        d["output_hash"] = event.output_hash
    if event.goal_text:
        d["goal_text"] = event.goal_text
    if event.error_message:
        d["error_message"] = event.error_message
    if event.tokens_in:
        d["tokens_in"] = event.tokens_in
    if event.tokens_out:
        d["tokens_out"] = event.tokens_out
    if event.agent_id:
        d["agent_id"] = event.agent_id
    return json.dumps(d)


def stream_scenario(
    scenario_name: str,
    delay: float = 0.15,
    output: IO[str] | None = None,
) -> None:
    """Stream a single scenario's events with delays.

    Args:
        scenario_name: Key from CAFT_GENERATORS.
        delay: Seconds between events.
        output: Where to write JSONL (default: stdout).
    """
    gen_fn = CAFT_GENERATORS[scenario_name]
    events, expected = gen_fn()
    dest = output or sys.stdout

    try:
        for event in events:
            line = _event_to_jsonl(event)
            dest.write(line + "\n")
            dest.flush()
            time.sleep(delay)
    except BrokenPipeError:
        sys.exit(0)


def stream_all_scenarios(
    delay: float = 0.15,
    pause_between: float = 1.0,
    output: IO[str] | None = None,
) -> None:
    """Stream all scenarios sequentially with pauses between.

    Args:
        delay: Seconds between events within a scenario.
        pause_between: Seconds between scenarios.
        output: Where to write JSONL (default: stdout).
    """
    dest = output or sys.stdout

    # Re-number steps globally so the monitor sees a continuous stream
    global_step = 0

    try:
        for scenario_name in ALL_CAFT_SCENARIO_NAMES:
            gen_fn = CAFT_GENERATORS[scenario_name]
            events, expected = gen_fn()

            # Write a comment-like marker (monitor will skip invalid JSON)
            dest.write(f"# === Scenario: {scenario_name} "
                        f"(expected: {sorted(expected) if expected else 'clean'}) ===\n")
            dest.flush()
            time.sleep(pause_between / 2)

            for event in events:
                global_step += 1
                event.step = global_step
                line = _event_to_jsonl(event)
                dest.write(line + "\n")
                dest.flush()
                time.sleep(delay)

            time.sleep(pause_between / 2)
    except BrokenPipeError:
        sys.exit(0)


def generate_demo_jsonl(
    scenario: str,
    delay: float = 0.3,
    output: IO[str] | None = None,
) -> None:
    """Write JSONL events to a file-like object for web visualization.

    Supports all synthetic scenarios, 'e2e', and 'showcase' (cycles through
    a curated sequence of scenarios with boundary markers).

    Args:
        scenario: Scenario name, 'e2e', 'all', or 'showcase'.
        delay: Seconds between events.
        output: Writable file-like object (e.g. QueueStream). Closed on completion.
    """
    dest = output or sys.stdout
    global_step = 0

    def _emit_events(events: list[TraceEvent], event_delay: float) -> int:
        nonlocal global_step
        for event in events:
            global_step += 1
            event.step = global_step
            line = _event_to_jsonl(event)
            dest.write(line + "\n")
            if hasattr(dest, "flush"):
                dest.flush()
            time.sleep(event_delay)
        return global_step

    def _emit_boundary(name: str, description: str) -> None:
        marker = json.dumps({
            "type": "scenario_boundary",
            "scenario": name,
            "description": description,
        })
        dest.write(marker + "\n")
        if hasattr(dest, "flush"):
            dest.flush()

    try:
        if scenario == "showcase":
            for scenario_name, pause in SHOWCASE_SEQUENCE:
                desc = _scenario_description(scenario_name)
                _emit_boundary(scenario_name, desc)
                time.sleep(1.0)

                events = _get_scenario_events(scenario_name)
                _emit_events(events, delay)
                time.sleep(pause)

        elif scenario == "e2e":
            from scripts.demo_e2e import build_demo_trace
            events = build_demo_trace()
            _emit_events(events, delay)

        elif scenario == "all":
            for scenario_name in ALL_CAFT_SCENARIO_NAMES:
                events = _get_scenario_events(scenario_name)
                _emit_events(events, delay)
                time.sleep(1.0)

        else:
            events = _get_scenario_events(scenario)
            _emit_events(events, delay)

    finally:
        if hasattr(dest, "close") and dest is not sys.stdout:
            dest.close()


def _get_scenario_events(name: str) -> list[TraceEvent]:
    """Get events for a scenario name (synthetic or e2e)."""
    if name == "e2e":
        from scripts.demo_e2e import build_demo_trace
        return build_demo_trace()
    gen_fn = CAFT_GENERATORS[name]
    events, _ = gen_fn()
    return events


def _scenario_description(name: str) -> str:
    """Human-readable description for showcase overlay."""
    descs = {
        "e2e": "End-to-End: 3 planted failures (repetition, termination, context loss)",
        "multi_agent": "Multi-Agent: Main agent spawns sub-agent via Task tool",
        "step_repetition": "Step Repetition: 10 identical file reads",
        "context_loss": "Context Loss: Re-reads forgotten file after extensive work",
        "premature_termination": "Premature Termination: Delivers without verification",
        "clean": "Clean Trace: Healthy agent lifecycle for contrast",
    }
    return descs.get(name, name.replace("_", " ").title())


def run_demo_plain(
    scenario: str | None = None,
    delay: float = 0.15,
) -> None:
    """Run the demo with plain text output (no rich dashboard).

    Prints each event and any CAFT detections inline.
    """
    from agentdiag.monitor import MonitorEngine
    from agentdiag.caft.synthetic import CAFT_GENERATORS

    if scenario and scenario != "all":
        scenarios = {scenario: CAFT_GENERATORS[scenario]}
    else:
        scenarios = CAFT_GENERATORS

    global_step = 0
    engine = MonitorEngine(goal="CAFT Demo")
    last_phase = None

    for scenario_name, gen_fn in scenarios.items():
        events, expected = gen_fn()
        exp_str = sorted(expected) if expected else "clean"
        print(f"\n{'='*60}")
        print(f"  Scenario: {scenario_name.upper()}")
        print(f"  Expected CAFT failures: {exp_str}")
        print(f"{'='*60}")

        for event in events:
            global_step += 1
            event.step = global_step
            new_dx = engine.push(event)
            state = engine.state

            # Show phase changes
            if state.hta_state and state.hta_state.current_phase != last_phase:
                last_phase = state.hta_state.current_phase
                print(f"  [PHASE] {last_phase.label.upper()}")

            # Show event
            tool_str = event.tool or event.type
            print(f"    step={event.step:>3}  {tool_str:<20}", end="")

            # Show detections
            for d in new_dx:
                print(f"  !! CAFT {d.caft_code} {d.failure_name} "
                      f"({d.severity.value})", end="")
            print()

            time.sleep(delay)

        # Check results for this scenario
        detected = {d.failure_name for d in engine._diagnoses}
        tp = expected & detected
        fn = expected - detected
        fp = detected - expected

        if fn:
            print(f"  MISSED: {sorted(fn)}")
        if fp:
            print(f"  EXTRA:  {sorted(fp)}")
        if not fn:
            if expected:
                print(f"  RESULT: All expected failures detected")
            else:
                fps = sorted(fp) if fp else "none"
                print(f"  RESULT: Clean scenario (false positives: {fps})")

        # Reset engine for next scenario
        engine.reset()
        last_phase = None

    print(f"\n{'='*60}")
    print(f"  Demo complete: {len(scenarios)} scenarios")
    print(f"{'='*60}")
