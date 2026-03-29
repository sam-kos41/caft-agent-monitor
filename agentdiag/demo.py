"""Demo mode — generates synthetic events through UniversalMonitor into visualization.

Generates a realistic agent session at ~2 events/sec with planted anomaly
patterns that the compositor should catch.  Serves the visualization UI
so you can watch sparklines build, anomalies highlight, phase markers
render, and working memory animate in real time.

Phases:
  1. Planning (steps 1-30):    high entropy, diverse reads, memory loads at L0/L1
  2. Execution (steps 31-200): lower entropy, focused tool calls, some L2 escalations
  3. Stuck loop (140-160):     INJECTED: repeat same 3 tools → mechanical_repetition
  4. Goal drift (200-230):     INJECTED: unrelated tools, low MI → goal_discontinuity
  5. QA (steps 231-270):       evaluation events with scores
  6. Recovery (271-300):       return to normal execution

Usage::

    python -m agentdiag.demo                   # opens browser
    python -m agentdiag.demo --port 8888       # custom port
    python -m agentdiag.demo --speed 5         # 5 events/sec
    python -m agentdiag.demo --json            # print JSON to stdout, no server
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import threading
from typing import Optional

from agentdiag.observable import (
    ObservableEvent,
    EventType,
    MemoryTier,
    HarnessPhase,
    AgentRole,
    memory_load_event,
    memory_store_event,
    tier_escalation_event,
    phase_boundary_event,
    evaluation_event,
    tool_call_event,
    file_read_event,
    file_write_event,
)
from agentdiag.universal_monitor import UniversalMonitor


# ── Synthetic event generation ─────────────────────────────────────────────

def _generate_events() -> list[ObservableEvent]:
    """Generate a deterministic synthetic event stream with planted anomalies."""
    events: list[ObservableEvent] = []
    rng = random.Random(42)
    t = 1000.0  # base timestamp

    def _ts():
        nonlocal t
        t += rng.uniform(0.3, 1.5)
        return t

    # === Phase 1: Planning (steps 1-30) ===
    events.append(phase_boundary_event(
        step=0, timestamp=_ts(), phase=HarnessPhase.PLANNING,
        agent_role=AgentRole.PLANNER,
    ))

    planning_tools = ["Read", "Grep", "Read", "Glob", "Read", "Grep", "Read"]
    planning_files = [
        "/src/main.py", "/src/config.ts", "/src/api/routes.py",
        "/src/models/user.py", "/tests/test_auth.py", "/README.md",
        "/package.json", "/src/utils.py", "/src/middleware.py",
    ]

    for i in range(1, 31):
        tool = rng.choice(planning_tools)
        path = rng.choice(planning_files)
        events.append(file_read_event(
            step=i, timestamp=_ts(), path=path,
            output_tokens=rng.randint(200, 2000),
        ))
        # Some memory loads during planning
        if i % 7 == 0:
            ns = rng.choice([
                "agent/planner/skills", "resources/reference_projects",
                "user/memories/preferences",
            ])
            events.append(memory_load_event(
                step=i, timestamp=_ts(),
                uri=f"viking://{ns}/item_{i}",
                tier=rng.choice([MemoryTier.L0, MemoryTier.L1]),
                token_count=rng.randint(100, 2000),
            ))

    # === Phase 2: Execution (steps 31-200) ===
    events.append(phase_boundary_event(
        step=31, timestamp=_ts(), phase=HarnessPhase.EXECUTING,
        agent_role=AgentRole.GENERATOR,
    ))

    exec_tools = ["Edit", "Bash", "Read", "Edit", "Bash", "Edit"]
    for i in range(31, 200):
        # === INJECTED: Stuck loop at steps 140-160 ===
        if 140 <= i <= 160:
            cycle = ["Read", "Edit", "Read"]
            tool = cycle[i % 3]
            events.append(tool_call_event(
                step=i, timestamp=_ts(), tool_name=tool,
                duration_ms=rng.uniform(50, 200),
                input_tokens=100, output_tokens=100,
            ))
            continue

        tool = rng.choice(exec_tools)
        if tool == "Read":
            events.append(file_read_event(
                step=i, timestamp=_ts(),
                path=rng.choice(planning_files),
                output_tokens=rng.randint(100, 1500),
            ))
        elif tool == "Edit":
            events.append(file_write_event(
                step=i, timestamp=_ts(),
                path=rng.choice(planning_files[:5]),
                input_tokens=rng.randint(50, 500),
            ))
        elif tool == "Bash":
            events.append(ObservableEvent(
                step=i, timestamp=_ts(),
                event_type=EventType.SHELL_COMMAND,
                tool_name="Bash",
                duration_ms=rng.uniform(100, 3000),
                input_tokens=50, output_tokens=rng.randint(50, 500),
            ))

        # Occasional L2 tier escalations during execution
        if i % 25 == 0:
            events.append(tier_escalation_event(
                step=i, timestamp=_ts(),
                uri=f"viking://agent/generator/skills/pattern_{i//25}",
                from_tier=MemoryTier.L1, to_tier=MemoryTier.L2,
                token_count=rng.randint(2000, 8000),
            ))

    # === Phase 3: Goal drift (steps 200-230) ===
    # Sudden shift to completely unrelated tools — should trigger incoherent_exploration
    drift_tools = ["Deploy", "Docker", "Terraform", "kubectl", "AWS", "GCloud"]
    for i in range(200, 231):
        events.append(tool_call_event(
            step=i, timestamp=_ts(),
            tool_name=rng.choice(drift_tools),
            duration_ms=rng.uniform(100, 2000),
            input_tokens=rng.randint(50, 300),
            output_tokens=rng.randint(50, 300),
        ))

    # === Phase 4: QA / Verification (steps 231-270) ===
    events.append(phase_boundary_event(
        step=231, timestamp=_ts(), phase=HarnessPhase.VERIFYING,
        agent_role=AgentRole.EVALUATOR,
    ))

    for i in range(231, 271):
        # Evaluator reads and tests
        if i % 3 == 0:
            events.append(file_read_event(
                step=i, timestamp=_ts(),
                path=rng.choice(planning_files),
                output_tokens=rng.randint(200, 1000),
            ))
        elif i % 3 == 1:
            events.append(ObservableEvent(
                step=i, timestamp=_ts(),
                event_type=EventType.SHELL_COMMAND,
                tool_name="pytest",
                duration_ms=rng.uniform(500, 5000),
                output_tokens=rng.randint(100, 500),
            ))
        else:
            events.append(tool_call_event(
                step=i, timestamp=_ts(),
                tool_name="Read",
                duration_ms=rng.uniform(50, 200),
                output_tokens=rng.randint(100, 500),
            ))

        # Evaluation results at key points
        if i in (240, 255, 265):
            score = rng.uniform(0.3, 0.95)
            criterion = rng.choice(["correctness", "completeness", "style", "tests"])
            events.append(evaluation_event(
                step=i, timestamp=_ts(),
                score=score, criterion=criterion, sprint_number=1,
            ))

    # === Phase 5: Recovery (steps 271-300) ===
    events.append(phase_boundary_event(
        step=271, timestamp=_ts(), phase=HarnessPhase.ITERATING,
        agent_role=AgentRole.GENERATOR,
    ))

    for i in range(271, 301):
        tool = rng.choice(exec_tools)
        if tool == "Edit":
            events.append(file_write_event(
                step=i, timestamp=_ts(),
                path=rng.choice(planning_files[:5]),
                input_tokens=rng.randint(50, 300),
            ))
        elif tool == "Bash":
            events.append(ObservableEvent(
                step=i, timestamp=_ts(),
                event_type=EventType.SHELL_COMMAND,
                tool_name="Bash",
                duration_ms=rng.uniform(100, 1000),
                output_tokens=rng.randint(50, 300),
            ))
        else:
            events.append(file_read_event(
                step=i, timestamp=_ts(),
                path=rng.choice(planning_files),
                output_tokens=rng.randint(100, 800),
            ))

        # Post-sprint memory stores (skill crystallization)
        if i % 10 == 0:
            events.append(memory_store_event(
                step=i, timestamp=_ts(),
                uri=f"viking://agent/generator/skills/learned_{i}",
                token_count=rng.randint(500, 2000),
            ))

    return events


# ── JSON mode (stdout) ─────────────────────────────────────────────────────

def _run_json(speed: float) -> None:
    """Print processed events as JSONL to stdout."""
    monitor = UniversalMonitor(calibration_window=80, sensitivity=2.5)
    events = _generate_events()
    delay = 1.0 / speed

    for event in events:
        result = monitor.process(event)
        print(json.dumps(result, default=str))
        sys.stdout.flush()
        time.sleep(delay)

    # Final state
    print(json.dumps({"type": "final_state", **monitor.get_state()}, default=str))


# ── Visualization server mode ──────────────────────────────────────────────

def _run_server(port: int, speed: float, open_browser: bool) -> None:
    """Run the demo with the visualization server."""
    try:
        import uvicorn
        from agentdiag.visualize import app, start_server
    except ImportError:
        print("Visualization requires: pip install uvicorn fastapi")
        print("Falling back to JSON mode.")
        _run_json(speed)
        return

    # We'll use a simpler approach: generate events into a JSONL temp file
    # and pipe it through the existing visualization server.
    import tempfile
    import os

    monitor = UniversalMonitor(calibration_window=80, sensitivity=2.5)
    events = _generate_events()

    # Write events as JSONL that the existing visualize.py ingestion can read
    tmpfile = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, prefix="agentdiag_demo_",
    )

    # Convert ObservableEvents to dicts the ingestion thread can parse
    for event in events:
        tmpfile.write(json.dumps(event.to_dict(), default=str) + "\n")
    tmpfile.close()

    print(f"Demo: {len(events)} synthetic events generated")
    print(f"Demo: serving visualization at http://localhost:{port}")
    print(f"Demo: events will replay at {speed} events/sec")
    print()
    print("Watch for:")
    print("  - Entropy drop around step 140 (stuck loop)")
    print("  - MI drop + surprisal spike around step 200 (goal drift)")
    print("  - Evaluation markers (red/green) during QA phase (steps 231-270)")
    print("  - Phase boundary vertical lines on sparklines")
    print("  - Working memory items appearing and fading")
    print()

    if open_browser:
        def _open():
            time.sleep(1.5)
            import webbrowser
            webbrowser.open(f"http://localhost:{port}")
        threading.Thread(target=_open, daemon=True).start()

    # Use the existing visualization server with the temp file as input
    stream = open(tmpfile.name, "r")
    try:
        start_server(
            stream=stream,
            goal="Demo: synthetic agent session with planted anomalies",
            port=port,
            delay=1.0 / speed,
            cognitive=True,
        )
    finally:
        stream.close()
        os.unlink(tmpfile.name)


# ── CLI entry point ────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="agentdiag demo — synthetic event stream visualization"
    )
    parser.add_argument("--port", type=int, default=8080, help="Server port (default: 8080)")
    parser.add_argument("--speed", type=float, default=2.0, help="Events per second (default: 2.0)")
    parser.add_argument("--json", action="store_true", help="Output JSON to stdout, no server")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser")
    args = parser.parse_args()

    if args.json:
        _run_json(args.speed)
    else:
        _run_server(args.port, args.speed, not args.no_browser)


if __name__ == "__main__":
    main()
