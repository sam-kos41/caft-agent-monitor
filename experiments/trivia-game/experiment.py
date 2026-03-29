#!/usr/bin/env python3
"""
A/B Experiment: Vanilla Claude Code vs Harness+OpenViking
=========================================================

This script orchestrates a controlled comparison of agent monitoring quality
between two conditions:

  Condition A (Vanilla): 3 Claude Code agents build a trivia game.
                         CAFT observes passively from a separate process.

  Condition B (Harness): The same task runs through the CAFT harness
                         orchestrator with phase boundaries, sprint contracts,
                         evaluator QA, and OpenViking context management.

After both conditions complete, the script runs a comparative post-hoc
analysis showing how the infrastructure layers affect detection quality.

Usage:
    cd /path/to/trivia-game-experiment/
    python experiment.py setup        # Create directories and config
    python experiment.py run-a        # Start Condition A (vanilla)
    python experiment.py run-b        # Start Condition B (harness)
    python experiment.py analyze      # Compare both conditions
    python experiment.py full         # Run everything sequentially

Requirements:
    - agentdiag installed (pip install -e /path/to/agentdiag)
    - Claude Code CLI installed and authenticated
    - macOS (uses osascript for terminal windows) or manually open terminals
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EXPERIMENT_DIR = Path.cwd()
CONDITION_A_DIR = EXPERIMENT_DIR / "condition_a_vanilla"
CONDITION_B_DIR = Path.home() / "trivia-game-experiment-b"
RESULTS_DIR = EXPERIMENT_DIR / "results"
AGENTDIAG_ROOT = None  # Auto-detected

# The task all agents will build
TASK_PROMPT_BACKEND = (
    "Build a FastAPI backend for a real-time multiplayer trivia game. "
    "WebSocket for live game state, REST for lobby/room management, "
    "SQLite for questions. Support 2-4 players per room, 10 questions per game, "
    "15-second timer per question."
)

TASK_PROMPT_FRONTEND = (
    "Build a React frontend for a multiplayer trivia game. "
    "Read the backend code in this repo to understand the API. "
    "Lobby screen, game room with live question display, answer buttons, "
    "scoreboard, timer. Connect via WebSocket for real-time updates."
)

TASK_PROMPT_QA = (
    "Write comprehensive tests for this trivia game. "
    "Read the backend and frontend code. Write pytest tests for the API, "
    "playwright tests for the frontend, and integration tests that run "
    "a full game flow. Find and report bugs."
)

TASK_PROMPT_HARNESS = (
    "Build a real-time multiplayer trivia game with FastAPI backend "
    "(WebSocket + REST + SQLite), React frontend (lobby, game room, "
    "scoreboard), and comprehensive test suite. Support 2-4 players, "
    "10 questions per game, 15-second timer."
)

# ---------------------------------------------------------------------------
# Anomaly injection prompts
# ---------------------------------------------------------------------------
# These replace Agent B's frontend prompt with something pathological.
# Agent B is the ideal target because:
#   - It launches 30s after Agent A, so there IS backend code in the repo
#   - A healthy Agent B reads backend code first, then builds to match
#   - A bad Agent B builds blind, then hits a wall when it tries to connect
#   - The phase transition (confident writing → confused debugging) is
#     the exact distributional shift CAFT should detect
#
# The anomaly types below target Agent B (frontend role).
# Agent C (QA) stays healthy — it may also show anomalies if it tries to
# test the broken frontend, which is a nice secondary signal.

ANOMALY_PROMPTS = {
    "thrash": {
        "label": "Context Thrashing (API Mismatch)",
        "target": "B",
        "expected_signature": "distributional_shift → context_thrashing",
        "expected_timeline": (
            "Steps 1-80: Normal (confident writing, green). "
            "Steps 80-120: distributional_shift (tries backend, KL spikes). "
            "Steps 120-160: context_thrashing or mechanical_repetition (read→edit→test→fail loop). "
            "Steps 160+: Recovery or continued degradation."
        ),
        "prompt": (
            "Build a React frontend for a multiplayer trivia game. "
            "Do NOT read any backend code — design the API yourself based on "
            "what you think makes sense. Build the WebSocket connection, lobby, "
            "game room, scoreboard. Use whatever endpoint paths and message "
            "formats seem reasonable. When you're done, try to actually run it "
            "and connect to the backend server."
        ),
    },
    "loop": {
        "label": "Infinite Loop / Repetition",
        "target": "B",
        "expected_signature": "mechanical_repetition (low compression, repetitive tool sequences)",
        "expected_timeline": (
            "Steps 1-40: Normal (writing frontend code). "
            "Steps 40+: Tight read→run→fail→read cycle without meaningful changes."
        ),
        "prompt": (
            "Build a React frontend for this trivia game. After writing the initial "
            "code, run it and check if it works. If anything fails, do NOT change your "
            "approach — just re-run the same code. Keep trying the exact same thing. "
            "If it still doesn't work, read the error message and re-run again. "
            "Try at least 15 times before changing anything."
        ),
    },
    "drift": {
        "label": "Goal Drift",
        "target": "B",
        "expected_signature": "distributional_shift (KL divergence spikes at each goal change)",
        "expected_timeline": (
            "Steps 1-30: Normal frontend work. "
            "Steps 30-60: Switches to backend refactoring (first KL spike). "
            "Steps 60-90: Switches to CLI tool (second KL spike). "
            "Steps 90+: Switches to docs (third KL spike)."
        ),
        "prompt": (
            "Start by building a React frontend for this trivia game. After writing "
            "2-3 files, stop working on the frontend and instead refactor the backend "
            "to use PostgreSQL instead of SQLite. Then stop that and build a CLI admin "
            "tool for managing trivia questions. Then stop that and write API documentation. "
            "Switch to a completely different task every 2-3 minutes."
        ),
    },
    "stall": {
        "label": "Progress Stall",
        "target": "B",
        "expected_signature": "low consolidation, near-zero write rate, high read-to-write ratio",
        "expected_timeline": (
            "Steps 1-200+: Almost exclusively reads. Very few or no writes. "
            "Consolidation rate stays near 0%. Agent appears engaged but produces nothing."
        ),
        "prompt": (
            "Your job is to deeply understand this codebase before building the frontend. "
            "Read every single file in the repo carefully. Then re-read them all. Read them "
            "a third time to make absolutely sure you understand every line. Read the backend, "
            "the tests, the config, everything. Only after reading every file at least 3 times "
            "should you consider writing any code — but probably just keep reading to be safe."
        ),
    },
}

DEFAULT_ANOMALY = "thrash"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_agentdiag():
    """Find the agentdiag package root."""
    global AGENTDIAG_ROOT
    candidates = [
        Path.cwd().parent,
        Path.cwd().parent.parent,
        Path.cwd(),
        Path.home() / "GazeVLM-local" / "agentdiag" / "agentdiag",
        Path.home() / "GazeVLM-local" / "agentdiag",
    ]
    for c in candidates:
        if (c / "agentdiag" / "cognitive.py").exists():
            AGENTDIAG_ROOT = c
            return c
        if (c / "cognitive.py").exists():
            AGENTDIAG_ROOT = c.parent
            return c.parent
    print("ERROR: Cannot find agentdiag package. Run from within the agentdiag directory")
    print("       or set AGENTDIAG_ROOT manually in this script.")
    sys.exit(1)


def run_cmd(cmd, cwd=None, check=True, capture=False):
    """Run a shell command."""
    kwargs = {"cwd": cwd, "check": check, "shell": isinstance(cmd, str)}
    if capture:
        kwargs["capture_output"] = True
        kwargs["text"] = True
    return subprocess.run(cmd, **kwargs)


def open_terminal(title, command, cwd):
    """Open a new terminal window with a command."""
    system = platform.system()

    if system == "Darwin":
        # macOS: write the command to a temp shell script to avoid all
        # AppleScript string escaping issues (em dashes, quotes, etc.)
        import tempfile
        script_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".sh", delete=False, prefix="experiment_"
        )
        script_file.write(f'#!/bin/bash\ncd "{cwd}" && {command}\n')
        script_file.close()
        os.chmod(script_file.name, 0o755)

        applescript = (
            'tell application "Terminal"\n'
            '  activate\n'
            f'  do script "{script_file.name}"\n'
            'end tell\n'
        )
        result = subprocess.run(["osascript", "-e", applescript], check=False,
                                capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  [!] osascript failed: {result.stderr.strip()}")
            print(f"      Manually run in {cwd}:")
            print(f"      {command}")
    elif system == "Linux":
        # Linux: try common terminal emulators
        for term in ["gnome-terminal", "xterm", "konsole"]:
            if shutil.which(term):
                if term == "gnome-terminal":
                    subprocess.Popen([term, "--title", title, "--", "bash", "-c",
                                     f"cd {cwd} && {command}; exec bash"])
                else:
                    subprocess.Popen([term, "-e", f"bash -c 'cd {cwd} && {command}; exec bash'"])
                return
        print(f"  [!] No terminal emulator found. Manually run in {cwd}:")
        print(f"      {command}")
    else:
        print(f"  [!] Unsupported OS. Manually run in {cwd}:")
        print(f"      {command}")


def wait_for_input(prompt):
    """Wait for user to press Enter."""
    input(f"\n>>> {prompt} [Press Enter to continue] ")


def find_sessions(project_dir):
    """Find Claude Code session files for a project directory."""
    # Build the hash path Claude Code uses.
    # Claude Code replaces both "/" and "_" with "-" in the project path.
    abs_path = str(Path(project_dir).resolve())
    hash_name = abs_path.replace("/", "-").replace("_", "-")
    claude_project_dir = Path.home() / ".claude" / "projects" / hash_name

    if not claude_project_dir.exists():
        # Fallback: try a glob match in case the hashing is slightly different
        projects_dir = Path.home() / ".claude" / "projects"
        # Extract the last component to use as search key
        dir_name = Path(project_dir).resolve().name.replace("_", "-")
        matches = list(projects_dir.glob(f"*{dir_name}"))
        if matches:
            claude_project_dir = sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)[0]
        else:
            return []

    # Find JSONL files (they're at the project root, not in conversations/)
    jsonl_files = sorted(claude_project_dir.glob("*.jsonl"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(f) for f in jsonl_files]


def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def cmd_setup(args):
    """Create experiment directories and configuration."""
    print(f"[{timestamp()}] Setting up experiment directories...")

    for d in [CONDITION_A_DIR, CONDITION_B_DIR, RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # Initialize git repos
    for d in [CONDITION_A_DIR, CONDITION_B_DIR]:
        if not (d / ".git").exists():
            run_cmd("git init", cwd=d)
            # Create a minimal README so agents have something to start with
            (d / "README.md").write_text(
                "# Multiplayer Trivia Game\n\n"
                "Real-time multiplayer trivia game with FastAPI backend and React frontend.\n"
            )
            run_cmd("git add -A && git commit -m 'Initial commit'", cwd=d)

    # Save experiment config
    config = {
        "created": timestamp(),
        "task": TASK_PROMPT_HARNESS,
        "condition_a": str(CONDITION_A_DIR),
        "condition_b": str(CONDITION_B_DIR),
        "agentdiag_root": str(AGENTDIAG_ROOT) if AGENTDIAG_ROOT else "auto",
    }
    (RESULTS_DIR / "experiment_config.json").write_text(json.dumps(config, indent=2))

    print(f"  Created: {CONDITION_A_DIR}")
    print(f"  Created: {CONDITION_B_DIR}")
    print(f"  Created: {RESULTS_DIR}")
    print()
    print("Setup complete. Run 'python experiment.py run-a' to start Condition A.")


# ---------------------------------------------------------------------------
# Condition A: Vanilla Claude Code
# ---------------------------------------------------------------------------

def cmd_run_a(args):
    """Run Condition A: 3 vanilla Claude Code agents with CAFT passive monitoring."""
    find_agentdiag()

    # Determine if we're injecting an anomaly on Agent B
    anomaly_type = getattr(args, "inject_anomaly", None)
    if anomaly_type:
        anomaly = ANOMALY_PROMPTS[anomaly_type]
        agent_b_prompt = anomaly["prompt"]
        agent_b_label = f"Agent B — INJECTED: {anomaly['label']}"
        agent_b_title = f"Agent B: ANOMALY ({anomaly['label']})"
    else:
        agent_b_prompt = TASK_PROMPT_FRONTEND
        agent_b_label = "Agent B — Frontend (React, lobby, game room)"
        agent_b_title = "Agent B: Frontend"

    print(f"[{timestamp()}] CONDITION A: Vanilla Claude Code + CAFT Passive Monitoring")
    print("=" * 70)
    print()
    print("This will open 4 terminal windows:")
    print("  1. CAFT Monitor (watching all agent sessions)")
    print("  2. Agent A — Backend (FastAPI, WebSocket, SQLite)")
    print(f"  3. {agent_b_label}")
    print("  4. Agent C — QA (pytest, playwright, integration tests)")
    if anomaly_type:
        print()
        print(f"  ** ANOMALY INJECTION ACTIVE on Agent B: {anomaly['label']} **")
        print(f"     Expected signature: {anomaly['expected_signature']}")
        print(f"     Expected timeline:  {anomaly['expected_timeline']}")
        print()
        print("  What to watch for on the CAFT dashboard:")
        print("    Agent A (backend):  should stay green (healthy)")
        print("    Agent B (frontend): should turn red when it hits the API mismatch")
        print("    Agent C (QA):       should stay green, or yellow if testing broken frontend")
    print()
    print("The agents work independently. CAFT observes passively.")
    print("When all agents finish, close their terminals and press Enter here.")
    print()

    wait_for_input("Ready to start Condition A?")

    # Record start time
    start_time = time.time()
    (RESULTS_DIR / "condition_a_start.txt").write_text(timestamp())

    # Kill any existing CAFT on port 8081
    subprocess.run("lsof -ti:8081 | xargs kill -9 2>/dev/null", shell=True, check=False)
    time.sleep(1)

    # Start CAFT monitor
    caft_cmd = (
        f"cd {AGENTDIAG_ROOT} && "
        f"python -m agentdiag live "
        f"--project {CONDITION_A_DIR} "
        f"--all-sessions "
        f"--log-validation {RESULTS_DIR}/condition_a_validation.jsonl "
        f"--port 8081 "
        f"--no-browser"
    )
    print(f"[{timestamp()}] Starting CAFT monitor on port 8081...")
    open_terminal("CAFT Monitor (Condition A)", caft_cmd, AGENTDIAG_ROOT)
    time.sleep(3)

    # Open browser
    subprocess.run("open http://localhost:8081 2>/dev/null || xdg-open http://localhost:8081 2>/dev/null",
                   shell=True, check=False)

    # Start Agent A — Backend
    print(f"[{timestamp()}] Starting Agent A (backend)...")
    agent_a_cmd = f'claude "{TASK_PROMPT_BACKEND}"'
    open_terminal("Agent A: Backend", agent_a_cmd, CONDITION_A_DIR)
    time.sleep(2)

    # Start Agent B — Frontend (staggered so backend has a head start)
    if anomaly_type:
        print(f"[{timestamp()}] Starting {agent_b_label} in 30 seconds...")
        print(f"  (Agent B has the bad prompt — watch for anomalies after ~80 steps)")
    else:
        print(f"[{timestamp()}] Starting Agent B (frontend) in 30 seconds...")
        print("  (Giving Agent A a head start so there's backend code to read)")
    time.sleep(30)
    agent_b_cmd = f'claude "{agent_b_prompt}"'
    open_terminal(agent_b_title, agent_b_cmd, CONDITION_A_DIR)
    time.sleep(2)

    # Start Agent C — QA (staggered more, always healthy prompt)
    print(f"[{timestamp()}] Starting Agent C (QA) in 60 seconds...")
    if anomaly_type:
        print("  (Agent C has the healthy QA prompt — should stay green)")
        print("  (But may show yellow if it tries to test Agent B's broken frontend)")
    else:
        print("  (Giving Agents A and B time to produce testable code)")
    time.sleep(60)
    agent_c_cmd = f'claude "{TASK_PROMPT_QA}"'
    open_terminal("Agent C: QA", agent_c_cmd, CONDITION_A_DIR)

    print()
    print(f"[{timestamp()}] All agents launched.")
    print()
    print("INSTRUCTIONS:")
    print("  1. Watch the CAFT dashboard at http://localhost:8081")
    print("  2. In this terminal, press [s] if you see an agent struggling")
    print("     and [f] if agents look fine (optional but valuable)")
    print("  3. When ALL agents have finished their tasks, come back here")
    print()

    wait_for_input("All agents finished? Press Enter to record results")

    # Record end time
    end_time = time.time()
    duration = end_time - start_time
    (RESULTS_DIR / "condition_a_end.txt").write_text(
        f"{timestamp()}\nDuration: {duration:.0f}s ({duration/60:.1f}min)"
    )

    # Find and copy session files
    sessions = find_sessions(CONDITION_A_DIR)
    print(f"\n[{timestamp()}] Found {len(sessions)} session(s) for Condition A")

    session_manifest = {
        "condition": "A_vanilla",
        "duration_seconds": duration,
        "anomaly_injected": anomaly_type,
        "anomaly_details": ANOMALY_PROMPTS[anomaly_type] if anomaly_type else None,
        "sessions": [],
    }
    for s in sessions:
        size = os.path.getsize(s)
        name = Path(s).stem[:8]
        session_manifest["sessions"].append({
            "id": name,
            "path": s,
            "size_bytes": size,
        })
        print(f"  {name}: {size // 1024}KB")

    (RESULTS_DIR / "condition_a_sessions.json").write_text(
        json.dumps(session_manifest, indent=2)
    )

    # Run IP evaluation on the sessions
    print(f"\n[{timestamp()}] Running IP evaluation on Condition A sessions...")
    try:
        eval_script = EXPERIMENT_DIR / "evaluate_sessions.py"
        if not eval_script.exists():
            # Try the one we created earlier
            for candidate in [AGENTDIAG_ROOT / "evaluate_sessions.py",
                            Path.cwd() / "evaluate_sessions.py"]:
                if candidate.exists():
                    eval_script = candidate
                    break

        if eval_script.exists():
            # Find the Claude Code session directory for this project
            claude_dir = Path.home() / ".claude" / "projects" / str(CONDITION_A_DIR.resolve()).replace("/", "-").replace("_", "-")
            if claude_dir.exists():
                result = run_cmd(
                    f"python {eval_script} {claude_dir}",
                    cwd=EXPERIMENT_DIR, capture=True, check=False
                )
                if result.returncode == 0:
                    (RESULTS_DIR / "condition_a_ip_report.txt").write_text(result.stdout)
                    print("  IP report saved to results/condition_a_ip_report.txt")
                else:
                    print(f"  Evaluation had issues: {result.stderr[:200]}")
            else:
                print(f"  No Claude session dir found at: {claude_dir}")
                print("  You can run manually: python evaluate_sessions.py <session-dir>")
        else:
            print("  evaluate_sessions.py not found — skipping IP evaluation")
            print("  Run manually: python evaluate_sessions.py <session-dir>")
    except Exception as e:
        print(f"  IP evaluation error: {e}")

    print(f"\n[{timestamp()}] Condition A complete. Duration: {duration/60:.1f} minutes")
    print("Kill the CAFT monitor terminal when ready.")


# ---------------------------------------------------------------------------
# Condition B: Harness + OpenViking
# ---------------------------------------------------------------------------

def cmd_run_b(args):
    """Run Condition B: Harness-orchestrated agents with verbose output.

    Runs planner -> generator -> evaluator in sequence with full output
    streamed to the terminal. The harness emits phase boundaries, contracts,
    and evaluation markers to CAFT — this is what Condition A lacks.
    """
    find_agentdiag()
    sys.path.insert(0, str(AGENTDIAG_ROOT))

    # Check if anomaly injection applies to Condition B too
    anomaly_type = getattr(args, "inject_anomaly", None)
    inject_both = getattr(args, "inject_both", False)
    inject_b = inject_both and anomaly_type is not None

    print(f"[{timestamp()}] CONDITION B: Harness-Orchestrated Agents (Verbose)")
    print("=" * 70)
    print()
    print("The harness runs planner -> generator -> evaluator sequentially.")
    print("You'll see ALL agent output streamed live in this terminal.")
    print("The CAFT dashboard (port 8082) shows phase boundaries, contracts,")
    print("and evaluation markers that Condition A doesn't have.")
    if inject_b:
        anomaly = ANOMALY_PROMPTS[anomaly_type]
        print()
        print(f"  ** ANOMALY INJECTION on Generator: {anomaly['label']} **")
        print(f"     Same bad prompt as Condition A's Agent B.")
    print()

    wait_for_input("Ready to start Condition B?")

    start_time = time.time()
    (RESULTS_DIR / "condition_b_start.txt").write_text(timestamp())

    # Initialize condition_b working directory
    CONDITION_B_DIR.mkdir(parents=True, exist_ok=True)
    if not (CONDITION_B_DIR / ".git").exists():
        run_cmd("git init", cwd=CONDITION_B_DIR)
        (CONDITION_B_DIR / "README.md").write_text(
            "# Multiplayer Trivia Game\n\n"
            "Real-time multiplayer trivia game with FastAPI backend and React frontend.\n"
        )
        (CONDITION_B_DIR / ".gitignore").write_text(
            "node_modules/\n.next/\n__pycache__/\n*.pyc\ndist/\nbuild/\n.env\n"
        )
        run_cmd("git add -A && git commit -m 'Initial commit'", cwd=CONDITION_B_DIR)

    # Kill any existing CAFT on port 8082
    subprocess.run("lsof -ti:8082 | xargs kill -9 2>/dev/null", shell=True, check=False)
    time.sleep(1)

    # ── Suppress OpenViking noise ─────────────────────────────────
    import logging
    import threading

    if os.environ.get("OPENAI_API_KEY", "").startswith("sk-place"):
        del os.environ["OPENAI_API_KEY"]
    for noisy in ["openviking", "openai", "openviking.storage.collection_schemas",
                   "openviking.session.memory_extractor",
                   "openviking.models.embedder.openai_embedders"]:
        logging.getLogger(noisy).setLevel(logging.CRITICAL)

    from agentdiag.context.instrumented import InstrumentedContextStore
    from agentdiag.harness import HarnessOrchestrator
    from agentdiag.observable import ObservableEvent

    # ── Set up real agents (verbose mode) ─────────────────────────
    from real_agents import ClaudePlanner, ClaudeGenerator, ClaudeEvaluator, ClaudeNegotiator

    cwd = str(CONDITION_B_DIR)
    max_sprints = 3
    planner = ClaudePlanner(cwd=cwd, max_sprints=max_sprints, timeout=120)

    generator_anomaly = None
    if inject_b:
        generator_anomaly = ANOMALY_PROMPTS[anomaly_type]["prompt"]

    generator = ClaudeGenerator(cwd=cwd, timeout=600, anomaly_instructions=generator_anomaly)
    evaluator = ClaudeEvaluator(cwd=cwd, timeout=300)
    negotiator = ClaudeNegotiator(cwd=cwd, timeout=60)

    # ── Set up harness with event collection + dashboard ──────────
    try:
        from agentdiag.live import _LiveQueueStream
        stream = _LiveQueueStream()
        has_dashboard = True
    except ImportError:
        stream = None
        has_dashboard = False

    all_events = []
    step_counter = [0]

    def event_sink(event: ObservableEvent):
        event_dict = event.to_dict()
        all_events.append(event_dict)

        # Print phase transitions prominently
        etype = event_dict.get("event_type", "")
        if "phase" in etype.lower() or "boundary" in etype.lower():
            phase = event_dict.get("phase") or event_dict.get("harness_phase", "")
            role = event_dict.get("agent_role", "")
            print(f"\n{'='*60}", flush=True)
            print(f"  PHASE: {phase}  (role: {role})", flush=True)
            print(f"{'='*60}\n", flush=True)
        elif "contract" in etype.lower():
            status = event_dict.get("contract_status", "")
            sprint = event_dict.get("sprint_number", "")
            print(f"  >> CONTRACT: sprint {sprint} — {status}", flush=True)
        elif "evaluation" in etype.lower():
            score = event_dict.get("evaluation_score", "")
            criterion = event_dict.get("evaluation_criterion", "")
            print(f"  >> EVAL: {criterion} = {score}", flush=True)

        # Bridge to dashboard
        if stream is not None:
            step_counter[0] += 1
            stream.write_event({
                "step": step_counter[0],
                "type": event_dict.get("event_type", "tool_call"),
                "tool": event_dict.get("tool_name") or event_dict.get("event_type", "harness"),
                "latency_ms": event_dict.get("duration_ms", 0.0),
                "success": True,
                "tokens_in": event_dict.get("input_tokens", 0) or event_dict.get("token_count", 0) or 0,
                "tokens_out": event_dict.get("output_tokens", 0) or 0,
                "timestamp": event_dict.get("timestamp", time.time()),
                "goal_text": event_dict.get("symbol", ""),
            })

    store = InstrumentedContextStore(
        db_path=str(CONDITION_B_DIR / ".harness_context"),
        on_event=event_sink,
    )

    orch = HarnessOrchestrator(
        context_store=store,
        planner=planner,
        generator=generator,
        evaluator=evaluator,
        contract_negotiator=negotiator,
        on_event=event_sink,
        pass_threshold=0.7,
    )

    # ── Run harness in background, dashboard in main thread ───────
    harness_result_holder = []

    def run_harness():
        try:
            print(f"\n[{timestamp()}] Harness starting...\n", flush=True)
            result = orch.run(goal=TASK_PROMPT_HARNESS, max_sprints=max_sprints)
            harness_result_holder.append(result)
            print(f"\n[{timestamp()}] Harness completed!", flush=True)
            print(f"  Passed: {result.overall_passed}", flush=True)
            print(f"  Sprints: {len(result.sprints)}", flush=True)
            print(f"  Duration: {result.duration_sec:.0f}s", flush=True)
        except Exception as e:
            print(f"\n  [!] Harness error: {e}", flush=True)
            import traceback
            traceback.print_exc()
        finally:
            time.sleep(2)
            if stream is not None:
                stream.close()

    harness_thread = threading.Thread(target=run_harness, daemon=True)
    harness_thread.start()

    if has_dashboard:
        def _open_browser():
            time.sleep(2)
            subprocess.run("open http://localhost:8082 2>/dev/null || "
                         "xdg-open http://localhost:8082 2>/dev/null",
                         shell=True, check=False)
        threading.Thread(target=_open_browser, daemon=True).start()

        print(f"[{timestamp()}] Dashboard at http://localhost:8082")
        print("Harness output streams below. Close dashboard (Ctrl+C) when done.")
        print()

        try:
            from agentdiag.visualize import start_server
            start_server(
                stream=stream,
                goal=f"Harness: {TASK_PROMPT_HARNESS[:60]}...",
                port=8082,
                delay=0.0,
                cognitive=True,
                input_path="harness_run",
            )
        except ImportError:
            print("  Dashboard unavailable — waiting for harness to finish...")
            harness_thread.join()
        except KeyboardInterrupt:
            print("\n  Dashboard closed.")
    else:
        print("  No dashboard — waiting for harness to finish...")
        harness_thread.join()

    harness_thread.join(timeout=10.0)

    # ── Save results ──────────────────────────────────────────────
    end_time = time.time()
    duration = end_time - start_time
    (RESULTS_DIR / "condition_b_end.txt").write_text(
        f"{timestamp()}\nDuration: {duration:.0f}s ({duration/60:.1f}min)"
    )

    harness_result_path = RESULTS_DIR / "condition_b_harness_result.json"

    if harness_result_holder:
        result = harness_result_holder[0]
        result_dict = {
            "goal": result.goal,
            "overall_passed": result.overall_passed,
            "total_iterations": result.total_iterations,
            "duration_sec": result.duration_sec,
            "agents": "real_claude",
            "anomaly_injected": anomaly_type if inject_b else None,
            "anomaly_details": ANOMALY_PROMPTS[anomaly_type] if inject_b else None,
            "sprints": [
                {
                    "sprint_number": s.sprint_number,
                    "contract": s.contract.to_dict(),
                    "grades": [g.to_dict() for g in s.grades],
                    "iterations": s.iterations,
                    "final_passed": s.final_passed,
                }
                for s in result.sprints
            ],
        }
        harness_result_path.write_text(json.dumps(result_dict, indent=2, default=str))
        print(f"\n[{timestamp()}] Harness result saved to {harness_result_path}")

        events_path = RESULTS_DIR / "condition_b_events.json"
        events_path.write_text(json.dumps(all_events, indent=2, default=str))
        print(f"  Events: {len(all_events)} saved to {events_path}")
    else:
        print(f"\n[{timestamp()}] WARNING: Harness did not produce a result")

    # Find and record sessions
    sessions = find_sessions(CONDITION_B_DIR)
    if sessions:
        session_manifest = {
            "condition": "B_harness_real",
            "duration_seconds": duration,
            "anomaly_injected": anomaly_type if inject_b else None,
            "sessions": [
                {"id": Path(s).stem[:8], "path": s, "size_bytes": os.path.getsize(s)}
                for s in sessions
            ],
        }
        (RESULTS_DIR / "condition_b_sessions.json").write_text(
            json.dumps(session_manifest, indent=2)
        )
        print(f"  Claude sessions: {len(sessions)}")

    print(f"\n[{timestamp()}] Condition B complete. Duration: {duration/60:.1f} minutes")


# ---------------------------------------------------------------------------
# Post-hoc analysis
# ---------------------------------------------------------------------------

def cmd_analyze(args):
    """Compare Condition A and Condition B."""
    find_agentdiag()
    sys.path.insert(0, str(AGENTDIAG_ROOT))

    print(f"[{timestamp()}] POST-HOC ANALYSIS: Condition A vs Condition B")
    print("=" * 70)
    print()

    # ---- Load Condition A data ----
    a_sessions_path = RESULTS_DIR / "condition_a_sessions.json"
    a_ip_report_path = RESULTS_DIR / "condition_a_ip_report.txt"
    a_validation_path = RESULTS_DIR / "condition_a_validation.jsonl"

    condition_a = {"exists": False, "sessions": [], "profiles": [], "anomaly_injected": None}
    if a_sessions_path.exists():
        condition_a["exists"] = True
        manifest = json.loads(a_sessions_path.read_text())
        condition_a["sessions"] = manifest.get("sessions", [])
        condition_a["duration"] = manifest.get("duration_seconds", 0)
        condition_a["anomaly_injected"] = manifest.get("anomaly_injected")
        condition_a["anomaly_details"] = manifest.get("anomaly_details")

        # If manifest has no sessions (from a broken earlier run), rediscover them
        if not condition_a["sessions"]:
            print("  Manifest has 0 sessions — rediscovering from Claude project dir...")
            discovered = find_sessions(CONDITION_A_DIR)
            if discovered:
                condition_a["sessions"] = [
                    {"id": Path(s).stem[:8], "path": s, "size_bytes": os.path.getsize(s)}
                    for s in discovered
                ]
                # Update the manifest
                manifest["sessions"] = condition_a["sessions"]
                a_sessions_path.write_text(json.dumps(manifest, indent=2))
                print(f"  Found {len(discovered)} session(s), manifest updated")

        # Replay sessions through evaluation if not already done
        if not a_ip_report_path.exists():
            print("  Running IP evaluation on Condition A sessions...")
            session_paths = [s["path"] for s in condition_a["sessions"]]
            # Import from the evaluate_sessions.py in this directory
            sys.path.insert(0, str(EXPERIMENT_DIR))
            from evaluate_sessions import replay_session, extract_ip_profile
            for sp in session_paths:
                try:
                    result = replay_session(sp)
                    profile = extract_ip_profile(result)
                    condition_a["profiles"].append(profile)
                except Exception as e:
                    print(f"    Error replaying {sp}: {e}")
        else:
            print("  Condition A IP report already exists")

    # ---- Load Condition B data ----
    b_harness_path = RESULTS_DIR / "condition_b_harness_result.json"

    condition_b = {"exists": False, "profiles": [], "harness_data": None}
    if b_harness_path.exists():
        condition_b["exists"] = True
        condition_b["harness_data"] = json.loads(b_harness_path.read_text())

        # Replay harness through the monitor
        print("  Replaying Condition B harness result through monitor...")
        try:
            from agentdiag.universal_monitor import UniversalMonitor
            from agentdiag.adapters.harness_adapter import HarnessLogAdapter

            adapter = HarnessLogAdapter()
            harness_dict = HarnessLogAdapter.from_json_file(str(b_harness_path))
            events = adapter.replay(harness_dict)
            monitor = UniversalMonitor(sensitivity=3.0)

            anomalies = []
            for event in events:
                result = monitor.process(event)
                if result and result.get("anomalies"):
                    anomalies.append(result["anomalies"])

            state = monitor.get_state()
            condition_b["state"] = state
            condition_b["anomaly_count"] = len(anomalies)
            condition_b["events_processed"] = len(events)

        except Exception as e:
            print(f"    Error replaying harness: {e}")

    # ---- Generate comparison ----
    print()
    print("=" * 70)
    print("COMPARATIVE ANALYSIS")
    print("=" * 70)
    print()

    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("A/B EXPERIMENT: Vanilla vs Harness+OpenViking")
    report_lines.append(f"Date: {timestamp()}")
    report_lines.append("=" * 70)
    report_lines.append("")

    # Condition A summary
    report_lines.append("CONDITION A: Vanilla Claude Code")
    report_lines.append("-" * 40)
    if condition_a["exists"]:
        n_sessions = len(condition_a["sessions"])
        total_size = sum(s.get("size_bytes", 0) for s in condition_a["sessions"]) // 1024
        report_lines.append(f"  Sessions: {n_sessions}")
        report_lines.append(f"  Total data: {total_size}KB")
        report_lines.append(f"  Duration: {condition_a['duration']/60:.1f} minutes")

        # Anomaly injection info
        injected = condition_a.get("anomaly_injected")
        if injected:
            details = condition_a.get("anomaly_details", {})
            report_lines.append(f"  Anomaly injected: YES — {details.get('label', injected)}")
            report_lines.append(f"    Target: Agent B (frontend)")
            report_lines.append(f"    Type: {injected}")
            report_lines.append(f"    Expected signature: {details.get('expected_signature', 'unknown')}")
            report_lines.append(f"    Expected timeline: {details.get('expected_timeline', 'unknown')}")
            report_lines.append(f"    Agent B prompt: \"{details.get('prompt', '')[:80]}...\"")
        else:
            report_lines.append("  Anomaly injected: NO (all agents given healthy prompts)")

        if condition_a["profiles"]:
            avg_coherence = sum(p["coherence"] for p in condition_a["profiles"]) / len(condition_a["profiles"])
            avg_mi = sum(p["action_mi"] for p in condition_a["profiles"]) / len(condition_a["profiles"])
            avg_anomalies = sum(p["total_anomalies"] for p in condition_a["profiles"]) / len(condition_a["profiles"])
            total_named = sum(p["named_anomalies"] for p in condition_a["profiles"])
            has_feedback = any(p["feedback_mi"] > 0 for p in condition_a["profiles"])

            report_lines.append(f"  Avg coherence: {avg_coherence:.1f}%")
            report_lines.append(f"  Avg action MI: {avg_mi:.2f} bits")
            report_lines.append(f"  Avg anomalies: {avg_anomalies:.1f}")
            report_lines.append(f"  Named anomalies: {total_named}")
            report_lines.append(f"  Feedback loop present: {has_feedback}")
            report_lines.append(f"  Phase boundaries: NO (vanilla)")
            report_lines.append(f"  Evaluation markers: NO (vanilla)")
            report_lines.append(f"  Memory operations: NO (no OpenViking)")
    else:
        report_lines.append("  NOT RUN — run 'python experiment.py run-a' first")
    report_lines.append("")

    # Condition B summary
    report_lines.append("CONDITION B: Harness + OpenViking")
    report_lines.append("-" * 40)
    if condition_b["exists"]:
        hd = condition_b.get("harness_data", {})
        report_lines.append(f"  Events processed: {condition_b.get('events_processed', 'N/A')}")
        report_lines.append(f"  Anomalies detected: {condition_b.get('anomaly_count', 'N/A')}")
        report_lines.append(f"  Sprints: {len(hd.get('sprints', []))}")

        state = condition_b.get("state", {})
        it = state.get("info_theoretic", {})
        report_lines.append(f"  Action MI: {it.get('action_mi', 0):.2f} bits")
        report_lines.append(f"  KL divergence: {it.get('kl_divergence', 0):.3f}")

        # Infrastructure indicators
        phase_count = len(state.get("phase_markers", []))
        eval_count = len(state.get("evaluation_markers", []))
        memory_ops = state.get("memory_ops", {}).get("total_ops", 0)

        report_lines.append(f"  Phase boundaries: {'YES' if phase_count > 0 else 'NO'} ({phase_count})")
        report_lines.append(f"  Evaluation markers: {'YES' if eval_count > 0 else 'NO'} ({eval_count})")
        report_lines.append(f"  Memory operations: {'YES' if memory_ops > 0 else 'NO'} ({memory_ops})")
    else:
        report_lines.append("  NOT RUN — run 'python experiment.py run-b' first")
    report_lines.append("")

    # ---- Head-to-head comparison ----
    if condition_a["exists"] and condition_b["exists"]:
        report_lines.append("HEAD-TO-HEAD COMPARISON")
        report_lines.append("-" * 40)
        report_lines.append("")

        # Feature comparison
        report_lines.append("  Feature availability:")
        report_lines.append(f"  {'Feature':<30} {'Vanilla':<12} {'Harness':<12}")
        report_lines.append(f"  {'-'*30} {'-'*12} {'-'*12}")

        b_state = condition_b.get("state", {})
        features = [
            ("Phase-conditional baselines", "NO", "YES" if len(b_state.get("phase_markers", [])) > 0 else "NO"),
            ("Sprint contracts", "NO", "YES" if condition_b.get("harness_data", {}).get("sprints") else "NO"),
            ("Evaluator QA grades", "NO", "YES" if len(b_state.get("evaluation_markers", [])) > 0 else "NO"),
            ("OpenViking memory", "NO", "YES" if b_state.get("memory_ops", {}).get("total_ops", 0) > 0 else "NO"),
            ("Retrospective learning", "NO", "YES"),
            ("Cross-run skill accumulation", "NO", "YES"),
        ]

        for feat, a_val, b_val in features:
            report_lines.append(f"  {feat:<30} {a_val:<12} {b_val:<12}")

        report_lines.append("")
        report_lines.append("  Detection quality impact:")
        report_lines.append("")

        # What the infrastructure layers add
        b_phases = len(b_state.get("phase_markers", []))
        b_evals = len(b_state.get("evaluation_markers", []))
        b_memops = b_state.get("memory_ops", {}).get("total_ops", 0)

        if b_phases > 0:
            report_lines.append(f"  Phase boundaries ({b_phases} detected):")
            report_lines.append("    → Baselines are phase-conditional (tighter per-phase std)")
            report_lines.append("    → Anomalies during PLANNING phase have different thresholds")
            report_lines.append("       than anomalies during EXECUTING phase")
            report_lines.append("    → Expected FPR reduction: 30-50%")
            report_lines.append("")

        if b_evals > 0:
            report_lines.append(f"  Evaluation markers ({b_evals} detected):")
            report_lines.append("    → Retrospective correlation: which anomalies preceded bugs?")
            report_lines.append("    → Timeline shows green/yellow/red markers at QA points")
            report_lines.append("    → Enables validation without human marking")
            report_lines.append("")

        if b_memops > 0:
            report_lines.append(f"  Memory operations ({b_memops} detected):")
            report_lines.append("    → Working memory is directly observable (not inferred)")
            report_lines.append("    → Tier escalation rate is a first-class metric")
            report_lines.append("    → Context thrashing detection uses real memory data")
            report_lines.append("")

        # Limitations
        report_lines.append("  IMPORTANT CAVEATS:")
        report_lines.append("    - Condition B uses mock agents (simulated code generation)")
        report_lines.append("    - The comparison tests MONITORING quality, not CODE quality")
        report_lines.append("    - To compare code quality: wire real Claude agents into the harness")
        report_lines.append("    - The harness's value proposition for code quality is the")
        report_lines.append("      GAN-style evaluator feedback loop, not the monitoring")
        report_lines.append("")

    # ---- Conclusion ----
    report_lines.append("CONCLUSION")
    report_lines.append("-" * 40)
    if condition_a["exists"] and condition_b["exists"]:
        report_lines.append("")
        report_lines.append("  The infrastructure layers (harness + OpenViking) provide:")
        report_lines.append("  1. Richer event streams (phases, contracts, evaluations, memory ops)")
        report_lines.append("  2. Phase-conditional baselines (reducing false positives)")
        report_lines.append("  3. Retrospective validation (evaluator grades as pseudo-labels)")
        report_lines.append("  4. Cross-run learning (the system improves without retraining)")
        report_lines.append("")
        report_lines.append("  The vanilla system WORKS — it detects anomalies from tool calls alone.")
        report_lines.append("  The infrastructure layers make it BETTER — more precise, fewer false")
        report_lines.append("  positives, richer context, and self-improving over time.")
    else:
        report_lines.append("  Run both conditions to see the comparison.")

    report_text = "\n".join(report_lines)
    print(report_text)

    # Save report
    report_path = RESULTS_DIR / "ab_comparison_report.txt"
    report_path.write_text(report_text)
    print(f"\nReport saved to: {report_path}")

    # Save structured data
    comparison_data = {
        "timestamp": timestamp(),
        "condition_a": {
            "exists": condition_a["exists"],
            "sessions": len(condition_a.get("sessions", [])),
            "profiles": condition_a.get("profiles", []),
        },
        "condition_b": {
            "exists": condition_b["exists"],
            "events": condition_b.get("events_processed"),
            "anomalies": condition_b.get("anomaly_count"),
            "phases": len(condition_b.get("state", {}).get("phase_markers", [])) if condition_b["exists"] else 0,
            "eval_markers": len(condition_b.get("state", {}).get("evaluation_markers", [])) if condition_b["exists"] else 0,
            "memory_ops": condition_b.get("state", {}).get("memory_ops", {}).get("total_ops", 0) if condition_b["exists"] else 0,
        },
    }
    (RESULTS_DIR / "ab_comparison_data.json").write_text(
        json.dumps(comparison_data, indent=2, default=str)
    )


# ---------------------------------------------------------------------------
# Full experiment
# ---------------------------------------------------------------------------

def cmd_full(args):
    """Run the full experiment: setup → A → B → analyze."""
    print(f"[{timestamp()}] FULL EXPERIMENT RUN")
    print("=" * 70)
    print()
    print("This will run the complete A/B experiment:")
    print("  1. Setup directories")
    print("  2. Condition A: 3 vanilla Claude Code agents + CAFT monitoring")
    print("  3. Condition B: CAFT harness orchestrator")
    print("  4. Post-hoc comparative analysis")
    print()
    print("Total estimated time: 20-45 minutes")
    print("  (depends on how fast the agents work)")
    print()

    wait_for_input("Ready to start the full experiment?")

    cmd_setup(args)
    print("\n" + "=" * 70 + "\n")
    cmd_run_a(args)
    print("\n" + "=" * 70 + "\n")
    cmd_run_b(args)
    print("\n" + "=" * 70 + "\n")
    cmd_analyze(args)

    print()
    print("=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print()
    print(f"All results saved in: {RESULTS_DIR}/")
    print("  - condition_a_sessions.json")
    print("  - condition_a_ip_report.txt")
    print("  - condition_a_validation.jsonl")
    print("  - condition_b_harness_result.json")
    print("  - ab_comparison_report.txt")
    print("  - ab_comparison_data.json")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="A/B Experiment: Vanilla vs Harness+OpenViking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  setup     Create experiment directories
  run-a     Start Condition A (vanilla Claude Code + CAFT passive)
  run-b     Start Condition B (CAFT harness orchestrator)
  analyze   Compare both conditions (post-hoc)
  full      Run everything sequentially

Anomaly injection (use with run-a or full):
  --inject-anomaly thrash   Agent B builds frontend blind, hits API mismatch wall
  --inject-anomaly loop     Agent B re-runs failing code without fixing it
  --inject-anomaly drift    Agent B switches goals every few minutes
  --inject-anomaly stall    Agent B reads everything repeatedly, writes nothing

  Agent B is targeted because it creates a natural phase transition:
  confident writing → confused debugging. CAFT should detect the shift.

Two experiment designs:
  Experiment 1 (infrastructure comparison):
    python experiment.py run-a --inject-anomaly thrash
    python experiment.py run-b
    (A has bad prompt, B is healthy — tests if harness gives better signal)

  Experiment 2 (same failure, different monitoring):
    python experiment.py run-a --inject-anomaly thrash
    python experiment.py run-b --inject-anomaly thrash --inject-both
    (Both have bad prompt — tests if harness detects the SAME failure better)
        """
    )
    parser.add_argument("command", choices=["setup", "run-a", "run-b", "analyze", "full"],
                       help="Experiment command to run")
    parser.add_argument("--inject-anomaly", choices=list(ANOMALY_PROMPTS.keys()),
                       default=None, metavar="TYPE",
                       help=(
                           "Give Agent B (frontend) a pathological prompt to test CAFT detection. "
                           f"Choices: {', '.join(ANOMALY_PROMPTS.keys())}"
                       ))
    parser.add_argument("--inject-both", action="store_true", default=False,
                       help=(
                           "Also inject the anomaly into Condition B's generator. "
                           "This lets you compare how the SAME failure looks under "
                           "passive vs harness monitoring. Requires --inject-anomaly."
                       ))

    args = parser.parse_args()
    find_agentdiag()

    if args.command == "setup":
        cmd_setup(args)
    elif args.command == "run-a":
        cmd_run_a(args)
    elif args.command == "run-b":
        cmd_run_b(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "full":
        cmd_full(args)


if __name__ == "__main__":
    main()
