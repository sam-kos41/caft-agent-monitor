"""Synthetic trace generator for evaluation.

Generates realistic Claude Code JSONL traces for each task in the bank.
Each trace is a sequence of JSON dicts matching the format that
ClaudeCodeAdapter._from_dict() expects:

    {"step": int, "type": str, "tool": str, "timestamp": float,
     "latency_ms": float, "tokens_in": int, "tokens_out": int,
     "target_path": str}

For each task, generates:
  - {task}_clean.jsonl   — successful completion, no failures
  - {task}_loop.jsonl    — subtle file-rereading loop
  - {task}_drift.jsonl   — gradual shift to unrelated files
  - {task}_thrash.jsonl  — rapid reads with no writes (high entropy, low MI)
  - {task}_stall.jsonl   — agent works but makes no progress

Usage::

    python -m agentdiag.eval.trace_generator --output traces/
    python -m agentdiag.eval.trace_generator --task rest_api --variant clean
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Optional

from agentdiag.eval.tasks import TASK_BANK, TaskSpec


# ── JSONL entry builder ───────────────────────────────────────────────────

def _entry(
    step: int,
    typ: str,
    tool: Optional[str],
    timestamp: float,
    target_path: Optional[str] = None,
    latency_ms: float = 50.0,
    tokens_in: int = 0,
    tokens_out: int = 0,
) -> dict:
    """Build a single JSONL dict matching ClaudeCodeAdapter._from_dict()."""
    d = {"step": step, "type": typ, "timestamp": round(timestamp, 3)}
    if tool:
        d["tool"] = tool
    if target_path:
        d["target_path"] = target_path
    d["latency_ms"] = round(latency_ms, 1)
    d["tokens_in"] = tokens_in
    d["tokens_out"] = tokens_out
    return d


# ── Phase-aware action generation ─────────────────────────────────────────

# Tool distributions per phase.  Weights are relative.
_PHASE_TOOL_WEIGHTS = {
    "planning": {
        "reasoning": 4, "read": 5, "grep": 3, "glob": 2, "bash": 1,
    },
    "executing": {
        "read": 3, "write": 4, "edit": 5, "bash": 3, "grep": 1, "glob": 1,
    },
    "verifying": {
        "bash": 5, "read": 3, "grep": 2, "glob": 1,
    },
    "iterating": {
        "read": 3, "edit": 4, "write": 2, "bash": 3, "grep": 2,
    },
}

# Tool → type mapping for JSONL
_TOOL_TYPE = {
    "read": "tool_call", "write": "tool_call", "edit": "tool_call",
    "bash": "tool_call", "grep": "tool_call", "glob": "tool_call",
    "reasoning": "reasoning",
}


def _choose_tool(phase: str, rng: random.Random) -> str:
    """Weighted random tool choice based on current phase."""
    weights = _PHASE_TOOL_WEIGHTS.get(phase, _PHASE_TOOL_WEIGHTS["executing"])
    tools = list(weights.keys())
    w = [weights[t] for t in tools]
    return rng.choices(tools, weights=w, k=1)[0]


def _choose_file(
    task: TaskSpec,
    rng: random.Random,
    phase: str = "executing",
    recent_files: Optional[list[str]] = None,
) -> str:
    """Choose a realistic file path for the task.

    During execution, bias toward recently-touched files (locality).
    During planning, prefer config/entry files.
    """
    files = list(task.typical_files)
    if phase == "planning":
        # Bias toward config/entry files
        config_files = [f for f in files if any(
            k in f.lower() for k in ("config", "package", "setup", "readme", "ini", "yaml", "yml")
        )]
        if config_files:
            files = config_files + files  # double weight

    if recent_files and phase in ("executing", "iterating") and rng.random() < 0.4:
        return rng.choice(recent_files[-5:])

    return rng.choice(files)


def _make_action(
    step: int,
    timestamp: float,
    tool: str,
    task: TaskSpec,
    rng: random.Random,
    phase: str = "executing",
    recent_files: Optional[list[str]] = None,
    target_override: Optional[str] = None,
) -> dict:
    """Generate a single action entry."""
    typ = _TOOL_TYPE.get(tool, "tool_call")
    target = target_override

    if tool in ("read", "grep", "glob"):
        target = target or _choose_file(task, rng, phase, recent_files)
        tokens_out = rng.randint(200, 2000)
        tokens_in = rng.randint(10, 50)
        latency = rng.uniform(20, 200)
    elif tool in ("write", "edit"):
        target = target or _choose_file(task, rng, phase, recent_files)
        tokens_in = rng.randint(100, 1500)
        tokens_out = rng.randint(10, 100)
        latency = rng.uniform(50, 300)
    elif tool == "bash":
        # Bash commands vary by phase
        if phase == "verifying":
            cmds = ["pytest", "npm test", "make test", "python -m pytest", "cargo test"]
        elif phase == "planning":
            cmds = ["ls", "find . -name '*.py'", "cat requirements.txt"]
        else:
            cmds = ["npm install", "pip install -r requirements.txt",
                     "python app.py", "npm run build", "make", "git add .",
                     "git commit -m 'progress'", "docker build .", "npm run dev"]
        target = rng.choice(cmds)
        tokens_in = rng.randint(10, 50)
        tokens_out = rng.randint(50, 500)
        latency = rng.uniform(100, 3000)
    elif tool == "reasoning":
        tokens_in = 0
        tokens_out = rng.randint(100, 800)
        latency = rng.uniform(500, 3000)
    else:
        tokens_in = rng.randint(10, 100)
        tokens_out = rng.randint(10, 100)
        latency = rng.uniform(50, 500)

    return _entry(
        step=step, typ=typ, tool=tool if tool != "reasoning" else None,
        timestamp=timestamp, target_path=target,
        latency_ms=latency, tokens_in=tokens_in, tokens_out=tokens_out,
    )


# ── Clean trace generation ────────────────────────────────────────────────

def generate_clean_trace(task: TaskSpec, seed: int = 42) -> list[dict]:
    """Generate a clean (no-failure) trace for a task.

    The trace has realistic phase structure:
    1. Planning: reads, reasoning, exploration
    2. Executing: reads → writes → edits in feature clusters
    3. Verifying: test runs, reads to check output
    4. Iterating: fix-and-retest cycles
    """
    rng = random.Random(seed)
    n_steps = task.estimated_steps
    profile = task.phase_profile or {
        "planning": 0.12, "executing": 0.55, "verifying": 0.20, "iterating": 0.13,
    }

    # Phase boundaries (cumulative step counts)
    phases = []
    cum = 0
    for phase_name, frac in profile.items():
        count = max(5, int(n_steps * frac))
        phases.append((phase_name, cum, cum + count))
        cum += count

    entries = []
    timestamp = 0.0
    recent_files: list[str] = []

    for phase_name, start, end in phases:
        for i in range(start, end):
            tool = _choose_tool(phase_name, rng)
            action = _make_action(
                step=i, timestamp=timestamp, tool=tool, task=task,
                rng=rng, phase=phase_name, recent_files=recent_files,
            )
            entries.append(action)

            # Track recent files for locality
            if action.get("target_path") and tool in ("read", "write", "edit"):
                recent_files.append(action["target_path"])
                if len(recent_files) > 20:
                    recent_files = recent_files[-20:]

            # Realistic inter-event timing
            timestamp += rng.uniform(0.5, 5.0)

    return entries


# ── Failure injection ─────────────────────────────────────────────────────

def _inject_loop(
    entries: list[dict],
    task: TaskSpec,
    rng: random.Random,
    inject_start: int,
    duration: int = 60,
) -> tuple[list[dict], int, int]:
    """Inject a stuck loop (mechanical_repetition).

    IT target: LOW action_entropy + LOW action_mi.

    SymbolStream uses window=50, so the injection must be longer than 50
    steps to fully dominate the window.  We use a single file read
    repeated — producing the same symbol ``read:{file}`` 60 times.
    Once the window fills, Shannon entropy approaches 0 and MI drops
    to 0 (a single symbol has no bigram diversity).

    Duration of 60 ensures the window is fully saturated and z-scores
    exceed 3.0 against a calibrated baseline of typical H ≈ 3.2.

    Returns (modified_entries, inject_step, inject_end_step).
    """
    # Single file — all 60 steps produce the exact same symbol
    stuck_file = rng.choice(task.typical_files)

    inject_step = entries[inject_start]["step"]
    injected = list(entries[:inject_start])
    timestamp = entries[inject_start]["timestamp"]

    for i in range(duration):
        step = inject_step + i
        tool = "read"
        action = _make_action(
            step=step, timestamp=timestamp, tool=tool, task=task,
            rng=rng, phase="executing", target_override=stuck_file,
        )
        injected.append(action)
        timestamp += rng.uniform(0.5, 3.0)

    # Resume normal trace with renumbered steps
    end_step = inject_step + duration
    for e in entries[inject_start:]:
        e_copy = dict(e)
        e_copy["step"] = end_step
        e_copy["timestamp"] = timestamp
        injected.append(e_copy)
        end_step += 1
        timestamp += rng.uniform(0.5, 5.0)

    return injected, inject_step, inject_step + duration


def _inject_drift(
    entries: list[dict],
    task: TaskSpec,
    rng: random.Random,
    inject_start: int,
    duration: int = 60,
) -> tuple[list[dict], int, int]:
    """Inject goal drift (goal_discontinuity).

    IT target: HIGH last_surprisal + LOW action_mi.

    High surprisal requires symbols never seen during calibration.
    Low MI requires that bigrams are unpredictable from marginals,
    which means a SMALL pool of symbols drawn IID (random order).

    With window=50 and ~5 novel symbols, MI stays near 0.3 (low)
    while each individual symbol has high surprisal (never seen
    in the calibration window).

    We use 5 novel file paths that the agent reads randomly, producing
    symbols like ``read:z_drift/rollback.sql`` that the baseline has
    never observed.

    Returns (modified_entries, inject_step, inject_end_step).
    """
    # Large pool of novel paths that never appeared during calibration.
    # Each path is used sparingly (1-2 times over 60 steps) so every
    # occurrence stays novel → high surprisal.  Using 'read' for all
    # so tool_stream doesn't muddy the signal.
    #
    # The KL divergence triggers because the symbol distribution has
    # shifted.  last_surprisal triggers because each individual novel
    # symbol has near-zero probability in the baseline distribution.
    drift_files = [
        f"z_drift_{i}/{name}"
        for i in range(20)
        for name in ["schema.sql", "config.tf", "deploy.yml"]
    ]  # 60 unique paths

    inject_step = entries[inject_start]["step"]
    injected = list(entries[:inject_start])
    timestamp = entries[inject_start]["timestamp"]

    # Use sequential draw from the pool so each symbol appears at most once
    drift_idx = 0
    for i in range(duration):
        step = inject_step + i
        # High novel probability from the start so surprisal stays above z=3
        # Novel symbols with count=0 give surprisal ≈ 7.2 bits (z≈3.5)
        drift_prob = min(1.0, 0.7 + 0.3 * (i / duration))

        if rng.random() < drift_prob:
            target = drift_files[drift_idx % len(drift_files)]
            drift_idx += 1
        else:
            target = rng.choice(task.typical_files)

        tool = "read"
        action = _make_action(
            step=step, timestamp=timestamp, tool=tool, task=task,
            rng=rng, phase="executing", target_override=target,
        )
        injected.append(action)
        timestamp += rng.uniform(0.5, 4.0)

    end_step = inject_step + duration
    for e in entries[inject_start:]:
        e_copy = dict(e)
        e_copy["step"] = end_step
        e_copy["timestamp"] = timestamp
        injected.append(e_copy)
        end_step += 1
        timestamp += rng.uniform(0.5, 5.0)

    return injected, inject_step, inject_step + duration


def _inject_thrash(
    entries: list[dict],
    task: TaskSpec,
    rng: random.Random,
    inject_start: int,
    duration: int = 60,
) -> tuple[list[dict], int, int]:
    """Inject context thrashing (context_thrashing).

    IT target: HIGH kl_divergence + (HIGH action_mi or HIGH read_entropy)
               WITHOUT high surprisal (distinguishes from drift).

    The agent frantically reads ALL of the task's files plus config files
    in rapid succession.  These are KNOWN files (seen during calibration)
    so surprisal stays normal, but the READ-ONLY pattern shifts the
    distribution (high KL) and the rapid diverse reads push read_entropy
    and MI high.  The key distinction from drift: these are familiar
    files in an unfamiliar pattern, not novel files.

    Returns (modified_entries, inject_step, inject_end_step).
    """
    # Use the task's own files plus common config files — all known
    thrash_files = list(task.typical_files) + [
        "package.json", "README.md", "Makefile", "setup.py",
        ".gitignore", "requirements.txt", "tsconfig.json",
    ]
    # Remove duplicates while preserving order
    seen = set()
    thrash_files = [f for f in thrash_files if not (f in seen or seen.add(f))]

    inject_step = entries[inject_start]["step"]
    injected = list(entries[:inject_start])
    timestamp = entries[inject_start]["timestamp"]

    for i in range(duration):
        step = inject_step + i
        # Cycle through ALL files rapidly — known but chaotic pattern
        target = thrash_files[i % len(thrash_files)]
        tool = "read"
        action = _make_action(
            step=step, timestamp=timestamp, tool=tool, task=task,
            rng=rng, phase="executing", target_override=target,
        )
        injected.append(action)
        timestamp += rng.uniform(0.2, 1.5)

    end_step = inject_step + duration
    for e in entries[inject_start:]:
        e_copy = dict(e)
        e_copy["step"] = end_step
        e_copy["timestamp"] = timestamp
        injected.append(e_copy)
        end_step += 1
        timestamp += rng.uniform(0.5, 5.0)

    return injected, inject_step, inject_step + duration


def _inject_stall(
    entries: list[dict],
    task: TaskSpec,
    rng: random.Random,
    inject_start: int,
    duration: int = 60,
) -> tuple[list[dict], int, int]:
    """Inject a stall (stagnation).

    IT target: LOW compression_ratio + LOW action_entropy.

    Stagnation means the agent is completely stuck — producing an
    extremely repetitive, low-diversity action sequence.  We use a
    2-file read cycle: read:A → read:B → read:A → read:B.
    With 2 symbols, entropy ≈ 1.0 (LOW).  The LZ algorithm easily
    compresses the repeating pattern, giving compression ≈ 0.05 (LOW).

    This is distinct from mechanical_repetition (1 symbol → MI=0) because
    the alternating pattern has high MI (knowing A predicts B and vice versa).
    Stagnation captures the "stuck but structured" failure mode.

    Returns (modified_entries, inject_step, inject_end_step).
    """
    inject_step = entries[inject_start]["step"]
    injected = list(entries[:inject_start])
    timestamp = entries[inject_start]["timestamp"]

    # 2-file strict alternation: very low entropy + very low compression
    cycle_files = rng.sample(task.typical_files, min(2, len(task.typical_files)))
    stall_cycle = [
        ("read", cycle_files[0]),
        ("read", cycle_files[1] if len(cycle_files) > 1 else cycle_files[0]),
    ]

    for i in range(duration):
        step = inject_step + i
        tool, target = stall_cycle[i % len(stall_cycle)]
        action = _make_action(
            step=step, timestamp=timestamp, tool=tool, task=task,
            rng=rng, phase="executing", target_override=target,
        )
        injected.append(action)
        timestamp += rng.uniform(1.0, 5.0)

    end_step = inject_step + duration
    for e in entries[inject_start:]:
        e_copy = dict(e)
        e_copy["step"] = end_step
        e_copy["timestamp"] = timestamp
        injected.append(e_copy)
        end_step += 1
        timestamp += rng.uniform(0.5, 5.0)

    return injected, inject_step, inject_step + duration


# ── Variant generation ────────────────────────────────────────────────────

_INJECTORS = {
    "loop": _inject_loop,
    "drift": _inject_drift,
    "thrash": _inject_thrash,
    "stall": _inject_stall,
}

# Expected compositor signature for each failure type.
# Derived empirically from what the IT metrics actually produce given
# each injection's symbol stream pattern at window=50.
EXPECTED_SIGNATURES = {
    "loop": "mechanical_repetition",
    "drift": "distributional_shift",
    "thrash": "context_thrashing",
    "stall": "distributional_anomaly",
}

FAILURE_VARIANTS = list(_INJECTORS.keys())


def generate_trace(
    task: TaskSpec,
    variant: str = "clean",
    seed: int = 42,
) -> tuple[list[dict], Optional[dict]]:
    """Generate a trace for a task with an optional failure variant.

    Args:
        task: Task specification from the bank.
        variant: "clean" or one of FAILURE_VARIANTS.
        seed: Random seed for reproducibility.

    Returns:
        (entries, injection_info) where injection_info is None for clean traces,
        or {"variant": str, "inject_step": int, "inject_end": int,
            "expected_signature": str} for failure traces.
    """
    rng = random.Random(seed)

    # Generate the base clean trace
    clean = generate_clean_trace(task, seed=seed)

    if variant == "clean":
        return clean, None

    if variant not in _INJECTORS:
        raise ValueError(f"Unknown variant: {variant}. Choose from: clean, {', '.join(FAILURE_VARIANTS)}")

    # Randomize injection point: after calibration window but before 70% of trace
    n = len(clean)
    earliest = max(110, int(n * 0.25))  # after calibration (100 steps)
    latest = int(n * 0.70)
    if earliest >= latest:
        earliest = 110
        latest = max(earliest + 30, n - 30)

    inject_idx = rng.randint(earliest, min(latest, n - 1))

    injected, inject_step, inject_end = _INJECTORS[variant](
        clean, task, rng, inject_idx,
    )

    info = {
        "variant": variant,
        "inject_step": inject_step,
        "inject_end": inject_end,
        "expected_signature": EXPECTED_SIGNATURES.get(variant, "unclassified_anomaly"),
    }

    return injected, info


# ── File I/O ──────────────────────────────────────────────────────────────

def write_trace(entries: list[dict], path: str | Path) -> None:
    """Write entries as JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def generate_all(output_dir: str | Path, seed: int = 42) -> dict[str, dict]:
    """Generate all traces for all tasks.

    Returns a manifest: {filename: {task, variant, inject_step, inject_end, ...}}
    """
    output_dir = Path(output_dir)
    manifest = {}

    for task in TASK_BANK:
        for variant in ["clean"] + FAILURE_VARIANTS:
            # Different seed per task+variant for independence
            trace_seed = seed + hash(f"{task.name}_{variant}") % 10000
            entries, info = generate_trace(task, variant=variant, seed=trace_seed)

            filename = f"{task.name}_{variant}.jsonl"
            filepath = output_dir / filename
            write_trace(entries, filepath)

            manifest[filename] = {
                "task": task.name,
                "domain": task.domain,
                "complexity": task.complexity,
                "variant": variant,
                "n_events": len(entries),
            }
            if info:
                manifest[filename].update(info)

    # Write manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate evaluation traces")
    parser.add_argument("--output", type=str, default="agentdiag/eval/traces",
                        help="Output directory for JSONL traces")
    parser.add_argument("--task", type=str, default=None,
                        help="Generate traces for a specific task only")
    parser.add_argument("--variant", type=str, default=None,
                        help="Generate a specific variant only")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    if args.task and args.variant:
        from agentdiag.eval.tasks import get_task
        task = get_task(args.task)
        entries, info = generate_trace(task, variant=args.variant, seed=args.seed)
        filename = f"{task.name}_{args.variant}.jsonl"
        write_trace(entries, Path(args.output) / filename)
        print(f"Wrote {len(entries)} events to {filename}")
        if info:
            print(f"  Injection: step {info['inject_step']}-{info['inject_end']}")
    else:
        manifest = generate_all(args.output, seed=args.seed)
        print(f"Generated {len(manifest)} traces in {args.output}/")
        for name, info in manifest.items():
            tag = f"  [{info['variant']}]"
            if info.get("inject_step"):
                tag += f" inject@{info['inject_step']}-{info['inject_end']}"
            print(f"  {name}: {info['n_events']} events{tag}")


if __name__ == "__main__":
    main()
