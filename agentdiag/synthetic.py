"""
Synthetic trace generators for testing agent diagnostics.

Each generator produces a realistic list[TraceEvent] exhibiting a specific
failure mode. Used for unit testing detectors and demos.
"""

from __future__ import annotations

import random

from agentdiag.models import TraceEvent


COMMON_TOOLS = ["search_docs", "read_file", "write_file", "run_code", "ask_user", "web_search"]


def _base_event(step: int, tool: str, **kw) -> TraceEvent:
    return TraceEvent(
        step=step,
        type="tool_call",
        tool=tool,
        latency_ms=kw.get("latency_ms", random.uniform(200, 1200)),
        success=kw.get("success", True),
        tokens_in=kw.get("tokens_in", random.randint(200, 800)),
        tokens_out=kw.get("tokens_out", random.randint(400, 1500)),
    )


def generate_normal_trace(n: int = 20, seed: int = 42) -> list[TraceEvent]:
    """A healthy agent trace: diverse tools, steady latency, low error rate."""
    random.seed(seed)
    events = []
    for i in range(n):
        tool = random.choice(COMMON_TOOLS)
        events.append(_base_event(
            step=i + 1,
            tool=tool,
            latency_ms=random.uniform(300, 1000),
            tokens_in=random.randint(300, 700),
            tokens_out=random.randint(500, 1200),
        ))
    return events


def generate_loop_trace(
    pattern: list[str] | None = None,
    repeats: int = 8,
    prefix: int = 3,
    seed: int = 42,
) -> list[TraceEvent]:
    """Agent gets stuck repeating the same tool pattern.

    Args:
        pattern: Tool sequence to repeat (default: ["search_docs", "search_docs"])
        repeats: Number of times the pattern repeats
        prefix: Normal events before the loop starts
    """
    random.seed(seed)
    if pattern is None:
        pattern = ["search_docs", "search_docs"]

    events = []
    step = 1

    # Normal prefix
    for _ in range(prefix):
        events.append(_base_event(step=step, tool=random.choice(COMMON_TOOLS)))
        step += 1

    # Looping section — same tokens_out to signal no state change
    for _ in range(repeats):
        for tool in pattern:
            events.append(_base_event(
                step=step,
                tool=tool,
                latency_ms=random.uniform(600, 1000) + (_ * 50),  # increasing latency
                tokens_in=500,
                tokens_out=1200,  # identical output
            ))
            step += 1

    return events


def generate_stall_trace(
    n: int = 20,
    stall_steps: list[int] | None = None,
    stall_latency_ms: float = 60000.0,
    seed: int = 42,
) -> list[TraceEvent]:
    """Agent stalls with abnormally long latencies at certain steps.

    Args:
        stall_steps: Which steps have stalls (default: [8, 14])
        stall_latency_ms: Latency for stall events (must exceed MIN_STALL_MS=30s)
    """
    random.seed(seed)
    if stall_steps is None:
        stall_steps = [8, 14]

    events = []
    for i in range(n):
        step = i + 1
        if step in stall_steps:
            events.append(_base_event(
                step=step,
                tool="read_file",
                latency_ms=stall_latency_ms,
                tokens_out=0,
            ))
        else:
            events.append(_base_event(
                step=step,
                tool=random.choice(COMMON_TOOLS),
                latency_ms=random.uniform(300, 800),
            ))

    return events


def generate_thrash_trace(
    n: int = 20,
    thrash_start: int = 5,
    thrash_length: int = 10,
    seed: int = 42,
) -> list[TraceEvent]:
    """Agent rapidly switches between tools with no progress.

    Args:
        thrash_start: Step where thrashing begins
        thrash_length: How many thrashing events
    """
    random.seed(seed)
    events = []
    step = 1

    # Normal prefix
    for _ in range(thrash_start - 1):
        events.append(_base_event(step=step, tool=random.choice(COMMON_TOOLS)))
        step += 1

    # Thrashing: cycle through tools, identical output hash (no real progress)
    thrash_tools = ["search_docs", "web_search", "read_file", "search_docs", "run_code"]
    for i in range(thrash_length):
        tool = thrash_tools[i % len(thrash_tools)]
        e = _base_event(
            step=step,
            tool=tool,
            latency_ms=random.uniform(100, 400),  # fast but useless
            tokens_in=500,
            tokens_out=200,  # identical low output
        )
        e.output_hash = "deadbeef"  # same hash = no state change
        events.append(e)
        step += 1

    # Tail
    for _ in range(n - thrash_start + 1 - thrash_length):
        if _ + step <= n:
            events.append(_base_event(step=step, tool=random.choice(COMMON_TOOLS)))
            step += 1

    return events


def generate_drift_trace(n: int = 24, seed: int = 42) -> list[TraceEvent]:
    """Agent gradually changes behavior: different tools, more errors, slower.

    First half: focused search+read. Second half: scattered tools, errors, high latency.
    """
    random.seed(seed)
    events = []
    mid = n // 2

    # First half: focused, healthy
    for i in range(mid):
        tool = random.choice(["search_docs", "read_file"])
        events.append(_base_event(
            step=i + 1,
            tool=tool,
            latency_ms=random.uniform(300, 600),
            success=True,
        ))

    # Second half: drifting — new tools, errors, high latency
    drift_tools = ["web_search", "ask_user", "run_code", "write_file"]
    for i in range(mid, n):
        tool = random.choice(drift_tools)
        events.append(_base_event(
            step=i + 1,
            tool=tool,
            latency_ms=random.uniform(1500, 4000),
            success=random.random() > 0.4,  # 40% error rate
            tokens_out=random.randint(100, 400),
        ))

    return events


def generate_cascade_trace(
    n: int = 20,
    cascade_start: int = 6,
    cascade_length: int = 6,
    seed: int = 42,
) -> list[TraceEvent]:
    """One error triggers a chain of downstream failures.

    Args:
        cascade_start: Step where the initial error occurs
        cascade_length: Length of the error cascade
    """
    random.seed(seed)
    events = []
    step = 1

    # Normal prefix
    for _ in range(cascade_start - 1):
        events.append(_base_event(step=step, tool=random.choice(COMMON_TOOLS)))
        step += 1

    # Cascade: consecutive failures with increasing latency
    cascade_tools = ["run_code", "read_file", "write_file", "run_code", "search_docs", "run_code"]
    for i in range(cascade_length):
        tool = cascade_tools[i % len(cascade_tools)]
        events.append(_base_event(
            step=step,
            tool=tool,
            latency_ms=random.uniform(2000, 5000) + i * 500,
            success=False,
            tokens_out=0,
            error_message=f"Error propagated from step {cascade_start}",
        ))
        step += 1

    # Recovery tail
    for _ in range(n - cascade_start + 1 - cascade_length):
        if step <= n:
            events.append(_base_event(step=step, tool=random.choice(COMMON_TOOLS)))
            step += 1

    return events


def generate_token_explosion_trace(n: int = 60, seed: int = 42) -> list[TraceEvent]:
    """Token usage grows exponentially over the trace.

    Uses n=60 (above MIN_EVENTS=50) with clear exponential growth:
    first quarter avg ~150 tokens, last quarter ~2000+ (growth >3x).
    """
    random.seed(seed)
    events = []

    for i in range(n):
        # Exponential growth: tokens ~ 100 * 1.06^i
        # Q1 (i=0..14): avg ~150, Q4 (i=45..59): avg ~1500+
        base_tokens = int(100 * (1.06 ** i))
        events.append(_base_event(
            step=i + 1,
            tool=random.choice(COMMON_TOOLS),
            latency_ms=random.uniform(300, 800) + i * 50,
            tokens_in=base_tokens // 3,
            tokens_out=base_tokens,
        ))

    return events


def generate_dead_end_trace(
    n: int = 20,
    dead_end_start: int = 6,
    dead_end_length: int = 8,
    seed: int = 42,
) -> list[TraceEvent]:
    """Agent gets stuck reasoning/planning without executing.

    Args:
        dead_end_start: Step where reasoning loop begins
        dead_end_length: How many consecutive reasoning steps
    """
    random.seed(seed)
    events = []
    step = 1

    # Normal prefix
    for _ in range(dead_end_start - 1):
        events.append(_base_event(step=step, tool=random.choice(COMMON_TOOLS)))
        step += 1

    # Dead end: pure reasoning, no execution
    for _ in range(dead_end_length):
        events.append(TraceEvent(
            step=step,
            type=random.choice(["reasoning", "planning"]),
            latency_ms=random.uniform(1000, 3000),
            tokens_in=random.randint(500, 1500),
            tokens_out=random.randint(800, 2000),
            success=True,
        ))
        step += 1

    # Tail
    for _ in range(n - dead_end_start + 1 - dead_end_length):
        if step <= n:
            events.append(_base_event(step=step, tool=random.choice(COMMON_TOOLS)))
            step += 1

    return events


def generate_recovery_failure_trace(
    n: int = 20,
    error_steps: list[int] | None = None,
    seed: int = 42,
) -> list[TraceEvent]:
    """Agent fails to recover after errors — retries same tool, keeps failing.

    Args:
        error_steps: Steps where errors occur (default: [5, 6, 7, 12, 13, 14])
    """
    random.seed(seed)
    if error_steps is None:
        error_steps = [5, 6, 7, 12, 13, 14]

    events = []
    for i in range(n):
        step = i + 1
        if step in error_steps:
            events.append(_base_event(
                step=step,
                tool="run_code",  # same tool every time
                latency_ms=random.uniform(2000, 5000),
                success=False,
                tokens_out=0,
                error_message="Execution failed",
            ))
        else:
            events.append(_base_event(
                step=step,
                tool=random.choice(COMMON_TOOLS),
            ))

    return events


# Convenience mapping
GENERATORS = {
    "normal": generate_normal_trace,
    "loop": generate_loop_trace,
    "stall": generate_stall_trace,
    "thrash": generate_thrash_trace,
    "drift": generate_drift_trace,
    "cascade": generate_cascade_trace,
    "token_explosion": generate_token_explosion_trace,
    "dead_end": generate_dead_end_trace,
    "recovery_failure": generate_recovery_failure_trace,
}
