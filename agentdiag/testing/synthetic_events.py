"""Synthetic event generator for analysis layer testing.

Produces realistic mixed-source ObservableEvent streams that simulate
full harness runs. Agent 1 owns this because Agent 1 knows what realistic
harness + OpenViking event sequences look like.

Two variants:
  - ``generate_healthy_run()`` — normal harness execution, 2 sprints, both pass
  - ``generate_anomalous_run()`` — same structure but with injected pathologies:
      * Stuck loop (mechanical repetition) around step 200
      * Goal drift (incoherent exploration) around step 350
      * Context thrashing (tier escalation storm) around step 500

Both produce ~600-700 events mixing tool calls, memory operations, phase
boundaries, contract events, and evaluation results.

Usage::

    from agentdiag.testing.synthetic_events import generate_healthy_run, generate_anomalous_run

    healthy = generate_healthy_run()
    anomalous = generate_anomalous_run()
"""

from __future__ import annotations

import random
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


# ---------------------------------------------------------------------------
# Constants for realistic distributions
# ---------------------------------------------------------------------------

_TOOL_NAMES_GATHERING = ["Read", "Glob", "Grep", "Bash", "WebSearch"]
_TOOL_NAMES_EXECUTING = ["Edit", "Write", "Bash", "Read", "Grep"]
_TOOL_NAMES_VERIFYING = ["Bash", "Read", "Grep"]
_TOOL_NAMES_DELIVERING = ["Bash", "Write"]

_FILE_PATHS = [
    "src/main.py", "src/auth.py", "src/models.py", "src/routes.py",
    "src/utils.py", "tests/test_main.py", "tests/test_auth.py",
    "package.json", "README.md", "src/config.py", "src/db.py",
    "src/middleware.py", "tests/test_routes.py", "src/schema.py",
]

_NAMESPACES = [
    "agent/planner/memories", "agent/generator/skills",
    "agent/evaluator/skills", "agent/evaluator/memories",
    "resources/current_project/contracts", "resources/current_project/qa_reports",
    "resources/reference_projects", "user/memories/preferences",
]

_EVAL_CRITERIA = ["correctness", "completeness", "code_quality", "test_coverage"]


def _rng(seed: int = 42) -> random.Random:
    return random.Random(seed)


# ---------------------------------------------------------------------------
# Phase block generators
# ---------------------------------------------------------------------------

def _make_session_start(step: int, t: float, goal: str) -> list[ObservableEvent]:
    return [ObservableEvent(
        step=step, timestamp=t, event_type=EventType.SESSION_START,
        metadata={"goal": goal, "source": "synthetic"},
    )]


def _make_planning_block(
    start_step: int, start_t: float, rng: random.Random,
) -> tuple[list[ObservableEvent], int, float]:
    """Generate planning phase: memory loads + planner tool calls."""
    events: list[ObservableEvent] = []
    step, t = start_step, start_t

    # Phase boundary
    step += 1; t += 0.05
    events.append(phase_boundary_event(
        step=step, timestamp=t, phase=HarnessPhase.PLANNING,
        previous_phase=None, agent_role=AgentRole.PLANNER,
    ))

    # Planner handoff
    step += 1; t += 0.02
    events.append(ObservableEvent(
        step=step, timestamp=t, event_type=EventType.AGENT_HANDOFF,
        agent_role=AgentRole.PLANNER,
    ))

    # Load baseline context at L0/L1
    for ns in ["agent/planner/memories", "user/memories/preferences", "resources/reference_projects"]:
        tier = MemoryTier.L0 if "reference" in ns else MemoryTier.L1
        step += 1; t += rng.uniform(0.1, 0.3)
        events.append(memory_load_event(
            step=step, timestamp=t,
            uri=f"viking://{ns}",
            tier=tier,
            token_count=rng.randint(50, 500),
            namespace=ns,
        ))

    # Planner reads some files
    for _ in range(rng.randint(3, 6)):
        step += 1; t += rng.uniform(0.2, 0.8)
        events.append(tool_call_event(
            step=step, timestamp=t,
            tool_name=rng.choice(["Read", "Glob", "Grep"]),
            target_path=rng.choice(_FILE_PATHS),
            duration_ms=rng.uniform(50, 300),
        ))

    return events, step, t


def _make_contract_block(
    start_step: int, start_t: float, sprint_n: int,
    prev_phase: HarnessPhase, rng: random.Random,
) -> tuple[list[ObservableEvent], int, float]:
    """Generate contract negotiation phase."""
    events: list[ObservableEvent] = []
    step, t = start_step, start_t

    step += 1; t += 0.05
    events.append(phase_boundary_event(
        step=step, timestamp=t, phase=HarnessPhase.CONTRACT_NEGOTIATION,
        previous_phase=prev_phase, sprint_number=sprint_n,
        agent_role=AgentRole.ORCHESTRATOR,
    ))

    # Proposal
    step += 1; t += 0.1
    events.append(ObservableEvent(
        step=step, timestamp=t, event_type=EventType.CONTRACT_EVENT,
        contract_status="proposed", sprint_number=sprint_n,
        agent_role=AgentRole.GENERATOR,
        metadata={"deliverables": 3, "criteria": 4, "goal": f"Sprint {sprint_n}"},
    ))

    # Evaluator loads context to review
    step += 1; t += 0.2
    events.append(memory_load_event(
        step=step, timestamp=t,
        uri="viking://agent/evaluator/memories/calibration",
        tier=MemoryTier.L2, token_count=rng.randint(200, 800),
        namespace="agent/evaluator/memories",
    ))

    # Accepted
    step += 1; t += 0.15
    events.append(ObservableEvent(
        step=step, timestamp=t, event_type=EventType.CONTRACT_EVENT,
        contract_status="accepted", sprint_number=sprint_n,
        agent_role=AgentRole.ORCHESTRATOR,
        metadata={"deliverables": 3, "criteria": 4},
    ))

    # Store contract
    step += 1; t += 0.05
    events.append(memory_store_event(
        step=step, timestamp=t,
        uri=f"viking://resources/current_project/contracts/sprint_{sprint_n}_accepted",
        token_count=rng.randint(100, 300),
        namespace="resources/current_project/contracts",
    ))

    return events, step, t


def _make_execution_block(
    start_step: int, start_t: float, sprint_n: int,
    n_events: int, rng: random.Random,
    prev_phase: HarnessPhase = HarnessPhase.CONTRACT_NEGOTIATION,
) -> tuple[list[ObservableEvent], int, float]:
    """Generate execution phase: tool calls + occasional memory loads."""
    events: list[ObservableEvent] = []
    step, t = start_step, start_t

    # Phase boundary BEFORE first event
    step += 1; t += 0.05
    events.append(phase_boundary_event(
        step=step, timestamp=t, phase=HarnessPhase.EXECUTING,
        previous_phase=prev_phase, sprint_number=sprint_n,
        agent_role=AgentRole.GENERATOR,
    ))

    step += 1; t += 0.02
    events.append(ObservableEvent(
        step=step, timestamp=t, event_type=EventType.AGENT_HANDOFF,
        agent_role=AgentRole.GENERATOR, sprint_number=sprint_n,
    ))

    # Generator loads skills
    step += 1; t += 0.1
    events.append(memory_load_event(
        step=step, timestamp=t,
        uri="viking://agent/generator/skills",
        tier=MemoryTier.L1, token_count=rng.randint(100, 400),
        namespace="agent/generator/skills",
    ))

    # Load contract at L2
    step += 1; t += 0.1
    events.append(memory_load_event(
        step=step, timestamp=t,
        uri=f"viking://resources/current_project/contracts/sprint_{sprint_n}",
        tier=MemoryTier.L2, token_count=rng.randint(100, 300),
        namespace="resources/current_project/contracts",
    ))

    # Main tool call loop
    for i in range(n_events):
        step += 1; t += rng.uniform(0.3, 2.0)

        # 70% tool calls, 15% file reads, 10% file writes, 5% memory ops
        r = rng.random()
        if r < 0.70:
            events.append(tool_call_event(
                step=step, timestamp=t,
                tool_name=rng.choice(_TOOL_NAMES_EXECUTING),
                target_path=rng.choice(_FILE_PATHS),
                duration_ms=rng.uniform(100, 2000),
            ))
        elif r < 0.85:
            events.append(file_read_event(
                step=step, timestamp=t,
                path=rng.choice(_FILE_PATHS),
                output_tokens=rng.randint(50, 2000),
            ))
        elif r < 0.95:
            events.append(file_write_event(
                step=step, timestamp=t,
                path=rng.choice(_FILE_PATHS),
                input_tokens=rng.randint(50, 1500),
            ))
        else:
            # Occasional memory load during execution
            ns = rng.choice(["agent/generator/skills", "agent/generator/memories"])
            events.append(memory_load_event(
                step=step, timestamp=t,
                uri=f"viking://{ns}",
                tier=rng.choice([MemoryTier.L0, MemoryTier.L1]),
                token_count=rng.randint(30, 200),
                namespace=ns,
            ))

    return events, step, t


def _make_verification_block(
    start_step: int, start_t: float, sprint_n: int,
    score: float, rng: random.Random,
) -> tuple[list[ObservableEvent], int, float]:
    """Generate verification phase: evaluator runs, grades, stores QA report."""
    events: list[ObservableEvent] = []
    step, t = start_step, start_t
    exec_start = start_step  # for step_range metadata

    # Phase boundary
    step += 1; t += 0.05
    events.append(phase_boundary_event(
        step=step, timestamp=t, phase=HarnessPhase.VERIFYING,
        previous_phase=HarnessPhase.EXECUTING, sprint_number=sprint_n,
        agent_role=AgentRole.EVALUATOR,
    ))

    step += 1; t += 0.02
    events.append(ObservableEvent(
        step=step, timestamp=t, event_type=EventType.AGENT_HANDOFF,
        agent_role=AgentRole.EVALUATOR, sprint_number=sprint_n,
    ))

    # Evaluator loads calibration + contract
    step += 1; t += 0.15
    events.append(memory_load_event(
        step=step, timestamp=t,
        uri="viking://agent/evaluator/skills",
        tier=MemoryTier.L2, token_count=rng.randint(200, 600),
        namespace="agent/evaluator/skills",
    ))

    # Evaluator tool calls (testing)
    for _ in range(rng.randint(5, 10)):
        step += 1; t += rng.uniform(0.5, 2.0)
        events.append(tool_call_event(
            step=step, timestamp=t,
            tool_name=rng.choice(_TOOL_NAMES_VERIFYING),
            target_path=rng.choice(_FILE_PATHS),
            duration_ms=rng.uniform(200, 3000),
        ))

    # Evaluation results
    eval_meta = {"step_range": (exec_start, step), "iteration": 1}
    for criterion in _EVAL_CRITERIA:
        c_score = score + rng.uniform(-0.1, 0.1)
        c_score = max(0.0, min(1.0, c_score))
        step += 1; t += 0.05
        events.append(ObservableEvent(
            step=step, timestamp=t, event_type=EventType.EVALUATION_RESULT,
            evaluation_score=round(c_score, 3),
            evaluation_criterion=criterion,
            sprint_number=sprint_n,
            agent_role=AgentRole.EVALUATOR,
            metadata=eval_meta,
        ))

    # Overall
    step += 1; t += 0.05
    events.append(ObservableEvent(
        step=step, timestamp=t, event_type=EventType.EVALUATION_RESULT,
        evaluation_score=round(score, 3),
        evaluation_criterion="overall",
        sprint_number=sprint_n,
        agent_role=AgentRole.EVALUATOR,
        metadata={**eval_meta, "passed": score >= 0.7},
    ))

    # Store QA report
    step += 1; t += 0.05
    events.append(memory_store_event(
        step=step, timestamp=t,
        uri=f"viking://resources/current_project/qa_reports/sprint_{sprint_n}",
        token_count=rng.randint(100, 400),
        namespace="resources/current_project/qa_reports",
    ))

    return events, step, t


def _make_retrospective_block(
    start_step: int, start_t: float, rng: random.Random,
    passed_sprints: list[int], failed_sprints: list[int],
) -> tuple[list[ObservableEvent], int, float]:
    """Generate retrospective phase: skill crystallization stores."""
    events: list[ObservableEvent] = []
    step, t = start_step, start_t

    step += 1; t += 0.05
    events.append(phase_boundary_event(
        step=step, timestamp=t, phase=HarnessPhase.RETROSPECTIVE,
        previous_phase=HarnessPhase.VERIFYING,
        agent_role=AgentRole.ORCHESTRATOR,
    ))

    # Store design patterns from passed sprints
    for n in passed_sprints:
        step += 1; t += 0.1
        events.append(memory_store_event(
            step=step, timestamp=t,
            uri=f"viking://agent/generator/skills/design_patterns/sprint_{n}",
            token_count=rng.randint(50, 200),
            namespace="agent/generator/skills",
        ))

    # Store bug patterns from failed sprints
    for n in failed_sprints:
        step += 1; t += 0.1
        events.append(memory_store_event(
            step=step, timestamp=t,
            uri=f"viking://agent/generator/memories/bug_patterns/sprint_{n}",
            token_count=rng.randint(50, 200),
            namespace="agent/generator/memories",
        ))
        for criterion in rng.sample(_EVAL_CRITERIA, k=min(2, len(_EVAL_CRITERIA))):
            step += 1; t += 0.05
            events.append(memory_store_event(
                step=step, timestamp=t,
                uri=f"viking://agent/evaluator/skills/effective_tests/{criterion}",
                token_count=50,
                namespace="agent/evaluator/skills",
            ))

    # Session summary store
    step += 1; t += 0.1
    events.append(memory_store_event(
        step=step, timestamp=t,
        uri="viking://resources/current_project/session_summary",
        token_count=rng.randint(200, 600),
        namespace="resources/current_project",
    ))

    return events, step, t


def _make_session_end(step: int, t: float) -> list[ObservableEvent]:
    return [ObservableEvent(
        step=step, timestamp=t, event_type=EventType.SESSION_END,
        metadata={"committed": True},
    )]


# ---------------------------------------------------------------------------
# Anomaly injection
# ---------------------------------------------------------------------------

def _inject_stuck_loop(
    events: list[ObservableEvent],
    center_step: int,
    rng: random.Random,
) -> list[ObservableEvent]:
    """Inject mechanical repetition: same tool+path repeated 40 times.

    Low entropy, low MI — the agent is stuck reading the same file.
    """
    injected: list[ObservableEvent] = []
    target_path = "src/main.py"
    tool = "Read"

    # Find the event nearest to center_step to get timestamp context
    base_t = 0.0
    for e in events:
        if e.step <= center_step:
            base_t = e.timestamp

    for i in range(40):
        step = center_step + i
        t = base_t + i * 0.5  # very regular cadence (mechanical)
        injected.append(tool_call_event(
            step=step, timestamp=t,
            tool_name=tool, target_path=target_path,
            duration_ms=rng.uniform(100, 150),  # consistent latency
        ))

    return injected


def _inject_goal_drift(
    events: list[ObservableEvent],
    center_step: int,
    rng: random.Random,
) -> list[ObservableEvent]:
    """Inject incoherent exploration: wild tool diversity, no pattern.

    High entropy, low MI — the agent has lost coherence.
    """
    injected: list[ObservableEvent] = []
    all_tools = ["Read", "Edit", "Write", "Bash", "Glob", "Grep",
                 "WebSearch", "WebFetch", "Agent", "NotebookEdit"]
    all_paths = _FILE_PATHS + [
        "package.json", "docker-compose.yml", ".env", "Makefile",
        "tsconfig.json", "webpack.config.js", "styles.css", ".gitignore",
    ]

    base_t = 0.0
    for e in events:
        if e.step <= center_step:
            base_t = e.timestamp

    for i in range(35):
        step = center_step + i
        t = base_t + i * rng.uniform(0.1, 3.0)
        # Random tools, random paths — no correlation between consecutive events
        injected.append(tool_call_event(
            step=step, timestamp=t,
            tool_name=rng.choice(all_tools),
            target_path=rng.choice(all_paths),
            duration_ms=rng.uniform(50, 5000),
        ))

    return injected


def _inject_context_thrashing(
    events: list[ObservableEvent],
    center_step: int,
    rng: random.Random,
) -> list[ObservableEvent]:
    """Inject context thrashing: rapid tier escalations across many namespaces.

    High tier_escalation_rate, high namespace_entropy — the agent is
    frantically loading and reloading context without settling.
    """
    injected: list[ObservableEvent] = []
    namespaces = _NAMESPACES.copy()

    base_t = 0.0
    for e in events:
        if e.step <= center_step:
            base_t = e.timestamp

    for i in range(40):
        step = center_step + i
        t = base_t + i * 0.3
        ns = rng.choice(namespaces)

        # Alternate between loads and escalations
        if i % 3 == 0:
            # Tier escalation L0 -> L1 -> L2
            from_tier = rng.choice([MemoryTier.L0, MemoryTier.L1])
            to_tier = MemoryTier.L1 if from_tier == MemoryTier.L0 else MemoryTier.L2
            injected.append(tier_escalation_event(
                step=step, timestamp=t,
                uri=f"viking://{ns}",
                from_tier=from_tier, to_tier=to_tier,
                token_count=rng.randint(200, 2000),
            ))
        elif i % 3 == 1:
            # Memory load at high tier
            injected.append(memory_load_event(
                step=step, timestamp=t,
                uri=f"viking://{ns}",
                tier=rng.choice([MemoryTier.L1, MemoryTier.L2]),
                token_count=rng.randint(200, 2000),
                namespace=ns,
            ))
        else:
            # Evict and reload different namespace
            injected.append(ObservableEvent(
                step=step, timestamp=t,
                event_type=EventType.MEMORY_EVICT,
                viking_uri=f"viking://{ns}",
                namespace=ns,
                token_count=rng.randint(100, 1000),
            ))

    return injected


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_healthy_run(seed: int = 42) -> list[ObservableEvent]:
    """Generate a realistic healthy harness run (~650 events).

    Structure:
      - Session start + planning (~15 events)
      - Sprint 1: contract + execution (150 events) + verification (pass at 0.82)
      - Sprint 2: contract + execution (150 events) + verification (pass at 0.88)
      - Retrospective + session end (~20 events)
    """
    rng = _rng(seed)
    all_events: list[ObservableEvent] = []
    step, t = 0, 0.0

    # Session start
    step += 1; t += 0.1
    all_events.extend(_make_session_start(step, t, "Build a login page with OAuth"))

    # Planning
    block, step, t = _make_planning_block(step, t, rng)
    all_events.extend(block)

    # Sprint 1
    block, step, t = _make_contract_block(step, t, 1, HarnessPhase.PLANNING, rng)
    all_events.extend(block)
    block, step, t = _make_execution_block(step, t, 1, 150, rng)
    all_events.extend(block)
    block, step, t = _make_verification_block(step, t, 1, 0.82, rng)
    all_events.extend(block)

    # Sprint 2
    block, step, t = _make_contract_block(step, t, 2, HarnessPhase.VERIFYING, rng)
    all_events.extend(block)
    block, step, t = _make_execution_block(step, t, 2, 150, rng)
    all_events.extend(block)
    block, step, t = _make_verification_block(step, t, 2, 0.88, rng)
    all_events.extend(block)

    # Retrospective
    block, step, t = _make_retrospective_block(step, t, rng, [1, 2], [])
    all_events.extend(block)

    # Session end
    step += 1; t += 0.1
    all_events.extend(_make_session_end(step, t))

    return all_events


def generate_anomalous_run(seed: int = 123) -> list[ObservableEvent]:
    """Generate a harness run with injected anomalies (~700 events).

    Same structure as healthy but with three pathologies:
      - ~step 200: stuck loop (mechanical repetition) — 40 identical Read events
      - ~step 350: goal drift (incoherent exploration) — 35 random tool calls
      - ~step 500: context thrashing — 40 rapid tier escalations

    The anomalies are spliced into the execution phases of a longer run
    (3 sprints with more events each).
    """
    rng = _rng(seed)
    all_events: list[ObservableEvent] = []
    step, t = 0, 0.0

    # Session start
    step += 1; t += 0.1
    all_events.extend(_make_session_start(step, t, "Build full-stack app with auth + dashboard"))

    # Planning
    block, step, t = _make_planning_block(step, t, rng)
    all_events.extend(block)

    # Sprint 1 — long execution, stuck loop injected around step 200
    block, step, t = _make_contract_block(step, t, 1, HarnessPhase.PLANNING, rng)
    all_events.extend(block)
    # First chunk of execution (up to ~step 180)
    block, step, t = _make_execution_block(step, t, 1, 160, rng)
    all_events.extend(block)

    # ANOMALY 1: Stuck loop at current position (~step 200)
    stuck = _inject_stuck_loop(all_events, step + 1, rng)
    all_events.extend(stuck)
    step += len(stuck)
    t = stuck[-1].timestamp + 0.5

    # More normal execution after recovery
    block, step, t = _make_execution_block(step, t, 1, 40, rng, HarnessPhase.EXECUTING)
    all_events.extend(block)
    block, step, t = _make_verification_block(step, t, 1, 0.55, rng)  # fails — stuck loop hurt quality
    all_events.extend(block)

    # Sprint 2 — goal drift injected around step 350
    block, step, t = _make_contract_block(step, t, 2, HarnessPhase.VERIFYING, rng)
    all_events.extend(block)
    block, step, t = _make_execution_block(step, t, 2, 60, rng)
    all_events.extend(block)

    # ANOMALY 2: Goal drift at current position (~step 350)
    drift = _inject_goal_drift(all_events, step + 1, rng)
    all_events.extend(drift)
    step += len(drift)
    t = drift[-1].timestamp + 0.5

    block, step, t = _make_execution_block(step, t, 2, 30, rng, HarnessPhase.EXECUTING)
    all_events.extend(block)
    block, step, t = _make_verification_block(step, t, 2, 0.45, rng)  # fails
    all_events.extend(block)

    # Sprint 3 — context thrashing injected around step 500
    block, step, t = _make_contract_block(step, t, 3, HarnessPhase.VERIFYING, rng)
    all_events.extend(block)
    block, step, t = _make_execution_block(step, t, 3, 60, rng)
    all_events.extend(block)

    # ANOMALY 3: Context thrashing at current position (~step 500)
    thrash = _inject_context_thrashing(all_events, step + 1, rng)
    all_events.extend(thrash)
    step += len(thrash)
    t = thrash[-1].timestamp + 0.5

    block, step, t = _make_execution_block(step, t, 3, 30, rng, HarnessPhase.EXECUTING)
    all_events.extend(block)
    block, step, t = _make_verification_block(step, t, 3, 0.60, rng)  # barely fails
    all_events.extend(block)

    # Retrospective
    block, step, t = _make_retrospective_block(step, t, rng, [], [1, 2, 3])
    all_events.extend(block)

    # Session end
    step += 1; t += 0.1
    all_events.extend(_make_session_end(step, t))

    # Re-number steps sequentially (injections may have gaps)
    for i, e in enumerate(all_events):
        e.step = i + 1

    return all_events
