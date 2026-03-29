"""CAFT Taxonomy: Unified mapping of all detectors to CAFT codes.

Maps the 8 original batch detectors AND 6 live CAFT detectors into
a single taxonomy with observable/latent classification.

The 33 CAFT failure types are organized into 8 top-level categories.
Each type is classified as OBSERVABLE (detectable from trace data alone)
or LATENT (requires deeper inference, not fair for automated detection).

Each type also carries an `ip_stage` mapping that identifies which stage
of the Information Processing model the failure occurs at:
  perception | attention | working_memory | decision_making | action | feedback

This module is the single source of truth for CAFT codes throughout
the system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Detectability(Enum):
    """Whether a CAFT failure type is directly observable in trace data."""
    OBSERVABLE = "observable"  # detectable from action/timing/tool patterns
    LATENT = "latent"          # requires semantic understanding or inference


@dataclass(frozen=True)
class CaftType:
    """A single CAFT failure type definition."""
    code: str              # e.g., "2.2"
    category: str          # e.g., "memory"
    name: str              # e.g., "step_repetition"
    label: str             # e.g., "Step Repetition"
    description: str       # one-line description
    detectability: Detectability
    detector_names: tuple[str, ...] = ()  # which detectors cover this type
    severity_default: str = "warning"     # default severity level
    ip_stage: str = ""     # IP model stage: perception|attention|working_memory|decision_making|action|feedback


# ──────────────────────────────────────────────────────────────────
# FULL CAFT TAXONOMY — 8 categories, 33 types
# ──────────────────────────────────────────────────────────────────

CAFT_TAXONOMY: dict[str, CaftType] = {}

def _register(*types: CaftType) -> None:
    for t in types:
        CAFT_TAXONOMY[t.code] = t


# ── Category 1: Perception Failures ──────────────────────────────
_register(
    CaftType("1.1", "perception", "input_misparse",
             "Input Misparse",
             "Agent misinterprets the user's request or input format",
             Detectability.LATENT, ip_stage="perception"),
    CaftType("1.2", "perception", "context_blindness",
             "Context Blindness",
             "Agent ignores relevant context available in the conversation",
             Detectability.LATENT, ip_stage="perception"),
    CaftType("1.3", "perception", "selective_attention",
             "Selective Attention",
             "Agent focuses on irrelevant details while missing critical information",
             Detectability.LATENT, ip_stage="perception"),
    CaftType("1.4", "perception", "format_confusion",
             "Format Confusion",
             "Agent misinterprets data format, schema, or structure",
             Detectability.LATENT, ip_stage="perception"),
)

# ── Category 2: Memory Failures ──────────────────────────────────
_register(
    CaftType("2.1", "memory", "context_loss",
             "Context Loss",
             "Agent re-reads resources it already processed",
             Detectability.OBSERVABLE,
             detector_names=("context_loss",), ip_stage="attention"),
    CaftType("2.2", "memory", "step_repetition",
             "Step Repetition",
             "Agent repeats the same operation multiple times consecutively",
             Detectability.OBSERVABLE,
             detector_names=("step_repetition", "loop"), ip_stage="working_memory"),
    CaftType("2.3", "memory", "state_amnesia",
             "State Amnesia",
             "Agent forgets modifications it made earlier in the session",
             Detectability.LATENT, ip_stage="working_memory"),
    CaftType("2.4", "memory", "goal_drift",
             "Goal Drift",
             "Agent's actions diverge from the original objective over time",
             Detectability.OBSERVABLE,
             detector_names=("goal_drift", "drift"), ip_stage="decision_making"),
)

# ── Category 3: Decision Failures ────────────────────────────────
_register(
    CaftType("3.1", "decision", "wrong_tool_selection",
             "Wrong Tool Selection",
             "Agent selects an inappropriate tool for the current task",
             Detectability.LATENT, ip_stage="decision_making"),
    CaftType("3.2", "decision", "overconfidence",
             "Overconfidence",
             "Agent proceeds with high confidence despite insufficient evidence",
             Detectability.LATENT, ip_stage="decision_making"),
    CaftType("3.3", "decision", "confirmation_bias",
             "Confirmation Bias",
             "Agent seeks evidence supporting its hypothesis while ignoring contradictions",
             Detectability.LATENT, ip_stage="decision_making"),
    CaftType("3.4", "decision", "analysis_paralysis",
             "Analysis Paralysis",
             "Agent gets stuck in extended reasoning without taking action",
             Detectability.OBSERVABLE,
             detector_names=("dead_end",), ip_stage="decision_making"),
    CaftType("3.5", "decision", "strategic_myopia",
             "Strategic Myopia",
             "Agent trapped in local optimization loops, unable to see broader strategy",
             Detectability.OBSERVABLE,
             detector_names=("strategic_myopia",), ip_stage="decision_making"),
)

# ── Category 4: Execution Failures ───────────────────────────────
_register(
    CaftType("4.1", "execution", "tool_misuse",
             "Tool Misuse",
             "Agent uses a tool with incorrect parameters or in wrong context",
             Detectability.OBSERVABLE,
             detector_names=("thrash",), ip_stage="action"),
    CaftType("4.2", "execution", "error_cascade",
             "Error Cascade",
             "A single error propagates through subsequent operations",
             Detectability.OBSERVABLE,
             detector_names=("cascade",), ip_stage="feedback"),
    CaftType("4.3", "execution", "recovery_failure",
             "Recovery Failure",
             "Agent fails to recover after encountering an error",
             Detectability.OBSERVABLE,
             detector_names=("recovery_failure",), ip_stage="feedback"),
    CaftType("4.4", "execution", "resource_exhaustion",
             "Resource Exhaustion",
             "Agent consumes excessive tokens, time, or API calls",
             Detectability.OBSERVABLE,
             detector_names=("token_explosion", "stall"), ip_stage="working_memory"),
)

# ── Category 5: Plan Structure Failures ──────────────────────────
_register(
    CaftType("5.1", "plan_structure", "incomplete_plan",
             "Incomplete Plan",
             "Agent's plan misses critical steps required for task completion",
             Detectability.LATENT, ip_stage="decision_making"),
    CaftType("5.2", "plan_structure", "wrong_ordering",
             "Wrong Ordering",
             "Agent executes correct steps but in wrong sequence",
             Detectability.LATENT, ip_stage="decision_making"),
    CaftType("5.3", "plan_structure", "missing_verification",
             "Missing Verification",
             "Agent completes changes without testing or reviewing results",
             Detectability.OBSERVABLE,
             detector_names=("missing_verification",), ip_stage="feedback"),
    CaftType("5.4", "plan_structure", "premature_termination",
             "Premature Termination",
             "Agent delivers results without completing verification phase",
             Detectability.OBSERVABLE,
             detector_names=("premature_termination",), ip_stage="feedback"),
)

# ── Category 6: Coordination Failures ────────────────────────────
_register(
    CaftType("6.1", "coordination", "subtask_interference",
             "Subtask Interference",
             "Work on one subtask undoes or conflicts with another",
             Detectability.LATENT, ip_stage="action"),
    CaftType("6.2", "coordination", "delegation_failure",
             "Delegation Failure",
             "Agent fails to properly delegate or coordinate with sub-agents",
             Detectability.LATENT, ip_stage="action"),
    CaftType("6.3", "coordination", "scope_creep",
             "Scope Creep",
             "Agent expands beyond the requested task scope",
             Detectability.LATENT, ip_stage="action"),
    CaftType("6.4", "coordination", "reasoning_action_mismatch",
             "Reasoning-Action Mismatch",
             "Agent's stated plan contradicts its next action",
             Detectability.OBSERVABLE,
             detector_names=("reasoning_action_mismatch",), ip_stage="decision_making"),
)

# ── Category 7: Communication Failures ───────────────────────────
_register(
    CaftType("7.1", "communication", "incomplete_output",
             "Incomplete Output",
             "Agent delivers partial results without indicating incompleteness",
             Detectability.LATENT, ip_stage="feedback"),
    CaftType("7.2", "communication", "wrong_audience",
             "Wrong Audience",
             "Agent produces output at wrong technical level for the user",
             Detectability.LATENT, ip_stage="feedback"),
    CaftType("7.3", "communication", "missing_explanation",
             "Missing Explanation",
             "Agent makes changes without explaining rationale",
             Detectability.LATENT, ip_stage="feedback"),
    CaftType("7.4", "communication", "hallucinated_status",
             "Hallucinated Status",
             "Agent claims success/completion when task is not actually done",
             Detectability.LATENT, ip_stage="feedback"),
)

# ── Category 8: Meta-Cognitive Failures ──────────────────────────
_register(
    CaftType("8.1", "metacognition", "capability_mismatch",
             "Capability Mismatch",
             "Agent attempts a task beyond its capabilities without recognizing limits",
             Detectability.LATENT, ip_stage="decision_making"),
    CaftType("8.2", "metacognition", "incomplete_model",
             "Incomplete Model",
             "Agent operates with incorrect assumptions about the environment",
             Detectability.LATENT, ip_stage="decision_making"),
    CaftType("8.3", "metacognition", "monitoring_failure",
             "Monitoring Failure",
             "Agent fails to track its own progress or detect its own errors",
             Detectability.LATENT, ip_stage="decision_making"),
    CaftType("8.4", "metacognition", "strategy_fixation",
             "Strategy Fixation",
             "Agent persists with a failing strategy instead of adapting",
             Detectability.OBSERVABLE,
             detector_names=("loop", "thrash"), ip_stage="decision_making"),
)


# ──────────────────────────────────────────────────────────────────
# Unified detector → CAFT mapping
# ──────────────────────────────────────────────────────────────────

# Maps original batch detector names to their primary CAFT codes
BATCH_DETECTOR_TO_CAFT: dict[str, str] = {
    "loop":               "2.2",  # Step Repetition (also contributes to 8.4)
    "thrash":             "4.1",  # Tool Misuse (also contributes to 8.4)
    "stall":              "4.4",  # Resource Exhaustion
    "drift":              "2.4",  # Goal Drift
    "cascade":            "4.2",  # Error Cascade
    "token_explosion":    "4.4",  # Resource Exhaustion
    "dead_end":           "3.4",  # Analysis Paralysis
    "recovery_failure":   "4.3",  # Recovery Failure
}

# Maps original failure_type strings to CAFT codes
FAILURE_TYPE_TO_CAFT: dict[str, str] = {
    "LOOP":               "2.2",
    "TOOL_THRASH":        "4.1",
    "STALL":              "4.4",
    "DRIFT":              "2.4",
    "CASCADE":            "4.2",
    "TOKEN_EXPLOSION":    "4.4",
    "DEAD_END":           "3.4",
    "RECOVERY_FAILURE":   "4.3",
}

# Secondary CAFT codes (detectors that provide evidence for multiple types)
BATCH_DETECTOR_SECONDARY_CAFT: dict[str, list[str]] = {
    "loop":  ["8.4"],     # Strategy Fixation
    "thrash": ["8.4"],    # Strategy Fixation
    "stall":  [],
    "drift":  [],
    "cascade": ["4.3"],   # Recovery Failure (cascade implies recovery failure)
    "token_explosion": [],
    "dead_end": [],
    "recovery_failure": [],
}


# ──────────────────────────────────────────────────────────────────
# Query helpers
# ──────────────────────────────────────────────────────────────────

def get_type(code: str) -> CaftType:
    """Get a CAFT type by code. Raises KeyError if not found."""
    return CAFT_TAXONOMY[code]

def get_type_by_name(name: str) -> Optional[CaftType]:
    """Get a CAFT type by failure name (e.g., 'step_repetition')."""
    for t in CAFT_TAXONOMY.values():
        if t.name == name:
            return t
    return None

def get_observable_types() -> list[CaftType]:
    """Get all observable (auto-detectable) CAFT types."""
    return [t for t in CAFT_TAXONOMY.values()
            if t.detectability == Detectability.OBSERVABLE]

def get_latent_types() -> list[CaftType]:
    """Get all latent (requires inference) CAFT types."""
    return [t for t in CAFT_TAXONOMY.values()
            if t.detectability == Detectability.LATENT]

def get_category_types(category: str) -> list[CaftType]:
    """Get all CAFT types in a category."""
    return [t for t in CAFT_TAXONOMY.values() if t.category == category]

def get_categories() -> list[str]:
    """Get all unique category names in order."""
    seen = []
    for t in CAFT_TAXONOMY.values():
        if t.category not in seen:
            seen.append(t.category)
    return seen

def map_batch_diagnosis(failure_type: str) -> Optional[str]:
    """Map a batch detector failure_type to its primary CAFT code."""
    return FAILURE_TYPE_TO_CAFT.get(failure_type)


# ──────────────────────────────────────────────────────────────────
# IP stage query helpers
# ──────────────────────────────────────────────────────────────────

def get_ip_stage_for_detector(failure_name: str) -> str:
    """Get the IP stage where a detector failure occurs."""
    caft_type = get_type_by_name(failure_name)
    return caft_type.ip_stage if caft_type else "unknown"

def get_detectors_by_ip_stage(stage: str) -> list[CaftType]:
    """Get all CAFT types that fail at a given IP stage."""
    return [t for t in CAFT_TAXONOMY.values() if t.ip_stage == stage]

def get_ip_stage_map() -> dict[str, str]:
    """Return a mapping of CAFT code → IP stage for all taxonomy entries."""
    return {t.code: t.ip_stage for t in CAFT_TAXONOMY.values()}
