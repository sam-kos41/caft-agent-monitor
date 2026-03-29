"""Prompt templates for automated CAFT trace annotation.

Contains the annotation criteria, CAFT taxonomy subset (observable types),
and output format specification used by Claude Code subagents.
"""

ANNOTATION_CRITERIA = {
    "step_repetition": (
        "Only TP if agent repeats SAME operation on SAME target with NO new info. "
        "Reading different files or different offsets = normal."
    ),
    "goal_drift": (
        "Only TP if agent loses track of objective unprompted. "
        "User-directed topic changes = normal lifecycle."
    ),
    "missing_verification": (
        "Only TP if code changes NEEDED testing and agent skipped entirely. "
        "Remote HPC workflows where user tests = normal."
    ),
    "context_loss": (
        "Only TP if agent re-reads because it FORGOT (not file changed, not "
        "verifying edits). System context continuation = infrastructure, "
        "but massive re-reading post-continuation is context loss."
    ),
    "premature_termination": (
        "Only TP if agent stopped before completing user's request."
    ),
    "reasoning_action_mismatch": (
        "Only TP if agent's stated reasoning contradicts its actions."
    ),
    "error_cascade": (
        "Only TP if a single error propagates through subsequent operations "
        "due to agent not adapting. Environmental cascades (SSHFS, sibling "
        "tool calls) are normal unless agent handles them poorly."
    ),
    "recovery_failure": (
        "Only TP if agent fails to recover after encountering an error "
        "despite viable alternatives."
    ),
    "analysis_paralysis": (
        "Only TP if agent gets stuck in extended reasoning/reading without "
        "producing action output."
    ),
}

OBSERVABLE_CAFT_CODES = {
    "2.1": {"name": "context_loss", "label": "Context Loss",
            "desc": "Agent re-reads resources it already processed"},
    "2.2": {"name": "step_repetition", "label": "Step Repetition",
            "desc": "Agent repeats the same operation multiple times consecutively"},
    "2.4": {"name": "goal_drift", "label": "Goal Drift",
            "desc": "Agent's actions diverge from the original objective over time"},
    "3.4": {"name": "analysis_paralysis", "label": "Analysis Paralysis",
            "desc": "Agent gets stuck in extended reasoning without taking action"},
    "4.1": {"name": "tool_misuse", "label": "Tool Misuse",
            "desc": "Agent uses a tool with incorrect parameters or in wrong context"},
    "4.2": {"name": "error_cascade", "label": "Error Cascade",
            "desc": "A single error propagates through subsequent operations"},
    "4.3": {"name": "recovery_failure", "label": "Recovery Failure",
            "desc": "Agent fails to recover after encountering an error"},
    "4.4": {"name": "resource_exhaustion", "label": "Resource Exhaustion",
            "desc": "Agent consumes excessive tokens, time, or API calls"},
    "5.3": {"name": "missing_verification", "label": "Missing Verification",
            "desc": "Agent completes changes without testing or reviewing results"},
    "5.4": {"name": "premature_termination", "label": "Premature Termination",
            "desc": "Agent delivers results without completing verification phase"},
    "6.4": {"name": "reasoning_action_mismatch", "label": "Reasoning-Action Mismatch",
            "desc": "Agent's stated plan contradicts its next action"},
    "8.4": {"name": "strategy_fixation", "label": "Strategy Fixation",
            "desc": "Agent persists with a failing strategy instead of adapting"},
}

OUTPUT_FORMAT = {
    "trace_num": "int (from input)",
    "session_id": "str (first 8 chars, from input)",
    "project": "str (from input)",
    "events": "int (from input)",
    "user_goal": "str: 1-2 sentence summary of what the user asked for",
    "agent_completed": "bool: did the agent finish the user's request?",
    "actual_failures": "list[str]: CAFT names of real failures (empty if clean)",
    "failure_details": [
        {
            "caft_code": "str (e.g. '5.4')",
            "caft_name": "str (e.g. 'premature_termination')",
            "onset_step": "int: approximate event index where failure began",
            "severity": "int 1-5 (1=trivial, 5=catastrophic)",
            "confidence": "int 1-5 (1=uncertain, 5=certain)",
            "rationale": "str: 2-3 sentences explaining WHY this is a real failure",
        }
    ],
    "annotations": "dict: empty {} (for auto-annotated traces)",
}


def build_annotation_prompt(batch_summaries: dict) -> str:
    """Build the full annotation prompt for a Claude Code agent.

    Args:
        batch_summaries: The batch_summaries.json content with traces and instructions.

    Returns:
        Prompt string for a Claude Code subagent.
    """
    traces = batch_summaries.get("traces", [])
    n = len(traces)

    return f"""\
You are annotating {n} Claude Code session traces for CAFT (Cognitive Agent Failure Taxonomy) failures.

## Your Task

For each trace summary below, determine:
1. What the user asked the agent to do (user_goal)
2. Whether the agent completed the task (agent_completed)
3. Whether any REAL failures occurred (actual_failures)
4. Details for each failure (failure_details)

## CRITICAL: Annotation Standards

Most traces are CLEAN (no failures). Only flag failures you are CONFIDENT about.
- A clean trace should have: actual_failures=[], failure_details=[], annotations={{}}
- Environmental issues (SSHFS drops, token limits) are NOT agent failures unless the agent handles them poorly
- User-directed topic changes are NOT goal drift
- Reading different files or different sections is NOT step repetition
- Remote HPC workflows where user tests on cluster = NOT missing verification

## CAFT Criteria (apply strictly)

{_format_criteria()}

## Output Format

Return a JSON array of annotations. Each element must have these fields:
{_format_output_spec()}

Return ONLY the JSON array, no markdown fences, no explanation."""


def _format_criteria() -> str:
    lines = []
    for name, criterion in ANNOTATION_CRITERIA.items():
        lines.append(f"- **{name}**: {criterion}")
    return "\n".join(lines)


def _format_output_spec() -> str:
    import json
    return json.dumps(OUTPUT_FORMAT, indent=2)
