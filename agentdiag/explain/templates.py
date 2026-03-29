"""Rule-based explanation templates for each failure mode (free tier)."""

from __future__ import annotations

from dataclasses import dataclass, field

from agentdiag.models import Diagnosis


@dataclass
class ExplanationResult:
    """Structured explanation with cause analysis and remediation steps."""
    failure_type: str
    description: str
    likely_cause: str
    remediation: list[str]

    def to_dict(self) -> dict:
        return {
            "failure_type": self.failure_type,
            "description": self.description,
            "likely_cause": self.likely_cause,
            "remediation": self.remediation,
        }


TEMPLATES: dict[str, dict] = {
    "LOOP": {
        "description": "Agent repeated the pattern {repeated_pattern} {pattern_count} times across {total_tool_calls} tool calls.",
        "likely_cause": "The agent lacks a termination condition or exit strategy for repeated actions.",
        "remediation": [
            "Add max-iteration guardrails to the agent loop",
            "Inject 'stop if no progress after N attempts' instruction",
            "Provide alternative tools for the stuck task",
            "Add state-change detection to break identical-output loops",
        ],
    },
    "TOOL_THRASH": {
        "description": "Agent rapidly switched between {unique_tools_in_window} in a window starting at step {window_start_step}.",
        "likely_cause": "The agent is indecisive about which tool to use, or lacks clear criteria for tool selection.",
        "remediation": [
            "Clarify tool selection criteria in the system prompt",
            "Reduce the number of available tools to minimize confusion",
            "Add a planning step before tool execution",
            "Implement tool-use history awareness to avoid redundant switches",
        ],
    },
    "STALL": {
        "description": "Agent stalled {stall_count} times. Worst at step {worst_step} ({max_latency_ms:.0f}ms vs median {median_latency_ms:.0f}ms).",
        "likely_cause": "External API timeouts, resource contention, or the agent waiting on unavailable dependencies.",
        "remediation": [
            "Add timeout limits to tool calls and API requests",
            "Implement retry-with-backoff for slow operations",
            "Add fallback tools for when primary tools are unresponsive",
            "Monitor and alert on latency thresholds",
        ],
    },
    "DRIFT": {
        "description": "Agent behavior shifted: tool distribution divergence {tool_distribution_shift:.3f}, error rate {error_rate_first_half:.1%} -> {error_rate_second_half:.1%}.",
        "likely_cause": "The agent gradually lost focus on the original goal, possibly due to error accumulation or context window limitations.",
        "remediation": [
            "Periodically re-inject the original goal into the context",
            "Add goal-alignment checks between steps",
            "Implement context summarization to prevent drift from long histories",
            "Set guardrails on tool distribution changes",
        ],
    },
    "CASCADE": {
        "description": "Error cascade of {longest_error_chain} consecutive failures from step {chain_start_step} to {chain_end_step}.",
        "likely_cause": "An initial error propagated downstream because the agent lacked error isolation or recovery logic.",
        "remediation": [
            "Add error isolation: catch and handle errors per-step",
            "Implement circuit breakers to stop cascading failures",
            "Add recovery strategies after the first error in a chain",
            "Log and surface the root-cause error separately from downstream effects",
        ],
    },
    "TOKEN_EXPLOSION": {
        "description": "Token usage grew {growth_ratio_last_vs_first_quarter:.1f}x from first to last quarter ({first_quarter_avg_tokens:.0f} -> {last_quarter_avg_tokens:.0f} tokens/step).",
        "likely_cause": "The agent is accumulating context without summarization, or generating increasingly verbose outputs.",
        "remediation": [
            "Implement context window management with summarization",
            "Add token budget limits per step and per conversation",
            "Prune irrelevant tool outputs from the conversation history",
            "Use streaming to detect and abort runaway token generation",
        ],
    },
    "DEAD_END": {
        "description": "Agent entered a dead end: {max_consecutive_reasoning} consecutive reasoning steps starting at step {dead_end_start_step}.",
        "likely_cause": "The agent is overthinking without executing, possibly stuck in analysis paralysis or lacking actionable tools.",
        "remediation": [
            "Add a max-reasoning-steps limit before requiring a tool call",
            "Inject 'take action now' prompts after N consecutive reasoning steps",
            "Ensure the agent has appropriate tools available for the current task",
            "Add a 'request help' escape hatch for when the agent is stuck",
        ],
    },
    "RECOVERY_FAILURE": {
        "description": "Agent failed to recover from errors {failed_recoveries} out of {total_errors} times ({recovery_failure_rate:.0%} failure rate).",
        "likely_cause": "The agent retries the same failing approach without changing strategy.",
        "remediation": [
            "Implement strategy rotation: try a different approach after N failures",
            "Add error analysis before retry to identify the root cause",
            "Provide fallback tools and alternative paths",
            "Add a 'give up gracefully' option with a clear error message to the user",
        ],
    },
}


def explain(diagnosis: Diagnosis) -> ExplanationResult:
    """Generate an explanation with remediation steps from a diagnosis.

    Uses template-based explanations (free tier). For LLM-powered
    context-aware explanations, see agentdiag.explain.llm.
    """
    template = TEMPLATES.get(diagnosis.failure_type)
    if template is None:
        return ExplanationResult(
            failure_type=diagnosis.failure_type,
            description=diagnosis.explanation,
            likely_cause="Unknown failure type — no template available.",
            remediation=["Investigate the failure mode manually."],
        )

    try:
        description = template["description"].format(**diagnosis.evidence)
    except (KeyError, ValueError):
        description = diagnosis.explanation

    return ExplanationResult(
        failure_type=diagnosis.failure_type,
        description=description,
        likely_cause=template["likely_cause"],
        remediation=template["remediation"],
    )
