"""Semantic confirmation layer for CAFT detectors.

Rule-based detectors are fast candidate generators (~67% precision ceiling).
This module adds LLM-based semantic confirmation to disambiguate real failures
from normal workflow patterns.

Pipeline:
    detector fires → candidate → confirm_diagnosis() → confirmed | rejected | uncertain

Supports LLM providers via AGENTDIAG_LLM_PROVIDER env var:
    - "claude_cli" (default): Claude Code CLI (`claude -p`), uses existing subscription
    - "ollama": Local models via ollama REST API

Graceful degradation: if no LLM is configured or the call fails, returns
"uncertain" with the original confidence (never crashes, never blocks).

V2 prompt design (2026-03-17):
    - Lead with failure definition, not normalcy bias
    - Detector-specific criteria templates with concrete evidence
    - Structured 3-criterion decision: match + explanation + senior engineer test
    - Calibrated toward confirmation (detector already found structural anomaly)
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from agentdiag.models import TraceEvent
from agentdiag.hta import HTAState, Phase
from agentdiag.caft.base import CaftDiagnosis, CaftSeverity
from agentdiag.caft.taxonomy import CAFT_TAXONOMY, get_type_by_name

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM call tracing — captures prompt + response for every LLM call
# ---------------------------------------------------------------------------

_llm_trace_path: Path | None = None


def enable_llm_tracing(path: Path | str) -> None:
    """Enable full prompt+response tracing to a JSONL file.

    Call this before running any LLM confirmations. Each LLM call will
    append a JSON line with: prompt, response, latency_ms, provider, model.
    """
    global _llm_trace_path
    _llm_trace_path = Path(path)


def _write_llm_trace(prompt: str, response: str, latency_ms: float, error: str | None = None) -> None:
    """Append one prompt+response record to the trace file."""
    if _llm_trace_path is None:
        return
    import time as _time
    record = {
        "timestamp": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
        "provider": _get_provider(),
        "model": _get_model(),
        "prompt": prompt,
        "response": response,
        "latency_ms": round(latency_ms, 1),
    }
    if error:
        record["error"] = error
    try:
        with open(_llm_trace_path, "a") as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        logger.warning("Failed to write LLM trace to %s", _llm_trace_path)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ConfirmationResult:
    """Result of semantic confirmation of a CAFT candidate."""
    confirmed: bool
    confidence: float          # 0.0 to 1.0
    reasoning: str             # 1-2 sentence explanation
    status: str                # "confirmed" | "rejected" | "uncertain"

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Detector-specific prompt templates (Task 2)
# ---------------------------------------------------------------------------

DETECTOR_CRITERIA: dict[str, str] = {
    "2.1": (
        "CONTEXT LOSS: The agent re-read a resource it already processed earlier "
        "in the session. Check: (1) Was there a user request to re-read between "
        "the two reads? (2) Did the file change between reads (edit/write in between)? "
        "(3) Is the agent reading a different section or using the content differently? "
        "If NONE of these apply, the agent forgot what it already read."
    ),
    "2.2": (
        "STEP REPETITION: The agent performed the same operation (same tool, same "
        "input) multiple times consecutively. Check: (1) Are the input_hashes "
        "identical across repetitions? (2) Are the outputs different (progressive "
        "work) or identical (stuck in a loop)? (3) Did the agent acknowledge the "
        "repetition or change strategy? If inputs are identical and outputs don't "
        "change, this is genuine repetition."
    ),
    "2.4": (
        "GOAL DRIFT: The agent changed what it's working on without the user "
        "asking for a change. IMPORTANT: Look at the SESSION END section below "
        "to see where the agent ended up — compare to the original goal in the "
        "Session Start section. Check: (1) Is there a user_input event between "
        "the old topic and new topic? If yes → NOT drift (user redirected). "
        "(2) Did the agent discover a blocker that required a detour? (3) Is "
        "the new activity (especially at session end) clearly unrelated to the "
        "original goal? (4) Did the user question whether the original work "
        "was finished? Drift requires topic change WITHOUT user direction. "
        "Focus on the overall trajectory, not just the onset step."
    ),
    "3.1": (
        "TOOL THRASHING: The agent performed many consecutive read-only operations "
        "without producing any output. Check: (1) Are the read operations targeting "
        "different files (exploring) or the same files repeatedly? (2) Is the agent "
        "making progress toward understanding (new files each time) or going in "
        "circles? (3) How many operations happened before any write/edit? "
        "Exploration with progress is normal; reading the same files repeatedly "
        "without action is thrashing."
    ),
    "4.2": (
        "ERROR CASCADE: Multiple errors occurred in sequence. IMPORTANT: Look at "
        "the Session End section to see whether the errors were ultimately resolved. "
        "Check: (1) Did the agent try a DIFFERENT approach after each error, or "
        "repeat the same failing command? (2) Did the errors compound (each error "
        "caused the next)? (3) Did the agent eventually recover by session end? "
        "A few search refinements that succeed is normal; repeating the same "
        "failure without strategy change, especially if unresolved at session end, "
        "is a cascade."
    ),
    "4.3": (
        "RECOVERY FAILURE: The agent encountered errors and failed to recover. "
        "IMPORTANT: Look at the Session End section — did the session end with "
        "errors still unresolved? Check: (1) How many errors occurred vs how "
        "many were resolved? (2) Did the agent try alternative approaches or "
        "just retry the same thing? (3) Are errors still present at session end? "
        "A low recovery rate (< 50%) with repeated retries and unresolved errors "
        "at session end is genuine recovery failure."
    ),
    "4.4": (
        "STALL: The detector uses IQR-based outlier detection on inter-event "
        "latencies within this session. If the detector flagged a stall, the "
        "latency is ALREADY statistically anomalous relative to this session's "
        "baseline — do NOT dismiss it by claiming a tool 'normally' has high "
        "latency. The detector already filters out cold-start (first 5 steps) "
        "and slow tools (Task/TaskOutput/Bash-only sessions).\n\n"
        "Your job is ONLY to check:\n"
        "1. Are there multiple stall events (not just one blip)?\n"
        "2. Did the stalls happen during active work (not user idle time)?\n"
        "3. Is there evidence the agent was stuck (repeated operations, no "
        "progress) rather than doing legitimate heavy work?\n\n"
        "DEFAULT TO CONFIRMED. The statistical detector is reliable. Only "
        "reject if you see clear evidence that the delay was productive work "
        "(e.g., a Bash command running a long build that succeeded)."
    ),
    "5.3": (
        "MISSING VERIFICATION: The session includes code changes but no testing "
        "or verification. IMPORTANT: Look at the Session End section — was any "
        "verification done before the session ended? Check: (1) Did the agent "
        "run ANY test command (pytest, npm test, make test, bash with 'test' in "
        "command) anywhere in the session, especially near the end? (2) Did the "
        "agent delegate testing to a Task subagent? (3) Did the user paste test "
        "results or say 'it works' / 'tests pass'? (4) Did the agent read output "
        "to verify its changes? If NONE of these verification patterns occurred "
        "between code changes and session end, verification is genuinely missing."
    ),
    "5.4": (
        "PREMATURE TERMINATION: The agent ACTIVELY declared completion or moved "
        "to delivery when the task was clearly not done. "
        "CRITICAL DISTINCTION — Premature termination is an AGENT DECISION, not "
        "a session ending: "
        "- PT IS: Agent says 'Done!', 'All changes complete', commits/pushes, "
        "or summarizes results — but deliverables are clearly missing. "
        "- PT IS NOT: Session ended because user stopped responding, context "
        "window filled, ExitPlanMode failed/crashed, or user moved on. These "
        "are session endings, not agent decisions to stop. "
        "STRUCTURED CHECK: "
        "(1) Find the agent's FINAL STATUS CLAIM: Did the agent explicitly "
        "declare the task done, summarize results, or move to delivery? If the "
        "session just... ends (last event is a tool call or reasoning with no "
        "completion claim), that is NOT premature termination. "
        "(2) If the agent DID claim completion: Are all deliverables actually "
        "done? Check Write/Edit for code, Bash for commands, user confirmation. "
        "(3) REJECT if: Session ended due to tool failure (ExitPlanMode FAIL), "
        "user was still actively sending messages at session end, or agent was "
        "mid-work with no completion claim. "
        "CONFIRM only if: Agent made a clear completion claim AND deliverables "
        "are verifiably incomplete. "
        "STRONG EVIDENCE FOR PT: User re-engaging later ('where did we leave off?', "
        "'do you remember what we were doing?') proves the agent abandoned work."
    ),
    "3.5": (
        "STRATEGIC MYOPIA: The agent is stuck in a local optimization loop, "
        "repeatedly applying the same approach without strategic progress. "
        "Strategic Myopia is NOT: "
        "- Normal iterative development (edit -> test -> edit -> test is fine "
        "if metrics improve and scope broadens). "
        "- Debugging a specific issue (concentrated edits are expected when "
        "fixing a bug). "
        "- Running tests after each change (that's good engineering practice). "
        "Strategic Myopia IS: "
        "- Metric barely moves despite many iterations (oscillation, not progress). "
        "- Solution search is narrow (same 2-3 files, same approach each time). "
        "- Agent never steps back to question the approach. "
        "- Ground truth gets modified instead of the system."
    ),
    "6.4": (
        "REASONING-ACTION MISMATCH: The agent's stated intent doesn't match "
        "its subsequent action. Check: (1) Did the agent say it would do X "
        "but then do Y? (2) Is the mismatch due to a user redirect between "
        "the reasoning and the action? (3) Did the agent adapt its plan "
        "(legitimate) or contradict itself (failure)? A user-caused redirect "
        "is NOT mismatch."
    ),
}

# Fallback for any detector type not in the dict above
_DEFAULT_CRITERIA = (
    "Check: (1) Does the observed behavior match the failure definition above? "
    "(2) Is there a specific, concrete reason this behavior is legitimate? "
    "(3) Would a senior engineer reviewing this trace flag it as problematic?"
)


# ---------------------------------------------------------------------------
# Few-shot examples from train split ground truth
# Each entry has a "confirmed" and "rejected" example so the LLM sees
# what both verdicts look like for this detector type.
# Source: annotations/ablation_ready/ (train split ONLY)
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES: dict[str, dict] = {
    "4.4": {  # stall — 7 TP in train, 0 FP
        "confirmed": {
            "evidence": '{"stall_steps": [7], "max_latency_ms": 25085.0, "median_latency_ms": 0.0, '
                        '"threshold_ms": 3000.0, "stall_count": 1, "stall_fraction": 0.1}',
            "events": "step 5: tool_call Read OK 12ms | step 6: tool_call Bash OK 45ms | "
                      "step 7: tool_call Task OK 1ms | step 8: tool_call TaskOutput OK 25085ms <<<ONSET | "
                      "step 9: tool_call Read OK 3ms | step 10: tool_call Write OK 8ms",
            "reasoning": "MATCH: Yes — 25085ms latency at step 8 is 8x the 3000ms threshold, "
                         "median is 0ms so this is a clear outlier. EXPLANATION: No — no large file "
                         "processing or expected computation at this step; TaskOutput should return "
                         "quickly. ENGINEER: Yes, a 25-second hang on TaskOutput indicates a blocked "
                         "subprocess. -> CONFIRMED (confidence: 0.85)",
        },
        "rejected": {
            "evidence": '{"stall_steps": [4], "stall_tool_names": ["Task"], "worst_tool": "Task", '
                        '"max_latency_ms": 64893.0, "median_latency_ms": 450.0, '
                        '"threshold_ms": 5000.0, "stall_count": 1, "stall_fraction": 0.10}',
            "events": "step 2: tool_call Read OK 25ms | step 3: tool_call Read OK 18ms | "
                      "step 4: tool_call Task OK 64893ms 'Find master table code' <<<ONSET | "
                      "step 5: tool_call TaskOutput OK 120ms | step 6: tool_call Grep OK 450ms | "
                      "step 7: tool_call Read OK 22ms",
            "reasoning": "MATCH: Superficially — 64893ms is 13x the 5000ms threshold. "
                         "EXPLANATION: Yes — stall_tool_names=['Task'] shows this was a subagent "
                         "spawn. Task tool calls delegate work to a subprocess that does its own "
                         "reading/searching/reasoning; 30-120 seconds is completely normal for Task. "
                         "Subsequent operations (Grep 450ms, Read 22ms) show no degradation. "
                         "ENGINEER: No, a 65-second Task call is expected behavior, not a stall. "
                         "-> REJECTED (confidence: 0.95)",
        },
    },
    "4.2": {  # error_cascade — 1 TP in train
        "confirmed": {
            "evidence": '{"longest_error_chain": 2, "chain_start_step": 4, "chain_end_step": 5, '
                        '"total_errors": 2, "tools_in_chain": ["Read", "Read"]}',
            "events": "step 3: tool_call Bash OK 120ms | step 4: tool_call Read FAIL 45ms | "
                      "step 5: tool_call Read FAIL 38ms <<<ONSET | step 6: tool_call Read OK 22ms | "
                      "step 7: tool_call Bash FAIL 80ms",
            "reasoning": "MATCH: Yes — 2 consecutive Read failures at steps 4-5, plus another "
                         "failure at step 7 shows errors propagating. EXPLANATION: No — the agent "
                         "retried the same tool (Read) and hit the same error, then moved to Bash "
                         "which also failed. No strategy change between attempts. ENGINEER: Yes, "
                         "cascading file read failures indicate a systemic issue the agent didn't "
                         "address. -> CONFIRMED (confidence: 0.80)",
        },
        "rejected": {
            "evidence": '{"longest_error_chain": 2, "chain_start_step": 8, "chain_end_step": 9, '
                        '"total_errors": 2, "tools_in_chain": ["Grep", "Grep"]}',
            "events": "step 7: tool_call Read OK 25ms | step 8: tool_call Grep FAIL 30ms | "
                      "step 9: tool_call Grep FAIL 35ms <<<ONSET | step 10: tool_call Grep OK 28ms | "
                      "step 11: tool_call Read OK 20ms | step 12: reasoning 'Now I can see the problem'",
            "reasoning": "MATCH: Superficially — 2 consecutive Grep failures. EXPLANATION: Yes — "
                         "the agent tried different search patterns (different input_hashes at steps "
                         "8 and 9), succeeded on the third attempt at step 10, and immediately "
                         "continued productive work. This is normal search refinement. ENGINEER: No, "
                         "trying 2 search patterns before finding the right one is expected. "
                         "-> REJECTED (confidence: 0.88)",
        },
    },
    "5.4": {  # premature_termination — 0 TP in train, 7 FP, 2 FN
        "confirmed": {
            "evidence": "{}",
            "events": "step 210: tool_call Edit OK 40ms | step 211: tool_call Write OK 35ms | "
                      "step 212: tool_call Read OK 20ms (reading unrelated file) | "
                      "step 214: user_input 'do u remember what we were doing last?' | "
                      "step 215: reasoning 'Let me recall...' | "
                      "step 217: tool_call Read OK 18ms <<<SESSION_END",
            "reasoning": "MATCH: Yes — original goal was 'Fix Templates + Retrain Phase 3' but "
                         "session ended at step 217 with agent reading unrelated files. User had to "
                         "re-engage at step 214 asking 'do u remember what we were doing last?' which "
                         "PROVES the agent didn't complete the goal. "
                         "ENGINEER: Yes, user needing to remind the agent of its goal means the work "
                         "was not completed. -> CONFIRMED (confidence: 0.80)",
        },
        "rejected": {
            "evidence": "{}",
            "events": "step 195: tool_call Edit OK 30ms | step 197: tool_call Write OK 25ms | "
                      "step 199: tool_call Read OK 18ms (reviewing changes) | "
                      "step 201: tool_call ExitPlanMode FAIL 50ms | "
                      "step 202: user_input 'what do i need to do next?' | "
                      "step 203: tool_call Read OK 20ms <<<SESSION_END",
            "reasoning": "MATCH: Superficially — session ended without formal delivery. "
                         "EXPLANATION: The session ended because ExitPlanMode FAILED at step 201 "
                         "(tool crash), not because the agent chose to stop. The agent did NOT "
                         "declare the task complete — it was mid-work when the tool crashed. "
                         "User asking 'what do I need to do next?' shows they are still engaged, "
                         "not that the agent abandoned work. Session endings due to tool failures "
                         "or user-initiated stops are NOT premature termination. "
                         "ENGINEER: No, a tool crash ending a session is not the agent's decision. "
                         "-> REJECTED (confidence: 0.92)",
        },
    },
    "2.4": {  # goal_drift — 0 TP in train, 5 FP, 1 FN
        "confirmed": {
            "evidence": '{"drift_blocks": 3, "total_blocks": 8, "unprompted_regressions": 5, '
                        '"baseline_tools": ["Read", "Edit", "Bash"]}',
            "events": "step 520: tool_call Edit OK 40ms (editing validation script) | "
                      "step 521: reasoning 'Now let me create a comprehension validation plan' | "
                      "step 522: tool_call Write OK 35ms (writing new plan document) | "
                      "step 523: tool_call ExitPlanMode OK 10ms | step 524: user_input 'no, rejected' | "
                      "step 525: reasoning <<<ONSET",
            "reasoning": "MATCH: Yes — agent finished its assigned 5 changes then started creating "
                         "a new comprehension validation plan (step 522) that was never requested. "
                         "User rejected at step 524. EXPLANATION: No — no user message asked for "
                         "comprehension validation. Agent invented a new task unprompted. ENGINEER: "
                         "Yes, starting unrequested work is clear goal drift. -> CONFIRMED (confidence: 0.85)",
        },
        "rejected": {
            "evidence": '{"drift_blocks": 2, "total_blocks": 19, "unprompted_regressions": 8, '
                        '"baseline_tools": ["Read", "Task"]}',
            "events": "step 95: tool_call Task OK 5ms | step 96: tool_call TaskOutput OK 80ms | "
                      "step 97: tool_call Read OK 20ms | step 98: tool_call Grep OK 15ms | "
                      "step 99: tool_call Edit OK 30ms | step 100: tool_call Bash OK 250ms | "
                      "step 101: tool_call Read OK 18ms | step 102: reasoning <<<ONSET",
            "reasoning": "MATCH: Superficially — 2 blocks with novel tools. EXPLANATION: Yes — "
                         "agent expanded from Read/Task to Grep/Edit/Bash as it moved from planning "
                         "to implementation. Using new tools in execution is normal progression, not "
                         "drift. The real failure was stall (4.4) at step 3, not goal drift. "
                         "ENGINEER: No, transitioning from reading to editing is expected workflow. "
                         "-> REJECTED (confidence: 0.88)",
        },
    },
    "5.3": {  # missing_verification — 0 TP in train, 3 FP
        "confirmed": {
            "evidence": '{"execution_events": 20, "verification_events": 0, '
                        '"delegated_verification": false, "total_events": 45}',
            "events": "step 38: tool_call Edit OK 35ms | step 39: tool_call Write OK 40ms | "
                      "step 40: tool_call Edit OK 30ms | step 41: reasoning 'Changes complete' | "
                      "step 42: tool_call git_commit OK 120ms | step 43: tool_call git_push OK 200ms | "
                      "step 44: reasoning 'Done! All changes pushed.' | step 45: end_of_session <<<ONSET",
            "reasoning": "MATCH: Yes — 20 execution events with code writes (Edit/Write at 38-40), "
                         "then git commit+push (42-43) without ANY test run, verification read, or "
                         "user confirmation. EXPLANATION: No — no pytest, no bash test, no Task "
                         "delegation, no user saying 'it works'. Agent committed and pushed untested "
                         "code. ENGINEER: Yes, pushing code without running tests is a clear failure. "
                         "-> CONFIRMED (confidence: 0.88)",
        },
        "rejected": {
            "evidence": '{"execution_events": 15, "verification_events": 0, '
                        '"delegated_verification": false, "total_events": 85}',
            "events": "step 78: tool_call Edit OK 25ms | step 79: tool_call Write OK 30ms | "
                      "step 80: tool_call Bash OK 3500ms 'python -m pytest tests/' | "
                      "step 81: tool_call Read OK 15ms | step 82: reasoning 'Tests pass' | "
                      "step 83: tool_call Edit OK 20ms | step 84: tool_call Bash OK 1200ms | "
                      "step 85: end_of_session <<<ONSET",
            "reasoning": "MATCH: Superficially — verification_events=0 in evidence. EXPLANATION: "
                         "Yes — agent ran pytest at step 80 (3500ms execution, not flagged as "
                         "verification by HTA but clearly a test run), read results at step 81, "
                         "and confirmed 'Tests pass'. The real failure was stall (4.4), not missing "
                         "verification. ENGINEER: No, tests were run and passed. "
                         "-> REJECTED (confidence: 0.92)",
        },
    },
    "4.3": {  # recovery_failure — 0 TP in train, 3 FP
        "confirmed": {
            "evidence": '{"total_errors": 5, "failed_recoveries": 3, "recovery_failure_rate": 0.75, '
                        '"same_tool_retries": 4, "worst_error_step": 15}',
            "events": "step 12: tool_call Bash FAIL 200ms 'npm install' | "
                      "step 13: tool_call Bash FAIL 180ms 'npm install' | "
                      "step 14: tool_call Bash FAIL 190ms 'npm install --force' | "
                      "step 15: tool_call Bash FAIL 195ms 'npm install --legacy-peer-deps' <<<ONSET | "
                      "step 16: reasoning 'Still failing' | step 17: tool_call Read OK 20ms",
            "reasoning": "MATCH: Yes — 4 consecutive failures with 75% recovery failure rate. "
                         "Agent retried npm install 4 times with minor flag changes but never "
                         "investigated the root cause. EXPLANATION: No — retrying with --force and "
                         "--legacy-peer-deps shows variation but not real strategy change (never "
                         "checked the error message or fixed the dependency). ENGINEER: Yes, 4 "
                         "retries without reading the error output is poor recovery. "
                         "-> CONFIRMED (confidence: 0.80)",
        },
        "rejected": {
            "evidence": '{"total_errors": 2, "failed_recoveries": 1, "recovery_failure_rate": 1.0, '
                        '"same_tool_retries": 1, "worst_error_step": 80}',
            "events": "step 78: tool_call Bash OK 300ms | step 79: tool_call Read OK 15ms | "
                      "step 80: tool_call Bash FAIL 250ms <<<ONSET | step 81: tool_call Read OK 20ms | "
                      "step 82: reasoning 'Permission denied, let me try differently' | "
                      "step 83: tool_call Bash OK 180ms",
            "reasoning": "MATCH: Superficially — 100% failure rate. EXPLANATION: Yes — only 1 "
                         "failed recovery out of 2 total errors. Agent read the error (step 81), "
                         "reasoned about it (step 82), and succeeded on the next attempt (step 83). "
                         "recovery_failure_rate=1.0 is misleading with only 1 sample. The real "
                         "failure was stall (4.4). ENGINEER: No, one failure followed by successful "
                         "adaptation is normal debugging. -> REJECTED (confidence: 0.90)",
        },
    },
    "3.1": {  # tool_thrashing — 0 TP in train, 1 FP
        "confirmed": {
            "evidence": '{"consecutive_read_only": 25, "threshold": 15, "phase": "executing"}',
            "events": "step 40: tool_call Read OK 15ms hash=abc123 | "
                      "step 41: tool_call Read OK 18ms hash=abc123 | "
                      "step 42: tool_call Grep OK 20ms hash=def456 | "
                      "step 43: tool_call Read OK 12ms hash=abc123 | "
                      "step 44: tool_call Grep OK 22ms hash=def456 | "
                      "step 45: tool_call Read OK 14ms hash=abc123 <<<ONSET",
            "reasoning": "MATCH: Yes — 25 consecutive read-only ops in EXECUTING phase (should "
                         "be writing code). Same files re-read repeatedly (hash=abc123 appears 4 "
                         "times in 6 steps). EXPLANATION: No — same input_hashes show the agent is "
                         "re-reading the same files, not exploring new ones. ENGINEER: Yes, reading "
                         "the same file 4+ times without writing anything is paralysis. "
                         "-> CONFIRMED (confidence: 0.82)",
        },
        "rejected": {
            "evidence": '{"consecutive_read_only": 16, "threshold": 15, "phase": "executing"}',
            "events": "step 50: tool_call Read OK 20ms hash=aaa | "
                      "step 51: tool_call Read OK 18ms hash=bbb | "
                      "step 52: tool_call Grep OK 25ms hash=ccc | "
                      "step 53: tool_call Read OK 15ms hash=ddd | "
                      "step 54: tool_call Read OK 22ms hash=eee | "
                      "step 55: tool_call Grep OK 19ms hash=fff | "
                      "step 56: tool_call Write OK 30ms <<<ONSET",
            "reasoning": "MATCH: Barely — 16 read-only ops, just 1 over the 15 threshold. "
                         "EXPLANATION: Yes — every operation targets a DIFFERENT file (unique "
                         "input_hashes aaa through fff), and the agent writes at step 56 "
                         "immediately after. This is exploration before implementation. The real "
                         "failure was stall (4.4). ENGINEER: No, reading 16 different files before "
                         "writing is reasonable for a complex task. -> REJECTED (confidence: 0.92)",
        },
    },
    "6.4": {  # reasoning_action_mismatch — 0 TP in train, 1 FP
        "confirmed": {
            "evidence": '{"reasoning_step": 30, "action_step": 31, '
                        '"planned_intent": "write tests", "actual_tool": "Edit"}',
            "events": "step 29: tool_call Write OK 40ms (wrote src/feature.py) | "
                      "step 30: reasoning 'Now I need to write tests for the new feature' | "
                      "step 31: tool_call Edit OK 35ms (editing src/feature.py again) <<<ONSET | "
                      "step 32: reasoning 'Let me also update the config'",
            "reasoning": "MATCH: Yes — agent said 'write tests' at step 30 but edited the "
                         "source file instead at step 31, not a test file. EXPLANATION: No — no "
                         "user redirect between steps 30-31, and editing src/feature.py is not "
                         "writing tests. ENGINEER: Yes, saying you'll write tests then editing "
                         "source code is a clear intent-action mismatch. -> CONFIRMED (confidence: 0.78)",
        },
        "rejected": {
            "evidence": '{"reasoning_step": 154, "action_step": 155, '
                        '"planned_intent": "read/review", "actual_tool": "Write"}',
            "events": "step 153: tool_call Read OK 20ms | "
                      "step 154: reasoning 'Let me review the current state and make changes' | "
                      "step 155: tool_call Write OK 35ms <<<ONSET | "
                      "step 156: tool_call Read OK 18ms",
            "reasoning": "MATCH: Superficially — planned 'read/review' but did Write. "
                         "EXPLANATION: Yes — 'review and make changes' includes both reading and "
                         "writing. The keyword detector matched 'review' but the full reasoning "
                         "says 'review...and make changes', which Write fulfills. ENGINEER: No, "
                         "'review and make changes' naturally leads to a Write. "
                         "-> REJECTED (confidence: 0.92)",
        },
    },
    "2.1": {  # context_loss — 0 TP in train, 0 FP, 2 FN
        "confirmed": {
            "evidence": '{"file": "config.yaml", "first_read_step": 3, "second_read_step": 18, '
                        '"intervening_ops": 14, "output_hash_changed": false}',
            "events": "step 3: tool_call Read OK 20ms hash=xyz (config.yaml) | "
                      "step 4-16: various tool_calls (Grep, Edit, Bash) | "
                      "step 17: reasoning 'Let me check the config' | "
                      "step 18: tool_call Read OK 18ms hash=xyz (config.yaml) <<<ONSET",
            "reasoning": "MATCH: Yes — re-read config.yaml at steps 3 and 18 with identical "
                         "output_hash (file unchanged). 14 intervening operations. EXPLANATION: "
                         "No — no user asked to re-check, no edit to config.yaml between reads, "
                         "agent said 'let me check the config' suggesting it forgot it already "
                         "read it. ENGINEER: Yes, re-reading an unchanged file after 14 ops with "
                         "'let me check' language indicates forgotten context. "
                         "-> CONFIRMED (confidence: 0.82)",
        },
        "rejected": {
            "evidence": '{"file": "src/app.py", "first_read_step": 5, "second_read_step": 15, '
                        '"intervening_ops": 9, "output_hash_changed": true}',
            "events": "step 5: tool_call Read OK 25ms hash=aaa (src/app.py) | "
                      "step 8: tool_call Edit OK 30ms (src/app.py) | "
                      "step 12: tool_call Edit OK 28ms (src/app.py) | "
                      "step 15: tool_call Read OK 22ms hash=bbb (src/app.py) <<<ONSET",
            "reasoning": "MATCH: Superficially — re-read with gap. EXPLANATION: Yes — agent "
                         "EDITED src/app.py at steps 8 and 12 (output_hash changed from aaa to "
                         "bbb). Re-reading after editing to verify changes is standard practice. "
                         "ENGINEER: No, verifying your own edits is good practice. "
                         "-> REJECTED (confidence: 0.92)",
        },
    },
    "2.2": {  # step_repetition
        "confirmed": {
            "evidence": '{"consecutive_count": 10, "tool": "read_file", '
                        '"unique_input_hashes": 1, "unique_output_hashes": 1}',
            "events": "step 4: tool_call Read OK 18ms hash=abc123 | "
                      "step 5: tool_call Read OK 20ms hash=abc123 | "
                      "step 6: tool_call Read OK 19ms hash=abc123 | "
                      "step 7: tool_call Read OK 17ms hash=abc123 | "
                      "step 8-13: tool_call Read OK ~18ms hash=abc123 <<<ONSET",
            "reasoning": "MATCH: Yes — 10 consecutive Read calls with identical input_hash "
                         "(abc123) and identical output_hash. Agent got the same result every "
                         "time. EXPLANATION: No — same input AND output means file didn't change "
                         "and agent didn't learn anything new between reads. ENGINEER: Yes, "
                         "reading the same unchanged file 10 times is a stuck loop. "
                         "-> CONFIRMED (confidence: 0.90)",
        },
        "rejected": {
            "evidence": '{"consecutive_count": 8, "tool": "read_file", '
                        '"unique_input_hashes": 8, "unique_output_hashes": 8}',
            "events": "step 5: tool_call Read OK 15ms hash=aaa | "
                      "step 6: tool_call Read OK 18ms hash=bbb | "
                      "step 7: tool_call Read OK 22ms hash=ccc | "
                      "step 8: tool_call Read OK 16ms hash=ddd | "
                      "step 9-12: tool_call Read OK ~17ms hash=eee,fff,ggg,hhh <<<ONSET",
            "reasoning": "MATCH: Superficially — 8 consecutive reads. EXPLANATION: Yes — "
                         "each read targets a DIFFERENT file (8 unique input_hashes, 8 unique "
                         "output_hashes). Agent is exploring a directory, not stuck in a loop. "
                         "ENGINEER: No, reading 8 different files is normal exploration. "
                         "-> REJECTED (confidence: 0.95)",
        },
    },
    "3.5": {  # strategic_myopia — V1 experimental
        "confirmed": {
            "evidence": '{"eval_cycles": 6, "eval_script": "python scripts/run_ablation.py", '
                        '"edit_concentration": 0.93, "top_files": ["detectors.py"], '
                        '"total_edits": 14, "metric_values": [0.73, 0.51, 0.52, 0.75], '
                        '"max_metric_delta": 0.043, "metric_stagnation": true, '
                        '"eval_edit_count": 3, "ground_truth_mutation": true, '
                        '"broad_reads": 0, "architecture_blindness": true, '
                        '"exploration_ratio": 0.02, "phase_stagnation": true}',
            "events": "step 50: tool_call Bash OK 3000ms 'python scripts/run_ablation.py' | "
                      "step 52: tool_call Edit OK 30ms (detectors.py:1110) | "
                      "step 54: tool_call Edit OK 28ms (detectors.py:1135) | "
                      "step 60: tool_call Bash OK 2800ms 'python scripts/run_ablation.py' | "
                      "step 65: tool_call Edit OK 25ms (annotation_ledger.jsonl) | "
                      "step 70: tool_call Bash OK 3100ms 'python scripts/run_ablation.py' | "
                      "step 75: tool_call Edit OK 22ms (detectors.py:1120) <<<ONSET",
            "reasoning": "MATCH: Yes — 6 eval cycles running run_ablation.py, 93% of 14 edits "
                         "target detectors.py (same file, same line range), metrics oscillate "
                         "73%->51%->52%->75% (net +2%, within noise). Agent edited annotation "
                         "files 3 times and never read architecture docs. Classic optimization "
                         "tunnel: hill-climbing on F1 by tuning thresholds in one file. "
                         "ENGINEER: Yes, oscillating metrics with concentrated edits and "
                         "ground truth mutation is textbook strategic myopia. "
                         "-> CONFIRMED (confidence: 0.80)",
        },
        "rejected": {
            "evidence": '{"eval_cycles": 5, "eval_script": "python -m pytest tests/", '
                        '"edit_concentration": 0.65, "top_files": ["auth.py", "test_auth.py", "routes.py"], '
                        '"total_edits": 12, "metric_values": [0.45, 0.78, 0.92], '
                        '"max_metric_delta": 0.511, "metric_stagnation": false, '
                        '"eval_edit_count": 0, "ground_truth_mutation": false, '
                        '"broad_reads": 3, "architecture_blindness": false, '
                        '"exploration_ratio": 0.15, "phase_stagnation": false}',
            "events": "step 10: tool_call Read OK 20ms (README.md) | "
                      "step 15: tool_call Edit OK 35ms (auth.py) | "
                      "step 20: tool_call Bash OK 1500ms 'python -m pytest tests/' | "
                      "step 30: tool_call Edit OK 30ms (routes.py) | "
                      "step 40: tool_call Bash OK 1200ms 'python -m pytest tests/' | "
                      "step 50: tool_call Edit OK 28ms (test_auth.py) <<<ONSET",
            "reasoning": "MATCH: Superficially — 5 eval cycles with concentrated edits. "
                         "EXPLANATION: Yes — metrics show genuine improvement 45%->78%->92% "
                         "(not oscillation). Edits spread across 3+ files including tests. "
                         "Agent read README (broad read). Exploration ratio 15% shows the "
                         "agent revisited GATHERING/PLANNING phases. This is iterative TDD "
                         "with real progress, not strategic myopia. "
                         "ENGINEER: No, steady metric improvement with broadening scope is "
                         "good engineering. -> REJECTED (confidence: 0.90)",
        },
    },
}


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

_PHASE_DESCRIPTIONS = {
    Phase.IDLE: "No activity yet — session hasn't started.",
    Phase.GATHERING: "Reading files, searching, exploring the codebase.",
    Phase.PLANNING: "Reasoning about approach, designing a solution.",
    Phase.EXECUTING: "Writing code, editing files, running commands.",
    Phase.VERIFYING: "Running tests, reviewing output, checking results.",
    Phase.DELIVERING: "Committing, pushing, summarizing results.",
}


def _format_event(e: TraceEvent, marker: str = "") -> str:
    """Format a single event as a human-readable log line.

    Target: readable by both humans and LLMs at a glance.
    """
    step = f"step {e.step:>4}"

    if e.type == "user_input":
        text = e.goal_text[:120] if e.goal_text else ""
        return f"  {step}: [user] {text}{marker}"

    if e.type == "planning":
        text = e.goal_text[:120] if e.goal_text else ""
        return f"  {step}: [thinking] {text}{marker}"

    if e.type == "reasoning":
        text = e.goal_text[:120] if e.goal_text else ""
        return f"  {step}: [response] {text}{marker}"

    # tool_call
    tool = e.tool or "unknown"
    success_str = "OK" if e.success else "FAIL"
    latency = f" {e.latency_ms:.0f}ms" if e.latency_ms > 0 else ""
    goal = ""
    if e.goal_text:
        if tool == "Bash":
            goal = f" `{e.goal_text[:100]}`"
        elif tool == "Grep":
            goal = f" {e.goal_text[:100]}"
        elif tool == "Glob":
            goal = f" {e.goal_text[:100]}"
        else:
            goal = f' "{e.goal_text[:100]}"'
    error = ""
    if not e.success and e.error_message:
        error = f" | {e.error_message[:100]}"
    return f"  {step}: {tool}{goal} {success_str}{latency}{error}{marker}"


def _format_event_window(events: list[TraceEvent], center_step: int, window: int = 15) -> str:
    """Format a window of events around the candidate onset step.

    Layout: onset event first (LLMs attend best to start/end of context),
    then context before, then context after. Based on Liu et al. 2023
    "Lost in the Middle" finding that mid-context items get lowest attention.
    """
    start_step = max(1, center_step - window)
    end_step = center_step + window * 2 // 3

    window_events = [
        e for e in events
        if start_step <= e.step <= end_step
    ]

    if not window_events:
        window_events = events[-25:]

    # Split into onset, before, after
    onset_event = None
    before = []
    after = []
    for e in window_events:
        if e.step == center_step:
            onset_event = e
        elif e.step < center_step:
            before.append(e)
        else:
            after.append(e)

    lines = []

    # Onset first — highest attention position
    if onset_event:
        lines.append("ONSET EVENT:")
        lines.append(_format_event(onset_event, " <<<ONSET"))
        lines.append("")

    # Context before
    if before:
        lines.append(f"Context before (steps {before[0].step}-{before[-1].step}):")
        for e in before:
            lines.append(_format_event(e))
        lines.append("")

    # Context after
    if after:
        lines.append(f"Context after (steps {after[0].step}-{after[-1].step}):")
        for e in after:
            lines.append(_format_event(e))

    return "\n".join(lines)


def _format_session_tail(events: list[TraceEvent], n: int = 10) -> str:
    """Format the last N events of the session (how the session ended)."""
    tail = events[-n:] if len(events) > n else events
    lines = []
    for i, e in enumerate(tail):
        marker = " <<<SESSION_END" if i == len(tail) - 1 else ""
        lines.append(_format_event(e, marker))
    return "\n".join(lines)


def _format_session_head(events: list[TraceEvent], n: int = 5) -> str:
    """Format the first N events (goal/setup area)."""
    head = events[:n]
    lines = []
    for e in head:
        lines.append(_format_event(e))
    return "\n".join(lines)


def _extract_agent_goal(events: list[TraceEvent]) -> str:
    """Extract the agent's goal from the first user message."""
    for e in events:
        if e.type == "user_input" and e.goal_text:
            text = e.goal_text.strip()
            if len(text) > 200:
                text = text[:200] + "..."
            return text
    return "(no explicit goal found)"


def _count_user_messages_in_window(events: list[TraceEvent], start: int, end: int) -> int:
    """Count user_input events in a step range."""
    return sum(1 for e in events if start <= e.step <= end and e.type == "user_input")


def _format_similar_cases(cases: list[dict]) -> str:
    """Format similar past cases for the confirmation prompt."""
    if not cases:
        return "No similar past cases available."

    lines = []
    for i, case in enumerate(cases[:3], 1):
        status = case.get("status", "unknown")
        failure = case.get("failure_name", "unknown")
        desc = case.get("description", "")[:100]
        phase = case.get("phase_at_onset", "unknown")
        lines.append(
            f"  Case {i}: {failure} in {phase} phase — {status}. {desc}"
        )

    statuses = [c.get("status", "unknown") for c in cases[:5]]
    fp_count = sum(1 for s in statuses if s == "false_positive")
    confirmed_count = sum(1 for s in statuses if s == "confirmed")
    total = len(statuses)
    if total > 0:
        lines.append(
            f"  Summary: {confirmed_count}/{total} confirmed, "
            f"{fp_count}/{total} false positives in similar cases."
        )

    return "\n".join(lines)


def _get_detector_criteria(caft_code: str, candidate: CaftDiagnosis) -> str:
    """Get detector-specific evaluation criteria."""
    criteria = DETECTOR_CRITERIA.get(caft_code, _DEFAULT_CRITERIA)
    # Fill in evidence values where template uses them
    evidence = candidate.evidence or {}
    try:
        criteria = criteria.format(**evidence)
    except (KeyError, IndexError):
        pass  # Template vars not in evidence dict — use as-is
    return criteria


def _format_few_shot_examples(caft_code: str) -> str:
    """Format few-shot examples for a detector type.

    Returns empty string if no examples exist for this detector.
    """
    examples = FEW_SHOT_EXAMPLES.get(caft_code)
    if not examples:
        return ""

    parts = []

    confirmed = examples.get("confirmed")
    if confirmed:
        parts.append("## Example: CONFIRMED failure")
        parts.append(f"Evidence: {confirmed['evidence']}")
        parts.append(f"Activity: {confirmed['events']}")
        parts.append(f"Analysis: {confirmed['reasoning']}")

    rejected = examples.get("rejected")
    if rejected:
        parts.append("")
        parts.append("## Example: REJECTED candidate (false positive)")
        parts.append(f"Evidence: {rejected['evidence']}")
        parts.append(f"Activity: {rejected['events']}")
        parts.append(f"Analysis: {rejected['reasoning']}")

    parts.append("")
    parts.append("## Now evaluate THIS candidate:")

    return "\n".join(parts)


def build_confirmation_prompt(
    candidate: CaftDiagnosis,
    events: list[TraceEvent],
    hta_state: HTAState,
    similar_cases: list[dict] | None = None,
) -> str:
    """Build the structured confirmation prompt for LLM review.

    V2 design principles:
    - Lead with failure definition (not "is this normal?")
    - Include detector-specific criteria
    - Structured 3-criterion decision framework
    - Calibrate toward confirmation (detector found structural anomaly)
    - Include user message count to help assess redirections
    """
    caft_type = get_type_by_name(candidate.failure_name)
    caft_def = caft_type.description if caft_type else candidate.description

    goal = _extract_agent_goal(events)
    phase = hta_state.current_phase
    phase_desc = _PHASE_DESCRIPTIONS.get(phase, "Unknown phase.")
    event_window = _format_event_window(events, candidate.at_step)
    evidence_str = json.dumps(candidate.evidence, indent=2, default=str)
    cases_str = _format_similar_cases(similar_cases or [])
    detector_criteria = _get_detector_criteria(candidate.caft_code, candidate)
    few_shot_str = _format_few_shot_examples(candidate.caft_code)
    total_events = len(events)
    last_step = events[-1].step if events else 0

    # Count user messages around onset to help LLM assess redirections
    user_msgs_before = _count_user_messages_in_window(
        events, max(1, candidate.at_step - 20), candidate.at_step)
    user_msgs_after = _count_user_messages_in_window(
        events, candidate.at_step, candidate.at_step + 10)

    # For detectors where the session END matters, add tail context
    _END_SENSITIVE = {"premature_termination", "goal_drift", "error_cascade",
                      "recovery_failure", "missing_verification"}
    session_context_section = ""
    if candidate.failure_name in _END_SENSITIVE:
        session_tail = _format_session_tail(events, n=12)
        session_head = _format_session_head(events, n=5)
        session_context_section = f"""
## Session Overview
Total events: {total_events} | Last step: {last_step} | Detector onset: step {candidate.at_step}

### Session Start (goal area):
{session_head}

### Session End (last 12 events):
{session_tail}

IMPORTANT: For {candidate.failure_name}, evaluate whether the session achieved its goal by looking at the SESSION END, not just the onset area.
"""

    # Build few-shot section (only if examples exist for this detector)
    few_shot_section = ""
    if few_shot_str:
        few_shot_section = f"""
## Calibration Examples

The following examples show what CONFIRMED and REJECTED verdicts look like for this detector type. Use them to calibrate your judgment.

{few_shot_str}
"""

    prompt = f"""You are a QA analyst reviewing an AI agent's work session. A detector has flagged a potential failure. Your job is to determine whether the evidence supports the failure diagnosis.

## Context
Agent's goal: {goal}
Current phase: {phase.label} — {phase_desc}
User messages near onset: {user_msgs_before} before, {user_msgs_after} after

## Failure Under Review
Type: {candidate.caft_code} — {candidate.failure_name}
Definition: {caft_def}
Onset step: {candidate.at_step}

## Detector-Specific Criteria
{detector_criteria}
{few_shot_section}
## Detector Evidence
{evidence_str}

## Activity Log (around step {candidate.at_step})
{event_window}
{session_context_section}
## Similar Past Cases
{cases_str}

## Decision Framework

The detector has identified a structural anomaly in the agent's behavior. Your job is NOT to decide whether agent behavior is "normal" — most agent behavior IS normal, but these candidates were flagged because they deviate from baselines.

Evaluate these three criteria:

1. **MATCH**: Does the evidence satisfy the failure definition for {candidate.failure_name}?
   Look at the detector evidence and activity log. Does the pattern match?

2. **EXPLANATION**: Is there a SPECIFIC, CONCRETE legitimate reason for this behavior?
   Acceptable reasons: user explicitly redirected the agent, file was modified between reads, agent was processing different inputs, agent delegated verification to a subagent.
   NOT acceptable: vague claims like "the agent is making progress" or "normal workflow."

3. **ENGINEER TEST**: Would a senior engineer reviewing this trace flag this as a problem?

Decision rules:
- MATCH=yes AND no specific EXPLANATION → confirmed
- MATCH=yes AND specific EXPLANATION exists → uncertain
- MATCH=no → rejected

Default to confirmed unless you identify a specific, concrete counter-explanation.

Respond with ONLY a JSON object:
{{"cited_evidence": ["step N: exact quote from activity log", "key: value from evidence JSON"], "confirmed": true/false, "confidence": 0.0-1.0, "reasoning": "one sentence citing specific evidence"}}

IMPORTANT: cited_evidence must contain 1-3 direct quotes from the activity log or evidence above. If you cannot cite specific evidence, set confirmed=false."""

    return prompt


# ---------------------------------------------------------------------------
# LLM provider abstraction
# ---------------------------------------------------------------------------

def _get_provider() -> str:
    """Get the configured LLM provider.

    Only claude_cli and ollama are supported.  Anthropic/OpenAI API
    providers have been removed — use claude_cli instead (runs under
    your existing Claude Code subscription, no API key needed).
    """
    provider = os.environ.get("AGENTDIAG_LLM_PROVIDER", "claude_cli").lower()
    if provider in ("anthropic", "openai"):
        logger.warning(
            "AGENTDIAG_LLM_PROVIDER=%s is no longer supported. "
            "Falling back to claude_cli (uses Claude Code subscription, no API key).",
            provider,
        )
        return "claude_cli"
    return provider


def _get_model() -> str:
    """Get the configured model name."""
    provider = _get_provider()
    default_models = {
        "claude_cli": "haiku",
        "ollama": "llama3.2",
    }
    return os.environ.get("AGENTDIAG_LLM_MODEL", default_models.get(provider, ""))


async def _call_ollama(prompt: str) -> str:
    """Call local Ollama API."""
    import httpx

    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{base_url}/api/generate",
            json={
                "model": _get_model(),
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1},
            },
        )
        response.raise_for_status()
        return response.json()["response"]


async def _call_claude_cli(prompt: str) -> str:
    """Call Claude Code CLI in print mode (no API key needed).

    Uses `claude -p --model <model>` which runs under the user's existing
    Claude Code subscription. Default model is haiku for speed/cost.
    """
    import asyncio
    import shutil
    import tempfile

    claude_bin = shutil.which("claude")
    if claude_bin is None:
        raise ValueError(
            "claude CLI not found on PATH. Install Claude Code or use a different provider."
        )

    model = _get_model()

    # Write prompt to a temp file to avoid shell escaping issues with large prompts
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(prompt)
        prompt_file = f.name

    try:
        proc = await asyncio.create_subprocess_exec(
            claude_bin, "-p",
            "--model", model,
            "--no-session-persistence",
            "--output-format", "text",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=prompt.encode()),
            timeout=120,
        )
        if proc.returncode != 0:
            err = stderr.decode().strip()
            raise RuntimeError(f"claude CLI failed (rc={proc.returncode}): {err[:500]}")
        return stdout.decode().strip()
    finally:
        import os as _os
        try:
            _os.unlink(prompt_file)
        except OSError:
            pass


_LLM_CALLERS = {
    "claude_cli": _call_claude_cli,
    "ollama": _call_ollama,
}


async def _call_llm(prompt: str) -> str:
    """Call the configured LLM provider, tracing prompt+response if enabled."""
    import time as _time

    provider = _get_provider()
    caller = _LLM_CALLERS.get(provider)
    if caller is None:
        raise ValueError(f"Unknown LLM provider: {provider}")

    t0 = _time.monotonic()
    try:
        response = await caller(prompt)
        ms = (_time.monotonic() - t0) * 1000
        _write_llm_trace(prompt, response, ms)
        return response
    except Exception as exc:
        ms = (_time.monotonic() - t0) * 1000
        _write_llm_trace(prompt, "", ms, error=str(exc))
        raise


def _extract_json_object(text: str) -> dict:
    """Extract the first JSON object from text, handling nested structures.

    Handles markdown fences, surrounding text, and nested arrays/objects
    (e.g. cited_evidence arrays).
    """
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    # Find first '{' and match to its closing '}'
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in response: {text[:200]}")

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])

    # Fallback: try the simple regex for flat objects
    match = re.search(r"\{[^{}]*\}", text)
    if match:
        return json.loads(match.group())
    raise ValueError(f"No JSON object found in response: {text[:200]}")


def _parse_llm_response(text: str) -> ConfirmationResult:
    """Parse LLM response JSON into ConfirmationResult.

    Handles common LLM response quirks:
    - Markdown code fences around JSON
    - Extra text before/after JSON
    - Nested arrays (cited_evidence)
    - Missing fields
    """
    data = _extract_json_object(text)

    confirmed = bool(data.get("confirmed", False))
    confidence = float(data.get("confidence", 0.5))
    confidence = max(0.0, min(1.0, confidence))
    reasoning = str(data.get("reasoning", "No reasoning provided."))

    # Log cited evidence for grounding audit (not stored in result,
    # but visible in debug logs)
    cited = data.get("cited_evidence", [])
    if cited:
        logger.debug("Cited evidence: %s", cited)

    if confirmed:
        status = "confirmed"
    elif confidence < 0.3:
        status = "rejected"
    else:
        status = "uncertain"

    return ConfirmationResult(
        confirmed=confirmed,
        confidence=confidence,
        reasoning=reasoning,
        status=status,
    )


# ---------------------------------------------------------------------------
# Main confirmation function
# ---------------------------------------------------------------------------

# Threshold above which detectors skip LLM confirmation.
# Raised from 0.9 to 0.95 in V2 so more candidates go through LLM
# (now that the prompt is better at distinguishing real failures).
# Configurable via AGENTDIAG_AUTOCONFIRM_THRESHOLD env var.
AUTOCONFIRM_THRESHOLD = float(os.environ.get("AGENTDIAG_AUTOCONFIRM_THRESHOLD", "0.9"))


def is_llm_available() -> bool:
    """LLM confirmation is disabled.

    Ablation V18 showed the LLM hurts performance (F1=29% with LLM vs
    F1=44% without).  All confirmation is now handled by rule-based
    auto-confirm (confidence >= 0.9 threshold).
    """
    return False


async def confirm_diagnosis(
    candidate: CaftDiagnosis,
    events: list[TraceEvent],
    hta_state: HTAState,
    context_cases: list[dict] | None = None,
) -> ConfirmationResult:
    """Semantically confirm a CAFT detector candidate via LLM.

    Args:
        candidate: The CAFT diagnosis to confirm.
        events: Full event list (last 15-20 events around candidate used).
        hta_state: Current HTA state.
        context_cases: Similar past cases from OpenViking.

    Returns:
        ConfirmationResult with confirmed/rejected/uncertain status.
        Returns "uncertain" (not rejected) if LLM call fails.
    """
    prompt = build_confirmation_prompt(
        candidate=candidate,
        events=events,
        hta_state=hta_state,
        similar_cases=context_cases,
    )

    try:
        response_text = await _call_llm(prompt)
        result = _parse_llm_response(response_text)
        logger.info(
            "Confirmation for %s at step %d: %s (confidence=%.2f) — %s",
            candidate.failure_name,
            candidate.at_step,
            result.status,
            result.confidence,
            result.reasoning,
        )
        return result

    except Exception as exc:
        # Graceful degradation: return uncertain, never crash
        logger.debug(
            "LLM confirmation failed for %s: %s",
            candidate.failure_name,
            exc,
        )
        return ConfirmationResult(
            confirmed=False,
            confidence=candidate.confidence * 0.7,
            reasoning=f"LLM confirmation unavailable ({type(exc).__name__}); using rule-based confidence.",
            status="uncertain",
        )


# ---------------------------------------------------------------------------
# Synchronous wrapper for non-async contexts
# ---------------------------------------------------------------------------

def confirm_diagnosis_sync(
    candidate: CaftDiagnosis,
    events: list[TraceEvent],
    hta_state: HTAState,
    context_cases: list[dict] | None = None,
) -> ConfirmationResult:
    """Synchronous wrapper around confirm_diagnosis.

    For use in contexts where async isn't available (CLI, tests).
    Uses a persistent event loop to avoid httpx 'Event loop is closed' warnings.
    """
    import asyncio
    import warnings

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(
                asyncio.run,
                confirm_diagnosis(candidate, events, hta_state, context_cases),
            )
            return future.result(timeout=60)
    else:
        # Use a persistent loop to avoid httpx cleanup warnings
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                confirm_diagnosis(candidate, events, hta_state, context_cases)
            )
        finally:
            # Suppress event-loop-closed warnings from httpx background cleanup
            try:
                _cancel_remaining_tasks(loop)
            except Exception:
                pass
            loop.close()


def _cancel_remaining_tasks(loop):
    """Cancel remaining async tasks to prevent 'Event loop is closed' noise."""
    import asyncio
    pending = asyncio.all_tasks(loop)
    for task in pending:
        task.cancel()
    if pending:
        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))


# ---------------------------------------------------------------------------
# Tier 2: Session-end assessment (PT + GoalDrift via single LLM call)
# ---------------------------------------------------------------------------

TIER_2_FAILURE_TYPES = {"premature_termination", "goal_drift"}


@dataclass
class SessionAssessment:
    """Result of Tier 2 session-end assessment."""
    classifications: list[str]  # e.g., ["B", "C"] = PT + GoalDrift
    premature_termination: bool
    goal_drift: bool
    confidence: float
    reasoning: str
    raw_response: str = ""


def build_session_assessment_prompt(
    events: list[TraceEvent],
    hta_state: HTAState,
) -> str:
    """Build the Tier 2 session-end assessment prompt.

    Single LLM call per session that evaluates completion status.
    Multi-select: can return multiple classifications (e.g., PT + GoalDrift).
    """
    goal = _extract_agent_goal(events)
    session_head = _format_session_head(events, n=8)
    session_tail = _format_session_tail(events, n=15)
    total_events = len(events)

    # Phase counts
    phase_counts = hta_state.phase_event_counts
    phases = ", ".join(
        f"{k}={v}" for k, v in sorted(phase_counts.items()) if v > 0
    )
    exec_count = phase_counts.get("executing", 0)
    verify_count = phase_counts.get("verifying", 0)
    deliver_count = phase_counts.get("delivering", 0)

    prompt = f"""You are reviewing a completed AI coding agent session.

USER'S ORIGINAL REQUEST: {goal}

SESSION START (first 8 events):
{session_head}

SESSION END (last 15 events):
{session_tail}

Total events: {total_events}  |  Phases: {phases}
Exec: {exec_count}  |  Verify: {verify_count}  |  Deliver: {deliver_count}

What happened in this session? Pick exactly ONE primary classification:
A) COMPLETED — agent addressed the user's request (fully or partially with user satisfied)
B) PREMATURE STOP — agent was actively working on deliverables but stopped mid-task
C) GOAL DRIFT — agent spent significant effort on work IT INITIATED that the user didn't request
D) EXTERNAL END — session ended due to crash, context limit, or disconnect
E) USER REDIRECTED — user explicitly changed the task direction mid-session

CRITICAL RULES for B (PREMATURE STOP):
- B means the agent was PRODUCING deliverables (writing code, editing files,
  running builds) and STOPPED before finishing. The user's original request
  must have concrete unfinished deliverables.
- B does NOT apply when:
  * The session is primarily Q&A or planning (user asks questions, agent answers)
  * The user's request was exploratory ("what do I do next?", "how does this work?")
  * The session ends with reasoning/planning events — that's normal session end
  * The user sent the last message and the session ended — that's the user leaving
  * The agent answered the user's question even if it didn't write code
- Most Claude Code sessions end with reasoning or user_input — this is NORMAL,
  not premature. Only choose B if the agent was mid-implementation.

CRITICAL RULES for C (GOAL DRIFT):
- C means the AGENT autonomously started working on something the user never
  asked for. The agent invented a new task on its own.
- C does NOT apply when the user redirected the conversation (that's E).
- C does NOT apply for brief exploration that supports the main task.
- To distinguish C from E: look for a user_input event that changed direction.
  If the user asked for the new topic → E. If the agent just started doing it → C.

Other rules:
- Default to A unless there is STRONG evidence otherwise. Most sessions are A.
- If the user asked a question and the agent answered it, that's A even if
  no code was written.
- D is for crashes, tool failures at the end, or context-limit cutoffs.
- B and C can co-occur only if BOTH clearly apply.

Respond with JSON:
{{"classifications": ["A"], "confidence": 0.85, "reasoning": "..."}}
Use ["B", "C"] if both premature stop AND goal drift clearly apply."""

    return prompt


def _parse_session_assessment(text: str) -> SessionAssessment:
    """Parse LLM response into SessionAssessment.

    Handles markdown fences and extra text around JSON.
    """
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    match = re.search(r"\{[^{}]*\}", text)
    if not match:
        raise ValueError(f"No JSON object found in response: {text[:200]}")

    data = json.loads(match.group())

    classifications = data.get("classifications", [])
    if isinstance(classifications, str):
        classifications = [classifications]
    # Normalize to uppercase single letters
    classifications = [c.strip().upper()[:1] for c in classifications if c.strip()]

    confidence = float(data.get("confidence", 0.5))
    confidence = max(0.0, min(1.0, confidence))
    reasoning = str(data.get("reasoning", "No reasoning provided."))

    return SessionAssessment(
        classifications=classifications,
        premature_termination="B" in classifications,
        goal_drift="C" in classifications,
        confidence=confidence,
        reasoning=reasoning,
        raw_response=text,
    )


async def assess_session_end(
    events: list[TraceEvent],
    hta_state: HTAState,
) -> SessionAssessment:
    """Run Tier 2 session-end assessment via single LLM call.

    Returns SessionAssessment with classifications.
    On error, returns empty classifications (never crashes).
    """
    prompt = build_session_assessment_prompt(events, hta_state)

    try:
        response_text = await _call_llm(prompt)
        result = _parse_session_assessment(response_text)
        logger.info(
            "Session assessment: classifications=%s (confidence=%.2f) — %s",
            result.classifications,
            result.confidence,
            result.reasoning,
        )
        return result

    except Exception as exc:
        logger.debug("Session assessment failed: %s", exc)
        return SessionAssessment(
            classifications=[],
            premature_termination=False,
            goal_drift=False,
            confidence=0.0,
            reasoning=f"Session assessment unavailable ({type(exc).__name__}).",
            raw_response="",
        )


def assess_session_end_sync(
    events: list[TraceEvent],
    hta_state: HTAState,
) -> SessionAssessment:
    """Synchronous wrapper around assess_session_end.

    Same pattern as confirm_diagnosis_sync.
    """
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(
                asyncio.run,
                assess_session_end(events, hta_state),
            )
            return future.result(timeout=60)
    else:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                assess_session_end(events, hta_state)
            )
        finally:
            try:
                _cancel_remaining_tasks(loop)
            except Exception:
                pass
            loop.close()


# ---------------------------------------------------------------------------
# Tier 2 v2: Binary PT and GoalDrift assessments (separate LLM calls)
# ---------------------------------------------------------------------------

@dataclass
class PTAssessment:
    """Result of binary premature termination assessment."""
    premature_termination: bool
    confidence: float
    reasoning: str
    raw_response: str = ""


@dataclass
class GDAssessment:
    """Result of binary goal drift assessment."""
    goal_drift: bool
    confidence: float
    reasoning: str
    raw_response: str = ""


def build_pt_assessment_prompt(
    events: list[TraceEvent],
    hta_state: HTAState,
    context_cases: list[dict] | None = None,
) -> str:
    """Build a binary YES/NO prompt focused solely on premature termination."""
    goal = _extract_agent_goal(events)
    session_head = _format_session_head(events, n=8)
    session_tail = _format_session_tail(events, n=15)
    total_events = len(events)
    cases_str = _format_similar_cases(context_cases or [])

    phase_counts = hta_state.phase_event_counts
    exec_count = phase_counts.get("executing", 0)
    verify_count = phase_counts.get("verifying", 0)
    deliver_count = phase_counts.get("delivering", 0)

    prompt = f"""You are reviewing a completed AI coding agent session for ONE specific question:

**Did the agent actively declare completion while deliverables were clearly incomplete?**

USER'S ORIGINAL REQUEST: {goal}

SESSION START (first 8 events):
{session_head}

SESSION END (last 15 events):
{session_tail}

Total events: {total_events}  |  Exec: {exec_count}  |  Verify: {verify_count}  |  Deliver: {deliver_count}

WHAT IS PREMATURE TERMINATION:
- The agent ACTIVELY declared "Done!", "All changes complete", committed/pushed, or summarized results
- BUT deliverables are clearly missing or incomplete

WHAT IS NOT PREMATURE TERMINATION:
- Session just... ends (last event is a tool call or reasoning with no completion claim)
- Session ended because user stopped responding, context window filled, or tool crashed
- The session is primarily Q&A or planning (no code deliverables expected)
- The user's request was exploratory ("what do I do next?", "how does this work?")
- The user sent the last message and the session ended — that's the user leaving
- ExitPlanMode failed/crashed — that's a tool failure, not agent decision

## Similar Past Cases
{cases_str}

DEFAULT TO NO. Most sessions end normally. Only say YES if:
1. You can identify the agent's explicit completion claim (quote it)
2. AND you can identify specific incomplete deliverables

Respond with ONLY a JSON object:
{{"premature_termination": true/false, "confidence": 0.0-1.0, "reasoning": "one sentence with specific evidence"}}"""

    return prompt


def build_gd_assessment_prompt(
    events: list[TraceEvent],
    hta_state: HTAState,
    context_cases: list[dict] | None = None,
) -> str:
    """Build a binary YES/NO prompt focused solely on goal drift."""
    goal = _extract_agent_goal(events)
    session_head = _format_session_head(events, n=8)
    session_tail = _format_session_tail(events, n=15)
    total_events = len(events)
    cases_str = _format_similar_cases(context_cases or [])

    prompt = f"""You are reviewing a completed AI coding agent session for ONE specific question:

**Did the agent autonomously start working on something the user never asked for?**

USER'S ORIGINAL REQUEST: {goal}

SESSION START (first 8 events):
{session_head}

SESSION END (last 15 events):
{session_tail}

Total events: {total_events}

WHAT IS GOAL DRIFT:
- The agent COMPLETELY ABANDONED the user's request and started working on an
  UNRELATED topic — different files, different feature, different purpose
- Example: User asks "fix the login bug" → agent starts refactoring the database schema

WHAT IS NOT GOAL DRIFT (common false positives — answer NO for these):
- The agent doing autonomous sub-tasks to accomplish the user's goal (reading files,
  fixing related bugs, running tests, investigating issues) — this is NORMAL agent behavior
- The agent discovering a blocker and taking a detour to resolve it
- The agent reading/editing files the user didn't name but that are related to the task
- The user explicitly redirecting the conversation to a new topic
- The agent doing more work than strictly necessary but still on the same topic
- The agent fixing a bug it discovered while working on the user's request

KEY TEST: Is the agent's work COMPLETELY UNRELATED to the original request?
- If the work is even loosely connected to the original goal → NOT drift
- Agents are EXPECTED to do autonomous investigation, debugging, and fixing — that's
  their job, not goal drift

## Similar Past Cases
{cases_str}

DEFAULT TO NO. Goal drift is RARE. Coding agents routinely do autonomous sub-tasks.
Only say YES if:
1. The agent started working on something COMPLETELY UNRELATED to the user's request
2. AND there is no user_input event that prompted the new direction
3. AND the new work is clearly a different project/feature/purpose, not a sub-task

Respond with ONLY a JSON object:
{{"goal_drift": true/false, "confidence": 0.0-1.0, "reasoning": "one sentence with specific evidence"}}"""

    return prompt


def _parse_pt_assessment(text: str) -> PTAssessment:
    """Parse LLM response into PTAssessment."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    match = re.search(r"\{[^{}]*\}", text)
    if not match:
        raise ValueError(f"No JSON object found in response: {text[:200]}")

    data = json.loads(match.group())
    return PTAssessment(
        premature_termination=bool(data.get("premature_termination", False)),
        confidence=max(0.0, min(1.0, float(data.get("confidence", 0.5)))),
        reasoning=str(data.get("reasoning", "No reasoning provided.")),
        raw_response=text,
    )


def _parse_gd_assessment(text: str) -> GDAssessment:
    """Parse LLM response into GDAssessment."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    match = re.search(r"\{[^{}]*\}", text)
    if not match:
        raise ValueError(f"No JSON object found in response: {text[:200]}")

    data = json.loads(match.group())
    return GDAssessment(
        goal_drift=bool(data.get("goal_drift", False)),
        confidence=max(0.0, min(1.0, float(data.get("confidence", 0.5)))),
        reasoning=str(data.get("reasoning", "No reasoning provided.")),
        raw_response=text,
    )


async def assess_pt(
    events: list[TraceEvent],
    hta_state: HTAState,
    context_cases: list[dict] | None = None,
) -> PTAssessment:
    """Run binary PT assessment via LLM."""
    prompt = build_pt_assessment_prompt(events, hta_state, context_cases)
    try:
        response_text = await _call_llm(prompt)
        result = _parse_pt_assessment(response_text)
        logger.info(
            "PT assessment: %s (confidence=%.2f) — %s",
            result.premature_termination, result.confidence, result.reasoning,
        )
        return result
    except Exception as exc:
        logger.debug("PT assessment failed: %s", exc)
        return PTAssessment(
            premature_termination=False, confidence=0.0,
            reasoning=f"PT assessment unavailable ({type(exc).__name__}).",
        )


async def assess_gd(
    events: list[TraceEvent],
    hta_state: HTAState,
    context_cases: list[dict] | None = None,
) -> GDAssessment:
    """Run binary goal drift assessment via LLM."""
    prompt = build_gd_assessment_prompt(events, hta_state, context_cases)
    try:
        response_text = await _call_llm(prompt)
        result = _parse_gd_assessment(response_text)
        logger.info(
            "GD assessment: %s (confidence=%.2f) — %s",
            result.goal_drift, result.confidence, result.reasoning,
        )
        return result
    except Exception as exc:
        logger.debug("GD assessment failed: %s", exc)
        return GDAssessment(
            goal_drift=False, confidence=0.0,
            reasoning=f"GD assessment unavailable ({type(exc).__name__}).",
        )


def assess_pt_sync(
    events: list[TraceEvent],
    hta_state: HTAState,
    context_cases: list[dict] | None = None,
) -> PTAssessment:
    """Synchronous wrapper around assess_pt."""
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, assess_pt(events, hta_state, context_cases))
            return future.result(timeout=60)
    else:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(assess_pt(events, hta_state, context_cases))
        finally:
            try:
                _cancel_remaining_tasks(loop)
            except Exception:
                pass
            loop.close()


def assess_gd_sync(
    events: list[TraceEvent],
    hta_state: HTAState,
    context_cases: list[dict] | None = None,
) -> GDAssessment:
    """Synchronous wrapper around assess_gd."""
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, assess_gd(events, hta_state, context_cases))
            return future.result(timeout=60)
    else:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(assess_gd(events, hta_state, context_cases))
        finally:
            try:
                _cancel_remaining_tasks(loop)
            except Exception:
                pass
            loop.close()
