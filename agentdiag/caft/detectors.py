"""Unified CAFT detectors — all detectors in one file.

All detectors implement check(events, hta_state) → Optional[CaftDiagnosis].
No batch/live split. MonitorEngine runs them all.

CAFT codes covered:
  2.1 -- context_loss (Memory: Context Loss)
  2.2 -- step_repetition (Memory: Repetition)
  2.4 -- goal_drift (Memory: Goal Drift) [disabled: 11 FP]
  3.1 -- tool_thrashing (Execution: Tool Thrashing) [disabled: 9+ FP]
  3.4 -- analysis_paralysis (Execution: Analysis Paralysis)
  3.5 -- strategic_myopia (Decision Making: Strategic Myopia)
  4.1 -- tool_misuse (Resource: Tool Misuse)
  4.2 -- error_cascade (Resource: Error Cascade)
  4.3 -- recovery_failure (Resource: Recovery Failure)
  4.4 -- stall (Resource: Stall)
  4.4 -- token_explosion (Resource: Token Explosion)
  5.3 -- missing_verification (Plan: Sequencing Error) [disabled: 14 FP]
  5.4 -- premature_termination (Plan: Missing Precondition)
  6.4 -- reasoning_action_mismatch (Coordination: Mismatch) [disabled]
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Optional

import numpy as np

from agentdiag.models import TraceEvent
from agentdiag.hta import HTAState, Phase
from agentdiag.caft.base import CaftDetector, CaftDiagnosis, CaftSeverity


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _synthesize_timestamps(events: list[TraceEvent]) -> list[TraceEvent]:
    """Fill in timestamps from cumulative latency if not provided."""
    if not events:
        return events
    if all(e.timestamp is not None for e in events):
        return events
    t = 0.0
    for e in events:
        e.timestamp = t
        t += e.latency_ms / 1000.0
    return events


def _compute_output_hashes(events: list[TraceEvent]) -> list[TraceEvent]:
    """Generate output hashes where missing, using (tool, tokens_out) as proxy."""
    for e in events:
        if e.output_hash is None:
            raw = f"{e.tool}:{e.tokens_out}:{e.success}"
            e.output_hash = hashlib.md5(raw.encode()).hexdigest()[:8]
    return events


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# Keywords in user messages that indicate delegated verification
_VERIFICATION_KEYWORDS = re.compile(
    r"(?:it works|tests? pass|looks? good|here(?:'|&apos;)?s the (?:output|error|result)|"
    r"(?:got|getting|see(?:ing)?|hit(?:ting)?) (?:this|an?|the) error|"
    r"traceback|stack ?trace|exception|error:|failed|"
    r"output (?:is|was|shows|from)|"
    r"ran (?:it|the|this)|"
    r"results? (?:from|of|show)|"
    r"successfully|completed|"
    r"CUDA|OOM|segfault|"
    r"\.py\b.*(?:line|error)|"  # Python error references
    r"(?:sbatch|srun|qsub|slurm))",  # HPC job submission = user running things
    re.IGNORECASE,
)

# Keywords in user messages that indicate context continuation
_CONTINUATION_KEYWORDS = re.compile(
    r"(?:context (?:was |)(?:compressed|compacted|truncated|continued|lost)|"
    r"previous conversation|"
    r"ran out of context|"
    r"continue (?:from )?where|"
    r"pick(?:ing)? up where|"
    r"summary (?:of|below|above|covers)|"
    r"earlier portion|"
    r"</?summary>|"  # literal summary tags in continuation messages
    r"this session is being continued)",
    re.IGNORECASE,
)


def _has_user_message_near(events: list[TraceEvent], step: int, window: int = 5) -> bool:
    """Check if there's a user_input event within `window` steps before `step`."""
    for e in events:
        if e.type == "user_input" and (step - window) <= e.step < step:
            return True
    return False


def _find_continuation_steps(events: list[TraceEvent]) -> set[int]:
    """Find steps where a context continuation event occurred.

    Returns the set of steps at which a continuation was detected,
    so that post-continuation re-reads can be excluded from context_loss.
    """
    continuation_steps: set[int] = set()
    for e in events:
        if e.type == "user_input" and e.goal_text:
            if _CONTINUATION_KEYWORDS.search(e.goal_text):
                continuation_steps.add(e.step)
    return continuation_steps


def _has_delegated_verification(events: list[TraceEvent]) -> bool:
    """Check if the user provided verification via messages.

    Returns True if any user_input event contains verification keywords
    (error output, test results, 'it works', HPC job references, etc.)
    """
    for e in events:
        if e.type == "user_input" and e.goal_text:
            if _VERIFICATION_KEYWORDS.search(e.goal_text):
                return True
    return False


def _segment_agent_blocks(events: list[TraceEvent]) -> list[list[TraceEvent]]:
    """Split events into blocks separated by user_input events.

    Each block represents a sequence of agent actions between user messages.
    Used by GoalDriftDetector to detect within-block drift (unprompted).
    """
    blocks: list[list[TraceEvent]] = []
    current: list[TraceEvent] = []
    for e in events:
        if e.type == "user_input":
            if current:
                blocks.append(current)
            current = []
        else:
            current.append(e)
    if current:
        blocks.append(current)
    return blocks


class StepRepetitionDetector:
    """CAFT 2.2 -- Detects truly identical operations repeated 6+ times.

    V3: Lowered threshold from 9 to 6, added output-diversity check.
    If 6+ consecutive identical (tool, input_hash) but >80% unique output_hashes,
    the agent is making progress (e.g., file changed between reads), NOT repetition.
    In GATHERING phase, threshold raised to 10 (exploration is normal early on).
    """
    name = "step_repetition"
    caft_code = "2.2"

    THRESHOLD = 9  # V3: kept at 9 (threshold=6 produced FP on trace 4)

    _last_snapshot: Optional[dict] = None

    def check(
        self,
        events: list[TraceEvent],
        hta_state: HTAState,
    ) -> Optional[CaftDiagnosis]:
        step = events[-1].step if events else 0
        self._last_snapshot = {
            "detector": self.name, "step": step,
            "fired": False, "confidence": 0.0,
            "gate_failed": None, "evidence_preview": {},
        }

        if len(events) < self.THRESHOLD:
            self._last_snapshot["gate_failed"] = "min_events"
            return None

        # Look at recent events for consecutive identical operations.
        recent = events[-self.THRESHOLD * 3:]
        tool_calls = [
            e for e in recent if e.type == "tool_call" and e.tool
        ]

        if len(tool_calls) < self.THRESHOLD:
            self._last_snapshot["gate_failed"] = "min_tool_calls"
            return None

        # Build operation identity: (tool_name, input_hash)
        def _op_id(e: TraceEvent) -> tuple:
            return (e.tool, e.input_hash or "_no_hash")

        # Count max consecutive IDENTICAL operations and track the run
        max_run = 1
        current_run = 1
        worst_op = _op_id(tool_calls[0])
        worst_tool = tool_calls[0].tool
        worst_run_end = 0
        worst_run_start = 0
        current_run_start = 0

        for i in range(1, len(tool_calls)):
            op_cur = _op_id(tool_calls[i])
            op_prev = _op_id(tool_calls[i - 1])

            if op_cur == op_prev:
                current_run += 1
                if current_run > max_run:
                    max_run = current_run
                    worst_op = op_cur
                    worst_tool = tool_calls[i].tool
                    worst_run_end = i
                    worst_run_start = current_run_start
            else:
                current_run = 1
                current_run_start = i

        if max_run < self.THRESHOLD:
            self._last_snapshot["gate_failed"] = "max_run_below_threshold"
            self._last_snapshot["evidence_preview"] = {"max_run": max_run, "threshold": self.THRESHOLD}
            return None

        # V3: Output diversity check
        # If the repeated ops all produce DIFFERENT outputs, the agent
        # is making progress (e.g., file changed between reads)
        run_events = tool_calls[worst_run_start:worst_run_end + 1]
        output_hashes = [e.output_hash for e in run_events if e.output_hash]
        if output_hashes:
            unique_ratio = len(set(output_hashes)) / len(output_hashes)
            if unique_ratio > 0.8:  # >80% unique outputs = progressive work
                self._last_snapshot["gate_failed"] = "output_diverse"
                self._last_snapshot["evidence_preview"] = {"max_run": max_run, "unique_ratio": round(unique_ratio, 3)}
                return None

        # V3: Phase-aware threshold
        # In GATHERING phase, exploration is normal -- raise threshold
        if hta_state.current_phase == Phase.GATHERING:
            early_threshold = max(self.THRESHOLD, 10)
            if max_run < early_threshold:
                self._last_snapshot["gate_failed"] = "gathering_threshold"
                self._last_snapshot["evidence_preview"] = {"max_run": max_run, "early_threshold": early_threshold}
                return None

        # Exclude meta-operations that are legitimately repeated
        _META_TOOLS = {
            "task", "exitplanmode", "enterplanmode", "askuserquestion",
            "todowrite", "taskcreate", "taskupdate", "taskget", "tasklist",
        }
        if worst_tool and worst_tool.lower() in _META_TOOLS:
            self._last_snapshot["gate_failed"] = "meta_tool"
            self._last_snapshot["evidence_preview"] = {"tool": worst_tool, "max_run": max_run}
            return None

        confidence = min(max_run / (self.THRESHOLD * 2), 1.0)
        self._last_snapshot["fired"] = True
        self._last_snapshot["confidence"] = round(confidence, 4)
        self._last_snapshot["evidence_preview"] = {"max_run": max_run, "tool": worst_tool}

        return CaftDiagnosis(
            caft_code="2.2",
            caft_category="memory",
            failure_name="step_repetition",
            severity=CaftSeverity.WARNING if max_run < 10 else CaftSeverity.CRITICAL,
            confidence=min(max_run / (self.THRESHOLD * 2), 1.0),
            description=(
                f"Identical operation ({worst_tool}, hash={worst_op[1][:8]}) "
                f"repeated {max_run} consecutive times -- possible memory lapse."
            ),
            evidence={
                "repeated_operation": worst_tool,
                "input_hash": worst_op[1],
                "consecutive_count": max_run,
                "threshold": self.THRESHOLD,
            },
            at_step=events[-1].step,
            remediation="Check if previous results were captured; add deduplication.",
        )


class ContextLossDetector:
    """CAFT 2.1 -- Detects re-reading of already-processed files/resources.

    V3: Scans ALL re-read candidates (not just first found), fires on
    strongest signal (largest gap). Adds relative staleness: gap must be
    >= max(MIN_INTERVENING, 8% of total events).

    Ground truth insight: V2 returned on first re-read found. If first pair
    was blocked by continuation/user message, later valid pairs were missed.
    """
    name = "context_loss"
    caft_code = "2.1"

    MIN_EVENTS = 6
    MIN_INTERVENING = 5
    STALENESS_RATIO = 0.08  # gap must be >= 8% of total events

    _last_snapshot: Optional[dict] = None

    def check(
        self,
        events: list[TraceEvent],
        hta_state: HTAState,
    ) -> Optional[CaftDiagnosis]:
        step = events[-1].step if events else 0
        self._last_snapshot = {
            "detector": self.name, "step": step,
            "fired": False, "confidence": 0.0,
            "gate_failed": None, "evidence_preview": {},
        }

        if len(events) < self.MIN_EVENTS:
            self._last_snapshot["gate_failed"] = "min_events"
            return None

        # Find all context continuation points
        continuation_steps = _find_continuation_steps(events)

        # Track read-like operations by output_hash (same content = same resource)
        read_tools = {"read_file", "read", "cat", "head", "search_docs", "fetch"}
        reads: dict[str, list[int]] = {}

        for e in events:
            if e.tool and any(t in e.tool.lower() for t in read_tools):
                key = e.output_hash
                if not key or key == e.tool:
                    continue
                if key not in reads:
                    reads[key] = []
                reads[key].append(e.step)

        # V3: Collect ALL re-read candidates, then fire on the strongest one
        candidates = []  # (key, steps_pair, intervening_count, gap_ratio)

        for key, steps in reads.items():
            if len(steps) < 2:
                continue

            # Check ALL pairs, not just first-last
            for idx_i in range(len(steps) - 1):
                for idx_j in range(idx_i + 1, len(steps)):
                    first, last = steps[idx_i], steps[idx_j]

                    # Skip if continuation between
                    if any(first < cs < last for cs in continuation_steps):
                        continue

                    # Skip if user message between
                    if any(
                        e.type == "user_input" and first < e.step < last
                        for e in events
                    ):
                        continue

                    # Count intervening non-read tool calls
                    intervening = [
                        e for e in events
                        if first < e.step < last
                        and e.type == "tool_call"
                        and e.tool
                        and not any(t in e.tool.lower() for t in read_tools)
                    ]

                    # V3: Relative staleness -- gap must be meaningful relative to session
                    min_gap = max(self.MIN_INTERVENING, int(len(events) * self.STALENESS_RATIO))
                    if len(intervening) >= min_gap:
                        candidates.append((key, [first, last], len(intervening)))

        if not candidates:
            self._last_snapshot["gate_failed"] = "no_reread_candidates"
            self._last_snapshot["evidence_preview"] = {"unique_reads": len(reads), "multi_reads": sum(1 for s in reads.values() if len(s) >= 2)}
            return None

        # Fire on strongest candidate (largest gap)
        best = max(candidates, key=lambda c: c[2])
        key, step_pair, interv_count = best

        confidence = min(interv_count / 10.0, 1.0)
        self._last_snapshot["fired"] = True
        self._last_snapshot["confidence"] = round(confidence, 4)
        self._last_snapshot["evidence_preview"] = {
            "total_candidates": len(candidates),
            "best_gap": interv_count,
            "resource_hash": key[:8] if key else "",
        }

        return CaftDiagnosis(
            caft_code="2.1",
            caft_category="memory",
            failure_name="context_loss",
            severity=CaftSeverity.WARNING,
            confidence=min(interv_count / 10.0, 1.0),
            description=(
                f"Re-read resource (hash={key[:8]}) at steps "
                f"{step_pair} with {interv_count} operations between -- "
                f"no context continuation detected, possible context loss."
            ),
            evidence={
                "resource_hash": key,
                "read_steps": step_pair,
                "read_count": 2,
                "intervening_operations": interv_count,
                "continuation_between": False,
                "total_candidates": len(candidates),
            },
            at_step=step_pair[-1],
            remediation="Summarize key findings after reads; use external scratchpad.",
        )


class PrematureTerminationDetector:
    """CAFT 5.4 -- Detects premature task termination (Tier 1.5).

    V13: Mode 1 only (delivering without verification). Modes 2+3 removed;
    semantic PT detection moved to Tier 2 session-end assessment.

    Mode 1 fires on structural signal (skip VERIFYING phase) and always
    requires LLM confirmation (force_llm_review=True).
    """
    name = "premature_termination"
    caft_code = "5.4"

    MIN_EXECUTE_EVENTS = 3
    MIN_DELIVERING_EVENTS = 3

    def check(
        self,
        events: list[TraceEvent],
        hta_state: HTAState,
    ) -> Optional[CaftDiagnosis]:
        if len(events) < 5:
            return None

        # Mode 1: Delivering without verification
        delivering_count = hta_state.phase_event_counts.get("delivering", 0)
        if hta_state.current_phase == Phase.DELIVERING and delivering_count >= self.MIN_DELIVERING_EVENTS:
            has_verified = any(
                t.to_phase == Phase.VERIFYING for t in hta_state.transitions
            )
            has_executed = (
                hta_state.phase_event_counts.get("executing", 0) >= self.MIN_EXECUTE_EVENTS
            )
            delegated = _has_delegated_verification(events)

            if has_executed and not has_verified and not delegated:
                return CaftDiagnosis(
                    caft_code="5.4",
                    caft_category="plan_structure",
                    failure_name="premature_termination",
                    severity=CaftSeverity.CRITICAL,
                    confidence=0.85,
                    description=(
                        f"Agent is delivering after "
                        f"{hta_state.phase_event_counts.get('executing', 0)} "
                        f"execution steps but NEVER verified its work."
                    ),
                    evidence={
                        "mode": "skip_verification",
                        "execution_events": hta_state.phase_event_counts.get("executing", 0),
                        "verification_events": 0,
                        "delegated_verification": False,
                        "phases_visited": [t.to_phase.label for t in hta_state.transitions],
                    },
                    at_step=events[-1].step if events else 0,
                    remediation="Add verification step: run tests, review output, check for errors.",
                    force_llm_review=True,
                )

        return None


class MissingVerificationDetector:
    """CAFT 5.3 -- Detects code changes without subsequent test/review.

    V3: Re-enabled with multi-pattern verification detection.
    Recognizes 5 patterns of verification:
    1. Formal HTA verification phase events
    2. Bash commands with test intent (pytest, npm test, etc.)
    3. Task subagent with testing intent
    4. Delegated verification (user pasting output)
    5. Agent acknowledges test results in reasoning

    Ground truth insight: V1/V2 failed because Claude Code runs tests via Bash
    (classified as EXECUTING). Multi-pattern detection catches all these.
    """
    name = "missing_verification"
    caft_code = "5.3"

    EXECUTE_THRESHOLD = 25

    _TEST_CMD_PATTERN = re.compile(
        r"(?:pytest|python -m pytest|npm test|jest|cargo test|"
        r"go test|make test|rspec|phpunit|"
        r"python.*test|bash.*test|\.\/test)",
        re.IGNORECASE,
    )

    _TEST_TASK_PATTERN = re.compile(
        r"(?:run.*tests?|test.*(?:pass|fail|results?)|"
        r"verify|validate|check.*(?:output|results?))",
        re.IGNORECASE,
    )

    _TEST_REASONING_PHRASES = [
        "tests pass", "all tests", "test output",
        "verified", "confirmed working", "tests are passing",
        "test suite", "tests succeeded",
    ]

    def _has_any_verification(self, events: list[TraceEvent], hta_state: HTAState) -> bool:
        """Multi-pattern verification detection."""
        # Pattern 1: Formal HTA verification
        if hta_state.phase_event_counts.get("verifying", 0) > 0:
            return True

        # Pattern 2: Bash commands with test intent
        for e in events:
            if e.tool and "bash" in e.tool.lower() and e.goal_text:
                if self._TEST_CMD_PATTERN.search(e.goal_text):
                    return True

        # Pattern 3: Task subagent with testing intent
        for e in events:
            if e.tool and e.tool.lower() == "task" and e.goal_text:
                if self._TEST_TASK_PATTERN.search(e.goal_text):
                    return True

        # Pattern 4: Delegated verification (user pasting output)
        if _has_delegated_verification(events):
            return True

        # Pattern 5: Agent acknowledges test results in reasoning
        for e in events:
            if e.type in ("reasoning", "planning") and e.goal_text:
                text = e.goal_text.lower()
                if any(p in text for p in self._TEST_REASONING_PHRASES):
                    return True

        return False

    def check(
        self,
        events: list[TraceEvent],
        hta_state: HTAState,
    ) -> Optional[CaftDiagnosis]:
        exec_count = hta_state.phase_event_counts.get("executing", 0)

        if exec_count < self.EXECUTE_THRESHOLD:
            return None

        # Multi-pattern verification check
        if self._has_any_verification(events, hta_state):
            return None

        # Check if there are write/edit operations
        # Note: "bash" excluded -- it's a generic execution tool, not necessarily writing files.
        # Consistent with PrematureTermination's write_tools check.
        write_tools = {"write_file", "edit_file", "edit", "write", "create_file"}
        has_writes = any(
            e.tool and any(t in e.tool.lower() for t in write_tools)
            for e in events
        )

        if not has_writes:
            return None

        # If the session is still in gathering/planning, don't fire prematurely
        if hta_state.current_phase in (Phase.GATHERING, Phase.PLANNING):
            return None

        return CaftDiagnosis(
            caft_code="5.3",
            caft_category="plan_structure",
            failure_name="missing_verification",
            severity=CaftSeverity.WARNING,
            confidence=min(exec_count / (self.EXECUTE_THRESHOLD * 3), 0.9),
            description=(
                f"Agent has {exec_count} execution events with file writes "
                f"but no verification detected (checked 5 verification patterns)."
            ),
            evidence={
                "execution_events": exec_count,
                "verification_events": 0,
                "delegated_verification": False,
                "total_events": hta_state.total_events,
            },
            at_step=events[-1].step if events else 0,
            remediation="Run tests after code changes; add review step before delivery.",
        )


class ReasoningActionMismatchDetector:
    """CAFT 6.4 -- Detects when reasoning says X but action does Y.

    Still DISABLED in V3 -- keyword matching is too brittle.
    (0 TP and 0 FP in ground truth, detector works but produces no signal)
    """
    name = "reasoning_action_mismatch"
    caft_code = "6.4"

    _READ_KEYWORDS = {"read", "look at", "examine", "check", "review", "inspect"}
    _WRITE_KEYWORDS = {"write", "edit", "modify", "change", "update", "fix"}
    _TEST_KEYWORDS = {"test", "run", "verify", "check", "validate"}
    _SEARCH_KEYWORDS = {"search", "find", "look for", "grep", "locate"}

    def check(
        self,
        events: list[TraceEvent],
        hta_state: HTAState,
    ) -> Optional[CaftDiagnosis]:
        if len(events) < 2:
            return None

        for i in range(len(events) - 1):
            reasoning = events[i]
            action = events[i + 1]

            if reasoning.type not in ("reasoning", "planning"):
                continue
            if action.type != "tool_call" or not action.tool:
                continue
            if not reasoning.goal_text:
                continue

            text = reasoning.goal_text.lower()
            tool = action.tool.lower()

            if any(k in text for k in self._READ_KEYWORDS):
                if any(t in tool for t in {"write", "edit", "create", "delete"}):
                    return CaftDiagnosis(
                        caft_code="6.4",
                        caft_category="coordination",
                        failure_name="reasoning_action_mismatch",
                        severity=CaftSeverity.WARNING,
                        confidence=0.7,
                        description=(
                            f"Agent planned to read/review at step {reasoning.step} "
                            f"but executed {action.tool} (write operation) at step {action.step}."
                        ),
                        evidence={
                            "reasoning_step": reasoning.step,
                            "action_step": action.step,
                            "planned_intent": "read/review",
                            "actual_tool": action.tool,
                        },
                        at_step=action.step,
                        remediation="Enforce plan-then-execute: verify action matches stated intent.",
                    )

            if any(k in text for k in self._TEST_KEYWORDS):
                if any(t in tool for t in {"write", "edit", "create"}):
                    return CaftDiagnosis(
                        caft_code="6.4",
                        caft_category="coordination",
                        failure_name="reasoning_action_mismatch",
                        severity=CaftSeverity.WARNING,
                        confidence=0.65,
                        description=(
                            f"Agent planned to test/verify at step {reasoning.step} "
                            f"but executed {action.tool} (write operation) at step {action.step}."
                        ),
                        evidence={
                            "reasoning_step": reasoning.step,
                            "action_step": action.step,
                            "planned_intent": "test/verify",
                            "actual_tool": action.tool,
                        },
                        at_step=action.step,
                        remediation="Add reasoning-action consistency check before tool execution.",
                    )

        return None


class GoalDriftDetector:
    """CAFT 2.4 -- Detects actions diverging from the stated objective.

    V3: Agent-block segmentation replaces half-session comparison.
    Segments events into "agent blocks" (sequences between user messages).
    Fires when agent shows sustained off-topic behavior WITHIN blocks.

    Ground truth insight: V2 half-session comparison always showed tool shift
    (first half=gathering, second half=executing). Agent-block segmentation
    detects drift without confounding lifecycle evolution.
    """
    name = "goal_drift"
    caft_code = "2.4"

    MIN_EVENTS = 20

    def check(
        self,
        events: list[TraceEvent],
        hta_state: HTAState,
    ) -> Optional[CaftDiagnosis]:
        if len(events) < self.MIN_EVENTS:
            return None

        # 1. Segment into agent blocks (actions between user messages)
        blocks = _segment_agent_blocks(events)
        if len(blocks) < 3:
            return None

        # 2. Establish "baseline" from first 2 blocks
        baseline_tools: set[str] = set()
        for block in blocks[:2]:
            baseline_tools |= {e.tool for e in block if e.tool}

        # 3. Score later blocks for novelty
        drift_blocks = 0
        for block in blocks[2:]:
            block_tools = {e.tool for e in block if e.tool}
            if not block_tools:
                continue
            novel_tools = block_tools - baseline_tools
            tool_novelty = len(novel_tools) / len(block_tools)

            # Block is "drifted" if >50% novel tools and has substance
            if tool_novelty > 0.5 and len(block) >= 3:
                drift_blocks += 1

        # 4. Count unprompted phase regressions
        unprompted_regressions = sum(
            1 for t in hta_state.transitions
            if t.is_regression and not _has_user_message_near(events, t.at_step, 3)
        )

        # 5. Composite: need drifted blocks + unprompted regressions
        fired = False
        confidence = 0.0
        if drift_blocks >= 3 and unprompted_regressions >= 3:
            fired = True
            confidence = min(drift_blocks / 5.0, 0.85)
        elif drift_blocks >= 2 and unprompted_regressions >= 5:
            fired = True
            confidence = min(drift_blocks / 4.0, 0.85)

        if not fired:
            return None

        return CaftDiagnosis(
            caft_code="2.4",
            caft_category="memory",
            failure_name="goal_drift",
            severity=CaftSeverity.WARNING if confidence < 0.7 else CaftSeverity.CRITICAL,
            confidence=round(confidence, 3),
            description=(
                f"Unprompted goal drift: {drift_blocks} agent blocks with >50% novel tools, "
                f"{unprompted_regressions} unprompted phase regressions. "
                f"Baseline tools: {sorted(baseline_tools)}."
            ),
            evidence={
                "drift_blocks": drift_blocks,
                "total_blocks": len(blocks),
                "unprompted_regressions": unprompted_regressions,
                "baseline_tools": sorted(baseline_tools),
            },
            at_step=events[-1].step,
            remediation="Re-inject original goal; check if agent is still pursuing stated objective.",
        )


class ToolThrashingDetector:
    """CAFT 3.1 -- Detects read-only thrashing during execution.

    Fires when the agent has 15+ consecutive read-only actions while
    HTA classifies as EXECUTING, or 25+ consecutive read-only in any phase.
    This indicates the agent is stuck in analysis paralysis.
    """
    name = "tool_thrashing"
    caft_code = "3.1"

    EXECUTING_THRESHOLD = 15
    ANY_PHASE_THRESHOLD = 25

    _READ_ONLY_TOOLS = {
        "read_file", "read", "cat", "head", "tail", "grep", "glob",
        "find", "list_files", "ls", "search", "fetch", "get",
        "describe", "search_docs", "web_search", "search_codebase",
    }

    def check(
        self,
        events: list[TraceEvent],
        hta_state: HTAState,
    ) -> Optional[CaftDiagnosis]:
        if len(events) < self.EXECUTING_THRESHOLD:
            return None

        # Count max consecutive read-only tool_call events
        tool_calls = [e for e in events if e.type == "tool_call" and e.tool]
        max_run = 0
        current_run = 0

        for e in tool_calls:
            if e.tool.lower() in self._READ_ONLY_TOOLS:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0

        # Phase-aware threshold
        threshold = self.EXECUTING_THRESHOLD
        if hta_state.current_phase != Phase.EXECUTING:
            threshold = self.ANY_PHASE_THRESHOLD

        if max_run < threshold:
            return None

        return CaftDiagnosis(
            caft_code="3.1",
            caft_category="execution",
            failure_name="tool_thrashing",
            severity=CaftSeverity.WARNING if max_run < 20 else CaftSeverity.CRITICAL,
            confidence=min(max_run / (threshold * 2), 0.85),
            description=(
                f"Agent performed {max_run} consecutive read-only "
                f"operations in {hta_state.current_phase.label} phase."
            ),
            evidence={
                "consecutive_read_only": max_run,
                "threshold": threshold,
                "phase": hta_state.current_phase.label,
            },
            at_step=events[-1].step if events else 0,
            remediation="Stop gathering and take a concrete action.",
        )


# ---------------------------------------------------------------------------
# Migrated batch detectors (now native CAFT format)
# ---------------------------------------------------------------------------

class ToolMisuseDetector:
    """CAFT 4.1 -- Rapid tool switching with no progress.

    Sliding window of size 5: high switch rate + low state change = thrashing.
    Migrated from batch ThrashDetector.
    """
    name = "tool_misuse"
    caft_code = "4.1"

    WINDOW = 5

    def check(
        self,
        events: list[TraceEvent],
        hta_state: HTAState,
    ) -> Optional[CaftDiagnosis]:
        events = _compute_output_hashes(events)
        tool_events = [e for e in events if e.type == "tool_call"]
        if len(tool_events) < self.WINDOW:
            return None

        max_thrash_score = 0.0
        worst_window_start = 0

        for i in range(len(tool_events) - self.WINDOW + 1):
            window = tool_events[i:i + self.WINDOW]
            tools = [e.tool for e in window]
            hashes = [e.output_hash for e in window]

            switches = sum(1 for j in range(1, len(tools)) if tools[j] != tools[j - 1])
            switch_rate = switches / (self.WINDOW - 1)

            state_changes = sum(1 for j in range(1, len(hashes)) if hashes[j] != hashes[j - 1])
            stagnation = 1.0 - (state_changes / (self.WINDOW - 1))

            score = switch_rate * stagnation
            if score > max_thrash_score:
                max_thrash_score = score
                worst_window_start = i

        if max_thrash_score < 0.4:
            return None

        worst_window = tool_events[worst_window_start:worst_window_start + self.WINDOW]
        tools_in_window = [e.tool for e in worst_window]
        unique_tools = list(set(tools_in_window))

        confidence = round(min(max_thrash_score, 0.85), 3)
        return CaftDiagnosis(
            caft_code="4.1",
            caft_category="resource",
            failure_name="tool_misuse",
            severity=CaftSeverity.WARNING if confidence < 0.7 else CaftSeverity.CRITICAL,
            confidence=confidence,
            description=(
                f"Rapid tool switching between {unique_tools} "
                f"in a {self.WINDOW}-step window with minimal state change."
            ),
            evidence={
                "worst_window_tools": tools_in_window,
                "unique_tools_in_window": unique_tools,
                "switch_rate": round(max_thrash_score, 3),
                "window_start_step": worst_window[0].step,
            },
            at_step=worst_window[-1].step,
            remediation="Pause and re-plan; select one tool and commit to it.",
        )


class StallDetector:
    """CAFT 4.4 -- Long gaps between productive actions.

    Uses IQR-based outlier detection on inter-event latencies, but only
    on INTRA-TURN latencies (tool_call chains). User idle time between
    turns (minutes/hours) is excluded because that's the user taking a
    break, not the agent stalling.

    A stall is only diagnosed when multiple consecutive long gaps occur
    during active work — a single pause is normal.

    Important: Claude Code traces have bimodal latency — 0ms text chunks
    interspersed with 5-90s API calls.  When the median is near zero the
    IQR collapses and ANY API call looks like an outlier.  The
    ``MIN_MEDIAN_MS`` guard rejects traces without meaningful latency
    data, and the ``MIN_STALL_MS`` floor (30s) ensures that normal
    API calls (5-20s) are not flagged.
    """
    name = "stall"
    caft_code = "4.4"

    IQR_MULTIPLIER = 1.5
    MIN_EVENTS = 10  # IQR unreliable below this
    MIN_STALL_MS = 30_000.0  # absolute floor — normal Claude API calls take 5-20s
    MAX_INTER_TURN_MS = 120_000.0  # >2min gap = user idle, not agent stall
    MIN_STALL_COUNT = 2  # need at least 2 stall events (fraction-based killed long sessions)
    MIN_MEDIAN_MS = 50.0  # skip traces where median ≈ 0 (no real latency data)

    _last_snapshot: Optional[dict] = None

    def check(
        self,
        events: list[TraceEvent],
        hta_state: HTAState,
    ) -> Optional[CaftDiagnosis]:
        step = events[-1].step if events else 0
        self._last_snapshot = {
            "detector": self.name, "step": step,
            "fired": False, "confidence": 0.0,
            "gate_failed": None, "evidence_preview": {},
        }

        events = _synthesize_timestamps(events)
        if len(events) < self.MIN_EVENTS:
            self._last_snapshot["gate_failed"] = "min_events"
            self._last_snapshot["evidence_preview"] = {"event_count": len(events), "min_required": self.MIN_EVENTS}
            return None

        latencies = np.array([e.latency_ms for e in events])

        # Filter out user idle time (inter-turn gaps)
        active_mask = latencies <= self.MAX_INTER_TURN_MS
        active_latencies = latencies[active_mask]
        if len(active_latencies) < self.MIN_EVENTS:
            self._last_snapshot["gate_failed"] = "min_active_events"
            self._last_snapshot["evidence_preview"] = {"active_events": int(active_mask.sum()), "min_required": self.MIN_EVENTS}
            return None

        # Median guard: when median ≈ 0 the trace has no real latency data
        # (e.g., Claude Code JSONL where most events are rapid text chunks).
        # The IQR collapses to 0 and ANY API call becomes an "outlier".
        median_latency = float(np.median(active_latencies))
        if median_latency < self.MIN_MEDIAN_MS:
            self._last_snapshot["gate_failed"] = "low_median"
            self._last_snapshot["evidence_preview"] = {
                "median_latency_ms": round(median_latency, 1),
                "min_median_ms": self.MIN_MEDIAN_MS,
                "active_events": int(len(active_latencies)),
            }
            return None

        q25, q75 = np.percentile(active_latencies, [25, 75])
        iqr = q75 - q25
        threshold = max(q75 + self.IQR_MULTIPLIER * iqr, self.MIN_STALL_MS)

        # Find stalls only among active (intra-turn) events
        stall_mask = active_mask & (latencies > threshold)
        stall_indices = np.where(stall_mask)[0]
        if len(stall_indices) == 0:
            self._last_snapshot["gate_failed"] = "no_outliers"
            self._last_snapshot["evidence_preview"] = {
                "threshold_ms": round(float(threshold), 1),
                "max_active_latency_ms": round(float(active_latencies.max()), 1),
                "median_latency_ms": round(float(np.median(active_latencies)), 1),
            }
            return None

        # Require a minimum number of stall events (not fraction-based,
        # because long sessions dilute the fraction even with real stalls)
        if len(stall_indices) < self.MIN_STALL_COUNT:
            self._last_snapshot["gate_failed"] = "min_stall_count"
            self._last_snapshot["evidence_preview"] = {
                "stall_count": int(len(stall_indices)),
                "threshold_ms": round(float(threshold), 1),
                "min_count": self.MIN_STALL_COUNT,
            }
            return None

        # Use only active latencies for stats
        stall_latencies = latencies[stall_mask]
        max_latency = float(stall_latencies.max())
        median_latency = float(np.median(active_latencies))
        max_idx = int(np.where(latencies == stall_latencies.max())[0][0])

        if threshold > 0:
            excess = (max_latency - threshold) / threshold
            confidence = min(0.5 + excess * 0.3, 1.0)
        else:
            confidence = 0.8

        # Stall never forces LLM review. The IQR detector is reliable:
        # confidence=1.0 means the latency is extremely anomalous.
        # Auto-confirm at 0.9 handles high-confidence cases; lower-confidence
        # cases still go to LLM via the normal path (no force override).
        needs_llm_review = False

        if confidence < 0.3:
            self._last_snapshot["gate_failed"] = "min_confidence"
            self._last_snapshot["confidence"] = round(confidence, 4)
            self._last_snapshot["evidence_preview"] = {
                "stall_count": int(len(stall_indices)),
                "max_latency_ms": round(max_latency, 1),
                "threshold_ms": round(float(threshold), 1),
            }
            return None

        # Slow-tool gate: if ALL stall tools are Task/TaskOutput/Bash, not a real stall
        _SLOW_TOOLS = {"Task", "TaskOutput", "Bash"}
        stall_tool_set = set(
            events[int(i)].tool or events[int(i)].type
            for i in stall_indices
        )
        if stall_tool_set and stall_tool_set <= _SLOW_TOOLS:
            self._last_snapshot["gate_failed"] = "slow_tools_only"
            self._last_snapshot["evidence_preview"] = {
                "stall_count": int(len(stall_indices)),
                "stall_tools": sorted(stall_tool_set),
            }
            return None

        # Cold-start gate: if all stalls in first 5 steps, it's loading latency
        if max(stall_indices) <= 5:
            self._last_snapshot["gate_failed"] = "cold_start"
            self._last_snapshot["evidence_preview"] = {
                "stall_count": int(len(stall_indices)),
                "max_stall_index": int(max(stall_indices)),
            }
            return None

        self._last_snapshot["fired"] = True
        self._last_snapshot["confidence"] = round(confidence, 4)
        self._last_snapshot["evidence_preview"] = {
            "stall_count": int(len(stall_indices)),
            "max_latency_ms": round(max_latency, 1),
            "threshold_ms": round(float(threshold), 1),
            "active_events": int(len(active_latencies)),
        }

        return CaftDiagnosis(
            caft_code="4.4",
            caft_category="resource",
            failure_name="stall",
            severity=CaftSeverity.WARNING if confidence < 0.7 else CaftSeverity.CRITICAL,
            confidence=round(confidence, 3),
            description=(
                f"Agent stalled {len(stall_indices)} times during active work "
                f"({len(active_latencies)} active events). "
                f"Worst at step {events[max_idx].step} "
                f"({max_latency:.0f}ms vs median {median_latency:.0f}ms)."
            ),
            evidence={
                "stall_steps": [int(i) for i in stall_indices[:20]],
                "stall_tool_names": [
                    events[int(i)].tool or events[int(i)].type
                    for i in stall_indices[:20]
                ],
                "max_latency_ms": round(max_latency, 1),
                "median_latency_ms": round(median_latency, 1),
                "threshold_ms": round(threshold, 1),
                "stall_count": len(stall_indices),
                "active_events": int(active_mask.sum()),
                "idle_events_excluded": int((~active_mask).sum()),
                "worst_step": events[max_idx].step,
                "worst_tool": events[max_idx].tool or events[max_idx].type,
            },
            at_step=events[max_idx].step,
            remediation="Check for blocking I/O or rate limits; consider timeout.",
            force_llm_review=needs_llm_review,
        )


class ErrorCascadeDetector:
    """CAFT 4.2 -- Error propagation chains.

    Detects consecutive errors that propagate downstream.
    Migrated from batch CascadeDetector.
    """
    name = "error_cascade"
    caft_code = "4.2"

    MIN_CHAIN = 3  # 2 consecutive failures is normal search refinement

    _last_snapshot: Optional[dict] = None

    def check(
        self,
        events: list[TraceEvent],
        hta_state: HTAState,
    ) -> Optional[CaftDiagnosis]:
        step = events[-1].step if events else 0
        self._last_snapshot = {
            "detector": self.name, "step": step,
            "fired": False, "confidence": 0.0,
            "gate_failed": None, "evidence_preview": {},
        }

        if len(events) < 3:
            self._last_snapshot["gate_failed"] = "min_events"
            return None

        chains = []
        current_chain: list[TraceEvent] = []
        for e in events:
            if not e.success:
                current_chain.append(e)
            else:
                if len(current_chain) >= self.MIN_CHAIN:
                    chains.append(current_chain)
                current_chain = []
        if len(current_chain) >= self.MIN_CHAIN:
            chains.append(current_chain)

        if not chains:
            # Compute the longest chain even if below threshold, for the snapshot
            max_chain_len = 0
            cur = 0
            for e in events:
                if not e.success:
                    cur += 1
                    max_chain_len = max(max_chain_len, cur)
                else:
                    cur = 0
            self._last_snapshot["gate_failed"] = "no_chains_above_min"
            self._last_snapshot["evidence_preview"] = {
                "longest_chain": max_chain_len,
                "min_chain": self.MIN_CHAIN,
                "total_errors": sum(1 for e in events if not e.success),
            }
            return None

        longest = max(chains, key=len)
        total_errors = sum(1 for e in events if not e.success)

        confidence = min(len(longest) / 10.0, 1.0)
        if confidence < 0.3:
            self._last_snapshot["gate_failed"] = "min_confidence"
            self._last_snapshot["confidence"] = round(confidence, 4)
            self._last_snapshot["evidence_preview"] = {
                "longest_chain": len(longest),
                "total_errors": total_errors,
            }
            return None

        # Request LLM review for short chains (< 5 errors).
        needs_llm_review = len(longest) < 5

        self._last_snapshot["fired"] = True
        self._last_snapshot["confidence"] = round(confidence, 4)
        self._last_snapshot["evidence_preview"] = {
            "longest_chain": len(longest),
            "total_chains": len(chains),
            "total_errors": total_errors,
        }

        return CaftDiagnosis(
            caft_code="4.2",
            caft_category="resource",
            failure_name="error_cascade",
            severity=CaftSeverity.WARNING if confidence < 0.7 else CaftSeverity.CRITICAL,
            confidence=round(confidence, 3),
            description=(
                f"Error cascade of {len(longest)} consecutive failures "
                f"from step {longest[0].step} to {longest[-1].step}."
            ),
            evidence={
                "longest_error_chain": len(longest),
                "chain_start_step": longest[0].step,
                "chain_end_step": longest[-1].step,
                "total_error_chains": len(chains),
                "total_errors": total_errors,
                "tools_in_chain": [e.tool for e in longest],
            },
            at_step=longest[-1].step,
            remediation="Fix the root cause error before retrying downstream steps.",
            force_llm_review=needs_llm_review,
        )


class TokenExplosionDetector:
    """CAFT 4.4 -- Exponential token growth.

    Fits regression on per-step tokens. High growth ratio + acceleration = explosion.
    Data-calibrated: median growth on real traces is 1.1x, P75 = 2.2x.
    Only fires for genuinely explosive growth (>3x) with acceleration.
    """
    name = "token_explosion"
    caft_code = "4.4"

    # Data-derived: P75 growth = 2.2x on real traces.
    # Growth >3x is clearly above P95, indicating a real problem.
    GROWTH_THRESHOLD = 3.0
    MIN_SCORE = 0.5  # higher threshold than synthetic-tuned 0.3
    MIN_EVENTS = 50  # need enough events for stable trend (not early noise)
    MIN_FIRST_Q_TOKENS = 50  # first quarter must be meaningful (not near-zero)

    def check(
        self,
        events: list[TraceEvent],
        hta_state: HTAState,
    ) -> Optional[CaftDiagnosis]:
        if len(events) < self.MIN_EVENTS:
            return None

        per_step = np.array([e.tokens_in + e.tokens_out for e in events], dtype=np.float64)
        if per_step.sum() == 0:
            return None

        steps = np.arange(len(per_step), dtype=np.float64)
        slope, _intercept = np.polyfit(steps, per_step, 1)

        if len(per_step) >= 5:
            coeffs = np.polyfit(steps, per_step, 2)
            acceleration = coeffs[0]
        else:
            acceleration = 0.0

        q = max(len(per_step) // 4, 1)
        first_q_mean = float(np.mean(per_step[:q]))
        last_q_mean = float(np.mean(per_step[-q:]))

        # Skip if first quarter is near-zero (growth ratio is meaningless)
        if first_q_mean < self.MIN_FIRST_Q_TOKENS:
            return None

        growth_ratio = last_q_mean / max(first_q_mean, 1.0)

        score = 0.0
        if growth_ratio > self.GROWTH_THRESHOLD:
            score += min((growth_ratio - self.GROWTH_THRESHOLD) / 5.0, 0.5)
        if acceleration > 0:
            score += min(acceleration / max(np.mean(per_step), 1) * 10, 0.5)

        if score < self.MIN_SCORE:
            return None

        confidence = round(min(score, 0.9), 3)
        return CaftDiagnosis(
            caft_code="4.4",
            caft_category="resource",
            failure_name="token_explosion",
            severity=CaftSeverity.WARNING if confidence < 0.7 else CaftSeverity.CRITICAL,
            confidence=confidence,
            description=(
                f"Token usage grew {growth_ratio:.1f}x from first to last quarter "
                f"({first_q_mean:.0f} -> {last_q_mean:.0f} tokens/step)."
            ),
            evidence={
                "growth_ratio_last_vs_first_quarter": round(growth_ratio, 2),
                "tokens_per_step_slope": round(slope, 2),
                "acceleration": round(acceleration, 4),
                "first_quarter_avg_tokens": round(first_q_mean, 0),
                "last_quarter_avg_tokens": round(last_q_mean, 0),
                "total_tokens": int(per_step.sum()),
            },
            at_step=events[-1].step if events else 0,
            remediation="Summarize context; reduce output verbosity.",
        )


class StrategicMyopiaDetector:
    """Detect agent trapped in local optimization loop.

    CAFT 3.5: Strategic Myopia (Decision Making)

    Fires when the agent exhibits a combination of:
    - Repeated evaluation cycles with minimal metric improvement
    - Concentrated edits in the same few files
    - No architectural review or strategic replanning
    - Optional: ground truth mutation, phase stagnation

    Grounded in GEMS knowledge-based mistakes and HFACS cognitive tunneling.
    """
    name = "strategic_myopia"
    caft_code = "3.5"

    MIN_EVAL_CYCLES = 4
    MIN_EDITS = 8
    EDIT_CONCENTRATION_THRESHOLD = 0.80
    MIN_EVENTS = 40
    MAX_METRIC_DELTA = 0.10
    EXPLORATION_RATIO_THRESHOLD = 0.05

    _EVAL_PATTERN = re.compile(
        r"(?:pytest|python\s+scripts/|run_ablation|npm\s+test|make\s+test|"
        r"python\s+-m\s+pytest|jest|mocha|cargo\s+test|go\s+test)",
        re.IGNORECASE,
    )
    _METRIC_PATTERN = re.compile(
        r"(?:F1|accuracy|precision|recall|score|AUC|loss|error)"
        r"[\s:=]*(\d+\.?\d*)\s*%?"
        r"|(\d+\.?\d*)\s*%\s*(?:precision|recall|accuracy|F1)",
        re.IGNORECASE,
    )
    _EDIT_TOOLS = {"Edit", "Write", "edit", "write", "edit_file", "write_file"}
    _EVAL_PATH_MARKERS = {
        "annotation", "test", "fixture", "ground_truth", "expected",
        "golden", "splits",
    }
    _BROAD_READ_MARKERS = {
        "docs/", "readme", "architecture", "claude.md", "design",
        "doc/", "contributing",
    }

    _last_snapshot: Optional[dict] = None

    def _is_eval_run(self, e: TraceEvent) -> bool:
        if e.type != "tool_call" or e.tool not in ("Bash", "bash"):
            return False
        text = (e.goal_text or "") + (e.input_hash or "")
        return bool(self._EVAL_PATTERN.search(text))

    def _is_edit(self, e: TraceEvent) -> bool:
        return e.type == "tool_call" and e.tool in self._EDIT_TOOLS

    def _extract_edit_target(self, e: TraceEvent) -> str:
        for src in (e.goal_text, e.input_hash):
            if src:
                return src[:120]
        return e.tool or "unknown"

    def _is_eval_path(self, path: str) -> bool:
        lower = path.lower()
        return any(m in lower for m in self._EVAL_PATH_MARKERS)

    def _is_broad_read(self, e: TraceEvent) -> bool:
        text = ((e.goal_text or "") + (e.input_hash or "")).lower()
        return any(m in text for m in self._BROAD_READ_MARKERS)

    def _extract_metrics(self, e: TraceEvent) -> list[float]:
        text = (e.goal_text or "") + (e.error_message or "")
        vals = []
        for m in self._METRIC_PATTERN.finditer(text):
            v = m.group(1) or m.group(2)
            if v:
                try:
                    vals.append(float(v))
                except ValueError:
                    pass
        return vals

    def check(
        self,
        events: list[TraceEvent],
        hta_state: HTAState,
    ) -> Optional[CaftDiagnosis]:
        step = events[-1].step if events else 0
        self._last_snapshot = {
            "detector": self.name, "step": step,
            "fired": False, "confidence": 0.0,
            "gate_failed": None, "evidence_preview": {},
        }

        if len(events) < self.MIN_EVENTS:
            self._last_snapshot["gate_failed"] = "min_events"
            self._last_snapshot["evidence_preview"] = {
                "event_count": len(events), "min_required": self.MIN_EVENTS,
            }
            return None

        # --- Signal 1: Eval Cycle Detection ---
        eval_indices = [i for i, e in enumerate(events) if self._is_eval_run(e)]
        eval_cycles = max(len(eval_indices) - 1, 0)
        eval_script = ""
        if eval_indices:
            e0 = events[eval_indices[0]]
            eval_script = (e0.goal_text or e0.input_hash or "")[:80]
        steps_between = []
        for j in range(1, len(eval_indices)):
            steps_between.append(eval_indices[j] - eval_indices[j - 1])

        if eval_cycles < self.MIN_EVAL_CYCLES:
            self._last_snapshot["gate_failed"] = "min_eval_cycles"
            self._last_snapshot["evidence_preview"] = {
                "eval_cycles": eval_cycles,
                "min_required": self.MIN_EVAL_CYCLES,
            }
            return None

        # --- Signal 3: Edit Concentration ---
        edit_events = [e for e in events if self._is_edit(e)]
        total_edits = len(edit_events)

        if total_edits < self.MIN_EDITS:
            self._last_snapshot["gate_failed"] = "min_edits"
            self._last_snapshot["evidence_preview"] = {
                "total_edits": total_edits, "min_required": self.MIN_EDITS,
            }
            return None

        edit_targets = [self._extract_edit_target(e) for e in edit_events]
        target_counts = Counter(edit_targets)
        top_3 = target_counts.most_common(3)
        top_3_count = sum(c for _, c in top_3)
        concentration = top_3_count / total_edits
        top_files = [f for f, _ in top_3]

        if concentration < self.EDIT_CONCENTRATION_THRESHOLD:
            self._last_snapshot["gate_failed"] = "edit_concentration"
            self._last_snapshot["evidence_preview"] = {
                "concentration": round(concentration, 3),
                "threshold": self.EDIT_CONCENTRATION_THRESHOLD,
                "total_edits": total_edits,
            }
            return None

        # --- Signal 2: Metric Stagnation ---
        metric_stagnation = False
        metric_values: list[float] = []
        for idx in eval_indices:
            # Look at the 3 events after the eval run for output
            for e in events[idx:min(idx + 4, len(events))]:
                vals = self._extract_metrics(e)
                if vals:
                    metric_values.append(vals[0])
                    break
        max_metric_delta = 0.0
        if len(metric_values) >= 3:
            max_val = max(metric_values)
            min_val = min(metric_values)
            if max_val > 0:
                max_metric_delta = (max_val - min_val) / max_val
            metric_stagnation = max_metric_delta < self.MAX_METRIC_DELTA

        # --- Signal 4: Ground Truth Mutation ---
        eval_edit_files = [
            self._extract_edit_target(e) for e in edit_events
            if self._is_eval_path(self._extract_edit_target(e))
        ]
        ground_truth_mutation = len(eval_edit_files) >= 2

        # --- Signal 5: Architecture Blindness ---
        read_events = [
            e for e in events
            if e.type == "tool_call" and e.tool in ("Read", "read", "read_file")
        ]
        total_reads = len(read_events)
        broad_reads = sum(1 for e in read_events if self._is_broad_read(e))
        narrow_reads = total_reads - broad_reads
        architecture_blindness = (
            total_reads >= 10
            and broad_reads == 0
            and (narrow_reads / total_reads >= 0.85 if total_reads > 0 else False)
        )

        # --- Signal 6: Phase Stagnation ---
        phase_stagnation = False
        exploration_ratio = 1.0
        phase_dist_late: dict[str, int] = {}
        cutoff = len(events) // 4
        if cutoff > 0:
            late_events = events[cutoff:]
            if late_events:
                from agentdiag.hta import classify_event
                late_phases: list[Phase] = []
                for e in late_events:
                    ph, _ = classify_event(e)
                    late_phases.append(ph)
                for ph in late_phases:
                    key = ph.name
                    phase_dist_late[key] = phase_dist_late.get(key, 0) + 1
                exploration_count = phase_dist_late.get("GATHERING", 0) + phase_dist_late.get("PLANNING", 0)
                exploration_ratio = exploration_count / len(late_phases) if late_phases else 1.0
                phase_stagnation = exploration_ratio < self.EXPLORATION_RATIO_THRESHOLD

        # --- Firing Logic ---
        # Required: eval_cycles AND edit_concentration (both already passed gates)
        # Plus at least 2 of the 4 optional signals
        optional_signals = sum([
            metric_stagnation,
            ground_truth_mutation,
            architecture_blindness,
            phase_stagnation,
        ])

        if optional_signals < 2:
            self._last_snapshot["gate_failed"] = "insufficient_optional_signals"
            self._last_snapshot["evidence_preview"] = {
                "eval_cycles": eval_cycles,
                "concentration": round(concentration, 3),
                "metric_stagnation": metric_stagnation,
                "ground_truth_mutation": ground_truth_mutation,
                "architecture_blindness": architecture_blindness,
                "phase_stagnation": phase_stagnation,
                "optional_count": optional_signals,
                "min_required": 2,
            }
            return None

        # --- Confidence ---
        confidence = round(min(0.4 + optional_signals * 0.15, 0.85), 3)

        self._last_snapshot["fired"] = True
        self._last_snapshot["confidence"] = confidence
        self._last_snapshot["evidence_preview"] = {
            "eval_cycles": eval_cycles,
            "concentration": round(concentration, 3),
            "optional_signals": optional_signals,
        }

        return CaftDiagnosis(
            caft_code="3.5",
            caft_category="decision_making",
            failure_name="strategic_myopia",
            severity=CaftSeverity.WARNING,
            confidence=confidence,
            description=(
                f"Agent trapped in local optimization loop: {eval_cycles} eval cycles, "
                f"{concentration:.0%} of {total_edits} edits concentrated in {len(top_files)} files"
                f"{', metrics stagnant' if metric_stagnation else ''}"
                f"{', ground truth mutated' if ground_truth_mutation else ''}"
                f"{', no architectural review' if architecture_blindness else ''}"
                f"{', stuck in execute/verify loop' if phase_stagnation else ''}."
            ),
            evidence={
                "eval_cycles": eval_cycles,
                "eval_script": eval_script,
                "steps_between_evals": steps_between,
                "edit_concentration": round(concentration, 3),
                "top_files": top_files,
                "total_edits": total_edits,
                "metric_values": [round(v, 3) for v in metric_values],
                "max_metric_delta": round(max_metric_delta, 4),
                "metric_stagnation": metric_stagnation,
                "eval_files_edited": eval_edit_files[:10],
                "eval_edit_count": len(eval_edit_files),
                "ground_truth_mutation": ground_truth_mutation,
                "narrow_reads": narrow_reads,
                "broad_reads": broad_reads,
                "architecture_blindness": architecture_blindness,
                "exploration_ratio": round(exploration_ratio, 4),
                "phase_dist_late": phase_dist_late,
                "phase_stagnation": phase_stagnation,
                "optional_signals_present": optional_signals,
            },
            at_step=events[eval_indices[-1]].step if eval_indices else step,
            remediation=(
                "Agent appears stuck in a local optimization loop. Consider: "
                "(1) stepping back to review architecture/design docs, "
                "(2) questioning whether the current approach is fundamentally right, "
                "(3) checking if the evaluation metric is meaningful at this sample size."
            ),
            force_llm_review=True,
        )


class AnalysisParalysisDetector:
    """CAFT 3.4 -- Agent stuck in reasoning without executing.

    Detects N+ consecutive reasoning/planning steps with no tool_call.
    Migrated from batch DeadEndDetector.
    """
    name = "analysis_paralysis"
    caft_code = "3.4"

    THRESHOLD = 4

    def check(
        self,
        events: list[TraceEvent],
        hta_state: HTAState,
    ) -> Optional[CaftDiagnosis]:
        if len(events) < self.THRESHOLD:
            return None

        max_run = 0
        current_run = 0
        run_start = 0
        worst_start = 0

        for i, e in enumerate(events):
            if e.type in ("reasoning", "planning"):
                if current_run == 0:
                    run_start = i
                current_run += 1
                if current_run > max_run:
                    max_run = current_run
                    worst_start = run_start
            else:
                current_run = 0

        if max_run < self.THRESHOLD:
            return None

        confidence = round(min(max_run / (self.THRESHOLD * 3), 0.85), 3)
        return CaftDiagnosis(
            caft_code="3.4",
            caft_category="execution",
            failure_name="analysis_paralysis",
            severity=CaftSeverity.WARNING if confidence < 0.7 else CaftSeverity.CRITICAL,
            confidence=confidence,
            description=(
                f"Agent stuck: {max_run} consecutive reasoning/planning steps "
                f"starting at step {events[worst_start].step} with no execution."
            ),
            evidence={
                "max_consecutive_reasoning": max_run,
                "dead_end_start_step": events[worst_start].step,
                "dead_end_end_step": events[min(worst_start + max_run - 1, len(events) - 1)].step,
                "total_reasoning_events": sum(1 for e in events if e.type in ("reasoning", "planning")),
                "total_events": len(events),
            },
            at_step=events[min(worst_start + max_run - 1, len(events) - 1)].step,
            remediation="Stop reasoning and take a concrete action.",
        )


class RecoveryFailureDetector:
    """CAFT 4.3 -- Failed recovery attempts after errors.

    Detects agents stuck in retry loops without adapting strategy.
    Three progress filters prevent false positives from normal development:

    1. Post-recovery progress: if the agent made 5+ unique successful
       operations after the last failed recovery, it recovered.
    2. Stuck retry detection: same tool + same input after error = no
       state change (stronger signal than diverse retries).
    3. Productive session filter: long sessions (50+ events, 80%+ success)
       need stuck retries to fire — late errors in productive sessions
       are not systematic recovery failure.
    """
    name = "recovery_failure"
    caft_code = "4.3"

    RECOVERY_WINDOW = 3
    MIN_RECOVERY_PROGRESS = 5  # unique successes after last failure = recovered
    PRODUCTIVE_SESSION_MIN_EVENTS = 50
    PRODUCTIVE_SESSION_SUCCESS_RATE = 0.8

    def check(
        self,
        events: list[TraceEvent],
        hta_state: HTAState,
    ) -> Optional[CaftDiagnosis]:
        if len(events) < 4:
            return None

        error_indices = [i for i, e in enumerate(events) if not e.success]
        if not error_indices:
            return None
        if len(error_indices) < 3:
            return None

        failed_recoveries = 0
        total_recovery_attempts = 0
        worst_error_step = None
        worst_failures_after = 0
        last_failed_recovery_idx = None

        for idx in error_indices:
            window = events[idx + 1:idx + 1 + self.RECOVERY_WINDOW]
            if not window:
                continue

            total_recovery_attempts += 1
            failures_after = sum(1 for e in window if not e.success)

            if failures_after > 0:
                failed_recoveries += 1
                last_failed_recovery_idx = idx
                if failures_after > worst_failures_after:
                    worst_failures_after = failures_after
                    worst_error_step = events[idx].step

        if total_recovery_attempts == 0:
            return None

        failure_rate = failed_recoveries / total_recovery_attempts
        if failure_rate < 0.4:
            return None

        # --- Progress filter 1: post-recovery progress ---
        # If the agent made 5+ unique successful operations after the last
        # failed recovery, it recovered and this is not a systemic failure.
        if last_failed_recovery_idx is not None:
            remaining = events[last_failed_recovery_idx + 1:]
            successful_tool_calls = [
                e for e in remaining
                if e.success and e.type == "tool_call"
            ]
            # Prefer output_hash for uniqueness (precise: different outputs =
            # real progress). When no hashes available, require a longer
            # unbroken streak of successes as stronger evidence of recovery.
            hashes = [e.output_hash for e in successful_tool_calls if e.output_hash]
            if hashes:
                unique_successes = len(set(hashes))
                threshold = self.MIN_RECOVERY_PROGRESS  # 5
            else:
                unique_successes = len(successful_tool_calls)
                threshold = self.MIN_RECOVERY_PROGRESS + 3  # 8
            if unique_successes >= threshold:
                return None

        # --- Stuck retry detection ---
        # Count retries where the agent used the exact same tool + input
        # (no state change = truly stuck, not trying different approaches).
        error_tools = [events[i].tool for i in error_indices]
        retry_tools = []
        stuck_retries = 0
        for idx in error_indices:
            if idx + 1 < len(events):
                next_e = events[idx + 1]
                retry_tools.append(next_e.tool)
                if (next_e.tool == events[idx].tool
                        and next_e.input_hash is not None
                        and next_e.input_hash == events[idx].input_hash):
                    stuck_retries += 1
            else:
                retry_tools.append(None)
        same_tool_retries = sum(
            1 for e, r in zip(error_tools, retry_tools) if e == r and r is not None
        )

        # --- Progress filter 2: productive session filter ---
        # Long, mostly-successful sessions that hit errors at the end are
        # not systemic recovery failures — unless there's a stuck loop.
        # Count only tool_call events (exclude reasoning/user_input/planning).
        tool_call_events = [e for e in events if e.type == "tool_call"]
        if len(tool_call_events) >= self.PRODUCTIVE_SESSION_MIN_EVENTS:
            total_successes = sum(1 for e in tool_call_events if e.success)
            success_rate = total_successes / len(tool_call_events)
            if success_rate >= self.PRODUCTIVE_SESSION_SUCCESS_RATE and stuck_retries == 0:
                return None

        confidence = round(min(failure_rate, 1.0), 3)

        return CaftDiagnosis(
            caft_code="4.3",
            caft_category="resource",
            failure_name="recovery_failure",
            severity=CaftSeverity.WARNING if confidence < 0.7 else CaftSeverity.CRITICAL,
            confidence=confidence,
            description=(
                f"Failed to recover from errors {failed_recoveries}/{total_recovery_attempts} "
                f"times ({failure_rate:.0%}). "
                f"{'Stuck retries (same tool+input): ' + str(stuck_retries) + '. ' if stuck_retries > 0 else ''}"
                f"{'Retried same tool ' + str(same_tool_retries) + ' times. ' if same_tool_retries > 0 else ''}"
            ),
            evidence={
                "total_errors": len(error_indices),
                "failed_recoveries": failed_recoveries,
                "recovery_failure_rate": round(failure_rate, 3),
                "same_tool_retries": same_tool_retries,
                "stuck_retries": stuck_retries,
                "worst_error_step": worst_error_step,
                "worst_consecutive_failures_after": worst_failures_after,
            },
            at_step=worst_error_step or (events[-1].step if events else 0),
            remediation="Change strategy after errors; don't retry the same approach.",
        )


# ---------------------------------------------------------------------------
# Detector lists
# ---------------------------------------------------------------------------

# Rules-only mode: high-precision detectors that work without LLM confirmation.
# These have been validated against 20 ground-truth traces at 66.7% precision.
ALL_CAFT_DETECTORS: list = [
    # GT-validated detectors only (V12b pruning based on ground_truth_76.json)
    # Removed: StepRepetitionDetector (1 GT, fires every session, 5% precision)
    # Removed: ToolMisuseDetector (0 GT), TokenExplosionDetector (0 GT)
    ContextLossDetector(),                 # 5 GTs
    PrematureTerminationDetector(),        # 8 GTs
    StallDetector(),                       # 4 GTs
    ErrorCascadeDetector(),                # 3 GTs
    AnalysisParalysisDetector(),           # 1 GT
    RecoveryFailureDetector(),             # 2 GTs
]

# V13: FULL == CAFT (GoalDrift moved to Tier 2 session-end assessment).
# GoalDrift is detected via assess_session_end_sync() in run_ablation.py,
# not via rule-based detectors.
ALL_CAFT_DETECTORS_FULL: list = list(ALL_CAFT_DETECTORS)

# Failure types detected by Tier 2 (session-end LLM assessment).
# These are NOT detected by rule-based detectors in CAFT/FULL lists.
TIER_2_FAILURE_TYPES = {"premature_termination", "goal_drift"}

# Disabled detectors (0 GT positives — pure FP generators):
# - MissingVerificationDetector: fires on ~every session, 0 TPs
# - ToolThrashingDetector: 0 TPs
# - ReasoningActionMismatchDetector: keyword-brittle, 0 TPs
# - StrategicMyopiaDetector: 0 TPs
# - ToolMisuseDetector: 0 TPs
# - TokenExplosionDetector: 0 TPs
# Disabled (abysmal signal): StepRepetitionDetector (1 TP in 80 traces)


# Detectors where re-firing at a different step is meaningful.
# stall: a stall at step 10 rejected by LLM shouldn't block a real stall at step 100.
# error_cascade: a cascade at step 50 is different from one at step 200.
# All other detectors fire once and are permanently deduplicated.
_REFIREABLE = {"stall", "error_cascade"}


def run_caft_detectors(
    events: list[TraceEvent],
    hta_state: HTAState,
    detectors: list | None = None,
    seen: dict[str, int] | None = None,
) -> list[CaftDiagnosis]:
    """Run CAFT detectors and return new diagnoses.

    Args:
        events: Current event window.
        hta_state: Current HTA state.
        detectors: Override detector list (default: ALL_CAFT_DETECTORS).
        seen: Dict mapping failure_name → at_step for deduplication.
              Refireable detectors can fire again at a different step;
              all others are fire-once.

    Returns:
        List of new CaftDiagnosis objects (deduplicated).
    """
    if detectors is None:
        detectors = ALL_CAFT_DETECTORS
    if seen is None:
        seen = {}

    results = []
    for det in detectors:
        diagnosis = det.check(events, hta_state)
        if diagnosis is None:
            continue
        if det.name in seen:
            # Refireable detectors: skip only if same at_step
            # All others: skip permanently (fire-once)
            if det.name not in _REFIREABLE or seen[det.name] == diagnosis.at_step:
                continue
        seen[det.name] = diagnosis.at_step
        results.append(diagnosis)

    return results


def run_all_detectors(
    events: list[TraceEvent],
    hta_state: HTAState,
    seen: dict[str, int] | None = None,
) -> list[CaftDiagnosis]:
    """Run ALL detectors (enabled + disabled) via the registry.

    Args:
        events: Event list.
        hta_state: Current HTA state.
        seen: Dict mapping failure_name → at_step for deduplication.

    Returns:
        List of new CaftDiagnosis objects (deduplicated).
    """
    from agentdiag.caft.registry import detector_registry

    if seen is None:
        seen = {}

    detectors = detector_registry.get_all()
    return run_caft_detectors(events, hta_state, detectors=detectors, seen=seen)


def run_caft_detectors_traced(
    events: list[TraceEvent],
    hta_state: HTAState,
    detectors: list | None = None,
    seen: dict[str, int] | None = None,
) -> tuple[list[CaftDiagnosis], list[dict]]:
    """Run CAFT detectors and return diagnoses AND per-detector snapshots.

    This is the decision-trace-aware variant of run_caft_detectors().
    For each detector, it harvests the _last_snapshot attribute (if set)
    to capture intermediate computation even when the detector didn't fire.

    Detectors that don't implement _last_snapshot get a minimal snapshot
    (just fired/not-fired + confidence from the diagnosis).

    Args:
        events: Current event window.
        hta_state: Current HTA state.
        detectors: Override detector list.
        seen: Deduplication dict (same semantics as run_caft_detectors).

    Returns:
        (diagnoses, snapshots) — diagnoses is the same as run_caft_detectors;
        snapshots is a list of dicts, one per detector per call.
    """
    if detectors is None:
        detectors = ALL_CAFT_DETECTORS
    if seen is None:
        seen = {}

    step = events[-1].step if events else 0
    results = []
    snapshots = []

    for det in detectors:
        diagnosis = det.check(events, hta_state)

        # Harvest snapshot from instrumented detectors
        snap = getattr(det, "_last_snapshot", None)
        if snap is not None:
            snapshots.append(snap)
        else:
            # Minimal snapshot for non-instrumented detectors
            if diagnosis is not None:
                snapshots.append({
                    "detector": det.name,
                    "step": step,
                    "fired": True,
                    "confidence": round(diagnosis.confidence, 4),
                    "gate_failed": None,
                    "evidence_preview": {},
                })
            else:
                snapshots.append({
                    "detector": det.name,
                    "step": step,
                    "fired": False,
                    "confidence": 0.0,
                    "gate_failed": None,
                    "evidence_preview": {},
                })

        # Deduplication (same logic as run_caft_detectors)
        if diagnosis is None:
            continue
        if det.name in seen:
            if det.name not in _REFIREABLE or seen[det.name] == diagnosis.at_step:
                continue
        seen[det.name] = diagnosis.at_step
        results.append(diagnosis)

    return results, snapshots
