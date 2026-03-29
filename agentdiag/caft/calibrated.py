"""Calibrated CAFT detectors that use normative baselines.

Wraps the 6 raw CAFT detectors with phase-specific percentile thresholds
learned from validation traces. Only fires when behavior genuinely deviates
from the learned "normal" baseline.

The key insight: normal agent behavior (reading multiple files, evolving
tool usage across phases) triggers raw detectors at 100%. With baselines,
we learn that e.g. 3 consecutive Reads in GATHERING is normal, so
step_repetition only fires when repetition exceeds the 95th percentile.

Usage:
    from agentdiag.caft.calibrated import make_calibrated_detectors

    profile = CalibrationProfile.load("baselines.json")
    detectors = make_calibrated_detectors(profile)
    # Use these instead of ALL_CAFT_DETECTORS
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

from agentdiag.models import TraceEvent
from agentdiag.hta import HTAState, Phase
from agentdiag.caft.base import CaftDiagnosis, CaftSeverity
from agentdiag.baselines import (
    CalibrationProfile,
    NormativePhaseModel,
    TransitionModel,
    ActionBaselineModel,
    extract_phase_segments,
    compute_phase_stats,
)


class CalibratedStepRepetition:
    """CAFT 2.2 — Step repetition, calibrated against phase norms.

    Only fires if the repetition rate in the current phase exceeds the
    95th percentile of the validation baseline for that phase.

    For example, in GATHERING phase, 3 consecutive Reads may be P50
    (completely normal), so we only flag when it reaches P95+.
    """
    name = "step_repetition"
    caft_code = "2.2"

    def __init__(self, phase_model: NormativePhaseModel):
        self._phase_model = phase_model

    # Minimum events before repetition detection is meaningful.
    # Early in a session, consecutive reads are normal startup behavior.
    MIN_EVENTS = 30
    # Minimum consecutive run length to be considered a real repetition
    # (as opposed to normal 2-3 consecutive reads)
    MIN_RUN_LENGTH = 4

    def check(
        self,
        events: list[TraceEvent],
        hta_state: HTAState,
    ) -> Optional[CaftDiagnosis]:
        if len(events) < self.MIN_EVENTS:
            return None

        phase = hta_state.current_phase.label

        # Compute current repetition rate (recent window)
        window = events[-30:]  # look at last 30 events
        tools = [e.tool or e.type for e in window if e.type == "tool_call"]
        if len(tools) < 5:
            return None

        consecutive = sum(
            1 for i in range(1, len(tools))
            if tools[i] == tools[i - 1]
        )
        rep_rate = consecutive / len(tools)

        # Check against phase-specific baseline
        if not self._phase_model.is_anomalous(phase, "repetition_rate", rep_rate):
            return None

        # Find the worst repeated tool
        max_run = 1
        current_run = 1
        worst_tool = tools[-1]
        for i in range(1, len(tools)):
            if tools[i] == tools[i - 1]:
                current_run += 1
                if current_run > max_run:
                    max_run = current_run
                    worst_tool = tools[i]
            else:
                current_run = 1

        # Require a minimum run length — 2-3 consecutive identical calls
        # is normal (e.g. Read Read Read); 4+ is suspicious
        if max_run < self.MIN_RUN_LENGTH:
            return None

        dist = self._phase_model.get_distribution(phase, "repetition_rate")
        threshold = dist.p95 if dist else 0.0

        return CaftDiagnosis(
            caft_code="2.2",
            caft_category="memory",
            failure_name="step_repetition",
            severity=CaftSeverity.WARNING if rep_rate < threshold * 1.5 else CaftSeverity.CRITICAL,
            confidence=min(rep_rate / max(threshold * 2, 0.01), 1.0),
            description=(
                f"Repetition rate {rep_rate:.0%} in {phase} phase exceeds "
                f"baseline P95 ({threshold:.0%}). "
                f"Worst: {worst_tool} repeated {max_run}x."
            ),
            evidence={
                "repetition_rate": round(rep_rate, 4),
                "baseline_p95": round(threshold, 4),
                "phase": phase,
                "worst_tool": worst_tool,
                "worst_run_length": max_run,
            },
            at_step=events[-1].step,
            remediation="Check if previous results were captured; add deduplication.",
        )


class CalibratedGoalDrift:
    """CAFT 2.4 — Goal drift, calibrated against transition and action norms.

    Only fires if:
    1. Anomalous transitions are present (P < min_probability), AND
    2. Tool usage within the current phase diverges from baseline, AND
    3. The drift is not simply normal lifecycle evolution (gather→execute)

    Normal lifecycle evolution (Read → Edit → Read) is expected and
    baselined; only genuine tangential drift triggers.

    Key insight: half-shift (tool distribution change between halves)
    is ALWAYS high in normal sessions because first half = gathering,
    second half = executing. This is lifecycle, not drift. We instead
    check if WITHIN-PHASE tool usage diverges from the baseline.
    """
    name = "goal_drift"
    caft_code = "2.4"

    MIN_EVENTS = 15

    def __init__(
        self,
        transition_model: TransitionModel,
        action_model: ActionBaselineModel,
    ):
        self._transition_model = transition_model
        self._action_model = action_model

    def check(
        self,
        events: list[TraceEvent],
        hta_state: HTAState,
    ) -> Optional[CaftDiagnosis]:
        if len(events) < self.MIN_EVENTS:
            return None

        # 1. Count anomalous transitions (never-before-seen phase jumps)
        anomalous_transitions = 0
        for t in hta_state.transitions:
            if self._transition_model.is_anomalous(
                t.from_phase.label, t.to_phase.label
            ):
                anomalous_transitions += 1

        # 2. Check within-phase tool divergence (NOT half-shift)
        phase = hta_state.current_phase.label
        recent_tools = [
            e.tool for e in events[-30:]
            if e.type == "tool_call" and e.tool
        ]
        divergence = self._action_model.compute_tool_divergence(
            phase, recent_tools
        )

        # 3. Check for novel tools in current phase
        novel = self._action_model.get_novel_tools(phase, recent_tools)
        novel_ratio = len(novel) / max(len(recent_tools), 1)

        # 4. Excessive regressions beyond baseline
        # Normal sessions have 2-5 regressions; >8 suggests confusion
        excessive_regressions = hta_state.regression_count > 8

        # Composite score — requires multiple signals to fire
        drift_score = 0.0

        # Anomalous transitions are strong evidence
        if anomalous_transitions >= 3:
            drift_score += 0.4
        elif anomalous_transitions >= 1:
            drift_score += 0.2

        # High tool divergence from phase baseline
        if divergence > 1.0:
            drift_score += 0.3
        elif divergence > 0.5:
            drift_score += 0.15

        # Novel tools (never seen in this phase during training)
        if novel_ratio > 0.3:
            drift_score += 0.2

        # Excessive regressions
        if excessive_regressions:
            drift_score += 0.2

        if drift_score < 0.5:
            return None

        new_tools_in_second_half = set()
        mid = len(events) // 2
        tools_first = {e.tool for e in events[:mid] if e.tool}
        tools_second = {e.tool for e in events[mid:] if e.tool}
        new_tools_in_second_half = tools_second - tools_first

        return CaftDiagnosis(
            caft_code="2.4",
            caft_category="memory",
            failure_name="goal_drift",
            severity=CaftSeverity.WARNING if drift_score < 0.7 else CaftSeverity.CRITICAL,
            confidence=round(drift_score, 3),
            description=(
                f"Drift detected: tool divergence={divergence:.2f} in {phase}, "
                f"anomalous transitions={anomalous_transitions}, "
                f"novel tools={novel}. "
                f"Regressions: {hta_state.regression_count}."
            ),
            evidence={
                "drift_score": round(drift_score, 3),
                "tool_divergence": round(divergence, 4),
                "anomalous_transitions": anomalous_transitions,
                "novel_tools": novel,
                "novel_ratio": round(novel_ratio, 4),
                "phase_regressions": hta_state.regression_count,
                "new_tools_second_half": list(new_tools_in_second_half),
            },
            at_step=events[-1].step,
            remediation="Re-inject original goal; check if agent is still pursuing stated objective.",
        )


class CalibratedMissingVerification:
    """CAFT 5.3 — Missing verification, calibrated.

    In real sessions, the executing phase_event_count accumulates across
    ALL executing segments (a session may enter executing 10+ times).
    The baseline P95 is per-SEGMENT, so we compare segment-level stats.

    Only fires if:
    1. Total executing events > 3x the P95 segment length (many segments,
       none verified), AND
    2. There are actual file writes, AND
    3. Zero verifying events anywhere in the session
    """
    name = "missing_verification"
    caft_code = "5.3"

    def __init__(self, phase_model: NormativePhaseModel):
        self._phase_model = phase_model

    def check(
        self,
        events: list[TraceEvent],
        hta_state: HTAState,
    ) -> Optional[CaftDiagnosis]:
        exec_count = hta_state.phase_event_counts.get("executing", 0)
        verify_count = hta_state.phase_event_counts.get("verifying", 0)

        # Need substantial execution events
        if exec_count < 10:
            return None

        # If there are ANY verification events, don't fire
        if verify_count > 0:
            return None

        # No verification at all — check if we have writes
        write_tools = {"write", "edit", "write_file", "edit_file", "create_file", "bash"}
        has_writes = any(
            e.tool and any(t in e.tool.lower() for t in write_tools)
            for e in events
        )

        if not has_writes:
            return None

        # The baseline P95 is for a SINGLE executing segment (~23 events).
        # A real session enters executing many times. Only flag if total
        # executing events exceed 3x the per-segment P95 — meaning the
        # agent has done multiple execution segments without ever verifying.
        dist = self._phase_model.get_distribution("executing", "step_count")
        if dist and dist.n >= 3:
            # Count how many executing segments we've completed
            exec_segments = sum(
                1 for n in hta_state.completed_nodes
                if n.phase == Phase.EXECUTING
            )
            # Allow 1 segment per regression cycle without verification
            # Only flag if many segments accumulated with zero verification
            segment_threshold = max(dist.p95 * 3, 50)
            if exec_count <= segment_threshold:
                return None

        return CaftDiagnosis(
            caft_code="5.3",
            caft_category="plan_structure",
            failure_name="missing_verification",
            severity=CaftSeverity.WARNING,
            confidence=min(exec_count / 100.0, 0.9),
            description=(
                f"Agent has {exec_count} execution events with file writes "
                f"but 0 verification events across the entire session."
            ),
            evidence={
                "execution_events": exec_count,
                "verification_events": 0,
                "total_events": hta_state.total_events,
            },
            at_step=events[-1].step if events else 0,
            remediation="Run tests after code changes; add review step before delivery.",
        )


class CalibratedContextLoss:
    """CAFT 2.1 — Context loss, calibrated.

    Uses the action baseline to determine if re-reading a resource
    is normal for the current phase. In GATHERING, multiple reads
    of different files are expected; context_loss only fires if
    the same resource is re-read with substantial intervening work.
    """
    name = "context_loss"
    caft_code = "2.1"

    def __init__(self, action_model: ActionBaselineModel):
        self._action_model = action_model

    def check(
        self,
        events: list[TraceEvent],
        hta_state: HTAState,
    ) -> Optional[CaftDiagnosis]:
        if len(events) < 6:
            return None

        # Track read-like operations by output_hash (same content = same resource)
        read_tools = {"read_file", "read", "cat", "head", "search_docs", "fetch"}
        reads: dict[str, list[int]] = {}

        for e in events:
            if e.tool and any(t in e.tool.lower() for t in read_tools):
                key = e.output_hash or e.tool
                if key not in reads:
                    reads[key] = []
                reads[key].append(e.step)

        # Find re-reads with significant intervening work
        for key, steps in reads.items():
            if len(steps) < 2:
                continue

            first, last = steps[0], steps[-1]
            intervening = [
                e for e in events
                if first < e.step < last
                and e.type == "tool_call"
                and e.tool
                and not any(t in e.tool.lower() for t in read_tools)
            ]

            # Require MORE intervening work than raw detector (3 instead of 2)
            # because normal development often involves re-reading after a few edits
            if len(intervening) < 3:
                continue

            # Check if re-reads of this frequency are normal
            phase = hta_state.current_phase.label
            read_tool = next(
                (e.tool for e in events if e.step == steps[0] and e.tool),
                "read"
            )
            baseline_freq = self._action_model.get_tool_frequency(phase, read_tool)

            # If reads are very common in this phase (>30% of actions), higher
            # re-read threshold
            min_intervening = 3 if baseline_freq < 0.3 else 5

            if len(intervening) < min_intervening:
                continue

            return CaftDiagnosis(
                caft_code="2.1",
                caft_category="memory",
                failure_name="context_loss",
                severity=CaftSeverity.WARNING,
                confidence=min(len(steps) / 4.0, 1.0),
                description=(
                    f"Re-read resource (hash={key[:8]}) at steps "
                    f"{steps} with {len(intervening)} operations between — "
                    f"exceeds baseline re-read pattern for {phase}."
                ),
                evidence={
                    "resource_hash": key,
                    "read_steps": steps,
                    "read_count": len(steps),
                    "intervening_operations": len(intervening),
                    "phase": phase,
                },
                at_step=last,
                remediation="Summarize key findings after reads; use external scratchpad.",
            )

        return None


def make_calibrated_detectors(
    profile: CalibrationProfile,
) -> list:
    """Create the full detector list, using calibrated versions where available.

    Returns all 13 detectors (9 active + 4 disabled). Detectors with
    calibrated versions use them; the rest are passed through unchanged.

    Calibrated versions exist for:
      - step_repetition (CalibratedStepRepetition)
      - context_loss (CalibratedContextLoss)
      - missing_verification (CalibratedMissingVerification)
      - goal_drift (CalibratedGoalDrift)

    Passed through unchanged (no calibration benefit):
      - premature_termination, reasoning_action_mismatch
      - tool_misuse, stall, error_cascade, token_explosion
      - analysis_paralysis, recovery_failure, tool_thrashing
    """
    from agentdiag.caft.detectors import (
        PrematureTerminationDetector,
        ReasoningActionMismatchDetector,
        ToolMisuseDetector,
        StallDetector,
        ErrorCascadeDetector,
        TokenExplosionDetector,
        AnalysisParalysisDetector,
        RecoveryFailureDetector,
        ToolThrashingDetector,
    )

    return [
        # Calibrated versions
        CalibratedStepRepetition(profile.phase_model),
        CalibratedContextLoss(profile.action_model),
        CalibratedMissingVerification(profile.phase_model),
        CalibratedGoalDrift(profile.transition_model, profile.action_model),
        # Unchanged (phase-transition / pair-based)
        PrematureTerminationDetector(),
        ReasoningActionMismatchDetector(),
        # Migrated batch detectors (no calibrated variant)
        ToolMisuseDetector(),
        StallDetector(),
        ErrorCascadeDetector(),
        TokenExplosionDetector(),
        AnalysisParalysisDetector(),
        RecoveryFailureDetector(),
        ToolThrashingDetector(),
    ]
