"""POC: Premature Termination — rules-only vs rules+LLM confirmation.

Validates the semantic confirmation approach on ONE detector:
  1. Current (strict) premature_termination has 100% precision but misses traces 15, 20
  2. Loose variant fires on any session with EXECUTING but poor delivery signals
  3. LLM confirmation layer filters loose FPs while keeping TPs

Compares three modes:
  - strict:      Current detector (high precision, low recall)
  - loose:       Lowered threshold (high recall, low precision)
  - loose+llm:   Loose + LLM confirmation (target: both high)

Usage:
    python scripts/poc_premature_termination.py                    # strict + loose comparison
    python scripts/poc_premature_termination.py --mode all         # all three modes
    python scripts/poc_premature_termination.py --mode loose+llm   # loose+LLM only
    python scripts/poc_premature_termination.py --dry-run           # show prompts without LLM
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

# Add package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agentdiag.adapters.claude_code import ClaudeCodeExtractor
from agentdiag.models import TraceEvent
from agentdiag.hta import HTAStateMachine, HTAState, Phase
from agentdiag.caft.base import CaftDiagnosis, CaftSeverity
from agentdiag.caft.detectors import (
    PrematureTerminationDetector,
    run_caft_detectors,
    _has_delegated_verification,
)


# ---------------------------------------------------------------------------
# Loose Premature Termination Detector
# ---------------------------------------------------------------------------

class LoosePrematureTerminationDetector:
    """Very liberal premature_termination — fires on any session that
    reaches EXECUTING but lacks strong delivery/completion signals.

    This is intentionally noisy. The LLM confirmation layer is expected
    to reject most firings, keeping only genuine premature terminations.

    Fires when ANY of these are true:
      A. (Original Mode 1) Delivering without verification
      B. (Original Mode 2) Plan-but-not-executed
      C. (NEW) Session has >=5 EXECUTING events + <3 DELIVERING events
         AND no clear completion signal (user "thanks", agent "done")
    """
    name = "premature_termination"
    caft_code = "5.4"

    _COMPLETION_SIGNAL = re.compile(
        r"(?:thanks?|thank you|that(?:'|&apos;)?s? (?:all|it|perfect|great|good)|"
        r"looks? good|done|perfect|great work|awesome|lgtm|ship it)",
        re.IGNORECASE,
    )

    _AGENT_DONE_SIGNAL = re.compile(
        r"(?:all (?:done|changes|tasks|modifications) (?:are |)(?:complete|done|made)|"
        r"(?:i(?:'|&apos;)?ve|i have) (?:completed|finished|done)|"
        r"everything (?:is|has been) (?:updated|fixed|implemented)|"
        r"summary of (?:all |)changes)",
        re.IGNORECASE,
    )

    _CANCEL_PATTERN = re.compile(
        r"(?:never ?mind|stop|cancel|forget it|don't|do not)",
        re.IGNORECASE,
    )

    def check(
        self,
        events: list[TraceEvent],
        hta_state: HTAState,
    ) -> Optional[CaftDiagnosis]:
        if len(events) < 5:
            return None

        exec_count = hta_state.phase_event_counts.get("executing", 0)
        delivering_count = hta_state.phase_event_counts.get("delivering", 0)

        # Check for user cancellation
        has_cancel = any(
            e.type == "user_input" and e.goal_text
            and self._CANCEL_PATTERN.search(e.goal_text)
            for e in events[-30:]
        )
        if has_cancel:
            return None

        # Mode 1: Original — delivering without verification (high confidence)
        if hta_state.current_phase == Phase.DELIVERING and delivering_count >= 3:
            has_verified = any(
                t.to_phase == Phase.VERIFYING for t in hta_state.transitions
            )
            has_executed = exec_count >= 3
            delegated = _has_delegated_verification(events)

            if has_executed and not has_verified and not delegated:
                return CaftDiagnosis(
                    caft_code="5.4",
                    caft_category="plan_structure",
                    failure_name="premature_termination",
                    severity=CaftSeverity.CRITICAL,
                    confidence=0.85,
                    description=(
                        f"Agent is delivering after {exec_count} "
                        f"execution steps but NEVER verified its work."
                    ),
                    evidence={
                        "mode": "skip_verification",
                        "execution_events": exec_count,
                        "verification_events": 0,
                        "delivering_events": delivering_count,
                    },
                    at_step=events[-1].step if events else 0,
                    remediation="Add verification step before delivery.",
                )

        # Mode C (LOOSE): Session has execution but weak delivery/completion
        if exec_count >= 5 and delivering_count < 3:
            # Check for completion signals from user
            has_user_completion = any(
                e.type == "user_input" and e.goal_text
                and self._COMPLETION_SIGNAL.search(e.goal_text)
                for e in events
            )

            # Check for agent completion signals
            has_agent_completion = any(
                e.type in ("reasoning", "planning", "assistant")
                and e.goal_text
                and self._AGENT_DONE_SIGNAL.search(e.goal_text)
                for e in events[-20:]
            )

            # Check for delegated verification
            has_delegated = _has_delegated_verification(events)

            if not has_user_completion and not has_agent_completion:
                # Also consider: did the session just... stop?
                # (No verification, no delivery, no user acknowledgment)
                verifying_count = hta_state.phase_event_counts.get("verifying", 0)

                confidence = 0.45  # Low base (loose rule)
                reasons = []

                if verifying_count == 0 and not has_delegated:
                    confidence += 0.15
                    reasons.append("no verification")

                if delivering_count == 0:
                    confidence += 0.10
                    reasons.append("no delivery")

                # Higher confidence if session has substantial writes
                write_tools = {
                    "write_file", "edit_file", "edit", "write",
                    "create_file", "notebookedit",
                }
                write_count = sum(
                    1 for e in events
                    if e.tool and e.tool.lower() in write_tools
                )
                if write_count >= 5:
                    confidence += 0.10
                    reasons.append(f"{write_count} writes")

                return CaftDiagnosis(
                    caft_code="5.4",
                    caft_category="plan_structure",
                    failure_name="premature_termination",
                    severity=CaftSeverity.WARNING,
                    confidence=min(confidence, 0.80),
                    description=(
                        f"Session has {exec_count} execution events and "
                        f"{delivering_count} delivery events with no "
                        f"completion signal. Issues: {', '.join(reasons)}."
                    ),
                    evidence={
                        "mode": "incomplete_session",
                        "execution_events": exec_count,
                        "delivering_events": delivering_count,
                        "verification_events": verifying_count,
                        "write_count": write_count,
                        "user_completion": has_user_completion,
                        "agent_completion": has_agent_completion,
                        "delegated_verification": has_delegated,
                        "reasons": reasons,
                    },
                    at_step=events[-1].step if events else 0,
                    remediation="Verify work and deliver results.",
                )

        return None


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def load_ground_truth():
    gt_path = Path(__file__).parent.parent / "annotations" / "ground_truth_20.json"
    with open(gt_path) as f:
        return json.load(f)


def run_detector_on_traces(detector, traces_data, extractor, all_sessions):
    """Run a single detector on all traces using MonitorEngine pattern.

    Processes events one-by-one through HTA + detector (same as
    MonitorEngine.push()) so detectors see every intermediate HTA state,
    not just the final state.
    """
    from agentdiag.monitor import MonitorEngine

    results = []
    for trace in traces_data:
        session_id = trace["session_id"]
        trace_num = trace["trace_num"]

        matching = [s for s in all_sessions if s.session_id.startswith(session_id)]
        if not matching:
            results.append({
                "trace_num": trace_num,
                "session_id": session_id,
                "status": "SKIP",
                "fired": False,
                "diagnosis": None,
            })
            continue

        try:
            events = extractor.parse_session(matching[0])
        except Exception as ex:
            results.append({
                "trace_num": trace_num,
                "session_id": session_id,
                "status": "ERROR",
                "fired": False,
                "diagnosis": None,
            })
            continue

        # Use MonitorEngine with only our detector
        engine = MonitorEngine(
            goal=f"Trace {trace_num}",
            detectors=[detector],
            confirm=False,
        )
        for event in events:
            engine.push(event)

        state = engine.state
        diagnoses = state.diagnoses
        fired = len(diagnoses) > 0

        results.append({
            "trace_num": trace_num,
            "session_id": session_id,
            "status": "OK",
            "fired": fired,
            "diagnosis": diagnoses[0] if diagnoses else None,
            "events": events,
            "hta_state": engine._hta.state,
        })

    return results


def _heuristic_confirm(events, hta_state, diag):
    """Heuristic LLM stand-in that uses trace-observable features.

    This simulates what a well-calibrated LLM would infer from the
    confirmation prompt. The prompt contains:
      - Agent goal (from first user_input)
      - Event window around the onset step
      - HTA phase distribution
      - Detector evidence

    The heuristic looks for signals a human/LLM reviewer would use:
      1. Does the agent produce a substantial deliverable?
      2. Are there user messages acknowledging completion?
      3. Does the session end mid-execution or at a natural boundary?
    """
    from agentdiag.caft.confirm import ConfirmationResult

    # Feature: last event type
    last_events = events[-10:] if len(events) >= 10 else events
    last_types = [e.type for e in last_events]
    last_tools = [e.tool for e in last_events if e.tool]

    # Feature: user messages in last 30 events
    user_msgs = [
        e for e in events[-30:]
        if e.type == "user_input" and e.goal_text
    ]

    # Feature: agent completion patterns
    completion_re = re.compile(
        r"(?:all (?:done|changes|tasks)|completed|finished|"
        r"summary of changes|let me know if|here(?:'|&apos;)?s the|"
        r"i(?:'|&apos;)?ve (?:implemented|created|updated|fixed|added))",
        re.IGNORECASE,
    )
    agent_said_done = any(
        e.type in ("reasoning", "planning", "assistant")
        and e.goal_text and completion_re.search(e.goal_text)
        for e in events[-20:]
    )

    # Feature: user said thanks/done
    user_thanks = re.compile(
        r"(?:thanks?|thank you|perfect|great|looks? good|awesome|lgtm)",
        re.IGNORECASE,
    )
    user_acknowledged = any(
        e.goal_text and user_thanks.search(e.goal_text)
        for e in user_msgs
    )

    # Feature: Did the agent do writes + verification?
    write_tools = {"write_file", "edit_file", "edit", "write", "create_file"}
    test_patterns = re.compile(
        r"(?:pytest|test|jest|cargo test|npm test)",
        re.IGNORECASE,
    )
    has_writes = any(e.tool and e.tool.lower() in write_tools for e in events)
    has_test_run = any(
        e.tool and "bash" in e.tool.lower() and e.goal_text
        and test_patterns.search(e.goal_text)
        for e in events
    )

    # Feature: How does the session end?
    # If the last few events are assistant reasoning with completion language,
    # it's likely a natural end. If they're tool_call with reads, it stopped abruptly.
    last_5_types = [e.type for e in (events[-5:] if len(events) >= 5 else events)]
    ends_with_reasoning = last_5_types[-1] in ("reasoning", "planning", "assistant") if last_5_types else False
    ends_with_tool = last_5_types[-1] == "tool_call" if last_5_types else False

    # Decision logic (what a good LLM would infer)
    if agent_said_done or user_acknowledged:
        return ConfirmationResult(
            confirmed=False,
            confidence=0.15,
            reasoning="Session has explicit completion signals (agent or user).",
            status="rejected",
        )

    if has_test_run and ends_with_reasoning:
        return ConfirmationResult(
            confirmed=False,
            confidence=0.20,
            reasoning="Agent ran tests and ended with reasoning — likely natural completion.",
            status="rejected",
        )

    # If session ends abruptly mid-tool-execution with many pending writes
    if ends_with_tool and has_writes:
        return ConfirmationResult(
            confirmed=True,
            confidence=0.80,
            reasoning="Session ends mid-execution with unverified writes — premature stop.",
            status="confirmed",
        )

    # Default: uncertain (most sessions end ambiguously)
    return ConfirmationResult(
        confirmed=False,
        confidence=0.35,
        reasoning="Insufficient evidence for premature termination — session may have completed naturally.",
        status="rejected",
    )


def run_llm_confirmation(results, dry_run=False, use_heuristic=False):
    """Run LLM confirmation on all fired candidates.

    Args:
        dry_run: Show prompts without calling LLM.
        use_heuristic: Use heuristic stand-in instead of real LLM.

    Returns updated results with confirmation status.
    """
    from agentdiag.caft.confirm import (
        build_confirmation_prompt,
        confirm_diagnosis_sync,
        is_llm_available,
        ConfirmationResult,
    )

    confirmed_results = []
    for r in results:
        r = dict(r)  # copy
        if not r["fired"] or r["diagnosis"] is None:
            r["llm_status"] = "not_fired"
            confirmed_results.append(r)
            continue

        diag = r["diagnosis"]
        events = r.get("events", [])
        hta_state = r.get("hta_state")

        if dry_run:
            # Show prompt but don't call LLM
            prompt = build_confirmation_prompt(diag, events, hta_state)
            r["llm_status"] = "dry_run"
            r["prompt"] = prompt
            r["prompt_len"] = len(prompt)
            confirmed_results.append(r)
            continue

        if use_heuristic:
            result = _heuristic_confirm(events, hta_state, diag)
            r["llm_status"] = result.status
            r["llm_confidence"] = result.confidence
            r["llm_reasoning"] = result.reasoning
            r["llm_confirmed"] = result.confirmed
            r["fired_after_llm"] = result.status != "rejected"
            confirmed_results.append(r)
            continue

        if not is_llm_available():
            r["llm_status"] = "unavailable"
            confirmed_results.append(r)
            continue

        try:
            result = confirm_diagnosis_sync(diag, events, hta_state)
            r["llm_status"] = result.status
            r["llm_confidence"] = result.confidence
            r["llm_reasoning"] = result.reasoning
            r["llm_confirmed"] = result.confirmed

            # If rejected, mark as not fired for metrics
            if result.status == "rejected":
                r["fired_after_llm"] = False
            else:
                r["fired_after_llm"] = True
        except Exception as ex:
            r["llm_status"] = f"error: {ex}"
            r["fired_after_llm"] = r["fired"]

        confirmed_results.append(r)

    return confirmed_results


def compute_metrics(results, gt_traces, use_llm_field=False):
    """Compute TP/FP/FN for premature_termination."""
    tp = fp = fn = 0
    details = []

    for r, gt in zip(results, gt_traces):
        is_real = "premature_termination" in gt.get("actual_failures", [])

        if use_llm_field:
            fired = r.get("fired_after_llm", r["fired"])
        else:
            fired = r["fired"]

        if fired and is_real:
            verdict = "TP"
            tp += 1
        elif fired and not is_real:
            verdict = "FP"
            fp += 1
        elif not fired and is_real:
            verdict = "FN"
            fn += 1
        else:
            verdict = "TN"

        details.append({
            "trace_num": r["trace_num"],
            "session_id": r.get("session_id", "?"),
            "verdict": verdict,
            "fired": fired,
            "is_real": is_real,
            "confidence": r["diagnosis"].confidence if r.get("diagnosis") else None,
        })

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "details": details,
    }


def print_results(mode_name, metrics, results=None, show_prompts=False):
    """Print formatted results for a single mode."""
    print(f"\n{'=' * 70}")
    print(f"  {mode_name}")
    print(f"{'=' * 70}")

    # Per-trace table
    print(f"\n  {'#':>3} {'Session':<12} {'Fired':>6} {'Verdict':>8} {'Conf':>6} {'Details'}")
    print(f"  {'─' * 65}")

    for d in metrics["details"]:
        fired_str = "YES" if d["fired"] else "—"
        conf_str = f"{d['confidence']:.2f}" if d["confidence"] else "—"
        detail = ""

        if results:
            r = next((r for r in results if r["trace_num"] == d["trace_num"]), None)
            if r and r.get("diagnosis"):
                mode = r["diagnosis"].evidence.get("mode", "?")
                detail = f"mode={mode}"
            if r and r.get("llm_status") and r["llm_status"] not in ("not_fired",):
                detail += f"  llm={r['llm_status']}"
                if r.get("llm_reasoning"):
                    detail += f" ({r['llm_reasoning'][:50]})"

        print(f"  {d['trace_num']:>3} {d['session_id'][:10]:<12} "
              f"{fired_str:>6} {d['verdict']:>8} {conf_str:>6}  {detail}")

        if show_prompts and results:
            r = next((r for r in results if r["trace_num"] == d["trace_num"]), None)
            if r and r.get("prompt"):
                print(f"\n      --- PROMPT ({r['prompt_len']} chars) ---")
                # Show first/last 200 chars
                p = r["prompt"]
                if len(p) > 500:
                    print(f"      {p[:250]}...")
                    print(f"      ...{p[-250:]}")
                else:
                    for line in p.split("\n"):
                        print(f"      {line}")
                print(f"      --- END PROMPT ---\n")

    # Summary
    m = metrics
    print(f"\n  {'─' * 65}")
    print(f"  TP={m['tp']}  FP={m['fp']}  FN={m['fn']}")
    print(f"  Precision: {m['precision']:.1%}")
    print(f"  Recall:    {m['recall']:.1%}")
    print(f"  F1:        {m['f1']:.1%}")


def main():
    parser = argparse.ArgumentParser(
        description="POC: premature_termination rules vs rules+LLM"
    )
    parser.add_argument(
        "--mode",
        choices=["strict", "loose", "loose+llm", "all"],
        default="all",
        help="Which mode(s) to evaluate",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show LLM prompts without calling LLM",
    )
    parser.add_argument(
        "--heuristic",
        action="store_true",
        help="Use heuristic LLM stand-in (no API key needed)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    args = parser.parse_args()

    gt = load_ground_truth()
    traces = gt["traces"]
    extractor = ClaudeCodeExtractor()

    traces_root = Path("~/.claude/projects").expanduser()
    all_sessions = extractor.discover(traces_root, min_lines=5)

    modes_to_run = []
    if args.mode in ("strict", "all"):
        modes_to_run.append("strict")
    if args.mode in ("loose", "all"):
        modes_to_run.append("loose")
    if args.mode in ("loose+llm", "all"):
        modes_to_run.append("loose+llm")
    # Always add oracle for comparison
    if args.mode == "all":
        modes_to_run.append("oracle")

    all_metrics = {}

    # Strict mode
    if "strict" in modes_to_run:
        print("\n  Running STRICT premature_termination detector...")
        strict_det = PrematureTerminationDetector()
        strict_results = run_detector_on_traces(strict_det, traces, extractor, all_sessions)
        strict_metrics = compute_metrics(strict_results, traces)
        all_metrics["strict"] = strict_metrics
        if not args.json:
            print_results("STRICT (current detector)", strict_metrics, strict_results)

    # Loose mode
    if "loose" in modes_to_run or "loose+llm" in modes_to_run:
        print("\n  Running LOOSE premature_termination detector...")
        loose_det = LoosePrematureTerminationDetector()
        loose_results = run_detector_on_traces(loose_det, traces, extractor, all_sessions)
        loose_metrics = compute_metrics(loose_results, traces)
        all_metrics["loose"] = loose_metrics
        if not args.json and "loose" in modes_to_run:
            print_results("LOOSE (lowered threshold)", loose_metrics, loose_results)

    # Loose + LLM mode
    if "loose+llm" in modes_to_run:
        print("\n  Running LLM confirmation on loose candidates...")
        llm_results = run_llm_confirmation(
            loose_results,
            dry_run=args.dry_run,
            use_heuristic=args.heuristic,
        )
        if not args.dry_run:
            llm_metrics = compute_metrics(llm_results, traces, use_llm_field=True)
            all_metrics["loose+llm"] = llm_metrics
            if not args.json:
                print_results(
                    "LOOSE + LLM CONFIRMATION",
                    llm_metrics,
                    llm_results,
                )
        else:
            if not args.json:
                print_results(
                    "LOOSE + LLM (DRY RUN — prompts only)",
                    loose_metrics,
                    llm_results,
                    show_prompts=True,
                )

    # Oracle mode: what a perfect LLM would achieve (loose + ground truth filter)
    if "oracle" in modes_to_run and "loose" in modes_to_run:
        oracle_results = []
        for r, gt in zip(loose_results, traces):
            r = dict(r)
            is_real = "premature_termination" in gt.get("actual_failures", [])
            r["fired_after_llm"] = r["fired"] and is_real
            r["llm_status"] = "oracle_confirmed" if is_real else "oracle_rejected"
            oracle_results.append(r)

        oracle_metrics = compute_metrics(oracle_results, traces, use_llm_field=True)
        all_metrics["oracle (ceiling)"] = oracle_metrics
        if not args.json:
            print_results("ORACLE (perfect LLM = ceiling)", oracle_metrics, oracle_results)

    # Comparison table
    if len(all_metrics) >= 2 and not args.json:
        print(f"\n{'=' * 70}")
        print(f"  COMPARISON: premature_termination detector modes")
        print(f"{'=' * 70}")
        print(f"  {'Mode':<25} {'TP':>4} {'FP':>4} {'FN':>4} {'Prec':>8} {'Rec':>8} {'F1':>8}")
        print(f"  {'─' * 60}")
        for mode_name, m in all_metrics.items():
            print(f"  {mode_name:<25} {m['tp']:>4} {m['fp']:>4} {m['fn']:>4} "
                  f"{m['precision']:>7.0%} {m['recall']:>7.0%} {m['f1']:>7.0%}")
        print(f"  {'─' * 60}")

        # Delta
        if "strict" in all_metrics and "loose" in all_metrics:
            s, l = all_metrics["strict"], all_metrics["loose"]
            print(f"\n  Loose vs Strict:")
            print(f"    Precision: {l['precision']:.0%} vs {s['precision']:.0%} "
                  f"(Δ {l['precision'] - s['precision']:+.0%})")
            print(f"    Recall:    {l['recall']:.0%} vs {s['recall']:.0%} "
                  f"(Δ {l['recall'] - s['recall']:+.0%})")

        if "loose+llm" in all_metrics and "loose" in all_metrics:
            l, llm = all_metrics["loose"], all_metrics["loose+llm"]
            print(f"\n  Loose+LLM vs Loose:")
            print(f"    Precision: {llm['precision']:.0%} vs {l['precision']:.0%} "
                  f"(Δ {llm['precision'] - l['precision']:+.0%})")
            print(f"    Recall:    {llm['recall']:.0%} vs {l['recall']:.0%} "
                  f"(Δ {llm['recall'] - l['recall']:+.0%})")

        print(f"{'=' * 70}")

    if args.json:
        # Serialize (drop non-serializable fields)
        output = {}
        for mode_name, m in all_metrics.items():
            output[mode_name] = {
                "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
            }
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
