#!/usr/bin/env python3
"""Run LLM confirmation layer on 46 detector candidates from the 30 new traces.

This is the V4 end-to-end validation: do LLM confirmations filter false positives
while preserving true positives?

Usage:
    # Source the API key first
    eval "$(grep ANTHROPIC_API_KEY ~/.zshrc)"
    python scripts/run_confirmation.py
"""

import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentdiag.adapters.claude_code import ClaudeCodeExtractor, SessionInfo
from agentdiag.hta import HTAStateMachine
from agentdiag.caft.detectors import ALL_CAFT_DETECTORS_FULL, run_caft_detectors
from agentdiag.caft.base import CaftDiagnosis, CaftSeverity
from agentdiag.caft.confirm import confirm_diagnosis_sync, is_llm_available
from agentdiag.annotation_store import AnnotationLedger

SUMMARIES_PATH = Path(__file__).resolve().parent.parent / "annotations" / "trace_summaries_30.json"
LEDGER_PATH = Path(__file__).resolve().parent.parent / "annotations" / "annotation_ledger_80.jsonl"
RESULTS_PATH = Path(__file__).resolve().parent.parent / "annotations" / "confirmation_results.json"


def parse_trace(jsonl_path: str):
    """Parse a trace and return (events, hta_state, diagnoses)."""
    extractor = ClaudeCodeExtractor()
    session = SessionInfo(
        session_id=Path(jsonl_path).stem,
        path=Path(jsonl_path),
        project_dir=Path(jsonl_path).parent.name,
    )

    events = extractor.parse_session(session)
    if not events:
        return [], None, []

    hta_machine = HTAStateMachine()
    hta_state = hta_machine.state
    for event in events:
        hta_state = hta_machine.push(event)

    diagnoses = run_caft_detectors(
        events, hta_state,
        detectors=ALL_CAFT_DETECTORS_FULL,
        seen={},
    )

    return events, hta_state, diagnoses


def main():
    if not is_llm_available():
        print("ERROR: No LLM available. Set ANTHROPIC_API_KEY.")
        sys.exit(1)

    print("LLM provider: anthropic (Sonnet 4.5)")
    print()

    summaries = json.loads(SUMMARIES_PATH.read_text())
    ledger = AnnotationLedger(LEDGER_PATH)

    # Get human annotations for comparison
    new_sids = set(s["session_id"] for s in summaries)
    human_records = {
        r.session_id: r for r in ledger.get_all()
        if r.annotator_type == "human" and r.session_id in new_sids
    }

    # Build set of human-confirmed failure codes per session
    human_failures = {}
    for sid, r in human_records.items():
        codes = set()
        if r.has_failure:
            codes.add(r.primary_caft_code)
            for sc in (r.secondary_caft_codes or []):
                codes.add(sc)
        human_failures[sid] = codes

    results = []
    total_confirmed = 0
    total_rejected = 0
    total_uncertain = 0
    tp = 0  # confirmed AND human says failure with matching code
    fp = 0  # confirmed BUT human says no failure or different code
    tn = 0  # rejected AND human says no failure
    fn = 0  # rejected BUT human says failure

    traces_with_candidates = [
        s for s in summaries
        if s["detector_results"].get("detectors_fired")
    ]
    total_candidates = sum(
        len(s["detector_results"]["detectors_fired"])
        for s in traces_with_candidates
    )
    print(f"Traces with candidates: {len(traces_with_candidates)}")
    print(f"Total candidates to confirm: {total_candidates}")
    print()

    candidate_num = 0
    start_time = time.time()

    for summary in traces_with_candidates:
        sid = summary["session_id"]
        det_fired = summary["detector_results"]["detectors_fired"]

        if not det_fired:
            continue

        # Parse trace to get events + HTA state
        events, hta_state, diagnoses = parse_trace(summary["path"])
        if not events or hta_state is None:
            print(f"  SKIP {sid[:8]}: no events parsed")
            continue

        human_codes = human_failures.get(sid, set())
        has_human_failure = bool(human_codes)

        print(f"[{sid[:8]}] {len(det_fired)} candidates, "
              f"human={'FAIL(' + ','.join(human_codes) + ')' if has_human_failure else 'CLEAN'}")

        for det_hit in det_fired:
            candidate_num += 1

            # Find matching diagnosis from re-parsed trace
            matching = [
                d for d in diagnoses
                if d.failure_name == det_hit["failure_name"]
            ]

            if not matching:
                # Detector may have different output on re-parse; construct from saved data
                sev_map = {"info": CaftSeverity.INFO, "warning": CaftSeverity.WARNING, "critical": CaftSeverity.CRITICAL}
                candidate = CaftDiagnosis(
                    caft_code=det_hit["caft_code"],
                    caft_category="",
                    failure_name=det_hit["failure_name"],
                    severity=sev_map.get(det_hit["severity"], CaftSeverity.WARNING),
                    confidence=det_hit["confidence"],
                    description=det_hit["description"],
                    evidence={},
                    at_step=det_hit["at_step"],
                    remediation="",
                )
            else:
                candidate = matching[0]

            # Run LLM confirmation
            try:
                result = confirm_diagnosis_sync(candidate, events, hta_state)
            except Exception as e:
                print(f"  ERROR confirming {candidate.failure_name}: {e}")
                result = None
                continue

            is_real = det_hit["caft_code"] in human_codes

            if result.status == "confirmed":
                total_confirmed += 1
                if is_real:
                    tp += 1
                else:
                    fp += 1
            elif result.status == "rejected":
                total_rejected += 1
                if is_real:
                    fn += 1
                else:
                    tn += 1
            else:
                total_uncertain += 1
                # Count uncertain as neither TP nor FP for now

            status_emoji = {
                "confirmed": "CONFIRMED",
                "rejected": "REJECTED",
                "uncertain": "UNCERTAIN",
            }

            print(f"  [{candidate_num}/{total_candidates}] "
                  f"{candidate.failure_name:25s} "
                  f"{status_emoji[result.status]:10s} "
                  f"conf={result.confidence:.2f} "
                  f"human={'TP' if is_real else 'FP'} "
                  f"| {result.reasoning[:80]}")

            results.append({
                "session_id": sid,
                "failure_name": candidate.failure_name,
                "caft_code": det_hit["caft_code"],
                "detector_confidence": det_hit["confidence"],
                "llm_confirmed": result.confirmed,
                "llm_confidence": result.confidence,
                "llm_status": result.status,
                "llm_reasoning": result.reasoning,
                "human_is_real": is_real,
                "human_codes": list(human_codes),
            })

            # Brief pause to avoid rate limiting
            time.sleep(0.5)

    elapsed = time.time() - start_time

    # Save results
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print()
    print("=" * 70)
    print("LLM CONFIRMATION RESULTS")
    print("=" * 70)
    print(f"\nTotal candidates:  {total_candidates}")
    print(f"Confirmed:         {total_confirmed}")
    print(f"Rejected:          {total_rejected}")
    print(f"Uncertain:         {total_uncertain}")
    print(f"Time:              {elapsed:.1f}s ({elapsed/max(total_candidates,1):.1f}s per candidate)")

    print(f"\n── Confusion Matrix (LLM vs Human) ──")
    print(f"  True Positives:  {tp} (LLM confirmed, human agrees)")
    print(f"  False Positives: {fp} (LLM confirmed, human disagrees)")
    print(f"  True Negatives:  {tn} (LLM rejected, human agrees)")
    print(f"  False Negatives: {fn} (LLM rejected, human says failure)")

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 0.001)

    print(f"\n── Metrics (confirmed-only) ──")
    print(f"  Precision: {tp}/{tp+fp} = {precision:.1%}")
    print(f"  Recall:    {tp}/{tp+fn} = {recall:.1%}")
    print(f"  F1:        {f1:.1%}")

    pre_llm_precision = (tp + fn) / max(total_candidates, 1)
    print(f"\n── Precision Improvement ──")
    print(f"  Pre-LLM (detectors only): {tp+fn}/{total_candidates} = {pre_llm_precision:.1%}")
    print(f"  Post-LLM (confirmed):     {tp}/{tp+fp} = {precision:.1%}")
    improvement = precision - pre_llm_precision
    print(f"  Improvement:              {'+' if improvement >= 0 else ''}{improvement:.1%}")

    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
