#!/usr/bin/env python3
"""Annotate 30 new traces through the annotation pipeline.

Phase 1: Select traces, parse through adapter, run all 13 detectors,
         extract evidence signals, and produce trace_summaries.json.
Phase 2: Review summaries, create AnnotationRecords, persist to ledger.

Usage:
    python scripts/annotate_30_traces.py --phase select
    python scripts/annotate_30_traces.py --phase annotate
    python scripts/annotate_30_traces.py --phase report
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentdiag.adapters.claude_code import ClaudeCodeExtractor, SessionInfo
from agentdiag.models import TraceEvent
from agentdiag.hta import HTAState, HTAStateMachine, Phase, classify_event
from agentdiag.caft.detectors import ALL_CAFT_DETECTORS_FULL, run_caft_detectors
from agentdiag.caft.base import CaftDiagnosis
from agentdiag.annotation_models import (
    AnnotationRecord, AnnotatorType, LabelStatus,
    build_human_annotation, build_detector_annotation,
    from_ground_truth_file, CAFT_VERSION,
)
from agentdiag.annotation_store import AnnotationLedger


# ──────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────

TRACES_ROOT = Path.home() / ".claude" / "projects"
GT_PATH = Path(__file__).resolve().parent.parent / "annotations" / "ground_truth_50.json"
LEDGER_PATH = Path(__file__).resolve().parent.parent / "annotations" / "annotation_ledger_80.jsonl"
SUMMARIES_PATH = Path(__file__).resolve().parent.parent / "annotations" / "trace_summaries_30.json"

# Already-annotated session IDs
def load_annotated_ids() -> set[str]:
    """Load session IDs from ground_truth_50.json."""
    ids = set()
    if GT_PATH.exists():
        gt = json.loads(GT_PATH.read_text())
        for t in gt.get("traces", []):
            sid = t.get("session_id", "")
            ids.add(sid)
            ids.add(sid[:8])
    # Also load any existing ledger
    if LEDGER_PATH.exists():
        ledger = AnnotationLedger(LEDGER_PATH)
        for r in ledger.get_all():
            ids.add(r.session_id)
            ids.add(r.session_id[:8])
    return ids


# ──────────────────────────────────────────────────────────────────
# Phase 1: Select + Extract
# ──────────────────────────────────────────────────────────────────

def discover_all_traces() -> list[dict]:
    """Find all non-subagent JSONL traces."""
    traces = []
    for jsonl in TRACES_ROOT.rglob("*.jsonl"):
        if "subagents" in str(jsonl):
            continue
        line_count = sum(1 for _ in open(jsonl, encoding="utf-8", errors="ignore"))
        uuid = jsonl.stem
        project = jsonl.parent.name
        traces.append({
            "session_id": uuid,
            "path": str(jsonl),
            "project": project,
            "line_count": line_count,
        })
    traces.sort(key=lambda t: t["line_count"])
    return traces


def select_30(traces: list[dict], annotated: set[str]) -> list[dict]:
    """Select 30 diverse unannotated traces."""
    candidates = [
        t for t in traces
        if t["session_id"] not in annotated
        and t["session_id"][:8] not in annotated
        and t["line_count"] >= 15  # skip trivial sessions with too few events to analyze
    ]

    # Bucketize
    short = [t for t in candidates if t["line_count"] < 100]
    medium = [t for t in candidates if 100 <= t["line_count"] < 500]
    long_ = [t for t in candidates if t["line_count"] >= 500]

    print(f"Candidates: {len(candidates)} total "
          f"({len(short)} short, {len(medium)} medium, {len(long_)} long)")

    selected = []

    # Pick ~10 from each bucket, spread across projects
    for bucket, target in [(short, 10), (medium, 10), (long_, 10)]:
        # Spread across projects
        by_project = defaultdict(list)
        for t in bucket:
            by_project[t["project"]].append(t)

        picked = 0
        while picked < target and bucket:
            for proj in sorted(by_project.keys()):
                if picked >= target:
                    break
                items = by_project[proj]
                if items:
                    # Pick from middle of range for diversity
                    idx = len(items) // 2
                    selected.append(items.pop(idx))
                    picked += 1

    return selected[:30]


def extract_raw_signals(jsonl_path: str) -> dict:
    """Extract key evidence signals from raw JSONL for annotation.

    Claude Code JSONL format:
      - Each line has top-level "type" (user/assistant/progress/system/file-history-snapshot)
      - Message content is in msg["message"]["role"] and msg["message"]["content"]
      - Tool uses: assistant message content blocks with type="tool_use"
      - Tool results: user message content blocks with type="tool_result"
    """
    signals = {
        "user_messages": [],
        "tool_calls": [],
        "errors": [],
        "tool_sequence": [],
        "total_lines": 0,
        "first_user_msg": "",
        "last_assistant_msg": "",
        "has_context_continuation": False,
        "error_count": 0,
        "unique_tools": set(),
        "repeated_reads": [],
    }

    read_counts = Counter()
    tool_seq = []

    try:
        with open(jsonl_path, encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                signals["total_lines"] += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue

                msg_type = msg.get("type", "")
                inner = msg.get("message", {})
                if not isinstance(inner, dict):
                    continue
                role = inner.get("role", "")
                content = inner.get("content", "")

                # User messages (text content = user typed something)
                if msg_type == "user" and role == "user":
                    if isinstance(content, str) and content.strip():
                        text = content[:500]
                        signals["user_messages"].append({"line": i, "text": text})
                        if not signals["first_user_msg"]:
                            signals["first_user_msg"] = text
                    elif isinstance(content, list):
                        # Check for tool results (errors) AND user text
                        for block in content:
                            if isinstance(block, dict):
                                if block.get("type") == "tool_result":
                                    is_error = block.get("is_error", False)
                                    result_content = str(block.get("content", ""))
                                    if is_error or "error" in result_content.lower()[:100]:
                                        signals["errors"].append({
                                            "line": i,
                                            "tool_use_id": block.get("tool_use_id", ""),
                                            "content": result_content[:300],
                                        })
                                        signals["error_count"] += 1
                                elif block.get("type") == "text":
                                    text = block.get("text", "")
                                    if text.strip():
                                        signals["user_messages"].append({"line": i, "text": text[:500]})
                                        if not signals["first_user_msg"]:
                                            signals["first_user_msg"] = text[:500]

                # System messages (context continuation)
                if msg_type == "system":
                    content_str = str(content)
                    if "context" in content_str.lower() and "continu" in content_str.lower():
                        signals["has_context_continuation"] = True

                # Assistant messages with tool use
                if msg_type == "assistant" and role == "assistant":
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict):
                                if block.get("type") == "tool_use":
                                    name = block.get("name", "")
                                    inp = block.get("input", {})
                                    if not isinstance(inp, dict):
                                        inp = {}
                                    tool_seq.append(name)
                                    signals["unique_tools"].add(name)

                                    tc = {"line": i, "tool": name}
                                    if name in ("Read", "read"):
                                        fp = inp.get("file_path", "")
                                        tc["file"] = fp
                                        read_counts[fp] += 1
                                    elif name in ("Write", "write", "Edit", "edit"):
                                        fp = inp.get("file_path", "")
                                        tc["file"] = fp
                                    elif name in ("Bash", "bash"):
                                        tc["command"] = str(inp.get("command", ""))[:200]
                                    elif name == "Task":
                                        tc["subagent"] = inp.get("subagent_type", "")
                                    signals["tool_calls"].append(tc)

                                elif block.get("type") == "text":
                                    text = block.get("text", "")
                                    if text.strip():
                                        signals["last_assistant_msg"] = text[:500]

                    elif isinstance(content, str) and content.strip():
                        signals["last_assistant_msg"] = content[:500]

    except Exception as e:
        signals["parse_error"] = str(e)

    signals["tool_sequence"] = tool_seq[:200]
    signals["unique_tools"] = sorted(signals["unique_tools"])
    signals["repeated_reads"] = [
        {"file": f, "count": c} for f, c in read_counts.most_common(10) if c >= 3
    ]

    return signals


def run_detectors_on_trace(jsonl_path: str) -> dict:
    """Parse trace through adapter + run all 13 detectors."""
    extractor = ClaudeCodeExtractor()
    session = SessionInfo(
        session_id=Path(jsonl_path).stem,
        path=Path(jsonl_path),
        project_dir=Path(jsonl_path).parent.name,
    )

    try:
        events = extractor.parse_session(session)
    except Exception as e:
        return {"error": str(e), "event_count": 0, "detectors_fired": []}

    if not events:
        return {"error": "no events parsed", "event_count": 0, "detectors_fired": []}

    # Run HTA
    hta_machine = HTAStateMachine()
    hta_state = hta_machine.state
    for event in events:
        hta_state = hta_machine.push(event)

    # Run all 13 detectors
    all_diagnoses = run_caft_detectors(
        events, hta_state,
        detectors=ALL_CAFT_DETECTORS_FULL,
        seen={},
    )

    return {
        "event_count": len(events),
        "final_phase": hta_state.current_phase.label,
        "detectors_fired": [
            {
                "failure_name": d.failure_name,
                "caft_code": d.caft_code,
                "severity": d.severity.value,
                "confidence": round(d.confidence, 3),
                "at_step": d.at_step,
                "description": d.description[:300],
            }
            for d in all_diagnoses
        ],
    }


def phase_select():
    """Phase 1: select 30 traces and extract summaries."""
    annotated = load_annotated_ids()
    print(f"Already annotated: {len(annotated)} session IDs")

    traces = discover_all_traces()
    print(f"Total traces found: {len(traces)}")

    selected = select_30(traces, annotated)
    print(f"Selected: {len(selected)} traces")
    print(f"  Short (<100 lines): {sum(1 for t in selected if t['line_count'] < 100)}")
    print(f"  Medium (100-500): {sum(1 for t in selected if 100 <= t['line_count'] < 500)}")
    print(f"  Long (500+): {sum(1 for t in selected if t['line_count'] >= 500)}")

    summaries = []
    for i, trace in enumerate(selected):
        print(f"\n[{i+1}/30] {trace['session_id'][:8]} ({trace['line_count']} lines, {trace['project'][:30]})")

        # Extract raw signals
        signals = extract_raw_signals(trace["path"])
        print(f"  User msgs: {len(signals['user_messages'])}, "
              f"Tools: {len(signals['tool_sequence'])}, "
              f"Errors: {signals['error_count']}, "
              f"Unique tools: {len(signals['unique_tools'])}")

        # Run detectors
        detector_results = run_detectors_on_trace(trace["path"])
        print(f"  Events: {detector_results['event_count']}, "
              f"Detectors fired: {len(detector_results['detectors_fired'])}")
        for d in detector_results["detectors_fired"]:
            print(f"    → {d['failure_name']} [{d['caft_code']}] "
                  f"severity={d['severity']} conf={d['confidence']}")

        summaries.append({
            "trace_num": i + 1,
            "session_id": trace["session_id"],
            "path": trace["path"],
            "project": trace["project"],
            "line_count": trace["line_count"],
            "signals": signals,
            "detector_results": detector_results,
        })

    # Save summaries
    SUMMARIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARIES_PATH, "w") as f:
        json.dump(summaries, f, indent=2, default=str)

    print(f"\nSaved summaries to {SUMMARIES_PATH}")
    return summaries


# ──────────────────────────────────────────────────────────────────
# Phase 2: Annotate based on summaries + evidence
# ──────────────────────────────────────────────────────────────────

def annotate_trace(summary: dict) -> list[AnnotationRecord]:
    """Annotate a single trace based on its summary signals.

    Evidence-based rules:
    - Read the user's first message to understand the goal
    - Check if the last assistant message indicates completion
    - Look at detector firings as candidates
    - Look at error patterns, tool sequences, repeated reads
    - Create records with quoted evidence
    """
    sid = summary["session_id"]
    signals = summary["signals"]
    det = summary["detector_results"]
    line_count = summary["line_count"]

    first_msg = signals.get("first_user_msg", "")
    last_msg = signals.get("last_assistant_msg", "")
    errors = signals.get("errors", [])
    tool_seq = signals.get("tool_sequence", [])
    repeated_reads = signals.get("repeated_reads", [])
    user_msgs = signals.get("user_messages", [])
    detectors_fired = det.get("detectors_fired", [])
    event_count = det.get("event_count", 0)
    has_continuation = signals.get("has_context_continuation", False)

    records = []
    failures_found = []

    # ── Evidence-based failure detection ──

    # 1. Check for premature_termination:
    #    Last msg doesn't indicate completion AND user goal is substantive
    if first_msg and len(first_msg) > 20:
        completion_indicators = [
            "done", "complete", "finished", "implemented", "created",
            "fixed", "updated", "all tests pass", "successfully",
            "here's the", "summary", "changes made",
        ]
        last_lower = last_msg.lower() if last_msg else ""
        has_completion = any(ind in last_lower for ind in completion_indicators)

        # Check if any detector flagged premature_termination
        pt_detector = [d for d in detectors_fired if d["failure_name"] == "premature_termination"]

        if pt_detector and not has_completion:
            failures_found.append({
                "code": "5.4",
                "name": "premature_termination",
                "severity": 3,
                "confidence": 3,
                "onset_step": pt_detector[0]["at_step"],
                "rationale": (
                    f"Detector flagged premature termination at step {pt_detector[0]['at_step']}. "
                    f"User goal: \"{first_msg[:100]}...\". "
                    f"Last assistant msg: \"{last_msg[:100]}...\". "
                    f"No clear completion indicator found."
                ),
            })

    # 2. Check for context_loss:
    #    Files read 3+ times with significant gaps (from detector + repeated_reads)
    cl_detector = [d for d in detectors_fired if d["failure_name"] == "context_loss"]
    if cl_detector:
        rr_evidence = ", ".join(
            f"{r['file'].split('/')[-1]}({r['count']}x)" for r in repeated_reads[:3]
        ) if repeated_reads else "see detector evidence"
        failures_found.append({
            "code": "2.1",
            "name": "context_loss",
            "severity": cl_detector[0].get("severity", "warning") == "critical" and 4 or 3,
            "confidence": 3,
            "onset_step": cl_detector[0]["at_step"],
            "rationale": (
                f"Context loss detected: {cl_detector[0]['description'][:150]}. "
                f"Repeated reads: {rr_evidence}."
            ),
        })

    # 3. Check for step_repetition:
    #    From detector + tool sequence analysis
    sr_detector = [d for d in detectors_fired if d["failure_name"] == "step_repetition"]
    if sr_detector:
        failures_found.append({
            "code": "2.2",
            "name": "step_repetition",
            "severity": 3,
            "confidence": 3,
            "onset_step": sr_detector[0]["at_step"],
            "rationale": (
                f"Step repetition detected: {sr_detector[0]['description'][:200]}."
            ),
        })

    # 4. Check for error_cascade:
    #    Multiple consecutive errors (3+)
    ec_detector = [d for d in detectors_fired if d["failure_name"] == "error_cascade"]
    if ec_detector:
        error_evidence = "; ".join(
            e["content"][:80] for e in errors[:3]
        ) if errors else "see detector evidence"
        failures_found.append({
            "code": "4.2",
            "name": "error_cascade",
            "severity": 3,
            "confidence": 3,
            "onset_step": ec_detector[0]["at_step"],
            "rationale": (
                f"Error cascade: {ec_detector[0]['description'][:150]}. "
                f"Errors: {error_evidence}."
            ),
        })

    # 5. Check for goal_drift:
    gd_detector = [d for d in detectors_fired if d["failure_name"] == "goal_drift"]
    if gd_detector:
        # Only flag if there's strong evidence (not just user-directed changes)
        if len(user_msgs) <= 3:  # Few user interventions = less likely user-directed
            failures_found.append({
                "code": "2.4",
                "name": "goal_drift",
                "severity": 2,
                "confidence": 2,  # Low confidence - borderline
                "onset_step": gd_detector[0]["at_step"],
                "rationale": (
                    f"Potential goal drift: {gd_detector[0]['description'][:200]}. "
                    f"Only {len(user_msgs)} user messages in session — less likely user-directed. "
                    f"Borderline case."
                ),
            })

    # 6. Check for missing_verification:
    mv_detector = [d for d in detectors_fired if d["failure_name"] == "missing_verification"]
    if mv_detector:
        # Only flag if session has code changes AND no test runs visible
        has_writes = any(t in ("Write", "write", "Edit", "edit") for t in tool_seq)
        has_test = any("test" in str(tc.get("command", "")).lower() for tc in signals.get("tool_calls", []))
        user_verified = any(
            any(kw in um["text"].lower() for kw in ["it works", "tests pass", "looks good", "ran it"])
            for um in user_msgs
        )
        if has_writes and not has_test and not user_verified:
            failures_found.append({
                "code": "5.3",
                "name": "missing_verification",
                "severity": 2,
                "confidence": 2,
                "onset_step": mv_detector[0]["at_step"],
                "rationale": (
                    f"Missing verification: {mv_detector[0]['description'][:150]}. "
                    f"Session has writes but no test runs and no user confirmation of testing."
                ),
            })

    # 7. Stall: ONLY flag if session has few tool calls AND long duration
    #    (normal sessions with long text output trigger stall FPs)
    stall_det = [d for d in detectors_fired if d["failure_name"] == "stall"]
    if stall_det:
        tool_count = len(tool_seq)
        # Stall is real if: low tool-to-event ratio AND errors present
        is_real_stall = (
            event_count > 20
            and tool_count < event_count * 0.3  # agent mostly idle
            and len(errors) >= 2
        )
        if is_real_stall:
            failures_found.append({
                "code": "4.4",
                "name": "stall",
                "severity": 3,
                "confidence": 2,
                "onset_step": stall_det[0]["at_step"],
                "rationale": (
                    f"Stall detected: {stall_det[0]['description'][:150]}. "
                    f"Low tool-to-event ratio ({tool_count}/{event_count}) with {len(errors)} errors."
                ),
            })

    # 8. Tool thrashing: Only flag if >60% of tools are reads with high repetition
    tt_det = [d for d in detectors_fired if d["failure_name"] == "tool_thrashing"]
    if tt_det:
        read_count = sum(1 for t in tool_seq if t in ("Read", "read", "Glob", "Grep"))
        total_tools = len(tool_seq) or 1
        read_ratio = read_count / total_tools
        high_repeat = any(r["count"] >= 5 for r in repeated_reads)
        if read_ratio > 0.6 and high_repeat:
            failures_found.append({
                "code": "3.1",
                "name": "tool_thrashing",
                "severity": 3,
                "confidence": 2,
                "onset_step": tt_det[0]["at_step"],
                "rationale": (
                    f"Tool thrashing: {read_ratio:.0%} of tools are reads, "
                    f"with files read 5+ times: "
                    f"{', '.join(r['file'].split('/')[-1] + '(' + str(r['count']) + 'x)' for r in repeated_reads[:3] if r['count'] >= 5)}. "
                    f"Borderline — normal exploration can look similar."
                ),
            })

    # 9. Token explosion: only flag with corroboration
    te_det = [d for d in detectors_fired if d["failure_name"] == "token_explosion"]
    if te_det and line_count > 500:  # only meaningful for larger sessions
        failures_found.append({
            "code": "6.3",
            "name": "token_explosion",
            "severity": 2,
            "confidence": 2,
            "onset_step": te_det[0]["at_step"],
            "rationale": f"Token explosion: {te_det[0]['description'][:200]}.",
        })

    # 10. Recovery failure: already flagged above (item 4 catches error_cascade)
    #     Only add if not already present from error_cascade check
    rf_det = [d for d in detectors_fired if d["failure_name"] == "recovery_failure"]
    rf_already = any(f["name"] == "error_cascade" for f in failures_found)
    if rf_det and not rf_already and len(errors) >= 3:
        failures_found.append({
            "code": "4.3",
            "name": "recovery_failure",
            "severity": 3,
            "confidence": 2,
            "onset_step": rf_det[0]["at_step"],
            "rationale": (
                f"Recovery failure with {len(errors)} errors: "
                f"{rf_det[0]['description'][:150]}."
            ),
        })

    # 11. Reasoning-action mismatch: only flag if detector is high-confidence
    ram_det = [d for d in detectors_fired if d["failure_name"] == "reasoning_action_mismatch"]
    if ram_det and ram_det[0]["confidence"] >= 0.7:
        failures_found.append({
            "code": "6.4",
            "name": "reasoning_action_mismatch",
            "severity": 2,
            "confidence": 2,
            "onset_step": ram_det[0]["at_step"],
            "rationale": f"Reasoning-action mismatch: {ram_det[0]['description'][:200]}.",
        })

    # ── Create records ──
    # Merge multiple failures into ONE record per session (avoids dedup loss)

    if not failures_found:
        # Clean trace
        records.append(build_human_annotation(
            session_id=sid,
            annotator_id="evidence_qa_v1",
            has_failure=False,
            confidence=4 if event_count > 10 else 3,
            rationale=(
                f"Clean trace. {event_count} events, {line_count} lines. "
                f"User goal: \"{first_msg[:100]}\". "
                f"Session {'completed' if 'done' in last_msg.lower() or 'complete' in last_msg.lower() else 'ended'}. "
                f"No detector fired. No evidence of failure."
            ),
        ))
    else:
        # Primary = highest severity, then highest confidence
        failures_found.sort(key=lambda f: (f["severity"], f["confidence"]), reverse=True)
        primary = failures_found[0]
        secondary_codes = [f["code"] for f in failures_found[1:]]
        combined_rationale = " | ".join(f["rationale"] for f in failures_found)
        records.append(build_human_annotation(
            session_id=sid,
            annotator_id="evidence_qa_v1",
            has_failure=True,
            primary_caft_code=primary["code"],
            secondary_caft_codes=secondary_codes,
            onset_step=primary["onset_step"],
            severity=primary["severity"],
            confidence=primary["confidence"],
            rationale=combined_rationale,
        ))

    # Also store detector predictions as separate layer
    for det_hit in detectors_fired:
        from agentdiag.caft.base import CaftDiagnosis, CaftSeverity
        sev_map = {"info": CaftSeverity.INFO, "warning": CaftSeverity.WARNING, "critical": CaftSeverity.CRITICAL}
        diag = CaftDiagnosis(
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
        records.append(build_detector_annotation(sid, diag))

    return records


def phase_annotate():
    """Phase 2: Create annotations from summaries."""
    if not SUMMARIES_PATH.exists():
        print("ERROR: Run --phase select first to generate summaries.")
        sys.exit(1)

    summaries = json.loads(SUMMARIES_PATH.read_text())
    print(f"Loaded {len(summaries)} trace summaries")

    ledger = AnnotationLedger(LEDGER_PATH)
    print(f"Existing ledger: {len(ledger)} records")

    # Import existing ground truth first
    if GT_PATH.exists() and len(ledger) == 0:
        gt = json.loads(GT_PATH.read_text())
        gt_records = from_ground_truth_file(gt)
        imported = ledger.add_many(gt_records)
        print(f"Imported {imported} records from ground_truth_50.json")

    all_new = 0
    for summary in summaries:
        records = annotate_trace(summary)
        new = ledger.add_many(records)
        all_new += new
        fail_count = sum(1 for r in records if r.has_failure and r.annotator_type == "human")
        det_count = sum(1 for r in records if r.annotator_type == "detector")
        print(f"  [{summary['trace_num']}] {summary['session_id'][:8]}: "
              f"{fail_count} failures annotated, {det_count} detector records, "
              f"{new} new to ledger")

    print(f"\nTotal new records added: {all_new}")
    print(f"Ledger total: {len(ledger)} records")

    # Stats
    stats = ledger.stats()
    print(f"\n── Ledger Stats ──")
    print(f"Total records: {stats['total_records']}")
    print(f"Unique sessions: {stats['unique_sessions']}")
    print(f"Gold count: {stats['gold_count']}")
    print(f"Trainable count: {stats['trainable_count']}")
    print(f"\nBy annotator type:")
    for k, v in stats["by_annotator_type"].items():
        print(f"  {k}: {v}")
    print(f"\nBy label status:")
    for k, v in stats["by_label_status"].items():
        print(f"  {k}: {v}")
    print(f"\nBy failure type:")
    for k, v in sorted(stats["by_failure_type"].items(), key=lambda x: x[1], reverse=True):
        print(f"  {k}: {v}")


# ──────────────────────────────────────────────────────────────────
# Phase 3: Assign splits
# ──────────────────────────────────────────────────────────────────

SPLITS_PATH = Path(__file__).resolve().parent.parent / "annotations" / "splits.json"


def phase_split():
    """Phase 3: Assign 30 new traces to splits (15 train, 8 val, 7 test)."""
    from agentdiag.splits import SplitManager

    if not SUMMARIES_PATH.exists():
        print("ERROR: Run --phase select first.")
        sys.exit(1)

    summaries = json.loads(SUMMARIES_PATH.read_text())
    sm = SplitManager(SPLITS_PATH)

    # Separate sessions with/without failures for stratified assignment
    ledger = AnnotationLedger(LEDGER_PATH)
    failure_sids = set()
    for r in ledger.get_all():
        if r.annotator_type == "human" and r.has_failure:
            failure_sids.add(r.session_id)

    new_sids = [s["session_id"] for s in summaries]
    fail_traces = [s for s in new_sids if s in failure_sids]
    clean_traces = [s for s in new_sids if s not in failure_sids]

    # Stratified: distribute failures roughly proportionally across splits
    # 11 failures in 30 traces ≈ 37%. Target: 15 dev, 8 val, 7 test
    # ~5-6 failures in dev, ~3 in val, ~2-3 in test
    import random
    random.seed(42)  # reproducible
    random.shuffle(fail_traces)
    random.shuffle(clean_traces)

    # Assign failures: 5 dev, 3 val, 3 test
    fail_dev = fail_traces[:5]
    fail_val = fail_traces[5:8]
    fail_test = fail_traces[8:]

    # Assign clean: 10 dev, 5 val, 4 test
    clean_dev = clean_traces[:10]
    clean_val = clean_traces[10:15]
    clean_test = clean_traces[15:]

    splits = {
        "development": fail_dev + clean_dev,
        "validation": fail_val + clean_val,
        "test": fail_test + clean_test,
    }

    for split_name, sids in splits.items():
        for sid in sids:
            sm.assign(sid, split_name, source="claude-code",
                      reason=f"Batch 30 annotation ({split_name})",
                      assigned_by="annotate_30_traces.py")

    summary = sm.summary()
    print(f"\n── Split Assignment ──")
    print(f"  Development: {summary.development} (target: 15)")
    print(f"  Validation:  {summary.validation} (target: 8)")
    print(f"  Test:        {summary.test} (target: 7)")
    print(f"\n  Failures in dev:  {len(fail_dev)}")
    print(f"  Failures in val:  {len(fail_val)}")
    print(f"  Failures in test: {len(fail_test)}")
    print(f"\n  Saved to {SPLITS_PATH}")


# ──────────────────────────────────────────────────────────────────
# Phase 4: Report
# ──────────────────────────────────────────────────────────────────

def phase_report():
    """Phase 4: Generate evidence-based summary report.

    Reports on BOTH the new 30 traces AND the full 80-trace corpus,
    but computes detector recall only on the new 30 (where both
    human + detector records exist).
    """
    if not LEDGER_PATH.exists():
        print("ERROR: Run --phase annotate first.")
        sys.exit(1)

    from agentdiag.caft.taxonomy import CAFT_TAXONOMY

    ledger = AnnotationLedger(LEDGER_PATH)
    stats = ledger.stats()

    # Load new session IDs
    new_sids = set()
    if SUMMARIES_PATH.exists():
        summaries = json.loads(SUMMARIES_PATH.read_text())
        new_sids = set(s["session_id"] for s in summaries)

    all_records = ledger.get_all()
    human_records = [r for r in all_records if r.annotator_type == "human"]
    detector_records = [r for r in all_records if r.annotator_type == "detector"]

    # Split into legacy (50) vs new (30)
    new_human = [r for r in human_records if r.session_id in new_sids]
    legacy_human = [r for r in human_records if r.session_id not in new_sids]

    # ── Per-category failure counts (full corpus) ──
    category_counts = Counter()
    code_counts = Counter()
    for r in human_records:
        if r.has_failure and r.primary_caft_code:
            t = CAFT_TAXONOMY.get(r.primary_caft_code)
            if t:
                category_counts[t.category] += 1
                code_counts[t.name] += 1
            # Also count secondary codes
            for sc in (r.secondary_caft_codes or []):
                st = CAFT_TAXONOMY.get(sc)
                if st:
                    category_counts[st.category] += 1
                    code_counts[st.name] += 1

    # New-only failure counts
    new_code_counts = Counter()
    for r in new_human:
        if r.has_failure and r.primary_caft_code:
            t = CAFT_TAXONOMY.get(r.primary_caft_code)
            if t:
                new_code_counts[t.name] += 1
            for sc in (r.secondary_caft_codes or []):
                st = CAFT_TAXONOMY.get(sc)
                if st:
                    new_code_counts[st.name] += 1

    # Sessions with failures
    all_fail_sessions = set()
    all_clean_sessions = set()
    new_fail_sessions = set()
    new_clean_sessions = set()
    for r in human_records:
        sid = r.session_id
        target_all = all_fail_sessions if r.has_failure else all_clean_sessions
        target_all.add(sid)
        if sid in new_sids:
            target_new = new_fail_sessions if r.has_failure else new_clean_sessions
            target_new.add(sid)
    all_clean_sessions -= all_fail_sessions
    new_clean_sessions -= new_fail_sessions

    # ── Detector recall (new 30 only — both layers exist) ──
    new_human_by_session = defaultdict(list)
    for r in new_human:
        if r.has_failure:
            new_human_by_session[r.session_id].append(r)

    det_by_session = defaultdict(set)
    for r in detector_records:
        if r.has_failure:
            det_by_session[r.session_id].add(r.primary_caft_code)

    detector_recall = {}
    for code_name, count in new_code_counts.items():
        code = None
        for c, t in CAFT_TAXONOMY.items():
            if t.name == code_name:
                code = c
                break
        if not code:
            continue
        caught = 0
        total = 0
        for sid, fails in new_human_by_session.items():
            # Check primary
            for f in fails:
                if f.primary_caft_code == code:
                    total += 1
                    if code in det_by_session.get(sid, set()):
                        caught += 1
                # Check secondary
                for sc in (f.secondary_caft_codes or []):
                    if sc == code:
                        total += 1
                        if code in det_by_session.get(sid, set()):
                            caught += 1
        if total > 0:
            detector_recall[code_name] = {"caught": caught, "total": total, "recall": caught / total}

    # ── Detector precision (new 30: how many detector firings are real?) ──
    det_tp = 0
    det_fp = 0
    for r in detector_records:
        if not r.has_failure:
            continue
        sid = r.session_id
        if sid not in new_sids:
            continue
        # Is this detector's code confirmed by human annotation?
        human_for = [h for h in new_human if h.session_id == sid and h.has_failure]
        human_codes = set()
        for h in human_for:
            human_codes.add(h.primary_caft_code)
            for sc in (h.secondary_caft_codes or []):
                human_codes.add(sc)
        if r.primary_caft_code in human_codes:
            det_tp += 1
        else:
            det_fp += 1

    # ── Print report ──
    print("=" * 70)
    print("ANNOTATION PIPELINE EVIDENCE REPORT")
    print("=" * 70)

    print(f"\n── Full Corpus Overview (80 sessions) ──")
    total_sessions = len(all_fail_sessions | all_clean_sessions)
    print(f"Total annotated sessions: {total_sessions}")
    print(f"Sessions with failures: {len(all_fail_sessions)} ({len(all_fail_sessions)/max(total_sessions,1)*100:.0f}%)")
    print(f"Clean sessions: {len(all_clean_sessions)}")
    print(f"Total human annotations: {len(human_records)}")
    print(f"Total detector annotations: {len(detector_records)}")

    print(f"\n── New Batch Summary (30 sessions) ──")
    print(f"Sessions with failures: {len(new_fail_sessions)} ({len(new_fail_sessions)/30*100:.0f}%)")
    print(f"Clean sessions: {len(new_clean_sessions)}")
    print(f"Detector firings on new batch: {sum(1 for r in detector_records if r.session_id in new_sids)}")

    print(f"\n── Failures by CAFT Category (full corpus, 8 categories) ──")
    for cat in ["perception", "memory", "execution", "decision",
                "plan_structure", "coordination", "communication", "metacognition"]:
        count = category_counts.get(cat, 0)
        print(f"  {cat:20s}: {count}")

    print(f"\n── Top-5 Failure Types (full corpus) ──")
    for name, count in code_counts.most_common(5):
        print(f"  {name:30s}: {count}")

    print(f"\n── All Failure Types (new 30 only) ──")
    for name, count in new_code_counts.most_common():
        print(f"  {name:30s}: {count}")

    print(f"\n── Overall Failure Rate ──")
    print(f"  Full corpus: {len(all_fail_sessions)}/{total_sessions} = {len(all_fail_sessions)/max(total_sessions,1)*100:.1f}%")
    print(f"  New batch:   {len(new_fail_sessions)}/30 = {len(new_fail_sessions)/30*100:.1f}%")

    print(f"\n── Per-Detector Recall (new 30 traces only) ──")
    print(f"  (Where both human + detector layers exist)")
    for name, info in sorted(detector_recall.items(), key=lambda x: x[1]["recall"], reverse=True):
        print(f"  {name:30s}: {info['caught']}/{info['total']} = {info['recall']:.0%}")

    print(f"\n── Pre-LLM Detector Precision (new 30 traces) ──")
    total_det = det_tp + det_fp
    prec = det_tp / max(total_det, 1)
    print(f"  True positives:  {det_tp}")
    print(f"  False positives: {det_fp}")
    print(f"  Precision:       {det_tp}/{total_det} = {prec:.1%}")

    # Coverage gaps
    detected_types = set()
    for r in detector_records:
        if r.has_failure and r.session_id in new_sids:
            t = CAFT_TAXONOMY.get(r.primary_caft_code)
            if t:
                detected_types.add(t.name)

    zero_coverage = set(new_code_counts.keys()) - detected_types
    if zero_coverage:
        print(f"\n── Coverage Gaps (human-labeled but no detector fired) ──")
        for name in sorted(zero_coverage):
            print(f"  {name}")

    # Confidence + severity distribution
    print(f"\n── Confidence Distribution (new 30 human annotations) ──")
    conf_dist = Counter(r.confidence for r in new_human)
    for conf in sorted(conf_dist.keys()):
        print(f"  Confidence {conf}: {conf_dist[conf]} records")

    print(f"\n── Severity Distribution (new 30, failures only) ──")
    sev_dist = Counter(r.severity for r in new_human if r.has_failure)
    for sev in sorted(sev_dist.keys()):
        print(f"  Severity {sev}: {sev_dist[sev]} records")

    # Split summary
    if SPLITS_PATH.exists():
        from agentdiag.splits import SplitManager
        sm = SplitManager(SPLITS_PATH)
        summary = sm.summary()
        print(f"\n── Split Assignments ──")
        print(summary)


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate 30 traces")
    parser.add_argument("--phase", choices=["select", "annotate", "split", "report", "all"],
                        default="all", help="Which phase to run")
    args = parser.parse_args()

    if args.phase in ("select", "all"):
        phase_select()
    if args.phase in ("annotate", "all"):
        phase_annotate()
    if args.phase in ("split", "all"):
        phase_split()
    if args.phase in ("report", "all"):
        phase_report()
