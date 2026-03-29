"""Automated CAFT trace annotation pipeline.

3-step workflow:
  1. PREPARE: Find unlabeled traces, extract summaries → batch_summaries.json
  2. ANNOTATE: User dispatches Claude Code subagents with the summaries (external)
  3. MERGE: Parse agent results, validate, merge into ground truth

Usage:
    # Prepare batch
    python -m agentdiag auto-annotate --prepare \\
        --traces ~/.claude/projects --n 30 \\
        --output annotations/batch_summaries.json

    # Merge results
    python -m agentdiag auto-annotate --merge \\
        annotations/auto_results.json \\
        --into annotations/ground_truth_50.json \\
        --output annotations/ground_truth_80.json

    # Validate auto vs manual agreement
    python -m agentdiag auto-annotate --validate \\
        --auto annotations/auto_labels.json \\
        --manual annotations/ground_truth_50.json
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

from agentdiag.auto_annotate_prompt import (
    ANNOTATION_CRITERIA,
    OBSERVABLE_CAFT_CODES,
)


# ---------------------------------------------------------------------------
# 1. Extract trace summary from raw JSONL
# ---------------------------------------------------------------------------

def extract_trace_summary(jsonl_path: Path) -> dict:
    """Extract an annotation-ready summary from a raw Claude Code session JSONL.

    Reads the raw JSONL file (not TraceEvents) to capture user messages,
    errors, tool sequences, and session boundaries. Also runs MonitorEngine
    on parsed TraceEvents for heuristic detector results.

    Args:
        jsonl_path: Path to a Claude Code session .jsonl file.

    Returns:
        Structured dict with summary fields ready for LLM consumption.
    """
    raw_lines = []
    with open(jsonl_path, "r", errors="replace") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw_lines.append((line_num, json.loads(line)))
            except json.JSONDecodeError:
                continue

    total_lines = len(raw_lines)

    # Extract user messages
    user_messages = []
    for line_num, event in raw_lines:
        if event.get("type") != "user":
            continue
        msg = event.get("message", {})
        content = msg.get("content", "")
        if isinstance(content, str) and content.strip():
            user_messages.append({
                "line": line_num,
                "content": content.strip()[:300],
            })

    # Extract errors from tool results
    errors = []
    for line_num, event in raw_lines:
        if event.get("type") != "user":
            continue
        msg = event.get("message", {})
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if block.get("type") == "tool_result" and block.get("is_error"):
                result_content = block.get("content", "")
                if isinstance(result_content, list):
                    result_content = " ".join(
                        b.get("text", "") for b in result_content
                        if isinstance(b, dict)
                    )
                errors.append({
                    "line": line_num,
                    "content": str(result_content).strip()[:200],
                })

    # Extract tool call sequence
    tool_sequence = []
    for line_num, event in raw_lines:
        if event.get("type") != "assistant":
            continue
        msg = event.get("message", {})
        for block in msg.get("content", []):
            if block.get("type") == "tool_use":
                tool_sequence.append({
                    "line": line_num,
                    "tool": block.get("name", "unknown"),
                })

    # First and last 3 events (for session start/end context)
    first_events = []
    for line_num, event in raw_lines[:3]:
        first_events.append({
            "line": line_num,
            "type": event.get("type", "unknown"),
            "preview": _event_preview(event),
        })

    last_events = []
    for line_num, event in raw_lines[-3:]:
        last_events.append({
            "line": line_num,
            "type": event.get("type", "unknown"),
            "preview": _event_preview(event),
        })

    # ExitPlanMode calls and outcomes
    plan_mode_exits = []
    for i, (line_num, event) in enumerate(raw_lines):
        if event.get("type") != "assistant":
            continue
        msg = event.get("message", {})
        for block in msg.get("content", []):
            if block.get("type") == "tool_use" and block.get("name") == "ExitPlanMode":
                # Check if next user event rejected it
                rejected = False
                for j in range(i + 1, min(i + 5, len(raw_lines))):
                    _, next_event = raw_lines[j]
                    if next_event.get("type") == "user":
                        next_content = next_event.get("message", {}).get("content", "")
                        if isinstance(next_content, str):
                            lower = next_content.lower()
                            if any(w in lower for w in ("reject", "no", "don't", "stop")):
                                rejected = True
                        break
                plan_mode_exits.append({"line": line_num, "rejected": rejected})

    # Run MonitorEngine for heuristic results
    heuristic_results = _run_heuristics(jsonl_path)

    return {
        "total_lines": total_lines,
        "summary": {
            "user_messages": user_messages,
            "errors": errors,
            "first_events": first_events,
            "last_events": last_events,
            "tool_sequence": [t["tool"] for t in tool_sequence],
            "tool_sequence_with_lines": tool_sequence[:200],  # cap for size
            "plan_mode_exits": plan_mode_exits,
            "error_count": len(errors),
        },
        "heuristic_results": heuristic_results,
    }


def _event_preview(event: dict) -> str:
    """Short preview string for a raw event."""
    etype = event.get("type", "")
    if etype == "user":
        content = event.get("message", {}).get("content", "")
        if isinstance(content, str):
            return content[:150]
        elif isinstance(content, list):
            types = [b.get("type", "?") for b in content if isinstance(b, dict)]
            return f"[{', '.join(types)}]"
    elif etype == "assistant":
        blocks = event.get("message", {}).get("content", [])
        types = [b.get("type", "?") for b in blocks if isinstance(b, dict)]
        return f"[{', '.join(types)}]"
    return etype


def _run_heuristics(jsonl_path: Path) -> dict:
    """Run MonitorEngine on a trace and return heuristic detector results."""
    try:
        from agentdiag.adapters.claude_code import ClaudeCodeExtractor
        from agentdiag.monitor import MonitorEngine

        extractor = ClaudeCodeExtractor()
        events = extractor.parse_session(jsonl_path)
        if not events:
            return {"detectors_fired": [], "trust_score": 1.0, "health": "healthy"}

        engine = MonitorEngine(goal="")
        for event in events:
            engine.push(event)

        state = engine.state
        return {
            "detectors_fired": [d.failure_name for d in state.diagnoses],
            "trust_score": round(state.trust_score, 3),
            "health": state.health,
        }
    except Exception as e:
        return {"detectors_fired": [], "trust_score": 1.0, "health": "error",
                "error": str(e)}


# ---------------------------------------------------------------------------
# 2. Prepare batch summaries
# ---------------------------------------------------------------------------

def prepare_batch(
    manifest_path: Path,
    ground_truth_path: Optional[Path],
    traces_root: Path,
    n: int,
    output_path: Path,
) -> dict:
    """Prepare batch_summaries.json for annotation.

    Reads manifest.csv, filters to unlabeled traces not in existing ground truth,
    extracts summaries, and writes the batch file.

    Args:
        manifest_path: Path to data/manifest.csv.
        ground_truth_path: Path to existing ground truth JSON (to skip already-labeled).
        traces_root: Root path to search for session JSONL files.
        n: Maximum number of traces to include.
        output_path: Where to write batch_summaries.json.

    Returns:
        The batch_summaries dict (also written to output_path).
    """
    # Load existing ground truth session IDs to skip
    existing_ids = set()
    if ground_truth_path and ground_truth_path.exists():
        with open(ground_truth_path) as f:
            gt = json.load(f)
        for trace in gt.get("traces", []):
            existing_ids.add(trace["session_id"])

    # Read manifest
    rows = []
    with open(manifest_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Filter: not already labeled, has tool use, enough lines
    candidates = []
    for row in rows:
        sid = row["session_id"]
        if sid[:8] in existing_ids or sid in existing_ids:
            continue
        if row.get("has_tool_use", "").lower() != "true":
            continue
        line_count = int(row.get("line_count", 0))
        if line_count < 50:
            continue
        candidates.append(row)

    # Sort by line count descending (larger traces more interesting)
    candidates.sort(key=lambda r: int(r.get("line_count", 0)), reverse=True)
    candidates = candidates[:n]

    # Discover session files
    from agentdiag.adapters.claude_code import ClaudeCodeExtractor
    extractor = ClaudeCodeExtractor()
    sessions = extractor.discover(traces_root, min_lines=5)
    session_map = {s.session_id: s for s in sessions}

    # Extract summaries
    traces = []
    next_num = len(existing_ids) + 1

    for row in candidates:
        sid = row["session_id"]
        session = session_map.get(sid)
        if session is None:
            print(f"  Skipping {sid[:8]}: JSONL not found under {traces_root}",
                  file=sys.stderr)
            continue

        print(f"  Extracting summary for {sid[:8]} "
              f"({row.get('line_count', '?')} lines)...",
              file=sys.stderr)

        summary = extract_trace_summary(session.path)

        # Derive project name from manifest
        project_raw = row.get("project", "")
        # Strip leading -Users-samkoscelny- prefix
        project = project_raw.split("-")[-1] if "-" in project_raw else project_raw

        traces.append({
            "trace_num": next_num,
            "session_id": sid[:8],
            "session_id_full": sid,
            "jsonl_path": str(session.path),
            "project": project,
            "total_lines": summary["total_lines"],
            "summary": summary["summary"],
            "heuristic_results": summary["heuristic_results"],
        })
        next_num += 1

    # Build output
    batch = {
        "annotation_instructions": {
            "criteria": ANNOTATION_CRITERIA,
            "output_format": {
                "trace_num": "int",
                "session_id": "str (8 chars)",
                "project": "str",
                "events": "int",
                "user_goal": "str",
                "agent_completed": "bool",
                "actual_failures": "list[str]",
                "failure_details": "list[dict] with caft_code, caft_name, "
                                   "onset_step, severity, confidence, rationale",
                "annotations": "dict (empty {})",
            },
            "caft_codes": {
                code: info for code, info in OBSERVABLE_CAFT_CODES.items()
            },
        },
        "traces": traces,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(batch, f, indent=2)

    print(f"\nPrepared {len(traces)} trace summaries → {output_path}",
          file=sys.stderr)
    return batch


# ---------------------------------------------------------------------------
# 3. Parse annotation response
# ---------------------------------------------------------------------------

VALID_CAFT_NAMES = set(info["name"] for info in OBSERVABLE_CAFT_CODES.values())


def parse_annotation_response(text: str) -> list[dict]:
    """Parse LLM annotation output into ground truth format.

    Accepts a JSON array string. Validates each annotation for required fields,
    valid CAFT codes, and severity/confidence ranges.

    Args:
        text: Raw text from LLM (expected to be a JSON array).

    Returns:
        List of validated trace annotations.

    Raises:
        ValueError: If the text cannot be parsed or contains invalid annotations.
    """
    # Strip markdown fences if present
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (fences)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}")

    if not isinstance(data, list):
        # Maybe it's wrapped in an object
        if isinstance(data, dict) and "traces" in data:
            data = data["traces"]
        else:
            raise ValueError(f"Expected JSON array, got {type(data).__name__}")

    validated = []
    for i, ann in enumerate(data):
        errors = _validate_annotation(ann, i)
        if errors:
            print(f"  Warning: trace {ann.get('trace_num', '?')}: "
                  f"{'; '.join(errors)}", file=sys.stderr)

        # Ensure required fields with defaults
        cleaned = {
            "trace_num": ann.get("trace_num", i + 1),
            "session_id": ann.get("session_id", ""),
            "project": ann.get("project", ""),
            "events": ann.get("events", 0),
            "user_goal": ann.get("user_goal", ""),
            "agent_completed": ann.get("agent_completed", True),
            "actual_failures": ann.get("actual_failures", []),
            "failure_details": _clean_failure_details(
                ann.get("failure_details", [])
            ),
            "annotations": ann.get("annotations", {}),
        }
        validated.append(cleaned)

    return validated


def _validate_annotation(ann: dict, idx: int) -> list[str]:
    """Validate a single annotation dict. Returns list of warning messages."""
    errors = []

    required = ["session_id", "user_goal", "agent_completed"]
    for field in required:
        if field not in ann:
            errors.append(f"missing '{field}'")

    # Validate actual_failures
    for name in ann.get("actual_failures", []):
        if name not in VALID_CAFT_NAMES:
            errors.append(f"unknown CAFT name '{name}'")

    # Validate failure_details
    for detail in ann.get("failure_details", []):
        code = detail.get("caft_code", "")
        if code and code not in OBSERVABLE_CAFT_CODES:
            errors.append(f"unknown CAFT code '{code}'")
        sev = detail.get("severity", 0)
        if not (1 <= sev <= 5):
            errors.append(f"severity {sev} out of range 1-5")
        conf = detail.get("confidence", 0)
        if not (1 <= conf <= 5):
            errors.append(f"confidence {conf} out of range 1-5")

    return errors


def _clean_failure_details(details: list) -> list[dict]:
    """Normalize failure_details entries."""
    cleaned = []
    for d in details:
        if not isinstance(d, dict):
            continue
        cleaned.append({
            "caft_code": str(d.get("caft_code", "")),
            "caft_name": str(d.get("caft_name", "")),
            "onset_step": int(d.get("onset_step", 0)),
            "severity": int(d.get("severity", 3)),
            "confidence": int(d.get("confidence", 3)),
            "rationale": str(d.get("rationale", "")),
        })
    return cleaned


# ---------------------------------------------------------------------------
# 4. Merge annotations into ground truth
# ---------------------------------------------------------------------------

def merge_annotations(
    new_annotations: list[dict],
    existing_path: Optional[Path],
    output_path: Path,
) -> dict:
    """Merge new annotations into existing ground truth file.

    Deduplicates by session_id. New annotations overwrite existing ones
    for the same session.

    Args:
        new_annotations: List of annotation dicts from parse_annotation_response().
        existing_path: Path to existing ground truth JSON (or None to start fresh).
        output_path: Where to write the merged ground truth.

    Returns:
        The merged ground truth dict.
    """
    # Load existing
    if existing_path and existing_path.exists():
        with open(existing_path) as f:
            gt = json.load(f)
    else:
        gt = {
            "annotator": "claude-sonnet-4-5 (auto)",
            "date": "",
            "method": "Automated annotation via auto_annotate pipeline",
            "criteria": ANNOTATION_CRITERIA,
            "traces": [],
        }

    # Index existing by session_id
    existing_by_sid = {}
    for trace in gt.get("traces", []):
        existing_by_sid[trace["session_id"]] = trace

    # Merge new
    added = 0
    updated = 0
    for ann in new_annotations:
        sid = ann["session_id"]
        if sid in existing_by_sid:
            existing_by_sid[sid].update(ann)
            updated += 1
        else:
            existing_by_sid[sid] = ann
            added += 1

    # Rebuild traces list sorted by trace_num
    all_traces = list(existing_by_sid.values())
    all_traces.sort(key=lambda t: t.get("trace_num", 999))

    gt["traces"] = all_traces

    # Update metadata
    import datetime
    gt["date"] = datetime.date.today().isoformat()

    # Write
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(gt, f, indent=2)

    print(f"\nMerge complete: {added} added, {updated} updated, "
          f"{len(all_traces)} total → {output_path}", file=sys.stderr)

    # Print summary statistics
    _print_summary_stats(all_traces)

    return gt


def _print_summary_stats(traces: list[dict]) -> None:
    """Print summary statistics for annotated traces."""
    total = len(traces)
    completed = sum(1 for t in traces if t.get("agent_completed", True))
    with_failures = sum(1 for t in traces if t.get("actual_failures"))

    # Count failure types
    failure_counts: Counter = Counter()
    for t in traces:
        for f in t.get("actual_failures", []):
            failure_counts[f] += 1

    print(f"\n{'=' * 50}", file=sys.stderr)
    print(f"  Ground Truth Summary ({total} traces)", file=sys.stderr)
    print(f"{'=' * 50}", file=sys.stderr)
    print(f"  Completed:      {completed}/{total} "
          f"({completed/total*100:.0f}%)", file=sys.stderr)
    print(f"  With failures:  {with_failures}/{total} "
          f"({with_failures/total*100:.0f}%)", file=sys.stderr)

    if failure_counts:
        print(f"\n  Failure distribution:", file=sys.stderr)
        for name, count in failure_counts.most_common():
            print(f"    {name:<30} {count}", file=sys.stderr)
    print(file=sys.stderr)


# ---------------------------------------------------------------------------
# 5. Validate auto vs manual agreement
# ---------------------------------------------------------------------------

def validate_agreement(
    auto_path: Path,
    manual_path: Path,
) -> dict:
    """Compare auto annotations against manual ground truth.

    Computes per-failure-type precision, recall, and Cohen's kappa
    on the overlap set (traces in both files).

    Args:
        auto_path: Path to auto-annotated ground truth JSON.
        manual_path: Path to manually-annotated ground truth JSON.

    Returns:
        Agreement statistics dict.
    """
    with open(auto_path) as f:
        auto_gt = json.load(f)
    with open(manual_path) as f:
        manual_gt = json.load(f)

    # Index by session_id
    auto_by_sid = {t["session_id"]: t for t in auto_gt.get("traces", [])}
    manual_by_sid = {t["session_id"]: t for t in manual_gt.get("traces", [])}

    # Find overlap
    overlap_sids = set(auto_by_sid) & set(manual_by_sid)
    if not overlap_sids:
        print("No overlapping traces found.", file=sys.stderr)
        return {"overlap": 0, "agreement": {}}

    # Per-failure-type agreement
    all_failure_types = set()
    for sid in overlap_sids:
        all_failure_types.update(auto_by_sid[sid].get("actual_failures", []))
        all_failure_types.update(manual_by_sid[sid].get("actual_failures", []))

    results = {}
    for ftype in sorted(all_failure_types):
        tp = fp = fn = tn = 0
        for sid in overlap_sids:
            auto_has = ftype in auto_by_sid[sid].get("actual_failures", [])
            manual_has = ftype in manual_by_sid[sid].get("actual_failures", [])
            if auto_has and manual_has:
                tp += 1
            elif auto_has and not manual_has:
                fp += 1
            elif not auto_has and manual_has:
                fn += 1
            else:
                tn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        results[ftype] = {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        }

    # Overall completion agreement
    completion_agree = sum(
        1 for sid in overlap_sids
        if auto_by_sid[sid].get("agent_completed") ==
           manual_by_sid[sid].get("agent_completed")
    )

    # Cohen's kappa for binary "has any failure"
    auto_any = [bool(auto_by_sid[sid].get("actual_failures")) for sid in overlap_sids]
    manual_any = [bool(manual_by_sid[sid].get("actual_failures")) for sid in overlap_sids]
    kappa = _cohens_kappa(auto_any, manual_any)

    summary = {
        "overlap": len(overlap_sids),
        "completion_agreement": f"{completion_agree}/{len(overlap_sids)}",
        "cohens_kappa": round(kappa, 3),
        "per_failure_type": results,
    }

    # Print
    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"  Auto vs Manual Agreement ({len(overlap_sids)} overlapping traces)",
          file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)
    print(f"  Completion agreement: {completion_agree}/{len(overlap_sids)}",
          file=sys.stderr)
    print(f"  Cohen's kappa (any failure): {kappa:.3f}", file=sys.stderr)
    print(f"\n  {'Failure Type':<30} {'Prec':>6} {'Rec':>6} {'F1':>6} "
          f"{'TP':>4} {'FP':>4} {'FN':>4}", file=sys.stderr)
    print(f"  {'─' * 64}", file=sys.stderr)
    for ftype, stats in sorted(results.items()):
        print(f"  {ftype:<30} {stats['precision']:>6.3f} "
              f"{stats['recall']:>6.3f} {stats['f1']:>6.3f} "
              f"{stats['tp']:>4} {stats['fp']:>4} {stats['fn']:>4}",
              file=sys.stderr)
    print(file=sys.stderr)

    return summary


def _cohens_kappa(a: list[bool], b: list[bool]) -> float:
    """Compute Cohen's kappa for two binary label lists."""
    n = len(a)
    if n == 0:
        return 0.0

    agree = sum(1 for x, y in zip(a, b) if x == y)
    p_o = agree / n

    # Expected agreement by chance
    a_pos = sum(a) / n
    b_pos = sum(b) / n
    p_e = a_pos * b_pos + (1 - a_pos) * (1 - b_pos)

    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1 - p_e)
