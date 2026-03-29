#!/usr/bin/env python3
"""Diagnostic script: analyze all stall-flagged sessions in the test split.

Pulls each session, runs the stall detector, cross-references with ground truth,
and outputs a comparison table showing TP vs FP characteristics.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agentdiag.adapters.claude_code import ClaudeCodeExtractor
from agentdiag.caft.detectors import StallDetector
from agentdiag.hta import HTAStateMachine


def load_test_sessions(annotations_path, splits_path):
    """Load test split session IDs and their stall annotation status."""
    with open(splits_path) as f:
        splits = json.load(f)
    test_ids = set(splits.get("test", []))

    stall_gt = set()
    with open(annotations_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            sid = d.get("session_id", "")
            if sid not in test_ids:
                continue
            failure = d.get("primary_caft_subtype") or d.get("failure_name", "")
            if not failure and d.get("annotator_type") == "detector":
                failure = d.get("annotator_id", "")
            if failure == "stall":
                stall_gt.add(sid)

    return test_ids, stall_gt


def analyze_session(session, extractor, detector):
    """Run stall detector and extract diagnostic features."""
    try:
        events = extractor.parse_session(session)
    except Exception as e:
        return None, None, str(e)

    if not events:
        return None, None, "no events"

    # Run HTA
    hta_machine = HTAStateMachine()
    for event in events:
        hta_state = hta_machine.push(event)

    # Run stall detector
    diagnosis = detector.check(events, hta_state)
    if diagnosis is None:
        return None, events, "no stall detected"

    # Extract diagnostic features
    evidence = diagnosis.evidence
    stall_steps = evidence.get("stall_steps", [])

    # What tools were used at stall events
    stall_tools = []
    stall_latencies = []
    for idx in stall_steps:
        if idx < len(events):
            e = events[idx]
            stall_tools.append(e.tool or e.type)
            stall_latencies.append(e.latency_ms)

    # Check for user_input events near stalls
    user_input_near_stall = False
    for idx in stall_steps:
        for offset in range(-2, 3):
            check_idx = idx + offset
            if 0 <= check_idx < len(events) and check_idx != idx:
                if events[check_idx].type == "user_input":
                    user_input_near_stall = True
                    break

    # Post-stall progress: unique successful ops after last stall
    last_stall_idx = max(stall_steps) if stall_steps else 0
    remaining = events[last_stall_idx + 1:]
    unique_post_stall = len(set(
        (e.tool, e.output_hash) for e in remaining
        if e.success and e.type == "tool_call" and e.output_hash
    ))
    total_post_stall = len([e for e in remaining if e.type == "tool_call"])

    # Consecutive stall pattern
    consecutive = 1
    max_consecutive = 1
    for i in range(1, len(stall_steps)):
        if stall_steps[i] == stall_steps[i-1] + 1:
            consecutive += 1
            max_consecutive = max(max_consecutive, consecutive)
        else:
            consecutive = 1

    # Same tool repeated at stall
    same_tool_stalls = len(stall_tools) > 1 and len(set(stall_tools)) == 1

    # Tool call events only
    tool_call_events = [e for e in events if e.type == "tool_call"]
    total_tool_calls = len(tool_call_events)
    success_rate = sum(1 for e in tool_call_events if e.success) / max(total_tool_calls, 1)

    return {
        "confidence": diagnosis.confidence,
        "stall_count": evidence.get("stall_count", 0),
        "stall_fraction": evidence.get("stall_fraction", 0),
        "max_latency_ms": evidence.get("max_latency_ms", 0),
        "median_latency_ms": evidence.get("median_latency_ms", 0),
        "threshold_ms": evidence.get("threshold_ms", 0),
        "active_events": evidence.get("active_events", 0),
        "stall_tools": stall_tools,
        "stall_latencies": stall_latencies,
        "user_input_near_stall": user_input_near_stall,
        "unique_post_stall": unique_post_stall,
        "total_post_stall": total_post_stall,
        "max_consecutive": max_consecutive,
        "same_tool_stalls": same_tool_stalls,
        "total_events": len(events),
        "total_tool_calls": total_tool_calls,
        "success_rate": round(success_rate, 3),
        "worst_step": evidence.get("worst_step", 0),
    }, events, None


def main():
    annotations_path = Path("annotations/ablation_ready/annotations.jsonl")
    splits_path = Path("annotations/ablation_ready/splits.json")
    traces_root = Path("~/.claude/projects").expanduser()

    test_ids, stall_gt = load_test_sessions(annotations_path, splits_path)
    print(f"Test split: {len(test_ids)} sessions, {len(stall_gt)} with stall annotations")
    print(f"Stall GT sessions: {sorted(s[:12] for s in stall_gt)}")
    print()

    extractor = ClaudeCodeExtractor()
    all_sessions = extractor.discover(traces_root, min_lines=5)
    session_map = {s.session_id: s for s in all_sessions}

    detector = StallDetector()
    results = []

    for sid in sorted(test_ids):
        session = session_map.get(sid)
        if session is None:
            matches = [s for s in all_sessions if s.session_id.startswith(sid)]
            if matches:
                session = matches[0]
        if session is None:
            continue

        features, events, error = analyze_session(session, extractor, detector)
        if features is None:
            continue

        is_tp = sid in stall_gt
        results.append({
            "session_id": sid[:12],
            "full_id": sid,
            "type": "TP" if is_tp else "FP",
            **features,
        })

    # Sort: TPs first, then FPs
    results.sort(key=lambda r: (r["type"] != "TP", -r["max_latency_ms"]))

    # Print comparison table
    print(f"{'Session':<14} {'Type':<4} {'Conf':<5} {'#Stall':<7} {'Frac':<6} "
          f"{'MaxLat':<9} {'MedLat':<9} {'Thresh':<9} {'Tools':<25} "
          f"{'Consec':<7} {'SameTool':<9} {'UserNear':<9} "
          f"{'PostProg':<9} {'TotalEvt':<9} {'SuccRate':<9}")
    print("-" * 180)

    for r in results:
        tools_str = ",".join(r["stall_tools"][:3])
        if len(r["stall_tools"]) > 3:
            tools_str += f"+{len(r['stall_tools'])-3}"
        print(f"{r['session_id']:<14} {r['type']:<4} {r['confidence']:<5.2f} "
              f"{r['stall_count']:<7} {r['stall_fraction']:<6.3f} "
              f"{r['max_latency_ms']:<9.0f} {r['median_latency_ms']:<9.0f} "
              f"{r['threshold_ms']:<9.0f} {tools_str:<25} "
              f"{r['max_consecutive']:<7} {str(r['same_tool_stalls']):<9} "
              f"{str(r['user_input_near_stall']):<9} "
              f"{r['unique_post_stall']:<9} {r['total_events']:<9} "
              f"{r['success_rate']:<9.3f}")

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    tps = [r for r in results if r["type"] == "TP"]
    fps = [r for r in results if r["type"] == "FP"]
    print(f"TPs: {len(tps)}, FPs: {len(fps)}")

    if fps:
        print("\nFP ANALYSIS:")
        for r in fps:
            print(f"\n  {r['session_id']} ({r['full_id']}):")
            print(f"    Stalls: {r['stall_count']} ({r['stall_fraction']:.1%} of active)")
            print(f"    Max latency: {r['max_latency_ms']:.0f}ms (threshold: {r['threshold_ms']:.0f}ms)")
            print(f"    Stall tools: {r['stall_tools']}")
            print(f"    Stall latencies: {[f'{l:.0f}ms' for l in r['stall_latencies']]}")
            print(f"    Consecutive: {r['max_consecutive']}, Same tool: {r['same_tool_stalls']}")
            print(f"    User input nearby: {r['user_input_near_stall']}")
            print(f"    Post-stall progress: {r['unique_post_stall']} unique ops, {r['total_post_stall']} total")
            print(f"    Session: {r['total_events']} events, {r['total_tool_calls']} tool calls, "
                  f"{r['success_rate']:.0%} success")

    if tps:
        print("\nTP ANALYSIS:")
        for r in tps:
            print(f"\n  {r['session_id']} ({r['full_id']}):")
            print(f"    Stalls: {r['stall_count']} ({r['stall_fraction']:.1%} of active)")
            print(f"    Max latency: {r['max_latency_ms']:.0f}ms (threshold: {r['threshold_ms']:.0f}ms)")
            print(f"    Stall tools: {r['stall_tools']}")
            print(f"    Stall latencies: {[f'{l:.0f}ms' for l in r['stall_latencies']]}")
            print(f"    Consecutive: {r['max_consecutive']}, Same tool: {r['same_tool_stalls']}")
            print(f"    User input nearby: {r['user_input_near_stall']}")
            print(f"    Post-stall progress: {r['unique_post_stall']} unique ops, {r['total_post_stall']} total")

    # Feature comparison
    print("\n" + "=" * 80)
    print("FEATURE COMPARISON (TP vs FP)")
    print("=" * 80)

    def avg(lst, key):
        vals = [r[key] for r in lst]
        return sum(vals) / len(vals) if vals else 0

    for feature in ["stall_count", "stall_fraction", "max_latency_ms", "median_latency_ms",
                     "max_consecutive", "unique_post_stall", "total_post_stall",
                     "total_events", "confidence", "success_rate"]:
        tp_avg = avg(tps, feature)
        fp_avg = avg(fps, feature)
        print(f"  {feature:<25} TP avg: {tp_avg:>10.1f}  FP avg: {fp_avg:>10.1f}  "
              f"{'<< SEPARATING' if abs(tp_avg - fp_avg) > max(abs(tp_avg), abs(fp_avg)) * 0.3 else ''}")

    tp_user = sum(1 for r in tps if r["user_input_near_stall"])
    fp_user = sum(1 for r in fps if r["user_input_near_stall"])
    print(f"  {'user_input_near_stall':<25} TP: {tp_user}/{len(tps)}  FP: {fp_user}/{len(fps)}")

    tp_same = sum(1 for r in tps if r["same_tool_stalls"])
    fp_same = sum(1 for r in fps if r["same_tool_stalls"])
    print(f"  {'same_tool_stalls':<25} TP: {tp_same}/{len(tps)}  FP: {fp_same}/{len(fps)}")


if __name__ == "__main__":
    main()
