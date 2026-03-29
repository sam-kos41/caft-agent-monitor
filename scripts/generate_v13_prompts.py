#!/usr/bin/env python3
"""Generate all LLM prompts for V13 ablation cache pre-population.

Runs the detection pipeline on test-split sessions, then generates:
1. Tier 1.5 confirmation prompts for each candidate (stall, error_cascade, PT)
2. Tier 2 session-end assessment prompts for every session

Outputs a JSONL file where each line is:
{
    "prompt_id": "tier1.5_<sid>_<detector>_<step>" or "tier2_<sid>",
    "tier": "1.5" or "2",
    "trace_id": "<session_id>",
    "detector": "<failure_name>" or "tier2_assessment",
    "candidate_step": <step> or -1,
    "prompt": "<the full prompt text>"
}

Usage:
    python scripts/generate_v13_prompts.py \
        --annotations annotations/ground_truth_76.json \
        --split test --splits-file annotations/ablation_ready/splits_76.json \
        --output prompts_v13.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agentdiag.adapters.claude_code import ClaudeCodeExtractor
from agentdiag.caft.base import CaftDiagnosis, CaftSeverity
from agentdiag.caft.confirm import (
    build_confirmation_prompt,
    build_session_assessment_prompt,
    AUTOCONFIRM_THRESHOLD,
)
from agentdiag.caft.detectors import ALL_CAFT_DETECTORS_FULL
from agentdiag.hta import HTAStateMachine
from agentdiag.monitor import MonitorEngine

# Reuse loading logic from run_ablation
from scripts.run_ablation import (
    load_annotations,
    load_gt_session_ids,
    diagnoses_to_detections,
    dedup_detections,
    run_detectors_on_traces,
    Detection,
    Annotation,
)


def main():
    parser = argparse.ArgumentParser(description="Generate V13 LLM prompts")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--splits-file", default=None)
    parser.add_argument("--traces", default="~/.claude/projects")
    parser.add_argument("--output", default="prompts_v13.jsonl")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    annotations_path = Path(args.annotations)
    traces_root = Path(args.traces).expanduser()
    output_path = Path(args.output)

    # Load annotations
    annotations = load_annotations(annotations_path)
    annotation_sids = set(a.trace_id for a in annotations)
    if annotations_path.suffix == ".json":
        all_gt_sids = load_gt_session_ids(annotations_path)
        session_ids = sorted(annotation_sids | all_gt_sids)
    else:
        session_ids = sorted(annotation_sids)

    # Filter by split
    if args.split and args.splits_file:
        splits_file = Path(args.splits_file)
        if splits_file.exists():
            from agentdiag.splits import SplitManager
            sm = SplitManager(splits_file)
            split_ids = set(sm.get_traces(args.split))
            split_prefixes = {sid[:8] for sid in split_ids}
            session_ids = [s for s in session_ids
                           if s in split_ids or s[:8] in split_prefixes]
            annotations = [a for a in annotations
                           if a.trace_id in split_ids or a.trace_id[:8] in split_prefixes]

    # Deduplicate
    prefix_groups: dict[str, list[str]] = defaultdict(list)
    for sid in session_ids:
        prefix_groups[sid[:8]].append(sid)
    canonical_ids = sorted(max(group, key=len) for group in prefix_groups.values())
    if len(canonical_ids) < len(session_ids):
        prefix_to_canonical = {cid[:8]: cid for cid in canonical_ids}
        seen_ann: set[tuple[str, str]] = set()
        deduped: list[Annotation] = []
        for a in annotations:
            canon = prefix_to_canonical.get(a.trace_id[:8], a.trace_id)
            key = (canon, a.failure_name)
            if key not in seen_ann:
                seen_ann.add(key)
                deduped.append(Annotation(
                    trace_id=canon, failure_name=a.failure_name,
                    caft_code=a.caft_code, onset_step=a.onset_step,
                ))
        annotations = deduped
        session_ids = canonical_ids

    print(f"Sessions: {len(session_ids)}, Annotations: {len(annotations)}")

    # Run detectors (loose mode)
    print("Running detectors...")
    results = run_detectors_on_traces(
        session_ids, list(ALL_CAFT_DETECTORS_FULL), traces_root,
        verbose=args.verbose,
    )
    dets_loose = dedup_detections(diagnoses_to_detections(results))
    print(f"Loose detections: {len(dets_loose)}")

    # Parse ALL sessions for events + HTA
    print("Parsing all sessions...")
    extractor = ClaudeCodeExtractor()
    all_sessions = extractor.discover(traces_root, min_lines=5)
    session_map = {s.session_id: s for s in all_sessions}

    session_events: dict[str, list] = {}
    session_hta: dict[str, object] = {}
    for sid in session_ids:
        session = session_map.get(sid)
        if session is None:
            matches = [s for s in all_sessions if s.session_id.startswith(sid)]
            if matches:
                session = matches[0]
        if session is None:
            continue
        try:
            events = extractor.parse_session(session)
            if not events:
                continue
            hta_machine = HTAStateMachine()
            for event in events:
                hta_state = hta_machine.push(event)
            session_events[sid] = events
            session_hta[sid] = hta_state
        except Exception as e:
            print(f"  Parse error for {sid[:10]}: {e}")

    print(f"Parsed {len(session_events)}/{len(session_ids)} sessions")

    # Generate prompts
    prompts = []

    # Tier 1.5: candidates that need LLM review
    for det in dets_loose:
        # Skip auto-confirmable (high confidence, no force_llm_review)
        if det.confidence >= AUTOCONFIRM_THRESHOLD and not det.force_llm_review:
            if args.verbose:
                print(f"  Auto-confirm: {det.trace_id[:10]} {det.failure_name} c={det.confidence:.2f}")
            continue

        events = session_events.get(det.trace_id)
        hta = session_hta.get(det.trace_id)
        if not events or hta is None:
            continue

        # Build CaftDiagnosis stub for prompt
        stub = CaftDiagnosis(
            caft_code=det.caft_code,
            caft_category="",
            failure_name=det.failure_name,
            severity=CaftSeverity.WARNING,
            confidence=det.confidence,
            description="",
            evidence={},
            at_step=det.onset_step,
            remediation="",
        )

        prompt = build_confirmation_prompt(stub, events, hta)
        prompt_id = f"tier1.5_{det.trace_id[:8]}_{det.failure_name}_{det.onset_step}"

        prompts.append({
            "prompt_id": prompt_id,
            "tier": "1.5",
            "trace_id": det.trace_id,
            "detector": det.failure_name,
            "candidate_step": det.onset_step,
            "confidence": det.confidence,
            "prompt": prompt,
        })

    # Tier 2: session-end assessment for ALL sessions
    for sid in session_ids:
        events = session_events.get(sid)
        hta = session_hta.get(sid)
        if not events or hta is None:
            continue

        prompt = build_session_assessment_prompt(events, hta)
        prompt_id = f"tier2_{sid[:8]}"

        prompts.append({
            "prompt_id": prompt_id,
            "tier": "2",
            "trace_id": sid,
            "detector": "tier2_assessment",
            "candidate_step": -1,
            "confidence": 0.0,
            "prompt": prompt,
        })

    # Write output
    with open(output_path, "w") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")

    tier15_count = sum(1 for p in prompts if p["tier"] == "1.5")
    tier2_count = sum(1 for p in prompts if p["tier"] == "2")
    print(f"\nGenerated {len(prompts)} prompts:")
    print(f"  Tier 1.5: {tier15_count} candidates")
    print(f"  Tier 2:   {tier2_count} sessions")
    print(f"Written to {output_path}")

    # Summary of Tier 1.5 candidates
    if args.verbose:
        print("\nTier 1.5 candidates:")
        for p in prompts:
            if p["tier"] == "1.5":
                print(f"  {p['trace_id'][:10]} {p['detector']}@{p['candidate_step']} c={p['confidence']:.2f}")
        print(f"\nTier 2: all {tier2_count} sessions")


if __name__ == "__main__":
    main()
