"""
agentdiag CLI
=============
6 commands:

    python -m agentdiag monitor --input stdin [--plain] [--json] [--demo scenario] [--web]
    python -m agentdiag evaluate --traces ~/.claude/projects [--json] [--split test] [--synthetic]
    python -m agentdiag calibrate --traces ~/.claude/projects [--pilot]
    python -m agentdiag annotate {queue,show,adjudicate,export-gold,stats,import-gt,auto-prepare,auto-merge,auto-validate}
    python -m agentdiag context {search,stats,cases,feedback,feedback-summary} [--context-db ./ctx]
    python -m agentdiag taxonomy [--observable-only]
"""

import argparse
import json
import sys
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════
# Command handlers
# ═══════════════════════════════════════════════════════════════════════

def cmd_monitor(args: argparse.Namespace) -> None:
    """Unified monitor: streaming TUI, plain text, JSON analysis, demo, or web."""

    # ── Compare mode ───────────────────────────────────────────────
    compare = getattr(args, "compare", None)
    if compare is not None:
        _handle_compare(args, compare)
        return

    # ── Demo mode ──────────────────────────────────────────────────
    demo = getattr(args, "demo", None)
    if demo is not None:
        _handle_demo(args, demo)
        return

    # ── Need --input for all other modes ───────────────────────────
    input_source = getattr(args, "input", None)
    if not input_source:
        print("Error: --input required (or use --demo)", file=sys.stderr)
        sys.exit(1)

    # ── JSON analysis mode (file → engine → JSON summary) ─────────
    if getattr(args, "json", False):
        _handle_json_analysis(args, input_source)
        return

    # ── Streaming modes: open stream ──────────────────────────────
    if input_source == "stdin":
        stream = sys.stdin
    else:
        path = Path(input_source)
        if not path.exists():
            print(f"Error: File not found: {path}", file=sys.stderr)
            sys.exit(1)
        stream = open(path, "r")

    context_store = _make_context_store(args)

    confirm = getattr(args, "confirm", False)

    try:
        if getattr(args, "web", False):
            # ── Web visualization ─────────────────────────────────
            from agentdiag.visualize import start_server
            start_server(
                stream=stream,
                goal=getattr(args, "goal", "") or "",
                port=getattr(args, "port", 8080),
                delay=getattr(args, "delay", 0.0),
                context_store=context_store,
                confirm=confirm,
                decision_trace=getattr(args, "decision_trace", False),
                cognitive=getattr(args, "cognitive", False),
                input_path=input_source if input_source != "stdin" else "",
            )
        elif getattr(args, "plain", False):
            from agentdiag.tui import run_plain_monitor
            run_plain_monitor(
                goal=getattr(args, "goal", "") or "",
                stream=stream,
                context_store=context_store,
            )
        else:
            from agentdiag.tui import run_dashboard, HAS_RICH
            if not HAS_RICH:
                print("rich not installed, falling back to plain output.",
                      file=sys.stderr)
                from agentdiag.tui import run_plain_monitor
                run_plain_monitor(
                    goal=getattr(args, "goal", "") or "",
                    stream=stream,
                    context_store=context_store,
                )
            else:
                run_dashboard(
                    goal=getattr(args, "goal", "") or "",
                    stream=stream,
                    context_store=context_store,
                )
    finally:
        if context_store is not None:
            context_store.close()
        if stream is not sys.stdin:
            stream.close()


def _handle_compare(args: argparse.Namespace, traces: list[str]) -> None:
    """Run comparison mode: two traces side-by-side in web UI."""
    if not getattr(args, "web", False):
        print("Error: --compare requires --web", file=sys.stderr)
        sys.exit(1)

    trace_a, trace_b = traces
    path_a = Path(trace_a)
    path_b = Path(trace_b)

    if not path_a.exists():
        print(f"Error: File not found: {path_a}", file=sys.stderr)
        sys.exit(1)
    if not path_b.exists():
        print(f"Error: File not found: {path_b}", file=sys.stderr)
        sys.exit(1)

    from agentdiag.visualize import start_compare_server

    stream_a = open(path_a, "r")
    stream_b = open(path_b, "r")
    try:
        start_compare_server(
            stream_a=stream_a,
            stream_b=stream_b,
            goal=getattr(args, "goal", "") or "",
            port=getattr(args, "port", 8080),
            delay=getattr(args, "delay", 0.0),
        )
    finally:
        stream_a.close()
        stream_b.close()


def _handle_demo(args: argparse.Namespace, scenario: str) -> None:
    """Run CAFT demo with synthetic traces."""
    import threading

    from agentdiag.caft.synthetic import CAFT_GENERATORS

    # Validate scenario name
    valid_scenarios = set(CAFT_GENERATORS.keys()) | {"e2e", "all", "showcase", "compare"}
    if scenario not in valid_scenarios:
        available = sorted(valid_scenarios)
        print(f"Error: Unknown scenario '{scenario}'. "
              f"Available: {available}", file=sys.stderr)
        sys.exit(1)

    delay = getattr(args, "delay", 0.3)
    confirm = getattr(args, "confirm", False)
    context_store = _make_context_store(args)

    # ── Web mode: pipe demo JSONL through QueueStream to web server ──
    if getattr(args, "web", False):
        # Compare demo: clean vs failing trace side-by-side
        if scenario == "compare":
            from agentdiag.caft.demo import generate_demo_jsonl
            from agentdiag.visualize import QueueStream, start_compare_server

            qs_a = QueueStream()
            qs_b = QueueStream()
            thread_a = threading.Thread(
                target=generate_demo_jsonl,
                args=("clean", delay, qs_a),
                daemon=True,
            )
            thread_b = threading.Thread(
                target=generate_demo_jsonl,
                args=("premature_termination", delay, qs_b),
                daemon=True,
            )
            thread_a.start()
            thread_b.start()

            start_compare_server(
                stream_a=qs_a,
                stream_b=qs_b,
                goal="Compare Demo",
                port=getattr(args, "port", 8080),
                delay=0.0,  # delay already applied in generators
            )
            return

        from agentdiag.caft.demo import generate_demo_jsonl
        from agentdiag.visualize import QueueStream, start_server

        qs = QueueStream()
        thread = threading.Thread(
            target=generate_demo_jsonl,
            args=(scenario, delay, qs),
            daemon=True,
        )
        thread.start()

        try:
            start_server(
                stream=qs,
                goal="CAFT Demo",
                confirm=confirm,
                port=getattr(args, "port", 8080),
                context_store=context_store,
                decision_trace=getattr(args, "decision_trace", False),
                cognitive=getattr(args, "cognitive", False),
            )
        finally:
            if context_store is not None:
                context_store.close()
        return

    # ── e2e demo (non-web): full pipeline walkthrough ──
    if scenario == "e2e":
        from scripts.demo_e2e import run_demo as run_e2e_demo
        run_e2e_demo(
            json_output=getattr(args, "json", False),
            confirm=confirm,
            delay=delay,
        )
        return

    # ── Synthetic scenario(s) in TUI or plain mode ──
    scenario_name = scenario if scenario not in ("all", "showcase") else None
    plain = getattr(args, "plain", False)

    if plain:
        from agentdiag.caft.demo import run_demo_plain
        run_demo_plain(scenario=scenario_name, delay=delay)
    else:
        from agentdiag.tui import run_dashboard_demo, HAS_RICH
        if not HAS_RICH:
            print("rich not installed, falling back to plain output.",
                  file=sys.stderr)
            from agentdiag.caft.demo import run_demo_plain
            run_demo_plain(scenario=scenario_name, delay=delay)
        else:
            run_dashboard_demo(scenario=scenario_name, delay=delay)


def _handle_json_analysis(args: argparse.Namespace, input_source: str) -> None:
    """Load trace → run through engine → output JSON summary."""
    from agentdiag.monitor import MonitorEngine

    if input_source == "stdin":
        # Stream stdin JSONL
        engine = MonitorEngine(goal=getattr(args, "goal", "") or "")
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                engine.push_raw(data)
            except json.JSONDecodeError:
                continue
    else:
        # Load file (supports .json and .jsonl via load_trace)
        from agentdiag.loading import load_trace
        path = Path(input_source)
        if not path.exists():
            print(f"Error: File not found: {path}", file=sys.stderr)
            sys.exit(1)
        events = load_trace(path)
        if not events:
            print("Error: No events found in trace file.", file=sys.stderr)
            sys.exit(1)
        engine = MonitorEngine(goal=getattr(args, "goal", "") or "")
        for event in events:
            engine.push(event)

    state = engine.state
    print(json.dumps({
        "total_events": state.total_events,
        "total_errors": state.total_errors,
        "trust_score": state.trust_score,
        "health": state.health,
        "diagnoses": [
            {
                "caft_code": d.caft_code,
                "failure_name": d.failure_name,
                "severity": d.severity.value,
                "confidence": d.confidence,
                "description": d.description,
                "evidence": d.evidence,
                "remediation": d.remediation,
            }
            for d in state.diagnoses
        ],
    }, indent=2, default=str))


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Unified evaluate: real traces, synthetic benchmark, splits, or ablation."""

    # --ablation: run 4-mode ablation study
    if getattr(args, "ablation", False):
        if not args.annotations:
            print("Error: --ablation requires --annotations", file=sys.stderr)
            sys.exit(1)
        import time as _time
        annotations_path = Path(args.annotations)
        if not annotations_path.exists():
            print(f"Error: Annotations file not found: {annotations_path}", file=sys.stderr)
            sys.exit(1)
        traces_root = Path(args.traces).expanduser()
        output_dir = Path(args.output_dir or f"results/ablation_{_time.strftime('%Y%m%d')}")
        detector_filter = set(args.detectors.split(",")) if args.detectors else None
        modes = args.modes.split(",") if args.modes else None
        splits_file = Path(args.splits_file) if getattr(args, "splits_file", None) else None

        # Web mode: launch ablation dashboard
        if getattr(args, "web", False):
            from agentdiag.visualize import start_ablation_server
            start_ablation_server(
                annotations_path=str(annotations_path),
                traces_root=str(traces_root),
                output_dir=str(output_dir),
                port=getattr(args, "port", 8080),
                split=args.split,
                splits_file=str(splits_file) if splits_file else None,
                llm_provider=args.llm_provider,
                match_window=args.match_window,
                bootstrap_n=args.bootstrap_n,
                skip_bootstrap=args.no_bootstrap,
                detector_filter=detector_filter,
                modes=modes,
                context_db=getattr(args, "context_db", None),
                verbose=True,
            )
            return

        # Batch mode (text-only)
        from scripts.run_ablation import run_ablation
        run_ablation(
            annotations_path=annotations_path,
            traces_root=traces_root,
            output_dir=output_dir,
            split=args.split,
            splits_file=splits_file,
            llm_provider=args.llm_provider,
            match_window=args.match_window,
            bootstrap_n=args.bootstrap_n,
            skip_bootstrap=args.no_bootstrap,
            detector_filter=detector_filter,
            modes=modes,
            dry_run=args.dry_run,
            context_db=getattr(args, "context_db", None),
        )
        return

    # --synthetic: run CAFT benchmark on synthetic traces
    if args.synthetic:
        from agentdiag.caft.benchmark import run_benchmark, print_report
        report = run_benchmark()
        print_report(report)
        sys.exit(0 if report.all_passed else 1)

    # --splits: manage train/val/test splits
    if args.splits:
        from agentdiag.splits import SplitManager
        sm = SplitManager(args.splits_file)

        if args.splits_init:
            traces_path = Path(args.traces).expanduser()
            if not traces_path.exists():
                print(f"Error: Traces path not found: {traces_path}", file=sys.stderr)
                sys.exit(1)
            summary = sm.auto_assign_claude_sessions(traces_path)
            print(f"Splits initialized and saved to {args.splits_file}")
            print(summary)
        else:
            print(sm.summary())
        return

    # Real trace evaluation (default)
    from agentdiag.evaluate import evaluate_claude_code, print_evaluation_report

    traces_path = Path(args.traces).expanduser()
    if not traces_path.exists():
        print(f"Error: Traces path not found: {traces_path}", file=sys.stderr)
        sys.exit(1)

    context_store = _make_context_store(args)

    try:
        report = evaluate_claude_code(
            traces_path=traces_path,
            session_id=args.session or None,
            min_lines=args.min_lines,
            max_sessions=args.n if args.n else None,
            context_store=context_store,
            annotations_path=args.annotations or None,
            split=args.split or None,
            splits_file=args.splits_file if args.split else None,
        )

        if args.json:
            print(report.to_json())
        else:
            print_evaluation_report(report)
    finally:
        if context_store is not None:
            context_store.close()


def cmd_calibrate(args: argparse.Namespace) -> None:
    from agentdiag.baselines import CalibrationPipeline

    traces_path = Path(args.traces).expanduser()
    if not traces_path.exists():
        print(f"Error: Traces path not found: {traces_path}", file=sys.stderr)
        sys.exit(1)

    if args.split == "test":
        print("WARNING: Calibrating on the test split risks data leakage.",
              file=sys.stderr)
        print("  Use --split validation (default) for calibration.",
              file=sys.stderr)

    splits_file = args.splits_file if args.splits_file else None

    pipeline = CalibrationPipeline()
    profile = pipeline.fit_from_traces_path(
        traces_path=traces_path,
        splits_file=splits_file,
        split=args.split,
        min_lines=args.min_lines,
    )

    output = Path(args.output)
    profile.save(output)

    print(f"Calibration complete:")
    print(f"  Sessions fitted:  {profile.n_sessions}")
    print(f"  Phase segments:   {profile.n_phase_segments}")
    print(f"  Transitions:      {profile.n_transitions}")
    print(f"  Phases modeled:   {', '.join(profile.phase_model.get_phases())}")
    print(f"  Profile saved to: {output}")

    if args.pilot:
        from agentdiag.caft.calibrated import make_calibrated_detectors
        from agentdiag.pilot import run_pilot, print_pilot_report

        cal_detectors = make_calibrated_detectors(profile)
        print(f"\n--- RAW DETECTORS (before calibration) ---")
        raw_report = run_pilot(traces_path, n=args.pilot_n, min_lines=args.min_lines)
        print_pilot_report(raw_report)

        print(f"\n--- CALIBRATED DETECTORS (after calibration) ---")
        cal_report = run_pilot(
            traces_path, n=args.pilot_n, min_lines=args.min_lines,
            detectors=cal_detectors,
        )
        print_pilot_report(cal_report)

        # Before/after comparison
        print(f"\n{'=' * 80}")
        print(f"  BEFORE vs AFTER COMPARISON")
        print(f"{'=' * 80}")

        all_detectors = set(raw_report.detector_counts) | set(cal_report.detector_counts)
        n_parsed = max(raw_report.n_parsed, 1)

        print(f"\n  {'Detector':<30} {'Before':>8} {'After':>8} {'Change':>10}")
        print(f"  {'─' * 60}")
        for det in sorted(all_detectors):
            before = raw_report.detector_counts.get(det, 0)
            after = cal_report.detector_counts.get(det, 0)
            before_pct = before / n_parsed
            after_pct = after / n_parsed
            change = after_pct - before_pct
            arrow = "↓" if change < 0 else ("↑" if change > 0 else "=")
            print(f"  {det:<30} {before_pct:>7.0%} {after_pct:>7.0%} {arrow} {abs(change):>7.0%}")

        print(f"\n  {'Metric':<30} {'Before':>8} {'After':>8}")
        print(f"  {'─' * 50}")
        print(f"  {'Clean sessions':<30} {raw_report.n_clean:>8} {cal_report.n_clean:>8}")
        print(f"  {'Detections':<30} {raw_report.n_detections:>8} {cal_report.n_detections:>8}")
        print(f"  {'HTA plausible':<30} {raw_report.n_hta_plausible:>8} {cal_report.n_hta_plausible:>8}")
        print()


def cmd_annotate(args: argparse.Namespace) -> None:
    """Annotation workflow: manual and automated."""
    action = getattr(args, "annotate_action", None)
    if not action:
        print("Usage: agentdiag annotate {queue,show,adjudicate,export-gold,stats,"
              "import-gt,auto-prepare,auto-merge,auto-validate}")
        print("Run 'agentdiag annotate -h' for help.")
        sys.exit(1)

    # ── Auto-annotate sub-actions ──────────────────────────────────
    if action.startswith("auto-"):
        _handle_auto_annotate(args, action)
        return

    # ── Manual annotation sub-actions ──────────────────────────────
    ledger_path = Path(args.ledger)

    if action == "queue":
        from agentdiag.annotation_store import AnnotationLedger
        from agentdiag.disagreement import rank_annotation_queue

        ledger = AnnotationLedger(ledger_path) if ledger_path.exists() else None

        # Build records_by_session from ledger
        records_by_session: dict = {}
        if ledger:
            for rec in ledger.get_all():
                sid = rec.effective_session_id
                if sid not in records_by_session:
                    records_by_session[sid] = []
                records_by_session[sid].append(rec)

        # Also include completely unlabeled sessions from manifest if available
        manifest_path = Path(getattr(args, "manifest", "data/manifest.csv"))
        if manifest_path.exists():
            import csv
            with open(manifest_path) as f:
                for row in csv.DictReader(f):
                    sid = row.get("session_id", "")[:8]
                    if sid and sid not in records_by_session:
                        records_by_session[sid] = []

        # Compute failure counts for novelty scoring
        failure_counts: dict[str, int] = {}
        if ledger:
            for rec in ledger.get_all():
                if rec.has_failure and rec.primary_caft_code:
                    name = rec.primary_caft_name or rec.primary_caft_code
                    failure_counts[name] = failure_counts.get(name, 0) + 1

        queue = rank_annotation_queue(records_by_session, failure_counts,
                                       limit=args.limit)

        print(f"\n{'=' * 70}")
        print(f"  ANNOTATION QUEUE ({len(queue)} sessions need review)")
        print(f"{'=' * 70}")
        print(f"  {'Session':<12} {'Score':>6} {'Reasons'}")
        print(f"  {'─' * 65}")
        for p in queue:
            reasons = ", ".join(p.reasons[:3]) if p.reasons else "—"
            print(f"  {p.session_id[:10]:<12} {p.score:>5.1f}  {reasons}")
        print()

    elif action == "show":
        from agentdiag.annotation_store import AnnotationLedger
        from agentdiag.disagreement import compute_session_disagreement_bundle

        session_id = args.session_id
        if not ledger_path.exists():
            print(f"No annotation ledger at {ledger_path}", file=sys.stderr)
            sys.exit(1)

        ledger = AnnotationLedger(ledger_path)
        records = ledger.get_for_session(session_id)

        if not records:
            print(f"No annotations for session {session_id}")
            sys.exit(0)

        print(f"\n{'=' * 70}")
        print(f"  ANNOTATIONS FOR {session_id}")
        print(f"{'=' * 70}")

        for rec in records:
            failure = rec.primary_caft_name if rec.has_failure else "clean"
            print(f"\n  [{rec.annotator_type}] by {rec.annotator_id}")
            print(f"    Status:     {rec.label_status}")
            print(f"    Failure:    {failure}")
            if rec.primary_caft_code:
                print(f"    CAFT code:  {rec.primary_caft_code}")
            if rec.severity:
                print(f"    Severity:   {rec.severity}/5")
            if rec.confidence:
                print(f"    Confidence: {rec.confidence}/5")
            if rec.free_text_rationale:
                rationale = rec.free_text_rationale[:200]
                print(f"    Rationale:  {rationale}")

        bundle = compute_session_disagreement_bundle(session_id, records)
        if bundle.any_disagreement:
            print(f"\n  DISAGREEMENTS ({bundle.total_disagreements})")
            for name in ["detector_vs_auto", "auto_vs_human",
                         "human_vs_adjudicated", "detector_vs_human"]:
                d = getattr(bundle, name)
                if d and d.has_disagreement:
                    print(f"    {name}: {d.description}")
        print()

    elif action == "adjudicate":
        from agentdiag.annotation_models import build_adjudicated_annotation
        from agentdiag.annotation_store import AnnotationLedger

        session_id = args.session_id
        ledger = AnnotationLedger(ledger_path)

        has_failure = bool(args.primary)
        ann = build_adjudicated_annotation(
            session_id=session_id,
            adjudicator_id=args.adjudicator or "human",
            has_failure=has_failure,
            primary_caft_code=args.primary or "",
            secondary_caft_codes=args.secondary.split(",") if args.secondary else [],
            severity=args.severity or 0,
            rationale=args.rationale or "",
        )
        ledger.add(ann)
        label = ann.primary_caft_name if has_failure else "clean"
        print(f"Adjudicated {session_id} → {label} (status=adjudicated)")

    elif action == "export-gold":
        from agentdiag.annotation_store import AnnotationLedger

        if not ledger_path.exists():
            print(f"No annotation ledger at {ledger_path}", file=sys.stderr)
            sys.exit(1)

        ledger = AnnotationLedger(ledger_path)
        status = args.status or "adjudicated"

        if status == "adjudicated":
            records = ledger.get_gold_annotations()
        elif status == "trainable":
            records = ledger.get_trainable_annotations()
        elif status == "eval":
            records = ledger.get_eval_annotations()
        else:
            records = ledger.get_by_status(status)

        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            for rec in records:
                f.write(rec.to_json() + "\n")

        print(f"Exported {len(records)} {status} annotations → {output}")

    elif action == "stats":
        from agentdiag.annotation_store import AnnotationLedger

        if not ledger_path.exists():
            print(f"No annotation ledger at {ledger_path}")
            sys.exit(0)

        ledger = AnnotationLedger(ledger_path)
        s = ledger.stats()

        print(f"\n{'=' * 50}")
        print(f"  ANNOTATION LEDGER STATS")
        print(f"{'=' * 50}")
        print(f"  Total records:    {s['total_records']}")
        print(f"  Unique sessions:  {s['unique_sessions']}")
        print(f"  Gold labels:      {s['gold_count']}")
        print(f"  Trainable labels: {s['trainable_count']}")

        if s["by_annotator_type"]:
            print(f"\n  By annotator type:")
            for t, c in sorted(s["by_annotator_type"].items()):
                print(f"    {t:<15} {c:>4}")

        if s["by_label_status"]:
            print(f"\n  By label status:")
            for st, c in sorted(s["by_label_status"].items()):
                print(f"    {st:<20} {c:>4}")

        if s["by_failure_type"]:
            print(f"\n  Failure distribution:")
            for name, c in sorted(s["by_failure_type"].items(), key=lambda x: -x[1]):
                print(f"    {name:<30} {c:>4}")
        print()

    elif action == "import-gt":
        from agentdiag.annotation_models import from_ground_truth_file
        from agentdiag.annotation_store import AnnotationLedger

        gt_path = Path(args.ground_truth_file)
        if not gt_path.exists():
            print(f"Error: File not found: {gt_path}", file=sys.stderr)
            sys.exit(1)

        with open(gt_path) as f:
            gt = json.load(f)

        records = from_ground_truth_file(gt)
        ledger = AnnotationLedger(ledger_path)
        count = ledger.merge_records(records)
        print(f"Imported {count} new records from {gt_path} "
              f"({len(records)} total, {len(records) - count} deduped) → {ledger_path}")

    else:
        print(f"Error: Unknown annotate action '{action}'", file=sys.stderr)
        sys.exit(1)


def _handle_auto_annotate(args: argparse.Namespace, action: str) -> None:
    """Automated CAFT trace annotation pipeline (sub-actions of annotate)."""
    from agentdiag.auto_annotate import (
        prepare_batch, parse_annotation_response,
        merge_annotations, validate_agreement,
    )

    if action == "auto-prepare":
        manifest = Path(args.manifest)
        if not manifest.exists():
            print(f"Error: Manifest not found: {manifest}", file=sys.stderr)
            sys.exit(1)
        gt_path = Path(args.ground_truth) if getattr(args, "ground_truth", None) else None
        traces_root = Path(args.traces).expanduser()
        prepare_batch(
            manifest_path=manifest,
            ground_truth_path=gt_path,
            traces_root=traces_root,
            n=args.n,
            output_path=Path(args.output),
        )

    elif action == "auto-merge":
        merge_file = getattr(args, "merge_file", None)
        if not merge_file:
            print("Error: auto-merge requires a FILE argument.", file=sys.stderr)
            sys.exit(1)
        merge_input = Path(merge_file)
        if not merge_input.exists():
            print(f"Error: File not found: {merge_input}", file=sys.stderr)
            sys.exit(1)
        with open(merge_input) as f:
            text = f.read()
        annotations = parse_annotation_response(text)
        into_path = Path(args.into) if getattr(args, "into", None) else None
        merge_annotations(
            new_annotations=annotations,
            existing_path=into_path,
            output_path=Path(args.output),
        )

    elif action == "auto-validate":
        auto_path = Path(args.auto)
        manual_path = Path(args.manual)
        if not auto_path.exists():
            print(f"Error: File not found: {auto_path}", file=sys.stderr)
            sys.exit(1)
        if not manual_path.exists():
            print(f"Error: File not found: {manual_path}", file=sys.stderr)
            sys.exit(1)
        result = validate_agreement(auto_path, manual_path)
        if getattr(args, "json", False):
            print(json.dumps(result, indent=2))


def cmd_taxonomy(args: argparse.Namespace) -> None:
    from agentdiag.caft.taxonomy import (
        CAFT_TAXONOMY, get_categories, get_category_types,
        get_observable_types, get_latent_types, Detectability,
    )

    obs = get_observable_types()
    lat = get_latent_types()
    print(f"\nCAFT Taxonomy: {len(CAFT_TAXONOMY)} types "
          f"({len(obs)} observable, {len(lat)} latent)")

    for cat in get_categories():
        types = get_category_types(cat)
        if args.observable_only:
            types = [t for t in types if t.detectability == Detectability.OBSERVABLE]
            if not types:
                continue
        print(f"\n  {cat.upper()}")
        for t in types:
            det = "OBS" if t.detectability == Detectability.OBSERVABLE else "LAT"
            detectors = ", ".join(t.detector_names) if t.detector_names else "—"
            print(f"    [{t.code}] {t.name:<30} [{det}]  detectors: {detectors}")
            print(f"          {t.description}")
    print()


def cmd_context(args: argparse.Namespace) -> None:
    from agentdiag.context import get_context_store

    action = getattr(args, "context_action", None)
    if not action:
        print("Usage: agentdiag context {search,stats}")
        print("Run 'agentdiag context -h' for help.")
        sys.exit(1)

    db_path = args.context_db or "./agentdiag_context"
    store = get_context_store(db_path)
    if store is None:
        print("Error: Could not initialize context store. "
              "Is openviking installed?", file=sys.stderr)
        sys.exit(1)

    try:
        action = args.context_action

        if action == "search":
            results = store.search(args.query, limit=args.limit)
            if not results:
                stats = store.get_stats()
                if not stats.get("search_available"):
                    print("Search requires an embedding API key.")
                    print("Set OPENAI_API_KEY or JINA_API_KEY and try again.")
                else:
                    print("No results found.")
                return
            for i, r in enumerate(results, 1):
                if isinstance(r, dict):
                    print(f"\n--- Result {i} ---")
                    for k, v in r.items():
                        val = str(v)
                        if len(val) > 200:
                            val = val[:200] + "..."
                        print(f"  {k}: {val}")
                else:
                    print(f"\n--- Result {i} ---")
                    print(f"  {r}")

        elif action == "stats":
            stats = store.get_stats()
            print(f"\nContext Database Statistics")
            print(f"{'=' * 40}")
            print(f"  DB path:         {stats.get('db_path', '?')}")
            print(f"  Sessions:        {stats.get('session_count', 0)}")
            print(f"  Active session:  {stats.get('active_session') or 'none'}")
            print(f"  Buffered events: {stats.get('buffered_events', 0)}")
            print(f"  Promoted cases:  {stats.get('promoted_cases', 0)}")
            print(f"  Search:          {'available' if stats.get('search_available') else 'unavailable (set OPENAI_API_KEY or JINA_API_KEY)'}")
            print(f"  Healthy:         {stats.get('healthy', '?')}")
            errors = stats.get("errors", [])
            if errors:
                print(f"  Errors:")
                for e in errors:
                    print(f"    - {e}")

        elif action == "cases":
            status_filter = getattr(args, "status", None)
            cases = store.load_cases(status_filter=status_filter)
            if not cases:
                print("No cases found." + (f" (filter: {status_filter})" if status_filter else ""))
                return

            print(f"\n{'=' * 80}")
            print(f"  DIAGNOSTIC CASES ({len(cases)} total"
                  f"{f', filtered: {status_filter}' if status_filter else ''})")
            print(f"{'=' * 80}")

            hdr = f"  {'Case ID':<40} {'Detector':<22} {'Sev':<9} {'Status':<14}"
            print(hdr)
            print(f"  {'─' * 76}")

            for case in cases:
                cid = case.get("case_id", "?")
                if len(cid) > 38:
                    cid = cid[:35] + "..."
                name = case.get("failure_name", "?")
                sev = case.get("severity", "?")
                status = case.get("status", "predicted")
                print(f"  {cid:<40} {name:<22} {sev:<9} {status:<14}")
            print()

        elif action == "feedback":
            case_id = args.case_id
            new_status = args.status
            notes = getattr(args, "notes", "") or ""

            ok = store.update_case_status(
                case_id=case_id,
                new_status=new_status,
                reviewer="human",
                notes=notes,
            )
            if ok:
                print(f"Updated case {case_id} → {new_status}")
                # Show current FP rates
                fp_rates = store.get_detector_fp_rates()
                if fp_rates:
                    print(f"\nUpdated detector FP rates:")
                    for name, rate in sorted(fp_rates.items()):
                        print(f"  {name:<30} {rate:.0%}")
            else:
                print(f"Error: Case '{case_id}' not found in ledger.", file=sys.stderr)
                sys.exit(1)

        elif action == "feedback-summary":
            summary = store.get_feedback_summary()
            total = summary["total_cases"]
            status_counts = summary["status_counts"]
            detector_stats = summary["detector_stats"]

            print(f"\n{'=' * 70}")
            print(f"  FEEDBACK SUMMARY — {total} total cases")
            print(f"{'=' * 70}")

            print(f"\n  Status breakdown:")
            for s, c in sorted(status_counts.items()):
                print(f"    {s:<18} {c:>4}")

            if detector_stats:
                print(f"\n  {'Detector':<25} {'Total':>5} {'Pred':>5} "
                      f"{'Conf':>5} {'FP':>5} {'Corr':>5} {'FP Rate':>8}")
                print(f"  {'─' * 64}")
                for name, stats in sorted(detector_stats.items()):
                    print(f"  {name:<25} {stats['total']:>5} "
                          f"{stats['predicted']:>5} {stats['confirmed']:>5} "
                          f"{stats['false_positive']:>5} {stats['corrected']:>5} "
                          f"{stats['fp_rate']:>7.0%}")
            print()

        else:
            print(f"Error: Unknown context action '{action}'", file=sys.stderr)
            sys.exit(1)
    finally:
        store.close()


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _make_context_store(args: argparse.Namespace):
    """Create a ContextStore from --context-db flag if provided."""
    db_path = getattr(args, "context_db", None)
    if not db_path:
        return None
    from agentdiag.context import get_context_store
    store = get_context_store(db_path)
    if store is None:
        print("Warning: Could not initialize context store "
              "(is openviking installed?)", file=sys.stderr)
    return store


# ═══════════════════════════════════════════════════════════════════════
# Backward-compat handlers (for hidden aliases)
# ═══════════════════════════════════════════════════════════════════════

def _cmd_live(args: argparse.Namespace) -> None:
    """Dispatch to the live observation module."""
    from agentdiag import live
    live._run_from_args(args)


def _cmd_harness(args: argparse.Namespace) -> None:
    """Run the three-agent harness with mock agents and visualization."""
    import json
    import random
    import threading
    import time
    from pathlib import Path

    prompt = args.prompt
    if not prompt:
        print("Usage: python -m agentdiag harness \"<task prompt>\"")
        sys.exit(1)

    port = getattr(args, "port", 8080)
    no_browser = getattr(args, "no_browser", False)
    output = getattr(args, "output", None)
    max_sprints = getattr(args, "max_sprints", 3)
    speed = getattr(args, "speed", 1.0)

    from agentdiag.context.instrumented import InstrumentedContextStore
    from agentdiag.harness import (
        HarnessOrchestrator, SprintContract, EvaluationGrade,
    )
    from agentdiag.observable import ObservableEvent
    from agentdiag.live import _LiveQueueStream

    # ── Mock agents ───────────────────────────────────────────────
    rng = random.Random(42)

    def mock_planner(goal: str, context: dict) -> list[dict]:
        """Split the goal into sprint specs based on the prompt."""
        words = goal.split()
        n_sprints = min(max_sprints, max(1, len(words) // 5))
        sprints = []
        for i in range(1, n_sprints + 1):
            sprints.append({
                "goal": f"Sprint {i}: {goal}" if n_sprints == 1 else f"Sprint {i}/{n_sprints} for: {goal}",
                "deliverables": [f"deliverable_{i}_{j}" for j in range(rng.randint(2, 4))],
                "success_criteria": [f"criterion_{i}_{j}" for j in range(rng.randint(2, 4))],
            })
        return sprints

    def mock_generator(
        contract: SprintContract,
        context: dict,
        feedback=None,
    ) -> dict:
        """Simulate code generation with realistic delay."""
        delay_sec = rng.uniform(0.5, 2.0) / speed
        time.sleep(delay_sec)
        return {
            "files": [f"src/{d}.py" for d in contract.deliverables],
            "iteration": feedback.iteration + 1 if feedback else 1,
        }

    def mock_evaluator(
        contract: SprintContract,
        artifacts: dict,
        context: dict,
    ) -> EvaluationGrade:
        """Simulate evaluation with realistic scoring."""
        delay_sec = rng.uniform(0.3, 1.0) / speed
        time.sleep(delay_sec)
        base_score = rng.uniform(0.6, 0.95)
        criteria = {}
        for c in contract.success_criteria:
            criteria[c] = round(base_score + rng.uniform(-0.1, 0.1), 3)
            criteria[c] = max(0.0, min(1.0, criteria[c]))
        overall = sum(criteria.values()) / len(criteria) if criteria else base_score
        return EvaluationGrade(
            sprint_number=contract.sprint_number,
            overall_score=round(overall, 3),
            criteria_scores=criteria,
            passed=overall >= 0.7,
            critique=f"Reviewed {len(artifacts.get('files', []))} files." if overall < 0.7 else "",
        )

    def mock_negotiator(contract: SprintContract, context: dict) -> SprintContract:
        """Simulate contract amendment."""
        if rng.random() < 0.3:
            contract.evaluator_amendments = "Added testability clause"
            contract.success_criteria.append("must_be_testable")
            contract.status = "amended"
        return contract

    # ── Event collection + viz stream bridge ──────────────────────
    stream = _LiveQueueStream()
    all_events: list[dict] = []
    step_counter = [0]

    def event_sink(event: ObservableEvent) -> None:
        """Receives events from both the InstrumentedContextStore and HarnessOrchestrator."""
        event_dict = event.to_dict()
        all_events.append(event_dict)
        # Bridge to visualization: convert to TraceEvent-compatible dict
        step_counter[0] += 1
        trace_event = {
            "step": step_counter[0],
            "type": event_dict.get("event_type", "tool_call"),
            "tool": event_dict.get("tool_name") or event_dict.get("event_type", "harness"),
            "latency_ms": event_dict.get("duration_ms", 0.0),
            "success": True,
            "tokens_in": event_dict.get("input_tokens", 0) or event_dict.get("token_count", 0) or 0,
            "tokens_out": event_dict.get("output_tokens", 0) or 0,
            "timestamp": event_dict.get("timestamp", time.time()),
            "goal_text": event_dict.get("symbol", ""),
        }
        stream.write_event(trace_event)

    # ── Create the harness ────────────────────────────────────────
    store = InstrumentedContextStore(
        db_path="./harness_context",
        on_event=event_sink,
    )

    orch = HarnessOrchestrator(
        context_store=store,
        planner=mock_planner,
        generator=mock_generator,
        evaluator=mock_evaluator,
        contract_negotiator=mock_negotiator,
        on_event=event_sink,
        pass_threshold=0.7,
    )

    # ── Run harness in background thread, viz in main thread ──────
    harness_result_holder: list = []

    def run_harness():
        try:
            result = orch.run(goal=prompt, max_sprints=max_sprints)
            harness_result_holder.append(result)
        except Exception as e:
            print(f"\nHarness error: {e}", file=sys.stderr)
        finally:
            time.sleep(1.0)  # let final events flush
            stream.close()

    harness_thread = threading.Thread(target=run_harness, daemon=True)

    print(f"Harness: \"{prompt}\"")
    print(f"Harness: {max_sprints} max sprints, mock agents")
    print(f"Harness: visualization at http://localhost:{port}")
    print()

    harness_thread.start()

    if not no_browser:
        def _open():
            time.sleep(1.5)
            import webbrowser
            webbrowser.open(f"http://localhost:{port}")
        threading.Thread(target=_open, daemon=True).start()

    try:
        from agentdiag.visualize import start_server
    except ImportError:
        print("Visualization requires: pip install uvicorn fastapi")
        sys.exit(1)

    start_server(
        stream=stream,
        goal=f"Harness: {prompt}",
        port=port,
        delay=0.0,
        cognitive=True,
        input_path="harness_run",
    )

    # ── Save result after viz server shuts down ───────────────────
    harness_thread.join(timeout=5.0)

    if harness_result_holder:
        result = harness_result_holder[0]
        result_dict = {
            "goal": result.goal,
            "overall_passed": result.overall_passed,
            "total_iterations": result.total_iterations,
            "duration_sec": result.duration_sec,
            "sprints": [
                {
                    "sprint_number": s.sprint_number,
                    "contract": s.contract.to_dict(),
                    "grades": [g.to_dict() for g in s.grades],
                    "iterations": s.iterations,
                    "final_passed": s.final_passed,
                }
                for s in result.sprints
            ],
        }

        out_path = output or "harness_result.json"
        Path(out_path).write_text(
            json.dumps(result_dict, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"\nHarness result saved to {out_path}")
        print(f"  Passed: {result.overall_passed}")
        print(f"  Sprints: {len(result.sprints)}")
        print(f"  Iterations: {result.total_iterations}")
        print(f"  Duration: {result.duration_sec:.1f}s")


def _cmd_analyze_compat(args: argparse.Namespace) -> None:
    """Backward-compatible analyze → monitor --input FILE --json."""
    from agentdiag.loading import load_trace
    from agentdiag.monitor import MonitorEngine

    path = Path(args.trace)
    if not path.exists():
        print(f"Error: File not found: {path}", file=sys.stderr)
        sys.exit(1)

    events = load_trace(path)
    if not events:
        print("Error: No events found in trace file.", file=sys.stderr)
        sys.exit(1)

    engine = MonitorEngine(goal="")
    for event in events:
        engine.push(event)

    state = engine.state

    if getattr(args, "json", False):
        print(json.dumps({
            "trace_id": path.stem,
            "total_events": state.total_events,
            "total_errors": state.total_errors,
            "trust_score": state.trust_score,
            "health": state.health,
            "diagnoses": [
                {
                    "caft_code": d.caft_code,
                    "failure_name": d.failure_name,
                    "severity": d.severity.value,
                    "confidence": d.confidence,
                    "description": d.description,
                    "evidence": d.evidence,
                    "remediation": d.remediation,
                }
                for d in state.diagnoses
            ],
        }, indent=2, default=str))
    else:
        if not state.diagnoses:
            print(f"Agent trace appears healthy. "
                  f"{state.total_events} events, "
                  f"{state.total_errors} errors, "
                  f"trust score {state.trust_score:.0%}.")
        else:
            print(f"DIAGNOSTIC SUMMARY: {len(state.diagnoses)} failure mode(s) detected "
                  f"in {state.total_events}-event trace.")
            print(f"\nHealth: {state.health.upper()}  Trust: {state.trust_score:.0%}")
            for d in state.diagnoses:
                print(f"\n  [{d.caft_code}] {d.failure_name} "
                      f"({d.severity.value}, confidence={d.confidence:.0%})")
                print(f"    {d.description}")
                if d.evidence:
                    for k, v in d.evidence.items():
                        print(f"    {k}: {v}")


def _cmd_demo_dispatch(args: argparse.Namespace) -> None:
    """Route demo command: Agent 2 visualization demo or legacy CAFT demo."""
    scenario = getattr(args, "scenario", None)
    if scenario is not None:
        # Legacy CAFT demo with scenario name
        _handle_demo(args, scenario)
    else:
        # Agent 2's visualization demo — call with pre-parsed args
        from agentdiag.demo import _run_json, _run_server
        if getattr(args, "json", False):
            _run_json(args.speed)
        else:
            _run_server(args.port, args.speed, not getattr(args, "no_browser", False))


def _cmd_demo_compat(args: argparse.Namespace) -> None:
    """Backward-compatible demo → monitor --demo."""
    _handle_demo(args, getattr(args, "scenario", "all"))


def _cmd_visualize_compat(args: argparse.Namespace) -> None:
    """Backward-compatible visualize → monitor --web."""
    from agentdiag.visualize import start_server

    if args.input == "stdin":
        stream = sys.stdin
    else:
        path = Path(args.input)
        if not path.exists():
            print(f"Error: File not found: {path}", file=sys.stderr)
            sys.exit(1)
        stream = open(path, "r")

    context_store = _make_context_store(args)

    try:
        start_server(
            stream=stream,
            goal=args.goal or "",
            port=args.port,
            delay=args.delay,
            context_store=context_store,
            decision_trace=getattr(args, "decision_trace", False),
            cognitive=getattr(args, "cognitive", False),
        )
    finally:
        if context_store is not None:
            context_store.close()
        if stream is not sys.stdin:
            stream.close()


def _cmd_auto_annotate_compat(args: argparse.Namespace) -> None:
    """Backward-compatible auto-annotate → annotate auto-*."""
    from agentdiag.auto_annotate import (
        prepare_batch, parse_annotation_response,
        merge_annotations, validate_agreement,
    )

    if getattr(args, "prepare", False):
        manifest = Path(args.manifest)
        if not manifest.exists():
            print(f"Error: Manifest not found: {manifest}", file=sys.stderr)
            sys.exit(1)
        gt_path = Path(args.ground_truth) if args.ground_truth else None
        traces_root = Path(args.traces).expanduser()
        prepare_batch(
            manifest_path=manifest,
            ground_truth_path=gt_path,
            traces_root=traces_root,
            n=args.n,
            output_path=Path(args.output),
        )
    elif getattr(args, "merge", None):
        merge_input = Path(args.merge)
        if not merge_input.exists():
            print(f"Error: File not found: {merge_input}", file=sys.stderr)
            sys.exit(1)
        with open(merge_input) as f:
            text = f.read()
        annotations = parse_annotation_response(text)
        into_path = Path(args.into) if args.into else None
        merge_annotations(
            new_annotations=annotations,
            existing_path=into_path,
            output_path=Path(args.output),
        )
    elif getattr(args, "validate", False):
        auto_path = Path(args.auto)
        manual_path = Path(args.manual)
        if not auto_path.exists():
            print(f"Error: File not found: {auto_path}", file=sys.stderr)
            sys.exit(1)
        if not manual_path.exists():
            print(f"Error: File not found: {manual_path}", file=sys.stderr)
            sys.exit(1)
        result = validate_agreement(auto_path, manual_path)
        if getattr(args, "json", False):
            print(json.dumps(result, indent=2))
    else:
        print("Usage: agentdiag auto-annotate --prepare|--merge|--validate",
              file=sys.stderr)
        sys.exit(1)


def _cmd_pilot_compat(args: argparse.Namespace) -> None:
    """Backward-compatible pilot command."""
    from agentdiag.pilot import run_pilot, print_pilot_report
    traces_path = Path(args.traces).expanduser()
    if not traces_path.exists():
        print(f"Error: Traces path not found: {traces_path}", file=sys.stderr)
        sys.exit(1)
    report = run_pilot(traces_path=traces_path, n=args.n, min_lines=args.min_lines)
    if args.json:
        print(report.to_json())
    else:
        print_pilot_report(report)


def _cmd_splits_compat(args: argparse.Namespace) -> None:
    """Backward-compatible splits command."""
    from agentdiag.splits import SplitManager
    sm = SplitManager(args.file)
    if args.init:
        traces_path = Path(args.traces).expanduser()
        if not traces_path.exists():
            print(f"Error: Traces path not found: {traces_path}", file=sys.stderr)
            sys.exit(1)
        summary = sm.auto_assign_claude_sessions(traces_path)
        print(f"Splits initialized and saved to {args.file}")
        print(summary)
    else:
        print(sm.summary())


# ═══════════════════════════════════════════════════════════════════════
# Parser definition
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="agentdiag",
        description="Anomaly detection for AI agent execution traces",
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── monitor (primary: absorbs analyze, demo, visualize) ───────
    p_monitor = subparsers.add_parser(
        "monitor",
        help="Monitor traces: streaming TUI, plain, JSON analysis, demo, or web",
    )
    p_monitor.add_argument(
        "--input", default=None,
        help="Input source: 'stdin' or path to JSONL file (required unless --demo)",
    )
    p_monitor.add_argument("--goal", default="", help="Goal text for HTA tracking")
    p_monitor.add_argument("--plain", action="store_true", help="Plain text output (no TUI)")
    p_monitor.add_argument("--json", action="store_true",
                           help="JSON analysis output (load file, output summary)")
    p_monitor.add_argument("--demo", nargs="?", const="all", default=None,
                           help="Run CAFT demo with synthetic traces (scenario name or 'all')")
    p_monitor.add_argument("--web", action="store_true",
                           help="Web-based diagnostic map visualization")
    p_monitor.add_argument("--port", type=int, default=8080,
                           help="HTTP port for --web mode (default: 8080)")
    p_monitor.add_argument("--delay", type=float, default=0.3,
                           help="Seconds between events (demo/web modes)")
    p_monitor.add_argument("--context-db", default=None,
                           help="OpenViking context DB path")
    p_monitor.add_argument("--confirm", action="store_true",
                           help="Enable LLM confirmation of CAFT detections")
    p_monitor.add_argument("--compare", nargs=2, metavar=("TRACE_A", "TRACE_B"),
                           default=None,
                           help="Compare two trace files side-by-side (requires --web)")
    p_monitor.add_argument("--decision-trace", action="store_true",
                           help="Enable per-step decision trace recording (web mode)")
    p_monitor.add_argument("--cognitive", action="store_true",
                           help="Enable cognitive load monitoring (web mode)")
    p_monitor.set_defaults(func=cmd_monitor)

    # ── evaluate ─────────────────────────────────────────────────────
    p_eval = subparsers.add_parser(
        "evaluate",
        help="Evaluate traces (real, synthetic, or annotated)",
    )
    p_eval.add_argument(
        "--traces", default="~/.claude/projects",
        help="Path to trace files (default: ~/.claude/projects)",
    )
    p_eval.add_argument("--session", default=None, help="Evaluate single session (prefix match)")
    p_eval.add_argument("--n", type=int, default=None, help="Limit to N sessions (largest first)")
    p_eval.add_argument("--min-lines", type=int, default=5, help="Skip sessions < N lines")
    p_eval.add_argument("--json", action="store_true", help="Output as JSON")
    p_eval.add_argument("--context-db", default=None, help="OpenViking context DB path")
    p_eval.add_argument("--annotations", default=None, help="Path to annotations JSONL for precision/recall")
    p_eval.add_argument(
        "--synthetic", action="store_true",
        help="Run CAFT synthetic benchmark (precision/recall/F1 on synthetic traces)",
    )
    p_eval.add_argument("--splits", action="store_true", help="Manage train/val/test splits")
    p_eval.add_argument("--splits-init", action="store_true", help="Auto-assign sessions to splits")
    p_eval.add_argument("--splits-file", default="splits.json", help="Splits file path")
    p_eval.add_argument(
        "--split", default=None,
        help="Only evaluate sessions in this split (requires --splits-file). "
             "One of: development, validation, test.",
    )
    p_eval.add_argument(
        "--ablation", action="store_true",
        help="Run 4-mode ablation study (strict/loose/loose+llm/oracle)",
    )
    p_eval.add_argument("--output-dir", default=None, help="Output directory for ablation results")
    p_eval.add_argument("--match-window", type=int, default=5, help="Step tolerance for matching (ablation)")
    p_eval.add_argument("--bootstrap-n", type=int, default=1000, help="Bootstrap iterations (ablation)")
    p_eval.add_argument("--no-bootstrap", action="store_true", help="Skip bootstrap CI (ablation)")
    p_eval.add_argument("--detectors", default=None, help="Comma-separated detector filter (ablation)")
    p_eval.add_argument("--modes", default=None, help="Comma-separated modes to run (ablation)")
    p_eval.add_argument("--dry-run", action="store_true", help="Show plan without running (ablation)")
    p_eval.add_argument("--llm-provider", default=None, help="LLM provider for loose+llm mode (ablation)")
    p_eval.add_argument("--web", action="store_true", help="Launch web dashboard for ablation (use with --ablation)")
    p_eval.add_argument("--port", type=int, default=8080, help="Port for web dashboard (default: 8080)")
    p_eval.set_defaults(func=cmd_evaluate)

    # ── calibrate ────────────────────────────────────────────────────
    p_cal = subparsers.add_parser(
        "calibrate",
        help="Fit normative baselines from validation traces",
    )
    p_cal.add_argument("--traces", default="~/.claude/projects", help="Session logs path")
    p_cal.add_argument("--splits-file", default=None, help="Path to splits.json")
    p_cal.add_argument("--split", default="validation", help="Split to fit on")
    p_cal.add_argument("--output", default="baselines.json", help="Output profile path")
    p_cal.add_argument("--min-lines", type=int, default=10, help="Skip sessions < N lines")
    p_cal.add_argument("--pilot", action="store_true", help="Before/after comparison")
    p_cal.add_argument("--pilot-n", type=int, default=20, help="Traces for comparison")
    p_cal.set_defaults(func=cmd_calibrate)

    # ── annotate (primary: absorbs auto-annotate) ────────────────
    p_ann = subparsers.add_parser(
        "annotate",
        help="Annotation workflow (manual + automated)",
    )
    p_ann.add_argument("--ledger", default="annotations/annotation_ledger.jsonl",
                        help="Path to annotation ledger JSONL")
    ann_sub = p_ann.add_subparsers(dest="annotate_action")

    # annotate queue
    p_ann_queue = ann_sub.add_parser("queue", help="Show sessions needing annotation")
    p_ann_queue.add_argument("--limit", type=int, default=20, help="Max results")
    p_ann_queue.add_argument("--manifest", default="data/manifest.csv",
                              help="Manifest CSV for unlabeled session discovery")

    # annotate show
    p_ann_show = ann_sub.add_parser("show", help="Show all annotations for a session")
    p_ann_show.add_argument("session_id", help="Session ID (full or prefix)")

    # annotate adjudicate
    p_ann_adj = ann_sub.add_parser("adjudicate",
                                    help="Record adjudicated gold label")
    p_ann_adj.add_argument("session_id", help="Session ID to adjudicate")
    p_ann_adj.add_argument("--primary", default=None,
                            help="Primary CAFT code (e.g., '2.2'). Omit for clean.")
    p_ann_adj.add_argument("--secondary", default=None,
                            help="Comma-separated secondary CAFT codes")
    p_ann_adj.add_argument("--severity", type=int, default=None, help="Severity 1-5")
    p_ann_adj.add_argument("--rationale", default=None, help="Rationale text")
    p_ann_adj.add_argument("--adjudicator", default=None, help="Adjudicator ID")

    # annotate export-gold
    p_ann_export = ann_sub.add_parser("export-gold",
                                       help="Export trusted annotations")
    p_ann_export.add_argument("--status", default="adjudicated",
                               choices=["adjudicated", "trainable", "eval", "all"],
                               help="Which label status to export")
    p_ann_export.add_argument("--output", default="annotations/gold_labels.jsonl",
                               help="Output JSONL path")

    # annotate stats
    ann_sub.add_parser("stats", help="Show annotation ledger statistics")

    # annotate import-gt
    p_ann_import = ann_sub.add_parser("import-gt",
                                       help="Import ground_truth_*.json into ledger")
    p_ann_import.add_argument("ground_truth_file", help="Path to ground truth JSON")

    # annotate auto-prepare
    p_ann_ap = ann_sub.add_parser("auto-prepare",
                                   help="Prepare batch summaries for annotation")
    p_ann_ap.add_argument("--traces", default="~/.claude/projects",
                           help="Root path to search for session JSONL files")
    p_ann_ap.add_argument("--manifest", default="data/manifest.csv",
                           help="Path to manifest.csv")
    p_ann_ap.add_argument("--ground-truth", default=None,
                           help="Existing ground truth JSON (skip already-labeled)")
    p_ann_ap.add_argument("--n", type=int, default=30,
                           help="Max traces to include in batch")
    p_ann_ap.add_argument("--output", default="annotations/batch_summaries.json",
                           help="Output path")

    # annotate auto-merge
    p_ann_am = ann_sub.add_parser("auto-merge",
                                   help="Merge annotation results into ground truth")
    p_ann_am.add_argument("merge_file", help="File containing annotation results")
    p_ann_am.add_argument("--into", default=None,
                           help="Existing ground truth to merge into")
    p_ann_am.add_argument("--output", default="annotations/batch_summaries.json",
                           help="Output path")

    # annotate auto-validate
    p_ann_av = ann_sub.add_parser("auto-validate",
                                   help="Validate auto vs manual agreement")
    p_ann_av.add_argument("--auto", required=True,
                           help="Auto-annotated ground truth JSON")
    p_ann_av.add_argument("--manual", required=True,
                           help="Manual ground truth JSON")
    p_ann_av.add_argument("--json", action="store_true",
                           help="Output validation results as JSON")

    p_ann.set_defaults(func=cmd_annotate)

    # ── context ──────────────────────────────────────────────────────
    p_ctx = subparsers.add_parser(
        "context",
        help="Manage persistent diagnostic context (OpenViking)",
    )
    p_ctx.add_argument("--context-db", default=None, help="Context DB path")
    ctx_sub = p_ctx.add_subparsers(dest="context_action")
    p_ctx_search = ctx_sub.add_parser("search", help="Search past sessions")
    p_ctx_search.add_argument("query", help="Search query text")
    p_ctx_search.add_argument("--limit", type=int, default=10, help="Max results")
    ctx_sub.add_parser("stats", help="Show context DB statistics")

    p_ctx_cases = ctx_sub.add_parser("cases", help="List promoted diagnostic cases")
    p_ctx_cases.add_argument(
        "--status", default=None,
        help="Filter by status: predicted, confirmed, false_positive, corrected",
    )

    p_ctx_fb = ctx_sub.add_parser("feedback", help="Update a case's review status")
    p_ctx_fb.add_argument("case_id", help="Case ID to update")
    p_ctx_fb.add_argument(
        "--status", required=True,
        choices=["confirmed", "false_positive", "corrected"],
        help="New status for the case",
    )
    p_ctx_fb.add_argument("--notes", default="", help="Resolution notes")

    ctx_sub.add_parser("feedback-summary", help="Show feedback statistics and FP rates")

    p_ctx.set_defaults(func=cmd_context)

    # ── taxonomy ─────────────────────────────────────────────────────
    p_tax = subparsers.add_parser(
        "taxonomy",
        help="Show the CAFT taxonomy (32 types, observable/latent)",
    )
    p_tax.add_argument("--observable-only", action="store_true", help="Observable types only")
    p_tax.set_defaults(func=cmd_taxonomy)

    # ── live — real-time Claude Code observation ───────────────────
    p_live = subparsers.add_parser(
        "live",
        help="Watch running Claude Code sessions in real-time",
    )
    p_live.add_argument("--session", type=str, default=None,
                        help="Path to specific JSONL trace file")
    p_live.add_argument("--project", type=str, default=".",
                        help="Project directory (default: cwd)")
    p_live.add_argument("--all-sessions", action="store_true",
                        help="Watch all active sessions")
    p_live.add_argument("--replay", type=str, default=None,
                        help="Replay a past session JSONL file")
    p_live.add_argument("--replay-harness", type=str, default=None,
                        help="Replay a saved HarnessResult JSON file")
    p_live.add_argument("--speed", type=float, default=1.0,
                        help="Replay speed multiplier")
    p_live.add_argument("--port", type=int, default=8080,
                        help="Visualization server port")
    p_live.add_argument("--no-browser", action="store_true",
                        help="Don't auto-open browser")
    p_live.add_argument("--z-threshold", type=float, default=3.0,
                        help="Anomaly detection sensitivity")
    p_live.add_argument("--log-validation", type=str, default=None,
                        help="Path to validation log JSONL (enables human mark mode)")
    p_live.set_defaults(func=lambda args: _cmd_live(args))

    # ── harness — orchestrated harness mode (placeholder) ────────
    p_harness = subparsers.add_parser(
        "harness",
        help="Run full planner/generator/evaluator harness",
    )
    p_harness.add_argument("prompt", nargs="?", default=None,
                           help="Task prompt for the harness")
    p_harness.add_argument("--max-sprints", type=int, default=3,
                           help="Maximum number of sprints (default: 3)")
    p_harness.add_argument("--port", type=int, default=8080,
                           help="Visualization server port (default: 8080)")
    p_harness.add_argument("--no-browser", action="store_true",
                           help="Don't auto-open browser")
    p_harness.add_argument("--output", type=str, default=None,
                           help="Output path for HarnessResult JSON (default: harness_result.json)")
    p_harness.add_argument("--speed", type=float, default=1.0,
                           help="Mock agent speed multiplier (default: 1.0)")
    p_harness.set_defaults(func=lambda args: _cmd_harness(args))

    # ── Backward compatibility aliases (hidden from help) ─────────
    # analyze → monitor --input FILE --json
    p_analyze = subparsers.add_parser("analyze")
    p_analyze.add_argument("trace", help="Path to trace file (.json or .jsonl)")
    p_analyze.add_argument("--json", action="store_true", help="Output as JSON")
    p_analyze.set_defaults(func=_cmd_analyze_compat)

    # demo → monitor --demo
    p_demo = subparsers.add_parser("demo",
                                   help="Run synthetic demo with planted anomalies")
    p_demo.add_argument("scenario", nargs="?", default=None)
    p_demo.add_argument("--port", type=int, default=8080, help="Server port")
    p_demo.add_argument("--speed", type=float, default=2.0, help="Events per second")
    p_demo.add_argument("--json", action="store_true", help="JSON to stdout, no server")
    p_demo.add_argument("--no-browser", action="store_true", help="Don't open browser")
    p_demo.add_argument("--delay", type=float, default=0.3)
    p_demo.add_argument("--plain", action="store_true")
    p_demo.set_defaults(func=_cmd_demo_dispatch)

    # visualize → monitor --web
    p_viz = subparsers.add_parser("visualize")
    p_viz.add_argument("--input", required=True, help="Input: 'stdin' or JSONL path")
    p_viz.add_argument("--goal", default="", help="Goal text for HTA")
    p_viz.add_argument("--port", type=int, default=8080, help="HTTP port")
    p_viz.add_argument("--delay", type=float, default=0.0, help="Replay pacing (seconds)")
    p_viz.add_argument("--context-db", default=None, help="OpenViking context DB path")
    p_viz.set_defaults(func=_cmd_visualize_compat)

    # auto-annotate → annotate auto-*
    p_aa = subparsers.add_parser("auto-annotate")
    p_aa.add_argument("--prepare", action="store_true")
    p_aa.add_argument("--merge", default=None, metavar="FILE")
    p_aa.add_argument("--validate", action="store_true")
    p_aa.add_argument("--traces", default="~/.claude/projects")
    p_aa.add_argument("--manifest", default="data/manifest.csv")
    p_aa.add_argument("--ground-truth", default=None)
    p_aa.add_argument("--n", type=int, default=30)
    p_aa.add_argument("--output", default="annotations/batch_summaries.json")
    p_aa.add_argument("--into", default=None)
    p_aa.add_argument("--auto", default=None)
    p_aa.add_argument("--manual", default=None)
    p_aa.add_argument("--json", action="store_true")
    p_aa.set_defaults(func=_cmd_auto_annotate_compat)

    # monitor-demo → monitor --demo
    p_mdemo = subparsers.add_parser("monitor-demo")
    p_mdemo.add_argument("scenario", nargs="?", default="all")
    p_mdemo.add_argument("--delay", type=float, default=0.3)
    p_mdemo.add_argument("--plain", action="store_true")
    p_mdemo.set_defaults(func=_cmd_demo_compat)

    # validate-caft → evaluate --synthetic
    p_vcaft = subparsers.add_parser("validate-caft")
    p_vcaft.set_defaults(func=lambda args: cmd_evaluate(args), synthetic=True,
                         traces="~/.claude/projects", session=None, n=None,
                         min_lines=5, json=False, context_db=None,
                         annotations=None, splits=False, splits_init=False,
                         splits_file="splits.json", split=None)

    # pilot
    p_pilot = subparsers.add_parser("pilot")
    p_pilot.add_argument("--traces", default="~/.claude/projects")
    p_pilot.add_argument("--n", type=int, default=20)
    p_pilot.add_argument("--min-lines", type=int, default=10)
    p_pilot.add_argument("--json", action="store_true")
    p_pilot.set_defaults(func=_cmd_pilot_compat)

    # splits
    p_splits = subparsers.add_parser("splits")
    p_splits.add_argument("--traces", default="~/.claude/projects")
    p_splits.add_argument("--init", action="store_true")
    p_splits.add_argument("--file", default="splits.json")
    p_splits.set_defaults(func=_cmd_splits_compat)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
