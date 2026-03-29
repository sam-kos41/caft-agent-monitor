"""Unified evaluation pipeline for agentdiag.

Runs real traces through HTA + CAFT and produces structured reports
with per-session diagnostics, aggregate statistics, detector firing rates,
and optional annotation comparison (precision/recall/F1).

Merges functionality from the former evaluate, pilot, and annotation modules.

Usage:
    agentdiag evaluate --dataset claude-code --traces ~/.claude/projects
    agentdiag evaluate --dataset claude-code --traces ~/.claude/projects --json
    agentdiag evaluate --dataset claude-code --traces ~/.claude/projects --annotations labels.jsonl
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from agentdiag.models import TraceEvent
from agentdiag.monitor import MonitorEngine, DashboardState
from agentdiag.adapters.claude_code import (
    ClaudeCodeExtractor,
    SessionInfo,
    discover_sessions,
)
from agentdiag.caft.base import CaftDiagnosis, CaftSeverity
from agentdiag.hta import Phase


@dataclass
class SessionResult:
    """Evaluation result for a single session.

    Combines fields from the former SessionResult and PilotResult into
    one unified per-session record.
    """
    session_id: str
    project_dir: str
    line_count: int
    event_count: int
    first_timestamp: Optional[str]
    last_timestamp: Optional[str]

    # HTA
    final_phase: str
    progress_pct: float
    hta_regressions: int

    # CAFT
    diagnoses: list[dict]
    trust_score: float
    health: str

    # Stats
    total_errors: int
    tool_counts: dict[str, int]
    event_type_counts: dict[str, int]

    # Extended fields (from former pilot module)
    parse_ok: bool = True
    parse_error: Optional[str] = None
    hta_plausible: bool = True
    hta_notes: str = ""
    phases_visited: list[str] = field(default_factory=list)
    detectors_fired: list[str] = field(default_factory=list)
    classification: str = ""  # "clean", "real", "false_alarm", "ambiguous", "parser_issue"
    analysis_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DetectorMetrics:
    """Precision/recall/F1 for a single detector (when annotations provided)."""
    detector_name: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 1.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "detector_name": self.detector_name,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


@dataclass
class EvaluationReport:
    """Aggregate evaluation report across all sessions."""
    dataset: str
    traces_path: str
    total_sessions: int
    total_events: int
    session_results: list[SessionResult]

    # Aggregate CAFT stats
    sessions_healthy: int = 0
    sessions_degraded: int = 0
    sessions_failing: int = 0
    total_diagnoses: int = 0
    detector_firing_rates: dict[str, int] = field(default_factory=dict)
    failure_name_counts: dict[str, int] = field(default_factory=dict)
    severity_counts: dict[str, int] = field(default_factory=dict)

    # Aggregate HTA stats
    avg_progress: float = 0.0
    avg_trust: float = 0.0
    total_regressions: int = 0

    # Annotation comparison (populated when annotations provided)
    detector_metrics: dict[str, DetectorMetrics] = field(default_factory=dict)
    annotations_path: Optional[str] = None
    annotated_sessions: int = 0

    # Timing
    total_ms: float = 0.0
    avg_ms_per_session: float = 0.0

    # Parse stats
    parse_failures: int = 0
    hta_implausible: int = 0

    # Classification
    classification_counts: dict[str, int] = field(default_factory=dict)

    @property
    def macro_precision(self) -> float:
        vals = [m.precision for m in self.detector_metrics.values()
                if m.true_positives + m.false_positives + m.false_negatives > 0]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def macro_recall(self) -> float:
        vals = [m.recall for m in self.detector_metrics.values()
                if m.true_positives + m.false_positives + m.false_negatives > 0]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def macro_f1(self) -> float:
        vals = [m.f1 for m in self.detector_metrics.values()
                if m.true_positives + m.false_positives + m.false_negatives > 0]
        return sum(vals) / len(vals) if vals else 0.0

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "traces_path": self.traces_path,
            "total_sessions": self.total_sessions,
            "total_events": self.total_events,
            "sessions_healthy": self.sessions_healthy,
            "sessions_degraded": self.sessions_degraded,
            "sessions_failing": self.sessions_failing,
            "total_diagnoses": self.total_diagnoses,
            "avg_progress": round(self.avg_progress, 3),
            "avg_trust": round(self.avg_trust, 3),
            "total_regressions": self.total_regressions,
            "detector_firing_rates": self.detector_firing_rates,
            "failure_name_counts": self.failure_name_counts,
            "severity_counts": self.severity_counts,
            "parse_failures": self.parse_failures,
            "hta_implausible": self.hta_implausible,
            "total_ms": round(self.total_ms, 1),
            "avg_ms_per_session": round(self.avg_ms_per_session, 1),
            "classification_counts": self.classification_counts,
            "detector_metrics": {
                k: v.to_dict() for k, v in self.detector_metrics.items()
            } if self.detector_metrics else {},
            "annotations_path": self.annotations_path,
            "annotated_sessions": self.annotated_sessions,
            "macro_precision": round(self.macro_precision, 4) if self.detector_metrics else None,
            "macro_recall": round(self.macro_recall, 4) if self.detector_metrics else None,
            "macro_f1": round(self.macro_f1, 4) if self.detector_metrics else None,
            "session_results": [r.to_dict() for r in self.session_results],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


# ── HTA plausibility check ──────────────────────────────────────────

def _assess_hta_plausibility(
    events: list[TraceEvent],
    engine: MonitorEngine,
) -> tuple[bool, str]:
    """Check if HTA phase inference is plausible."""
    state = engine.state
    hta = state.hta_state
    if hta is None:
        return False, "No HTA state (no events processed)"

    tool_calls = sum(1 for e in events if e.type == "tool_call")
    phase = hta.current_phase

    if tool_calls > 10 and phase == Phase.IDLE:
        return False, f"Stuck in IDLE after {tool_calls} tool calls"

    write_tools = {"write", "edit", "write_file", "edit_file", "create"}
    has_writes = any(e.tool and any(w in e.tool.lower() for w in write_tools)
                     for e in events if e.type == "tool_call")
    if has_writes and phase == Phase.GATHERING and tool_calls > 5:
        return False, "Has writes but stuck in GATHERING"

    if state.progress_pct == 0.0 and len(events) > 20:
        return False, f"Zero progress after {len(events)} events"

    return True, ""


def _classify_result(result: SessionResult) -> str:
    """Auto-classify session result for quick triage."""
    if not result.parse_ok:
        return "parser_issue"
    if not result.hta_plausible:
        return "parser_issue"
    if not result.detectors_fired:
        return "clean"

    if result.detectors_fired == ["goal_drift"] and result.event_count < 30:
        return "false_alarm"

    if result.detectors_fired == ["step_repetition"]:
        return "ambiguous"

    if len(result.detectors_fired) > 1 or any(
        d.get("severity") == "critical"
        for d in result.diagnoses
    ):
        return "real"

    return "ambiguous"


# ── Core evaluation function ────────────────────────────────────────

def evaluate_claude_code(
    traces_path: str | Path = "~/.claude/projects",
    session_id: Optional[str] = None,
    min_lines: int = 5,
    max_sessions: Optional[int] = None,
    context_store: "ContextStore | None" = None,
    annotations_path: Optional[str | Path] = None,
    detectors: list | None = None,
    split: Optional[str] = None,
    splits_file: Optional[str | Path] = None,
) -> EvaluationReport:
    """Evaluate Claude Code session traces through HTA + CAFT.

    This is the unified evaluation entry point. It runs each session
    through the full MonitorEngine pipeline, computes per-session
    diagnostics, aggregate stats, and optionally compares against
    gold-standard annotations.

    Args:
        traces_path: Root directory with Claude Code session logs.
        session_id: If provided, evaluate only this session (prefix match).
        min_lines: Skip sessions with fewer lines.
        max_sessions: Limit number of sessions (largest first).
        context_store: Optional OpenViking context store.
        annotations_path: Path to annotations JSONL for precision/recall.
        detectors: Custom detector list (e.g. calibrated).
        split: If provided, only evaluate sessions in this split
            (requires splits_file). One of: development, validation, test.
        splits_file: Path to splits.json for split filtering.

    Returns:
        EvaluationReport with per-session and aggregate results.
    """
    traces_path = Path(traces_path).expanduser()
    extractor = ClaudeCodeExtractor()
    sessions = extractor.discover(traces_path, min_lines=min_lines)

    if session_id:
        sessions = [s for s in sessions if s.session_id.startswith(session_id)]
        if not sessions:
            raise ValueError(f"No session found matching '{session_id}'")

    # Filter by split if requested
    if split and splits_file:
        from agentdiag.splits import SplitManager
        splits_path = Path(splits_file)
        if splits_path.exists():
            sm = SplitManager(splits_path)
            split_ids = set(sm.get_traces(split))
            sessions = [s for s in sessions if s.session_id in split_ids]

    if max_sessions is not None:
        sessions.sort(key=lambda s: s.line_count, reverse=True)
        sessions = sessions[:max_sessions]

    # Load annotations if provided
    annotations: dict[str, dict] = {}
    if annotations_path:
        annotations = _load_annotations(annotations_path)

    session_results: list[SessionResult] = []
    total_events = 0
    total_start = time.time()

    for session in sessions:
        trace_start = time.time()

        # Parse
        try:
            events = extractor.parse_session(session)
        except Exception as ex:
            result = SessionResult(
                session_id=session.session_id,
                project_dir=session.project_dir,
                line_count=session.line_count,
                event_count=0,
                first_timestamp=session.first_timestamp,
                last_timestamp=session.last_timestamp,
                final_phase="unknown",
                progress_pct=0.0,
                hta_regressions=0,
                diagnoses=[],
                trust_score=1.0,
                health="unknown",
                total_errors=0,
                tool_counts={},
                event_type_counts={},
                parse_ok=False,
                parse_error=str(ex)[:200],
                analysis_ms=(time.time() - trace_start) * 1000,
                classification="parser_issue",
            )
            session_results.append(result)
            continue

        if not events:
            result = SessionResult(
                session_id=session.session_id,
                project_dir=session.project_dir,
                line_count=session.line_count,
                event_count=0,
                first_timestamp=session.first_timestamp,
                last_timestamp=session.last_timestamp,
                final_phase="unknown",
                progress_pct=0.0,
                hta_regressions=0,
                diagnoses=[],
                trust_score=1.0,
                health="unknown",
                total_errors=0,
                tool_counts={},
                event_type_counts={},
                parse_ok=False,
                parse_error="No events extracted",
                analysis_ms=(time.time() - trace_start) * 1000,
                classification="parser_issue",
            )
            session_results.append(result)
            continue

        total_events += len(events)

        # Run through HTA + CAFT
        engine = MonitorEngine(
            goal=f"Session {session.session_id[:8]}",
            context_store=context_store,
            detectors=detectors,
        )
        if context_store is not None:
            engine.start_context_session(
                goal=f"Session {session.session_id[:8]}",
                source=session.session_id,
            )

        for event in events:
            engine.push(event)

        state = engine.state

        if context_store is not None:
            engine.end_context_session()
        hta = state.hta_state

        # HTA plausibility
        plausible, hta_note = _assess_hta_plausibility(events, engine)

        # Count tools and event types
        tool_counts: dict[str, int] = {}
        event_type_counts: dict[str, int] = {}
        for e in events:
            if e.tool:
                tool_counts[e.tool] = tool_counts.get(e.tool, 0) + 1
            event_type_counts[e.type] = event_type_counts.get(e.type, 0) + 1

        result = SessionResult(
            session_id=session.session_id,
            project_dir=session.project_dir,
            line_count=session.line_count,
            event_count=len(events),
            first_timestamp=session.first_timestamp,
            last_timestamp=session.last_timestamp,
            final_phase=hta.current_phase.label if hta else "unknown",
            progress_pct=round(state.progress_pct, 3),
            hta_regressions=hta.regression_count if hta else 0,
            diagnoses=[d.to_dict() for d in state.diagnoses],
            trust_score=round(state.trust_score, 3),
            health=state.health,
            total_errors=state.total_errors,
            tool_counts=tool_counts,
            event_type_counts=event_type_counts,
            parse_ok=True,
            hta_plausible=plausible,
            hta_notes=hta_note,
            phases_visited=list(dict.fromkeys(
                t.to_phase.label for t in hta.transitions
            )) if hta else [],
            detectors_fired=[d.failure_name for d in state.diagnoses],
            analysis_ms=(time.time() - trace_start) * 1000,
        )
        result.classification = _classify_result(result)
        session_results.append(result)

    total_ms = (time.time() - total_start) * 1000

    # Build aggregate report
    report = _build_aggregate_report(
        dataset="claude-code",
        traces_path=str(traces_path),
        session_results=session_results,
        total_events=total_events,
        total_ms=total_ms,
    )

    # Annotation comparison
    if annotations:
        report.annotations_path = str(annotations_path)
        _compute_annotation_metrics(report, annotations)

    return report


def _build_aggregate_report(
    dataset: str,
    traces_path: str,
    session_results: list[SessionResult],
    total_events: int,
    total_ms: float,
) -> EvaluationReport:
    """Build aggregate stats from session results."""
    report = EvaluationReport(
        dataset=dataset,
        traces_path=traces_path,
        total_sessions=len(session_results),
        total_events=total_events,
        session_results=session_results,
        total_ms=round(total_ms, 1),
        avg_ms_per_session=round(total_ms / max(len(session_results), 1), 1),
    )

    parsed = [r for r in session_results if r.parse_ok]
    report.parse_failures = sum(1 for r in session_results if not r.parse_ok)
    report.hta_implausible = sum(1 for r in parsed if not r.hta_plausible)

    if parsed:
        report.sessions_healthy = sum(1 for r in parsed if r.health == "healthy")
        report.sessions_degraded = sum(1 for r in parsed if r.health == "degraded")
        report.sessions_failing = sum(1 for r in parsed if r.health == "failing")
        report.total_diagnoses = sum(len(r.diagnoses) for r in parsed)
        report.avg_progress = sum(r.progress_pct for r in parsed) / len(parsed)
        report.avg_trust = sum(r.trust_score for r in parsed) / len(parsed)
        report.total_regressions = sum(r.hta_regressions for r in parsed)

        # Detector firing rates
        for r in parsed:
            for d in r.diagnoses:
                code = d.get("caft_code", "?")
                name = d.get("failure_name", "?")
                sev = d.get("severity", "?")
                report.detector_firing_rates[code] = report.detector_firing_rates.get(code, 0) + 1
                report.failure_name_counts[name] = report.failure_name_counts.get(name, 0) + 1
                report.severity_counts[sev] = report.severity_counts.get(sev, 0) + 1

    # Classification counts
    for r in session_results:
        cls = r.classification or "unknown"
        report.classification_counts[cls] = report.classification_counts.get(cls, 0) + 1

    return report


# ── Annotation loading and comparison ─────────────────────────────

def _load_annotations(path: str | Path) -> dict[str, dict]:
    """Load annotations from JSONL. Returns {trace_id: annotation_dict}."""
    path = Path(path)
    if not path.exists():
        return {}

    annotations = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ann = json.loads(line)
                trace_id = ann.get("trace_id", "")
                if trace_id:
                    annotations[trace_id] = ann
            except json.JSONDecodeError:
                continue
    return annotations


def load_annotation_ledger_for_eval(
    ledger_path: str | Path,
    status_filter: str = "gold",
) -> dict[str, dict]:
    """Load AnnotationRecords from a ledger and convert to eval-compatible format.

    This bridges the new AnnotationLedger with the existing evaluation
    pipeline, filtering by label status to ensure only trusted labels
    are used for final metrics.

    Args:
        ledger_path: Path to annotation_ledger.jsonl.
        status_filter: One of:
            - "gold": adjudicated only (for final metrics)
            - "trainable": adjudicated + human_reviewed (for threshold tuning)
            - "eval": adjudicated + held_out_test (for evaluation)
            - "all": all records (exploratory only)

    Returns:
        {session_id: annotation_dict} compatible with _compute_annotation_metrics.
    """
    from agentdiag.annotation_store import AnnotationLedger

    path = Path(ledger_path)
    if not path.exists():
        return {}

    ledger = AnnotationLedger(path)

    if status_filter == "gold":
        records = ledger.get_gold_annotations()
    elif status_filter == "trainable":
        records = ledger.get_trainable_annotations()
    elif status_filter == "eval":
        records = ledger.get_eval_annotations()
    else:
        records = ledger.get_all()

    # Convert to eval-compatible format: pick best label per session
    annotations: dict[str, dict] = {}
    for rec in records:
        sid = rec.effective_session_id
        if sid in annotations:
            continue  # first one wins (ledger returns highest-trust first)

        # Build a dict compatible with _compute_annotation_metrics
        ann: dict = {
            "trace_id": sid,
            "has_failure": rec.has_failure,
            "primary_caft_subtype": rec.primary_caft_name,
            "primary_caft_code": rec.primary_caft_code,
            "secondary_failures": [
                _code_to_name(c) for c in rec.secondary_caft_codes
            ],
            "severity": rec.severity,
            "label_status": rec.label_status,
            "annotator_type": rec.annotator_type,
        }
        annotations[sid] = ann

    return annotations


def _code_to_name(code: str) -> str:
    """Resolve a CAFT code to failure name."""
    from agentdiag.caft.taxonomy import CAFT_TAXONOMY
    t = CAFT_TAXONOMY.get(code)
    return t.name if t else code


def _compute_annotation_metrics(
    report: EvaluationReport,
    annotations: dict[str, dict],
) -> None:
    """Compare detector results against gold-standard annotations."""
    # Build a metrics dict for each detector that appeared
    all_detector_names: set[str] = set()
    for r in report.session_results:
        for d in r.diagnoses:
            all_detector_names.add(d.get("failure_name", "?"))

    # Also include detectors from annotations
    for ann in annotations.values():
        if ann.get("has_failure") and ann.get("primary_caft_subtype"):
            all_detector_names.add(ann["primary_caft_subtype"])
        for code in ann.get("secondary_failures", []):
            all_detector_names.add(code)

    metrics = {name: DetectorMetrics(detector_name=name) for name in all_detector_names}

    annotated_count = 0
    for r in report.session_results:
        ann = annotations.get(r.session_id)
        if ann is None:
            continue
        annotated_count += 1

        # What the detector found
        detected = set(r.detectors_fired)

        # What was annotated as ground truth
        expected: set[str] = set()
        if ann.get("has_failure"):
            if ann.get("primary_caft_subtype"):
                expected.add(ann["primary_caft_subtype"])
            for code in ann.get("secondary_failures", []):
                expected.add(code)

        # Score
        for name in all_detector_names:
            is_expected = name in expected
            is_detected = name in detected

            if is_expected and is_detected:
                metrics[name].true_positives += 1
            elif is_expected and not is_detected:
                metrics[name].false_negatives += 1
            elif not is_expected and is_detected:
                metrics[name].false_positives += 1

    report.detector_metrics = metrics
    report.annotated_sessions = annotated_count


# ── Pretty printing ─────────────────────────────────────────────────

def print_evaluation_report(report: EvaluationReport) -> None:
    """Print a human-readable evaluation report."""
    print(f"\n{'=' * 70}")
    print(f"  AGENTDIAG EVALUATION REPORT")
    print(f"{'=' * 70}")
    print(f"  Dataset:    {report.dataset}")
    print(f"  Traces:     {report.traces_path}")
    print(f"  Sessions:   {report.total_sessions}")
    print(f"  Events:     {report.total_events}")
    if report.total_ms:
        print(f"  Time:       {report.total_ms:.0f}ms ({report.avg_ms_per_session:.0f}ms/session)")
    if report.parse_failures:
        print(f"  Parse fail: {report.parse_failures}")
    print()

    # Health distribution
    print(f"  HEALTH DISTRIBUTION")
    print(f"  {'─' * 40}")
    total = report.total_sessions or 1
    print(f"    Healthy:   {report.sessions_healthy:>3} ({report.sessions_healthy/total:.0%})")
    print(f"    Degraded:  {report.sessions_degraded:>3} ({report.sessions_degraded/total:.0%})")
    print(f"    Failing:   {report.sessions_failing:>3} ({report.sessions_failing/total:.0%})")
    print()

    # Aggregate metrics
    print(f"  AGGREGATE METRICS")
    print(f"  {'─' * 40}")
    print(f"    Avg progress:   {report.avg_progress:.0%}")
    print(f"    Avg trust:      {report.avg_trust:.0%}")
    print(f"    Total CAFT dx:  {report.total_diagnoses}")
    print(f"    Total regressions: {report.total_regressions}")
    print()

    # CAFT detector firing rates
    if report.failure_name_counts:
        print(f"  CAFT FAILURE FREQUENCY")
        print(f"  {'─' * 40}")
        for name, count in sorted(report.failure_name_counts.items(), key=lambda x: -x[1]):
            rate = count / total
            print(f"    {name:<30} {count:>3} ({rate:.0%} of sessions)")
        print()

    # Severity distribution
    if report.severity_counts:
        print(f"  SEVERITY DISTRIBUTION")
        print(f"  {'─' * 40}")
        for sev, count in sorted(report.severity_counts.items()):
            print(f"    {sev:<12} {count:>3}")
        print()

    # Classification counts
    if report.classification_counts:
        print(f"  AUTO-CLASSIFICATION")
        print(f"  {'─' * 40}")
        for cls, count in sorted(report.classification_counts.items(), key=lambda x: -x[1]):
            print(f"    {cls:<15} {count:>3}")
        print()

    # Annotation comparison (precision/recall/F1)
    if report.detector_metrics:
        _print_annotation_metrics(report)

    # Per-session details
    print(f"  PER-SESSION DETAILS")
    print(f"  {'─' * 70}")
    print(f"  {'Session':<12} {'Events':>6} {'Phase':<12} {'Progress':>8} {'Trust':>6} {'Health':<10} {'CAFT':>4}")
    print(f"  {'─' * 70}")
    for r in report.session_results:
        caft_count = len(r.diagnoses)
        print(f"  {r.session_id[:10]:<12} {r.event_count:>6} {r.final_phase:<12} "
              f"{r.progress_pct:>7.0%} {r.trust_score:>5.0%} {r.health:<10} {caft_count:>4}")
        if r.hta_notes:
            print(f"      note: {r.hta_notes}")
        if r.parse_error:
            print(f"      error: {r.parse_error}")
    print(f"  {'=' * 70}")


def _print_annotation_metrics(report: EvaluationReport) -> None:
    """Print per-detector precision/recall/F1 from annotation comparison."""
    print(f"  DETECTOR METRICS (vs {report.annotated_sessions} annotated sessions)")
    print(f"  {'─' * 70}")
    print(f"  {'Detector':<30} {'Prec':>6} {'Rec':>6} {'F1':>6} {'TP':>4} {'FP':>4} {'FN':>4}")
    print(f"  {'─' * 30} {'─' * 6} {'─' * 6} {'─' * 6} {'─' * 4} {'─' * 4} {'─' * 4}")

    for name in sorted(report.detector_metrics):
        m = report.detector_metrics[name]
        # Skip detectors with no activity
        if m.true_positives + m.false_positives + m.false_negatives == 0:
            continue
        print(
            f"  {name:<30} "
            f"{m.precision:>5.0%} "
            f"{m.recall:>5.0%} "
            f"{m.f1:>5.0%} "
            f"{m.true_positives:>4} "
            f"{m.false_positives:>4} "
            f"{m.false_negatives:>4}"
        )

    print()
    print(f"  Macro Precision:  {report.macro_precision:.0%}")
    print(f"  Macro Recall:     {report.macro_recall:.0%}")
    print(f"  Macro F1:         {report.macro_f1:.0%}")
    print()
