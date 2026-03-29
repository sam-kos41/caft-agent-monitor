"""Pilot evaluation: backwards-compatible wrapper around evaluate.

The pilot functionality is now part of the unified evaluation pipeline
in evaluate.py. This module provides the old API for backward compatibility.

Usage:
    agentdiag pilot --traces ~/.claude/projects --n 20
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from agentdiag.evaluate import (
    evaluate_claude_code,
    EvaluationReport,
    SessionResult,
    _classify_result,
    print_evaluation_report,
)


@dataclass
class PilotResult:
    """Result of running one trace through the pilot.

    Backwards-compatible wrapper. New code should use SessionResult.
    """
    trace_num: int
    session_id: str
    project: str
    raw_lines: int

    # Parse
    parse_ok: bool = True
    parse_error: Optional[str] = None
    event_count: int = 0
    tool_calls: int = 0
    reasoning_events: int = 0
    user_inputs: int = 0

    # HTA
    hta_plausible: bool = True
    final_phase: str = "unknown"
    progress_pct: float = 0.0
    phases_visited: list[str] = field(default_factory=list)
    regressions: int = 0
    hta_notes: str = ""

    # CAFT
    detectors_fired: list[str] = field(default_factory=list)
    caft_codes: list[str] = field(default_factory=list)
    severities: list[str] = field(default_factory=list)
    trust_score: float = 1.0
    health: str = "healthy"

    # Classification
    main_failure: str = ""
    notes: str = ""

    # Timing
    analysis_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PilotReport:
    """Aggregate pilot results. Backwards-compatible wrapper."""
    n_traces: int
    n_parsed: int
    n_parse_failures: int
    n_hta_plausible: int
    n_hta_implausible: int
    n_detections: int
    n_clean: int
    results: list[PilotResult]

    # Detector frequency
    detector_counts: dict[str, int] = field(default_factory=dict)
    phase_distribution: dict[str, int] = field(default_factory=dict)
    health_distribution: dict[str, int] = field(default_factory=dict)

    # Timing
    total_ms: float = 0.0
    avg_ms_per_trace: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["results"] = [r.to_dict() for r in self.results]
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


def _session_to_pilot(i: int, r: SessionResult) -> PilotResult:
    """Convert a SessionResult to a PilotResult for backwards compatibility."""
    return PilotResult(
        trace_num=i + 1,
        session_id=r.session_id,
        project=r.project_dir,
        raw_lines=r.line_count,
        parse_ok=r.parse_ok,
        parse_error=r.parse_error,
        event_count=r.event_count,
        tool_calls=r.tool_counts.get("Read", 0) + r.tool_counts.get("Edit", 0) + sum(
            v for k, v in r.tool_counts.items() if k not in ("Read", "Edit")
        ) if r.tool_counts else 0,
        reasoning_events=r.event_type_counts.get("reasoning", 0) + r.event_type_counts.get("planning", 0),
        user_inputs=r.event_type_counts.get("user_input", 0),
        hta_plausible=r.hta_plausible,
        final_phase=r.final_phase,
        progress_pct=r.progress_pct,
        phases_visited=r.phases_visited,
        regressions=r.hta_regressions,
        hta_notes=r.hta_notes,
        detectors_fired=r.detectors_fired,
        caft_codes=[d.get("caft_code", "?") for d in r.diagnoses],
        severities=[d.get("severity", "?") for d in r.diagnoses],
        trust_score=r.trust_score,
        health=r.health,
        main_failure=r.classification,
        analysis_ms=r.analysis_ms,
    )


def run_pilot(
    traces_path: str | Path = "~/.claude/projects",
    n: int = 20,
    min_lines: int = 10,
    detectors: list | None = None,
) -> PilotReport:
    """Run the pilot evaluation on N real traces.

    This is a backwards-compatible wrapper around evaluate_claude_code().

    Args:
        traces_path: Root directory with Claude Code sessions.
        n: Number of traces to evaluate.
        min_lines: Skip sessions with fewer lines.
        detectors: Custom detector list (e.g. calibrated).

    Returns:
        PilotReport with per-trace results and aggregate stats.
    """
    report = evaluate_claude_code(
        traces_path=traces_path,
        min_lines=min_lines,
        max_sessions=n,
        detectors=detectors,
    )

    # Convert to pilot-format results
    results = [_session_to_pilot(i, r) for i, r in enumerate(report.session_results)]

    n_parsed = sum(1 for r in results if r.parse_ok)
    n_parse_fail = sum(1 for r in results if not r.parse_ok)
    n_hta_ok = sum(1 for r in results if r.hta_plausible)
    n_hta_bad = sum(1 for r in results if r.parse_ok and not r.hta_plausible)
    n_detected = sum(1 for r in results if r.detectors_fired)
    n_clean = sum(1 for r in results if r.parse_ok and not r.detectors_fired)

    detector_counts: dict[str, int] = {}
    phase_dist: dict[str, int] = {}
    health_dist: dict[str, int] = {}
    for r in results:
        for d in r.detectors_fired:
            detector_counts[d] = detector_counts.get(d, 0) + 1
        if r.parse_ok:
            phase_dist[r.final_phase] = phase_dist.get(r.final_phase, 0) + 1
            health_dist[r.health] = health_dist.get(r.health, 0) + 1

    return PilotReport(
        n_traces=len(results),
        n_parsed=n_parsed,
        n_parse_failures=n_parse_fail,
        n_hta_plausible=n_hta_ok,
        n_hta_implausible=n_hta_bad,
        n_detections=n_detected,
        n_clean=n_clean,
        results=results,
        detector_counts=detector_counts,
        phase_distribution=phase_dist,
        health_distribution=health_dist,
        total_ms=report.total_ms,
        avg_ms_per_trace=report.avg_ms_per_session,
    )


def print_pilot_report(report: PilotReport) -> None:
    """Print the pilot results table."""
    print(f"\n{'=' * 100}")
    print(f"  AGENTDIAG PILOT EVALUATION — {report.n_traces} traces")
    print(f"{'=' * 100}")
    print(f"  Parsed: {report.n_parsed}/{report.n_traces}  "
          f"HTA OK: {report.n_hta_plausible}  "
          f"Detections: {report.n_detections}  "
          f"Clean: {report.n_clean}  "
          f"Time: {report.total_ms:.0f}ms ({report.avg_ms_per_trace:.0f}ms/trace)")
    print()

    # Results table
    hdr = (f"  {'#':>3} {'Session':<12} {'Lines':>5} {'Events':>6} "
           f"{'Parse':>5} {'HTA':>5} {'Phase':<12} {'Prog':>5} "
           f"{'Trust':>5} {'Health':<9} {'Detectors':<30} {'Class':<12}")
    print(hdr)
    print(f"  {'─' * 96}")

    for r in report.results:
        parse = "OK" if r.parse_ok else "FAIL"
        hta = "OK" if r.hta_plausible else "BAD"
        detectors = ", ".join(r.detectors_fired) if r.detectors_fired else "—"
        if len(detectors) > 28:
            detectors = detectors[:25] + "..."

        print(f"  {r.trace_num:>3} {r.session_id[:10]:<12} {r.raw_lines:>5} "
              f"{r.event_count:>6} {parse:>5} {hta:>5} "
              f"{r.final_phase:<12} {r.progress_pct:>4.0%} "
              f"{r.trust_score:>4.0%} {r.health:<9} "
              f"{detectors:<30} {r.main_failure:<12}")

        if r.hta_notes:
            print(f"      note: {r.hta_notes}")
        if r.parse_error:
            print(f"      error: {r.parse_error}")

    # Aggregate stats
    print(f"\n  {'─' * 96}")
    print(f"\n  DETECTOR FREQUENCY:")
    for name, count in sorted(report.detector_counts.items(), key=lambda x: -x[1]):
        pct = count / max(report.n_parsed, 1)
        print(f"    {name:<30} {count:>3} ({pct:.0%})")

    print(f"\n  PHASE DISTRIBUTION:")
    for phase, count in sorted(report.phase_distribution.items()):
        print(f"    {phase:<15} {count:>3}")

    print(f"\n  HEALTH DISTRIBUTION:")
    for health, count in sorted(report.health_distribution.items()):
        print(f"    {health:<15} {count:>3}")

    # Classification summary
    class_counts: dict[str, int] = {}
    for r in report.results:
        class_counts[r.main_failure] = class_counts.get(r.main_failure, 0) + 1
    print(f"\n  AUTO-CLASSIFICATION:")
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"    {cls:<15} {count:>3}")

    print(f"\n{'=' * 100}")
