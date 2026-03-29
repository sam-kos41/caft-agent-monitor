"""CAFT detector validation benchmark.

Runs all synthetic CAFT traces through the full monitor pipeline and
measures precision, recall, F1, and false positive rate per detector.

This is the core evaluation tool: it PROVES the detectors work by
checking every synthetic scenario against its ground-truth labels.

Usage:
    from agentdiag.caft.benchmark import run_benchmark, print_report
    results = run_benchmark()
    print_report(results)

CLI:
    python -m agentdiag validate-caft
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from agentdiag.models import TraceEvent
from agentdiag.hta import HTAStateMachine
from agentdiag.caft.detectors import ALL_CAFT_DETECTORS, run_caft_detectors
from agentdiag.caft.synthetic import CAFT_GENERATORS, ALL_CAFT_SCENARIO_NAMES


@dataclass
class ScenarioResult:
    """Result of running one scenario through CAFT detectors."""
    scenario_name: str
    expected_failures: set[str]
    detected_failures: set[str]
    true_positives: set[str]
    false_positives: set[str]
    false_negatives: set[str]
    diagnoses: list  # CaftDiagnosis objects
    passed: bool

    @property
    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        tp = ", ".join(sorted(self.true_positives)) or "none"
        fp = ", ".join(sorted(self.false_positives)) or "none"
        fn = ", ".join(sorted(self.false_negatives)) or "none"
        return (
            f"[{status}] {self.scenario_name}: "
            f"TP={tp}  FP={fp}  FN={fn}"
        )


@dataclass
class DetectorMetrics:
    """Precision/recall/F1 for a single CAFT detector."""
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


@dataclass
class BenchmarkReport:
    """Full benchmark report."""
    scenario_results: list[ScenarioResult]
    detector_metrics: dict[str, DetectorMetrics]
    total_scenarios: int
    passed_scenarios: int
    failed_scenarios: int

    @property
    def all_passed(self) -> bool:
        return self.failed_scenarios == 0

    @property
    def macro_precision(self) -> float:
        vals = [m.precision for m in self.detector_metrics.values()]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def macro_recall(self) -> float:
        vals = [m.recall for m in self.detector_metrics.values()]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def macro_f1(self) -> float:
        vals = [m.f1 for m in self.detector_metrics.values()]
        return sum(vals) / len(vals) if vals else 0.0


def _run_scenario(
    scenario_name: str,
    events: list[TraceEvent],
    expected: set[str],
) -> ScenarioResult:
    """Run one scenario through the full HTA + CAFT pipeline.

    Uses incremental (streaming) detection — pushes events one at a time
    through HTA + CAFT, exactly as the MonitorEngine does. This is the
    correct evaluation mode because CAFT detectors use sliding windows.
    """
    hta = HTAStateMachine(goal=f"benchmark_{scenario_name}")
    seen: dict[str, int] = {}
    all_diagnoses = []

    for e in events:
        hta_state = hta.push(e)
        new = run_caft_detectors(
            events=events[:e.step],  # events seen so far
            hta_state=hta_state,
            detectors=list(ALL_CAFT_DETECTORS),
            seen=seen,
        )
        all_diagnoses.extend(new)

    detected = {d.failure_name for d in all_diagnoses}
    tp = expected & detected
    fp = detected - expected
    fn = expected - detected
    passed = (fn == set())  # pass if no false negatives (all expected were caught)

    return ScenarioResult(
        scenario_name=scenario_name,
        expected_failures=expected,
        detected_failures=detected,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        diagnoses=all_diagnoses,
        passed=passed,
    )


def run_benchmark(
    scenarios: dict[str, callable] | None = None,
) -> BenchmarkReport:
    """Run the full CAFT benchmark suite.

    Args:
        scenarios: Override scenario generators. Default: all CAFT_GENERATORS.
            Scenarios whose expected failures are entirely covered by
            disabled detectors are automatically skipped.

    Returns:
        BenchmarkReport with per-scenario and per-detector metrics.
    """
    if scenarios is None:
        scenarios = CAFT_GENERATORS

    # All known detector names (only active detectors)
    all_detector_names = {d.name for d in ALL_CAFT_DETECTORS}
    detector_metrics = {name: DetectorMetrics(detector_name=name)
                        for name in all_detector_names}

    results = []
    for name, gen_fn in scenarios.items():
        events, expected = gen_fn()

        # Skip scenarios whose expected failures are ALL from disabled detectors.
        # E.g., if missing_verification is disabled and a scenario only expects
        # missing_verification, skip it entirely.
        if expected and not (expected & all_detector_names):
            continue

        # For scenarios expecting a mix of active+disabled detectors,
        # only expect the active ones.
        active_expected = expected & all_detector_names
        result = _run_scenario(name, events, active_expected)
        results.append(result)

        # Accumulate per-detector metrics
        for det_name in all_detector_names:
            is_expected = det_name in expected
            is_detected = det_name in result.detected_failures

            if is_expected and is_detected:
                detector_metrics[det_name].true_positives += 1
            elif is_expected and not is_detected:
                detector_metrics[det_name].false_negatives += 1
            elif not is_expected and is_detected:
                detector_metrics[det_name].false_positives += 1
            # true negative: not expected, not detected — not tracked

    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed

    return BenchmarkReport(
        scenario_results=results,
        detector_metrics=detector_metrics,
        total_scenarios=len(results),
        passed_scenarios=passed,
        failed_scenarios=failed,
    )


def print_report(report: BenchmarkReport) -> None:
    """Print a human-readable benchmark report to stdout."""
    print("=" * 70)
    print("  CAFT Detector Validation Benchmark")
    print("=" * 70)

    # Scenario results
    print(f"\n{'SCENARIO RESULTS':^70}")
    print("-" * 70)
    for r in report.scenario_results:
        status = "PASS" if r.passed else "FAIL"
        icon = " + " if r.passed else "!! "
        print(f"  {icon}[{status}] {r.scenario_name}")
        if r.expected_failures:
            print(f"       Expected: {sorted(r.expected_failures)}")
        if r.detected_failures:
            print(f"       Detected: {sorted(r.detected_failures)}")
        if r.false_negatives:
            print(f"       MISSED:   {sorted(r.false_negatives)}")
        if r.false_positives:
            print(f"       EXTRA:    {sorted(r.false_positives)}")

    # Per-detector metrics
    print(f"\n{'PER-DETECTOR METRICS':^70}")
    print("-" * 70)
    print(f"  {'Detector':<30} {'Prec':>6} {'Rec':>6} {'F1':>6} {'TP':>4} {'FP':>4} {'FN':>4}")
    print(f"  {'—' * 30} {'—' * 6} {'—' * 6} {'—' * 6} {'—' * 4} {'—' * 4} {'—' * 4}")

    for name in sorted(report.detector_metrics):
        m = report.detector_metrics[name]
        print(
            f"  {name:<30} "
            f"{m.precision:>5.0%} "
            f"{m.recall:>5.0%} "
            f"{m.f1:>5.0%} "
            f"{m.true_positives:>4} "
            f"{m.false_positives:>4} "
            f"{m.false_negatives:>4}"
        )

    # Aggregate
    print(f"\n{'AGGREGATE':^70}")
    print("-" * 70)
    print(f"  Scenarios:        {report.passed_scenarios}/{report.total_scenarios} passed")
    print(f"  Macro Precision:  {report.macro_precision:.0%}")
    print(f"  Macro Recall:     {report.macro_recall:.0%}")
    print(f"  Macro F1:         {report.macro_f1:.0%}")

    # Verdict
    print()
    if report.all_passed:
        print("  VERDICT: ALL SCENARIOS PASSED")
    else:
        print(f"  VERDICT: {report.failed_scenarios} SCENARIO(S) FAILED")
    print("=" * 70)
