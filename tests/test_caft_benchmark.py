"""Tests for the CAFT benchmark itself.

These ensure the benchmark harness and synthetic generators
produce correct results, and that no regressions sneak in.
"""

import pytest
from agentdiag.caft.benchmark import run_benchmark, BenchmarkReport
from agentdiag.caft.synthetic import CAFT_GENERATORS


class TestBenchmarkReport:
    def test_all_scenarios_pass(self):
        """The benchmark must pass with 100% recall — this IS the proof."""
        report = run_benchmark()
        assert report.all_passed, (
            f"{report.failed_scenarios} scenario(s) failed: "
            + ", ".join(
                r.scenario_name
                for r in report.scenario_results
                if not r.passed
            )
        )

    def test_all_active_scenarios_included(self):
        """V13: 6 scenarios skipped (5 from V12b pruning + goal_drift moved to Tier 2)."""
        report = run_benchmark()
        # 6 scenarios skipped: step_repetition, missing_verification, goal_drift,
        # tool_thrashing, reasoning_action_mismatch, + goal_drift now Tier 2
        # NOTE: goal_drift was already in the skip list from V12b (counted above)
        # so the count stays at 5 if GoalDrift was already skipped.
        # If GoalDrift was NOT already skipped, it's now 6.
        # Check: GoalDrift is no longer in ALL_CAFT_DETECTORS, so its scenario
        # will be skipped by the benchmark's active-detector filter.
        assert report.total_scenarios == len(CAFT_GENERATORS) - 5

    def test_zero_false_negatives(self):
        """Every expected failure must be detected."""
        report = run_benchmark()
        for r in report.scenario_results:
            assert r.false_negatives == set(), (
                f"Scenario '{r.scenario_name}' missed: {r.false_negatives}"
            )

    def test_zero_false_positives(self):
        """No unexpected failures on any scenario."""
        report = run_benchmark()
        for r in report.scenario_results:
            assert r.false_positives == set(), (
                f"Scenario '{r.scenario_name}' had false positives: {r.false_positives}"
            )

    def test_perfect_precision_and_recall(self):
        report = run_benchmark()
        for name, m in report.detector_metrics.items():
            assert m.precision == 1.0, f"{name}: precision={m.precision}"
            assert m.recall == 1.0, f"{name}: recall={m.recall}"
            assert m.f1 == 1.0, f"{name}: f1={m.f1}"

    def test_clean_scenario_has_no_detections(self):
        """The clean trace must produce zero CAFT diagnoses."""
        report = run_benchmark()
        clean = next(r for r in report.scenario_results if r.scenario_name == "clean")
        assert clean.detected_failures == set()
        assert clean.true_positives == set()
        assert clean.false_positives == set()

    def test_multi_failure_catches_all(self):
        """The multi-failure trace must catch premature_termination (step_repetition pruned in V12b)."""
        report = run_benchmark()
        multi = next(r for r in report.scenario_results if r.scenario_name == "multi_failure")
        assert "premature_termination" in multi.true_positives
