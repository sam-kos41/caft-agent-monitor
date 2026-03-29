"""Tests for the evaluation metrics engine (agentdiag/metrics.py)."""

import pytest
from agentdiag.metrics import (
    Annotation,
    BootstrapCI,
    ComparisonTable,
    Detection,
    DetectorResult,
    EvalReport,
    MatchResult,
    ModeComparison,
    bootstrap_ci,
    compare_modes,
    compute_evaluation,
    format_comparison_table,
    match_detections,
    mcnemar_test,
    _normal_sf,
)


# ── Fixtures ────────────────────────────────────────────────────────

def _det(trace_id: str, name: str, step: int, conf: float = 0.7,
         code: str = "", confirmed: bool | None = None) -> Detection:
    return Detection(
        trace_id=trace_id,
        failure_name=name,
        caft_code=code or f"x.{name[:1]}",
        onset_step=step,
        confidence=conf,
        confirmed=confirmed,
    )


def _ann(trace_id: str, name: str, step: int = 0,
         code: str = "", latent: bool = False) -> Annotation:
    return Annotation(
        trace_id=trace_id,
        failure_name=name,
        caft_code=code or f"x.{name[:1]}",
        onset_step=step,
        is_latent=latent,
    )


# ── match_detections ────────────────────────────────────────────────

class TestMatchDetections:
    def test_exact_match(self):
        """Detection with same name and step within window = full match."""
        dets = [_det("t1", "step_repetition", 10)]
        anns = [_ann("t1", "step_repetition", 12)]
        matches, unmatched = match_detections(dets, anns, match_window=5)
        assert len(matches) == 1
        assert matches[0].match_type == "full"
        assert matches[0].tp_weight == 1.0
        assert len(unmatched) == 0

    def test_partial_match_wrong_step(self):
        """Same name but step outside window = partial match (0.5)."""
        dets = [_det("t1", "step_repetition", 10)]
        anns = [_ann("t1", "step_repetition", 25)]
        matches, unmatched = match_detections(dets, anns, match_window=5)
        assert len(matches) == 1
        assert matches[0].match_type == "partial"
        assert matches[0].tp_weight == 0.5
        assert len(unmatched) == 0

    def test_false_positive(self):
        """Detection with no matching annotation = FP."""
        dets = [_det("t1", "step_repetition", 10)]
        anns = [_ann("t1", "context_loss", 10)]
        matches, unmatched = match_detections(dets, anns, match_window=5)
        assert len(matches) == 1
        assert matches[0].match_type == "fp"
        assert matches[0].tp_weight == 0.0
        assert len(unmatched) == 1  # context_loss unmatched

    def test_false_negative(self):
        """Annotation with no detection = FN."""
        dets = []
        anns = [_ann("t1", "step_repetition", 10)]
        matches, unmatched = match_detections(dets, anns, match_window=5)
        assert len(matches) == 0
        assert len(unmatched) == 1
        assert unmatched[0].failure_name == "step_repetition"

    def test_zero_onset_always_matches(self):
        """Annotation with onset_step=0 matches any detection step."""
        dets = [_det("t1", "premature_termination", 100)]
        anns = [_ann("t1", "premature_termination", 0)]
        matches, unmatched = match_detections(dets, anns, match_window=5)
        assert len(matches) == 1
        assert matches[0].match_type == "full"

    def test_greedy_best_match_first(self):
        """Higher-confidence detection gets matched first."""
        dets = [
            _det("t1", "step_repetition", 10, conf=0.5),
            _det("t1", "step_repetition", 12, conf=0.9),
        ]
        anns = [_ann("t1", "step_repetition", 12)]
        matches, unmatched = match_detections(dets, anns, match_window=5)
        # Higher-confidence det (step 12) should match the annotation
        tp_matches = [m for m in matches if m.match_type == "full"]
        assert len(tp_matches) == 1
        assert tp_matches[0].detection.onset_step == 12

    def test_latent_annotations_excluded(self):
        """Latent annotations should not participate in matching."""
        dets = [_det("t1", "step_repetition", 10)]
        anns = [_ann("t1", "step_repetition", 10, latent=True)]
        matches, unmatched = match_detections(dets, anns, match_window=5)
        # Detection has no non-latent annotation to match → FP
        assert len(matches) == 1
        assert matches[0].match_type == "fp"
        assert len(unmatched) == 0  # latent annotations excluded from FN

    def test_multiple_annotations_different_traces(self):
        """Annotations from different traces don't cross-match."""
        dets = [_det("t1", "step_repetition", 10)]
        anns = [_ann("t2", "step_repetition", 10)]
        matches, unmatched = match_detections(dets, anns, match_window=5)
        assert len(matches) == 1
        assert matches[0].match_type == "fp"
        assert len(unmatched) == 1

    def test_empty_inputs(self):
        """No detections, no annotations."""
        matches, unmatched = match_detections([], [], match_window=5)
        assert matches == []
        assert unmatched == []

    def test_multiple_failures_same_trace(self):
        """Multiple different failure types in one trace."""
        dets = [
            _det("t1", "step_repetition", 10),
            _det("t1", "premature_termination", 25),
        ]
        anns = [
            _ann("t1", "step_repetition", 12),
            _ann("t1", "premature_termination", 25),
        ]
        matches, unmatched = match_detections(dets, anns, match_window=5)
        tp_matches = [m for m in matches if m.match_type in ("full", "partial")]
        assert len(tp_matches) == 2
        assert len(unmatched) == 0


# ── DetectorResult properties ───────────────────────────────────────

class TestDetectorResult:
    def test_precision_basic(self):
        dr = DetectorResult(detector="test", caft_code="1.1", tp=3.0, fp=1, fn=0)
        assert dr.precision == pytest.approx(0.75)

    def test_recall_basic(self):
        dr = DetectorResult(detector="test", caft_code="1.1", tp=2.0, fp=0, fn=2)
        assert dr.recall == pytest.approx(0.5)

    def test_f1_basic(self):
        dr = DetectorResult(detector="test", caft_code="1.1", tp=2.0, fp=1, fn=1)
        p = 2.0 / 3.0
        r = 2.0 / 3.0
        expected_f1 = 2 * p * r / (p + r)
        assert dr.f1 == pytest.approx(expected_f1)

    def test_precision_no_detections_no_fn(self):
        """No detections, no annotations → precision 1.0 (vacuous)."""
        dr = DetectorResult(detector="test", caft_code="1.1", tp=0.0, fp=0, fn=0)
        assert dr.precision == 1.0

    def test_precision_no_detections_with_fn(self):
        """No detections but some annotations → precision 0.0."""
        dr = DetectorResult(detector="test", caft_code="1.1", tp=0.0, fp=0, fn=2)
        assert dr.precision == 0.0

    def test_partial_tp(self):
        """Fractional TP from partial matches."""
        dr = DetectorResult(detector="test", caft_code="1.1", tp=1.5, fp=1, fn=1)
        assert dr.precision == pytest.approx(1.5 / 2.5)

    def test_to_dict(self):
        dr = DetectorResult(detector="test", caft_code="1.1", tp=2.0, fp=1, fn=1)
        d = dr.to_dict()
        assert "precision" in d
        assert "recall" in d
        assert "f1" in d
        assert d["detector"] == "test"


# ── compute_evaluation ──────────────────────────────────────────────

class TestComputeEvaluation:
    def test_basic_evaluation(self):
        """Single TP detection → precision=1, recall=1."""
        anns = [_ann("t1", "premature_termination", 25)]
        dets = [_det("t1", "premature_termination", 27)]
        report = compute_evaluation(anns, dets, "strict", skip_bootstrap=True)
        assert report.mode == "strict"
        assert report.macro_p == pytest.approx(1.0)
        assert report.macro_r == pytest.approx(1.0)
        assert report.macro_f1 == pytest.approx(1.0)
        assert report.n_traces == 1
        assert report.n_annotations == 1
        assert report.n_candidates == 1

    def test_mixed_tp_fp_fn(self):
        """2 TPs, 1 FP, 1 FN across 2 detectors."""
        anns = [
            _ann("t1", "step_repetition", 10),
            _ann("t1", "premature_termination", 25),
            _ann("t1", "context_loss", 15),
        ]
        dets = [
            _det("t1", "step_repetition", 12),      # TP
            _det("t1", "premature_termination", 27), # TP
            _det("t1", "goal_drift", 5),             # FP
        ]
        report = compute_evaluation(anns, dets, "test", skip_bootstrap=True)
        assert report.micro_p == pytest.approx(2.0 / 3.0)  # 2 TP / (2 TP + 1 FP)
        assert report.micro_r == pytest.approx(2.0 / 3.0)  # 2 TP / (2 TP + 1 FN)

    def test_empty_detections(self):
        """No detections → 0 precision, 0 recall."""
        anns = [_ann("t1", "step_repetition", 10)]
        report = compute_evaluation(anns, [], "strict", skip_bootstrap=True)
        assert report.macro_r == 0.0
        assert report.n_candidates == 0

    def test_latent_annotations_tracked(self):
        """Latent annotations → tracked in latent_fn, not penalized."""
        anns = [_ann("t1", "step_repetition", 10, latent=True)]
        report = compute_evaluation(anns, [], "strict", skip_bootstrap=True)
        assert report.latent_fn == 1
        # Latent annotations shouldn't contribute to FN in per-detector stats
        for d in report.per_detector:
            if d.detector == "step_repetition":
                assert d.fn == 0

    def test_candidates_per_trace(self):
        """Candidates per trace is correctly computed."""
        anns = [_ann("t1", "x", 5), _ann("t2", "y", 10)]
        dets = [_det("t1", "x", 5), _det("t1", "z", 8), _det("t2", "y", 10)]
        report = compute_evaluation(anns, dets, "test", skip_bootstrap=True)
        assert report.candidates_per_trace == pytest.approx(3.0 / 2.0)

    def test_to_dict_roundtrip(self):
        """EvalReport serializes cleanly."""
        anns = [_ann("t1", "premature_termination", 25)]
        dets = [_det("t1", "premature_termination", 27)]
        report = compute_evaluation(anns, dets, "strict", skip_bootstrap=True)
        d = report.to_dict()
        assert d["mode"] == "strict"
        assert isinstance(d["per_detector"], list)
        assert isinstance(d["macro_f1"], float)

    def test_to_json(self):
        """EvalReport JSON serializes without error."""
        anns = [_ann("t1", "premature_termination", 25)]
        dets = [_det("t1", "premature_termination", 27)]
        report = compute_evaluation(anns, dets, "strict", skip_bootstrap=True)
        import json
        parsed = json.loads(report.to_json())
        assert parsed["mode"] == "strict"


# ── bootstrap_ci ────────────────────────────────────────────────────

class TestBootstrapCI:
    def test_produces_three_metrics(self):
        """Bootstrap CI returns CIs for macro P/R/F1."""
        anns = [_ann("t1", "x", 5), _ann("t2", "y", 10)]
        dets = [_det("t1", "x", 5), _det("t2", "y", 10)]
        cis = bootstrap_ci(anns, dets, n_iterations=50)
        assert "macro_precision" in cis
        assert "macro_recall" in cis
        assert "macro_f1" in cis

    def test_ci_bounds_order(self):
        """CI lower <= point estimate <= CI upper."""
        anns = [_ann("t1", "x", 5), _ann("t2", "y", 10)]
        dets = [_det("t1", "x", 5), _det("t2", "y", 10)]
        cis = bootstrap_ci(anns, dets, n_iterations=100)
        for key, ci in cis.items():
            assert ci.ci_lower <= ci.point_estimate + 0.01  # small tolerance
            assert ci.ci_upper >= ci.point_estimate - 0.01

    def test_single_trace_returns_empty(self):
        """Single trace → insufficient for bootstrap, returns empty."""
        anns = [_ann("t1", "x", 5)]
        dets = [_det("t1", "x", 5)]
        cis = bootstrap_ci(anns, dets, n_iterations=50)
        assert cis == {}  # n < 2 → empty

    def test_perfect_score_narrow_ci(self):
        """Perfect detection → CI should be near [1.0, 1.0]."""
        anns = [_ann(f"t{i}", "x", 5) for i in range(10)]
        dets = [_det(f"t{i}", "x", 5) for i in range(10)]
        cis = bootstrap_ci(anns, dets, n_iterations=200)
        f1_ci = cis["macro_f1"]
        assert f1_ci.ci_lower >= 0.9
        assert f1_ci.ci_upper <= 1.01

    def test_to_dict(self):
        ci = BootstrapCI(
            metric="test", point_estimate=0.5,
            ci_lower=0.3, ci_upper=0.7, n_iterations=100,
        )
        d = ci.to_dict()
        assert d["metric"] == "test"
        assert d["n_iterations"] == 100


# ── mcnemar_test ────────────────────────────────────────────────────

class TestMcNemarTest:
    def test_identical_modes(self):
        """Identical results → McNemar statistic 0, p-value 1.0."""
        anns = [_ann("t1", "x", 5)]
        dets = [_det("t1", "x", 5)]
        result = mcnemar_test(anns, dets, dets, match_window=5)
        assert result.mcnemar_statistic == 0.0
        assert result.p_value == 1.0
        assert not result.significant

    def test_different_modes(self):
        """One mode detects, other doesn't → should have statistic > 0."""
        anns = [
            _ann("t1", "x", 5),
            _ann("t2", "y", 10),
            _ann("t3", "z", 15),
        ]
        dets_a = [_det("t1", "x", 5), _det("t2", "y", 10), _det("t3", "z", 15)]
        dets_b = []  # mode B detects nothing
        result = mcnemar_test(anns, dets_a, dets_b, match_window=5)
        assert result.a_better
        # With 3 discordant pairs, statistic should be > 0
        assert result.mcnemar_statistic > 0

    def test_p_value_range(self):
        """p-value should be in [0, 1]."""
        anns = [_ann("t1", "x", 5)]
        dets_a = [_det("t1", "x", 5)]
        dets_b = []
        result = mcnemar_test(anns, dets_a, dets_b, match_window=5)
        assert 0.0 <= result.p_value <= 1.0


# ── normal_sf ───────────────────────────────────────────────────────

class TestNormalSF:
    def test_zero(self):
        assert _normal_sf(0) == pytest.approx(0.5, abs=0.01)

    def test_large_z(self):
        assert _normal_sf(5.0) < 0.001

    def test_negative_z(self):
        assert _normal_sf(-2.0) == pytest.approx(1.0 - _normal_sf(2.0), abs=0.01)

    def test_known_value(self):
        """sf(1.96) ~ 0.025."""
        assert _normal_sf(1.96) == pytest.approx(0.025, abs=0.005)


# ── compare_modes ───────────────────────────────────────────────────

class TestCompareModes:
    def test_basic_comparison(self):
        anns = [_ann("t1", "x", 5)]
        dets_strict = [_det("t1", "x", 5)]
        dets_loose = [_det("t1", "x", 5), _det("t1", "y", 10)]

        r_strict = compute_evaluation(anns, dets_strict, "strict", skip_bootstrap=True)
        r_loose = compute_evaluation(anns, dets_loose, "loose", skip_bootstrap=True)

        reports = {"strict": r_strict, "loose": r_loose}
        det_map = {"strict": dets_strict, "loose": dets_loose}

        comparison = compare_modes(reports, anns, det_map)
        assert isinstance(comparison, ComparisonTable)
        assert isinstance(comparison.per_detector_winners, dict)

    def test_per_detector_winners(self):
        """Per-detector winner should be the mode with highest F1."""
        anns = [_ann("t1", "x", 5)]
        dets_a = [_det("t1", "x", 5)]  # TP
        dets_b = [_det("t1", "x", 5), _det("t1", "x", 20)]  # TP + FP

        r_a = compute_evaluation(anns, dets_a, "strict", skip_bootstrap=True)
        r_b = compute_evaluation(anns, dets_b, "loose", skip_bootstrap=True)

        reports = {"strict": r_a, "loose": r_b}
        det_map = {"strict": dets_a, "loose": dets_b}

        comparison = compare_modes(reports, anns, det_map)
        assert comparison.per_detector_winners.get("x") == "strict"


# ── format_comparison_table ─────────────────────────────────────────

class TestFormatComparisonTable:
    def test_produces_text(self):
        anns = [_ann("t1", "x", 5)]
        dets = [_det("t1", "x", 5)]
        report = compute_evaluation(anns, dets, "strict", skip_bootstrap=True)
        reports = {"strict": report}
        comparison = ComparisonTable(pairwise=[], per_detector_winners={"x": "strict"})
        text = format_comparison_table(reports, comparison, "2026-03-17")
        assert "AGENTDIAG ABLATION STUDY" in text
        assert "strict" in text
        assert "Precision" in text
