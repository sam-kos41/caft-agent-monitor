"""Tests for the annotation schema, store, and agreement metrics."""

import json

import pytest

from agentdiag.annotation import (
    Annotation,
    AnnotationStore,
    AgreementReport,
    compute_agreement,
    _cohens_kappa,
)


class TestAnnotation:
    def test_clean_annotation_validates(self):
        a = Annotation(trace_id="t1", annotator_id="ann1", has_failure=False)
        assert a.validate() == []

    def test_failure_annotation_validates(self):
        a = Annotation(
            trace_id="t1",
            annotator_id="ann1",
            has_failure=True,
            primary_caft_code="2.2",
            primary_caft_category="memory",
            primary_caft_subtype="step_repetition",
            failure_onset_step=5,
            severity=3,
            annotator_confidence=4,
            evidence_steps=[5, 6, 7],
        )
        assert a.validate() == []

    def test_missing_code_fails_validation(self):
        a = Annotation(trace_id="t1", annotator_id="ann1", has_failure=True,
                       severity=3, annotator_confidence=4, evidence_steps=[1],
                       failure_onset_step=1)
        issues = a.validate()
        assert any("primary_caft_code" in i for i in issues)

    def test_invalid_code_fails_validation(self):
        a = Annotation(trace_id="t1", annotator_id="ann1", has_failure=True,
                       primary_caft_code="99.99", severity=3,
                       annotator_confidence=4, evidence_steps=[1],
                       failure_onset_step=1)
        issues = a.validate()
        assert any("Unknown CAFT code" in i for i in issues)

    def test_missing_onset_fails_validation(self):
        a = Annotation(trace_id="t1", annotator_id="ann1", has_failure=True,
                       primary_caft_code="2.2", severity=3,
                       annotator_confidence=4, evidence_steps=[1])
        issues = a.validate()
        assert any("onset" in i for i in issues)

    def test_severity_out_of_range(self):
        a = Annotation(trace_id="t1", annotator_id="ann1", has_failure=True,
                       primary_caft_code="2.2", failure_onset_step=1,
                       severity=6, annotator_confidence=4, evidence_steps=[1])
        issues = a.validate()
        assert any("Severity" in i for i in issues)

    def test_to_dict_roundtrip(self):
        a = Annotation(
            trace_id="t1", annotator_id="ann1", has_failure=True,
            primary_caft_code="2.2", failure_onset_step=5,
            failure_onset_window=(3, 7),
            severity=3, annotator_confidence=4,
            evidence_steps=[5, 6],
        )
        d = a.to_dict()
        a2 = Annotation.from_dict(d)
        assert a2.trace_id == "t1"
        assert a2.primary_caft_code == "2.2"
        assert a2.failure_onset_window == (3, 7)


class TestAnnotationStore:
    def test_add_and_retrieve(self, tmp_path):
        store = AnnotationStore(tmp_path / "ann.jsonl")
        a = Annotation(trace_id="t1", annotator_id="ann1", has_failure=False)
        store.add(a)
        assert len(store) == 1
        assert store.get_all()[0].trace_id == "t1"

    def test_persistence(self, tmp_path):
        path = tmp_path / "ann.jsonl"
        store1 = AnnotationStore(path)
        store1.add(Annotation(trace_id="t1", annotator_id="ann1", has_failure=False))
        store1.add(Annotation(trace_id="t2", annotator_id="ann1", has_failure=False))

        # Reload from disk
        store2 = AnnotationStore(path)
        assert len(store2) == 2

    def test_get_for_trace(self, tmp_path):
        store = AnnotationStore(tmp_path / "ann.jsonl")
        store.add(Annotation(trace_id="t1", annotator_id="ann1", has_failure=False))
        store.add(Annotation(trace_id="t2", annotator_id="ann1", has_failure=False))
        store.add(Annotation(trace_id="t1", annotator_id="ann2", has_failure=False))

        t1_anns = store.get_for_trace("t1")
        assert len(t1_anns) == 2

    def test_get_by_annotator(self, tmp_path):
        store = AnnotationStore(tmp_path / "ann.jsonl")
        store.add(Annotation(trace_id="t1", annotator_id="ann1", has_failure=False))
        store.add(Annotation(trace_id="t2", annotator_id="ann2", has_failure=False))

        assert len(store.get_by_annotator("ann1")) == 1
        assert len(store.get_by_annotator("ann2")) == 1

    def test_rejects_invalid_annotation(self, tmp_path):
        store = AnnotationStore(tmp_path / "ann.jsonl")
        a = Annotation(trace_id="t1", annotator_id="ann1", has_failure=True,
                       severity=3, annotator_confidence=4, evidence_steps=[1],
                       failure_onset_step=1)
        # Missing primary_caft_code
        with pytest.raises(ValueError, match="Invalid annotation"):
            store.add(a)

    def test_trace_ids(self, tmp_path):
        store = AnnotationStore(tmp_path / "ann.jsonl")
        store.add(Annotation(trace_id="t1", annotator_id="ann1", has_failure=False))
        store.add(Annotation(trace_id="t2", annotator_id="ann1", has_failure=False))
        assert store.trace_ids == {"t1", "t2"}

    def test_annotator_ids(self, tmp_path):
        store = AnnotationStore(tmp_path / "ann.jsonl")
        store.add(Annotation(trace_id="t1", annotator_id="ann1", has_failure=False))
        store.add(Annotation(trace_id="t1", annotator_id="ann2", has_failure=False))
        assert store.annotator_ids == {"ann1", "ann2"}


class TestCohensKappa:
    def test_perfect_agreement(self):
        a = ["A", "B", "A", "B"]
        b = ["A", "B", "A", "B"]
        assert _cohens_kappa(a, b) == 1.0

    def test_no_agreement(self):
        a = ["A", "A", "A", "A"]
        b = ["B", "B", "B", "B"]
        k = _cohens_kappa(a, b)
        assert k <= 0.0  # no agreement (0.0 when all-constant labels)

    def test_partial_agreement(self):
        a = ["A", "B", "A", "B", "A"]
        b = ["A", "B", "B", "B", "A"]
        k = _cohens_kappa(a, b)
        assert 0.0 < k < 1.0

    def test_empty_lists(self):
        assert _cohens_kappa([], []) == 0.0

    def test_single_label(self):
        a = ["A", "A", "A"]
        b = ["A", "A", "A"]
        assert _cohens_kappa(a, b) == 1.0


class TestComputeAgreement:
    def _make_store(self, tmp_path, annotations):
        store = AnnotationStore(tmp_path / "ann.jsonl")
        for a in annotations:
            store.add(a)
        return store

    def test_perfect_binary_agreement(self, tmp_path):
        store = self._make_store(tmp_path, [
            Annotation(trace_id="t1", annotator_id="A", has_failure=False),
            Annotation(trace_id="t1", annotator_id="B", has_failure=False),
            Annotation(trace_id="t2", annotator_id="A", has_failure=True,
                       primary_caft_code="2.2", primary_caft_category="memory",
                       failure_onset_step=5, severity=3, annotator_confidence=4,
                       evidence_steps=[5]),
            Annotation(trace_id="t2", annotator_id="B", has_failure=True,
                       primary_caft_code="2.2", primary_caft_category="memory",
                       failure_onset_step=5, severity=3, annotator_confidence=4,
                       evidence_steps=[5]),
        ])
        report = compute_agreement(store, "A", "B")
        assert report.n_traces == 2
        assert report.binary_kappa == 1.0
        assert report.binary_agreement == 1.0

    def test_no_common_traces(self, tmp_path):
        store = self._make_store(tmp_path, [
            Annotation(trace_id="t1", annotator_id="A", has_failure=False),
            Annotation(trace_id="t2", annotator_id="B", has_failure=False),
        ])
        report = compute_agreement(store, "A", "B")
        assert report.n_traces == 0

    def test_disagreement_lowers_kappa(self, tmp_path):
        store = self._make_store(tmp_path, [
            Annotation(trace_id="t1", annotator_id="A", has_failure=False),
            Annotation(trace_id="t1", annotator_id="B", has_failure=True,
                       primary_caft_code="2.2", primary_caft_category="memory",
                       failure_onset_step=5, severity=3, annotator_confidence=4,
                       evidence_steps=[5]),
            Annotation(trace_id="t2", annotator_id="A", has_failure=True,
                       primary_caft_code="4.2", primary_caft_category="execution",
                       failure_onset_step=10, severity=4, annotator_confidence=3,
                       evidence_steps=[10]),
            Annotation(trace_id="t2", annotator_id="B", has_failure=False),
        ])
        report = compute_agreement(store, "A", "B")
        assert report.binary_kappa < 0.5

    def test_report_summary(self, tmp_path):
        store = self._make_store(tmp_path, [
            Annotation(trace_id="t1", annotator_id="A", has_failure=False),
            Annotation(trace_id="t1", annotator_id="B", has_failure=False),
        ])
        report = compute_agreement(store, "A", "B")
        s = report.summary()
        assert "A vs B" in s
        assert "Binary" in s
