"""Comprehensive tests for the first-class annotation pipeline.

Tests cover:
  1. AnnotationRecord creation, validation, serialization
  2. Builder functions (detector, auto, human, adjudicated)
  3. Legacy conversion from ground_truth_*.json format
  4. AnnotationLedger (CRUD, dedup, persistence, lifecycle filters, merge, stats)
  5. Disagreement computation (pairwise, session bundle, priority, queue)
  6. Evaluation filtering (load_annotation_ledger_for_eval)
  7. OpenViking annotation persistence (mocked — no live API keys required)
"""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agentdiag.annotation_models import (
    CAFT_VERSION,
    CODEBOOK_VERSION,
    ANNOTATION_PROMPT_VERSION,
    LabelStatus,
    AnnotatorType,
    AnnotationRecord,
    build_detector_annotation,
    build_auto_annotation,
    build_human_annotation,
    build_adjudicated_annotation,
    from_ground_truth_trace,
    from_ground_truth_file,
    _severity_to_int,
    _name_to_code,
)
from agentdiag.annotation_store import AnnotationLedger
from agentdiag.disagreement import (
    DisagreementSummary,
    compare_annotations,
    SessionDisagreementBundle,
    compute_session_disagreement_bundle,
    AnnotationPriority,
    compute_annotation_priority,
    rank_annotation_queue,
)


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def sample_record():
    """A valid failure AnnotationRecord."""
    return AnnotationRecord(
        session_id="abc12345",
        trace_id="abc12345",
        annotator_type=AnnotatorType.HUMAN.value,
        annotator_id="test_user",
        label_status=LabelStatus.HUMAN_REVIEWED.value,
        has_failure=True,
        primary_caft_code="2.2",
        onset_step=15,
        severity=4,
        confidence=3,
        free_text_rationale="Repeated same read 12 times",
    )


@pytest.fixture
def clean_record():
    """A valid clean (no failure) AnnotationRecord."""
    return AnnotationRecord(
        session_id="def67890",
        trace_id="def67890",
        annotator_type=AnnotatorType.HUMAN.value,
        annotator_id="test_user",
        label_status=LabelStatus.HUMAN_REVIEWED.value,
        has_failure=False,
        confidence=5,
    )


@pytest.fixture
def ledger_path(tmp_path):
    return tmp_path / "test_ledger.jsonl"


@pytest.fixture
def populated_ledger(ledger_path):
    """Ledger with records across all 4 annotation layers for one session."""
    ledger = AnnotationLedger(ledger_path)

    # Detector layer
    det = AnnotationRecord(
        session_id="sess_001",
        trace_id="sess_001",
        annotator_type=AnnotatorType.DETECTOR.value,
        annotator_id="context_loss",
        label_status=LabelStatus.UNLABELED.value,
        has_failure=True,
        primary_caft_code="2.1",
        onset_step=20,
        severity=3,
        confidence=2,
    )

    # Auto layer
    auto = AnnotationRecord(
        session_id="sess_001",
        trace_id="sess_001",
        annotator_type=AnnotatorType.AUTO.value,
        annotator_id="claude-sonnet-4-5",
        label_status=LabelStatus.AUTO_LABELED.value,
        has_failure=True,
        primary_caft_code="2.2",
        onset_step=18,
        severity=4,
        confidence=3,
    )

    # Human layer
    human = AnnotationRecord(
        session_id="sess_001",
        trace_id="sess_001",
        annotator_type=AnnotatorType.HUMAN.value,
        annotator_id="reviewer_a",
        label_status=LabelStatus.HUMAN_REVIEWED.value,
        has_failure=True,
        primary_caft_code="2.2",
        onset_step=16,
        severity=4,
        confidence=4,
    )

    # Adjudicated layer
    adj = AnnotationRecord(
        session_id="sess_001",
        trace_id="sess_001",
        annotator_type=AnnotatorType.ADJUDICATED.value,
        annotator_id="lead_reviewer",
        label_status=LabelStatus.ADJUDICATED.value,
        has_failure=True,
        primary_caft_code="2.2",
        onset_step=16,
        severity=4,
        confidence=5,
    )

    # A clean session with only human review
    clean = AnnotationRecord(
        session_id="sess_002",
        trace_id="sess_002",
        annotator_type=AnnotatorType.HUMAN.value,
        annotator_id="reviewer_a",
        label_status=LabelStatus.HUMAN_REVIEWED.value,
        has_failure=False,
        confidence=5,
    )

    ledger.add(det)
    ledger.add(auto)
    ledger.add(human)
    ledger.add(adj)
    ledger.add(clean)

    return ledger


# ======================================================================
# 1. AnnotationRecord creation and validation
# ======================================================================

class TestAnnotationRecord:
    def test_default_values(self):
        r = AnnotationRecord()
        assert r.annotator_type == AnnotatorType.DETECTOR.value
        assert r.label_status == LabelStatus.UNLABELED.value
        assert r.has_failure is False
        assert r.severity == 0
        assert r.confidence == 0
        assert len(r.annotation_id) == 12

    def test_unique_annotation_ids(self):
        r1 = AnnotationRecord()
        r2 = AnnotationRecord()
        assert r1.annotation_id != r2.annotation_id

    def test_version_constants_pinned(self):
        r = AnnotationRecord()
        assert r.caft_version == CAFT_VERSION
        assert r.codebook_version == CODEBOOK_VERSION
        assert r.annotation_prompt_version == ANNOTATION_PROMPT_VERSION

    def test_validate_clean_record(self, clean_record):
        issues = clean_record.validate()
        assert issues == []

    def test_validate_failure_record(self, sample_record):
        issues = sample_record.validate()
        assert issues == []

    def test_validate_missing_session_id(self):
        r = AnnotationRecord(annotator_type=AnnotatorType.HUMAN.value)
        issues = r.validate()
        assert any("session_id" in i or "trace_id" in i for i in issues)

    def test_validate_unknown_annotator_type(self):
        r = AnnotationRecord(session_id="x", annotator_type="alien")
        issues = r.validate()
        assert any("annotator_type" in i for i in issues)

    def test_validate_unknown_label_status(self):
        r = AnnotationRecord(session_id="x", label_status="pending_review")
        issues = r.validate()
        assert any("label_status" in i for i in issues)

    def test_validate_failure_missing_code(self):
        r = AnnotationRecord(
            session_id="x",
            has_failure=True,
            severity=3,
            confidence=3,
        )
        issues = r.validate()
        assert any("primary_caft_code" in i for i in issues)

    def test_validate_failure_invalid_code(self):
        r = AnnotationRecord(
            session_id="x",
            has_failure=True,
            primary_caft_code="99.99",
            severity=3,
            confidence=3,
        )
        issues = r.validate()
        assert any("Unknown CAFT code" in i for i in issues)

    def test_validate_severity_out_of_range(self):
        r = AnnotationRecord(
            session_id="x",
            has_failure=True,
            primary_caft_code="2.2",
            severity=0,
            confidence=3,
        )
        issues = r.validate()
        assert any("severity" in i for i in issues)

    def test_validate_confidence_out_of_range(self):
        r = AnnotationRecord(
            session_id="x",
            has_failure=True,
            primary_caft_code="2.2",
            severity=3,
            confidence=6,
        )
        issues = r.validate()
        assert any("confidence" in i for i in issues)

    def test_validate_invalid_secondary_code(self):
        r = AnnotationRecord(
            session_id="x",
            has_failure=True,
            primary_caft_code="2.2",
            secondary_caft_codes=["2.1", "99.1"],
            severity=3,
            confidence=3,
        )
        issues = r.validate()
        assert any("secondary" in i.lower() for i in issues)


class TestAnnotationRecordProperties:
    def test_primary_caft_name(self, sample_record):
        assert sample_record.primary_caft_name == "step_repetition"

    def test_primary_caft_category(self, sample_record):
        assert sample_record.primary_caft_category == "memory"

    def test_is_observable(self, sample_record):
        assert sample_record.is_observable is True

    def test_is_observable_latent(self):
        r = AnnotationRecord(primary_caft_code="1.1")
        assert r.is_observable is False

    def test_effective_session_id_prefers_trace_id(self):
        r = AnnotationRecord(session_id="sess", trace_id="trace")
        assert r.effective_session_id == "trace"

    def test_effective_session_id_falls_back_to_session_id(self):
        r = AnnotationRecord(session_id="sess", trace_id="")
        assert r.effective_session_id == "sess"

    def test_dedup_key(self, sample_record):
        key = sample_record.dedup_key
        assert key == ("abc12345", "human", "test_user", CAFT_VERSION)

    def test_unknown_code_properties(self):
        r = AnnotationRecord(primary_caft_code="99.99")
        assert r.primary_caft_name == ""
        assert r.primary_caft_category == ""


class TestAnnotationRecordSerialization:
    def test_to_dict_roundtrip(self, sample_record):
        d = sample_record.to_dict()
        r2 = AnnotationRecord.from_dict(d)
        assert r2.session_id == sample_record.session_id
        assert r2.primary_caft_code == sample_record.primary_caft_code
        assert r2.severity == sample_record.severity
        assert r2.annotator_type == sample_record.annotator_type
        assert r2.has_failure == sample_record.has_failure

    def test_from_dict_ignores_unknown_keys(self):
        d = {
            "session_id": "x",
            "annotator_type": "human",
            "unknown_field": "ignored",
            "another_unknown": 42,
        }
        r = AnnotationRecord.from_dict(d)
        assert r.session_id == "x"
        assert not hasattr(r, "unknown_field")

    def test_to_json_produces_valid_json(self, sample_record):
        j = sample_record.to_json()
        d = json.loads(j)
        assert d["session_id"] == "abc12345"
        assert d["has_failure"] is True

    def test_json_roundtrip(self, sample_record):
        j = sample_record.to_json()
        d = json.loads(j)
        r2 = AnnotationRecord.from_dict(d)
        assert r2.session_id == sample_record.session_id
        assert r2.evidence_steps == sample_record.evidence_steps

    def test_secondary_codes_serialized(self):
        r = AnnotationRecord(
            session_id="x",
            secondary_caft_codes=["2.1", "2.3"],
        )
        d = r.to_dict()
        assert d["secondary_caft_codes"] == ["2.1", "2.3"]
        r2 = AnnotationRecord.from_dict(d)
        assert r2.secondary_caft_codes == ["2.1", "2.3"]


# ======================================================================
# 2. Builder functions
# ======================================================================

class TestBuilderFunctions:
    def test_build_detector_annotation(self):
        from agentdiag.caft.base import CaftDiagnosis, CaftSeverity

        diag = CaftDiagnosis(
            caft_code="2.1",
            caft_category="memory",
            failure_name="context_loss",
            severity=CaftSeverity.WARNING,
            confidence=0.75,
            description="Re-read same file 5 times",
            evidence={"file": "main.py", "count": 5},
            at_step=42,
            remediation="Use cached content",
        )
        r = build_detector_annotation("sess_x", diag, detector_name="context_loss_v3")
        assert r.annotator_type == AnnotatorType.DETECTOR.value
        assert r.annotator_id == "context_loss_v3"
        assert r.label_status == LabelStatus.UNLABELED.value
        assert r.has_failure is True
        assert r.primary_caft_code == "2.1"
        assert r.onset_step == 42
        assert r.severity == 3  # warning → 3
        assert 1 <= r.confidence <= 5
        assert r.observable_vs_latent == "observable"
        assert "Re-read" in r.free_text_rationale

    def test_build_detector_uses_failure_name_as_default_id(self):
        from agentdiag.caft.base import CaftDiagnosis, CaftSeverity

        diag = CaftDiagnosis(
            caft_code="2.2",
            caft_category="memory",
            failure_name="step_repetition",
            severity=CaftSeverity.INFO,
            confidence=0.5,
            description="test",
            evidence={},
            at_step=1,
            remediation="",
        )
        r = build_detector_annotation("sess_x", diag)
        assert r.annotator_id == "step_repetition"

    def test_build_auto_annotation(self):
        r = build_auto_annotation(
            session_id="sess_y",
            has_failure=True,
            primary_caft_code="2.2",
            onset_step=10,
            severity=4,
            confidence=3,
            rationale="LLM detected repetition",
        )
        assert r.annotator_type == AnnotatorType.AUTO.value
        assert r.label_status == LabelStatus.AUTO_LABELED.value
        assert r.annotator_id == "claude-sonnet-4-5"
        assert r.has_failure is True
        assert r.primary_caft_code == "2.2"

    def test_build_auto_annotation_custom_model(self):
        r = build_auto_annotation(
            session_id="s",
            has_failure=False,
            annotator_id="gpt-4o",
        )
        assert r.annotator_id == "gpt-4o"

    def test_build_human_annotation(self):
        r = build_human_annotation(
            session_id="sess_z",
            annotator_id="reviewer_b",
            has_failure=True,
            primary_caft_code="4.2",
            evidence_steps=[10, 15, 20],
            severity=5,
            confidence=4,
            rationale="Clear error cascade",
        )
        assert r.annotator_type == AnnotatorType.HUMAN.value
        assert r.label_status == LabelStatus.HUMAN_REVIEWED.value
        assert r.evidence_steps == [10, 15, 20]
        assert r.severity == 5

    def test_build_adjudicated_annotation(self):
        r = build_adjudicated_annotation(
            session_id="sess_w",
            adjudicator_id="lead",
            has_failure=True,
            primary_caft_code="2.2",
            severity=4,
            rationale="Confirmed after reviewing all layers",
        )
        assert r.annotator_type == AnnotatorType.ADJUDICATED.value
        assert r.label_status == LabelStatus.ADJUDICATED.value
        assert r.confidence == 5  # default for adjudicated

    def test_build_adjudicated_clean(self):
        r = build_adjudicated_annotation(
            session_id="s",
            adjudicator_id="lead",
            has_failure=False,
        )
        assert r.has_failure is False
        assert r.primary_caft_code == ""


# ======================================================================
# 3. Legacy conversion
# ======================================================================

class TestLegacyConversion:
    def test_clean_trace(self):
        trace = {
            "session_id": "abc123",
            "actual_failures": [],
            "agent_completed": True,
        }
        records = from_ground_truth_trace(trace, annotator_id="manual")
        assert len(records) == 1
        assert records[0].has_failure is False
        assert records[0].annotator_id == "manual"
        assert "Clean" in records[0].free_text_rationale

    def test_failure_with_details(self):
        trace = {
            "session_id": "def456",
            "actual_failures": ["step_repetition"],
            "failure_details": [
                {
                    "caft_code": "2.2",
                    "onset_step": 15,
                    "severity": 4,
                    "confidence": 3,
                    "rationale": "Repeated read operations",
                }
            ],
        }
        records = from_ground_truth_trace(trace)
        assert len(records) == 1
        assert records[0].has_failure is True
        assert records[0].primary_caft_code == "2.2"
        assert records[0].onset_step == 15

    def test_failure_without_details_uses_name_lookup(self):
        trace = {
            "session_id": "ghi789",
            "actual_failures": ["context_loss"],
        }
        records = from_ground_truth_trace(trace)
        assert len(records) == 1
        assert records[0].has_failure is True
        assert records[0].primary_caft_code == "2.1"
        assert "context_loss" in records[0].free_text_rationale

    def test_multiple_failures_produce_multiple_records(self):
        trace = {
            "session_id": "multi",
            "actual_failures": ["step_repetition", "context_loss"],
            "failure_details": [
                {"caft_code": "2.2", "onset_step": 10, "severity": 3, "confidence": 3, "rationale": "a"},
                {"caft_code": "2.1", "onset_step": 20, "severity": 2, "confidence": 4, "rationale": "b"},
            ],
        }
        records = from_ground_truth_trace(trace)
        assert len(records) == 2
        codes = {r.primary_caft_code for r in records}
        assert codes == {"2.1", "2.2"}

    def test_from_ground_truth_file(self):
        gt = {
            "annotator": "test_annotator",
            "traces": [
                {"session_id": "s1", "actual_failures": [], "agent_completed": True},
                {"session_id": "s2", "actual_failures": ["step_repetition"],
                 "failure_details": [{"caft_code": "2.2", "onset_step": 5, "severity": 3, "confidence": 3, "rationale": "r"}]},
            ],
        }
        records = from_ground_truth_file(gt)
        assert len(records) == 2
        assert records[0].annotator_id == "test_annotator"
        assert records[0].has_failure is False
        assert records[1].has_failure is True

    def test_unknown_failure_name_produces_empty_code(self):
        trace = {
            "session_id": "unk",
            "actual_failures": ["totally_made_up_failure"],
        }
        records = from_ground_truth_trace(trace)
        assert len(records) == 1
        assert records[0].primary_caft_code == ""


class TestHelperFunctions:
    def test_severity_to_int(self):
        assert _severity_to_int("info") == 2
        assert _severity_to_int("warning") == 3
        assert _severity_to_int("critical") == 5
        assert _severity_to_int("unknown") == 3

    def test_name_to_code(self):
        assert _name_to_code("step_repetition") == "2.2"
        assert _name_to_code("context_loss") == "2.1"
        assert _name_to_code("nonexistent") == ""


# ======================================================================
# 4. AnnotationLedger
# ======================================================================

class TestAnnotationLedger:
    def test_create_empty(self, ledger_path):
        ledger = AnnotationLedger(ledger_path)
        assert len(ledger) == 0
        assert ledger.get_all() == []

    def test_add_single(self, ledger_path):
        ledger = AnnotationLedger(ledger_path)
        r = AnnotationRecord(session_id="s1", annotator_type="human", annotator_id="u1")
        is_new = ledger.add(r)
        assert is_new is True
        assert len(ledger) == 1

    def test_persistence_roundtrip(self, ledger_path):
        ledger1 = AnnotationLedger(ledger_path)
        ledger1.add(AnnotationRecord(session_id="s1", annotator_type="human", annotator_id="u1"))
        ledger1.add(AnnotationRecord(session_id="s2", annotator_type="auto", annotator_id="m1"))

        # Reload from disk
        ledger2 = AnnotationLedger(ledger_path)
        assert len(ledger2) == 2
        sessions = ledger2.get_sessions()
        assert "s1" in sessions
        assert "s2" in sessions

    def test_dedup_by_key(self, ledger_path):
        ledger = AnnotationLedger(ledger_path)
        r1 = AnnotationRecord(
            session_id="s1",
            annotator_type="human",
            annotator_id="u1",
            caft_version="1.0",
            severity=2,
        )
        r2 = AnnotationRecord(
            session_id="s1",
            annotator_type="human",
            annotator_id="u1",
            caft_version="1.0",
            severity=4,  # updated severity
        )
        assert ledger.add(r1) is True
        assert ledger.add(r2) is False  # deduped (same key, newer replaces)
        assert len(ledger) == 1
        # The record should be updated to the newer one
        assert ledger.get_all()[0].severity == 4

    def test_different_annotator_types_not_deduped(self, ledger_path):
        ledger = AnnotationLedger(ledger_path)
        det = AnnotationRecord(session_id="s1", annotator_type="detector", annotator_id="ctx")
        human = AnnotationRecord(session_id="s1", annotator_type="human", annotator_id="ctx")
        ledger.add(det)
        ledger.add(human)
        assert len(ledger) == 2

    def test_get_for_session(self, populated_ledger):
        records = populated_ledger.get_for_session("sess_001")
        assert len(records) == 4  # detector, auto, human, adjudicated
        types = {r.annotator_type for r in records}
        assert types == {"detector", "auto", "human", "adjudicated"}

    def test_get_by_type(self, populated_ledger):
        humans = populated_ledger.get_by_type("human")
        assert len(humans) == 2  # reviewer_a for sess_001 + sess_002

    def test_get_by_status(self, populated_ledger):
        adjudicated = populated_ledger.get_by_status(LabelStatus.ADJUDICATED.value)
        assert len(adjudicated) == 1
        assert adjudicated[0].session_id == "sess_001"

    def test_get_sessions(self, populated_ledger):
        sessions = populated_ledger.get_sessions()
        assert "sess_001" in sessions
        assert "sess_002" in sessions

    def test_add_many(self, ledger_path):
        ledger = AnnotationLedger(ledger_path)
        records = [
            AnnotationRecord(session_id=f"s{i}", annotator_type="human", annotator_id="u1")
            for i in range(10)
        ]
        count = ledger.add_many(records)
        assert count == 10
        assert len(ledger) == 10

    def test_add_many_with_dedup(self, ledger_path):
        ledger = AnnotationLedger(ledger_path)
        r1 = AnnotationRecord(session_id="s1", annotator_type="human", annotator_id="u1")
        ledger.add(r1)
        records = [
            AnnotationRecord(session_id="s1", annotator_type="human", annotator_id="u1"),
            AnnotationRecord(session_id="s2", annotator_type="human", annotator_id="u1"),
        ]
        count = ledger.add_many(records)
        assert count == 1  # s1 deduped, s2 new
        assert len(ledger) == 2

    def test_corrupted_lines_skipped_on_load(self, ledger_path):
        # Write some valid and invalid JSONL
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ledger_path, "w") as f:
            r = AnnotationRecord(session_id="valid", annotator_type="human", annotator_id="u")
            f.write(r.to_json() + "\n")
            f.write("THIS IS NOT JSON\n")
            f.write("{}\n")  # empty dict — will create a record with defaults
            f.write("\n")  # empty line — skipped
        ledger = AnnotationLedger(ledger_path)
        assert len(ledger) >= 1  # at least the valid one loaded


class TestAnnotationLedgerLifecycleFilters:
    def test_get_gold_annotations(self, populated_ledger):
        gold = populated_ledger.get_gold_annotations()
        assert len(gold) == 1
        assert all(r.label_status == LabelStatus.ADJUDICATED.value for r in gold)

    def test_get_trainable_annotations(self, populated_ledger):
        trainable = populated_ledger.get_trainable_annotations()
        # adjudicated (1) + human_reviewed (2) = 3
        assert len(trainable) == 3
        statuses = {r.label_status for r in trainable}
        assert statuses == {LabelStatus.ADJUDICATED.value, LabelStatus.HUMAN_REVIEWED.value}

    def test_get_eval_annotations(self, ledger_path):
        ledger = AnnotationLedger(ledger_path)
        ledger.add(AnnotationRecord(
            session_id="s1", annotator_type="adjudicated",
            annotator_id="lead", label_status=LabelStatus.ADJUDICATED.value,
        ))
        ledger.add(AnnotationRecord(
            session_id="s2", annotator_type="adjudicated",
            annotator_id="lead", label_status=LabelStatus.HELD_OUT_TEST.value,
        ))
        ledger.add(AnnotationRecord(
            session_id="s3", annotator_type="auto",
            annotator_id="llm", label_status=LabelStatus.AUTO_LABELED.value,
        ))
        eval_recs = ledger.get_eval_annotations()
        assert len(eval_recs) == 2
        session_ids = {r.session_id for r in eval_recs}
        assert session_ids == {"s1", "s2"}

    def test_get_unlabeled_sessions(self, ledger_path):
        ledger = AnnotationLedger(ledger_path)
        # Session with only detector prediction
        ledger.add(AnnotationRecord(
            session_id="det_only", annotator_type="detector",
            annotator_id="ctx", label_status="unlabeled",
        ))
        # Session with human review
        ledger.add(AnnotationRecord(
            session_id="reviewed", annotator_type="human",
            annotator_id="u1", label_status="human_reviewed",
        ))
        unlabeled = ledger.get_unlabeled_sessions()
        assert "det_only" in unlabeled
        assert "reviewed" not in unlabeled

    def test_get_best_label_trust_hierarchy(self, populated_ledger):
        best = populated_ledger.get_best_label("sess_001")
        assert best is not None
        assert best.annotator_type == AnnotatorType.ADJUDICATED.value

    def test_get_best_label_human_over_auto(self, ledger_path):
        ledger = AnnotationLedger(ledger_path)
        ledger.add(AnnotationRecord(
            session_id="s1", annotator_type="auto", annotator_id="llm",
        ))
        ledger.add(AnnotationRecord(
            session_id="s1", annotator_type="human", annotator_id="u1",
        ))
        best = ledger.get_best_label("s1")
        assert best.annotator_type == "human"

    def test_get_best_label_none_for_unknown(self, populated_ledger):
        assert populated_ledger.get_best_label("nonexistent") is None


class TestAnnotationLedgerMerge:
    def test_merge_from_other_ledger(self, tmp_path):
        path_a = tmp_path / "a.jsonl"
        path_b = tmp_path / "b.jsonl"

        ledger_a = AnnotationLedger(path_a)
        ledger_a.add(AnnotationRecord(session_id="s1", annotator_type="human", annotator_id="u1"))

        ledger_b = AnnotationLedger(path_b)
        ledger_b.add(AnnotationRecord(session_id="s2", annotator_type="human", annotator_id="u1"))
        ledger_b.add(AnnotationRecord(session_id="s3", annotator_type="auto", annotator_id="llm"))

        count = ledger_a.merge_from(ledger_b)
        assert count == 2
        assert len(ledger_a) == 3

    def test_merge_records_dedup(self, ledger_path):
        ledger = AnnotationLedger(ledger_path)
        ledger.add(AnnotationRecord(session_id="s1", annotator_type="human", annotator_id="u1"))

        new_records = [
            AnnotationRecord(session_id="s1", annotator_type="human", annotator_id="u1"),  # dup
            AnnotationRecord(session_id="s2", annotator_type="human", annotator_id="u1"),  # new
        ]
        count = ledger.merge_records(new_records)
        assert count == 1
        assert len(ledger) == 2


class TestAnnotationLedgerStats:
    def test_stats(self, populated_ledger):
        s = populated_ledger.stats()
        assert s["total_records"] == 5
        assert s["unique_sessions"] == 2
        assert "human" in s["by_annotator_type"]
        assert s["gold_count"] == 1
        assert s["trainable_count"] == 3

    def test_stats_empty_ledger(self, ledger_path):
        ledger = AnnotationLedger(ledger_path)
        s = ledger.stats()
        assert s["total_records"] == 0
        assert s["unique_sessions"] == 0
        assert s["gold_count"] == 0

    def test_stats_failure_types(self, ledger_path):
        ledger = AnnotationLedger(ledger_path)
        ledger.add(AnnotationRecord(
            session_id="s1", annotator_type="human", annotator_id="u",
            has_failure=True, primary_caft_code="2.2",
        ))
        ledger.add(AnnotationRecord(
            session_id="s2", annotator_type="human", annotator_id="u",
            has_failure=True, primary_caft_code="2.2",
        ))
        s = ledger.stats()
        assert s["by_failure_type"]["step_repetition"] == 2


# ======================================================================
# 5. Disagreement computation
# ======================================================================

class TestCompareAnnotations:
    def test_perfect_agreement(self):
        a = AnnotationRecord(
            session_id="s1", annotator_type="detector", annotator_id="d",
            has_failure=True, primary_caft_code="2.2", severity=3, onset_step=10,
            confidence=3,
        )
        b = AnnotationRecord(
            session_id="s1", annotator_type="human", annotator_id="h",
            has_failure=True, primary_caft_code="2.2", severity=3, onset_step=10,
            confidence=3,
        )
        ds = compare_annotations(a, b)
        assert ds.binary_agree is True
        assert ds.code_agree is True
        assert ds.severity_delta == 0
        assert ds.onset_delta == 0
        assert not ds.has_disagreement
        assert ds.description == "agree"

    def test_binary_disagreement(self):
        a = AnnotationRecord(
            session_id="s1", annotator_type="detector", annotator_id="d",
            has_failure=True, primary_caft_code="2.2", severity=3, onset_step=10,
        )
        b = AnnotationRecord(
            session_id="s1", annotator_type="human", annotator_id="h",
            has_failure=False,
        )
        ds = compare_annotations(a, b)
        assert ds.binary_agree is False
        assert ds.has_disagreement is True
        assert "binary" in ds.description

    def test_code_disagreement(self):
        a = AnnotationRecord(
            session_id="s1", annotator_type="auto",
            has_failure=True, primary_caft_code="2.1",
        )
        b = AnnotationRecord(
            session_id="s1", annotator_type="human",
            has_failure=True, primary_caft_code="2.2",
        )
        ds = compare_annotations(a, b)
        assert ds.binary_agree is True
        assert ds.code_agree is False
        assert ds.has_disagreement is True
        assert "code" in ds.description

    def test_severity_delta_in_description(self):
        a = AnnotationRecord(
            session_id="s1", has_failure=True, primary_caft_code="2.2",
            severity=1, annotator_type="auto",
        )
        b = AnnotationRecord(
            session_id="s1", has_failure=True, primary_caft_code="2.2",
            severity=5, annotator_type="human",
        )
        ds = compare_annotations(a, b)
        assert ds.severity_delta == 4
        assert "severity" in ds.description

    def test_onset_delta(self):
        a = AnnotationRecord(
            session_id="s1", has_failure=True, primary_caft_code="2.2",
            onset_step=5, annotator_type="auto",
        )
        b = AnnotationRecord(
            session_id="s1", has_failure=True, primary_caft_code="2.2",
            onset_step=50, annotator_type="human",
        )
        ds = compare_annotations(a, b)
        assert ds.onset_delta == 45
        assert "onset" in ds.description


class TestSessionDisagreementBundle:
    def test_full_bundle(self):
        det = AnnotationRecord(
            session_id="s1", annotator_type="detector", annotator_id="d",
            has_failure=True, primary_caft_code="2.1",
        )
        auto = AnnotationRecord(
            session_id="s1", annotator_type="auto", annotator_id="llm",
            has_failure=True, primary_caft_code="2.2",
        )
        human = AnnotationRecord(
            session_id="s1", annotator_type="human", annotator_id="u",
            has_failure=True, primary_caft_code="2.2",
        )
        adj = AnnotationRecord(
            session_id="s1", annotator_type="adjudicated", annotator_id="lead",
            has_failure=True, primary_caft_code="2.2",
        )
        bundle = compute_session_disagreement_bundle("s1", [det, auto, human, adj])

        assert bundle.detector_vs_auto is not None
        assert bundle.detector_vs_auto.has_disagreement  # 2.1 vs 2.2
        assert bundle.auto_vs_human is not None
        assert not bundle.auto_vs_human.has_disagreement  # both 2.2
        assert bundle.human_vs_adjudicated is not None
        assert bundle.detector_vs_human is not None

    def test_partial_bundle(self):
        det = AnnotationRecord(
            session_id="s1", annotator_type="detector", annotator_id="d",
            has_failure=True, primary_caft_code="2.1",
        )
        human = AnnotationRecord(
            session_id="s1", annotator_type="human", annotator_id="u",
            has_failure=False,
        )
        bundle = compute_session_disagreement_bundle("s1", [det, human])
        assert bundle.detector_vs_auto is None  # no auto record
        assert bundle.auto_vs_human is None     # no auto record
        assert bundle.detector_vs_human is not None
        assert bundle.detector_vs_human.has_disagreement  # failure vs clean

    def test_empty_records(self):
        bundle = compute_session_disagreement_bundle("s1", [])
        assert bundle.total_disagreements == 0
        assert bundle.any_disagreement is False

    def test_total_disagreements(self):
        det = AnnotationRecord(
            session_id="s1", annotator_type="detector", has_failure=True,
            primary_caft_code="2.1",
        )
        auto = AnnotationRecord(
            session_id="s1", annotator_type="auto", has_failure=False,
        )
        human = AnnotationRecord(
            session_id="s1", annotator_type="human", has_failure=True,
            primary_caft_code="2.2",
        )
        bundle = compute_session_disagreement_bundle("s1", [det, auto, human])
        # det vs auto: disagree (binary). auto vs human: disagree (binary). det vs human: disagree (code)
        assert bundle.total_disagreements >= 2

    def test_to_dict(self):
        bundle = SessionDisagreementBundle(session_id="s1")
        d = bundle.to_dict()
        assert d["session_id"] == "s1"
        assert d["total_disagreements"] == 0
        assert d["detector_vs_auto"] is None


class TestAnnotationPriority:
    def test_unlabeled_session_high_priority(self):
        p = compute_annotation_priority("s1", [])
        assert p.score >= 10.0
        assert "unlabeled" in " ".join(p.reasons).lower()

    def test_adjudicated_session_zero_priority(self):
        adj = AnnotationRecord(
            session_id="s1", annotator_type="adjudicated",
            label_status=LabelStatus.ADJUDICATED.value,
            has_failure=True, primary_caft_code="2.2", severity=5, confidence=5,
        )
        p = compute_annotation_priority("s1", [adj])
        assert p.score == 0.0

    def test_high_severity_increases_priority(self):
        r = AnnotationRecord(
            session_id="s1", annotator_type="detector",
            has_failure=True, primary_caft_code="2.2",
            severity=5, confidence=4,
        )
        p = compute_annotation_priority("s1", [r])
        assert p.severity_score >= 8.0

    def test_low_confidence_increases_priority(self):
        r = AnnotationRecord(
            session_id="s1", annotator_type="auto",
            has_failure=True, primary_caft_code="2.2",
            severity=3, confidence=1,
        )
        p = compute_annotation_priority("s1", [r])
        assert p.uncertainty_score >= 6.0
        assert "confidence" in " ".join(p.reasons).lower()

    def test_disagreement_increases_priority(self):
        det = AnnotationRecord(
            session_id="s1", annotator_type="detector",
            has_failure=True, primary_caft_code="2.1",
        )
        auto = AnnotationRecord(
            session_id="s1", annotator_type="auto",
            has_failure=False,
        )
        p = compute_annotation_priority("s1", [det, auto])
        assert p.disagreement_score > 0

    def test_novelty_score_rare_type(self):
        r = AnnotationRecord(
            session_id="s1", annotator_type="detector",
            has_failure=True, primary_caft_code="2.2",
            severity=3, confidence=3,
        )
        failure_counts = {"step_repetition": 1}
        p = compute_annotation_priority("s1", [r], failure_counts)
        assert p.novelty_score >= 5.0

    def test_no_human_review_bonus(self):
        r = AnnotationRecord(
            session_id="s1", annotator_type="auto",
            has_failure=True, primary_caft_code="2.2",
            severity=3, confidence=3,
        )
        p = compute_annotation_priority("s1", [r])
        assert p.unlabeled_score > 0


class TestRankAnnotationQueue:
    def test_ranking_order(self):
        records_by_session = {
            "adjudicated": [AnnotationRecord(
                session_id="adjudicated", annotator_type="adjudicated",
                label_status=LabelStatus.ADJUDICATED.value,
                has_failure=False, confidence=5,
            )],
            "high_severity": [AnnotationRecord(
                session_id="high_severity", annotator_type="detector",
                has_failure=True, primary_caft_code="2.2",
                severity=5, confidence=2,
            )],
            "low_severity": [AnnotationRecord(
                session_id="low_severity", annotator_type="detector",
                has_failure=True, primary_caft_code="2.2",
                severity=1, confidence=5,
            )],
        }
        queue = rank_annotation_queue(records_by_session)
        # Adjudicated should be excluded (score=0), high_severity before low_severity
        assert len(queue) == 2
        assert queue[0].session_id == "high_severity"
        assert queue[1].session_id == "low_severity"

    def test_limit_parameter(self):
        records_by_session = {
            f"s{i}": [AnnotationRecord(
                session_id=f"s{i}", annotator_type="detector",
                has_failure=True, primary_caft_code="2.2",
                severity=3, confidence=3,
            )]
            for i in range(10)
        }
        queue = rank_annotation_queue(records_by_session, limit=3)
        assert len(queue) == 3

    def test_empty_input(self):
        queue = rank_annotation_queue({})
        assert queue == []


# ======================================================================
# 6. Evaluation filtering
# ======================================================================

class TestLoadAnnotationLedgerForEval:
    def test_gold_filter(self, tmp_path):
        from agentdiag.evaluate import load_annotation_ledger_for_eval

        ledger_path = tmp_path / "eval_ledger.jsonl"
        ledger = AnnotationLedger(ledger_path)
        ledger.add(AnnotationRecord(
            session_id="s1", annotator_type="adjudicated", annotator_id="lead",
            label_status=LabelStatus.ADJUDICATED.value,
            has_failure=True, primary_caft_code="2.2", severity=4, confidence=5,
        ))
        ledger.add(AnnotationRecord(
            session_id="s2", annotator_type="human", annotator_id="u1",
            label_status=LabelStatus.HUMAN_REVIEWED.value,
            has_failure=False, confidence=4,
        ))

        result = load_annotation_ledger_for_eval(ledger_path, "gold")
        assert "s1" in result
        assert "s2" not in result  # human_reviewed excluded from gold
        assert result["s1"]["has_failure"] is True
        assert result["s1"]["primary_caft_code"] == "2.2"

    def test_trainable_filter(self, tmp_path):
        from agentdiag.evaluate import load_annotation_ledger_for_eval

        ledger_path = tmp_path / "eval_ledger.jsonl"
        ledger = AnnotationLedger(ledger_path)
        ledger.add(AnnotationRecord(
            session_id="s1", annotator_type="adjudicated", annotator_id="lead",
            label_status=LabelStatus.ADJUDICATED.value,
            has_failure=True, primary_caft_code="2.2",
        ))
        ledger.add(AnnotationRecord(
            session_id="s2", annotator_type="human", annotator_id="u1",
            label_status=LabelStatus.HUMAN_REVIEWED.value,
            has_failure=False,
        ))
        ledger.add(AnnotationRecord(
            session_id="s3", annotator_type="auto", annotator_id="llm",
            label_status=LabelStatus.AUTO_LABELED.value,
            has_failure=True, primary_caft_code="2.1",
        ))

        result = load_annotation_ledger_for_eval(ledger_path, "trainable")
        assert "s1" in result
        assert "s2" in result
        assert "s3" not in result  # auto excluded from trainable

    def test_nonexistent_path_returns_empty(self, tmp_path):
        from agentdiag.evaluate import load_annotation_ledger_for_eval

        result = load_annotation_ledger_for_eval(tmp_path / "missing.jsonl")
        assert result == {}

    def test_eval_format_compatibility(self, tmp_path):
        from agentdiag.evaluate import load_annotation_ledger_for_eval

        ledger_path = tmp_path / "eval_ledger.jsonl"
        ledger = AnnotationLedger(ledger_path)
        ledger.add(AnnotationRecord(
            session_id="s1", annotator_type="adjudicated", annotator_id="lead",
            label_status=LabelStatus.ADJUDICATED.value,
            has_failure=True, primary_caft_code="2.2",
            secondary_caft_codes=["2.1"],
            severity=4, confidence=5,
        ))

        result = load_annotation_ledger_for_eval(ledger_path, "gold")
        ann = result["s1"]
        # Verify fields expected by _compute_annotation_metrics
        assert "trace_id" in ann
        assert "has_failure" in ann
        assert "primary_caft_subtype" in ann
        assert "primary_caft_code" in ann
        assert "secondary_failures" in ann
        assert "severity" in ann
        assert ann["primary_caft_subtype"] == "step_repetition"
        assert ann["secondary_failures"] == ["context_loss"]


# ======================================================================
# 7. OpenViking annotation persistence (mocked)
# ======================================================================

class TestOpenVikingAnnotation:
    """Tests for annotation methods on ContextStore.

    All OpenViking calls are mocked — no live API keys required.
    """

    @pytest.fixture
    def mock_context_store(self, tmp_path):
        """Create a ContextStore with mocked OpenViking client."""
        db_path = str(tmp_path / "test_context")

        with patch("agentdiag.context.openviking._ensure_ov_config"):
            with patch("agentdiag.context.openviking.SyncOpenViking") as MockOV:
                mock_client = MagicMock()
                mock_client.create_session.return_value = {"session_id": "mock_sess_001"}
                mock_client.add_message.return_value = None
                mock_client.commit_session.return_value = {}
                MockOV.return_value = mock_client

                from agentdiag.context.openviking import ContextStore
                store = ContextStore(db_path=db_path)
                store._client = mock_client
                yield store

    def test_record_annotation_writes_to_ledger(self, mock_context_store, tmp_path):
        store = mock_context_store
        record = build_human_annotation(
            session_id="sess_test",
            annotator_id="reviewer",
            has_failure=True,
            primary_caft_code="2.2",
            severity=3,
            confidence=4,
        )
        result = store.record_annotation(record)
        assert result is True

        # Verify annotation was written to local JSONL ledger
        ledger = store._annotation_ledger_path
        assert ledger.exists()
        with open(ledger) as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["session_id"] == "sess_test"
        assert data["has_failure"] is True

    def test_record_annotation_also_writes_to_ov(self, mock_context_store):
        store = mock_context_store
        store._session_id = "mock_sess_001"
        record = build_auto_annotation(
            session_id="mock_sess_001",
            has_failure=False,
        )
        store.record_annotation(record)
        # OV add_message should have been called
        store._client.add_message.assert_called()

    def test_get_annotations_for_session(self, mock_context_store):
        store = mock_context_store
        # Write two annotations for same session
        r1 = build_detector_annotation.__wrapped__ if hasattr(build_detector_annotation, '__wrapped__') else None
        rec1 = build_human_annotation(
            session_id="sess_abc", annotator_id="u1",
            has_failure=True, primary_caft_code="2.2", severity=3, confidence=3,
        )
        rec2 = build_auto_annotation(
            session_id="sess_abc", has_failure=True,
            primary_caft_code="2.2", severity=3, confidence=2,
        )
        store.record_annotation(rec1)
        store.record_annotation(rec2)

        annotations = store.get_annotations_for_session("sess_abc")
        assert len(annotations) == 2

    def test_get_annotations_empty_for_unknown(self, mock_context_store):
        annotations = mock_context_store.get_annotations_for_session("nonexistent")
        assert annotations == []

    def test_find_annotation_needed_cases(self, mock_context_store):
        store = mock_context_store
        # Write a case to the case ledger
        case_data = {
            "case_id": "sess_001_10_step_repetition",
            "session_id": "sess_001",
            "failure_name": "step_repetition",
            "severity": "warning",
            "status": "predicted",
        }
        store._ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with open(store._ledger_path, "a") as f:
            f.write(json.dumps(case_data) + "\n")

        needed = store.find_annotation_needed_cases()
        assert len(needed) == 1
        assert needed[0]["session_id"] == "sess_001"

    def test_find_annotation_needed_excludes_annotated(self, mock_context_store):
        store = mock_context_store
        # Write a case
        case_data = {
            "case_id": "sess_001_10_step_repetition",
            "session_id": "sess_001",
            "failure_name": "step_repetition",
            "severity": "warning",
            "status": "predicted",
        }
        store._ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with open(store._ledger_path, "a") as f:
            f.write(json.dumps(case_data) + "\n")

        # Write a human annotation for that session
        rec = build_human_annotation(
            session_id="sess_001", annotator_id="u1",
            has_failure=True, primary_caft_code="2.2", severity=3, confidence=4,
        )
        store.record_annotation(rec)

        needed = store.find_annotation_needed_cases()
        assert len(needed) == 0  # sess_001 is now annotated

    def test_record_adjudicated_label(self, mock_context_store):
        store = mock_context_store
        result = store.record_adjudicated_label(
            session_id="sess_adj",
            has_failure=True,
            primary_caft_code="2.2",
            adjudicator_id="lead",
            rationale="Confirmed step repetition",
            severity=4,
        )
        assert result is True

        # Verify it was written to the annotation ledger
        annotations = store.get_annotations_for_session("sess_adj")
        assert len(annotations) == 1
        assert annotations[0]["annotator_type"] == "adjudicated"
        assert annotations[0]["label_status"] == "adjudicated"

    def test_record_annotation_graceful_failure(self, mock_context_store):
        store = mock_context_store
        # Force an exception on ledger write
        store._append_to_annotation_ledger = MagicMock(side_effect=Exception("disk full"))
        record = build_human_annotation(
            session_id="s", annotator_id="u", has_failure=False,
        )
        # Should not raise — graceful degradation
        result = store.record_annotation(record)
        # Result is False because the exception is caught
        assert result is False


# ======================================================================
# 8. Edge cases and integration
# ======================================================================

class TestEdgeCases:
    def test_ledger_repr(self, ledger_path):
        ledger = AnnotationLedger(ledger_path)
        r = repr(ledger)
        assert "AnnotationLedger" in r
        assert "0 records" in r

    def test_ledger_len(self, populated_ledger):
        assert len(populated_ledger) == 5

    def test_disagreement_summary_to_dict(self):
        ds = DisagreementSummary(
            source_a="detector", source_b="human", session_id="s1",
            binary_agree=False, description="binary disagreement",
        )
        d = ds.to_dict()
        assert d["source_a"] == "detector"
        assert d["binary_agree"] is False

    def test_annotation_priority_to_dict(self):
        p = AnnotationPriority(session_id="s1", score=15.0, reasons=["high severity"])
        d = p.to_dict()
        assert d["score"] == 15.0
        assert d["reasons"] == ["high severity"]

    def test_label_status_values(self):
        """Ensure all expected statuses exist."""
        assert LabelStatus.UNLABELED.value == "unlabeled"
        assert LabelStatus.AUTO_LABELED.value == "auto_labeled"
        assert LabelStatus.HUMAN_REVIEWED.value == "human_reviewed"
        assert LabelStatus.ADJUDICATED.value == "adjudicated"
        assert LabelStatus.HELD_OUT_TEST.value == "held_out_test"

    def test_annotator_type_values(self):
        """Ensure all expected types exist."""
        assert AnnotatorType.DETECTOR.value == "detector"
        assert AnnotatorType.AUTO.value == "auto"
        assert AnnotatorType.HUMAN.value == "human"
        assert AnnotatorType.ADJUDICATED.value == "adjudicated"

    def test_ledger_handles_concurrent_like_writes(self, ledger_path):
        """Multiple add operations in sequence behave correctly."""
        ledger = AnnotationLedger(ledger_path)
        for i in range(50):
            ledger.add(AnnotationRecord(
                session_id=f"s{i}",
                annotator_type="human",
                annotator_id="u1",
            ))
        assert len(ledger) == 50
        # Reload and verify
        ledger2 = AnnotationLedger(ledger_path)
        assert len(ledger2) == 50
