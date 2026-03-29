"""Tests for ground-truth authority rules and annotation cleaning.

Verifies that:
1. Human CLEAN annotations suppress detector firings on the same session
2. Unlabeled detector annotations are excluded from ground truth
3. DRAFT annotations are included but flagged
4. Session ID prefix matching works (short ↔ full UUID)
5. Validation catches circular ground truth
6. The clean annotation loader produces honest ground truth
"""

import json
import tempfile
from pathlib import Path

import pytest

from agentdiag.annotation_models import (
    AnnotationRecord,
    AnnotatorType,
    LabelStatus,
    build_detector_annotation,
    build_human_annotation,
)
from agentdiag.annotation_store import AnnotationLedger
from agentdiag.metrics import (
    Annotation,
    validate_annotations_jsonl,
)


# ── Helpers ──────────────────────────────────────────────────────────

def _write_jsonl(records: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _make_human_clean(session_id: str) -> dict:
    return AnnotationRecord(
        session_id=session_id,
        trace_id=session_id,
        annotator_type="human",
        annotator_id="test-human",
        label_status="human_reviewed",
        has_failure=False,
        confidence=5,
        free_text_rationale="Clean trace",
    ).to_dict()


def _make_human_failure(session_id: str, caft_code: str, failure_name: str) -> dict:
    return AnnotationRecord(
        session_id=session_id,
        trace_id=session_id,
        annotator_type="human",
        annotator_id="test-human",
        label_status="human_reviewed",
        has_failure=True,
        primary_caft_code=caft_code,
        onset_step=0,
        severity=3,
        confidence=4,
        free_text_rationale=f"Human identified {failure_name}",
    ).to_dict()


def _make_detector(session_id: str, failure_name: str, caft_code: str) -> dict:
    return AnnotationRecord(
        session_id=session_id,
        trace_id=session_id,
        annotator_type="detector",
        annotator_id=failure_name,
        label_status="unlabeled",
        has_failure=True,
        primary_caft_code=caft_code,
        onset_step=0,
        severity=3,
        confidence=2,
        free_text_rationale=f"Detector fired: {failure_name}",
    ).to_dict()


def _make_draft(session_id: str, caft_code: str, failure_name: str) -> dict:
    return AnnotationRecord(
        session_id=session_id,
        trace_id=session_id,
        annotator_type="human",
        annotator_id="auto-script",
        label_status="auto_labeled",
        has_failure=True,
        primary_caft_code=caft_code,
        onset_step=0,
        severity=3,
        confidence=2,
        free_text_rationale=f"DRAFT: {failure_name}",
    ).to_dict()


# Import the loader under test (from scripts/)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from run_ablation import load_annotations_from_jsonl, _load_clean_session_ids_from_jsonl


# ── Test: Human CLEAN suppresses detector ────────────────────────────

class TestHumanCLEANSuppression:
    def test_human_clean_excludes_detector_stall(self, tmp_path):
        """Core fix: human says CLEAN → detector stall is NOT ground truth."""
        ann_file = tmp_path / "anns.jsonl"
        _write_jsonl([
            _make_human_clean("abc12345"),
            _make_detector("abc12345-full-uuid-here", "stall", "4.4"),
        ], ann_file)

        annotations = load_annotations_from_jsonl(ann_file)
        assert len(annotations) == 0, "Human CLEAN should suppress detector stall"

    def test_human_clean_makes_session_fp_eligible(self, tmp_path):
        """CLEAN sessions should appear in clean_session_ids for FP counting."""
        ann_file = tmp_path / "anns.jsonl"
        _write_jsonl([
            _make_human_clean("abc12345"),
            _make_detector("abc12345-full-uuid-here", "stall", "4.4"),
        ], ann_file)

        clean_ids = _load_clean_session_ids_from_jsonl(ann_file)
        assert len(clean_ids) == 1
        # Should be the canonical (longest) ID
        assert any("abc12345" in sid for sid in clean_ids)

    def test_multiple_clean_sessions(self, tmp_path):
        ann_file = tmp_path / "anns.jsonl"
        _write_jsonl([
            _make_human_clean("aaa11111"),
            _make_detector("aaa11111-full-uuid", "stall", "4.4"),
            _make_human_clean("bbb22222"),
            _make_detector("bbb22222-full-uuid", "error_cascade", "4.2"),
            _make_human_clean("ccc33333"),
        ], ann_file)

        annotations = load_annotations_from_jsonl(ann_file)
        assert len(annotations) == 0

        clean_ids = _load_clean_session_ids_from_jsonl(ann_file)
        assert len(clean_ids) == 3


# ── Test: Unlabeled detector exclusion ───────────────────────────────

class TestDetectorExclusion:
    def test_detector_only_not_ground_truth(self, tmp_path):
        """Detector annotations without human review are NOT ground truth."""
        ann_file = tmp_path / "anns.jsonl"
        _write_jsonl([
            _make_detector("xyz99999", "stall", "4.4"),
        ], ann_file)

        annotations = load_annotations_from_jsonl(ann_file)
        assert len(annotations) == 0

    def test_detector_with_human_failure_is_included(self, tmp_path):
        """If human agrees with detector (marks failure), it IS ground truth."""
        ann_file = tmp_path / "anns.jsonl"
        _write_jsonl([
            _make_human_failure("xyz99999", "4.4", "stall"),
            _make_detector("xyz99999-full-uuid", "stall", "4.4"),
        ], ann_file)

        annotations = load_annotations_from_jsonl(ann_file)
        assert len(annotations) == 1
        assert annotations[0].failure_name == "stall"


# ── Test: DRAFT annotations ─────────────────────────────────────────

class TestDraftAnnotations:
    def test_drafts_are_included(self, tmp_path):
        """DRAFT (auto_labeled) annotations are included in ground truth."""
        ann_file = tmp_path / "anns.jsonl"
        _write_jsonl([
            _make_draft("ddd44444", "4.4", "stall"),
        ], ann_file)

        annotations = load_annotations_from_jsonl(ann_file)
        assert len(annotations) == 1

    def test_draft_suppressed_by_human_clean(self, tmp_path):
        """If human says CLEAN, DRAFT is also suppressed."""
        ann_file = tmp_path / "anns.jsonl"
        _write_jsonl([
            _make_human_clean("ddd44444"),
            _make_draft("ddd44444-full-uuid", "4.4", "stall"),
        ], ann_file)

        annotations = load_annotations_from_jsonl(ann_file)
        assert len(annotations) == 0


# ── Test: Session ID prefix matching ─────────────────────────────────

class TestSessionIDPrefix:
    def test_short_and_long_same_session(self, tmp_path):
        """Short 8-char ID and full UUID with same prefix = same session."""
        ann_file = tmp_path / "anns.jsonl"
        _write_jsonl([
            _make_human_clean("eee55555"),  # short
            _make_detector("eee55555-aaaa-bbbb-cccc-dddddddddddd", "stall", "4.4"),  # long
        ], ann_file)

        annotations = load_annotations_from_jsonl(ann_file)
        assert len(annotations) == 0, "Same session via prefix match"

    def test_canonical_id_is_longest(self, tmp_path):
        """The canonical session ID should be the longest form."""
        ann_file = tmp_path / "anns.jsonl"
        full_uuid = "fff66666-aaaa-bbbb-cccc-dddddddddddd"
        _write_jsonl([
            _make_human_failure("fff66666", "4.2", "error_cascade"),
            _make_detector(full_uuid, "error_cascade", "4.2"),
        ], ann_file)

        annotations = load_annotations_from_jsonl(ann_file)
        assert len(annotations) == 1
        assert annotations[0].trace_id == full_uuid

    def test_different_prefixes_different_sessions(self, tmp_path):
        """Sessions with different 8-char prefixes are different."""
        ann_file = tmp_path / "anns.jsonl"
        _write_jsonl([
            _make_human_failure("aaa11111", "4.2", "error_cascade"),
            _make_human_failure("bbb22222", "4.4", "stall"),
        ], ann_file)

        annotations = load_annotations_from_jsonl(ann_file)
        assert len(annotations) == 2


# ── Test: AnnotationLedger prefix awareness ──────────────────────────

class TestLedgerPrefixAwareness:
    def test_get_for_session_finds_both_ids(self, tmp_path):
        ledger_path = tmp_path / "ledger.jsonl"
        ledger = AnnotationLedger(ledger_path)

        rec_short = build_human_annotation(
            session_id="ggg77777",
            annotator_id="human",
            has_failure=False,
            confidence=5,
            rationale="Clean",
        )
        from agentdiag.caft.base import CaftDiagnosis, CaftSeverity
        diag = CaftDiagnosis(
            caft_code="4.4", caft_category="execution",
            failure_name="stall", severity=CaftSeverity.WARNING,
            confidence=0.8, description="stall", evidence={},
            at_step=5, remediation="",
        )
        rec_long = build_detector_annotation(
            session_id="ggg77777-aaaa-bbbb-cccc-dddddddddddd",
            diagnosis=diag,
        )

        ledger.add(rec_short)
        ledger.add(rec_long)

        # Looking up short ID should find BOTH records
        results = ledger.get_for_session("ggg77777")
        assert len(results) == 2

        # Looking up long ID should also find BOTH
        results = ledger.get_for_session("ggg77777-aaaa-bbbb-cccc-dddddddddddd")
        assert len(results) == 2

    def test_get_best_label_prefers_human(self, tmp_path):
        ledger_path = tmp_path / "ledger.jsonl"
        ledger = AnnotationLedger(ledger_path)

        rec_human = build_human_annotation(
            session_id="hhh88888",
            annotator_id="human",
            has_failure=False,
            confidence=5,
            rationale="Clean",
        )
        from agentdiag.caft.base import CaftDiagnosis, CaftSeverity
        diag = CaftDiagnosis(
            caft_code="4.4", caft_category="execution",
            failure_name="stall", severity=CaftSeverity.WARNING,
            confidence=0.8, description="stall", evidence={},
            at_step=5, remediation="",
        )
        rec_det = build_detector_annotation(
            session_id="hhh88888-full-uuid",
            diagnosis=diag,
        )

        ledger.add(rec_human)
        ledger.add(rec_det)

        best = ledger.get_best_label("hhh88888")
        assert best is not None
        assert best.annotator_type == "human"
        assert best.has_failure is False

    def test_get_sessions_deduplicates(self, tmp_path):
        ledger_path = tmp_path / "ledger.jsonl"
        ledger = AnnotationLedger(ledger_path)

        rec1 = build_human_annotation(
            session_id="iii99999",
            annotator_id="human",
            has_failure=False,
        )
        from agentdiag.caft.base import CaftDiagnosis, CaftSeverity
        diag = CaftDiagnosis(
            caft_code="4.4", caft_category="execution",
            failure_name="stall", severity=CaftSeverity.WARNING,
            confidence=0.8, description="stall", evidence={},
            at_step=5, remediation="",
        )
        rec2 = build_detector_annotation(
            session_id="iii99999-full-uuid",
            diagnosis=diag,
        )

        ledger.add(rec1)
        ledger.add(rec2)

        sessions = ledger.get_sessions()
        assert len(sessions) == 1
        # Should be canonical (longest)
        assert "iii99999-full-uuid" in sessions


# ── Test: Validation function ────────────────────────────────────────

class TestValidation:
    def test_detects_circular_ground_truth(self, tmp_path):
        """Validation should flag detector-only sessions."""
        ann_file = tmp_path / "anns.jsonl"
        _write_jsonl([
            _make_detector("jjj00001", "stall", "4.4"),
            _make_detector("jjj00002", "error_cascade", "4.2"),
        ], ann_file)

        report = validate_annotations_jsonl(str(ann_file))
        assert not report.is_valid
        assert any("CIRCULAR" in e for e in report.errors)

    def test_clean_data_is_valid(self, tmp_path):
        ann_file = tmp_path / "anns.jsonl"
        _write_jsonl([
            _make_human_clean("kkk00001"),
            _make_human_failure("kkk00002", "4.2", "error_cascade"),
        ], ann_file)

        report = validate_annotations_jsonl(str(ann_file))
        assert report.is_valid
        assert len(report.errors) == 0

    def test_warns_on_drafts(self, tmp_path):
        ann_file = tmp_path / "anns.jsonl"
        _write_jsonl([
            _make_draft("lll00001", "4.4", "stall"),
        ], ann_file)

        report = validate_annotations_jsonl(str(ann_file))
        assert any("DRAFT" in w for w in report.warnings)

    def test_warns_on_class_imbalance(self, tmp_path):
        ann_file = tmp_path / "anns.jsonl"
        records = []
        for i in range(10):
            records.append(_make_human_failure(f"mmm{i:05d}", "4.4", "stall"))
        records.append(_make_human_failure("mmm99999", "4.2", "error_cascade"))
        _write_jsonl(records, ann_file)

        report = validate_annotations_jsonl(str(ann_file))
        assert any("IMBALANCE" in w for w in report.warnings)

    def test_warns_on_low_sample_size(self, tmp_path):
        ann_file = tmp_path / "anns.jsonl"
        _write_jsonl([
            _make_human_failure("nnn00001", "4.2", "error_cascade"),
        ], ann_file)

        report = validate_annotations_jsonl(str(ann_file))
        assert any("SAMPLE SIZE" in w for w in report.warnings)

    def test_warns_on_mixed_ids(self, tmp_path):
        ann_file = tmp_path / "anns.jsonl"
        _write_jsonl([
            _make_human_clean("ooo00001"),
            _make_detector("ooo00001-full-uuid-here", "stall", "4.4"),
        ], ann_file)

        report = validate_annotations_jsonl(str(ann_file))
        assert any("MIXED" in w for w in report.warnings)

    def test_human_detector_conflict_warning(self, tmp_path):
        ann_file = tmp_path / "anns.jsonl"
        _write_jsonl([
            _make_human_clean("ppp00001"),
            _make_detector("ppp00001-full-uuid", "stall", "4.4"),
        ], ann_file)

        report = validate_annotations_jsonl(str(ann_file))
        assert any("CONFLICT" in w for w in report.warnings)


# ── Test: Dedup within sessions ──────────────────────────────────────

class TestDedup:
    def test_same_failure_deduped(self, tmp_path):
        """Multiple annotations for same failure on same session = one GT entry."""
        ann_file = tmp_path / "anns.jsonl"
        _write_jsonl([
            _make_human_failure("qqq00001", "4.4", "stall"),
            _make_human_failure("qqq00001", "4.4", "stall"),
        ], ann_file)

        annotations = load_annotations_from_jsonl(ann_file)
        assert len(annotations) == 1

    def test_different_failures_both_kept(self, tmp_path):
        """Different failures on same session = both in GT."""
        ann_file = tmp_path / "anns.jsonl"
        _write_jsonl([
            _make_human_failure("rrr00001", "4.4", "stall"),
            _make_human_failure("rrr00001", "4.2", "error_cascade"),
        ], ann_file)

        annotations = load_annotations_from_jsonl(ann_file)
        assert len(annotations) == 2
        names = {a.failure_name for a in annotations}
        assert "stall" in names or "4.4" in names  # depends on derive logic
