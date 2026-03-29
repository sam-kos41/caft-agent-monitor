"""Tests for the feedback loop (case ledger, FP rates, confidence adjustment).

These tests exercise the case ledger logic without requiring OpenViking.
They test the JSONL persistence, status updates, FP rate computation,
and confidence adjustment that drive the feedback loop.
"""

import json
import tempfile
from pathlib import Path

import pytest

from agentdiag.caft.base import CaftDiagnosis, CaftSeverity


# ── Case ledger helpers (extracted from ContextStore for testability) ──

def _write_cases(ledger_path: Path, cases: list[dict]) -> None:
    """Write cases to a JSONL ledger file."""
    with open(ledger_path, "w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, default=str) + "\n")


def _load_cases(ledger_path: Path, status_filter: str | None = None) -> list[dict]:
    """Load cases from a JSONL ledger file."""
    cases = []
    if not ledger_path.exists():
        return cases
    with open(ledger_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            if status_filter and case.get("status") != status_filter:
                continue
            cases.append(case)
    return cases


def _update_case_status(
    ledger_path: Path, case_id: str, new_status: str,
    reviewer: str = "human", notes: str = "",
) -> bool:
    """Update a case's status in the ledger."""
    if not ledger_path.exists():
        return False
    lines = []
    found = False
    with open(ledger_path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                lines.append(line)
                continue
            case = json.loads(stripped)
            if case.get("case_id") == case_id:
                case["status"] = new_status
                case["reviewer"] = reviewer
                case["resolution_notes"] = notes
                found = True
            lines.append(json.dumps(case, default=str) + "\n")
    if found:
        with open(ledger_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
    return found


def _compute_fp_rates(cases: list[dict]) -> dict[str, float]:
    """Compute FP rates from reviewed cases."""
    reviewed: dict[str, int] = {}
    fp_count: dict[str, int] = {}
    for case in cases:
        status = case.get("status", "predicted")
        if status == "predicted":
            continue
        name = case.get("failure_name", "")
        if not name:
            continue
        reviewed[name] = reviewed.get(name, 0) + 1
        if status == "false_positive":
            fp_count[name] = fp_count.get(name, 0) + 1
    return {name: fp_count.get(name, 0) / total for name, total in reviewed.items()}


def _adjust_confidence(diagnosis: CaftDiagnosis, fp_rates: dict[str, float]) -> float:
    """Apply FP-rate discount to diagnosis confidence."""
    fp_rate = fp_rates.get(diagnosis.failure_name, 0.0)
    if fp_rate > 0.0:
        return round(diagnosis.confidence * (1.0 - fp_rate * 0.5), 4)
    return diagnosis.confidence


# ── Test fixtures ──

def _make_case(case_id: str, failure_name: str, status: str = "predicted") -> dict:
    return {
        "case_id": case_id,
        "session_id": "sess_001",
        "failure_name": failure_name,
        "caft_code": "2.2",
        "severity": "warning",
        "confidence": 0.7,
        "status": status,
        "description": f"Test case {case_id}",
    }


def _make_diagnosis(failure_name: str, confidence: float = 0.8) -> CaftDiagnosis:
    return CaftDiagnosis(
        caft_code="2.2",
        caft_category="memory",
        failure_name=failure_name,
        severity=CaftSeverity.WARNING,
        confidence=confidence,
        description="test",
        evidence={},
        at_step=1,
        remediation="fix",
    )


# ── Tests ──

class TestCaseLedger:
    def test_write_and_load_cases(self, tmp_path):
        ledger = tmp_path / "cases.jsonl"
        cases = [
            _make_case("c1", "step_repetition"),
            _make_case("c2", "context_loss"),
        ]
        _write_cases(ledger, cases)
        loaded = _load_cases(ledger)
        assert len(loaded) == 2
        assert loaded[0]["case_id"] == "c1"
        assert loaded[1]["case_id"] == "c2"

    def test_load_empty_file(self, tmp_path):
        ledger = tmp_path / "cases.jsonl"
        ledger.touch()
        assert _load_cases(ledger) == []

    def test_load_nonexistent_file(self, tmp_path):
        ledger = tmp_path / "doesnotexist.jsonl"
        assert _load_cases(ledger) == []

    def test_status_filter(self, tmp_path):
        ledger = tmp_path / "cases.jsonl"
        cases = [
            _make_case("c1", "step_repetition", status="predicted"),
            _make_case("c2", "context_loss", status="confirmed"),
            _make_case("c3", "step_repetition", status="false_positive"),
        ]
        _write_cases(ledger, cases)
        fp = _load_cases(ledger, status_filter="false_positive")
        assert len(fp) == 1
        assert fp[0]["case_id"] == "c3"

    def test_update_case_status(self, tmp_path):
        ledger = tmp_path / "cases.jsonl"
        cases = [
            _make_case("c1", "step_repetition"),
            _make_case("c2", "context_loss"),
        ]
        _write_cases(ledger, cases)

        ok = _update_case_status(ledger, "c1", "false_positive", notes="FP: normal exploration")
        assert ok

        loaded = _load_cases(ledger)
        assert loaded[0]["status"] == "false_positive"
        assert loaded[0]["reviewer"] == "human"
        assert loaded[0]["resolution_notes"] == "FP: normal exploration"
        assert loaded[1]["status"] == "predicted"  # unchanged

    def test_update_nonexistent_case(self, tmp_path):
        ledger = tmp_path / "cases.jsonl"
        _write_cases(ledger, [_make_case("c1", "step_repetition")])
        ok = _update_case_status(ledger, "c_nope", "confirmed")
        assert not ok


class TestFPRates:
    def test_no_reviewed_cases(self):
        cases = [_make_case("c1", "step_repetition", status="predicted")]
        rates = _compute_fp_rates(cases)
        assert rates == {}

    def test_all_confirmed(self):
        cases = [
            _make_case("c1", "step_repetition", status="confirmed"),
            _make_case("c2", "step_repetition", status="confirmed"),
        ]
        rates = _compute_fp_rates(cases)
        assert rates["step_repetition"] == 0.0

    def test_all_false_positive(self):
        cases = [
            _make_case("c1", "step_repetition", status="false_positive"),
            _make_case("c2", "step_repetition", status="false_positive"),
        ]
        rates = _compute_fp_rates(cases)
        assert rates["step_repetition"] == 1.0

    def test_mixed_reviews(self):
        cases = [
            _make_case("c1", "step_repetition", status="confirmed"),
            _make_case("c2", "step_repetition", status="false_positive"),
            _make_case("c3", "step_repetition", status="confirmed"),
            _make_case("c4", "step_repetition", status="false_positive"),
        ]
        rates = _compute_fp_rates(cases)
        assert rates["step_repetition"] == 0.5

    def test_multiple_detectors(self):
        cases = [
            _make_case("c1", "step_repetition", status="confirmed"),
            _make_case("c2", "step_repetition", status="false_positive"),
            _make_case("c3", "context_loss", status="confirmed"),
            _make_case("c4", "context_loss", status="confirmed"),
            _make_case("c5", "context_loss", status="confirmed"),
        ]
        rates = _compute_fp_rates(cases)
        assert rates["step_repetition"] == 0.5
        assert rates["context_loss"] == 0.0

    def test_unreviewed_cases_ignored(self):
        cases = [
            _make_case("c1", "step_repetition", status="predicted"),
            _make_case("c2", "step_repetition", status="predicted"),
            _make_case("c3", "step_repetition", status="confirmed"),
        ]
        rates = _compute_fp_rates(cases)
        assert rates["step_repetition"] == 0.0  # 0 FP / 1 reviewed


class TestConfidenceAdjustment:
    def test_no_adjustment_when_no_fp(self):
        d = _make_diagnosis("step_repetition", confidence=0.8)
        adjusted = _adjust_confidence(d, {"step_repetition": 0.0})
        assert adjusted == 0.8

    def test_no_adjustment_when_detector_not_in_rates(self):
        d = _make_diagnosis("step_repetition", confidence=0.8)
        adjusted = _adjust_confidence(d, {})
        assert adjusted == 0.8

    def test_50_percent_fp_rate(self):
        d = _make_diagnosis("step_repetition", confidence=0.8)
        adjusted = _adjust_confidence(d, {"step_repetition": 0.5})
        # 0.8 * (1 - 0.5 * 0.5) = 0.8 * 0.75 = 0.6
        assert adjusted == 0.6

    def test_100_percent_fp_rate(self):
        d = _make_diagnosis("step_repetition", confidence=0.8)
        adjusted = _adjust_confidence(d, {"step_repetition": 1.0})
        # 0.8 * (1 - 1.0 * 0.5) = 0.8 * 0.5 = 0.4
        assert adjusted == 0.4

    def test_25_percent_fp_rate(self):
        d = _make_diagnosis("context_loss", confidence=0.6)
        adjusted = _adjust_confidence(d, {"context_loss": 0.25})
        # 0.6 * (1 - 0.25 * 0.5) = 0.6 * 0.875 = 0.525
        assert adjusted == 0.525

    def test_other_detector_unaffected(self):
        d = _make_diagnosis("context_loss", confidence=0.9)
        adjusted = _adjust_confidence(d, {"step_repetition": 1.0})
        assert adjusted == 0.9  # context_loss not in rates


class TestFeedbackEndToEnd:
    """End-to-end test: write cases → review → check FP rates → adjust confidence."""

    def test_full_feedback_cycle(self, tmp_path):
        ledger = tmp_path / "cases.jsonl"

        # 1. Promote cases
        cases = [
            _make_case("s1_5_step_repetition", "step_repetition"),
            _make_case("s1_8_step_repetition", "step_repetition"),
            _make_case("s2_3_context_loss", "context_loss"),
            _make_case("s3_7_step_repetition", "step_repetition"),
        ]
        _write_cases(ledger, cases)

        # 2. No reviewed cases yet → no FP rates
        all_cases = _load_cases(ledger)
        rates = _compute_fp_rates(all_cases)
        assert rates == {}

        # 3. Human reviews: 2/3 step_repetition are FP, context_loss is confirmed
        _update_case_status(ledger, "s1_5_step_repetition", "false_positive",
                            notes="Normal file exploration")
        _update_case_status(ledger, "s1_8_step_repetition", "false_positive",
                            notes="Post-continuation re-read")
        _update_case_status(ledger, "s2_3_context_loss", "confirmed")
        # s3_7_step_repetition stays "predicted" (unreviewed)

        # 4. Compute FP rates
        all_cases = _load_cases(ledger)
        rates = _compute_fp_rates(all_cases)
        assert rates["step_repetition"] == 1.0  # 2 FP / 2 reviewed (3rd unreviewed)
        assert rates["context_loss"] == 0.0      # 0 FP / 1 reviewed

        # 5. New diagnosis comes in → adjust confidence
        new_diag = _make_diagnosis("step_repetition", confidence=0.8)
        adjusted = _adjust_confidence(new_diag, rates)
        # FP rate = 1.0 → 0.8 * (1 - 1.0 * 0.5) = 0.4
        assert adjusted == 0.4

        # context_loss unaffected
        cl_diag = _make_diagnosis("context_loss", confidence=0.6)
        cl_adjusted = _adjust_confidence(cl_diag, rates)
        assert cl_adjusted == 0.6

        # 6. Now human reviews last step_repetition as confirmed
        _update_case_status(ledger, "s3_7_step_repetition", "confirmed")
        all_cases = _load_cases(ledger)
        rates = _compute_fp_rates(all_cases)
        # 2 FP / 3 reviewed = 0.667
        assert abs(rates["step_repetition"] - 2 / 3) < 0.001

        # Confidence less discounted now
        new_diag2 = _make_diagnosis("step_repetition", confidence=0.8)
        adjusted2 = _adjust_confidence(new_diag2, rates)
        # 0.8 * (1 - 0.667 * 0.5) ≈ 0.8 * 0.667 = 0.5333
        assert 0.53 < adjusted2 < 0.54
