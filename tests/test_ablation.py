"""Tests for the ablation study runner (scripts/run_ablation.py).

All tests use synthetic data — no real traces or LLM calls required.
"""

import json
import sys
import tempfile
from pathlib import Path

import pytest

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from run_ablation import (
    LLMCache,
    LLMCacheEntry,
    apply_oracle_filter,
    diagnoses_to_detections,
    load_annotations,
    load_annotations_from_gt,
    load_annotations_from_jsonl,
)

from agentdiag.caft.base import CaftDiagnosis, CaftSeverity
from agentdiag.metrics import Annotation, Detection


# ── Ground truth loading ────────────────────────────────────────────

class TestLoadAnnotationsGT:
    def test_load_basic(self, tmp_path):
        gt = {
            "traces": [
                {
                    "session_id": "session_1",
                    "actual_failures": ["premature_termination"],
                    "failure_details": [
                        {
                            "failure_name": "premature_termination",
                            "caft_code": "5.4",
                            "onset_step": 25,
                        }
                    ],
                },
                {
                    "session_id": "session_2",
                    "actual_failures": [],
                },
            ]
        }
        p = tmp_path / "gt.json"
        p.write_text(json.dumps(gt))

        anns = load_annotations_from_gt(p)
        assert len(anns) == 1
        assert anns[0].trace_id == "session_1"
        assert anns[0].failure_name == "premature_termination"
        # onset_step forced to 0: JSON onset_step is a JSONL line number,
        # not a TraceEvent index, so it's incomparable with detector steps.
        assert anns[0].onset_step == 0

    def test_fallback_without_details(self, tmp_path):
        """When failure_details is absent, use actual_failures names."""
        gt = {
            "traces": [
                {
                    "session_id": "s1",
                    "actual_failures": ["context_loss"],
                }
            ]
        }
        p = tmp_path / "gt.json"
        p.write_text(json.dumps(gt))

        anns = load_annotations_from_gt(p)
        assert len(anns) == 1
        assert anns[0].failure_name == "context_loss"
        assert anns[0].onset_step == 0  # unknown step

    def test_clean_trace_no_annotations(self, tmp_path):
        gt = {"traces": [{"session_id": "clean", "actual_failures": []}]}
        p = tmp_path / "gt.json"
        p.write_text(json.dumps(gt))

        anns = load_annotations_from_gt(p)
        assert len(anns) == 0


class TestLoadAnnotationsJSONL:
    def test_load_basic(self, tmp_path):
        lines = [
            json.dumps({
                "session_id": "s1000001",
                "has_failure": True,
                "primary_caft_subtype": "step_repetition",
                "primary_caft_code": "2.2",
                "onset_step": 10,
                "annotator_type": "human",
                "label_status": "human_reviewed",
            }),
            json.dumps({
                "session_id": "s2000002",
                "has_failure": False,
                "annotator_type": "human",
                "label_status": "human_reviewed",
            }),
        ]
        p = tmp_path / "anns.jsonl"
        p.write_text("\n".join(lines))

        anns = load_annotations_from_jsonl(p)
        assert len(anns) == 1
        assert anns[0].failure_name == "step_repetition"

    def test_latent_annotation(self, tmp_path):
        lines = [json.dumps({
            "session_id": "s1000001",
            "has_failure": True,
            "primary_caft_subtype": "reasoning_loop",
            "primary_caft_code": "6.1",
            "onset_step": 5,
            "observable_vs_latent": "latent",
            "annotator_type": "human",
            "label_status": "human_reviewed",
        })]
        p = tmp_path / "anns.jsonl"
        p.write_text("\n".join(lines))

        anns = load_annotations_from_jsonl(p)
        assert len(anns) == 1
        assert anns[0].is_latent is True


class TestLoadAnnotationsAutoDetect:
    def test_json_extension(self, tmp_path):
        gt = {"traces": [{"session_id": "s1", "actual_failures": ["x"], "failure_details": [{"failure_name": "x", "caft_code": "1.1", "onset_step": 1}]}]}
        p = tmp_path / "gt.json"
        p.write_text(json.dumps(gt))
        anns = load_annotations(p)
        assert len(anns) == 1

    def test_jsonl_extension(self, tmp_path):
        p = tmp_path / "anns.jsonl"
        p.write_text(json.dumps({
            "session_id": "s1000001",
            "has_failure": True,
            "primary_caft_subtype": "x",
            "primary_caft_code": "1.1",
            "annotator_type": "human",
            "label_status": "human_reviewed",
        }))
        anns = load_annotations(p)
        assert len(anns) == 1


# ── LLM Cache ──────────────────────────────────────────────────────

class TestLLMCache:
    def test_put_and_get(self, tmp_path):
        cache = LLMCache(tmp_path / "cache.jsonl")
        entry = LLMCacheEntry(
            trace_id="t1",
            detector="step_repetition",
            candidate_step=10,
            confirmed=True,
            confidence=0.85,
            reasoning="Real repetition",
            latency_ms=150.0,
            tokens=200,
        )
        cache.put(entry)
        result = cache.get("t1", "step_repetition", 10)
        assert result is not None
        assert result.confirmed is True
        assert result.confidence == 0.85

    def test_cache_miss(self, tmp_path):
        cache = LLMCache(tmp_path / "cache.jsonl")
        assert cache.get("missing", "x", 0) is None

    def test_persistence(self, tmp_path):
        path = tmp_path / "cache.jsonl"
        cache1 = LLMCache(path)
        cache1.put(LLMCacheEntry("t1", "x", 5, True, 0.9, "ok", 100.0, 50))

        # Load from same file
        cache2 = LLMCache(path)
        assert cache2.get("t1", "x", 5) is not None
        assert len(cache2) == 1

    def test_len(self, tmp_path):
        cache = LLMCache(tmp_path / "cache.jsonl")
        assert len(cache) == 0
        cache.put(LLMCacheEntry("t1", "x", 5, True, 0.9, "ok", 100.0, 50))
        assert len(cache) == 1


# ── Oracle filter ───────────────────────────────────────────────────

class TestApplyOracleFilter:
    def test_keeps_matching_detections(self):
        dets = [
            Detection(trace_id="t1", failure_name="x", caft_code="1.1", onset_step=10, confidence=0.8),
        ]
        anns = [Annotation(trace_id="t1", failure_name="x", caft_code="1.1", onset_step=12)]
        filtered = apply_oracle_filter(dets, anns, match_window=5)
        assert len(filtered) == 1
        assert filtered[0].confirmed is True

    def test_removes_fp_detections(self):
        dets = [
            Detection(trace_id="t1", failure_name="x", caft_code="1.1", onset_step=10, confidence=0.8),
            Detection(trace_id="t1", failure_name="y", caft_code="2.1", onset_step=20, confidence=0.6),
        ]
        anns = [Annotation(trace_id="t1", failure_name="x", caft_code="1.1", onset_step=12)]
        filtered = apply_oracle_filter(dets, anns, match_window=5)
        assert len(filtered) == 1
        assert filtered[0].failure_name == "x"

    def test_empty_annotations(self):
        dets = [Detection(trace_id="t1", failure_name="x", caft_code="1.1", onset_step=10, confidence=0.8)]
        filtered = apply_oracle_filter(dets, [], match_window=5)
        assert len(filtered) == 0

    def test_oracle_ignores_step_window(self):
        """Oracle matches on (trace_id, failure_name) only — no step window.

        The oracle is the ceiling metric: it shows what a perfect LLM
        confirmation layer could achieve.  Step-window evaluation is
        handled separately by compute_evaluation / match_detections.
        """
        dets = [
            Detection(trace_id="t1", failure_name="x", caft_code="1.1", onset_step=10, confidence=0.8),
        ]
        anns = [Annotation(trace_id="t1", failure_name="x", caft_code="1.1", onset_step=50)]
        filtered = apply_oracle_filter(dets, anns, match_window=5)
        assert len(filtered) == 1
        assert filtered[0].confirmed is True

    def test_zero_onset_matches_any_step(self):
        """Annotation with onset_step=0 matches any detection step."""
        dets = [
            Detection(trace_id="t1", failure_name="x", caft_code="1.1", onset_step=100, confidence=0.8),
        ]
        anns = [Annotation(trace_id="t1", failure_name="x", caft_code="1.1", onset_step=0)]
        filtered = apply_oracle_filter(dets, anns, match_window=5)
        assert len(filtered) == 1


# ── diagnoses_to_detections ─────────────────────────────────────────

class TestDiagnosesToDetections:
    def test_basic_conversion(self):
        diag = CaftDiagnosis(
            caft_code="2.2",
            caft_category="memory",
            failure_name="step_repetition",
            severity=CaftSeverity.CRITICAL,
            confidence=0.85,
            description="test",
            evidence={},
            at_step=10,
            remediation="test",
        )
        results = {"session_1": [(diag, 50.0)]}
        dets = diagnoses_to_detections(results)
        assert len(dets) == 1
        assert dets[0].trace_id == "session_1"
        assert dets[0].failure_name == "step_repetition"
        assert dets[0].confidence == 0.85
        assert dets[0].latency_ms == 50.0

    def test_empty_results(self):
        dets = diagnoses_to_detections({})
        assert dets == []

    def test_multiple_sessions(self):
        diag1 = CaftDiagnosis(
            caft_code="2.2", caft_category="memory",
            failure_name="step_repetition", severity=CaftSeverity.WARNING,
            confidence=0.7, description="", evidence={}, at_step=5, remediation="",
        )
        diag2 = CaftDiagnosis(
            caft_code="5.4", caft_category="plan",
            failure_name="premature_termination", severity=CaftSeverity.CRITICAL,
            confidence=0.9, description="", evidence={}, at_step=25, remediation="",
        )
        results = {
            "s1": [(diag1, 30.0)],
            "s2": [(diag2, 40.0)],
        }
        dets = diagnoses_to_detections(results)
        assert len(dets) == 2
        names = {d.failure_name for d in dets}
        assert names == {"step_repetition", "premature_termination"}
