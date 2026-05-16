"""Tests for the pilot feature extractor + audit pure pieces.

The real audit runs on the spec sample; here we pin the deterministic
machinery (aggregation, design matrix, frozen constants).
"""

from __future__ import annotations

from agentdiag.validation.pilot_features import (
    _aggregate, extract_one, IT_FEATURES, IT_BASE,
)
from agentdiag.validation.pilot_audit import (
    _design_matrix, SEED, GATE_FEATURES, GATE_THRESHOLD,
)


def test_aggregate_math():
    a = _aggregate([1.0, 2.0, 3.0, 4.0])
    assert a["mean"] == 2.5
    assert a["final"] == 4.0
    assert a["max"] == 4.0
    assert round(a["slope"], 6) == 1.0          # perfectly linear
    z = _aggregate([])
    assert z == {"mean": 0.0, "final": 0.0, "max": 0.0, "slope": 0.0}
    assert _aggregate([5.0])["slope"] == 0.0     # single point -> 0 slope


def test_it_feature_set_is_frozen_20():
    assert len(IT_FEATURES) == 20
    assert len(IT_BASE) == 5
    assert "action_mi.mean" in IT_FEATURES
    assert "compression_ratio.mean" in IT_FEATURES


def test_extract_one_shape():
    row = {
        "target": True,
        "generated_patch": "x",
        "model_name": "swe-agent-llama-70b",
        "exit_status": "submitted",
        "trajectory": [
            {"role": "user", "text": "ISSUE"},
            {"role": "ai", "text": 'go\n```\nsearch_dir "q"\n```'},
            {"role": "user", "text": "found"},
            {"role": "ai", "text": "open\n```\nopen a/b.py\n```"},
        ],
    }
    r = extract_one(row)
    assert set(r["it"].keys()) == set(IT_FEATURES)
    assert r["target"] is True
    assert r["n_events"] == 2
    assert r["tool_counts"].get("search_dir") == 1
    assert r["tool_counts"].get("open") == 1


def test_design_matrix_columns_sorted_union():
    rows = [
        {"tool_counts": {"open": 2, "edit": 1}},
        {"tool_counts": {"search_dir": 3, "open": 1}},
    ]
    X, cols = _design_matrix(rows)
    assert cols == ["edit", "open", "search_dir"]
    assert X.shape == (2, 3)
    assert list(X[0]) == [1.0, 2.0, 0.0]      # edit,open,search_dir
    assert list(X[1]) == [0.0, 1.0, 3.0]


def test_frozen_audit_constants():
    assert SEED == 20260515
    assert GATE_FEATURES == ("action_mi.mean", "compression_ratio.mean")
    assert GATE_THRESHOLD == 0.80
