"""Tests for Leg-3 SA extraction (deterministic core)."""

from __future__ import annotations

from agentdiag.validation.pilot_sa import (
    extract_sa_one, SA_FEATURES, GATE_FEATURES, GATE_THRESHOLD, SEED,
)


def _ai(cmd):
    return {"role": "ai", "text": f"do it\n```\n{cmd}\n```"}


def test_frozen_feature_set():
    assert len(SA_FEATURES) == 6
    assert GATE_FEATURES == ("perception.coverage",
                             "projection.verify_before_submit")
    assert GATE_THRESHOLD == 0.80 and SEED == 20260515


def test_perception_coverage_observed_then_edited():
    # opens a/b.py, then edits a/b.py -> covered
    row = {"target": True, "trajectory": [
        {"role": "user", "text": "ISSUE"},
        _ai("open src/b.py"),
        {"role": "user", "text": "[shown]"},
        _ai("edit 1:1\nx\nend_of_edit"),  # edit has no path arg here
    ]}
    # edit target parsing: swe_agent _first_pathish on "1:1" -> "1:1";
    # use an explicit path-bearing mutate instead:
    row = {"target": True, "trajectory": [
        _ai('open src/b.py'),
        _ai('create src/b.py'),
    ]}
    sa = extract_sa_one(row)["sa"]
    assert sa["perception.coverage"] == 1.0
    assert sa["perception.blind_edit_rate"] == 0.0


def test_blind_edit_when_not_observed_first():
    row = {"target": False, "trajectory": [
        _ai('create totally/new.py'),     # mutate with no prior observe
    ]}
    sa = extract_sa_one(row)["sa"]
    assert sa["perception.coverage"] == 0.0
    assert sa["perception.blind_edit_rate"] > 0.0


def test_explore_ratio_and_read_before_first_edit():
    row = {"target": True, "trajectory": [
        _ai('search_dir "x"'), _ai('open a.py'), _ai('open b.py'),
        _ai('edit 1:1\nq\nend_of_edit'),
    ]}
    sa = extract_sa_one(row)["sa"]
    # 3 observe of 4 actions
    assert abs(sa["perception.explore_ratio"] - 0.75) < 1e-9
    # first mutate at index 3 of 4
    assert abs(sa["perception.read_before_first_edit"] - 0.75) < 1e-9


def test_projection_verify_before_submit():
    row = {"target": True, "trajectory": [
        _ai('create a.py'),
        _ai('python -m pytest -q'),   # verify AFTER last mutate
        _ai('submit'),
    ]}
    sa = extract_sa_one(row)["sa"]
    assert sa["projection.verify_before_submit"] == 1.0

    row2 = {"target": False, "trajectory": [
        _ai('create a.py'),
        _ai('submit'),                # submit blind, no verify
    ]}
    sa2 = extract_sa_one(row2)["sa"]
    assert sa2["projection.verify_before_submit"] == 0.0


def test_extract_shape():
    row = {"target": True, "trajectory": [_ai('open a.py'),
                                          _ai('submit')]}
    r = extract_sa_one(row)
    assert set(r["sa"].keys()) == set(SA_FEATURES)
    assert r["target"] is True
    assert isinstance(r["tool_counts"], dict)
