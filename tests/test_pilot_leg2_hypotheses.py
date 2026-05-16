"""Tests for the locked Leg-2 §9 decision rule + frozen constants."""

from __future__ import annotations

from agentdiag.validation.pilot_leg2_hypotheses import (
    _decide, H1_FLOOR, H2_DELTA_MIN, H3_DELTA_MIN,
)


def test_decision_rule_branches_locked():
    assert _decide(False, False)["code"] == "W_H1_FAIL"
    assert _decide(False, True)["code"] == "W_H1_FAIL"      # W-H1 gates first
    assert _decide(True, False)["code"] == "W_H1_PASS_W_H3_FAIL"
    assert _decide(True, True)["code"] == "W_H1_PASS_W_H3_PASS"
    assert "redundant with Leg-1 IT" in _decide(True, False)["conclusion"]
    assert "distinct, validated" in _decide(True, True)["conclusion"]
    # every branch proceeds to Leg 3 (program sequencing)
    for a, b in [(False, False), (True, False), (True, True)]:
        assert "Leg 3" in _decide(a, b)["action"]


def test_frozen_thresholds():
    assert H1_FLOOR == 0.55
    assert H2_DELTA_MIN == 0.03
    assert H3_DELTA_MIN == 0.02      # leg-defining, pre-stated lower
