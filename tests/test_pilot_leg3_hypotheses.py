"""Tests for the locked Leg-3 §9 decision rule + frozen constants."""

from __future__ import annotations

from agentdiag.validation.pilot_leg3_hypotheses import (
    _decide, H1_FLOOR, H2_DELTA_MIN, H3_DELTA_MIN,
)


def test_decision_rule_branches_locked():
    assert _decide(False, False)["code"] == "SA_H1_FAIL"
    assert _decide(False, True)["code"] == "SA_H1_FAIL"   # H1 gates first
    assert _decide(True, False)["code"] == "SA_H1_PASS_SA_H3_FAIL"
    assert _decide(True, True)["code"] == "SA_H1_PASS_SA_H3_PASS"
    assert "IT is the" in _decide(True, False)["conclusion"]
    assert "multi-leg framework" in _decide(True, True)["conclusion"]
    for a, b in [(False, False), (True, False), (True, True)]:
        assert "Leg 4" in _decide(a, b)["action"]


def test_frozen_thresholds():
    assert H1_FLOOR == 0.55
    assert H2_DELTA_MIN == 0.03
    assert H3_DELTA_MIN == 0.02
