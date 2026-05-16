"""Tests for the locked Leg-4 §7 decision rule + frozen constants."""

from __future__ import annotations

from agentdiag.validation.pilot_leg4_hypotheses import (
    _decide, H1_FLOOR, H2_DELTA_MIN, H3_DELTA_MIN,
)


def test_decision_rule_branches_locked():
    assert _decide(False, False)["code"] == "E_H1_FAIL"
    assert _decide(False, True)["code"] == "E_H1_FAIL"   # H1 gates first
    assert _decide(True, False)["code"] == "E_H1_PASS_E_H3_FAIL"
    assert _decide(True, True)["code"] == "E_H1_PASS_E_H3_PASS"
    assert "COMPLETES the parsimony" in _decide(True, False)["conclusion"]
    assert "RE-OPENS" in _decide(True, True)["conclusion"]
    # FINAL leg: every branch ends the empirical phase
    for a, b in [(False, False), (True, False), (True, True)]:
        assert "END empirical phase" in _decide(a, b)["action"]


def test_frozen_thresholds():
    assert H1_FLOOR == 0.55
    assert H2_DELTA_MIN == 0.03
    assert H3_DELTA_MIN == 0.02
