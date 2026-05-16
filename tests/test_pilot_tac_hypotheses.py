"""Tests for the locked agent-native TAC §8 decision rule."""

from __future__ import annotations

from agentdiag.validation.pilot_tac_hypotheses import (
    _decide, H1_FLOOR, H2_DELTA_MIN, H3_DELTA_MIN,
)


def test_decision_rule_branches_locked():
    assert _decide(False, False)["code"] == "TAC_H1_FAIL"
    assert _decide(False, True)["code"] == "TAC_H1_FAIL"   # H1 gates first
    assert _decide(True, False)["code"] == "TAC_H1_PASS_TAC_H3_FAIL"
    assert _decide(True, True)["code"] == "TAC_H1_PASS_TAC_H3_PASS"
    assert "DEEPENS the parsimony" in _decide(True, False)["conclusion"]
    assert "FIRST to survive" in _decide(True, True)["conclusion"]


def test_frozen_thresholds():
    assert H1_FLOOR == 0.55
    assert H2_DELTA_MIN == 0.03
    assert H3_DELTA_MIN == 0.02
