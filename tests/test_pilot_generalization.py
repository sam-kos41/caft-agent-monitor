"""Tests for Generalization G — frozen parse rule, Spearman, §5 rule."""

from __future__ import annotations

import numpy as np

from agentdiag.validation.pilot_generalization import (
    parse_g, _sp, _decide, SEED, H1_FLOOR, H2_DELTA_MIN, H3_DELTA_MIN,
)


def test_frozen_constants():
    assert SEED == 20260515
    assert H1_FLOOR == 0.10
    assert H2_DELTA_MIN == 0.03
    assert H3_DELTA_MIN == 0.02


def test_parse_g_frozen_rule():
    assert parse_g("... 8 passed, 2 failed ...") == 8 / 10
    assert parse_g("3 passed") == 1.0
    assert parse_g("5 failed") == 0.0
    assert parse_g("2 passed, 1 failed, 1 error") == 2 / 4
    assert parse_g("no test summary here") is None
    assert parse_g("") is None
    assert parse_g(None) is None
    # total zero -> None
    assert parse_g("0 passed") is None


def test_spearman_numpy_matches_known():
    a = np.array([1, 2, 3, 4, 5], float)
    assert abs(_sp(a, a) - 1.0) < 1e-9
    assert abs(_sp(a, a[::-1]) + 1.0) < 1e-9
    assert _sp(a, np.ones(5)) == 0.0          # constant -> 0
    # monotone but nonlinear -> spearman 1.0
    assert abs(_sp(a, a ** 3) - 1.0) < 1e-9


def test_decision_rule_locked_branches():
    assert _decide(False, False, [])["code"] == "G_H1_FAIL"
    assert _decide(False, True, ["x"])["code"] == "G_H1_FAIL"  # H1 gates
    assert _decide(True, True, [])["code"] == "PARSIMONY_GENERALIZES"
    assert _decide(True, False, [])["code"] == "PARSIMONY_GENERALIZES"
    r = _decide(True, True, ["workload", "thought_action_coherence"])
    assert r["code"] == "RESEPARATION"
    assert "workload" in r["conclusion"]
    assert "REJECTED" in r["action"]
