"""Tests for the locked decision rule + H1/H2 plumbing.

The decision rule is the load-bearing locked logic — all three
post-audit branches are pinned. A tiny end-to-end run on synthetic
separable/noise data confirms the stats wiring is sane.
"""

from __future__ import annotations

import json
import numpy as np

from agentdiag.validation.pilot_hypotheses import (
    _decide, SEED, H1_AUC_FLOOR, H2_DELTA_MIN, N_PERM, run,
)
from agentdiag.validation.pilot_features import IT_FEATURES


def test_decision_rule_all_branches_locked():
    assert _decide(False, False)["code"] == "H1_FAIL"
    assert _decide(False, True)["code"] == "H1_FAIL"       # H1 gates first
    assert _decide(True, False)["code"] == "H1_PASS_H2_FAIL"
    assert _decide(True, True)["code"] == "H1_PASS_H2_PASS"
    assert "Bank" in _decide(False, False)["action"]
    assert "Commit" in _decide(True, True)["action"]


def test_frozen_constants():
    assert SEED == 20260515
    assert H1_AUC_FLOOR == 0.55
    assert H2_DELTA_MIN == 0.03
    assert N_PERM == 1000


def _synth_rows(tmp_path, separable: bool, n=240, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        tgt = i % 2
        it = {}
        for f in IT_FEATURES:
            base = rng.normal(0, 1)
            if separable and f == "action_mi.mean":
                base += 3.0 * tgt          # strong real signal
            it[f] = float(base)
        rows.append({
            "it": it, "target": bool(tgt),
            "baseline": {"n_turns": int(rng.integers(5, 50)),
                         "n_parsed_actions": int(rng.integers(3, 30)),
                         "patch_len": int(rng.integers(0, 2000)),
                         "exit_status": "submitted",
                         "model_name": "swe-agent-llama-70b"},
        })
    p = tmp_path / "feat.json"
    p.write_text(json.dumps(rows))
    return str(p)


def test_endtoend_separable_triggers_h1_pass(tmp_path):
    cache = _synth_rows(tmp_path, separable=True, seed=1)
    r = run(cache, str(tmp_path))
    assert r["H1"]["observed_cv_auc"] > 0.55
    assert r["H1"]["pass"] is True


def test_endtoend_pure_noise_h1_fails(tmp_path):
    cache = _synth_rows(tmp_path, separable=False, seed=2)
    r = run(cache, str(tmp_path))
    # noise: AUC ~0.5, must not clear the null+floor
    assert r["H1"]["pass"] is False
    assert r["decision"]["code"] == "H1_FAIL"
