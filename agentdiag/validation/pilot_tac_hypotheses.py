"""Agent-native TAC — TAC-H1/H2/H3 + locked §8 decision rule.

PREREG_AGENT_NATIVE_TAC.md §5-§8, PRIMARY deterministic
operationalization only (decision-bearing). Identical machinery to the
validated pilot / Legs 2-4; TAC features swapped in. Frozen-sample
caches (IT + TAC) aligned by row, verified (0 target mismatches).
TAC-H3 is leg-defining: does an agent-native construct survive the
discriminant gate the 3 HF ports failed 3/3? Thresholds frozen; §8
executes itself from the booleans. Robustness 1/2 are NOT here (run
separately, non-gating, after this decision is recorded).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from agentdiag.validation.pilot_features import IT_FEATURES
from agentdiag.validation.pilot_tac import TAC_FEATURES
from agentdiag.validation.pilot_hypotheses import (
    _it_pipeline, _full_pipeline, _cv_mean_auc, SEED, N_PERM, N_BOOT,
)

H1_FLOOR = 0.55
H2_DELTA_MIN = 0.03
H3_DELTA_MIN = 0.02


def _load():
    F = json.loads(Path("/tmp/caft_pilot/features.json").read_text())
    T = json.loads(Path("/tmp/caft_pilot/tac.json").read_text())
    assert len(F) == len(T)
    assert all(bool(f["target"]) == bool(t["target"]) for f, t in zip(F, T))
    y = np.array([1 if f["target"] else 0 for f in F])
    IT = np.array([[f["it"][k] for k in IT_FEATURES] for f in F], float)
    TA = np.array([[t["tac"][k] for k in TAC_FEATURES] for t in T], float)
    num = np.array([[f["baseline"]["n_turns"],
                     f["baseline"]["n_parsed_actions"],
                     f["baseline"]["patch_len"]] for f in F], float)
    cat = np.array([[str(f["baseline"]["exit_status"]),
                     str(f["baseline"]["model_name"])] for f in F])
    return y, IT, TA, num, cat


def _oof_auc(pipe, X, y, skf):
    p = cross_val_predict(pipe, X, y, cv=skf, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p)), p


def _delta_ci(y, p_lo, p_hi, seed):
    rng = np.random.default_rng(seed)
    n = len(y)
    b = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            b[i] = np.nan
            continue
        b[i] = (roc_auc_score(y[idx], p_hi[idx])
                - roc_auc_score(y[idx], p_lo[idx]))
    b = b[~np.isnan(b)]
    return (float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)),
            int(len(b)))


def _decide(h1: bool, h3: bool) -> dict:
    """LOCKED §8 (audit passed). TAC-H3 leg-defining."""
    if not h1:
        return {"code": "TAC_H1_FAIL",
                "conclusion": "TAC carries no outcome signal.",
                "action": "Document negative; stop."}
    if not h3:
        return {"code": "TAC_H1_PASS_TAC_H3_FAIL",
                "conclusion": "TAC reduces to IT — even an agent-native "
                              "construct (on an axis that ignores the "
                              "action-sequence shape and uses the thought "
                              "channel) collapses. DEEPENS the parsimony / "
                              "IT-load-bearing thesis.",
                "action": "Document; stop."}
    return {"code": "TAC_H1_PASS_TAC_H3_PASS",
            "conclusion": "TAC is a DISTINCT, validated agent-native "
                          "construct — the FIRST to survive the "
                          "discriminant gate the 3 HF ports failed 3/3. "
                          "Evidence agent-native measurement is a viable "
                          "research program.",
            "action": "Document; this becomes the lead for any future "
                      "program."}


def run(out_dir: str) -> dict:
    y, IT, TA, num, cat = _load()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    obs = _cv_mean_auc(_it_pipeline(), TA, y, skf)
    rng = np.random.default_rng(SEED)
    null = np.array([_cv_mean_auc(
        _it_pipeline(), TA, rng.permutation(y),
        StratifiedKFold(5, shuffle=True, random_state=SEED))
        for _ in range(N_PERM)])
    p95 = float(np.percentile(null, 95))
    h1 = bool(obs > p95 and obs >= H1_FLOOR)

    Xb = np.hstack([num, cat])
    Xbt = np.hstack([np.hstack([num, TA]), cat])
    Xbi = np.hstack([np.hstack([num, IT]), cat])
    Xbit = np.hstack([np.hstack([num, IT, TA]), cat])

    auc_b, p_b = _oof_auc(_full_pipeline(num.shape[1]), Xb, y, skf)
    auc_bt, p_bt = _oof_auc(_full_pipeline(num.shape[1] + TA.shape[1]),
                            Xbt, y, skf)
    d2 = auc_bt - auc_b
    lo2, hi2, nb2 = _delta_ci(y, p_b, p_bt, SEED)
    h2 = bool(d2 >= H2_DELTA_MIN and lo2 > 0)

    auc_bi, p_bi = _oof_auc(_full_pipeline(num.shape[1] + IT.shape[1]),
                            Xbi, y, skf)
    auc_bit, p_bit = _oof_auc(
        _full_pipeline(num.shape[1] + IT.shape[1] + TA.shape[1]),
        Xbit, y, skf)
    d3 = auc_bit - auc_bi
    lo3, hi3, nb3 = _delta_ci(y, p_bi, p_bit, SEED + 1)
    h3 = bool(d3 >= H3_DELTA_MIN and lo3 > 0)

    corr = np.corrcoef(np.hstack([TA, IT]), rowvar=False)
    ta_it = corr[:TA.shape[1], TA.shape[1]:]
    max_abs_r = float(np.nanmax(np.abs(ta_it)))

    decision = _decide(h1, h3)
    res = {
        "n": int(len(y)),
        "TAC_H1": {"obs_auc": obs, "null_p95": p95,
                   "null_mean": float(null.mean()), "floor": H1_FLOOR,
                   "n_perm": N_PERM, "pass": h1},
        "TAC_H2": {"auc_baseline": auc_b, "auc_baseline_TAC": auc_bt,
                   "delta": d2, "ci": [lo2, hi2], "n_boot": nb2,
                   "min": H2_DELTA_MIN, "pass": h2},
        "TAC_H3": {"auc_baseline_IT": auc_bi,
                   "auc_baseline_IT_TAC": auc_bit, "delta": d3,
                   "ci": [lo3, hi3], "n_boot": nb3, "min": H3_DELTA_MIN,
                   "pass": h3},
        "discriminant_max_abs_corr_TAC_vs_IT": max_abs_r,
        "decision": decision,
    }
    o = Path(out_dir)
    o.mkdir(parents=True, exist_ok=True)
    (o / "tac_hypotheses.json").write_text(json.dumps(res, indent=2))
    (o / "tac_hypotheses.md").write_text(_md(res))
    print(_md(res))
    return res


def _md(r: dict) -> str:
    h1, h2, h3, d = r["TAC_H1"], r["TAC_H2"], r["TAC_H3"], r["decision"]
    return "\n".join([
        "# Agent-Native TAC — H1/H2/H3 (locked PREREG §5-§8)", "",
        f"n={r['n']}  (PRIMARY deterministic operationalization; "
        f"robustness 1/2 reported separately, non-gating)", "",
        "## TAC-H1 — TAC-only vs label-perm null",
        f"- obs CV AUC **{h1['obs_auc']:.4f}** vs null p95 "
        f"{h1['null_p95']:.4f} (null mean {h1['null_mean']:.4f}, "
        f"{h1['n_perm']} perms), floor {h1['floor']}",
        f"- **TAC-H1 {'PASS' if h1['pass'] else 'FAIL'}**", "",
        "## TAC-H2 — beyond trivial baseline",
        f"- AUC base {h2['auc_baseline']:.4f} -> base+TAC "
        f"{h2['auc_baseline_TAC']:.4f}; ΔAUC **{h2['delta']:+.4f}** "
        f"CI [{h2['ci'][0]:+.4f},{h2['ci'][1]:+.4f}] ({h2['n_boot']} boot)",
        f"- **TAC-H2 {'PASS' if h2['pass'] else 'FAIL'}** "
        f"(needs ≥{h2['min']} & CI>0)", "",
        "## TAC-H3 — LEG-DEFINING: beyond baseline+Leg-1 IT",
        f"- AUC base+IT {h3['auc_baseline_IT']:.4f} -> +TAC "
        f"{h3['auc_baseline_IT_TAC']:.4f}; ΔAUC **{h3['delta']:+.4f}** "
        f"CI [{h3['ci'][0]:+.4f},{h3['ci'][1]:+.4f}] ({h3['n_boot']} boot)",
        f"- **TAC-H3 {'PASS' if h3['pass'] else 'FAIL'}** "
        f"(needs ≥{h3['min']} & CI>0)", "",
        f"- descriptive: max |corr| any TAC vs any IT feature = "
        f"{r['discriminant_max_abs_corr_TAC_vs_IT']:.3f}", "",
        f"## DECISION (locked §8): {d['code']}", "",
        f"**{d['conclusion']}**", "", f"Action: {d['action']}", "",
        "_Objective; rule executed from booleans; thresholds not "
        "re-weighed. Robustness checks cannot alter this decision._",
    ])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/caft_pilot")
    a = ap.parse_args()
    run(a.out)


if __name__ == "__main__":
    main()
