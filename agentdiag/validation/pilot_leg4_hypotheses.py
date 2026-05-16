"""Leg 4 — E-H1/E-H2/E-H3 + locked §7 decision rule (FINAL leg).

PREREG_LEG4_ERROR_RECOVERY.md §6-§7. Identical machinery to the
validated pilot / Legs 2-3; ER features swapped in. Frozen-sample
caches (IT + ER) aligned by row, verified (0 target mismatches).
E-H3 (signal beyond Leg-1 IT) is leg-defining and is the decisive
test for the whole program's parsimony thesis. Thresholds frozen;
§7 executes itself from the booleans. Every branch ends the empirical
phase.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from agentdiag.validation.pilot_features import IT_FEATURES
from agentdiag.validation.pilot_error import ERROR_FEATURES
from agentdiag.validation.pilot_hypotheses import (
    _it_pipeline, _full_pipeline, _cv_mean_auc, SEED, N_PERM, N_BOOT,
)

H1_FLOOR = 0.55
H2_DELTA_MIN = 0.03
H3_DELTA_MIN = 0.02


def _load():
    F = json.loads(Path("/tmp/caft_pilot/features.json").read_text())
    E = json.loads(Path("/tmp/caft_pilot/error.json").read_text())
    assert len(F) == len(E)
    assert all(bool(f["target"]) == bool(e["target"]) for f, e in zip(F, E))
    y = np.array([1 if f["target"] else 0 for f in F])
    IT = np.array([[f["it"][k] for k in IT_FEATURES] for f in F], float)
    ER = np.array([[e["error"][k] for k in ERROR_FEATURES] for e in E],
                  float)
    num = np.array([[f["baseline"]["n_turns"],
                     f["baseline"]["n_parsed_actions"],
                     f["baseline"]["patch_len"]] for f in F], float)
    cat = np.array([[str(f["baseline"]["exit_status"]),
                     str(f["baseline"]["model_name"])] for f in F])
    return y, IT, ER, num, cat


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
    """LOCKED §7 (audit passed). E-H3 leg-defining. FINAL leg —
    every branch ends the empirical phase."""
    if not h1:
        return {"code": "E_H1_FAIL",
                "conclusion": "Error recovery carries no outcome signal.",
                "action": "Document negative leg; END empirical phase "
                          "-> methodology paper."}
    if not h3:
        return {"code": "E_H1_PASS_E_H3_FAIL",
                "conclusion": "Error recovery redundant with Leg-1 IT — "
                              "not distinct. COMPLETES the parsimony "
                              "result: all 3 tested HF analogs reduce to "
                              "IT (airtight; strongest-distinct axis too).",
                "action": "Document; END empirical phase -> paper."}
    return {"code": "E_H1_PASS_E_H3_PASS",
            "conclusion": "Error recovery is a DISTINCT, validated leg. "
                          "RE-OPENS the parsimony thesis — a genuine "
                          "second construct exists. Paper-defining the "
                          "other way.",
            "action": "Document (thesis reframed); END empirical phase "
                      "-> paper."}


def run(out_dir: str) -> dict:
    y, IT, ER, num, cat = _load()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    obs = _cv_mean_auc(_it_pipeline(), ER, y, skf)
    rng = np.random.default_rng(SEED)
    null = np.array([_cv_mean_auc(
        _it_pipeline(), ER, rng.permutation(y),
        StratifiedKFold(5, shuffle=True, random_state=SEED))
        for _ in range(N_PERM)])
    p95 = float(np.percentile(null, 95))
    h1 = bool(obs > p95 and obs >= H1_FLOOR)

    Xb = np.hstack([num, cat])
    Xbe = np.hstack([np.hstack([num, ER]), cat])
    Xbi = np.hstack([np.hstack([num, IT]), cat])
    Xbie = np.hstack([np.hstack([num, IT, ER]), cat])

    auc_b, p_b = _oof_auc(_full_pipeline(num.shape[1]), Xb, y, skf)
    auc_be, p_be = _oof_auc(_full_pipeline(num.shape[1] + ER.shape[1]),
                            Xbe, y, skf)
    d2 = auc_be - auc_b
    lo2, hi2, nb2 = _delta_ci(y, p_b, p_be, SEED)
    h2 = bool(d2 >= H2_DELTA_MIN and lo2 > 0)

    auc_bi, p_bi = _oof_auc(_full_pipeline(num.shape[1] + IT.shape[1]),
                            Xbi, y, skf)
    auc_bie, p_bie = _oof_auc(
        _full_pipeline(num.shape[1] + IT.shape[1] + ER.shape[1]),
        Xbie, y, skf)
    d3 = auc_bie - auc_bi
    lo3, hi3, nb3 = _delta_ci(y, p_bi, p_bie, SEED + 1)
    h3 = bool(d3 >= H3_DELTA_MIN and lo3 > 0)

    corr = np.corrcoef(np.hstack([ER, IT]), rowvar=False)
    er_it = corr[:ER.shape[1], ER.shape[1]:]
    max_abs_r = float(np.nanmax(np.abs(er_it)))

    decision = _decide(h1, h3)
    res = {
        "n": int(len(y)),
        "E_H1": {"obs_auc": obs, "null_p95": p95,
                 "null_mean": float(null.mean()), "floor": H1_FLOOR,
                 "n_perm": N_PERM, "pass": h1},
        "E_H2": {"auc_baseline": auc_b, "auc_baseline_ER": auc_be,
                 "delta": d2, "ci": [lo2, hi2], "n_boot": nb2,
                 "min": H2_DELTA_MIN, "pass": h2},
        "E_H3": {"auc_baseline_IT": auc_bi,
                 "auc_baseline_IT_ER": auc_bie, "delta": d3,
                 "ci": [lo3, hi3], "n_boot": nb3, "min": H3_DELTA_MIN,
                 "pass": h3},
        "discriminant_max_abs_corr_ER_vs_IT": max_abs_r,
        "decision": decision,
    }
    o = Path(out_dir)
    o.mkdir(parents=True, exist_ok=True)
    (o / "leg4_hypotheses.json").write_text(json.dumps(res, indent=2))
    (o / "leg4_hypotheses.md").write_text(_md(res))
    print(_md(res))
    return res


def _md(r: dict) -> str:
    h1, h2, h3, d = r["E_H1"], r["E_H2"], r["E_H3"], r["decision"]
    return "\n".join([
        "# Leg 4 — E-H1/H2/H3 (locked PREREG_LEG4 §6-§7) — FINAL LEG", "",
        f"n={r['n']}", "",
        "## E-H1 — ER-only vs label-perm null",
        f"- obs CV AUC **{h1['obs_auc']:.4f}** vs null p95 "
        f"{h1['null_p95']:.4f} (null mean {h1['null_mean']:.4f}, "
        f"{h1['n_perm']} perms), floor {h1['floor']}",
        f"- **E-H1 {'PASS' if h1['pass'] else 'FAIL'}**", "",
        "## E-H2 — beyond trivial baseline",
        f"- AUC base {h2['auc_baseline']:.4f} -> base+ER "
        f"{h2['auc_baseline_ER']:.4f}; ΔAUC **{h2['delta']:+.4f}** "
        f"CI [{h2['ci'][0]:+.4f},{h2['ci'][1]:+.4f}] ({h2['n_boot']} boot)",
        f"- **E-H2 {'PASS' if h2['pass'] else 'FAIL'}** "
        f"(needs ≥{h2['min']} & CI>0)", "",
        "## E-H3 — LEG-DEFINING / DECISIVE: beyond baseline+Leg-1 IT",
        f"- AUC base+IT {h3['auc_baseline_IT']:.4f} -> +ER "
        f"{h3['auc_baseline_IT_ER']:.4f}; ΔAUC **{h3['delta']:+.4f}** "
        f"CI [{h3['ci'][0]:+.4f},{h3['ci'][1]:+.4f}] ({h3['n_boot']} boot)",
        f"- **E-H3 {'PASS' if h3['pass'] else 'FAIL'}** "
        f"(needs ≥{h3['min']} & CI>0)", "",
        f"- descriptive: max |corr| any ER vs any IT feature = "
        f"{r['discriminant_max_abs_corr_ER_vs_IT']:.3f}", "",
        f"## DECISION (locked §7): {d['code']}", "",
        f"**{d['conclusion']}**", "", f"Action: {d['action']}", "",
        "_Objective; rule executed from booleans; thresholds not "
        "re-weighed. Empirical phase ends here regardless._",
    ])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/caft_pilot")
    a = ap.parse_args()
    run(a.out)


if __name__ == "__main__":
    main()
