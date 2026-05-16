"""Leg 3 — SA-H1/SA-H2/SA-H3 + locked §9 decision rule.

PREREG_LEG3_SA.md §6-§9. Identical machinery to the validated pilot
and Leg 2; SA features swapped in. Frozen-sample caches (IT + SA)
aligned by row, verified (0 target mismatches). SA-H3 (signal beyond
Leg-1 IT) is leg-defining and decides the 4-leg-vs-IT-core
meta-question for the strongest distinctness candidate. Thresholds
frozen; §9 executes itself from the booleans.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from agentdiag.validation.pilot_features import IT_FEATURES
from agentdiag.validation.pilot_sa import SA_FEATURES
from agentdiag.validation.pilot_hypotheses import (
    _it_pipeline, _full_pipeline, _cv_mean_auc, SEED, N_PERM, N_BOOT,
)

H1_FLOOR = 0.55
H2_DELTA_MIN = 0.03
H3_DELTA_MIN = 0.02


def _load():
    F = json.loads(Path("/tmp/caft_pilot/features.json").read_text())
    S = json.loads(Path("/tmp/caft_pilot/sa.json").read_text())
    assert len(F) == len(S)
    assert all(bool(f["target"]) == bool(s["target"]) for f, s in zip(F, S))
    y = np.array([1 if f["target"] else 0 for f in F])
    IT = np.array([[f["it"][k] for k in IT_FEATURES] for f in F], float)
    SA = np.array([[s["sa"][k] for k in SA_FEATURES] for s in S], float)
    num = np.array([[f["baseline"]["n_turns"],
                     f["baseline"]["n_parsed_actions"],
                     f["baseline"]["patch_len"]] for f in F], float)
    cat = np.array([[str(f["baseline"]["exit_status"]),
                     str(f["baseline"]["model_name"])] for f in F])
    return y, IT, SA, num, cat


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
    """LOCKED §9 (audit already passed). SA-H3 is leg-defining."""
    if not h1:
        return {"code": "SA_H1_FAIL",
                "conclusion": "SA carries no outcome signal.",
                "action": "Document negative leg; proceed to Leg 4."}
    if not h3:
        return {"code": "SA_H1_PASS_SA_H3_FAIL",
                "conclusion": "SA redundant with Leg-1 IT — not distinct. "
                              "Strong evidence for the 'IT is the "
                              "load-bearing construct' thesis (SA was the "
                              "strongest distinctness candidate).",
                "action": "Documented major finding; proceed to Leg 4."}
    return {"code": "SA_H1_PASS_SA_H3_PASS",
            "conclusion": "SA is a distinct, validated leg. Evidence for "
                          "the multi-leg framework thesis.",
            "action": "Leg 3 validated; proceed to Leg 4."}


def run(out_dir: str) -> dict:
    y, IT, SA, num, cat = _load()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    obs = _cv_mean_auc(_it_pipeline(), SA, y, skf)
    rng = np.random.default_rng(SEED)
    null = np.array([_cv_mean_auc(
        _it_pipeline(), SA, rng.permutation(y),
        StratifiedKFold(5, shuffle=True, random_state=SEED))
        for _ in range(N_PERM)])
    p95 = float(np.percentile(null, 95))
    h1 = bool(obs > p95 and obs >= H1_FLOOR)

    Xb = np.hstack([num, cat])
    Xbs = np.hstack([np.hstack([num, SA]), cat])
    Xbi = np.hstack([np.hstack([num, IT]), cat])
    Xbis = np.hstack([np.hstack([num, IT, SA]), cat])

    auc_b, p_b = _oof_auc(_full_pipeline(num.shape[1]), Xb, y, skf)
    auc_bs, p_bs = _oof_auc(_full_pipeline(num.shape[1] + SA.shape[1]),
                            Xbs, y, skf)
    d2 = auc_bs - auc_b
    lo2, hi2, nb2 = _delta_ci(y, p_b, p_bs, SEED)
    h2 = bool(d2 >= H2_DELTA_MIN and lo2 > 0)

    auc_bi, p_bi = _oof_auc(_full_pipeline(num.shape[1] + IT.shape[1]),
                            Xbi, y, skf)
    auc_bis, p_bis = _oof_auc(
        _full_pipeline(num.shape[1] + IT.shape[1] + SA.shape[1]),
        Xbis, y, skf)
    d3 = auc_bis - auc_bi
    lo3, hi3, nb3 = _delta_ci(y, p_bi, p_bis, SEED + 1)
    h3 = bool(d3 >= H3_DELTA_MIN and lo3 > 0)

    corr = np.corrcoef(np.hstack([SA, IT]), rowvar=False)
    sa_it = corr[:SA.shape[1], SA.shape[1]:]
    max_abs_r = float(np.nanmax(np.abs(sa_it)))

    decision = _decide(h1, h3)
    res = {
        "n": int(len(y)),
        "SA_H1": {"obs_auc": obs, "null_p95": p95,
                  "null_mean": float(null.mean()), "floor": H1_FLOOR,
                  "n_perm": N_PERM, "pass": h1},
        "SA_H2": {"auc_baseline": auc_b, "auc_baseline_SA": auc_bs,
                  "delta": d2, "ci": [lo2, hi2], "n_boot": nb2,
                  "min": H2_DELTA_MIN, "pass": h2},
        "SA_H3": {"auc_baseline_IT": auc_bi,
                  "auc_baseline_IT_SA": auc_bis, "delta": d3,
                  "ci": [lo3, hi3], "n_boot": nb3, "min": H3_DELTA_MIN,
                  "pass": h3},
        "discriminant_max_abs_corr_SA_vs_IT": max_abs_r,
        "decision": decision,
    }
    o = Path(out_dir)
    o.mkdir(parents=True, exist_ok=True)
    (o / "leg3_hypotheses.json").write_text(json.dumps(res, indent=2))
    (o / "leg3_hypotheses.md").write_text(_md(res))
    print(_md(res))
    return res


def _md(r: dict) -> str:
    h1, h2, h3, d = r["SA_H1"], r["SA_H2"], r["SA_H3"], r["decision"]
    return "\n".join([
        "# Leg 3 — SA-H1/H2/H3 (locked PREREG_LEG3_SA §6-§9)", "",
        f"n={r['n']}", "",
        "## SA-H1 — SA-only vs label-perm null",
        f"- obs CV AUC **{h1['obs_auc']:.4f}** vs null p95 "
        f"{h1['null_p95']:.4f} (null mean {h1['null_mean']:.4f}, "
        f"{h1['n_perm']} perms), floor {h1['floor']}",
        f"- **SA-H1 {'PASS' if h1['pass'] else 'FAIL'}**", "",
        "## SA-H2 — beyond trivial baseline",
        f"- AUC base {h2['auc_baseline']:.4f} -> base+SA "
        f"{h2['auc_baseline_SA']:.4f}; ΔAUC **{h2['delta']:+.4f}** "
        f"CI [{h2['ci'][0]:+.4f},{h2['ci'][1]:+.4f}] ({h2['n_boot']} boot)",
        f"- **SA-H2 {'PASS' if h2['pass'] else 'FAIL'}** "
        f"(needs ≥{h2['min']} & CI>0)", "",
        "## SA-H3 — LEG-DEFINING: beyond baseline+Leg-1 IT",
        f"- AUC base+IT {h3['auc_baseline_IT']:.4f} -> +SA "
        f"{h3['auc_baseline_IT_SA']:.4f}; ΔAUC **{h3['delta']:+.4f}** "
        f"CI [{h3['ci'][0]:+.4f},{h3['ci'][1]:+.4f}] ({h3['n_boot']} boot)",
        f"- **SA-H3 {'PASS' if h3['pass'] else 'FAIL'}** "
        f"(needs ≥{h3['min']} & CI>0)", "",
        f"- descriptive: max |corr| any SA vs any IT feature = "
        f"{r['discriminant_max_abs_corr_SA_vs_IT']:.3f}", "",
        f"## DECISION (locked §9): {d['code']}", "",
        f"**{d['conclusion']}**", "", f"Action: {d['action']}", "",
        "_Objective; rule executed from booleans; thresholds not "
        "re-weighed after seeing numbers._",
    ])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/caft_pilot")
    a = ap.parse_args()
    run(a.out)


if __name__ == "__main__":
    main()
