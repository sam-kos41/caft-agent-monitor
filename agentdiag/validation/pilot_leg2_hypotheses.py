"""Leg 2 — W-H1/W-H2/W-H3 + locked §9 decision rule.

PREREG_LEG2_WORKLOAD.md §6-§9. Reuses the validated pilot's estimator
and CV machinery and the frozen-sample caches (IT + workload aligned
by row, verified). Nothing here is a judgement call: thresholds are
frozen; the §9 rule executes itself from the booleans. W-H3 (signal
beyond Leg-1 IT) is the leg-defining gate.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from agentdiag.validation.pilot_features import IT_FEATURES
from agentdiag.validation.pilot_workload import WORKLOAD_FEATURES
from agentdiag.validation.pilot_hypotheses import (
    _it_pipeline, _full_pipeline, _cv_mean_auc, SEED, N_PERM, N_BOOT,
)

H1_FLOOR = 0.55
H2_DELTA_MIN = 0.03
H3_DELTA_MIN = 0.02


def _load():
    F = json.loads(Path("/tmp/caft_pilot/features.json").read_text())
    W = json.loads(Path("/tmp/caft_pilot/workload.json").read_text())
    assert len(F) == len(W)
    y = np.array([1 if f["target"] else 0 for f in F])
    assert all(bool(f["target"]) == bool(w["target"]) for f, w in zip(F, W))
    IT = np.array([[f["it"][k] for k in IT_FEATURES] for f in F], float)
    WL = np.array([[w["workload"][k] for k in WORKLOAD_FEATURES]
                   for w in W], float)
    num = np.array([[f["baseline"]["n_turns"],
                     f["baseline"]["n_parsed_actions"],
                     f["baseline"]["patch_len"]] for f in F], float)
    cat = np.array([[str(f["baseline"]["exit_status"]),
                     str(f["baseline"]["model_name"])] for f in F])
    return y, IT, WL, num, cat


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


def run(out_dir: str) -> dict:
    y, IT, WL, num, cat = _load()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # ---- W-H1: workload-only vs label-permutation null ----
    obs = _cv_mean_auc(_it_pipeline(), WL, y, skf)
    rng = np.random.default_rng(SEED)
    null = np.array([_cv_mean_auc(
        _it_pipeline(), WL, rng.permutation(y),
        StratifiedKFold(5, shuffle=True, random_state=SEED))
        for _ in range(N_PERM)])
    p95 = float(np.percentile(null, 95))
    wh1 = bool(obs > p95 and obs >= H1_FLOOR)

    Xb = np.hstack([num, cat])
    Xbw = np.hstack([np.hstack([num, WL]), cat])
    Xbi = np.hstack([np.hstack([num, IT]), cat])
    Xbiw = np.hstack([np.hstack([num, IT, WL]), cat])

    # ---- W-H2: baseline+workload vs baseline ----
    auc_b, p_b = _oof_auc(_full_pipeline(num.shape[1]), Xb, y, skf)
    auc_bw, p_bw = _oof_auc(_full_pipeline(num.shape[1] + WL.shape[1]),
                            Xbw, y, skf)
    d2 = auc_bw - auc_b
    lo2, hi2, nb2 = _delta_ci(y, p_b, p_bw, SEED)
    wh2 = bool(d2 >= H2_DELTA_MIN and lo2 > 0)

    # ---- W-H3 (leg-defining): +workload beyond baseline+IT ----
    auc_bi, p_bi = _oof_auc(_full_pipeline(num.shape[1] + IT.shape[1]),
                            Xbi, y, skf)
    auc_biw, p_biw = _oof_auc(
        _full_pipeline(num.shape[1] + IT.shape[1] + WL.shape[1]),
        Xbiw, y, skf)
    d3 = auc_biw - auc_bi
    lo3, hi3, nb3 = _delta_ci(y, p_bi, p_biw, SEED + 1)
    wh3 = bool(d3 >= H3_DELTA_MIN and lo3 > 0)

    # descriptive discriminant evidence (no threshold)
    corr = np.corrcoef(np.hstack([WL, IT]), rowvar=False)
    wl_it = corr[:WL.shape[1], WL.shape[1]:]
    max_abs_r = float(np.nanmax(np.abs(wl_it)))

    decision = _decide(wh1, wh3)
    res = {
        "n": int(len(y)),
        "W_H1": {"obs_auc": obs, "null_p95": p95,
                 "null_mean": float(null.mean()), "floor": H1_FLOOR,
                 "n_perm": N_PERM, "pass": wh1},
        "W_H2": {"auc_baseline": auc_b, "auc_baseline_workload": auc_bw,
                 "delta": d2, "ci": [lo2, hi2], "n_boot": nb2,
                 "min": H2_DELTA_MIN, "pass": wh2},
        "W_H3": {"auc_baseline_IT": auc_bi,
                 "auc_baseline_IT_workload": auc_biw, "delta": d3,
                 "ci": [lo3, hi3], "n_boot": nb3, "min": H3_DELTA_MIN,
                 "pass": wh3},
        "discriminant_max_abs_corr_workload_vs_IT": max_abs_r,
        "decision": decision,
    }
    o = Path(out_dir)
    o.mkdir(parents=True, exist_ok=True)
    (o / "leg2_hypotheses.json").write_text(json.dumps(res, indent=2))
    (o / "leg2_hypotheses.md").write_text(_md(res))
    print(_md(res))
    return res


def _decide(wh1: bool, wh3: bool) -> dict:
    """LOCKED §9 (audit already passed). W-H3 is leg-defining."""
    if not wh1:
        return {"code": "W_H1_FAIL",
                "conclusion": "Workload carries no outcome signal.",
                "action": "Document negative leg; proceed to Leg 3."}
    if not wh3:
        return {"code": "W_H1_PASS_W_H3_FAIL",
                "conclusion": "Workload is redundant with Leg-1 IT — "
                              "not a distinct construct.",
                "action": "Fold into Leg 1; proceed to Leg 3."}
    return {"code": "W_H1_PASS_W_H3_PASS",
            "conclusion": "Workload is a distinct, validated leg.",
            "action": "Leg 2 validated; proceed to Leg 3."}


def _md(r: dict) -> str:
    h1, h2, h3, d = r["W_H1"], r["W_H2"], r["W_H3"], r["decision"]
    return "\n".join([
        "# Leg 2 — W-H1/H2/H3 (locked PREREG_LEG2_WORKLOAD §6-§9)", "",
        f"n={r['n']}", "",
        "## W-H1 — workload-only vs label-perm null",
        f"- obs CV AUC **{h1['obs_auc']:.4f}** vs null p95 "
        f"{h1['null_p95']:.4f} (null mean {h1['null_mean']:.4f}, "
        f"{h1['n_perm']} perms), floor {h1['floor']}",
        f"- **W-H1 {'PASS' if h1['pass'] else 'FAIL'}**", "",
        "## W-H2 — beyond trivial baseline (controls volume)",
        f"- AUC base {h2['auc_baseline']:.4f} -> base+workload "
        f"{h2['auc_baseline_workload']:.4f}; ΔAUC **{h2['delta']:+.4f}** "
        f"CI [{h2['ci'][0]:+.4f},{h2['ci'][1]:+.4f}] ({h2['n_boot']} boot)",
        f"- **W-H2 {'PASS' if h2['pass'] else 'FAIL'}** "
        f"(needs ≥{h2['min']} & CI>0)", "",
        "## W-H3 — LEG-DEFINING: beyond baseline+Leg-1 IT",
        f"- AUC base+IT {h3['auc_baseline_IT']:.4f} -> +workload "
        f"{h3['auc_baseline_IT_workload']:.4f}; ΔAUC **{h3['delta']:+.4f}** "
        f"CI [{h3['ci'][0]:+.4f},{h3['ci'][1]:+.4f}] ({h3['n_boot']} boot)",
        f"- **W-H3 {'PASS' if h3['pass'] else 'FAIL'}** "
        f"(needs ≥{h3['min']} & CI>0)", "",
        f"- descriptive: max |corr| any workload vs any IT feature = "
        f"{r['discriminant_max_abs_corr_workload_vs_IT']:.3f}", "",
        f"## DECISION (locked §9): {d['code']}", "",
        f"**{d['conclusion']}**  Action: {d['action']}", "",
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
