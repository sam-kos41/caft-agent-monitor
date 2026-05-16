"""Pilot H1/H2 + the locked decision rule (PILOT_PREREGISTRATION §6-§8).

Reuses the SAME cached feature matrix the symbolization audit built,
so every number is identical across phases. Nothing here is a
judgement call: the thresholds are frozen and the 4-outcome decision
rule executes itself from the booleans.

H1 (§6): L2 logistic regression on the 20 IT features only;
  statistic = mean test ROC AUC over StratifiedKFold(5, shuffle,
  random_state=20260515); null = 1000 label permutations re-running
  the same CV; PASS iff observed > null 95th pct AND observed >= 0.55.

H2 (§7): nested LR, baseline vs baseline+IT, same folds; ΔAUC on
  out-of-fold predictions; 95% CI by bootstrap over instances; PASS
  iff mean ΔAUC >= 0.03 AND 95% CI excludes 0.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from agentdiag.validation.pilot_features import IT_FEATURES

SEED = 20260515
N_PERM = 1000
H1_AUC_FLOOR = 0.55
H2_DELTA_MIN = 0.03
N_BOOT = 2000


def _matrices(rows: list[dict]):
    y = np.array([1 if r["target"] else 0 for r in rows])
    X_it = np.array([[r["it"][f] for f in IT_FEATURES] for r in rows],
                    dtype=float)
    num = np.array([[r["baseline"]["n_turns"],
                     r["baseline"]["n_parsed_actions"],
                     r["baseline"]["patch_len"]] for r in rows], dtype=float)
    exit_s = np.array([[str(r["baseline"]["exit_status"])] for r in rows])
    model = np.array([[str(r["baseline"]["model_name"])] for r in rows])
    cat = np.hstack([exit_s, model])
    return y, X_it, num, cat


def _lr():
    # sklearn LogisticRegression default penalty IS "l2" (pre-reg §6/§7);
    # passing it explicitly is deprecated in sklearn 1.8 — the default
    # is the identical L2 estimator, not a spec change.
    return LogisticRegression(max_iter=2000)


def _it_pipeline():
    return Pipeline([("sc", StandardScaler()), ("lr", _lr())])


def _full_pipeline(n_num: int):
    ohe = OneHotEncoder(handle_unknown="ignore")
    pre = ColumnTransformer([
        ("num", StandardScaler(), list(range(n_num))),
        ("cat", ohe, [n_num, n_num + 1]),
    ])
    return Pipeline([("pre", pre), ("lr", _lr())])


def _cv_mean_auc(pipe, X, y, skf) -> float:
    aucs = []
    for tr, te in skf.split(X, y):
        pipe.fit(X[tr], y[tr])
        p = pipe.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs))


def run(cache: str, out_dir: str) -> dict:
    rows = json.loads(Path(cache).read_text())
    y, X_it, num, cat = _matrices(rows)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # ---- H1: IT-only vs label-permutation null ----
    obs_auc = _cv_mean_auc(_it_pipeline(), X_it, y, skf)
    rng = np.random.default_rng(SEED)
    null = np.empty(N_PERM)
    for i in range(N_PERM):
        yp = rng.permutation(y)
        null[i] = _cv_mean_auc(_it_pipeline(), X_it, yp,
                               StratifiedKFold(5, shuffle=True,
                                               random_state=SEED))
    null_p95 = float(np.percentile(null, 95))
    h1_pass = bool(obs_auc > null_p95 and obs_auc >= H1_AUC_FLOOR)

    # ---- H2: nested baseline vs baseline+IT (out-of-fold) ----
    Xb = np.hstack([num, cat])
    Xbi_num = np.hstack([num, X_it])
    Xbi = np.hstack([Xbi_num, cat])
    nnum_b, nnum_bi = num.shape[1], Xbi_num.shape[1]
    oof_b = cross_val_predict(_full_pipeline(nnum_b), Xb, y, cv=skf,
                              method="predict_proba")[:, 1]
    oof_bi = cross_val_predict(_full_pipeline(nnum_bi), Xbi, y, cv=skf,
                               method="predict_proba")[:, 1]
    auc_b = float(roc_auc_score(y, oof_b))
    auc_bi = float(roc_auc_score(y, oof_bi))
    d_auc = auc_bi - auc_b
    brng = np.random.default_rng(SEED)
    n = len(y)
    boot = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = brng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            boot[i] = np.nan
            continue
        boot[i] = (roc_auc_score(y[idx], oof_bi[idx])
                   - roc_auc_score(y[idx], oof_b[idx]))
    boot = boot[~np.isnan(boot)]
    ci = (float(np.percentile(boot, 2.5)),
          float(np.percentile(boot, 97.5)))
    h2_pass = bool(d_auc >= H2_DELTA_MIN and ci[0] > 0)

    decision = _decide(h1_pass, h2_pass)
    result = {
        "n": int(n), "n_resolved": int(y.sum()),
        "H1": {"observed_cv_auc": obs_auc, "null_p95": null_p95,
               "null_mean": float(null.mean()), "n_perm": N_PERM,
               "floor": H1_AUC_FLOOR, "pass": h1_pass},
        "H2": {"auc_baseline": auc_b, "auc_baseline_plus_IT": auc_bi,
               "delta_auc": d_auc, "delta_min": H2_DELTA_MIN,
               "ci95": ci, "n_boot": int(len(boot)), "pass": h2_pass},
        "decision": decision,
    }
    o = Path(out_dir)
    o.mkdir(parents=True, exist_ok=True)
    (o / "hypotheses_result.json").write_text(json.dumps(result, indent=2))
    (o / "hypotheses_result.md").write_text(_md(result))
    print(_md(result))
    return result


def _decide(h1: bool, h2: bool) -> dict:
    """The LOCKED §8 decision rule (audit already passed). Executes
    itself from the booleans — no thresholds re-weighed here."""
    if not h1:
        return {"code": "H1_FAIL",
                "conclusion": "IT carries no population signal even on a "
                              "clean independent outcome.",
                "action": "Bank the methodology paper. Do NOT commit the "
                          "year-long program. Clean stop."}
    if h1 and not h2:
        return {"code": "H1_PASS_H2_FAIL",
                "conclusion": "IT separates above chance but adds nothing "
                              "beyond trivial features (length etc.).",
                "action": "Documented honest finding; default to banking; "
                          "forward case is weak."}
    return {"code": "H1_PASS_H2_PASS",
            "conclusion": "IT adds genuine incremental signal on a clean "
                          "independent outcome.",
            "action": "Commit to the fuller four-leg program; "
                      "operationalize the other legs."}


def _md(r: dict) -> str:
    h1, h2, d = r["H1"], r["H2"], r["decision"]
    return "\n".join([
        "# Pilot H1/H2 Result (locked pre-registration §6-§8)", "",
        f"n={r['n']} ({r['n_resolved']} resolved / "
        f"{r['n']-r['n_resolved']} unresolved)", "",
        "## H1 — IT-only vs label-permutation null", "",
        f"- observed CV AUC: **{h1['observed_cv_auc']:.4f}**",
        f"- null 95th pct: {h1['null_p95']:.4f} "
        f"(null mean {h1['null_mean']:.4f}, {h1['n_perm']} perms)",
        f"- floor: {h1['floor']}",
        f"- **H1 {'PASS' if h1['pass'] else 'FAIL'}** "
        f"(needs AUC > null p95 AND >= {h1['floor']})", "",
        "## H2 — incremental value over trivial baseline", "",
        f"- AUC baseline: {h2['auc_baseline']:.4f}",
        f"- AUC baseline + IT: {h2['auc_baseline_plus_IT']:.4f}",
        f"- ΔAUC: **{h2['delta_auc']:+.4f}**  "
        f"(95% CI [{h2['ci95'][0]:+.4f}, {h2['ci95'][1]:+.4f}], "
        f"{h2['n_boot']} boot)",
        f"- **H2 {'PASS' if h2['pass'] else 'FAIL'}** "
        f"(needs ΔAUC >= {h2['delta_min']} AND CI excludes 0)", "",
        f"## DECISION (locked §8, audit already passed): {d['code']}", "",
        f"**{d['conclusion']}**", "",
        f"Action: {d['action']}", "",
        "_Objective. The rule executed from the H1/H2 booleans; "
        "thresholds were not re-weighed after seeing the numbers._",
    ])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/caft_pilot/features.json")
    ap.add_argument("--out", default="/tmp/caft_pilot")
    a = ap.parse_args()
    run(a.cache, a.out)


if __name__ == "__main__":
    main()
