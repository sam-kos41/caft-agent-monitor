"""Generalization Discriminant G — outcome granularity.

PREREG_GENERALIZATION_G.md (FROZEN, greenlit 2026-05-16). Tests
whether collapse-into-IT survives a GRADED outcome on the same frozen
sample. No new symbolization gate (feature sets unchanged, already
gated per leg). The locked §5 rule executes itself.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from agentdiag.validation.pilot_features import IT_FEATURES
from agentdiag.validation.pilot_workload import WORKLOAD_FEATURES
from agentdiag.validation.pilot_sa import SA_FEATURES
from agentdiag.validation.pilot_error import ERROR_FEATURES
from agentdiag.validation.pilot_tac import TAC_FEATURES

SEED = 20260515
N_PERM = 1000
N_BOOT = 2000
H1_FLOOR = 0.10
H2_DELTA_MIN = 0.03
H3_DELTA_MIN = 0.02

# FROZEN graded-outcome parse (pre-reg §1)
_P = re.compile(r"(\d+)\s+passed", re.I)
_F = re.compile(r"(\d+)\s+failed", re.I)
_E = re.compile(r"(\d+)\s+error", re.I)


def parse_g(eval_logs: str):
    """Return g in [0,1] or None if not deterministically parseable."""
    el = eval_logs or ""
    p, f, e = _P.search(el), _F.search(el), _E.search(el)
    if not (p or f):
        return None
    np_ = int(p.group(1)) if p else 0
    nf_ = int(f.group(1)) if f else 0
    ne_ = int(e.group(1)) if e else 0
    tot = np_ + nf_ + ne_
    if tot == 0:
        return None
    return np_ / tot


def _ridge_num():
    return Pipeline([("sc", StandardScaler()), ("r", Ridge(alpha=1.0))])


def _ridge_full(n_num: int):
    pre = ColumnTransformer([
        ("num", StandardScaler(), list(range(n_num))),
        ("cat", OneHotEncoder(handle_unknown="ignore"),
         [n_num, n_num + 1]),
    ])
    return Pipeline([("pre", pre), ("r", Ridge(alpha=1.0))])


def _oof(pipe, X, y, skf):
    return cross_val_predict(pipe, X, y, cv=skf)


def _sp(a, b):
    """Spearman rho, numpy-only (package is deliberately scipy-free,
    matching eval/stats.py). 0.0 if either side is constant."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0          # constant input -> undefined; return 0
    ar = np.argsort(np.argsort(a))
    br = np.argsort(np.argsort(b))
    return float(np.corrcoef(ar, br)[0, 1])


def _delta_ci(y, p_lo, p_hi, seed, pct=(2.5, 97.5)):
    rng = np.random.default_rng(seed)
    n = len(y)
    b = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        b[i] = _sp(p_hi[idx], y[idx]) - _sp(p_lo[idx], y[idx])
    return (float(np.percentile(b, pct[0])),
            float(np.percentile(b, pct[1])), b)


def run(out_dir: str) -> dict:
    base = Path("/tmp/caft_pilot")
    F = json.loads((base / "features.json").read_text())
    WL = json.loads((base / "workload.json").read_text())
    SA = json.loads((base / "sa.json").read_text())
    ER = json.loads((base / "error.json").read_text())
    TA = json.loads((base / "tac.json").read_text())
    sample = [json.loads(l) for l in open(base / "sample.jsonl")]
    n0 = len(sample)
    assert all(len(x) == n0 for x in (F, WL, SA, ER, TA)), "cache misalign"

    keep, g = [], []
    for i, row in enumerate(sample):
        gi = parse_g(row.get("eval_logs"))
        if gi is not None:
            keep.append(i)
            g.append(gi)
    g = np.array(g, float)
    K = np.array(keep)

    def mat(cache, feats, sub):
        return np.array([[cache[i][sub][k] for k in feats] for i in K],
                        float)

    IT = mat(F, IT_FEATURES, "it")
    W = mat(WL, WORKLOAD_FEATURES, "workload")
    S = mat(SA, SA_FEATURES, "sa")
    Er = mat(ER, ERROR_FEATURES, "error")
    T = mat(TA, TAC_FEATURES, "tac")
    num = np.array([[F[i]["baseline"]["n_turns"],
                     F[i]["baseline"]["n_parsed_actions"],
                     F[i]["baseline"]["patch_len"]] for i in K], float)
    cat = np.array([[str(F[i]["baseline"]["exit_status"]),
                     str(F[i]["baseline"]["model_name"])] for i in K])

    skf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    # G-H1: IT-only vs permuted-g null
    p_it = _oof(_ridge_num(), IT, g, skf)
    obs = _sp(p_it, g)
    rng = np.random.default_rng(SEED)
    null = np.array([_sp(_oof(_ridge_num(), IT, gp := rng.permutation(g),
                              KFold(5, shuffle=True, random_state=SEED)),
                         gp) for _ in range(N_PERM)])
    p95 = float(np.percentile(null, 95))
    gh1 = bool(obs > p95 and abs(obs) >= H1_FLOOR)

    Xb = np.hstack([num, cat])
    Xbi = np.hstack([np.hstack([num, IT]), cat])
    pb = _oof(_ridge_full(num.shape[1]), Xb, g, skf)
    pbi = _oof(_ridge_full(num.shape[1] + IT.shape[1]), Xbi, g, skf)
    sp_b, sp_bi = _sp(pb, g), _sp(pbi, g)
    d2 = sp_bi - sp_b
    lo2, hi2, _ = _delta_ci(g, pb, pbi, SEED)
    gh2 = bool(d2 >= H2_DELTA_MIN and lo2 > 0)

    # G-H3: each collapsed construct beyond baseline+IT
    constructs = {"workload": W, "situation_awareness": S,
                  "error_recovery": Er, "thought_action_coherence": T}
    h3 = {}
    any_resep = []
    for name, M in constructs.items():
        Xbic = np.hstack([np.hstack([num, IT, M]), cat])
        pbic = _oof(_ridge_full(num.shape[1] + IT.shape[1] + M.shape[1]),
                    Xbic, g, skf)
        d = _sp(pbic, g) - sp_bi
        lo, hi, _ = _delta_ci(g, pbi, pbic, SEED + 1)
        # Bonferroni alpha/4 -> 98.75% CI, reported not gating
        blo, bhi, _ = _delta_ci(g, pbi, pbic, SEED + 1,
                                pct=(0.625, 99.375))
        resep = bool(d >= H3_DELTA_MIN and lo > 0)
        h3[name] = {"delta_spearman": d, "ci95": [lo, hi],
                    "ci_bonf_9875": [blo, bhi],
                    "re_separates": resep,
                    "survives_bonferroni": bool(d >= H3_DELTA_MIN
                                                and blo > 0)}
        if resep:
            any_resep.append(name)

    decision = _decide(gh1, gh2, any_resep)
    res = {
        "n_analysis": int(len(K)), "n_total": n0,
        "g_mean": float(g.mean()), "g_frac_eq_1": float((g == 1).mean()),
        "G_H1": {"obs_spearman": obs, "null_p95": p95,
                 "null_mean": float(null.mean()), "floor": H1_FLOOR,
                 "pass": gh1},
        "G_H2": {"spearman_baseline": sp_b,
                 "spearman_baseline_IT": sp_bi, "delta": d2,
                 "ci95": [lo2, hi2], "min": H2_DELTA_MIN, "pass": gh2},
        "G_H3": h3,
        "constructs_that_reseparate": any_resep,
        "decision": decision,
    }
    o = Path(out_dir)
    o.mkdir(parents=True, exist_ok=True)
    (o / "generalization_G.json").write_text(json.dumps(res, indent=2))
    (o / "generalization_G.md").write_text(_md(res))
    print(_md(res))
    return res


def _decide(gh1, gh2, resep_list) -> dict:
    if not gh1:
        return {"code": "G_H1_FAIL",
                "conclusion": "IT does not predict the graded outcome; "
                              "the binary IT result may be "
                              "binarization-specific.",
                "action": "Major caveat on the program; document; "
                          "reassess before any build."}
    if resep_list:
        return {"code": "RESEPARATION",
                "conclusion": "Collapse was partly a binarization "
                              "artifact: " + ", ".join(resep_list)
                              + " carry distinct signal under "
                              "partial-credit grading.",
                "action": "Paper-defining reframe; 'agents are flat' "
                          "interpretation REJECTED; reassess."}
    return {"code": "PARSIMONY_GENERALIZES",
            "conclusion": "No construct re-separates under a graded "
                          "outcome. Parsimony generalizes across "
                          "outcome granularity (H_flat supported over "
                          "H_artifact, on this corpus/axis).",
            "action": "Strong corroboration; IT-load-bearing finding "
                      "stands (conditional on §1 limits)."}


def _md(r: dict) -> str:
    h1, h2, d = r["G_H1"], r["G_H2"], r["decision"]
    L = ["# Generalization G — outcome granularity (locked §5)", "",
         f"analysis n={r['n_analysis']}/{r['n_total']} "
         f"(parseable subset; §1 selection bias applies). "
         f"g mean={r['g_mean']:.3f}, frac(g==1)={r['g_frac_eq_1']:.0%}",
         "", "## G-H1 — IT predicts graded outcome (vs permuted-g null)",
         f"- OOF Spearman **{h1['obs_spearman']:.4f}** vs null p95 "
         f"{h1['null_p95']:.4f} (null mean {h1['null_mean']:.4f}), "
         f"floor |ρ|≥{h1['floor']}",
         f"- **G-H1 {'PASS' if h1['pass'] else 'FAIL'}**", "",
         "## G-H2 — IT beyond trivial baseline",
         f"- Spearman base {h2['spearman_baseline']:.4f} -> base+IT "
         f"{h2['spearman_baseline_IT']:.4f}; ΔSpearman "
         f"**{h2['delta']:+.4f}** CI [{h2['ci95'][0]:+.4f},"
         f"{h2['ci95'][1]:+.4f}]",
         f"- **G-H2 {'PASS' if h2['pass'] else 'FAIL'}** "
         f"(needs ≥{h2['min']} & CI>0)", "",
         "## G-H3 — does ANY collapsed construct re-separate (beyond "
         "baseline+IT)?", "",
         "| construct | ΔSpearman | 95% CI | re-separates? | Bonf α/4 |",
         "|---|---:|---|---|---|"]
    for nm, v in r["G_H3"].items():
        L.append(f"| {nm} | {v['delta_spearman']:+.4f} | "
                 f"[{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}] | "
                 f"{v['re_separates']} | "
                 f"{'survives' if v['survives_bonferroni'] else 'no'} |")
    L += ["", f"## DECISION (locked §5): {d['code']}", "",
          f"**{d['conclusion']}**", "", f"Action: {d['action']}", "",
          "_Conclusions conditional on §1 selection bias + single "
          "corpus/agent family. Outcome-granularity test only._"]
    return "\n".join(L)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/caft_pilot")
    a = ap.parse_args()
    run(a.out)


if __name__ == "__main__":
    main()
