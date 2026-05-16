"""Pilot symbolization-audit GATE (PILOT_PREREGISTRATION §5 + A2).

Question: can an IT feature be reconstructed from raw tool-API surface
counts alone? If yes, the IT signal is a tool-API artifact and the
pilot is INVALID (a methods finding, NOT a verdict about information
theory, and H1/H2 are not interpreted).

Frozen procedure (A2): design matrix = per-trajectory count of each
distinct tool_name; estimator = Ridge(alpha=1.0); R² = mean test R²
under KFold(5, shuffle, random_state=20260515). GATE: pilot INVALID if
CV R² >= 0.80 for `action_mi.mean` OR `compression_ratio.mean`.

This module computes and reports. The gate outcome is objective and
locked — it is not a judgement call and is not re-litigated.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score

from agentdiag.validation.pilot_features import build_matrix, IT_FEATURES

SEED = 20260515
GATE_FEATURES = ("action_mi.mean", "compression_ratio.mean")
GATE_THRESHOLD = 0.80


def _design_matrix(rows: list[dict]) -> tuple[np.ndarray, list[str]]:
    tools = sorted({t for r in rows for t in r["tool_counts"]})
    X = np.array([[r["tool_counts"].get(t, 0) for t in tools]
                  for r in rows], dtype=float)
    return X, tools


def run_audit(sample_path: str, cache: str, out_dir: str) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = build_matrix(sample_path, cache=cache)
    X, tool_cols = _design_matrix(rows)
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    r2 = {}
    for feat in IT_FEATURES:
        y = np.array([r["it"][feat] for r in rows], dtype=float)
        if np.std(y) < 1e-12:          # degenerate target -> R² undefined
            r2[feat] = {"cv_r2": None, "note": "near-constant target"}
            continue
        scores = cross_val_score(Ridge(alpha=1.0), X, y,
                                 cv=kf, scoring="r2")
        r2[feat] = {"cv_r2": float(np.mean(scores)),
                    "fold_r2": [float(s) for s in scores]}

    # Objective, locked gate
    gate = {}
    invalid = False
    for gf in GATE_FEATURES:
        v = r2[gf]["cv_r2"]
        if v is None:
            gate[gf] = {"cv_r2": None, "fails_gate": None,
                        "note": "near-constant — see report"}
        else:
            fails = v >= GATE_THRESHOLD
            gate[gf] = {"cv_r2": v, "fails_gate": bool(fails)}
            invalid = invalid or fails

    result = {
        "n_trajectories": len(rows),
        "n_tool_columns": len(tool_cols),
        "tool_columns": tool_cols,
        "gate_features": GATE_FEATURES,
        "gate_threshold": GATE_THRESHOLD,
        "gate": gate,
        "pilot_invalid": invalid,
        "all_feature_cv_r2": {k: v.get("cv_r2") for k, v in r2.items()},
        "detail": r2,
    }
    (out / "symbolization_audit.json").write_text(json.dumps(result, indent=2))
    (out / "symbolization_audit.md").write_text(_md(result))
    print(_md(result))
    return result


def _md(r: dict) -> str:
    L = ["# Pilot Symbolization Audit", "",
         f"- trajectories: {r['n_trajectories']}",
         f"- tool-name columns (design matrix): {r['n_tool_columns']}",
         f"- estimator: Ridge(alpha=1.0), KFold(5, shuffle, "
         f"random_state={SEED})  [pre-reg A2]",
         f"- gate: pilot INVALID if CV R² >= {r['gate_threshold']} for "
         f"{' or '.join(r['gate_features'])}", "",
         "## Gated features", "",
         "| feature | CV R² | fails gate? |", "|---|---:|---|"]
    for gf, g in r["gate"].items():
        v = g["cv_r2"]
        L.append(f"| {gf} | {'n/a' if v is None else f'{v:.4f}'} | "
                 f"{g.get('fails_gate')} |")
    L += ["", f"## GATE OUTCOME: "
          f"{'PILOT INVALID (symbolization confound)' if r['pilot_invalid'] else 'audit passes — IT not reconstructible from tool-API counts'}",
          "", "## All 20 IT features — CV R² (tool-counts -> feature)", "",
          "| feature | CV R² |", "|---|---:|"]
    for k, v in r["all_feature_cv_r2"].items():
        L.append(f"| {k} | {'n/a (near-constant)' if v is None else f'{v:.4f}'} |")
    L += ["", "_Locked, objective. Not re-litigated. Per pre-reg, if the "
          "gate fails the pilot is INVALID and H1/H2 are not interpreted._"]
    return "\n".join(L)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="/tmp/caft_pilot/sample.jsonl")
    ap.add_argument("--cache", default="/tmp/caft_pilot/features.json")
    ap.add_argument("--out", default="/tmp/caft_pilot")
    a = ap.parse_args()
    run_audit(a.sample, a.cache, a.out)


if __name__ == "__main__":
    main()
