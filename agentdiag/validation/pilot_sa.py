"""Leg 3 — situation-awareness-analog features + symbolization gate.

Per PREREG_LEG3_SA.md. ONLY Endsley L1 (perception) + L3 (projection),
6 deterministic features from the parsed action stream (tool_name +
target_path). L2 comprehension is excluded by construction (no NLP).
Reuses the frozen Leg-1 sample. Runs the standing symbolization gate
then STOPS for the mandatory human checkpoint.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score

from agentdiag.adapters.swe_agent import row_to_events

SEED = 20260515
GATE_FEATURES = ("perception.coverage", "projection.verify_before_submit")
GATE_THRESHOLD = 0.80

SA_FEATURES = (
    "perception.coverage", "perception.explore_ratio",
    "perception.read_before_first_edit", "perception.blind_edit_rate",
    "projection.verify_before_submit", "projection.verify_rate",
)

_OBSERVE_PATH = {"open", "cat", "scroll_down", "scroll_up", "goto",
                 "head", "tail"}
_OBSERVE_ANY = _OBSERVE_PATH | {"search_dir", "search_file",
                                "find_file", "grep", "ls"}
_MUTATE = {"edit", "create", "insert", "append"}
_VERIFY = {"python", "pytest", "tox", "make", "nosetests", "test",
           "grep", "search_dir", "search_file", "find_file"}


def _base(p):
    if not p:
        return ""
    return os.path.basename(str(p).rstrip("/")) or str(p)


def extract_sa_one(row: dict) -> dict:
    ev = row_to_events(row)
    tools = [(e.tool_name or "").lower() for e in ev]
    targs = [_base(e.target_path) for e in ev]
    n = len(ev)

    observed_paths: set[str] = set()
    mutate_targets: list[str] = []
    blind_edits = 0
    first_mutate = None
    last_mutate = -1
    n_observe = 0
    n_verify = 0
    submit_idx = None

    for i, (tl, tg) in enumerate(zip(tools, targs)):
        if tl in _OBSERVE_ANY:
            n_observe += 1
        if tl in _VERIFY:
            n_verify += 1
        if tl in _MUTATE:
            if first_mutate is None:
                first_mutate = i
            last_mutate = i
            mutate_targets.append(tg)
            if tg and tg not in observed_paths:
                blind_edits += 1
        if tl in _OBSERVE_PATH and tg:
            observed_paths.add(tg)
        if tl == "submit" and submit_idx is None:
            submit_idx = i

    distinct_mut = [t for t in dict.fromkeys(mutate_targets) if t]
    if distinct_mut:
        # was each distinct mutate target observed at SOME earlier obs?
        # (recompute with ordering: observed-before-first-mutation-of-it)
        seen: set[str] = set()
        covered = set()
        for i, (tl, tg) in enumerate(zip(tools, targs)):
            if tl in _OBSERVE_PATH and tg:
                seen.add(tg)
            if tl in _MUTATE and tg and tg in seen:
                covered.add(tg)
        coverage = len(covered) / len(distinct_mut)
    else:
        coverage = 0.0  # no demonstrated perception-before-action

    explore_ratio = (n_observe / n) if n else 0.0
    read_before = ((first_mutate / n) if (first_mutate is not None and n)
                   else (1.0 if n else 0.0))
    blind_rate = (blind_edits / n) if n else 0.0

    # L3 projection: verification strictly after last mutate, before submit
    end = submit_idx if submit_idx is not None else n
    vbs = 0
    if last_mutate >= 0:
        for i in range(last_mutate + 1, end):
            if tools[i] in _VERIFY:
                vbs = 1
                break
    verify_rate = (n_verify / n) if n else 0.0

    feats = {
        "perception.coverage": float(coverage),
        "perception.explore_ratio": float(explore_ratio),
        "perception.read_before_first_edit": float(read_before),
        "perception.blind_edit_rate": float(blind_rate),
        "projection.verify_before_submit": float(vbs),
        "projection.verify_rate": float(verify_rate),
    }
    tc: dict = {}
    for tl in tools:
        tc[tl or "?"] = tc.get(tl or "?", 0) + 1
    return {"sa": feats, "target": bool(row.get("target")),
            "tool_counts": tc}


def build_sa(sample_path: str, cache: str | None = None) -> list:
    if cache and Path(cache).exists():
        return json.loads(Path(cache).read_text())
    out = []
    with open(sample_path, encoding="utf-8") as f:
        for k, line in enumerate(f, 1):
            line = line.strip()
            if line:
                out.append(extract_sa_one(json.loads(line)))
            if k % 250 == 0:
                print(f"  ...SA extracted {k}", flush=True)
    if cache:
        Path(cache).write_text(json.dumps(out))
    return out


def run_gate(sample_path: str, cache: str, out_dir: str) -> dict:
    rows = build_sa(sample_path, cache=cache)
    tools = sorted({t for r in rows for t in r["tool_counts"]})
    X = np.array([[r["tool_counts"].get(t, 0) for t in tools]
                  for r in rows], dtype=float)
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    r2 = {}
    for feat in SA_FEATURES:
        y = np.array([r["sa"][feat] for r in rows], dtype=float)
        if np.std(y) < 1e-12:
            r2[feat] = None
            continue
        r2[feat] = float(np.mean(
            cross_val_score(Ridge(alpha=1.0), X, y, cv=kf, scoring="r2")))
    invalid = False
    gate = {}
    for gf in GATE_FEATURES:
        v = r2.get(gf)
        fails = (v is not None and v >= GATE_THRESHOLD)
        gate[gf] = {"cv_r2": v, "fails_gate": bool(fails)}
        invalid = invalid or fails
    res = {"n": len(rows), "n_tool_columns": len(tools),
           "gate": gate, "leg3_invalid": invalid,
           "all_sa_cv_r2": r2}
    o = Path(out_dir)
    o.mkdir(parents=True, exist_ok=True)
    (o / "leg3_sa_audit.json").write_text(json.dumps(res, indent=2))
    (o / "leg3_sa_audit.md").write_text(_md(res))
    print(_md(res))
    return res


def _md(r: dict) -> str:
    L = ["# Leg 3 (Situation Awareness) Symbolization Audit", "",
         f"- trajectories: {r['n']}  tool columns: {r['n_tool_columns']}",
         "- Ridge(alpha=1.0), KFold(5, random_state=20260515)",
         f"- gate: Leg-3 INVALID if CV R² >= {GATE_THRESHOLD} for "
         f"{' or '.join(GATE_FEATURES)} (the relational features)", "",
         "| gate feature | CV R² | fails? |", "|---|---:|---|"]
    for gf, g in r["gate"].items():
        v = g["cv_r2"]
        L.append(f"| {gf} | {'n/a' if v is None else f'{v:.4f}'} "
                 f"| {g['fails_gate']} |")
    L += ["", f"## GATE: {'LEG-3 INVALID (SA = tool-API artifact)' if r['leg3_invalid'] else 'passes — relational SA not reconstructible from tool counts'}",
          "", "| SA feature | CV R² |", "|---|---:|"]
    for k, v in r["all_sa_cv_r2"].items():
        L.append(f"| {k} | {'n/a (near-constant)' if v is None else f'{v:.4f}'} |")
    L += ["", "_Locked, objective. STOP for the mandatory human "
          "checkpoint before SA-H1/H2/H3 (PROGRAM.md rule 3)._"]
    return "\n".join(L)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="/tmp/caft_pilot/sample.jsonl")
    ap.add_argument("--cache", default="/tmp/caft_pilot/sa.json")
    ap.add_argument("--out", default="/tmp/caft_pilot")
    a = ap.parse_args()
    run_gate(a.sample, a.cache, a.out)


if __name__ == "__main__":
    main()
