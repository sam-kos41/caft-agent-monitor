"""Agent-native pilot — thought-action coherence (TAC) + gate.

Per PREREG_AGENT_NATIVE_TAC.md (FROZEN). PRIMARY deterministic
operationalization only here (the decision-bearing one). Robustness 1
(semantic) / 2 (LLM-graded) are separate, non-gating, and only touched
after the gate checkpoint. Reuses the frozen Legs-1-4 sample.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score

from agentdiag.adapters.swe_agent import _extract_action, _FENCE

SEED = 20260515
GATE_FEATURES = ("tac.mean", "tac.target_match_rate")
GATE_THRESHOLD = 0.80

TAC_FEATURES = (
    "tac.mean", "tac.min", "tac.final", "tac.slope",
    "tac.verb_align_rate", "tac.target_match_rate",
)

# FROZEN verb-class lexicon (pre-reg §3)
_LEX = {
    "observe": {"look", "read", "open", "view", "inspect", "check",
                "see", "examine"},
    "search": {"search", "find", "locate", "grep", "where"},
    "mutate": {"edit", "change", "modify", "fix", "add", "create",
               "write", "update", "replace", "implement"},
    "verify": {"run", "execute", "test", "verify", "confirm", "check"},
    "submit": {"submit", "done", "finish", "complete"},
}
# action command verb -> category
_CAT = {}
for v in ("open", "goto", "scroll_down", "scroll_up", "cat", "head",
          "tail"):
    _CAT[v] = "observe"
for v in ("search_dir", "search_file", "find_file", "grep", "ls"):
    _CAT[v] = "search"
for v in ("edit", "create", "insert", "append"):
    _CAT[v] = "mutate"
for v in ("python", "pytest", "tox", "make", "nosetests", "test",
          "bash"):
    _CAT[v] = "verify"
_CAT["submit"] = "submit"

_WORD = re.compile(r"[a-z_]+")


def _thought(ai_text: str) -> str:
    return _FENCE.sub("", ai_text or "").strip()


def _target_token(args: str) -> str:
    m = re.search(r'["\']([^"\']+)["\']', args or "")
    raw = m.group(1) if m else (
        args.split(None, 1)[0] if (args or "").strip() else "")
    base = os.path.basename(str(raw).rstrip("/")) or str(raw)
    # for a search query take its first alpha token
    if not base:
        return ""
    mt = re.search(r"[A-Za-z_][A-Za-z0-9_]+", base)
    return (mt.group(0) if mt else base).lower()


def extract_tac_one(row: dict) -> dict:
    per_turn = []
    verb_aligns = []
    target_flags = []     # only for actions that have a target
    tool_counts: dict = {}
    for t in row.get("trajectory") or []:
        if t.get("role") not in ("ai", "assistant"):
            continue
        txt = t.get("text") or ""
        act = _extract_action(txt)
        if act is None:
            continue
        verb, args = act
        verb = verb.lower()
        tool_counts[verb] = tool_counts.get(verb, 0) + 1
        cat = _CAT.get(verb, "other")
        thought = _thought(txt).lower()
        tw = set(_WORD.findall(thought))

        if cat == "other":
            verb_align = 0
        else:
            verb_align = 1 if (tw & _LEX[cat]) else 0
        verb_aligns.append(verb_align)

        tgt = _target_token(args)
        if tgt:
            tp = 1 if tgt in thought else 0
            target_flags.append(tp)
        else:
            tp = verb_align  # target-less action not penalized
        per_turn.append(0.5 * verb_align + 0.5 * tp)

    if per_turn:
        arr = np.asarray(per_turn, float)
        slope = (float(np.polyfit(np.arange(len(arr)), arr, 1)[0])
                 if len(arr) > 1 else 0.0)
        feats = {
            "tac.mean": float(arr.mean()),
            "tac.min": float(arr.min()),
            "tac.final": float(arr[-1]),
            "tac.slope": slope,
            "tac.verb_align_rate": float(np.mean(verb_aligns)),
            "tac.target_match_rate": (float(np.mean(target_flags))
                                      if target_flags else 0.0),
        }
    else:
        feats = {k: 0.0 for k in TAC_FEATURES}
    return {"tac": feats, "target": bool(row.get("target")),
            "tool_counts": tool_counts}


def build_tac(sample_path: str, cache: str | None = None) -> list:
    if cache and Path(cache).exists():
        return json.loads(Path(cache).read_text())
    out = []
    with open(sample_path, encoding="utf-8") as f:
        for k, line in enumerate(f, 1):
            line = line.strip()
            if line:
                out.append(extract_tac_one(json.loads(line)))
            if k % 250 == 0:
                print(f"  ...TAC extracted {k}", flush=True)
    if cache:
        Path(cache).write_text(json.dumps(out))
    return out


def run_gate(sample_path: str, cache: str, out_dir: str) -> dict:
    rows = build_tac(sample_path, cache=cache)
    tools = sorted({t for r in rows for t in r["tool_counts"]})
    X = np.array([[r["tool_counts"].get(t, 0) for t in tools]
                  for r in rows], dtype=float)
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    r2 = {}
    for feat in TAC_FEATURES:
        y = np.array([r["tac"][feat] for r in rows], dtype=float)
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
           "gate": gate, "tac_invalid": invalid,
           "all_tac_cv_r2": r2}
    o = Path(out_dir)
    o.mkdir(parents=True, exist_ok=True)
    (o / "tac_audit.json").write_text(json.dumps(res, indent=2))
    (o / "tac_audit.md").write_text(_md(res))
    print(_md(res))
    return res


def _md(r: dict) -> str:
    L = ["# Agent-Native TAC Symbolization Audit", "",
         f"- trajectories: {r['n']}  tool columns: {r['n_tool_columns']}",
         "- Ridge(alpha=1.0), KFold(5, random_state=20260515)",
         f"- gate: INVALID if CV R² >= {GATE_THRESHOLD} for "
         f"{' or '.join(GATE_FEATURES)} (construct-bearing pair)", "",
         "| gate feature | CV R² | fails? |", "|---|---:|---|"]
    for gf, g in r["gate"].items():
        v = g["cv_r2"]
        L.append(f"| {gf} | {'n/a' if v is None else f'{v:.4f}'} "
                 f"| {g['fails_gate']} |")
    L += ["", f"## GATE: {'INVALID (TAC = tool-API artifact)' if r['tac_invalid'] else 'passes — TAC not reconstructible from tool counts'}",
          "", "| TAC feature | CV R² |", "|---|---:|"]
    for k, v in r["all_tac_cv_r2"].items():
        L.append(f"| {k} | {'n/a (near-constant)' if v is None else f'{v:.4f}'} |")
    L += ["", "_Locked, objective. STOP for the mandatory human "
          "checkpoint before TAC-H1/H2/H3 (PROGRAM.md rule 3)._"]
    return "\n".join(L)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="/tmp/caft_pilot/sample.jsonl")
    ap.add_argument("--cache", default="/tmp/caft_pilot/tac.json")
    ap.add_argument("--out", default="/tmp/caft_pilot")
    a = ap.parse_args()
    run_gate(a.sample, a.cache, a.out)


if __name__ == "__main__":
    main()
