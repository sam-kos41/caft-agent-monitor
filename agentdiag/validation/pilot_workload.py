"""Leg 2 — cognitive-workload-analog features + symbolization gate.

Per PREREG_LEG2_WORKLOAD.md. Extracts ONLY the 9 observable workload
features (§4); non-observable signals are excluded by construction
(no timestamps in nebius -> no timing features here at all). Reuses
the frozen Leg-1 spec sample. Runs the standing symbolization-audit
gate and STOPS — W-H1/H2/H3 are a separate module gated by the
mandatory human checkpoint (PROGRAM.md rule 3).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score

from agentdiag.adapters.swe_agent import row_to_events  # for n_parsed_actions
from agentdiag.validation.pilot_features import _aggregate

SEED = 20260515
_FENCE = re.compile(r"```[a-zA-Z0-9_]*\n?.*?```", re.DOTALL)
_ERR = re.compile(r"error|traceback|no such file|command not found|"
                  r"not found|exception", re.I)
GATE_FEATURES = ("reasoning_len.mean", "context_cum.final")
GATE_THRESHOLD = 0.80

WORKLOAD_FEATURES = (
    "reasoning_len.mean", "reasoning_len.max",
    "reasoning_len.total", "reasoning_len.slope",
    "context_cum.final", "context_cum.slope",
    "reasoning_density.mean",
    "error_recovery.n_episodes", "error_recovery.mean_latency_turns",
)


def _reasoning_text(ai_text: str) -> str:
    """ai turn text with all fenced action blocks stripped = the
    sustained-processing 'thought'."""
    return _FENCE.sub("", ai_text or "").strip()


def extract_workload_one(row: dict) -> dict:
    traj = row.get("trajectory") or []
    reasoning_lens: list[int] = []
    cum: list[int] = []
    running = 0
    n_actions = 0
    # error-recovery over observation (non-ai) turns after the issue
    err_episodes = 0
    latencies: list[int] = []
    pending_err_at: int | None = None

    for i, t in enumerate(traj):
        txt = t.get("text") or ""
        running += len(txt)
        cum.append(running)
        role = t.get("role")
        if role in ("ai", "assistant"):
            rt = _reasoning_text(txt)
            if _FENCE.search(txt):
                n_actions += 1
            reasoning_lens.append(len(rt))
            # an ai turn after an error closes the recovery episode
            if pending_err_at is not None:
                latencies.append(i - pending_err_at)
                pending_err_at = None
        else:
            # observation turn: error signature opens an episode
            if i > 1 and _ERR.search(txt):
                if pending_err_at is None:
                    err_episodes += 1
                    pending_err_at = i

    rl = _aggregate([float(x) for x in reasoning_lens])
    total_rl = float(sum(reasoning_lens))
    cseries = [float(x) for x in cum]
    cfinal = cseries[-1] if cseries else 0.0
    cslope = (float(np.polyfit(range(len(cseries)), cseries, 1)[0])
              if len(cseries) > 1 else 0.0)
    density = (total_rl / n_actions) if n_actions else 0.0
    mean_lat = float(np.mean(latencies)) if latencies else 0.0

    feats = {
        "reasoning_len.mean": rl["mean"],
        "reasoning_len.max": rl["max"],
        "reasoning_len.total": total_rl,
        "reasoning_len.slope": rl["slope"],
        "context_cum.final": cfinal,
        "context_cum.slope": cslope,
        "reasoning_density.mean": density,
        "error_recovery.n_episodes": float(err_episodes),
        "error_recovery.mean_latency_turns": mean_lat,
    }
    return {"workload": feats, "target": bool(row.get("target")),
            "tool_counts": _tool_counts(row)}


def _tool_counts(row: dict) -> dict:
    c: dict = {}
    for e in row_to_events(row):
        k = e.tool_name or "?"
        c[k] = c.get(k, 0) + 1
    return c


def build_workload(sample_path: str, cache: str | None = None) -> list:
    if cache and Path(cache).exists():
        return json.loads(Path(cache).read_text())
    out = []
    with open(sample_path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if line:
                out.append(extract_workload_one(json.loads(line)))
            if n % 250 == 0:
                print(f"  ...workload extracted {n}", flush=True)
    if cache:
        Path(cache).write_text(json.dumps(out))
    return out


def run_gate(sample_path: str, cache: str, out_dir: str) -> dict:
    rows = build_workload(sample_path, cache=cache)
    tools = sorted({t for r in rows for t in r["tool_counts"]})
    X = np.array([[r["tool_counts"].get(t, 0) for t in tools]
                  for r in rows], dtype=float)
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    r2 = {}
    for feat in WORKLOAD_FEATURES:
        y = np.array([r["workload"][feat] for r in rows], dtype=float)
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
           "gate": gate, "leg2_invalid": invalid,
           "all_workload_cv_r2": r2}
    o = Path(out_dir)
    o.mkdir(parents=True, exist_ok=True)
    (o / "leg2_workload_audit.json").write_text(json.dumps(res, indent=2))
    (o / "leg2_workload_audit.md").write_text(_md(res))
    print(_md(res))
    return res


def _md(r: dict) -> str:
    L = ["# Leg 2 (Workload) Symbolization Audit", "",
         f"- trajectories: {r['n']}  tool columns: {r['n_tool_columns']}",
         "- Ridge(alpha=1.0), KFold(5, random_state=20260515) [standing rule]",
         f"- gate: Leg-2 INVALID if CV R² >= {GATE_THRESHOLD} for "
         f"{' or '.join(GATE_FEATURES)}", "", "| gate feature | CV R² | fails? |",
         "|---|---:|---|"]
    for gf, g in r["gate"].items():
        v = g["cv_r2"]
        L.append(f"| {gf} | {'n/a' if v is None else f'{v:.4f}'} "
                 f"| {g['fails_gate']} |")
    L += ["", f"## GATE: {'LEG-2 INVALID (workload = tool-API artifact)' if r['leg2_invalid'] else 'passes — workload not reconstructible from tool counts'}",
          "", "| workload feature | CV R² |", "|---|---:|"]
    for k, v in r["all_workload_cv_r2"].items():
        L.append(f"| {k} | {'n/a (near-constant)' if v is None else f'{v:.4f}'} |")
    L += ["", "_Locked, objective. STOP here for the mandatory human "
          "checkpoint before W-H1/H2/H3 (PROGRAM.md rule 3)._"]
    return "\n".join(L)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="/tmp/caft_pilot/sample.jsonl")
    ap.add_argument("--cache", default="/tmp/caft_pilot/workload.json")
    ap.add_argument("--out", default="/tmp/caft_pilot")
    a = ap.parse_args()
    run_gate(a.sample, a.cache, a.out)


if __name__ == "__main__":
    main()
