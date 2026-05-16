"""Leg 4 — error-recovery / adaptive-behavior features + gate.

Per PREREG_LEG4_ERROR_RECOVERY.md (FROZEN, greenlit 2026-05-15).
6 deterministic features, Reason/Hollnagel-grounded, deliberately NOT
IT-flavored. Wall-clock latency + semantic ack excluded by
construction. Reuses the frozen Leg-1 sample. Runs the standing
symbolization gate then STOPS for the mandatory human checkpoint.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score

from agentdiag.adapters.swe_agent import _extract_action

SEED = 20260515
K_RECOVER = 3
GATE_FEATURES = ("error.strategy_change_rate", "error.recovery_success_rate")
GATE_THRESHOLD = 0.80

ERROR_FEATURES = (
    "error.n_episodes", "error.recurrence_rate",
    "error.strategy_change_rate", "error.recovery_success_rate",
    "error.mean_latency_turns", "error.terminal_unresolved",
)

# FROZEN error signature (pre-reg §4) — case-insensitive
_ERR = re.compile(
    r"error|traceback|no such file|command not found|not found|"
    r"exception|failed|cannot|invalid syntax|permission denied", re.I)


def _base(p):
    if not p:
        return ""
    return os.path.basename(str(p).rstrip("/")) or str(p)


def _sequence(row: dict):
    """Ordered combined stream: ('act',(tool,target)) | ('obs',is_err).
    The initial issue (turn idx<=1) is not an observation episode."""
    seq = []
    for i, t in enumerate(row.get("trajectory") or []):
        role = t.get("role")
        txt = t.get("text") or ""
        if role in ("ai", "assistant"):
            a = _extract_action(txt)
            if a is not None:
                cmd, args = a
                seq.append(("act", (cmd.lower(), _base(
                    re.search(r'["\']([^"\']+)["\']', args).group(1)
                    if re.search(r'["\']([^"\']+)["\']', args)
                    else (args.split(None, 1)[0] if args.strip() else "")))))
        elif role in ("user", "tool"):
            if i > 1:
                seq.append(("obs", bool(_ERR.search(txt))))
    return seq


def extract_error_one(row: dict) -> dict:
    seq = _sequence(row)
    n = len(seq)

    # episodes: error obs, consecutive error-obs with no act between merge
    episodes = []  # list of dict(fail, resp, seq_idx)
    prev_act = None
    last_was_err_obs = False
    for idx, (kind, val) in enumerate(seq):
        if kind == "act":
            prev_act = val
            last_was_err_obs = False
        else:  # obs
            if val:  # error
                if not last_was_err_obs:
                    episodes.append({"fail": prev_act, "resp": None,
                                     "idx": idx})
                last_was_err_obs = True
            else:
                last_was_err_obs = False

    # response action for each episode = next 'act' after its obs idx
    for ep in episodes:
        for j in range(ep["idx"] + 1, n):
            if seq[j][0] == "act":
                ep["resp"] = seq[j][1]
                break

    ne = len(episodes)
    if ne:
        recur = 0
        seen_fail = []
        chg = 0
        rec = 0
        lats = []
        for ep in episodes:
            f = ep["fail"]
            if f is not None and f in seen_fail:
                recur += 1
            if f is not None:
                seen_fail.append(f)
            if f is not None and ep["resp"] is not None and ep["resp"] != f:
                chg += 1
            # recovery: a non-error obs within K_RECOVER steps after idx
            recovered_at = None
            for j in range(ep["idx"] + 1,
                           min(n, ep["idx"] + 1 + K_RECOVER)):
                if seq[j][0] == "obs" and seq[j][1] is False:
                    recovered_at = j
                    break
            if recovered_at is not None:
                rec += 1
            # latency: turns to the NEXT non-error obs (unbounded);
            # only episodes that ever recover contribute to the mean
            for j in range(ep["idx"] + 1, n):
                if seq[j][0] == "obs" and seq[j][1] is False:
                    lats.append(j - ep["idx"])
                    break
        recurrence_rate = recur / ne
        strategy_change_rate = chg / ne
        recovery_success_rate = rec / ne
        mean_latency = float(np.mean(lats)) if lats else 0.0
    else:
        recurrence_rate = strategy_change_rate = 0.0
        recovery_success_rate = mean_latency = 0.0

    # terminal_unresolved: last obs is error AND no submit act after it
    last_err_obs_idx = None
    for idx in range(n - 1, -1, -1):
        if seq[idx][0] == "obs":
            last_err_obs_idx = idx if seq[idx][1] else None
            break
    terminal_unresolved = 0.0
    if last_err_obs_idx is not None:
        submitted_after = any(
            seq[j][0] == "act" and seq[j][1][0] == "submit"
            for j in range(last_err_obs_idx + 1, n))
        terminal_unresolved = 0.0 if submitted_after else 1.0

    feats = {
        "error.n_episodes": float(ne),
        "error.recurrence_rate": float(recurrence_rate),
        "error.strategy_change_rate": float(strategy_change_rate),
        "error.recovery_success_rate": float(recovery_success_rate),
        "error.mean_latency_turns": float(mean_latency),
        "error.terminal_unresolved": float(terminal_unresolved),
    }
    tc: dict = {}
    for kind, val in seq:
        if kind == "act":
            tc[val[0]] = tc.get(val[0], 0) + 1
    return {"error": feats, "target": bool(row.get("target")),
            "tool_counts": tc}


def build_error(sample_path: str, cache: str | None = None) -> list:
    if cache and Path(cache).exists():
        return json.loads(Path(cache).read_text())
    out = []
    with open(sample_path, encoding="utf-8") as f:
        for k, line in enumerate(f, 1):
            line = line.strip()
            if line:
                out.append(extract_error_one(json.loads(line)))
            if k % 250 == 0:
                print(f"  ...error extracted {k}", flush=True)
    if cache:
        Path(cache).write_text(json.dumps(out))
    return out


def run_gate(sample_path: str, cache: str, out_dir: str) -> dict:
    rows = build_error(sample_path, cache=cache)
    tools = sorted({t for r in rows for t in r["tool_counts"]})
    X = np.array([[r["tool_counts"].get(t, 0) for t in tools]
                  for r in rows], dtype=float)
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    r2 = {}
    for feat in ERROR_FEATURES:
        y = np.array([r["error"][feat] for r in rows], dtype=float)
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
           "gate": gate, "leg4_invalid": invalid,
           "all_error_cv_r2": r2}
    o = Path(out_dir)
    o.mkdir(parents=True, exist_ok=True)
    (o / "leg4_error_audit.json").write_text(json.dumps(res, indent=2))
    (o / "leg4_error_audit.md").write_text(_md(res))
    print(_md(res))
    return res


def _md(r: dict) -> str:
    L = ["# Leg 4 (Error Recovery) Symbolization Audit", "",
         f"- trajectories: {r['n']}  tool columns: {r['n_tool_columns']}",
         "- Ridge(alpha=1.0), KFold(5, random_state=20260515)",
         f"- gate: Leg-4 INVALID if CV R² >= {GATE_THRESHOLD} for "
         f"{' or '.join(GATE_FEATURES)} (the adaptive/relational pair)",
         "", "| gate feature | CV R² | fails? |", "|---|---:|---|"]
    for gf, g in r["gate"].items():
        v = g["cv_r2"]
        L.append(f"| {gf} | {'n/a' if v is None else f'{v:.4f}'} "
                 f"| {g['fails_gate']} |")
    L += ["", f"## GATE: {'LEG-4 INVALID (ER = tool-API artifact)' if r['leg4_invalid'] else 'passes — adaptive ER not reconstructible from tool counts'}",
          "", "| ER feature | CV R² |", "|---|---:|"]
    for k, v in r["all_error_cv_r2"].items():
        L.append(f"| {k} | {'n/a (near-constant)' if v is None else f'{v:.4f}'} |")
    L += ["", "_Locked, objective. STOP for the mandatory human "
          "checkpoint before E-H1/H2/H3 (PROGRAM.md rule 3)._"]
    return "\n".join(L)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="/tmp/caft_pilot/sample.jsonl")
    ap.add_argument("--cache", default="/tmp/caft_pilot/error.json")
    ap.add_argument("--out", default="/tmp/caft_pilot")
    a = ap.parse_args()
    run_gate(a.sample, a.cache, a.out)


if __name__ == "__main__":
    main()
