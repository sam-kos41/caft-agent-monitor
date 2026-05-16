"""Per-trajectory feature extraction — the shared spine for the pilot.

Computes, for each trajectory, exactly the features frozen in
docs/PILOT_PREREGISTRATION.md §4:

  IT features (20): for base in {action_mi, action_entropy,
    tool_entropy, compression_ratio, kl_divergence}, the per-trajectory
    {mean, final, max, slope} of the pipeline's own per-step
    result["metrics"] (slope = OLS slope over step index).
  Trivial baseline: n_turns, n_parsed_actions, patch_len,
    exit_status (categorical), model_name (categorical).
  Plus: per-trajectory Counter of event tool_name (the audit design
    matrix, per amendment A2).

Used by the symbolization audit AND H1/H2 so the numbers are
identical across phases. No statistics or decisions here — extraction
only.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterator

import numpy as np

from agentdiag.adapters.swe_agent import row_to_events, baseline_features
from agentdiag.universal_monitor import UniversalMonitor

IT_BASE = ("action_mi", "action_entropy", "tool_entropy",
           "compression_ratio", "kl_divergence")
IT_SUFFIX = ("mean", "final", "max", "slope")
IT_FEATURES = tuple(f"{b}.{s}" for b in IT_BASE for s in IT_SUFFIX)


def _aggregate(series: list[float]) -> dict:
    if not series:
        return {"mean": 0.0, "final": 0.0, "max": 0.0, "slope": 0.0}
    arr = np.asarray(series, dtype=float)
    slope = (float(np.polyfit(np.arange(len(arr)), arr, 1)[0])
             if len(arr) > 1 else 0.0)
    return {"mean": float(arr.mean()), "final": float(arr[-1]),
            "max": float(arr.max()), "slope": slope}


def extract_one(row: dict) -> dict:
    """Return {it: {...20...}, tool_counts: Counter, baseline: {...},
    target: bool, n_events: int}."""
    events = row_to_events(row)
    mon = UniversalMonitor(sensitivity=2.0)
    series = {b: [] for b in IT_BASE}
    tool_counts: Counter = Counter()
    for e in events:
        tool_counts[e.tool_name or "?"] += 1
        res = mon.process(e)
        if res.get("type") == "observation":
            mt = res.get("metrics", {})
            for b in IT_BASE:
                series[b].append(float(mt.get(b, 0.0)))
    it = {}
    for b in IT_BASE:
        agg = _aggregate(series[b])
        for s in IT_SUFFIX:
            it[f"{b}.{s}"] = agg[s]
    return {
        "it": it,
        "tool_counts": dict(tool_counts),
        "baseline": baseline_features(row),
        "target": bool(row.get("target")),
        "n_events": len(events),
    }


def iter_sample(sample_path: str | Path) -> Iterator[dict]:
    with Path(sample_path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_matrix(sample_path: str | Path, cache: str | Path | None = None):
    """Extract features for the whole sample. Returns a list of dicts.

    Caches to JSON so the audit and H1/H2 reuse identical numbers
    without recompute (and without the 127MB raw sample).
    """
    cache = Path(cache) if cache else None
    if cache and cache.exists():
        return json.loads(cache.read_text())
    out = []
    n = 0
    for row in iter_sample(sample_path):
        out.append(extract_one(row))
        n += 1
        if n % 250 == 0:
            print(f"  ...extracted {n}", flush=True)
    if cache:
        cache.write_text(json.dumps(out))
    return out
