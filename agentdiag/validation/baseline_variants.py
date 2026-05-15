"""Steelman test: does CAFT detect looping under a BETTER baseline?

The self-calibrating-on-first-~100-events baseline fires on a clock
(calibration-window close), in 7/7 sessions, loop or not. That indicts
the calibration mechanism, not necessarily the information-theoretic
approach. This module re-runs the lead/lag test under two principled
alternatives and compares all three on the same event axis:

  self      — the real pipeline's SelfCalibratingBaseline (baseline)
  corpus    — z-score each session's per-step metrics against a
              reference distribution pooled over the OTHER sessions
              (leave-one-out). "anomaly" = "unlike normal sessions".
  changept  — abandon z-vs-baseline; flag the first event where the
              IT trajectory (action_mi / compression_ratio) shows a
              sustained level shift (windowed mean-shift, pre-stated
              w / k). Sidesteps calibration entirely.

Honesty constraints (identical to leadlag.py):
  * One event axis (the ObservableEvent stream CAFT consumes).
  * Per-step metrics are the pipeline's own result["metrics"] — not a
    reimplementation.
  * Trivial detector pre-stated: a run of `literal_threshold` (8)
    consecutive identical ObservableEvents.
  * A method that fires in ~all sessions at a clustered index has
    ~zero discriminative power, no matter how large its raw "lead" —
    the artifact guard reports this rather than calling it detection.
  * Corpus baseline uses LEAVE-ONE-OUT (score S against corpus minus
    S) to avoid circularity. n is small: directional only.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from agentdiag.validation.leadlag import _obs_sig

_CP_METRICS = ("action_mi", "compression_ratio")
_CP_WINDOW = 20      # pre-stated
_CP_K = 2.5          # pre-stated standardized-shift threshold
_CORPUS_Z = 2.0      # pre-stated, matches pipeline sensitivity default
_CORPUS_MIN_METRICS = 2  # mirrors compositor's "2+ anomalous" rule


@dataclass
class SessionTrace:
    session_id: str
    n_events: int
    t_literal: Optional[int]
    t_self: Optional[int]
    per_step: list[dict]      # pipeline result["metrics"] per observation


def collect(jsonl_path: str | Path,
            literal_threshold: int = 8,
            sensitivity: float = 2.0) -> SessionTrace:
    from agentdiag.universal_monitor import UniversalMonitor
    from agentdiag.live import _extract_trace_events_from_cc
    from agentdiag.cognitive import trace_event_to_observable
    from agentdiag.models import TraceEvent

    path = Path(jsonl_path)
    monitor = UniversalMonitor(sensitivity=sensitivity)
    idx = 0
    run_sig, run_len = None, 0
    t_literal = t_self = None
    per_step: list[dict] = []
    step_counter = [0]

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                raw = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            try:
                extracted = _extract_trace_events_from_cc(raw, step_counter)
            except Exception:
                continue
            for te in extracted:
                try:
                    tev = TraceEvent(**{
                        k: te.get(k)
                        for k in TraceEvent.__dataclass_fields__ if k in te
                    })
                    obs = trace_event_to_observable(tev)
                except Exception:
                    obs = None
                if obs is None:
                    continue
                s = _obs_sig(obs)
                if s == run_sig:
                    run_len += 1
                else:
                    run_sig, run_len = s, 1
                if t_literal is None and run_len >= literal_threshold:
                    t_literal = idx
                result = monitor.process(obs)
                if result and result.get("type") == "observation":
                    per_step.append(result.get("metrics", {}))
                    if t_self is None and result.get("anomalies"):
                        t_self = idx
                    idx += 1
    return SessionTrace(path.stem, idx, t_literal, t_self, per_step)


def _corpus_ref(traces: list[SessionTrace], exclude: str
                ) -> tuple[dict, dict]:
    """Leave-one-out pooled mean/std per metric over all OTHER sessions."""
    cols: dict[str, list[float]] = {}
    for t in traces:
        if t.session_id == exclude:
            continue
        for m in t.per_step:
            for k, v in m.items():
                if isinstance(v, (int, float)):
                    cols.setdefault(k, []).append(float(v))
    mean = {k: float(np.mean(v)) for k, v in cols.items() if v}
    std = {k: float(np.std(v)) for k, v in cols.items() if v}
    return mean, std


def corpus_fire(per_step: list[dict], mean: dict, std: dict,
                z: float = _CORPUS_Z,
                min_metrics: int = _CORPUS_MIN_METRICS) -> Optional[int]:
    for i, m in enumerate(per_step):
        hot = 0
        for k, v in m.items():
            s = std.get(k, 0.0)
            if s <= 1e-9 or k not in mean:
                continue
            if abs((float(v) - mean[k]) / s) >= z:
                hot += 1
        if hot >= min_metrics:
            return i
    return None


def changepoint_fire(per_step: list[dict],
                     metrics=_CP_METRICS,
                     w: int = _CP_WINDOW,
                     k: float = _CP_K) -> Optional[int]:
    series = {mn: np.array([float(s.get(mn, 0.0)) for s in per_step])
              for mn in metrics}
    n = len(per_step)
    if n < 2 * w:
        return None
    for t in range(w, n - w):
        for mn, arr in series.items():
            before, after = arr[t - w:t], arr[t:t + w]
            # within-window pooled SD (NOT std of the concatenation,
            # which is inflated by the very shift we're testing for).
            pooled = np.sqrt(
                (before.var() + after.var()) / 2.0) + 1e-9
            if abs(before.mean() - after.mean()) / pooled >= k:
                return t
    return None


@dataclass
class Row:
    sid: str
    n: int
    t_literal: Optional[int]
    t_self: Optional[int]
    t_corpus: Optional[int]
    t_cp: Optional[int]


def _fire_stats(name: str, fires: list[Optional[int]], n_total: int) -> str:
    present = [f for f in fires if f is not None]
    rate = len(present) / n_total if n_total else 0.0
    if present:
        spread = max(present) - min(present)
        clustered = spread <= 120
        tag = (" ARTIFACT(fires in {:.0%}, clustered {}-{})".format(
                   rate, min(present), max(present))
               if rate >= 0.8 and clustered
               else " (rate {:.0%}, spread {})".format(rate, spread))
    else:
        tag = " (never fires)"
    return f"  {name:9} fire-rate {rate:.0%}{tag}"


def run(corpus_dir: str | Path, literal_threshold: int = 8) -> str:
    corpus = Path(corpus_dir)
    traces = []
    for p in sorted(corpus.glob("*.jsonl")):
        try:
            traces.append(collect(p, literal_threshold))
        except Exception as e:  # noqa: BLE001
            print(f"  {p.stem[:16]}: ERROR {e}")
    rows: list[Row] = []
    for t in traces:
        mean, std = _corpus_ref(traces, t.session_id)
        rows.append(Row(
            sid=t.session_id, n=t.n_events,
            t_literal=t.t_literal, t_self=t.t_self,
            t_corpus=corpus_fire(t.per_step, mean, std),
            t_cp=changepoint_fire(t.per_step),
        ))

    L = []
    L.append("=" * 82)
    L.append("STEELMAN: CAFT looping detection under 3 baselines "
             f"(literal threshold = {literal_threshold})")
    L.append("=" * 82)
    L.append(f"{'session':18} {'n':>5} {'t_lit':>6} {'t_self':>7} "
             f"{'t_corpus':>9} {'t_cp':>6}")
    L.append("-" * 82)
    for r in rows:
        f = lambda x: "-" if x is None else str(x)
        L.append(f"{r.sid[:18]:18} {r.n:5d} {f(r.t_literal):>6} "
                 f"{f(r.t_self):>7} {f(r.t_corpus):>9} {f(r.t_cp):>6}")
    L.append("")
    L.append("PER-METHOD DISCRIMINATIVE POWER (a method that fires in ~all")
    L.append("sessions at a clustered index detects nothing, regardless of")
    L.append("raw 'lead'):")
    nt = len(rows)
    L.append(_fire_stats("self", [r.t_self for r in rows], nt))
    L.append(_fire_stats("corpus", [r.t_corpus for r in rows], nt))
    L.append(_fire_stats("changept", [r.t_cp for r in rows], nt))
    L.append("")
    # Lead vs literal, per method, only on sessions with a literal loop
    elig = [r for r in rows if r.t_literal is not None]
    L.append(f"LEAD vs literal detector (eligible sessions: {len(elig)}/{nt})")
    if elig:
        for method, attr in (("self", "t_self"), ("corpus", "t_corpus"),
                             ("changept", "t_cp")):
            leads = [r.t_literal - getattr(r, attr)
                     for r in elig if getattr(r, attr) is not None]
            if leads:
                L.append(f"  {method:9} median lead "
                         f"{statistics.median(leads):+.0f}  (n={len(leads)})")
            else:
                L.append(f"  {method:9} never fired on an eligible session")
    L.append("")
    L.append("READ THIS HONESTLY: a positive median lead means NOTHING if")
    L.append("that method is flagged ARTIFACT above. Detection requires")
    L.append("(a) fire rate well below 100%, (b) fire point that VARIES")
    L.append("with session content and tracks t_literal. n is tiny and the")
    L.append("corpus is not a random sample — directional only.")
    L.append("=" * 82)
    return "\n".join(L)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--threshold", type=int, default=8)
    args = ap.parse_args()
    print(run(args.corpus, args.threshold))


if __name__ == "__main__":
    main()
