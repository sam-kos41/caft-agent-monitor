"""Lead/lag analysis: does CAFT fire BEFORE the trivial loop detector?

This is the decisive convergent-validity test (see
docs/CONSTRUCT_REVISION.md). Session-level kappa already showed CAFT
and a literal-loop regex barely agree. The only way CAFT-the-detector
carries real information is if it fires *earlier* than the trivial
detector — i.e. it is an early-warning signal, not a more expensive
regex that co-fires.

Honesty constraints baked in:

  * SAME EVENT AXIS. Both fire-points are measured on the exact
    ObservableEvent stream CAFT consumes (the real
    _analyze_session extraction path), indexed by processed-event
    count. No two different clocks.
  * CAFT's fire = the REAL pipeline. t_caft is the first event where
    UniversalMonitor.process() actually emits a named anomaly
    signature. It is not a reimplemented or hand-picked threshold.
  * TRIVIAL DETECTOR pre-stated. t_literal is the event index at
    which a run of `literal_threshold` (default 8) consecutive
    identical ObservableEvents completes. Stated before looking at
    results; matches the 8x definition used in signals.py.
  * lead = t_literal - t_caft. Positive => CAFT led (fired earlier).
    ~0 => co-fire (CAFT adds nothing temporally). Negative => CAFT
    lagged. None => no literal loop in the session (not eligible for
    the lead statistic; reported separately).

Only sessions that actually contain a literal loop yield a lead
number. Sessions where CAFT fires with NO literal loop are reported
separately — they are either genuine non-literal thrash detections
(information beyond the regex) or false positives; this analysis
flags them for inspection, it does not adjudicate them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


@dataclass
class LeadLagResult:
    session_id: str
    n_events: int
    t_literal: Optional[int]      # event idx the 8x run completes
    t_caft: Optional[int]         # event idx CAFT first emits a signature
    caft_signature: str           # the signature CAFT first fired (if any)
    has_literal_loop: bool
    caft_fired: bool
    lead: Optional[int]           # t_literal - t_caft (None if no loop)

    def to_dict(self) -> dict:
        return asdict(self)

    def verdict(self) -> str:
        if not self.has_literal_loop:
            if self.caft_fired:
                return "caft_only (no literal loop — inspect: thrash or FP)"
            return "neither fired"
        if not self.caft_fired:
            return "literal_only (CAFT missed a real loop)"
        if self.lead is None:
            return "n/a"
        if self.lead > 3:
            return f"CAFT LED by {self.lead} events"
        if self.lead < -3:
            return f"CAFT lagged by {-self.lead} events"
        return f"co-fire (|lead|={abs(self.lead)} <=3 — no real lead)"


def _obs_sig(obs) -> tuple:
    """Normalized identity of an ObservableEvent for literal-run detection."""
    return (
        getattr(getattr(obs, "event_type", None), "value",
                str(getattr(obs, "event_type", ""))),
        getattr(obs, "tool_name", None),
        getattr(obs, "target_path", None),
    )


def lead_lag(jsonl_path: str | Path,
             literal_threshold: int = 8,
             sensitivity: float = 2.0) -> LeadLagResult:
    """Replay one session once; record both fire-points on one axis."""
    from agentdiag.universal_monitor import UniversalMonitor
    from agentdiag.live import _extract_trace_events_from_cc
    from agentdiag.cognitive import trace_event_to_observable
    from agentdiag.models import TraceEvent

    path = Path(jsonl_path)
    monitor = UniversalMonitor(sensitivity=sensitivity)

    idx = 0
    run_sig = None
    run_len = 0
    t_literal: Optional[int] = None
    t_caft: Optional[int] = None
    caft_sig = ""
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
                        for k in TraceEvent.__dataclass_fields__
                        if k in te
                    })
                    obs = trace_event_to_observable(tev)
                except Exception:
                    obs = None
                if obs is None:
                    continue

                # --- trivial literal-loop detector (pre-stated) ---
                s = _obs_sig(obs)
                if s == run_sig:
                    run_len += 1
                else:
                    run_sig, run_len = s, 1
                if t_literal is None and run_len >= literal_threshold:
                    t_literal = idx

                # --- CAFT's REAL fire point ---
                result = monitor.process(obs)
                if (t_caft is None and result
                        and result.get("anomalies")):
                    t_caft = idx
                    anom = result["anomalies"]
                    if isinstance(anom, dict):
                        caft_sig = anom.get("signature", "unnamed")
                    else:
                        caft_sig = "unnamed"

                idx += 1

    has_loop = t_literal is not None
    caft_fired = t_caft is not None
    lead = (t_literal - t_caft) if (has_loop and caft_fired) else None

    return LeadLagResult(
        session_id=path.stem,
        n_events=idx,
        t_literal=t_literal,
        t_caft=t_caft,
        caft_signature=caft_sig,
        has_literal_loop=has_loop,
        caft_fired=caft_fired,
        lead=lead,
    )


def run_corpus(corpus_dir: str | Path,
               literal_threshold: int = 8) -> list[LeadLagResult]:
    corpus = Path(corpus_dir)
    out = []
    for p in sorted(corpus.glob("*.jsonl")):
        try:
            out.append(lead_lag(p, literal_threshold=literal_threshold))
        except Exception as e:  # noqa: BLE001 — report, don't abort the batch
            print(f"  {p.stem[:16]}: ERROR {e}")
    return out


def format_report(results: list[LeadLagResult],
                  literal_threshold: int = 8) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append("CAFT LEAD/LAG vs the trivial literal-loop detector "
                 f"(threshold = {literal_threshold} identical events)")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"{'session':18} {'n_evt':>6} {'t_lit':>6} {'t_caft':>7} "
                 f"{'lead':>6}  verdict")
    lines.append("-" * 78)
    eligible = []
    for r in results:
        tl = "-" if r.t_literal is None else str(r.t_literal)
        tc = "-" if r.t_caft is None else str(r.t_caft)
        ld = "-" if r.lead is None else f"{r.lead:+d}"
        lines.append(f"{r.session_id[:18]:18} {r.n_events:6d} {tl:>6} "
                     f"{tc:>7} {ld:>6}  {r.verdict()}")
        if r.lead is not None:
            eligible.append(r.lead)
    lines.append("")
    lines.append("INTERPRETATION")
    lines.append("-" * 78)
    n_loop = sum(1 for r in results if r.has_literal_loop)
    n_caft_only = sum(1 for r in results
                      if r.caft_fired and not r.has_literal_loop)
    n_missed = sum(1 for r in results
                   if r.has_literal_loop and not r.caft_fired)
    lines.append(f"  Sessions with a literal loop (eligible): {n_loop}/{len(results)}")
    # ---- artifact guard (do NOT call a constant an early-warning) ----
    # If CAFT fires in (almost) every session at a tightly clustered
    # early index regardless of whether a literal loop exists, the
    # "lead" is an artifact of constant firing, not detection. A
    # detector that fires in ~100% of sessions has ~zero discriminative
    # power about the thing it claims to detect.
    fired = [r for r in results if r.caft_fired]
    fire_rate = len(fired) / len(results) if results else 0.0
    t_caft_vals = [r.t_caft for r in fired if r.t_caft is not None]
    if t_caft_vals:
        spread = max(t_caft_vals) - min(t_caft_vals)
        clustered = spread <= 120  # ~ the baseline calibration window
    else:
        clustered = False
    artifact = fire_rate >= 0.8 and clustered

    if eligible:
        import statistics
        med = statistics.median(eligible)
        led = sum(1 for x in eligible if x > 3)
        cofire = sum(1 for x in eligible if abs(x) <= 3)
        lagged = sum(1 for x in eligible if x < -3)
        lines.append(f"  Median raw lead: {med:+.0f} events  "
                     f"(LED {led}, co-fire {cofire}, lagged {lagged})")
        lines.append(f"  CAFT fire rate across ALL sessions: "
                     f"{fire_rate:.0%}  (t_caft spread "
                     f"{min(t_caft_vals)}-{max(t_caft_vals)})")
        lines.append("")
        if artifact:
            lines.append("  => ARTIFACT, NOT EARLY DETECTION. CAFT fires in")
            lines.append(f"     {fire_rate:.0%} of sessions at a clustered early")
            lines.append("     index (~the self-baseline calibration window),")
            lines.append("     loop or no loop. The apparent 'lead' is constant")
            lines.append("     firing, not prescience. CAFT carries ~zero")
            lines.append("     discriminative information about looping here.")
            lines.append("     This CONFIRMS the self-baseline pathology with")
            lines.append("     timing data (docs/CONSTRUCT_REVISION.md).")
        elif led > lagged + cofire:
            lines.append("  => CAFT fires earlier than the trivial detector")
            lines.append("     AND does not fire indiscriminately — a real")
            lines.append("     early-warning claim. (Verify on a larger corpus.)")
        elif cofire >= max(led, lagged):
            lines.append("  => CAFT CO-FIRES — no temporal information added.")
        else:
            lines.append("  => CAFT LAGS the trivial detector.")
    else:
        lines.append("  No eligible sessions — cannot compute a lead "
                     "statistic on this corpus.")
        lines.append(f"  (CAFT fire rate across ALL sessions: {fire_rate:.0%}"
                     + (f", clustered at ~{min(t_caft_vals)}-{max(t_caft_vals)}"
                        if t_caft_vals else "") + ")")
    lines.append("")
    lines.append(f"  CAFT fired with NO literal loop: {n_caft_only} "
                 f"(inspect: non-literal thrash vs false positive)")
    lines.append(f"  CAFT MISSED a real literal loop: {n_missed}")
    lines.append("")
    lines.append("  n is small and this corpus is not a random sample — "
                 "treat as directional, not definitive.")
    lines.append("=" * 78)
    return "\n".join(lines)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="CAFT lead/lag analysis")
    ap.add_argument("corpus", help="Directory of session JSONL files")
    ap.add_argument("--threshold", type=int, default=8,
                    help="Literal-loop trigger (consecutive identical events)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    results = run_corpus(args.corpus, literal_threshold=args.threshold)
    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
    else:
        print(format_report(results, literal_threshold=args.threshold))


if __name__ == "__main__":
    main()
