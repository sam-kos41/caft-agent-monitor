"""Rule-based mapping from CAFT's continuous metrics to validation labels.

Runs the existing CAFT analysis pipeline on a session and converts the
output (action_mi, kl_divergence, signature counts, etc.) into the same
1-5 Likert + categorical schema the human and Ollama raters use, so all
three are directly comparable.

The rules below are documented and deterministic. They are *the
hypothesis under test* — if Sam↔CAFT kappa is low, refining these
rules (or refining the underlying signatures) is the response.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from agentdiag.validation.digest import (
    SessionDigest, DIMENSIONS, LIKERT_DIMS, SCALE_ANCHORS, HEALTH_ANCHORS,
)
from agentdiag.validation.ledger import Rating


RATER_ID = "caft-v0.3"  # v0.1 abstains-on-nothing; v0.2 abstains on
                         # user_satisfied + goal_drifted; v0.3 ALSO abstains
                         # on overall_health (retired — see
                         # docs/CONSTRUCT_REVISION.md). CAFT now reports only
                         # stuck_in_loop + coherent_progress.


@dataclass
class CaftMetrics:
    """Subset of CAFT's per-session metrics used by the rater."""
    action_mi: float
    tool_entropy: float
    kl_divergence: float
    compression_ratio: float
    anomaly_count: int
    event_count: int
    health: str

    @property
    def anomaly_rate(self) -> float:
        return self.anomaly_count / max(self.event_count, 1)


def compute_caft_metrics(jsonl_path: str) -> CaftMetrics:
    """Run the existing CAFT analysis and extract the metrics we need."""
    from agentdiag.plugin import CAFT
    caft = CAFT(sensitivity=2.0)
    result = caft.analyze(jsonl_path)
    metrics = result.get("metrics", {})
    return CaftMetrics(
        action_mi=float(metrics.get("action_mi", 0.0)),
        tool_entropy=float(metrics.get("tool_entropy", 0.0)),
        kl_divergence=float(metrics.get("kl_divergence", 0.0)),
        compression_ratio=float(metrics.get("compression_ratio", 1.0)),
        anomaly_count=int(result.get("anomaly_count", 0)),
        event_count=int(result.get("events", 0)),
        health=str(result.get("behavioral_state",
                               result.get("health", "unknown"))),
    )


def _likert_from_threshold(value: float, low: float, high: float,
                           inverted: bool = False) -> int:
    """Bucket a continuous metric into 1-5.

    If inverted=False, higher values map to higher Likert.
    If inverted=True, higher values map to lower Likert (e.g. low MI = stuck).
    """
    if inverted:
        if value <= low: return 5
        if value >= high: return 1
        frac = (high - value) / (high - low)
    else:
        if value <= low: return 1
        if value >= high: return 5
        frac = (value - low) / (high - low)
    return max(1, min(5, int(round(1 + frac * 4))))


def rate_with_caft(digest: SessionDigest,
                   metrics: Optional[CaftMetrics] = None) -> list[Rating]:
    """Produce 5 Rating rows for a session using CAFT's existing metrics.

    Mapping rules (the hypothesis under test):

      stuck_in_loop:    inverse of action_mi
                        MI < 0.9 -> 5 (very stuck), MI > 1.7 -> 1 (not stuck)
      goal_drifted:     scaled kl_divergence
                        KL < 0.2 -> 1 (no drift), KL > 0.5 -> 5 (heavy drift)
      coherent_progress: action_mi
                        MI < 0.9 -> 1, MI > 1.7 -> 5
      user_satisfied:   inverse of anomaly_rate (proxy: high anomaly rate
                        suggests the session was problematic)
                        rate < 0.1 -> 5, rate > 0.5 -> 1
      overall_health:   passes through caft.health (red->pathological,
                        yellow->degraded, green->healthy)
    """
    if metrics is None:
        metrics = compute_caft_metrics(digest.source_path)

    stuck = _likert_from_threshold(metrics.action_mi, 0.9, 1.7, inverted=True)
    coherent = _likert_from_threshold(metrics.action_mi, 0.9, 1.7)

    metric_basis = (
        f"action_mi={metrics.action_mi:.2f} "
        f"behavioral_state={metrics.health}"
    )

    # v0.3 (see docs/CONSTRUCT_REVISION.md): CAFT reports ONLY the two
    # dimensions its IT metrics actually bear on. It abstains on:
    #   - user_satisfied: no user-sentiment model (v0.1 mapped it from
    #     inverse anomaly_rate and confidently mis-rated a rage session)
    #   - goal_drifted:  KL-divergence is not a valid drift construct
    #   - overall_health: RETIRED. The red/yellow/green verdict was not
    #     supported by the math (kappa = -0.04 vs domain expert). CAFT
    #     no longer issues a session quality verdict at all.
    rated = {
        "stuck_in_loop": stuck,
        "coherent_progress": coherent,
    }
    abstained = {
        "user_satisfied": "no user-sentiment model — IT metrics do not "
                          "observe user satisfaction",
        "goal_drifted": "KL-divergence is not a construct-valid drift "
                        "signal (conflates user-directed re-scoping)",
        "overall_health": "retired — unvalidated quality verdict "
                          "(see docs/CONSTRUCT_REVISION.md)",
    }

    out: list[Rating] = []
    for dim in DIMENSIONS:
        if dim in abstained:
            out.append(Rating(
                session_id=digest.session_id,
                rater_type="caft", rater_id=RATER_ID, dimension=dim,
                value=None, confidence="",
                reasoning=f"ABSTAIN: {abstained[dim]}",
            ))
            continue
        v = rated[dim]
        anchor = (SCALE_ANCHORS[dim][v] if dim in LIKERT_DIMS
                  else HEALTH_ANCHORS.get(v, ""))
        out.append(Rating(
            session_id=digest.session_id,
            rater_type="caft", rater_id=RATER_ID, dimension=dim,
            value=v, confidence="high",
            reasoning=f"{metric_basis} -> {dim}={v}: {anchor}",
        ))
    return out
