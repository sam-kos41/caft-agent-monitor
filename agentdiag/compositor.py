"""Compositional anomaly detector — interprets multi-metric co-occurrences.

Receives anomaly dicts from SelfCalibratingBaseline, ignores single-metric
blips (noise), and interprets multi-metric co-occurrences as named anomaly
signatures derived from information theory.

Key signatures (empirically validated against SymbolStream window=50):

  - low entropy + low MI                = mechanical_repetition (stuck in a loop)
  - low entropy + high MI               = tight_iteration (focused, possibly stuck)
  - kl_divergence HIGH + entropy HIGH    = distributional_shift (goal drift)
  - kl_divergence HIGH + MI HIGH        = context_thrashing (frantic re-reading)
    (without entropy HIGH)
  - compression LOW + entropy LOW       = execution_regression (degenerate loop)
  - high tier_escalation + high ns_H    = memory_thrashing (OpenViking overload)
  - low entropy + low MI (aggregate)    = stagnation (completely stuck)
  - anything else multi-metric          = unclassified_anomaly (novel — still flagged)

Design note: signatures use kl_divergence because it is the most sensitive
change detector (baseline std≈0 during calibration, so any KL shift triggers).
Stream-specific metrics (read_entropy, tool_entropy) are used alongside the
aggregate action_entropy because they decouple — a flood of reads can spike
read_entropy without affecting tool_entropy.

No training data required.  Interpretations come from IT theory, not labels.

Usage::

    compositor = CompositionalAnomalyDetector()
    anomalies = baseline.observe(metrics)  # from SelfCalibratingBaseline
    if anomalies:
        result = compositor.analyze(anomalies)
        # result = {"signature": "mechanical_repetition", "severity": "warning",
        #           "metrics": {...}, "interpretation": "..."}
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AnomalySignature:
    """A named multi-metric anomaly pattern mapped to a Wickens IP stage."""
    name: str
    severity: str  # "info", "warning", "critical"
    interpretation: str
    metrics: dict[str, dict]  # the raw anomaly dict from baseline
    step: Optional[int] = None
    wickens_stage: str = "unknown"  # which Wickens stage is failing

    def to_dict(self) -> dict:
        return {
            "signature": self.name,
            "severity": self.severity,
            "interpretation": self.interpretation,
            "metrics": self.metrics,
            "step": self.step,
            "wickens_stage": self.wickens_stage,
        }


# ── Signature definitions ──────────────────────────────────────────────────
# Each signature is a (name, required_conditions, severity, interpretation)
# tuple.  Conditions are functions over the anomaly dict.

def _has(anomalies: dict, metric: str, direction: str) -> bool:
    """Check if a metric is anomalous in a given direction."""
    return metric in anomalies and anomalies[metric].get("direction") == direction


def _has_any_prefix(anomalies: dict, prefix: str, direction: str) -> bool:
    """Check if any metric with the given prefix is anomalous."""
    return any(
        k.startswith(prefix) and v.get("direction") == direction
        for k, v in anomalies.items()
    )


# Each signature: (name, condition, severity, interpretation, wickens_stage)
# wickens_stage identifies which stage in the Wickens IP model is failing.
#
# Priority order matters: more specific signatures come first so they match
# before broader ones.  Signatures are derived from empirical observation of
# what metric co-occurrences actually occur at window=50 with the current
# SymbolStream implementation.
#
# Key insight: kl_divergence calibrates with std≈0 (distribution is stable
# during the first 100 events), so ANY distributional change registers as
# z=infinity.  This makes KL the most sensitive single change detector.
# Signatures that include KL alongside a second metric are highly specific.

_SIGNATURES: list[tuple[str, callable, str, str, str]] = [
    # ── Entropy + MI signatures (classic IT pairs) ────────────────────
    (
        "mechanical_repetition",
        lambda a: (_has(a, "action_entropy", "low") or _has(a, "tool_entropy", "low"))
                  and _has(a, "action_mi", "low"),
        "warning",
        "Response selection stuck — low entropy, low MI. "
        "The agent is repeating the same actions with no coherent structure.",
        "response_selection",
    ),
    (
        "tight_iteration",
        lambda a: (_has(a, "action_entropy", "low") or _has(a, "tool_entropy", "low"))
                  and _has(a, "action_mi", "high"),
        "info",
        "Response execution cycling — low entropy, high MI. "
        "Focused on a narrow set of actions with high sequential coherence.",
        "response_execution",
    ),

    # ── KL-based signatures (distributional change detectors) ─────────
    (
        "distributional_shift",
        lambda a: _has(a, "kl_divergence", "high")
                  and (_has(a, "last_surprisal", "high") or _has(a, "action_entropy", "high")
                       or _has(a, "read_entropy", "high")),
        "warning",
        "Perceptual processing disrupted — high KL divergence with high entropy. "
        "The action distribution has shifted toward novel, high-diversity behavior. "
        "Indicates goal drift or sudden context switch to unfamiliar territory.",
        "perceptual",
    ),
    (
        "context_thrashing",
        lambda a: _has(a, "kl_divergence", "high")
                  and _has(a, "action_mi", "high")
                  and not _has(a, "action_entropy", "high"),
        "warning",
        "Working memory overloaded — high KL divergence with high MI but stable entropy. "
        "The agent is frantically re-reading known files in rapid succession. "
        "Distribution shifted but symbol diversity hasn't increased — chaotic not exploratory.",
        "working_memory",
    ),

    # ── Compression-based signatures ──────────────────────────────────
    (
        "execution_regression",
        lambda a: _has(a, "compression_ratio", "low")
                  and (_has(a, "action_entropy", "low") or _has(a, "read_entropy", "low")),
        "warning",
        "Response execution degraded — low compression ratio, low entropy. "
        "Action sequence is highly compressible and low-diversity. "
        "The agent is stuck in a degenerate loop.",
        "response_execution",
    ),
    (
        "stagnation",
        lambda a: _has(a, "compression_ratio", "low") and _has(a, "kl_divergence", "low"),
        "critical",
        "Response selection stuck — low compression, stable distribution. "
        "The sequence is repetitive but the overall distribution looks normal. "
        "Almost no new information is being generated.",
        "response_selection",
    ),

    # ── KL + entropy catch-all (broad distributional anomaly) ─────────
    (
        "distributional_anomaly",
        lambda a: _has(a, "kl_divergence", "high")
                  and (_has(a, "action_entropy", "low") or _has(a, "action_entropy", "high")),
        "info",
        "Action distribution has shifted — high KL divergence with abnormal entropy. "
        "The agent's behavior has changed from its calibrated baseline.",
        "response_selection",
    ),

    # ── Memory-specific signatures (OpenViking domain) ────────────────
    (
        "memory_thrashing",
        lambda a: (_has(a, "memory_escalation_rate", "high")
                   and _has(a, "namespace_entropy", "high")),
        "warning",
        "Working memory overloaded — high tier escalation, high namespace entropy. "
        "Loading and unloading context from many namespaces rapidly.",
        "working_memory",
    ),
]


class CompositionalAnomalyDetector:
    """Interprets multi-metric anomaly co-occurrences as named signatures.

    Single-metric anomalies are ignored (treated as noise).  Only when
    2+ metrics are simultaneously anomalous does the compositor check
    for known signature patterns.  Unknown multi-metric patterns are
    reported as 'unclassified_anomaly'.

    Tracks signature frequency for the operator to review post-session.
    """

    def __init__(self, min_metrics: int = 2) -> None:
        self._min_metrics = min_metrics
        self._history: deque[AnomalySignature] = deque(maxlen=500)
        self._signature_counts: dict[str, int] = {}

    def analyze(
        self,
        anomalies: dict[str, dict],
        step: Optional[int] = None,
    ) -> Optional[AnomalySignature]:
        """Analyze a set of anomalous metrics for known signatures.

        Args:
            anomalies: {metric: {"value", "z_score", "direction", ...}}
                       from SelfCalibratingBaseline.observe().
            step: Current step number for timeline correlation.

        Returns:
            AnomalySignature if a multi-metric pattern is found, else None.
            Returns None for single-metric blips (noise).
        """
        if len(anomalies) < self._min_metrics:
            return None

        # Check known signatures in priority order
        for name, condition, severity, interpretation, wickens_stage in _SIGNATURES:
            try:
                if condition(anomalies):
                    sig = AnomalySignature(
                        name=name,
                        severity=severity,
                        interpretation=interpretation,
                        metrics=anomalies,
                        wickens_stage=wickens_stage,
                        step=step,
                    )
                    self._record(sig)
                    return sig
            except Exception:
                continue

        # Multi-metric anomaly but no known signature → novel failure
        sig = AnomalySignature(
            name="unclassified_anomaly",
            severity="info",
            interpretation=(
                f"Multiple metrics are simultaneously anomalous "
                f"({', '.join(anomalies.keys())}) but the pattern doesn't "
                f"match any known information-theoretic signature.  "
                f"This may be a novel failure mode worth investigating."
            ),
            metrics=anomalies,
            step=step,
        )
        self._record(sig)
        return sig

    @property
    def history(self) -> list[AnomalySignature]:
        return list(self._history)

    @property
    def signature_counts(self) -> dict[str, int]:
        return dict(self._signature_counts)

    def get_summary(self) -> dict:
        """Summary for debugging/display."""
        return {
            "total_anomalies": len(self._history),
            "signature_counts": self._signature_counts,
            "recent": [s.to_dict() for s in list(self._history)[-10:]],
        }

    def _record(self, sig: AnomalySignature) -> None:
        self._history.append(sig)
        self._signature_counts[sig.name] = (
            self._signature_counts.get(sig.name, 0) + 1
        )
