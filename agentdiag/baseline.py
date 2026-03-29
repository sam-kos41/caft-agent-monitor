"""Self-calibrating baseline for real-time anomaly detection.

Collects raw metric values for the first `calibration_window` events,
locks per-metric (mean, std) baselines, then returns z-scores for each
metric on every subsequent event.  No training data, no labeled examples.

When the harness provides phase boundaries, maintains SEPARATE baselines
per phase.  Falls back to a single global baseline when no phases are
declared.  This is the graceful degradation path — the system works
without a harness, but gets more sensitive with one.

Usage::

    baseline = SelfCalibratingBaseline(calibration_window=100, sensitivity=2.0)

    # During live processing:
    baseline.set_phase("executing")   # optional — from PHASE_BOUNDARY events
    anomalies = baseline.observe({"tool_entropy": 1.82, "action_mi": 0.6})

    # anomalies is {} during calibration, or:
    # {"tool_entropy": {"value": 3.5, "z_score": 2.8, "expected_mean": 1.8,
    #                   "expected_std": 0.6, "direction": "high"}}
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Optional


class SelfCalibratingBaseline:
    """Real-time self-calibrating baseline with optional per-phase tracking.

    During calibration (first `calibration_window` observations), collects
    values for each metric.  After calibration, locks baselines and returns
    z-scores.  Metrics exceeding `sensitivity` standard deviations from the
    mean are flagged as anomalous.

    If phases are declared (via set_phase()), maintains per-phase baselines.
    Falls back to global baseline for phases with too few samples.
    """

    MIN_PHASE_SAMPLES = 30  # fall back to global if phase has fewer

    def __init__(
        self,
        calibration_window: int = 100,
        sensitivity: float = 2.0,
    ) -> None:
        self.calibration_window = calibration_window
        self.sensitivity = sensitivity

        self._calibrating = True
        self._observation_count = 0

        # Global buffers (used during calibration)
        self._buffers: dict[str, list[float]] = defaultdict(list)

        # Per-phase buffers (used during calibration when phases are declared)
        self._phase_buffers: dict[tuple[str, str], list[float]] = defaultdict(list)

        # Locked baselines: {metric: (mean, std)}
        self._baselines: dict[str, tuple[float, float]] = {}

        # Per-phase locked baselines: {phase: {metric: (mean, std)}}
        self._phase_baselines: dict[str, dict[str, tuple[float, float]]] = {}

        self._current_phase: Optional[str] = None

        # Post-calibration metric histories for trend detection
        self._metric_histories: dict[str, list[float]] = defaultdict(list)
        self._gradual_declines: dict[str, dict] = {}
        self._trend_window = 30  # steps to look back for slope

    @property
    def is_calibrating(self) -> bool:
        return self._calibrating

    @property
    def current_phase(self) -> Optional[str]:
        return self._current_phase

    def set_phase(self, phase: Optional[str]) -> None:
        """Set the current phase context.

        Called when a PHASE_BOUNDARY event arrives from the harness.
        Pass None to clear phase context (global baseline only).
        """
        self._current_phase = phase

    def observe(
        self,
        metrics: dict[str, float],
        phase: Optional[str] = None,
    ) -> dict[str, dict]:
        """Observe a set of metrics and return anomalies.

        Args:
            metrics: {metric_name: value} — the IT measures for this event.
            phase: Optional phase override (if not set via set_phase()).

        Returns:
            {} during calibration.
            {metric_name: {"value", "z_score", "expected_mean",
             "expected_std", "direction"}} for anomalous metrics post-cal.
        """
        effective_phase = phase or self._current_phase
        self._observation_count += 1

        if self._calibrating:
            for k, v in metrics.items():
                if not math.isfinite(v):
                    continue
                self._buffers[k].append(v)
                if effective_phase:
                    self._phase_buffers[(effective_phase, k)].append(v)

            if self._observation_count >= self.calibration_window:
                self._lock_baselines()
            return {}

        # Post-calibration: compute z-scores against best available baseline
        baseline = self._get_baseline(effective_phase)
        anomalies: dict[str, dict] = {}

        for k, v in metrics.items():
            if not math.isfinite(v):
                continue
            if k not in baseline:
                continue
            mean, std = baseline[k]
            if std < 1e-10:
                # Constant during calibration — any deviation is novel
                if abs(v - mean) > 1e-6:
                    anomalies[k] = {
                        "value": round(v, 4),
                        "z_score": round(self.sensitivity + 1, 2),
                        "expected_mean": round(mean, 4),
                        "expected_std": 0.0,
                        "direction": "high" if v > mean else "low",
                    }
                continue
            z = abs(v - mean) / std
            if z > self.sensitivity:
                anomalies[k] = {
                    "value": round(v, 4),
                    "z_score": round(z, 2),
                    "expected_mean": round(mean, 4),
                    "expected_std": round(std, 4),
                    "direction": "high" if v > mean else "low",
                }

        # Track metric histories for gradual decline detection
        for k, v in metrics.items():
            if math.isfinite(v):
                self._metric_histories[k].append(v)
                # Keep bounded
                if len(self._metric_histories[k]) > 200:
                    self._metric_histories[k] = self._metric_histories[k][-200:]

        # Check for gradual declines (trend-based, separate from threshold-based)
        self._gradual_declines = {}
        for k in metrics:
            if k not in baseline:
                continue
            mean, std = baseline[k]
            if std < 1e-10:
                continue
            decline = self._detect_gradual_decline(
                self._metric_histories.get(k, []), mean, std
            )
            if decline is not None:
                self._gradual_declines[k] = decline

        return anomalies

    def _detect_gradual_decline(
        self,
        history: list[float],
        baseline_mean: float,
        baseline_std: float,
    ) -> Optional[dict]:
        """Detect a gradual downward trend over the last N steps.

        Uses linear regression. If slope is significantly negative (> 2 SE
        below zero) AND current value is in the bottom 20th percentile of
        the calibration distribution, flag as gradual decline.
        """
        if len(history) < self._trend_window:
            return None
        window = history[-self._trend_window:]
        n = len(window)
        x_mean = (n - 1) / 2.0
        y_mean = sum(window) / n

        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(window))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        if denominator == 0:
            return None
        slope = numerator / denominator

        # Standard error of slope
        residuals_ss = sum(
            (y - (y_mean + slope * (i - x_mean))) ** 2
            for i, y in enumerate(window)
        )
        if n <= 2:
            return None
        se = math.sqrt(residuals_ss / (n - 2)) / math.sqrt(denominator)
        if se < 1e-12:
            return None

        # 20th percentile of normal distribution: mean - 0.84 * std
        threshold = baseline_mean - 0.84 * baseline_std
        current = window[-1]

        if slope < -2 * se and current < threshold:
            return {
                "slope": round(slope, 6),
                "se": round(se, 6),
                "current": round(current, 4),
                "threshold": round(threshold, 4),
                "baseline_mean": round(baseline_mean, 4),
            }
        return None

    @property
    def gradual_declines(self) -> dict[str, dict]:
        """Currently detected gradual declines, keyed by metric name."""
        return dict(self._gradual_declines)

    def get_baseline_summary(self) -> dict:
        """Return locked baselines for debugging/display."""
        return {
            "calibrating": self._calibrating,
            "observation_count": self._observation_count,
            "global": {
                k: {"mean": round(m, 4), "std": round(s, 4)}
                for k, (m, s) in self._baselines.items()
            },
            "phases": {
                phase: {
                    k: {"mean": round(m, 4), "std": round(s, 4)}
                    for k, (m, s) in metrics.items()
                }
                for phase, metrics in self._phase_baselines.items()
            },
            "gradual_declines": self._gradual_declines,
        }

    def manual_baseline_lock(self) -> None:
        """Force-lock baselines early.  Escape hatch for the operator."""
        if self._calibrating:
            self._lock_baselines()

    # ── Private ──

    def _lock_baselines(self) -> None:
        """Compute and freeze (mean, std) from calibration buffers."""
        self._calibrating = False

        # Global baselines
        for k, values in self._buffers.items():
            if len(values) >= 2:
                mean = sum(values) / len(values)
                variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
                std = math.sqrt(variance)
                self._baselines[k] = (mean, std)

        # Per-phase baselines
        phase_metrics: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for (phase, metric), values in self._phase_buffers.items():
            phase_metrics[phase][metric] = values

        for phase, metrics in phase_metrics.items():
            self._phase_baselines[phase] = {}
            for k, values in metrics.items():
                if len(values) >= 2:
                    mean = sum(values) / len(values)
                    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
                    std = math.sqrt(variance)
                    self._phase_baselines[phase][k] = (mean, std)

        # Free calibration buffers
        self._buffers.clear()
        self._phase_buffers.clear()

    def _get_baseline(self, phase: Optional[str]) -> dict[str, tuple[float, float]]:
        """Get the best available baseline for the current context.

        Uses per-phase baseline if available and has enough samples.
        Falls back to global baseline otherwise.
        """
        if phase and phase in self._phase_baselines:
            phase_bl = self._phase_baselines[phase]
            if len(phase_bl) >= 1:  # has at least one metric
                return phase_bl
        return self._baselines
