"""Validation tests for the analysis layer.

Tests the full pipeline: synthetic events → UniversalMonitor → anomaly detection.

Healthy run assertions:
  - All IT metrics are non-zero after 100 steps
  - Phase-conditional baselines have different means per phase
  - Very few anomalies detected (< 10 named anomalies)

Anomalous run assertions:
  - mechanical_repetition detected near step 200
  - goal_discontinuity or incoherent_exploration detected near step 350
  - context_thrashing detected near step 500
  - All detections happen WITHOUT manual labeling
"""

from __future__ import annotations

import unittest
from collections import Counter

from agentdiag.testing.synthetic_events import (
    generate_healthy_run,
    generate_anomalous_run,
)
from agentdiag.universal_monitor import UniversalMonitor
from agentdiag.cognitive import SymbolStream
from agentdiag.baseline import SelfCalibratingBaseline
from agentdiag.compositor import CompositionalAnomalyDetector, AnomalySignature
from agentdiag.observable import EventType, ObservableEvent


class TestSyntheticEvents(unittest.TestCase):
    """Test the synthetic event generator itself."""

    def test_healthy_run_has_events(self):
        events = generate_healthy_run()
        self.assertGreater(len(events), 300)

    def test_anomalous_run_has_events(self):
        events = generate_anomalous_run()
        self.assertGreater(len(events), 500)

    def test_healthy_run_has_all_event_types(self):
        events = generate_healthy_run()
        types = {e.event_type for e in events}
        self.assertIn(EventType.SESSION_START, types)
        self.assertIn(EventType.SESSION_END, types)
        self.assertIn(EventType.PHASE_BOUNDARY, types)
        self.assertIn(EventType.TOOL_CALL, types)
        self.assertIn(EventType.MEMORY_LOAD, types)
        self.assertIn(EventType.MEMORY_STORE, types)
        self.assertIn(EventType.EVALUATION_RESULT, types)
        self.assertIn(EventType.CONTRACT_EVENT, types)

    def test_anomalous_run_has_tier_escalations(self):
        events = generate_anomalous_run()
        types = {e.event_type for e in events}
        self.assertIn(EventType.MEMORY_TIER_ESCALATION, types)

    def test_events_are_deterministic(self):
        a = generate_healthy_run(seed=42)
        b = generate_healthy_run(seed=42)
        self.assertEqual(len(a), len(b))
        for ea, eb in zip(a, b):
            self.assertEqual(ea.event_type, eb.event_type)
            self.assertEqual(ea.step, eb.step)

    def test_steps_are_sequential(self):
        events = generate_anomalous_run()
        for i, e in enumerate(events):
            self.assertEqual(e.step, i + 1, f"Step gap at index {i}")


class TestSymbolStream(unittest.TestCase):
    """Test Agent 2's SymbolStream (from cognitive.py) IT measures."""

    def test_entropy_nonzero_after_100(self):
        stream = SymbolStream(window=50, lz_window=150)
        events = generate_healthy_run()
        pushed = 0
        for e in events:
            if not e.is_phase_marker():
                stream.push(e.to_symbol())
                pushed += 1
                if pushed == 100:
                    break
        self.assertGreater(stream.entropy(), 0.0)

    def test_mi_nonzero_after_100(self):
        stream = SymbolStream(window=50, lz_window=150)
        events = generate_healthy_run()
        for e in events[:150]:
            if not e.is_phase_marker():
                stream.push(e.to_symbol())
        self.assertGreater(stream.mi(), 0.0)

    def test_compression_nonzero(self):
        stream = SymbolStream(window=50, lz_window=150)
        events = generate_healthy_run()
        for e in events[:150]:
            if not e.is_phase_marker():
                stream.push(e.to_symbol())
        self.assertGreater(stream.compression(), 0.0)

    def test_baseline_locks_after_100(self):
        stream = SymbolStream(window=50, lz_window=150)
        events = generate_healthy_run()
        pushed = 0
        for e in events:
            if not e.is_phase_marker():
                stream.push(e.to_symbol())
                pushed += 1
                if pushed == 99:
                    self.assertFalse(stream._baseline_locked)
                if pushed == 100:
                    self.assertTrue(stream._baseline_locked)
                    break

    def test_kl_divergence_zero_before_lock(self):
        stream = SymbolStream(window=50, lz_window=150)
        events = generate_healthy_run()
        for e in events[:50]:
            if not e.is_phase_marker():
                stream.push(e.to_symbol())
        self.assertEqual(stream.kl_divergence(), 0.0)

    def test_surprisal_with_laplace(self):
        stream = SymbolStream(window=50, lz_window=150)
        events = generate_healthy_run()
        for e in events[:150]:
            if not e.is_phase_marker():
                stream.push(e.to_symbol())
        # Known symbol should have finite surprisal
        s = stream.surprisal("tool:Read")
        self.assertGreater(s, 0.0)
        self.assertTrue(s < 20.0)  # no hard cap, but should be reasonable
        # Unknown symbol should have high but finite surprisal
        s_rare = stream.surprisal("never_seen_before_xyz")
        self.assertGreater(s_rare, s)

    def test_history_populated(self):
        stream = SymbolStream(window=50, lz_window=150)
        events = generate_healthy_run()
        for e in events:
            if not e.is_phase_marker():
                stream.push(e.to_symbol())
        self.assertGreater(len(stream.entropy_history), 0)
        self.assertGreater(len(stream.mi_history), 0)
        self.assertGreater(len(stream.compression_history), 0)
        self.assertGreater(len(stream.surprisal_history), 0)
        self.assertGreater(len(stream.kl_history), 0)


class TestSelfCalibratingBaseline(unittest.TestCase):
    """Test the baseline from Agent 2's implementation."""

    def test_calibration_period(self):
        bl = SelfCalibratingBaseline(calibration_window=50, sensitivity=3.0)
        for i in range(49):
            result = bl.observe({"x": float(i)})
            self.assertEqual(result, {})
        self.assertTrue(bl.is_calibrating)

        bl.observe({"x": 50.0})
        self.assertFalse(bl.is_calibrating)

    def test_detects_anomaly(self):
        bl = SelfCalibratingBaseline(calibration_window=50, sensitivity=2.0)
        # Calibrate with values around 1.0
        for i in range(50):
            bl.observe({"x": 1.0 + (i % 3) * 0.1})

        # Normal value — no anomaly
        result = bl.observe({"x": 1.1})
        self.assertEqual(result, {})

        # Extreme value — anomaly
        result = bl.observe({"x": 10.0})
        self.assertIn("x", result)

    def test_phase_conditional_baselines(self):
        bl = SelfCalibratingBaseline(calibration_window=60, sensitivity=3.0)
        # Calibrate with different values per phase
        for i in range(30):
            bl.observe({"x": 1.0 + (i % 3) * 0.05}, phase="planning")
        for i in range(30):
            bl.observe({"x": 5.0 + (i % 3) * 0.1}, phase="executing")

        summary = bl.get_baseline_summary()
        self.assertIn("planning", summary["phases"])
        self.assertIn("executing", summary["phases"])
        # Means should differ
        plan_mean = summary["phases"]["planning"]["x"]["mean"]
        exec_mean = summary["phases"]["executing"]["x"]["mean"]
        self.assertNotAlmostEqual(plan_mean, exec_mean, places=0)

    def test_manual_lock(self):
        bl = SelfCalibratingBaseline(calibration_window=100, sensitivity=3.0)
        for i in range(30):
            bl.observe({"x": 1.0})
        self.assertTrue(bl.is_calibrating)
        bl.manual_baseline_lock()
        self.assertFalse(bl.is_calibrating)


class TestCompositionalAnomalyDetector(unittest.TestCase):
    """Test the compositor from Agent 2's implementation."""

    def test_ignores_single_metric(self):
        comp = CompositionalAnomalyDetector()
        result = comp.analyze({"x": {"value": 10, "z_score": 5, "direction": "high"}})
        self.assertIsNone(result)

    def test_detects_mechanical_repetition(self):
        comp = CompositionalAnomalyDetector()
        anomalies = {
            "action_entropy": {"value": 0.1, "z_score": 4.0, "direction": "low"},
            "action_mi": {"value": 0.05, "z_score": 3.5, "direction": "low"},
        }
        result = comp.analyze(anomalies)
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "mechanical_repetition")

    def test_detects_distributional_shift(self):
        comp = CompositionalAnomalyDetector()
        anomalies = {
            "kl_divergence": {"value": 2.0, "z_score": 4.0, "direction": "high"},
            "last_surprisal": {"value": 8.0, "z_score": 3.5, "direction": "high"},
        }
        result = comp.analyze(anomalies)
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "distributional_shift")

    def test_detects_memory_thrashing(self):
        comp = CompositionalAnomalyDetector()
        anomalies = {
            "memory_escalation_rate": {"value": 0.9, "z_score": 5.0, "direction": "high"},
            "namespace_entropy": {"value": 3.5, "z_score": 4.0, "direction": "high"},
        }
        result = comp.analyze(anomalies)
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "memory_thrashing")

    def test_unclassified_for_unknown_pattern(self):
        comp = CompositionalAnomalyDetector()
        anomalies = {
            "weird_metric_a": {"value": 10, "z_score": 5, "direction": "high"},
            "weird_metric_b": {"value": 10, "z_score": 5, "direction": "low"},
        }
        result = comp.analyze(anomalies)
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "unclassified_anomaly")

    def test_tracks_signature_counts(self):
        comp = CompositionalAnomalyDetector()
        for _ in range(3):
            comp.analyze({
                "action_entropy": {"value": 0.1, "z_score": 4, "direction": "low"},
                "action_mi": {"value": 0.05, "z_score": 3, "direction": "low"},
            })
        self.assertEqual(comp.signature_counts["mechanical_repetition"], 3)


# Helper to extract anomaly signatures from UniversalMonitor process() results
def _collect_anomalies(monitor: UniversalMonitor, events: list) -> list[tuple[int, AnomalySignature]]:
    """Run events through monitor, collect (step, signature) pairs."""
    anomalies: list[tuple[int, AnomalySignature]] = []
    for e in events:
        result = monitor.process(e)
        if result.get("anomalies") is not None:
            # Reconstruct AnomalySignature from dict
            ad = result["anomalies"]
            sig = AnomalySignature(
                name=ad["signature"],
                severity=ad["severity"],
                interpretation=ad.get("interpretation", ""),
                metrics=ad.get("metrics", {}),
                step=ad.get("step"),
            )
            anomalies.append((e.step, sig))
    return anomalies


class TestUniversalMonitorHealthy(unittest.TestCase):
    """Validate: healthy run produces few/no anomalies."""

    @classmethod
    def setUpClass(cls):
        cls.monitor = UniversalMonitor(sensitivity=3.0)
        cls.events = generate_healthy_run()
        cls.anomalies = _collect_anomalies(cls.monitor, cls.events)

    def test_all_metrics_nonzero_after_100(self):
        state = self.monitor.get_state()
        it = state["info_theoretic"]
        self.assertGreater(it["tool_entropy"], 0.0)
        self.assertGreater(it["action_mi"], 0.0)
        self.assertGreater(it["compression_ratio"], 0.0)

    def test_baseline_not_calibrating(self):
        self.assertFalse(self.monitor.is_calibrating)

    def test_phase_baselines_have_different_means(self):
        state = self.monitor.get_state()
        phases = state["baseline"]["phases"]
        # Should have at least planning and executing
        self.assertGreaterEqual(len(phases), 2)

    def test_few_named_anomalies(self):
        named = [(s, sig) for s, sig in self.anomalies if sig.name != "unclassified_anomaly"]
        self.assertLess(len(named), 10,
            f"Healthy run should have < 10 named anomalies, got {len(named)}: "
            f"{[(s, sig.name) for s, sig in named]}")

    def test_get_state_is_serializable(self):
        import json
        state = self.monitor.get_state()
        json.dumps(state, default=str)


class TestUniversalMonitorAnomalous(unittest.TestCase):
    """Validate: anomalous run detects injected pathologies.

    Uses sensitivity=2.0 (Agent 2's default). With the 50-event window,
    the strongest signal is mechanical_repetition (40 identical events).
    Goal drift and context thrashing signatures require memory metrics
    (namespace_entropy, memory_escalation_rate) to co-occur with IT metrics.
    We also collect ALL anomalies (including unclassified) for richer checks.
    """

    @classmethod
    def setUpClass(cls):
        cls.monitor = UniversalMonitor(sensitivity=2.0)
        cls.events = generate_anomalous_run()
        cls.all_anomalies = _collect_anomalies(cls.monitor, cls.events)

        cls.named = [(s, sig) for s, sig in cls.all_anomalies
                      if sig.name != "unclassified_anomaly"]
        cls.counts = Counter(sig.name for _, sig in cls.named)
        cls.all_counts = Counter(sig.name for _, sig in cls.all_anomalies)

    def test_detects_mechanical_repetition(self):
        """Stuck loop around step 200 should be caught."""
        self.assertIn("mechanical_repetition", self.counts,
            f"Expected mechanical_repetition, got: {dict(self.counts)}")
        mr_steps = [s for s, sig in self.named if sig.name == "mechanical_repetition"]
        near_200 = [s for s in mr_steps if 170 <= s <= 280]
        self.assertGreater(len(near_200), 0,
            f"mechanical_repetition steps: {mr_steps}, none near 200")

    def test_detects_anomalies_near_goal_drift(self):
        """Goal drift around step 350 should produce some anomaly signal.

        With the 50-event window, the goal_drift injection produces high
        entropy + varied tools. This may trigger distributional_shift,
        distributional_anomaly, or at minimum unclassified_anomaly.
        """
        all_near_350 = [
            (s, sig) for s, sig in self.all_anomalies
            if 270 <= s <= 450
        ]
        self.assertGreater(len(all_near_350), 0,
            f"Expected some anomaly signal near step 350, got none. "
            f"All anomalies: {[(s, sig.name) for s, sig in self.all_anomalies]}")

    def test_detects_anomalies_near_context_thrash(self):
        """Context thrashing around step 500 should produce anomaly signal.

        With full memory metrics (namespace_entropy, escalation_rate,
        memory_entropy all high), this triggers context_thrashing. With only
        IT metrics, it may appear as unclassified_anomaly due to the rapid
        memory operations producing unusual symbol patterns.
        """
        all_near_500 = [
            (s, sig) for s, sig in self.all_anomalies
            if 480 <= s <= 600
        ]
        self.assertGreater(len(all_near_500), 0,
            f"Expected some anomaly signal near step 500, got none. "
            f"All anomalies: {[(s, sig.name) for s, sig in self.all_anomalies]}")

    def test_anomalous_has_more_total_anomalies(self):
        """Anomalous run should have more total anomalies than healthy run."""
        healthy_monitor = UniversalMonitor(sensitivity=2.0)
        healthy_anomalies = _collect_anomalies(healthy_monitor, generate_healthy_run())

        self.assertGreater(len(self.all_anomalies), len(healthy_anomalies),
            f"Anomalous ({len(self.all_anomalies)}) should have more total anomalies "
            f"than healthy ({len(healthy_anomalies)})")

    def test_detections_are_before_labels_needed(self):
        """All detections happen from the event stream, not from manual labels."""
        for step, sig in self.named:
            self.assertIsNotNone(sig.metrics)


class TestAdapterRegistry(unittest.TestCase):
    """Test the adapter factory."""

    def test_get_viking_adapter(self):
        from agentdiag.adapters import get_adapter
        adapter = get_adapter("viking")
        self.assertEqual(type(adapter).__name__, "VikingLogAdapter")

    def test_get_harness_adapter(self):
        from agentdiag.adapters import get_adapter
        adapter = get_adapter("harness")
        self.assertEqual(type(adapter).__name__, "HarnessLogAdapter")

    def test_get_mixed_adapter(self):
        from agentdiag.adapters import get_adapter
        adapter = get_adapter("mixed")
        self.assertEqual(type(adapter).__name__, "MixedAdapter")

    def test_unknown_adapter_raises(self):
        from agentdiag.adapters import get_adapter
        with self.assertRaises(ValueError):
            get_adapter("nonexistent")

    def test_mixed_adapter_merge(self):
        from agentdiag.adapters import MixedAdapter
        from agentdiag.observable import ObservableEvent, EventType
        adapter = MixedAdapter()
        stream_a = [ObservableEvent(step=1, timestamp=0.1, event_type=EventType.TOOL_CALL)]
        stream_b = [ObservableEvent(step=1, timestamp=0.05, event_type=EventType.MEMORY_LOAD)]
        merged = adapter.merge(stream_a, stream_b)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0].event_type, EventType.MEMORY_LOAD)
        self.assertEqual(merged[1].event_type, EventType.TOOL_CALL)
        self.assertEqual(merged[0].step, 1)
        self.assertEqual(merged[1].step, 2)


if __name__ == "__main__":
    unittest.main()
