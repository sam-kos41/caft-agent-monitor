"""Tests for the validation logging and comparison system."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from agentdiag.validation_log import ValidationLog


class TestValidationLog(unittest.TestCase):
    """Test the validation log write/read cycle."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._log_path = os.path.join(self._tmpdir, "test_validation.jsonl")

    def test_session_lifecycle(self):
        vlog = ValidationLog(self._log_path)
        vlog.start_session(goal="Test goal", source="test")
        vlog.end_session(event_count=100)

        records = ValidationLog.load(self._log_path)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["type"], "session_start")
        self.assertEqual(records[0]["goal"], "Test goal")
        self.assertEqual(records[1]["type"], "session_end")
        self.assertEqual(records[1]["event_count"], 100)

    def test_human_marks(self):
        vlog = ValidationLog(self._log_path)
        vlog.start_session()
        vlog.update_position(step=10, event_count=10)
        vlog.mark_struggling()
        vlog.update_position(step=20, event_count=20)
        vlog.mark_fine()
        vlog.end_session()

        records = ValidationLog.load(self._log_path)
        marks = [r for r in records if r["type"] == "human_mark"]
        self.assertEqual(len(marks), 2)
        self.assertEqual(marks[0]["label"], "struggling")
        self.assertEqual(marks[0]["step"], 10)
        self.assertEqual(marks[1]["label"], "fine")
        self.assertEqual(marks[1]["step"], 20)
        self.assertEqual(vlog.human_marks_count, 2)

    def test_explicit_step(self):
        vlog = ValidationLog(self._log_path)
        vlog.start_session()
        vlog.mark_struggling(step=42, event_count=100)

        records = ValidationLog.load(self._log_path)
        marks = [r for r in records if r["type"] == "human_mark"]
        self.assertEqual(marks[0]["step"], 42)
        self.assertEqual(marks[0]["event_count"], 100)

    def test_system_detections(self):
        vlog = ValidationLog(self._log_path)
        vlog.start_session()
        vlog.record_detection(
            step=42,
            signature="mechanical_repetition",
            severity="warning",
            metrics={"action_entropy": 0.1},
        )
        vlog.end_session()

        records = ValidationLog.load(self._log_path)
        dets = [r for r in records if r["type"] == "system_detection"]
        self.assertEqual(len(dets), 1)
        self.assertEqual(dets[0]["signature"], "mechanical_repetition")
        self.assertEqual(dets[0]["step"], 42)
        self.assertEqual(vlog.detection_count, 1)


class TestCompareValidation(unittest.TestCase):
    """Test the agreement computation logic."""

    def test_perfect_agreement(self):
        """Human says struggling where system detects, fine elsewhere."""
        from scripts.compare_validation import compute_agreement

        records = [
            {"type": "session_start", "timestamp": 0.0},
            {"type": "human_mark", "timestamp": 1.0, "step": 10, "label": "struggling"},
            {"type": "system_detection", "timestamp": 1.0, "step": 12, "signature": "mechanical_repetition", "severity": "warning"},
            {"type": "human_mark", "timestamp": 2.0, "step": 50, "label": "fine"},
            {"type": "session_end", "timestamp": 3.0},
        ]

        result = compute_agreement(records, window=15)
        self.assertEqual(result["confusion_matrix"]["true_positive"], 1)
        self.assertEqual(result["confusion_matrix"]["true_negative"], 1)
        self.assertEqual(result["confusion_matrix"]["false_positive"], 0)
        self.assertEqual(result["confusion_matrix"]["false_negative"], 0)
        self.assertAlmostEqual(result["rates"]["agreement_rate"], 1.0)

    def test_false_negative(self):
        """Human says struggling but system misses it."""
        from scripts.compare_validation import compute_agreement

        records = [
            {"type": "human_mark", "timestamp": 1.0, "step": 10, "label": "struggling"},
            # No system detection nearby
        ]

        result = compute_agreement(records, window=15)
        self.assertEqual(result["confusion_matrix"]["false_negative"], 1)
        self.assertEqual(result["rates"]["recall"], 0.0)

    def test_false_positive(self):
        """Human says fine but system detects anomaly nearby."""
        from scripts.compare_validation import compute_agreement

        records = [
            {"type": "human_mark", "timestamp": 1.0, "step": 50, "label": "fine"},
            {"type": "system_detection", "timestamp": 1.0, "step": 48, "signature": "stagnation", "severity": "critical"},
        ]

        result = compute_agreement(records, window=15)
        self.assertEqual(result["confusion_matrix"]["false_positive"], 1)
        self.assertEqual(result["rates"]["precision"], 0.0)

    def test_window_boundary(self):
        """Detection just outside the window should not match."""
        from scripts.compare_validation import compute_agreement

        records = [
            {"type": "human_mark", "timestamp": 1.0, "step": 100, "label": "struggling"},
            {"type": "system_detection", "timestamp": 1.0, "step": 120, "signature": "mechanical_repetition", "severity": "warning"},
        ]

        # Window of 15: step 100 ± 15 = [85, 115]. Detection at 120 is outside.
        result = compute_agreement(records, window=15)
        self.assertEqual(result["confusion_matrix"]["false_negative"], 1)

        # Window of 25: step 100 ± 25 = [75, 125]. Detection at 120 is inside.
        result = compute_agreement(records, window=25)
        self.assertEqual(result["confusion_matrix"]["true_positive"], 1)

    def test_unclassified_ignored(self):
        """Unclassified anomalies should not count as detections for agreement."""
        from scripts.compare_validation import compute_agreement

        records = [
            {"type": "human_mark", "timestamp": 1.0, "step": 10, "label": "struggling"},
            {"type": "system_detection", "timestamp": 1.0, "step": 10, "signature": "unclassified_anomaly", "severity": "info"},
        ]

        result = compute_agreement(records, window=15)
        # unclassified should NOT count as a detection
        self.assertEqual(result["confusion_matrix"]["false_negative"], 1)

    def test_matched_signatures_tracked(self):
        """Per-signature breakdown should show which signatures matched."""
        from scripts.compare_validation import compute_agreement

        records = [
            {"type": "human_mark", "timestamp": 1.0, "step": 10, "label": "struggling"},
            {"type": "system_detection", "timestamp": 1.0, "step": 10, "signature": "mechanical_repetition", "severity": "warning"},
            {"type": "system_detection", "timestamp": 1.0, "step": 12, "signature": "stagnation", "severity": "critical"},
        ]

        result = compute_agreement(records, window=15)
        self.assertIn("mechanical_repetition", result["matched_signatures"])
        self.assertIn("stagnation", result["matched_signatures"])

    def test_empty_log(self):
        """Empty log should produce zero-everything."""
        from scripts.compare_validation import compute_agreement

        result = compute_agreement([], window=15)
        self.assertEqual(result["summary"]["human_marks"], 0)
        self.assertAlmostEqual(result["rates"]["agreement_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
