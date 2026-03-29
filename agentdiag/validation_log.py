"""Validation logging — human judgment marks alongside system detections.

Records two parallel streams into a single JSONL file:
  1. **Human marks** — operator presses a key to annotate "struggling" or "fine"
  2. **System detections** — anomaly signatures from UniversalMonitor/compositor

After the session, ``scripts/compare_validation.py`` computes agreement rate
between human judgment and system detection using a temporal window.

The log is append-only JSONL. Each line is one of:

    {"type": "human_mark", "timestamp": ..., "step": ..., "label": "struggling", "event_count": ...}
    {"type": "human_mark", "timestamp": ..., "step": ..., "label": "fine", "event_count": ...}
    {"type": "system_detection", "timestamp": ..., "step": ..., "signature": "mechanical_repetition", "severity": "warning", "metrics": {...}}
    {"type": "session_start", "timestamp": ..., "goal": ..., "source": ...}
    {"type": "session_end", "timestamp": ..., "event_count": ..., "human_marks": ..., "system_detections": ...}

Usage::

    log = ValidationLog("validation_2026-03-28_14-30.jsonl")
    log.start_session(goal="Fix login bug", source="live")

    # From keyboard listener:
    log.mark_struggling(step=42, event_count=150)
    log.mark_fine(step=55, event_count=200)

    # From UniversalMonitor/compositor:
    log.record_detection(step=42, signature="mechanical_repetition",
                         severity="warning", metrics={...})

    log.end_session(event_count=500)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional


class ValidationLog:
    """Append-only JSONL log pairing human marks with system detections."""

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._human_count = 0
        self._detection_count = 0
        self._current_step = 0
        self._current_event_count = 0

    def _append(self, record: dict) -> None:
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    # -- Session lifecycle --

    def start_session(self, goal: str = "", source: str = "") -> None:
        self._append({
            "type": "session_start",
            "timestamp": time.time(),
            "goal": goal,
            "source": source,
        })

    def end_session(self, event_count: int = 0) -> None:
        self._append({
            "type": "session_end",
            "timestamp": time.time(),
            "event_count": event_count,
            "human_marks": self._human_count,
            "system_detections": self._detection_count,
        })

    # -- Human marks --

    def update_position(self, step: int, event_count: int) -> None:
        """Update the current step/event position (called on each event)."""
        self._current_step = step
        self._current_event_count = event_count

    def mark_struggling(
        self,
        step: Optional[int] = None,
        event_count: Optional[int] = None,
    ) -> None:
        """Record that the human observer thinks the agent is struggling."""
        self._human_count += 1
        self._append({
            "type": "human_mark",
            "timestamp": time.time(),
            "step": step if step is not None else self._current_step,
            "event_count": event_count if event_count is not None else self._current_event_count,
            "label": "struggling",
        })

    def mark_fine(
        self,
        step: Optional[int] = None,
        event_count: Optional[int] = None,
    ) -> None:
        """Record that the human observer thinks the agent is fine."""
        self._human_count += 1
        self._append({
            "type": "human_mark",
            "timestamp": time.time(),
            "step": step if step is not None else self._current_step,
            "event_count": event_count if event_count is not None else self._current_event_count,
            "label": "fine",
        })

    # -- System detections --

    def record_detection(
        self,
        step: int,
        signature: str,
        severity: str,
        metrics: Optional[dict] = None,
    ) -> None:
        """Record a system anomaly detection."""
        self._detection_count += 1
        self._append({
            "type": "system_detection",
            "timestamp": time.time(),
            "step": step,
            "signature": signature,
            "severity": severity,
            "metrics": metrics or {},
        })

    # -- Accessors --

    @property
    def path(self) -> Path:
        return self._path

    @property
    def human_marks_count(self) -> int:
        return self._human_count

    @property
    def detection_count(self) -> int:
        return self._detection_count

    @staticmethod
    def load(path: str) -> list[dict]:
        """Load all records from a validation log file."""
        records: list[dict] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return records
