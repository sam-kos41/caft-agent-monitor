"""Unified detector registry.

Single registry for ALL detectors. All detectors are native CAFT format.
Supports optional calibration: if a baselines.json exists, detectors
are swapped with calibrated versions that use phase-specific percentile
thresholds.

Usage:
    from agentdiag.caft.registry import detector_registry

    # Get all detectors (calibrated if baselines available)
    all_dets = detector_registry.get_all()

    # Get by name
    det = detector_registry.get("step_repetition")

    # Manually load calibration
    detector_registry.load_calibration("baselines.json")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from agentdiag.caft.base import CaftDetector

logger = logging.getLogger(__name__)

# Default search paths for calibration baselines (tried in order)
DEFAULT_CALIBRATION_PATHS = [
    Path("baselines.json"),
    Path("~/.agentdiag/baselines.json").expanduser(),
]


class DetectorRegistry:
    """Unified registry for all detectors."""

    def __init__(self):
        self._detectors: dict[str, CaftDetector] = {}
        self._calibrated: bool = False

    @property
    def is_calibrated(self) -> bool:
        """True if calibrated detectors are loaded."""
        return self._calibrated

    def register(self, detector: CaftDetector) -> None:
        """Register a detector."""
        self._detectors[detector.name] = detector

    def unregister(self, name: str) -> None:
        """Remove a detector."""
        self._detectors.pop(name, None)

    def get(self, name: str) -> CaftDetector:
        """Get a detector by name. Raises KeyError if not found."""
        return self._detectors[name]

    def get_all(self) -> list[CaftDetector]:
        """All registered detectors."""
        return list(self._detectors.values())

    def get_enabled(self) -> list[CaftDetector]:
        """Get all enabled detectors (excludes disabled ones)."""
        return [
            d for d in self._detectors.values()
            if d.name not in self._disabled
        ]

    def names(self) -> list[str]:
        """All registered detector names."""
        return list(self._detectors.keys())

    def clear(self) -> None:
        """Remove all detectors."""
        self._detectors.clear()
        self._disabled.clear()
        self._calibrated = False

    def disable(self, name: str) -> None:
        """Disable a detector (keeps it registered but excluded from enabled)."""
        self._disabled.add(name)

    def enable(self, name: str) -> None:
        """Re-enable a previously disabled detector."""
        self._disabled.discard(name)

    def load_calibration(self, path: str | Path) -> bool:
        """Load calibration profile and swap in calibrated detectors.

        Returns True if calibration was loaded successfully.
        """
        path = Path(path)
        if not path.exists():
            return False

        try:
            from agentdiag.baselines import CalibrationProfile
            from agentdiag.caft.calibrated import make_calibrated_detectors

            profile = CalibrationProfile.load(path)
            calibrated = make_calibrated_detectors(profile)

            # Preserve disabled set across reload
            disabled = set(self._disabled)

            # Replace all detectors with calibrated versions
            self._detectors.clear()
            for det in calibrated:
                self._detectors[det.name] = det

            # Restore disabled set
            self._disabled_set = disabled
            self._calibrated = True

            logger.info(
                "Loaded calibration from %s (%d detectors, %d sessions)",
                path, len(calibrated), profile.n_sessions,
            )
            return True
        except Exception as e:
            logger.warning("Failed to load calibration from %s: %s", path, e)
            return False

    @property
    def _disabled(self) -> set:
        if not hasattr(self, "_disabled_set"):
            self._disabled_set: set[str] = set()
        return self._disabled_set

    def __len__(self) -> int:
        return len(self._detectors)

    def __contains__(self, name: str) -> bool:
        return name in self._detectors


def _build_default_registry() -> DetectorRegistry:
    """Build the default registry with all detectors.

    Tries to auto-load calibration from default paths. If no calibration
    file is found, falls back to raw (uncalibrated) detectors.
    """
    registry = DetectorRegistry()

    from agentdiag.caft.detectors import ALL_CAFT_DETECTORS_FULL

    for det in ALL_CAFT_DETECTORS_FULL:
        registry.register(det)

    # Disable detectors with high FP rates on real traces
    registry.disable("missing_verification")         # 14 FP on 20 real traces
    registry.disable("goal_drift")                    # 11 FP on 20 real traces
    registry.disable("tool_thrashing")               # 9+ FP on 20 real traces
    registry.disable("reasoning_action_mismatch")    # keyword matching too brittle
    registry.disable("strategic_myopia")              # V1 experimental — needs LLM

    # Try auto-loading calibration from default paths
    for path in DEFAULT_CALIBRATION_PATHS:
        if path.exists():
            registry.load_calibration(path)
            break

    return registry


# Module-level singleton
detector_registry = _build_default_registry()
