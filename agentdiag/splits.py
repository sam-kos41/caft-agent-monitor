"""Data split manager for evaluation integrity.

Enforces strict development / validation / test boundaries to prevent
data leakage. Every trace is assigned to exactly one split. The system
warns if you try to tune thresholds on test data.

Splits:
  DEVELOPMENT — Synthetic traces + 5 Claude Code traces (parser debugging)
  VALIDATION  — 15 Claude Code traces (threshold tuning) + MAST pilot (~100)
  TEST        — Remaining Claude Code + full MAST + full AgentRx + future

Usage:
    from agentdiag.splits import SplitManager

    sm = SplitManager("splits.json")
    sm.assign("session-abc", "development", source="claude-code")
    sm.auto_assign_claude_sessions("~/.claude/projects", n_dev=5, n_val=15)
    sm.check_leakage("session-abc", "test")  # warns if already in dev/val
"""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


VALID_SPLITS = {"development", "validation", "test"}
VALID_SOURCES = {"synthetic", "claude-code", "mast", "agentrx", "external"}


@dataclass
class TraceAssignment:
    """A single trace's split assignment with metadata."""
    trace_id: str
    split: str               # development, validation, test
    source: str              # synthetic, claude-code, mast, agentrx, external
    assigned_at: float = field(default_factory=time.time)
    assigned_by: str = "auto"
    reason: str = ""
    locked: bool = False     # locked assignments can't be changed

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TraceAssignment":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class SplitSummary:
    """Summary of current split assignments."""
    development: int = 0
    validation: int = 0
    test: int = 0
    unassigned: int = 0
    by_source: dict[str, dict[str, int]] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            f"Split Summary:",
            f"  Development: {self.development}",
            f"  Validation:  {self.validation}",
            f"  Test:        {self.test}",
            f"  Unassigned:  {self.unassigned}",
        ]
        if self.by_source:
            lines.append("  By source:")
            for source, splits in sorted(self.by_source.items()):
                parts = ", ".join(f"{s}={n}" for s, n in sorted(splits.items()))
                lines.append(f"    {source}: {parts}")
        return "\n".join(lines)


class SplitManager:
    """Manages trace split assignments with leakage prevention."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._assignments: dict[str, TraceAssignment] = {}
        self._access_log: list[dict] = []
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        with open(self.path) as f:
            data = json.load(f)

        if "assignments" in data:
            # Full format: {"assignments": [TraceAssignment, ...]}
            for d in data.get("assignments", []):
                a = TraceAssignment.from_dict(d)
                self._assignments[a.trace_id] = a
            self._access_log = data.get("access_log", [])
        else:
            # Simple format: {"train": [...], "val": [...], "test": [...]}
            # Keep split names as-is (train/val/test) for direct lookup
            for key, ids in data.items():
                if key == "metadata" or not isinstance(ids, list):
                    continue
                for trace_id in ids:
                    self._assignments[trace_id] = TraceAssignment(
                        trace_id=trace_id,
                        split=key,
                        source="unknown",
                        reason=f"Loaded from simple splits file ({key})",
                        assigned_by="splits_file",
                    )

    def _save(self) -> None:
        data = {
            "assignments": [a.to_dict() for a in self._assignments.values()],
            "access_log": self._access_log[-1000:],  # keep last 1000
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

    def assign(
        self,
        trace_id: str,
        split: str,
        source: str = "unknown",
        reason: str = "",
        assigned_by: str = "auto",
    ) -> None:
        """Assign a trace to a split."""
        if split not in VALID_SPLITS:
            raise ValueError(f"Invalid split '{split}'. Must be one of {VALID_SPLITS}")

        existing = self._assignments.get(trace_id)
        if existing and existing.locked:
            raise ValueError(
                f"Trace '{trace_id}' is locked in split '{existing.split}'. "
                f"Cannot reassign to '{split}'."
            )
        if existing and existing.split != split:
            warnings.warn(
                f"Trace '{trace_id}' reassigned from '{existing.split}' to '{split}'. "
                f"This may indicate data leakage.",
                UserWarning,
                stacklevel=2,
            )

        self._assignments[trace_id] = TraceAssignment(
            trace_id=trace_id,
            split=split,
            source=source,
            reason=reason,
            assigned_by=assigned_by,
        )
        self._save()

    def get_split(self, trace_id: str) -> Optional[str]:
        """Get the split for a trace. Returns None if unassigned."""
        a = self._assignments.get(trace_id)
        return a.split if a else None

    def get_traces(self, split: str) -> list[str]:
        """Get all trace IDs in a split."""
        return [a.trace_id for a in self._assignments.values() if a.split == split]

    def check_leakage(self, trace_id: str, intended_use: str) -> bool:
        """Check if using a trace for intended_use would cause leakage.

        Returns True if there's a leakage risk (and issues a warning).

        Leakage rules:
        - Development traces should not be used for validation metrics
        - Development/validation traces should NEVER be used for test metrics
        - Test traces should NEVER be used for threshold tuning
        """
        current = self.get_split(trace_id)
        if current is None:
            return False

        is_leak = False
        if intended_use == "test" and current in ("development", "validation"):
            warnings.warn(
                f"LEAKAGE: Trace '{trace_id}' is in '{current}' split but "
                f"you're trying to use it for testing. This invalidates results.",
                UserWarning,
                stacklevel=2,
            )
            is_leak = True
        elif intended_use == "threshold_tuning" and current == "test":
            warnings.warn(
                f"LEAKAGE: Trace '{trace_id}' is in 'test' split but "
                f"you're trying to use it for threshold tuning.",
                UserWarning,
                stacklevel=2,
            )
            is_leak = True

        self._access_log.append({
            "trace_id": trace_id,
            "intended_use": intended_use,
            "current_split": current,
            "is_leak": is_leak,
            "timestamp": time.time(),
        })
        self._save()
        return is_leak

    def lock(self, trace_id: str) -> None:
        """Lock a trace's assignment (prevent reassignment)."""
        a = self._assignments.get(trace_id)
        if a is None:
            raise ValueError(f"Trace '{trace_id}' not assigned to any split")
        a.locked = True
        self._save()

    def summary(self) -> SplitSummary:
        """Get a summary of current split assignments."""
        s = SplitSummary()
        for a in self._assignments.values():
            if a.split == "development":
                s.development += 1
            elif a.split == "validation":
                s.validation += 1
            elif a.split == "test":
                s.test += 1

            # By source
            if a.source not in s.by_source:
                s.by_source[a.source] = {}
            src = s.by_source[a.source]
            src[a.split] = src.get(a.split, 0) + 1

        return s

    def auto_assign_claude_sessions(
        self,
        traces_path: str | Path,
        n_dev: int = 5,
        n_val: int = 15,
    ) -> SplitSummary:
        """Auto-assign Claude Code sessions to splits.

        Discovers sessions, sorts by size (largest first), and assigns:
        - First n_dev → development
        - Next n_val → validation
        - Rest → test

        Larger sessions go to dev/val because they're more useful for
        debugging and tuning. Smaller sessions (potentially incomplete)
        go to test to avoid biasing results.
        """
        from agentdiag.adapters.claude_code import discover_sessions

        sessions = discover_sessions(traces_path)
        # Sort by line count descending (larger sessions first for dev/val)
        sessions.sort(key=lambda s: s.line_count, reverse=True)

        for i, session in enumerate(sessions):
            trace_id = session.session_id
            if trace_id in self._assignments:
                continue  # don't reassign existing

            if i < n_dev:
                split = "development"
                reason = f"Auto-assigned (top {n_dev} by size for debugging)"
            elif i < n_dev + n_val:
                split = "validation"
                reason = f"Auto-assigned (next {n_val} by size for tuning)"
            else:
                split = "test"
                reason = "Auto-assigned (remaining for held-out testing)"

            self._assignments[trace_id] = TraceAssignment(
                trace_id=trace_id,
                split=split,
                source="claude-code",
                reason=reason,
                assigned_by="auto_assign",
            )

        # Also assign all synthetic traces to development
        self._assignments["synthetic_all"] = TraceAssignment(
            trace_id="synthetic_all",
            split="development",
            source="synthetic",
            reason="All synthetic traces are development-only",
            assigned_by="auto_assign",
        )

        self._save()
        return self.summary()
