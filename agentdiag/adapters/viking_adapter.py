"""OpenViking log replay adapter — maps stored access logs to ObservableEvent.

The InstrumentedContextStore emits events in real-time during live sessions.
This adapter serves the complementary purpose: replaying stored session data
from OpenViking's JSONL logs for offline analysis.

It reads the case ledger and session archives, reconstructing the event stream
that would have been emitted by the instrumented client during the original run.

Usage::

    from agentdiag.adapters.viking_adapter import VikingLogAdapter

    adapter = VikingLogAdapter()
    events = adapter.replay("./agentdiag_context")
    for event in events:
        monitor.push(event)  # feed into UniversalMonitor
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Iterator, Optional

from agentdiag.observable import (
    ObservableEvent,
    EventType,
    MemoryTier,
    AgentRole,
    memory_load_event,
    memory_store_event,
)

logger = logging.getLogger(__name__)


# Map case status values to semantic meaning for event reconstruction
_STATUS_TO_EVENT_TYPE = {
    "predicted": EventType.MEMORY_STORE,
    "confirmed": EventType.MEMORY_STORE,
    "false_positive": EventType.MEMORY_STORE,
    "corrected": EventType.MEMORY_STORE,
}


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _namespace_from_failure(failure_name: str) -> str:
    """Derive a viking:// namespace from a failure name."""
    return f"resources/current_project/cases"


class VikingLogAdapter:
    """Replays OpenViking session data as ObservableEvent streams.

    Reads from:
    - case_ledger.jsonl — promoted diagnostic cases
    - annotation_ledger.jsonl — annotation records
    - viking/ directory — session archives with .abstract.md/.overview.md

    Produces a chronologically ordered stream of ObservableEvent objects
    that reconstruct the memory operations from the original session.
    """

    def replay(
        self,
        db_path: str,
        session_filter: Optional[str] = None,
    ) -> list[ObservableEvent]:
        """Replay all stored events from an OpenViking context database.

        Args:
            db_path: Path to the agentdiag_context directory.
            session_filter: If provided, only replay events from sessions
                whose ID starts with this prefix.

        Returns:
            Chronologically sorted list of ObservableEvent.
        """
        root = Path(db_path)
        events: list[ObservableEvent] = []

        # Replay case ledger
        events.extend(self._replay_case_ledger(root, session_filter))

        # Replay annotation ledger
        events.extend(self._replay_annotation_ledger(root, session_filter))

        # Replay Viking filesystem summaries
        events.extend(self._replay_viking_archives(root, session_filter))

        # Sort by timestamp, then step
        events.sort(key=lambda e: (e.timestamp, e.step))

        # Re-number steps sequentially
        for i, event in enumerate(events, start=1):
            event.step = i

        return events

    def replay_iter(
        self,
        db_path: str,
        session_filter: Optional[str] = None,
    ) -> Iterator[ObservableEvent]:
        """Iterator version for large datasets."""
        yield from self.replay(db_path, session_filter)

    # ------------------------------------------------------------------
    # Case ledger replay
    # ------------------------------------------------------------------

    def _replay_case_ledger(
        self,
        root: Path,
        session_filter: Optional[str],
    ) -> list[ObservableEvent]:
        """Reconstruct MEMORY_STORE events from the case ledger."""
        ledger = root / "case_ledger.jsonl"
        if not ledger.exists():
            return []

        events: list[ObservableEvent] = []
        step = 0

        try:
            with open(ledger, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        case = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    sid = case.get("session_id", "")
                    if session_filter and not sid.startswith(session_filter):
                        continue

                    step += 1
                    failure_name = case.get("failure_name", "unknown")
                    timestamp = case.get("created_at", 0.0)
                    if isinstance(timestamp, str):
                        timestamp = 0.0

                    events.append(memory_store_event(
                        step=step,
                        timestamp=timestamp,
                        uri=f"viking://resources/current_project/cases/{failure_name}",
                        token_count=_estimate_tokens(json.dumps(case, default=str)),
                        namespace=_namespace_from_failure(failure_name),
                    ))

                    # If the case was reviewed (confirmed/FP), emit the
                    # retrospective skill/anti-skill store
                    status = case.get("status", "predicted")
                    if status == "confirmed":
                        step += 1
                        events.append(memory_store_event(
                            step=step,
                            timestamp=timestamp + 0.001,
                            uri=f"viking://agent/evaluator/skills/{failure_name}",
                            token_count=_estimate_tokens(json.dumps(case, default=str)),
                            namespace="agent/evaluator/skills",
                        ))
                    elif status == "false_positive":
                        step += 1
                        events.append(memory_store_event(
                            step=step,
                            timestamp=timestamp + 0.001,
                            uri=f"viking://agent/evaluator/memories/false_positives/{failure_name}",
                            token_count=_estimate_tokens(json.dumps(case, default=str)),
                            namespace="agent/evaluator/memories",
                        ))

        except Exception:
            logger.debug("Failed to replay case ledger", exc_info=True)

        return events

    # ------------------------------------------------------------------
    # Annotation ledger replay
    # ------------------------------------------------------------------

    def _replay_annotation_ledger(
        self,
        root: Path,
        session_filter: Optional[str],
    ) -> list[ObservableEvent]:
        """Reconstruct MEMORY_STORE events from the annotation ledger."""
        ledger = root / "annotation_ledger.jsonl"
        if not ledger.exists():
            return []

        events: list[ObservableEvent] = []
        step = 0

        try:
            with open(ledger, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ann = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    sid = ann.get("session_id", "")
                    if session_filter and not sid.startswith(session_filter):
                        continue

                    step += 1
                    ann_type = ann.get("annotator_type", "unknown")
                    timestamp = ann.get("created_at", 0.0)
                    if isinstance(timestamp, str):
                        timestamp = 0.0

                    events.append(memory_store_event(
                        step=step,
                        timestamp=timestamp,
                        uri=f"viking://resources/current_project/annotations/{ann_type}",
                        token_count=_estimate_tokens(json.dumps(ann, default=str)),
                        namespace="resources/current_project/annotations",
                    ))

        except Exception:
            logger.debug("Failed to replay annotation ledger", exc_info=True)

        return events

    # ------------------------------------------------------------------
    # Viking filesystem archive replay
    # ------------------------------------------------------------------

    def _replay_viking_archives(
        self,
        root: Path,
        session_filter: Optional[str],
    ) -> list[ObservableEvent]:
        """Reconstruct MEMORY_LOAD events from Viking .abstract.md and .overview.md files."""
        viking_dir = root / "viking"
        if not viking_dir.exists():
            return []

        events: list[ObservableEvent] = []
        step = 0

        for abstract in sorted(viking_dir.rglob(".abstract.md")):
            try:
                # Derive namespace from path relative to viking dir
                rel = abstract.parent.relative_to(viking_dir)
                parts = list(rel.parts)
                # Skip the "default" scope prefix if present
                if parts and parts[0] == "default":
                    parts = parts[1:]
                namespace = "/".join(parts[:3]) if parts else "root"

                content = abstract.read_text(encoding="utf-8")
                if not content.strip():
                    continue

                step += 1
                events.append(memory_load_event(
                    step=step,
                    timestamp=abstract.stat().st_mtime,
                    uri=f"viking://{'/'.join(parts)}",
                    tier=MemoryTier.L0,
                    token_count=_estimate_tokens(content),
                    namespace=namespace,
                ))

                # Check for overview (L1) at same level
                overview = abstract.parent / ".overview.md"
                if overview.exists():
                    overview_content = overview.read_text(encoding="utf-8")
                    if overview_content.strip():
                        step += 1
                        events.append(memory_load_event(
                            step=step,
                            timestamp=overview.stat().st_mtime,
                            uri=f"viking://{'/'.join(parts)}",
                            tier=MemoryTier.L1,
                            token_count=_estimate_tokens(overview_content),
                            namespace=namespace,
                        ))

            except Exception:
                logger.debug("Failed to replay archive %s", abstract, exc_info=True)

        return events
