"""JSONL-backed annotation ledger with lifecycle-aware filtering.

Provides AnnotationLedger — a persistent store for AnnotationRecords
with deduplication, conflict resolution, and status-based filtering.

The ledger preserves all annotation layers (detector, auto, human,
adjudicated) as separate records. Deduplication is by
(session_id, annotator_type, annotator_id, caft_version).

Filtering helpers enforce the rule that only adjudicated/reviewed
labels are trusted for calibration and final evaluation.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from agentdiag.annotation_models import (
    AnnotationRecord,
    AnnotatorType,
    LabelStatus,
)


class AnnotationLedger:
    """JSONL-backed storage for AnnotationRecords.

    Supports:
      - append with dedup by (session_id, annotator_type, annotator_id, version)
      - filter by label_status, annotator_type, session_id
      - gold/trainable/eval subset retrieval
      - merge from another ledger or record list
      - prefix-aware session resolution (short 8-char IDs ↔ full UUIDs)
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._records: list[AnnotationRecord] = []
        self._index: dict[tuple, int] = {}  # dedup_key → index in _records
        self._prefix_map: dict[str, set[str]] = {}  # 8-char prefix → {full session IDs}
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    rec = AnnotationRecord.from_dict(d)
                    self._insert(rec, persist=False)
                except (json.JSONDecodeError, TypeError):
                    continue

    def _register_prefix(self, session_id: str) -> None:
        """Track 8-char prefix → full session ID mapping."""
        prefix = session_id[:8]
        if prefix not in self._prefix_map:
            self._prefix_map[prefix] = set()
        self._prefix_map[prefix].add(session_id)

    def _resolve_session_ids(self, session_id: str) -> set[str]:
        """Resolve a session ID to all known aliases (short prefix ↔ full UUID).

        This ensures that ``e2eff792`` and ``e2eff792-fb22-4f3c-...`` are
        treated as the same physical session.
        """
        prefix = session_id[:8]
        aliases = self._prefix_map.get(prefix, set())
        return aliases | {session_id}

    def _insert(self, rec: AnnotationRecord, persist: bool = True) -> bool:
        """Insert or update a record. Returns True if new, False if updated."""
        self._register_prefix(rec.effective_session_id)
        key = rec.dedup_key
        if key in self._index:
            idx = self._index[key]
            existing = self._records[idx]
            # Newer record replaces older (by updated_at)
            if rec.updated_at >= existing.updated_at:
                self._records[idx] = rec
                if persist:
                    self._rewrite()
                return False
            return False
        else:
            self._index[key] = len(self._records)
            self._records.append(rec)
            if persist:
                self._append_line(rec)
            return True

    def _append_line(self, rec: AnnotationRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(rec.to_json() + "\n")

    def _rewrite(self) -> None:
        """Rewrite the full ledger (for dedup updates)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            for rec in self._records:
                f.write(rec.to_json() + "\n")

    # ── Public API ──────────────────────────────────────────────

    def add(self, record: AnnotationRecord) -> bool:
        """Add a record. Deduplicates by key. Returns True if new."""
        record.updated_at = time.time()
        return self._insert(record)

    def add_many(self, records: list[AnnotationRecord]) -> int:
        """Add multiple records. Returns count of new insertions."""
        count = 0
        for rec in records:
            rec.updated_at = time.time()
            if self._insert(rec, persist=False):
                count += 1
        self._rewrite()
        return count

    def get_all(self) -> list[AnnotationRecord]:
        return list(self._records)

    def get_for_session(self, session_id: str) -> list[AnnotationRecord]:
        """Get all annotation layers for a session.

        Uses prefix-aware resolution so short IDs (``e2eff792``) and full
        UUIDs (``e2eff792-fb22-4f3c-...``) are treated as the same session.
        """
        aliases = self._resolve_session_ids(session_id)
        return [r for r in self._records
                if r.session_id in aliases or r.trace_id in aliases]

    def get_by_type(self, annotator_type: str) -> list[AnnotationRecord]:
        return [r for r in self._records if r.annotator_type == annotator_type]

    def get_by_status(self, label_status: str) -> list[AnnotationRecord]:
        return [r for r in self._records if r.label_status == label_status]

    def get_sessions(self) -> set[str]:
        """All unique session IDs in the ledger (canonical form).

        When both short and full UUID exist for the same session,
        returns the longest (canonical) ID only.
        """
        canonical: set[str] = set()
        for group in self._prefix_map.values():
            canonical.add(max(group, key=len))
        return canonical

    # ── Lifecycle-aware filters ─────────────────────────────────

    def get_gold_annotations(self) -> list[AnnotationRecord]:
        """Get only adjudicated labels — the only trusted gold standard.

        These are the ONLY records that should be used for:
          - final performance metrics
          - published evaluation results
          - threshold decisions
        """
        return [r for r in self._records
                if r.label_status == LabelStatus.ADJUDICATED.value]

    def get_trainable_annotations(self) -> list[AnnotationRecord]:
        """Get annotations suitable for threshold tuning.

        Includes adjudicated + human-reviewed labels but excludes:
          - held_out_test (reserved for eval)
          - unlabeled (no signal)
          - auto_labeled only (not yet verified)
        """
        trainable = {LabelStatus.ADJUDICATED.value, LabelStatus.HUMAN_REVIEWED.value}
        return [r for r in self._records if r.label_status in trainable]

    def get_eval_annotations(self) -> list[AnnotationRecord]:
        """Get annotations suitable for evaluation.

        Includes adjudicated + held_out_test labels. Auto-labeled records
        are included only for exploratory analysis when flagged.
        """
        eval_statuses = {LabelStatus.ADJUDICATED.value, LabelStatus.HELD_OUT_TEST.value}
        return [r for r in self._records if r.label_status in eval_statuses]

    def get_unlabeled_sessions(self) -> set[str]:
        """Sessions that have no annotation beyond detector predictions."""
        labeled = set()
        non_detector = {AnnotatorType.AUTO.value, AnnotatorType.HUMAN.value,
                        AnnotatorType.ADJUDICATED.value}
        for r in self._records:
            if r.annotator_type in non_detector:
                labeled.add(r.effective_session_id)
        return self.get_sessions() - labeled

    def get_best_label(self, session_id: str) -> Optional[AnnotationRecord]:
        """Get the highest-trust annotation for a session.

        Trust hierarchy: adjudicated > human > auto > detector.
        Returns None if no records exist.
        """
        records = self.get_for_session(session_id)
        if not records:
            return None

        priority = {
            AnnotatorType.ADJUDICATED.value: 4,
            AnnotatorType.HUMAN.value: 3,
            AnnotatorType.AUTO.value: 2,
            AnnotatorType.DETECTOR.value: 1,
        }
        records.sort(key=lambda r: priority.get(r.annotator_type, 0), reverse=True)
        return records[0]

    # ── Merge ───────────────────────────────────────────────────

    def merge_from(self, other: "AnnotationLedger") -> int:
        """Merge records from another ledger. Returns count of new records."""
        return self.add_many(other.get_all())

    def merge_records(self, records: list[AnnotationRecord]) -> int:
        """Merge a list of records. Returns count of new records."""
        return self.add_many(records)

    # ── Stats ───────────────────────────────────────────────────

    def stats(self) -> dict:
        """Summary statistics."""
        by_type: dict[str, int] = {}
        by_status: dict[str, int] = {}
        by_failure: dict[str, int] = {}
        sessions = set()

        for r in self._records:
            by_type[r.annotator_type] = by_type.get(r.annotator_type, 0) + 1
            by_status[r.label_status] = by_status.get(r.label_status, 0) + 1
            if r.has_failure and r.primary_caft_code:
                name = r.primary_caft_name or r.primary_caft_code
                by_failure[name] = by_failure.get(name, 0) + 1
            sessions.add(r.effective_session_id)

        return {
            "total_records": len(self._records),
            "unique_sessions": len(sessions),
            "by_annotator_type": by_type,
            "by_label_status": by_status,
            "by_failure_type": by_failure,
            "gold_count": len(self.get_gold_annotations()),
            "trainable_count": len(self.get_trainable_annotations()),
        }

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return f"AnnotationLedger({self.path}, {len(self._records)} records)"
