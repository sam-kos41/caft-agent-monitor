"""JSONL-backed rating store for the validation harness.

Schema (one JSON object per line):
    {
      "session_id": str,
      "rater_type": "human" | "ollama" | "caft",
      "rater_id":   str,             # username, "llama3.2:3b", "caft-v0.1", ...
      "dimension":  str,             # one of DIMENSIONS
      "value":      int | str | None,# 1-5 / label / None=abstain ("can't tell")
      "confidence": str,             # "" | "low" | "med" | "high"
      "reasoning":  str,             # free-text justification
      "timestamp":  str,             # ISO-8601
    }

A value of None (JSON null) means the rater explicitly abstained — the
session/dimension was not assessable from the evidence shown. Abstentions
are EXCLUDED from agreement statistics rather than guessed, so a
gold-standard rater is never forced to inject noise.

Append-only — re-rating writes a new row rather than overwriting, so the
full history is preserved. Readers de-duplicate by keeping the latest row
per (session, rater_type, rater_id, dimension).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional


RATER_TYPES = ("human", "ollama", "caft", "signal")
CONFIDENCE_LEVELS = ("", "low", "med", "high")


@dataclass
class Rating:
    session_id: str
    rater_type: str
    rater_id: str
    dimension: str
    value: int | str | None          # None == abstain ("can't tell")
    confidence: str = ""             # "", "low", "med", "high"
    reasoning: str = ""
    timestamp: str = ""

    def __post_init__(self):
        if self.rater_type not in RATER_TYPES:
            raise ValueError(
                f"rater_type must be one of {RATER_TYPES}, got {self.rater_type!r}"
            )
        if self.confidence not in CONFIDENCE_LEVELS:
            raise ValueError(
                f"confidence must be one of {CONFIDENCE_LEVELS}, "
                f"got {self.confidence!r}"
            )
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    @property
    def is_abstention(self) -> bool:
        return self.value is None


class Ledger:
    """Append-only JSONL ratings store with read-back deduplication."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def append(self, rating: Rating) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(rating), ensure_ascii=False) + "\n")

    def append_many(self, ratings: list[Rating]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            for r in ratings:
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    def all_rows(self) -> list[dict]:
        rows: list[dict] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows

    def latest(self) -> dict[tuple[str, str, str, str], dict]:
        """Return latest rating per (session, rater_type, rater_id, dimension).

        Iterates rows in file order; later rows override earlier ones.
        """
        latest: dict[tuple[str, str, str, str], dict] = {}
        for row in self.all_rows():
            key = (row["session_id"], row["rater_type"],
                   row["rater_id"], row["dimension"])
            latest[key] = row
        return latest

    def session_ratings(self, session_id: str) -> list[dict]:
        """All latest ratings for a session, across all raters and dims."""
        latest = self.latest()
        return [v for k, v in latest.items() if k[0] == session_id]

    def has_rating(self, session_id: str, rater_type: str,
                   rater_id: str, dimension: str) -> bool:
        return (session_id, rater_type, rater_id, dimension) in self.latest()

    def sessions_rated_by(self, rater_type: str,
                          rater_id: Optional[str] = None) -> set[str]:
        """Sessions that have at least one rating from this rater."""
        out: set[str] = set()
        for k in self.latest():
            sess, rt, rid, _dim = k
            if rt == rater_type and (rater_id is None or rid == rater_id):
                out.add(sess)
        return out

    def __iter__(self) -> Iterator[dict]:
        return iter(self.all_rows())
