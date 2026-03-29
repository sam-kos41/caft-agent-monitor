"""OpenViking-backed persistent diagnostic context.

Architecture
------------
Two-tier persistence model:

1. **In-memory event buffer** — every ``record_event()`` call appends to a
   lightweight in-memory list.  Zero I/O in the hot path.

2. **Promoted case memories** — only *meaningful* events are written to
   OpenViking as structured ``DiagnosticCase`` records:
   - First diagnosis of each CAFT type
   - Critical-severity diagnoses
   - Session-end summary (always)

3. **Case ledger** — local JSONL file tracking all promoted cases with their
   human-reviewed status (PREDICTED, CONFIRMED, FALSE_POSITIVE, CORRECTED).
   This enables the feedback loop: detectors whose cases are frequently
   marked FALSE_POSITIVE get their confidence discounted in future runs.

Pattern memories (cross-session aggregates) are NOT auto-derived.
They are a future offline concern — see ``get_failure_patterns()`` docstring.

All public methods are wrapped in try/except so that OpenViking failures
never crash the diagnostic pipeline.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional

from openviking import SyncOpenViking

from agentdiag.models import TraceEvent
from agentdiag.hta import Phase
from agentdiag.caft.base import CaftDiagnosis, CaftSeverity

logger = logging.getLogger(__name__)

_SEVERITY_RANK = {
    CaftSeverity.INFO: 0,
    CaftSeverity.WARNING: 1,
    CaftSeverity.CRITICAL: 2,
}


# ======================================================================
# Structured diagnostic case schema (Points 2 + 5)
# ======================================================================

class CaseStatus(Enum):
    """Lifecycle status of a diagnostic case."""
    PREDICTED = "predicted"          # raw detector output, unverified
    CONFIRMED = "confirmed"          # human-validated as correct
    CORRECTED = "corrected"          # human corrected the diagnosis
    FALSE_POSITIVE = "false_positive"  # human rejected as incorrect


@dataclass
class DiagnosticCase:
    """A single promoted diagnostic case with structured fields.

    This is what gets written to OpenViking — not raw events or free text.
    Each case represents a meaningful diagnostic finding that is worth
    remembering across sessions.
    """
    # Identity
    case_id: str                      # unique ID (session_id + step)
    session_id: str

    # Source context
    source: str = ""                  # trace filename or origin
    trace_length: int = 0             # total events in session so far

    # HTA context
    phase_at_onset: str = ""          # HTA phase when diagnosed

    # CAFT classification
    caft_code: str = ""               # e.g., "2.2"
    caft_category: str = ""           # e.g., "memory"
    failure_name: str = ""            # e.g., "step_repetition"
    severity: str = ""                # info / warning / critical
    confidence: float = 0.0
    onset_step: int = 0

    # Content
    description: str = ""             # human-readable explanation
    evidence_summary: str = ""        # condensed evidence dict
    remediation: str = ""             # suggested fix

    # Trust trajectory
    trust_at_onset: float = 1.0

    # Confirmation status (Point 5)
    status: str = CaseStatus.PREDICTED.value
    reviewer: str = ""                # "human" or "system"
    resolution_notes: str = ""

    # Timestamp
    created_at: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

    def to_summary_text(self) -> str:
        """One-line summary for L0 display."""
        return (
            f"[{self.caft_code}] {self.failure_name} ({self.severity}) "
            f"at step {self.onset_step} in {self.phase_at_onset} phase — "
            f"{self.description}"
        )

    @classmethod
    def from_diagnosis(
        cls,
        diagnosis: CaftDiagnosis,
        session_id: str,
        source: str = "",
        trace_length: int = 0,
        phase: str = "",
        trust: float = 1.0,
    ) -> "DiagnosticCase":
        """Create a case from a CAFT diagnosis."""
        evidence_str = json.dumps(diagnosis.evidence, default=str)
        if len(evidence_str) > 500:
            evidence_str = evidence_str[:497] + "..."

        return cls(
            case_id=f"{session_id}_{diagnosis.at_step}_{diagnosis.failure_name}",
            session_id=session_id,
            source=source,
            trace_length=trace_length,
            phase_at_onset=phase,
            caft_code=diagnosis.caft_code,
            caft_category=diagnosis.caft_category,
            failure_name=diagnosis.failure_name,
            severity=diagnosis.severity.value,
            confidence=diagnosis.confidence,
            onset_step=diagnosis.at_step,
            description=diagnosis.description,
            evidence_summary=evidence_str,
            remediation=diagnosis.remediation,
            trust_at_onset=trust,
            status=CaseStatus.PREDICTED.value,
            created_at=time.time(),
        )


# ======================================================================
# Buffered event record (Point 1 — in-memory only, never persisted)
# ======================================================================

@dataclass
class _BufferedEvent:
    """Lightweight in-memory event record. Never written to OpenViking."""
    step: int
    tool: str
    phase: str
    success: bool
    latency_ms: float
    has_diagnosis: bool


# ======================================================================
# OpenViking config helpers
# ======================================================================

def _build_ov_config(db_path: str) -> dict:
    """Build a minimal OpenViking config dict for local operation.

    Detects available API keys from environment variables and configures
    the embedding provider accordingly.
    """
    workspace = str(Path(db_path).resolve())

    config: dict = {
        "storage": {
            "workspace": workspace,
            "agfs": {
                "mode": "binding-client",
                "backend": "local",
            },
            "vectordb": {
                "backend": "local",
            },
        },
        "vlm": {},
        "rerank": {},
    }

    openai_key = os.environ.get("OPENAI_API_KEY", "")
    jina_key = os.environ.get("JINA_API_KEY", "")

    if openai_key:
        config["embedding"] = {
            "dense": {
                "provider": "openai",
                "model": "text-embedding-3-small",
                "api_key": openai_key,
                "dimension": 1536,
            }
        }
    elif jina_key:
        config["embedding"] = {
            "dense": {
                "provider": "jina",
                "model": "jina-embeddings-v3",
                "api_key": jina_key,
                "dimension": 1024,
            }
        }
    else:
        # Placeholder — search won't work but session recording will.
        config["embedding"] = {
            "dense": {
                "provider": "openai",
                "model": "text-embedding-3-small",
                "api_key": "sk-placeholder-no-search",
                "dimension": 1536,
            }
        }

    return config


def _has_real_embedding_key() -> bool:
    """Check if a real embedding API key is available."""
    return bool(
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("JINA_API_KEY")
    )


def _ensure_ov_config(db_path: str) -> None:
    """Create a minimal OpenViking config if none exists."""
    if os.environ.get("OPENVIKING_CONFIG_FILE"):
        return

    candidates = [
        Path.home() / ".openviking" / "ov.conf",
        Path("/etc/openviking/ov.conf"),
    ]
    if any(p.exists() for p in candidates):
        return

    conf_dir = Path.home() / ".openviking"
    conf_dir.mkdir(parents=True, exist_ok=True)
    conf_path = conf_dir / "ov.conf"
    config = _build_ov_config(db_path)
    conf_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    logger.debug("Created minimal OpenViking config at %s", conf_path)


# ======================================================================
# ContextStore
# ======================================================================

class ContextStore:
    """Persistent diagnostic context backed by OpenViking.

    Uses a two-tier persistence model:

    - **Hot path** (``record_event``): appends to an in-memory buffer.
      Zero I/O per event.
    - **Promotion** (``_promote_case``): only writes to OpenViking when a
      meaningful diagnosis is detected.  Called automatically on first
      diagnosis of each CAFT type and on critical-severity findings.
    - **Commit** (``end_session``): flushes promoted cases + session
      summary to OpenViking and triggers memory extraction.

    All public methods silently catch exceptions so that context failures
    never disrupt the diagnostic pipeline.
    """

    def __init__(self, db_path: str = "./agentdiag_context"):
        self._db_path = db_path
        _ensure_ov_config(db_path)
        self._client = SyncOpenViking(path=db_path)
        self._client.initialize()

        # Session state
        self._session_id: Optional[str] = None
        self._session_goal: str = ""
        self._session_source: str = ""

        # In-memory buffers (Point 1+6 — zero I/O in the hot path)
        self._event_buffer: list[_BufferedEvent] = []
        self._promoted_cases: list[DiagnosticCase] = []
        self._seen_case_types: set[str] = set()  # track first-of-type

        # Case ledger — local JSONL file for feedback loop
        self._ledger_path = Path(db_path) / "case_ledger.jsonl"
        self._ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self._fp_rate_cache: Optional[dict[str, float]] = None

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def start_session(self, goal: str = "", source: str = "") -> str:
        """Start a new diagnostic session.

        Args:
            goal: Goal text for the monitored agent session.
            source: Source identifier (e.g., trace filename).

        Returns:
            The new session ID, or empty string on failure.
        """
        try:
            result = self._client.create_session()
            self._session_id = result["session_id"]
            self._session_goal = goal
            self._session_source = source

            # Reset buffers
            self._event_buffer.clear()
            self._promoted_cases.clear()
            self._seen_case_types.clear()

            # Record session metadata as the first message
            meta = json.dumps({
                "type": "session_start",
                "goal": goal,
                "source": source,
            })
            self._client.add_message(
                session_id=self._session_id,
                role="user",
                content=meta,
            )
            logger.debug("Context session started: %s", self._session_id)
            return self._session_id
        except Exception:
            logger.debug("Failed to start context session", exc_info=True)
            self._session_id = None
            return ""

    def record_event(
        self,
        event: TraceEvent,
        phase: Phase,
        diagnoses: list[CaftDiagnosis],
        trust_score: float = 1.0,
    ) -> None:
        """Buffer a trace event in memory. Zero I/O.

        Only promotes to OpenViking when meaningful diagnoses are present:
        - First occurrence of each CAFT failure type
        - Any critical-severity diagnosis

        Args:
            event: The trace event to record.
            phase: Current HTA phase when this event occurred.
            diagnoses: New CAFT diagnoses detected on this event (may be empty).
            trust_score: Current trust score at time of event.
        """
        if self._session_id is None:
            return

        # Always buffer in memory (cheap)
        self._event_buffer.append(_BufferedEvent(
            step=event.step,
            tool=event.tool or event.type,
            phase=phase.label,
            success=event.success,
            latency_ms=event.latency_ms,
            has_diagnosis=bool(diagnoses),
        ))

        # Promote meaningful diagnoses to case memory
        for d in diagnoses:
            is_first_of_type = d.failure_name not in self._seen_case_types
            is_critical = d.severity == CaftSeverity.CRITICAL

            if is_first_of_type or is_critical:
                self._seen_case_types.add(d.failure_name)
                case = DiagnosticCase.from_diagnosis(
                    diagnosis=d,
                    session_id=self._session_id,
                    source=self._session_source,
                    trace_length=len(self._event_buffer),
                    phase=phase.label,
                    trust=trust_score,
                )
                self._promoted_cases.append(case)
                self._promote_case(case)

    def _promote_case(self, case: DiagnosticCase) -> None:
        """Write a single promoted case to OpenViking and the local ledger."""
        if self._session_id is None:
            return

        try:
            # Store as structured JSON, not free text (Point 2)
            self._client.add_message(
                session_id=self._session_id,
                role="assistant",
                parts=[
                    {
                        "type": "tool",
                        "tool_name": f"caft_{case.failure_name}",
                        "tool_id": case.case_id,
                        "tool_status": case.severity,
                        "tool_output": case.to_json(),
                        "duration_ms": 0,
                    },
                    {
                        "type": "text",
                        "text": case.to_summary_text(),
                    },
                ],
            )
        except Exception:
            logger.debug(
                "Failed to promote case %s to OpenViking", case.case_id,
                exc_info=True,
            )

        # Also write to local case ledger (for feedback loop)
        self._append_to_ledger(case)
        self._fp_rate_cache = None  # invalidate cache

    def _append_to_ledger(self, case: DiagnosticCase) -> None:
        """Append a case record to the local JSONL ledger."""
        try:
            with open(self._ledger_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(case.to_dict(), default=str) + "\n")
        except Exception:
            logger.debug("Failed to write to case ledger", exc_info=True)

    def end_session(self, dashboard_state: "DashboardState") -> dict:
        """Commit the session with a diagnostic summary.

        Writes the session summary (L0/L1/L2) and commits, triggering
        OpenViking's archival and memory extraction.

        Args:
            dashboard_state: Final dashboard state from MonitorEngine.

        Returns:
            Commit result dict, or empty dict on failure.
        """
        if self._session_id is None:
            return {}

        try:
            # Build L0 (abstract)
            diag_names = [c.failure_name for c in self._promoted_cases]
            diag_summary = ", ".join(diag_names) if diag_names else "none"
            n_buffered = len(self._event_buffer)
            n_promoted = len(self._promoted_cases)

            l0 = (
                f"Diagnostic session: {n_buffered} events, "
                f"{n_promoted} promoted cases ({diag_summary}), "
                f"trust={dashboard_state.trust_score:.2f} {dashboard_state.health}"
            )

            # Build L1 (overview)
            l1_lines = [
                f"Events: {dashboard_state.total_events} "
                f"(errors: {dashboard_state.total_errors}, "
                f"buffered: {n_buffered}, promoted: {n_promoted})",
                f"Phase: {dashboard_state.hta_state.current_phase.label if dashboard_state.hta_state else 'unknown'}",
                f"Trust: {dashboard_state.trust_score:.2f} ({dashboard_state.health})",
            ]
            if self._promoted_cases:
                l1_lines.append("Promoted cases:")
                for c in self._promoted_cases:
                    l1_lines.append(f"  {c.to_summary_text()}")
            l1 = "\n".join(l1_lines)

            # Build L2 (structured detail)
            l2_cases = [c.to_dict() for c in self._promoted_cases]
            l2 = json.dumps(l2_cases, indent=2, default=str) if l2_cases else "[]"

            summary_text = f"## L0\n{l0}\n\n## L1\n{l1}\n\n## L2\n{l2}"

            # Write summary as final message
            self._client.add_message(
                session_id=self._session_id,
                role="assistant",
                content=summary_text,
            )

            # Commit the session
            result = self._client.commit_session(self._session_id)
            logger.debug("Context session committed: %s", self._session_id)
            return result
        except Exception:
            logger.debug("Failed to end context session", exc_info=True)
            return {}
        finally:
            self._session_id = None
            self._event_buffer.clear()
            self._promoted_cases.clear()
            self._seen_case_types.clear()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def find_similar_failures(
        self,
        diagnosis: CaftDiagnosis,
        limit: int = 5,
    ) -> list[dict]:
        """Semantic search for past sessions with similar failure patterns."""
        if not _has_real_embedding_key():
            return []
        try:
            query = (
                f"{diagnosis.failure_name}: {diagnosis.description} "
                f"(severity={diagnosis.severity.value})"
            )
            results = self._client.search(
                query=query,
                target_uri="viking://",
                limit=limit,
            )
            return self._normalize_results(results)
        except Exception:
            logger.debug("Failed to search for similar failures", exc_info=True)
            return []

    def find_sessions_by_tool(self, tool_name: str, limit: int = 10) -> list[dict]:
        """Find past sessions that used a specific tool."""
        if not _has_real_embedding_key():
            return []
        try:
            results = self._client.search(
                query=f"tool_name: {tool_name}",
                target_uri="viking://",
                limit=limit,
            )
            return self._normalize_results(results)
        except Exception:
            logger.debug("Failed to search by tool %s", tool_name, exc_info=True)
            return []

    def search(self, query: str, limit: int = 10) -> list[dict]:
        """General semantic search across all stored context.

        Requires a valid embedding API key (OPENAI_API_KEY or JINA_API_KEY).
        Returns empty list with a logged warning if no key is available.
        """
        if not _has_real_embedding_key():
            logger.debug("Search skipped: no embedding API key available")
            return []
        try:
            results = self._client.search(
                query=query,
                target_uri="viking://",
                limit=limit,
            )
            return self._normalize_results(results)
        except Exception:
            logger.debug("Failed to search: %s", query, exc_info=True)
            return []

    @staticmethod
    def _normalize_results(results) -> list[dict]:
        if isinstance(results, list):
            return results
        if isinstance(results, dict):
            return results.get("results", [])
        return []

    # ------------------------------------------------------------------
    # Patterns (Point 4 — explicit boundary, future offline concern)
    # ------------------------------------------------------------------

    def get_failure_patterns(self) -> list[dict]:
        """Get accumulated failure patterns across all sessions.

        **Note:** Patterns are *derived aggregates* built from multiple
        cases (e.g., "23% of sessions show step_repetition during file
        search").  They are NOT the same as individual cases.

        Pattern generation is a future offline process:
        - Batch aggregation over committed cases
        - Threshold-based clustering
        - Manual promotion

        For now, this method searches for any memories that OpenViking's
        auto-extraction may have created during session commits.
        """
        try:
            results = self._client.search(
                query="CAFT failure pattern diagnostic",
                target_uri="viking://",
                limit=20,
            )
            return self._normalize_results(results)
        except Exception:
            logger.debug("Failed to get failure patterns", exc_info=True)
            return []

    # ------------------------------------------------------------------
    # Agent registration (Point 3 — internal API, not CLI-exposed)
    # ------------------------------------------------------------------

    def register_agents(self, agents_dir: str) -> int:
        """Bulk-register agent definitions as OpenViking Skills.

        .. note::

            This is an internal API kept for future use (expected-vs-observed
            behavior, per-agent failure priors, remediation grounded in agent
            definitions).  It is intentionally NOT exposed in the CLI.

        Args:
            agents_dir: Path to directory containing agent ``.md`` files.

        Returns:
            Number of agents successfully registered.
        """
        agents_path = Path(agents_dir).expanduser()
        if not agents_path.is_dir():
            logger.warning("Agents directory not found: %s", agents_path)
            return 0

        registered = 0
        for md_file in sorted(agents_path.glob("*.md")):
            try:
                content = md_file.read_text(encoding="utf-8")
                name = md_file.stem
                skill_data = {
                    "name": name,
                    "description": content[:500],
                    "content": content,
                }
                self._client.add_skill(data=skill_data)
                registered += 1
                logger.debug("Registered agent skill: %s", name)
            except Exception:
                logger.debug(
                    "Failed to register agent %s", md_file.name, exc_info=True
                )

        return registered

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Get context database statistics."""
        try:
            sessions = self._client.list_sessions()
            session_count = len(sessions) if isinstance(sessions, list) else 0

            status = self._client.get_status()
            healthy = True
            errors: list[str] = []
            if status is not None:
                if hasattr(status, "is_healthy"):
                    healthy = status.is_healthy
                if hasattr(status, "errors") and status.errors:
                    errors = [str(e) for e in status.errors]

            return {
                "db_path": self._db_path,
                "session_count": session_count,
                "active_session": self._session_id,
                "buffered_events": len(self._event_buffer),
                "promoted_cases": len(self._promoted_cases),
                "search_available": _has_real_embedding_key(),
                "healthy": healthy,
                "errors": errors,
            }
        except Exception:
            logger.debug("Failed to get stats", exc_info=True)
            return {
                "db_path": self._db_path,
                "session_count": 0,
                "active_session": self._session_id,
                "buffered_events": len(self._event_buffer),
                "promoted_cases": len(self._promoted_cases),
                "search_available": _has_real_embedding_key(),
                "healthy": False,
                "errors": ["Failed to connect to context database"],
            }

    # ------------------------------------------------------------------
    # Feedback loop — case status updates and FP rate computation
    # ------------------------------------------------------------------

    def load_cases(self, status_filter: Optional[str] = None) -> list[dict]:
        """Load all cases from the local JSONL ledger.

        Args:
            status_filter: If set, only return cases with this status
                (e.g., "predicted", "false_positive", "confirmed").

        Returns:
            List of case dicts, newest first.
        """
        cases: list[dict] = []
        if not self._ledger_path.exists():
            return cases

        try:
            with open(self._ledger_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        case = json.loads(line)
                        if status_filter and case.get("status") != status_filter:
                            continue
                        cases.append(case)
                    except json.JSONDecodeError:
                        continue
        except Exception:
            logger.debug("Failed to load case ledger", exc_info=True)

        cases.reverse()  # newest first
        return cases

    def update_case_status(
        self,
        case_id: str,
        new_status: str,
        reviewer: str = "human",
        notes: str = "",
    ) -> bool:
        """Update a case's status in the ledger.

        Rewrites the ledger with the updated status. This is safe because
        ledgers are small (hundreds of entries, not millions).

        Args:
            case_id: The case_id to update.
            new_status: New status value (use CaseStatus enum values).
            reviewer: Who made this determination.
            notes: Optional resolution notes.

        Returns:
            True if the case was found and updated.
        """
        valid_statuses = {s.value for s in CaseStatus}
        if new_status not in valid_statuses:
            logger.warning("Invalid status '%s', valid: %s", new_status, valid_statuses)
            return False

        if not self._ledger_path.exists():
            return False

        try:
            lines: list[str] = []
            found = False
            with open(self._ledger_path, encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if not stripped:
                        lines.append(line)
                        continue
                    try:
                        case = json.loads(stripped)
                        if case.get("case_id") == case_id:
                            case["status"] = new_status
                            case["reviewer"] = reviewer
                            case["resolution_notes"] = notes
                            found = True
                        lines.append(json.dumps(case, default=str) + "\n")
                    except json.JSONDecodeError:
                        lines.append(line)

            if found:
                with open(self._ledger_path, "w", encoding="utf-8") as f:
                    f.writelines(lines)
                self._fp_rate_cache = None  # invalidate
                logger.debug("Updated case %s to %s", case_id, new_status)

            return found
        except Exception:
            logger.debug("Failed to update case status", exc_info=True)
            return False

    def get_detector_fp_rates(self) -> dict[str, float]:
        """Compute false positive rate per detector from reviewed cases.

        Only considers cases that have been reviewed (status != "predicted").
        Rate = false_positive_count / total_reviewed_count per detector.

        Returns:
            Dict mapping failure_name → FP rate (0.0 to 1.0).
            Empty dict if no cases have been reviewed.
        """
        if self._fp_rate_cache is not None:
            return self._fp_rate_cache

        cases = self.load_cases()

        # Count reviewed cases per detector
        reviewed: dict[str, int] = {}
        fp_count: dict[str, int] = {}

        for case in cases:
            status = case.get("status", CaseStatus.PREDICTED.value)
            if status == CaseStatus.PREDICTED.value:
                continue  # skip unreviewed

            name = case.get("failure_name", "")
            if not name:
                continue

            reviewed[name] = reviewed.get(name, 0) + 1
            if status == CaseStatus.FALSE_POSITIVE.value:
                fp_count[name] = fp_count.get(name, 0) + 1

        rates: dict[str, float] = {}
        for name, total in reviewed.items():
            rates[name] = fp_count.get(name, 0) / total

        self._fp_rate_cache = rates
        return rates

    def get_feedback_summary(self) -> dict:
        """Get a summary of feedback statistics.

        Returns:
            Dict with per-detector review counts, FP rates, and totals.
        """
        cases = self.load_cases()
        total = len(cases)

        # Count by status
        status_counts: dict[str, int] = {}
        for case in cases:
            s = case.get("status", CaseStatus.PREDICTED.value)
            status_counts[s] = status_counts.get(s, 0) + 1

        fp_rates = self.get_detector_fp_rates()

        # Per-detector breakdown
        detector_stats: dict[str, dict] = {}
        for case in cases:
            name = case.get("failure_name", "unknown")
            if name not in detector_stats:
                detector_stats[name] = {
                    "total": 0,
                    "predicted": 0,
                    "confirmed": 0,
                    "false_positive": 0,
                    "corrected": 0,
                    "fp_rate": fp_rates.get(name, 0.0),
                }
            detector_stats[name]["total"] += 1
            s = case.get("status", CaseStatus.PREDICTED.value)
            if s in detector_stats[name]:
                detector_stats[name][s] += 1

        return {
            "total_cases": total,
            "status_counts": status_counts,
            "detector_stats": detector_stats,
            "fp_rates": fp_rates,
        }

    def adjust_diagnosis_confidence(
        self,
        diagnosis: "CaftDiagnosis",
    ) -> "CaftDiagnosis":
        """Adjust a diagnosis's confidence based on historical FP rates.

        If this detector has a high FP rate from past reviews, discount
        the confidence proportionally:

            adjusted = original * (1 - fp_rate * 0.5)

        The 0.5 factor ensures even a 100% FP rate only halves confidence
        (rather than zeroing it), since the detector may still catch real
        failures.

        Args:
            diagnosis: The diagnosis to adjust.

        Returns:
            The same diagnosis object with adjusted confidence (mutated in place).
        """
        fp_rates = self.get_detector_fp_rates()
        fp_rate = fp_rates.get(diagnosis.failure_name, 0.0)

        if fp_rate > 0.0:
            original = diagnosis.confidence
            diagnosis.confidence = round(
                original * (1.0 - fp_rate * 0.5), 4
            )
            logger.debug(
                "Adjusted %s confidence: %.3f → %.3f (FP rate: %.1f%%)",
                diagnosis.failure_name, original, diagnosis.confidence,
                fp_rate * 100,
            )

        return diagnosis

    # ------------------------------------------------------------------
    # LLM Confirmation integration
    # ------------------------------------------------------------------

    def record_confirmation(
        self,
        candidate: "CaftDiagnosis",
        result: "ConfirmationResult",
    ) -> None:
        """Record an LLM confirmation result for a candidate diagnosis.

        Updates the case ledger with the confirmation status and stores
        the reasoning for future case-based retrieval.

        Args:
            candidate: The original CAFT diagnosis candidate.
            result: The LLM confirmation result.
        """
        try:
            # Map confirmation status to case status
            status_map = {
                "confirmed": CaseStatus.CONFIRMED.value,
                "rejected": CaseStatus.FALSE_POSITIVE.value,
                "uncertain": CaseStatus.PREDICTED.value,
            }
            new_status = status_map.get(result.status, CaseStatus.PREDICTED.value)

            # Build a case ID for this confirmation
            session_id = self._session_id or "no_session"
            case_id = f"{session_id}_{candidate.at_step}_{candidate.failure_name}"

            # Try to update existing case in ledger
            updated = self.update_case_status(
                case_id=case_id,
                new_status=new_status,
                reviewer="llm",
                notes=result.reasoning,
            )

            if not updated:
                # Case not in ledger yet — create it with confirmation status
                case = DiagnosticCase.from_diagnosis(
                    diagnosis=candidate,
                    session_id=session_id,
                    source=self._session_source,
                    trace_length=len(self._event_buffer),
                    phase=(
                        self._event_buffer[-1].phase
                        if self._event_buffer
                        else ""
                    ),
                    trust=1.0,
                )
                case.status = new_status
                case.reviewer = "llm"
                case.resolution_notes = result.reasoning
                self._append_to_ledger(case)
                self._fp_rate_cache = None

        except Exception:
            logger.debug(
                "Failed to record confirmation for %s",
                candidate.failure_name,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Annotation persistence (first-class annotation integration)
    # ------------------------------------------------------------------

    def record_annotation(self, annotation: "AnnotationRecord") -> bool:
        """Store an AnnotationRecord linked to its session.

        Annotations are stored in:
          1. The local annotation ledger (JSONL, always)
          2. OpenViking session (if session exists, best-effort)

        Args:
            annotation: The AnnotationRecord to persist.

        Returns:
            True if stored successfully.
        """
        try:
            # Always write to local annotation ledger
            self._append_to_annotation_ledger(annotation)

            # If we have an active session or can find the session, store in OV too
            session_id = self._session_id or annotation.session_id
            if session_id:
                try:
                    self._client.add_message(
                        session_id=session_id,
                        role="assistant",
                        parts=[
                            {
                                "type": "tool",
                                "tool_name": f"annotation_{annotation.annotator_type}",
                                "tool_id": annotation.annotation_id,
                                "tool_status": annotation.label_status,
                                "tool_output": annotation.to_json(),
                                "duration_ms": 0,
                            },
                            {
                                "type": "text",
                                "text": (
                                    f"[{annotation.annotator_type}] "
                                    f"session={annotation.session_id[:8]} "
                                    f"failure={annotation.has_failure} "
                                    f"code={annotation.primary_caft_code} "
                                    f"status={annotation.label_status}"
                                ),
                            },
                        ],
                    )
                except Exception:
                    logger.debug(
                        "Failed to store annotation in OpenViking session",
                        exc_info=True,
                    )

            return True
        except Exception:
            logger.debug("Failed to record annotation", exc_info=True)
            return False

    def get_annotations_for_session(
        self,
        session_id: str,
    ) -> list[dict]:
        """Retrieve all annotations for a session from the local ledger.

        Args:
            session_id: Session ID (full or 8-char prefix).

        Returns:
            List of annotation dicts, newest first.
        """
        annotations: list[dict] = []
        ledger = self._annotation_ledger_path
        if not ledger.exists():
            return annotations

        try:
            with open(ledger, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ann = json.loads(line)
                        sid = ann.get("session_id", "")
                        tid = ann.get("trace_id", "")
                        if (sid.startswith(session_id) or tid.startswith(session_id)
                                or session_id.startswith(sid[:8])):
                            annotations.append(ann)
                    except json.JSONDecodeError:
                        continue
        except Exception:
            logger.debug("Failed to read annotation ledger", exc_info=True)

        annotations.reverse()  # newest first
        return annotations

    def find_annotation_needed_cases(
        self,
        limit: int = 20,
    ) -> list[dict]:
        """Find cases that need annotation (status=predicted, no annotation).

        Returns cases sorted by severity (critical first) that have no
        corresponding annotation record.

        Args:
            limit: Maximum number of cases to return.

        Returns:
            List of case dicts needing annotation.
        """
        # Load all cases
        cases = self.load_cases()

        # Load annotated session IDs
        annotated_sessions: set[str] = set()
        ledger = self._annotation_ledger_path
        if ledger.exists():
            try:
                with open(ledger, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            ann = json.loads(line)
                            atype = ann.get("annotator_type", "")
                            if atype in ("human", "adjudicated"):
                                sid = ann.get("session_id", "")
                                if sid:
                                    annotated_sessions.add(sid)
                                    annotated_sessions.add(sid[:8])
                        except json.JSONDecodeError:
                            continue
            except Exception:
                pass

        # Filter to unannotated predicted cases
        severity_rank = {"critical": 3, "warning": 2, "info": 1}
        needed = []
        for case in cases:
            status = case.get("status", "predicted")
            if status != "predicted":
                continue
            sid = case.get("session_id", "")
            if sid in annotated_sessions or sid[:8] in annotated_sessions:
                continue
            needed.append(case)

        # Sort by severity (critical first)
        needed.sort(
            key=lambda c: severity_rank.get(c.get("severity", "info"), 0),
            reverse=True,
        )
        return needed[:limit]

    def record_adjudicated_label(
        self,
        session_id: str,
        has_failure: bool,
        primary_caft_code: str = "",
        adjudicator_id: str = "human",
        rationale: str = "",
        severity: int = 0,
    ) -> bool:
        """Record the final adjudicated gold label for a session.

        This creates an adjudicated AnnotationRecord and also updates
        any matching case in the case ledger to CONFIRMED or FALSE_POSITIVE.

        Args:
            session_id: Session to adjudicate.
            has_failure: Whether there was a real failure.
            primary_caft_code: The CAFT code (if failure).
            adjudicator_id: Who adjudicated.
            rationale: Explanation of the decision.
            severity: Severity 1-5 (if failure).

        Returns:
            True if recorded successfully.
        """
        try:
            from agentdiag.annotation_models import build_adjudicated_annotation

            annotation = build_adjudicated_annotation(
                session_id=session_id,
                adjudicator_id=adjudicator_id,
                has_failure=has_failure,
                primary_caft_code=primary_caft_code,
                severity=severity,
                rationale=rationale,
            )
            self.record_annotation(annotation)

            # Update matching cases in the case ledger
            cases = self.load_cases()
            for case in cases:
                if not case.get("session_id", "").startswith(session_id[:8]):
                    continue
                case_id = case.get("case_id", "")
                if not case_id:
                    continue
                case_name = case.get("failure_name", "")
                from agentdiag.caft.taxonomy import CAFT_TAXONOMY
                expected_name = ""
                if primary_caft_code and primary_caft_code in CAFT_TAXONOMY:
                    expected_name = CAFT_TAXONOMY[primary_caft_code].name

                if has_failure and case_name == expected_name:
                    self.update_case_status(case_id, "confirmed",
                                            reviewer=adjudicator_id, notes=rationale)
                elif not has_failure:
                    self.update_case_status(case_id, "false_positive",
                                            reviewer=adjudicator_id, notes=rationale)

            return True
        except Exception:
            logger.debug("Failed to record adjudicated label", exc_info=True)
            return False

    @property
    def _annotation_ledger_path(self) -> Path:
        return Path(self._db_path) / "annotation_ledger.jsonl"

    def _append_to_annotation_ledger(self, annotation: "AnnotationRecord") -> None:
        """Append an annotation record to the local JSONL annotation ledger."""
        try:
            ledger = self._annotation_ledger_path
            ledger.parent.mkdir(parents=True, exist_ok=True)
            with open(ledger, "a", encoding="utf-8") as f:
                f.write(annotation.to_json() + "\n")
        except Exception:
            logger.debug("Failed to write to annotation ledger", exc_info=True)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the OpenViking client connection."""
        try:
            self._client.close()
        except Exception:
            pass
