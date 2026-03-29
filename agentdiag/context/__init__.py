"""Persistent diagnostic context backed by OpenViking.

Provides searchable, session-based storage for CAFT diagnoses and trace events.
Degrades gracefully to no-ops when OpenViking is not installed.

Usage::

    from agentdiag.context import get_context_store

    store = get_context_store("./my_context_db")
    if store:
        sid = store.start_session(goal="Fix login bug")
        # ... record events ...
        store.end_session(dashboard_state)
        store.close()
"""

from __future__ import annotations

from typing import Optional

_HAS_OPENVIKING = False
try:
    from agentdiag.context.openviking import ContextStore, DiagnosticCase, CaseStatus
    _HAS_OPENVIKING = True
except ImportError:
    ContextStore = None  # type: ignore[assignment,misc]
    DiagnosticCase = None  # type: ignore[assignment,misc]
    CaseStatus = None  # type: ignore[assignment,misc]

from agentdiag.context.instrumented import InstrumentedContextStore


def get_context_store(db_path: Optional[str] = None) -> Optional["ContextStore"]:
    """Create a ContextStore if OpenViking is available.

    Returns None if OpenViking is not installed, allowing callers to
    skip context operations without checking for the dependency.

    Args:
        db_path: Path for the context database. Defaults to ``./agentdiag_context``.
    """
    if not _HAS_OPENVIKING:
        return None

    path = db_path or "./agentdiag_context"
    try:
        return ContextStore(db_path=path)
    except Exception:
        return None


def get_instrumented_store(
    db_path: Optional[str] = None,
    on_event=None,
) -> InstrumentedContextStore:
    """Create an InstrumentedContextStore (always available, degrades gracefully).

    Unlike get_context_store(), this never returns None — it always provides
    event emission even if OpenViking is not installed.
    """
    path = db_path or "./agentdiag_context"
    return InstrumentedContextStore(db_path=path, on_event=on_event)


__all__ = [
    "ContextStore", "DiagnosticCase", "CaseStatus", "get_context_store",
    "InstrumentedContextStore", "get_instrumented_store",
]
