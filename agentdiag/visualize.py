"""Web-based diagnostic visualization server.

Serves a Canvas-based visualization where HTA phases are rooms,
agent movement encodes phase transitions, and CAFT states are
visible as agent body language.

Architecture:
  - Ingestion thread reads JSONL from stdin/file, pushes to MonitorEngine
  - WebSocket handler polls MonitorEngine.state at 10Hz, pushes diffs to clients
  - Single HTML page with inline Canvas JS renders the spatial map

Usage:
    agentdiag visualize --input stdin
    agentdiag visualize --input trace.jsonl --delay 0.2
    agentdiag visualize --input stdin --port 8080
"""

from __future__ import annotations

import asyncio
import json
import queue
import sys
import threading
import time
from pathlib import Path
from typing import IO, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from agentdiag.monitor import MonitorEngine, DashboardState, ActionEntry
from agentdiag.hta import Phase


app = FastAPI(title="agentdiag visualize")


class QueueStream:
    """Thread-safe file-like object backed by a queue.

    The demo thread writes JSONL lines via write(); the ingestion thread
    reads them via iteration. A None sentinel signals end-of-stream.
    """

    def __init__(self) -> None:
        self._q: queue.Queue[Optional[str]] = queue.Queue()

    def write(self, line: str) -> None:
        self._q.put(line)

    def close(self) -> None:
        self._q.put(None)

    def __iter__(self):
        return self

    def __next__(self) -> str:
        item = self._q.get()
        if item is None:
            raise StopIteration
        return item


# ── Module-level state shared between ingestion thread and WS handler ──
_engine: Optional[MonitorEngine] = None
_context_store = None  # Optional[ContextStore]
_status_engine = None  # Optional[StatusEngine]
_current_state_json: str = '{"type":"waiting"}'
_event_count: int = 0
_stream_ended: bool = False

# Session metadata (populated once during ingestion, static after that)
_session_meta: dict = {}

# Multi-agent tracking
_agent_phases: dict[str, str] = {}      # agent_id -> current phase
_agent_active: dict[str, bool] = {}     # agent_id -> still active?
_next_agent_idx: int = 1

# Ablation mode state
_ablation_mode: bool = False
_ablation_state: dict = {"type": "ablation_waiting", "phase": "idle"}
_ablation_events: list[dict] = []  # Recent LLM/context events (ring buffer, max 200)
_ablation_event_count: int = 0
_ablation_complete: bool = False


def _dashboard_to_dict(state: DashboardState) -> dict:
    """Convert DashboardState to a WebSocket-friendly dict.

    This is the single serialization boundary between the Python pipeline
    and the Canvas frontend. Every field here drives a visual element.
    """
    hta = state.hta_state
    current_phase = hta.current_phase.label if hta else "idle"

    result = {
        "type": "state_update",
        # HTA phase → which room the agent is in
        "phase": current_phase,
        "phase_index": int(hta.current_phase) if hta else 0,
        "progress_pct": round(state.progress_pct, 4),
        # Trust → agent health color (deprecated, kept for compat)
        "trust_score": round(state.trust_score, 4),
        "health": state.health,
        # V5: Session health metrics (grounded in observable behavior)
        "completion_rate": state.completion_rate,
        "failure_density": round(state.failure_density, 2),
        "active_diagnosis_count": state.active_diagnosis_count,
        "completion_history": state.completion_history,
        # Counters
        "total_events": state.total_events,
        "total_errors": state.total_errors,
        "events_per_minute": state.events_per_minute,
        # CAFT diagnoses → badge color + side panel
        "diagnoses": [
            {
                "caft_code": d.caft_code,
                "failure_name": d.failure_name,
                "severity": d.severity.value,
                "description": d.description,
                "at_step": d.at_step,
                "confidence": round(d.confidence, 3),
                "remediation": d.remediation,
                "evidence": {
                    k: str(v)[:200] for k, v in d.evidence.items()
                },
            }
            for d in state.diagnoses
        ],
        # Recent actions → side panel log
        "actions": [
            {
                "step": a.step,
                "tool": a.tool,
                "phase": a.phase.label,
                "success": a.success,
                "latency_ms": round(a.latency_ms, 1),
            }
            for a in state.actions[-30:]
        ],
        # Phase transitions → agent movement history
        "transitions": [
            {
                "from_phase": t.from_phase.label,
                "to_phase": t.to_phase.label,
                "at_step": t.at_step,
                "is_regression": t.is_regression,
            }
            for t in (hta.transitions if hta else [])
        ],
        # Per-phase event counts → room activity indicators
        "phase_event_counts": hta.phase_event_counts if hta else {},
        "regression_count": hta.regression_count if hta else 0,
        # Multi-agent state
        "agents": {
            aid: {
                "phase": phase,
                "label": aid,
                "active": _agent_active.get(aid, True),
            }
            for aid, phase in _agent_phases.items()
        },
        # Confirmation stats (V4)
        "candidates_total": state.candidates_total,
        "candidates_confirmed": state.candidates_confirmed,
        "candidates_rejected": state.candidates_rejected,
        "candidates_uncertain": state.candidates_uncertain,
        "candidates_autoconfirmed": state.candidates_autoconfirmed,
        # Session context (populated once during ingestion, static after that)
        "session_context": {
            "session_id": _session_meta.get("session_id", ""),
            "project_dir": _session_meta.get("project_dir", ""),
            "user_goal": _session_meta.get("user_goal", "") or (
                _engine._hta._goal if _engine else ""
            ),
            "first_timestamp": _session_meta.get("first_timestamp", ""),
            "last_timestamp": _session_meta.get("last_timestamp", ""),
            "source_events": _session_meta.get("total_events", 0),
        },
        # Duration/timing from DashboardState
        "start_time": state.start_time,
        "last_event_time": state.last_event_time,
    }

    # V7: Attach cognitive state if available
    if _engine is not None and _engine.cognitive_state is not None:
        tracker = _engine.cognitive_state
        result["cognitive_state"] = tracker.state.to_dict()
        result["working_memory"] = tracker.working_memory.to_dict()
        result["decision_points"] = tracker.decision_log.to_dict()
        result["info_theoretic"] = tracker.symbol_stream.to_dict()

    # V8: Human-readable status summary
    if _status_engine is not None:
        result["status_summary"] = _status_engine.get_summary()

    return result


# ── Ablation progress callback ─────────────────────────────────────────

def _ablation_progress_callback(event: dict):
    """Called from ablation background thread — updates global state for WebSocket."""
    global _ablation_state, _ablation_event_count, _ablation_complete

    etype = event.get("type", "")

    # Append detailed events to the ring buffer
    if etype in ("llm_start", "llm_result", "fp_adjustment", "auto_confirm"):
        _ablation_events.append(event)
        if len(_ablation_events) > 200:
            _ablation_events.pop(0)

    if etype == "setup":
        # Setup phase: annotation loading, split filtering, dedup
        step = event.get("step", "")
        _ablation_state = {
            "type": "ablation_update",
            "phase": "setup",
            "setup_step": step,
            "total_sessions": event.get("total_sessions", 0),
            "total_annotations": event.get("total_annotations", 0),
            "modes": event.get("modes", []),
            "modes_completed": [],
            "reports": {},
            "recent_events": [],
        }
    elif etype == "mode_start":
        _ablation_state = {
            "type": "ablation_update",
            "phase": "running",
            "current_mode": event.get("mode", ""),
            "total_sessions": event.get("total_sessions", 0),
            "sessions_processed": 0,
            "current_session": "",
            "current_session_detections": [],
            "modes_completed": _ablation_state.get("modes_completed", []),
            "reports": _ablation_state.get("reports", {}),
            "recent_events": _ablation_events[-30:],
        }
    elif etype == "mode_complete":
        completed = list(_ablation_state.get("modes_completed", []))
        completed.append(event.get("mode", ""))
        reports = dict(_ablation_state.get("reports", {}))
        reports[event.get("mode", "")] = event.get("report", {})
        _ablation_state = {
            "type": "ablation_update",
            "phase": "running",
            "current_mode": _ablation_state.get("current_mode", ""),
            "total_sessions": _ablation_state.get("total_sessions", 0),
            "modes_completed": completed,
            "reports": reports,
            "recent_events": _ablation_events[-30:],
        }
    elif etype in ("discovering", "scanning_start"):
        # Pre-scan phase: discovering traces or about to start scanning
        _ablation_state = dict(_ablation_state)
        _ablation_state["type"] = "ablation_update"
        if etype == "scanning_start":
            _ablation_state["sessions_processed"] = 0
            _ablation_state["total_sessions"] = event.get("to_scan", 0)
            _ablation_state["current_session"] = ""
            _ablation_state["current_session_detections"] = []
    elif etype == "session_scan":
        # Per-session progress during detector scanning
        _ablation_state = dict(_ablation_state)
        _ablation_state["type"] = "ablation_update"
        _ablation_state["sessions_processed"] = event.get("session_idx", 0)
        _ablation_state["total_sessions"] = event.get("total_sessions", 0)
        _ablation_state["current_session"] = event.get("session_id", "")[:8]
        _ablation_state["current_session_detections"] = event.get("detections", [])
    elif etype in ("llm_result", "auto_confirm", "fp_adjustment", "llm_start"):
        # Update recent_events in the current state
        _ablation_state = dict(_ablation_state)
        _ablation_state["recent_events"] = _ablation_events[-30:]
    elif etype == "ablation_complete":
        _ablation_state = {
            "type": "ablation_complete",
            "phase": "complete",
            "reports": event.get("reports", {}),
            "comparison_table": event.get("comparison_table", ""),
            "modes_completed": _ablation_state.get("modes_completed", []),
            "recent_events": _ablation_events[-50:],
        }
        _ablation_complete = True

    _ablation_event_count += 1


# ── Routes ──────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    """Serve the visualization HTML."""
    html_path = Path(__file__).parent / "static" / "visualize.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api/mode")
async def api_mode():
    """Return the current visualization mode (normal, compare, or ablation)."""
    if _ablation_mode:
        return JSONResponse({"mode": "ablation"})
    return JSONResponse({"mode": "compare" if _compare_mode else "normal"})


@app.get("/api/sprites")
async def api_sprites():
    """Serve pre-decoded character sprite data for pixel art rendering."""
    sprites_path = Path(__file__).parent / "static" / "sprites.json"
    if sprites_path.exists():
        return JSONResponse(json.loads(sprites_path.read_text(encoding="utf-8")))
    return JSONResponse({"error": "sprites not built — run scripts/build_pixel_sprites.py"}, status_code=404)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """Push DashboardState snapshots to the browser at 10Hz."""
    await ws.accept()
    last_count = -1
    last_send_time = 0.0

    try:
        while True:
            if _ablation_mode:
                # Ablation mode: send ablation state on change OR every 2s
                current_count = _ablation_event_count
                now = time.time()
                if current_count != last_count or (now - last_send_time) > 2.0:
                    last_count = current_count
                    last_send_time = now
                    await ws.send_text(json.dumps(_ablation_state))

                if _ablation_complete and current_count == last_count:
                    final = dict(_ablation_state)
                    final["type"] = "ablation_complete"
                    await ws.send_json(final)
                    break
            else:
                # Normal monitoring mode
                current_count = _event_count
                if current_count != last_count:
                    last_count = current_count
                    await ws.send_text(_current_state_json)

                if _stream_ended and current_count == last_count:
                    final = json.loads(_current_state_json)
                    final["type"] = "stream_end"
                    await ws.send_json(final)
                    break

            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


# ── Ablation API endpoints ─────────────────────────────────────────────

@app.get("/api/ablation/events")
async def api_ablation_events(last: int = 50):
    """Return recent ablation LLM/context events."""
    return JSONResponse({"events": _ablation_events[-last:]})


@app.get("/api/ablation/status")
async def api_ablation_status():
    """Return current ablation progress state."""
    return JSONResponse(_ablation_state)


# ── Context API endpoints ───────────────────────────────────────────────

@app.get("/api/cases")
async def api_cases(status: Optional[str] = None):
    """List all diagnostic cases from the context store ledger."""
    if _context_store is None:
        return JSONResponse({"cases": [], "error": "no context store"})
    cases = _context_store.load_cases(status_filter=status)
    return JSONResponse({"cases": cases})


@app.post("/api/cases/{case_id}/feedback")
async def api_case_feedback(case_id: str, action: str = "confirm"):
    """Update a case's status: action = 'confirm' | 'false_positive'."""
    if _context_store is None:
        return JSONResponse({"ok": False, "error": "no context store"}, status_code=503)
    status_map = {
        "confirm": "confirmed",
        "false_positive": "false_positive",
    }
    new_status = status_map.get(action)
    if not new_status:
        return JSONResponse({"ok": False, "error": f"invalid action: {action}"}, status_code=400)
    ok = _context_store.update_case_status(case_id, new_status, reviewer="dashboard")
    return JSONResponse({"ok": ok, "case_id": case_id, "new_status": new_status})


@app.get("/api/stats")
async def api_stats():
    """Context store statistics and detector FP rates."""
    if _context_store is None:
        return JSONResponse({"error": "no context store"})
    stats = _context_store.get_stats()
    feedback = _context_store.get_feedback_summary()
    return JSONResponse({"stats": stats, "feedback": feedback})


@app.get("/api/similar-cases")
async def api_similar_cases(caft_code: str = "", step: int = 0):
    """Search for past cases similar to a given CAFT diagnosis."""
    if _context_store is None:
        return JSONResponse({"cases": [], "error": "no context store"})
    from agentdiag.caft.base import CaftDiagnosis, CaftSeverity
    query_diag = CaftDiagnosis(
        caft_code=caft_code, caft_category="", failure_name="",
        severity=CaftSeverity.INFO, confidence=0.5,
        description=f"CAFT {caft_code} at step {step}",
        evidence={}, at_step=step, remediation="",
    )
    results = _context_store.find_similar_failures(query_diag, limit=5)
    return JSONResponse({"cases": results})


@app.get("/api/decision-trace")
async def api_decision_trace(
    step: Optional[int] = None,
    detector: Optional[str] = None,
):
    """Return the decision trace (if enabled).

    Query params:
        step: Return only records for this step.
        detector: Return only snapshots for this detector name.
        (no params): Return the full trace.
    """
    if _engine is None or _engine.decision_trace is None:
        return JSONResponse(
            {"error": "decision trace not enabled (start with --decision-trace)"},
            status_code=404,
        )
    trace = _engine.decision_trace
    if step is not None:
        return JSONResponse({
            "step_record": trace.get_step(step),
            "detector_snapshots": trace.get_snapshots_at_step(step),
        })
    if detector is not None:
        return JSONResponse({
            "detector": detector,
            "timeline": trace.get_detector_timeline(detector),
        })
    return JSONResponse(trace.to_dict())


@app.get("/api/cognitive")
async def api_cognitive():
    """Return current cognitive state snapshot."""
    if _engine is None or _engine.cognitive_state is None:
        return JSONResponse(
            {"error": "cognitive monitoring not enabled (start with --cognitive)"},
            status_code=404,
        )
    return JSONResponse(_engine.cognitive_state.to_dict())


@app.get("/api/cognitive/memory")
async def api_cognitive_memory():
    """Return working memory model snapshot."""
    if _engine is None or _engine.cognitive_state is None:
        return JSONResponse(
            {"error": "cognitive monitoring not enabled"},
            status_code=404,
        )
    return JSONResponse(_engine.cognitive_state.working_memory.to_dict())


@app.get("/api/cognitive/decisions")
async def api_cognitive_decisions(last: int = 20):
    """Return recent decision points."""
    if _engine is None or _engine.cognitive_state is None:
        return JSONResponse(
            {"error": "cognitive monitoring not enabled"},
            status_code=404,
        )
    log = _engine.cognitive_state.decision_log
    decisions = [d.to_dict() for d in log.get_recent(last)]
    return JSONResponse({
        "decisions": decisions,
        "metrics": log.get_decision_quality_metrics(),
    })


@app.get("/api/cognitive/info-theoretic")
async def api_info_theoretic():
    """Return full info-theoretic state with histories."""
    if _engine is None or _engine.cognitive_state is None:
        return JSONResponse(
            {"error": "cognitive monitoring not enabled"},
            status_code=404,
        )
    return JSONResponse(_engine.cognitive_state.symbol_stream.to_dict())


@app.get("/api/cognitive/memory-ops")
async def api_memory_ops():
    """Return recent memory operation events for timeline."""
    if _engine is None or _engine.cognitive_state is None:
        return JSONResponse(
            {"error": "cognitive monitoring not enabled"},
            status_code=404,
        )
    return JSONResponse({
        "events": list(_engine.cognitive_state.symbol_stream.memory_events),
    })


@app.get("/api/cognitive/ip-stage-map")
async def api_ip_stage_map():
    """Return the IP stage mapping for all CAFT types."""
    from agentdiag.caft.taxonomy import CAFT_TAXONOMY
    return JSONResponse({
        code: {
            "name": t.name,
            "label": t.label,
            "ip_stage": t.ip_stage,
            "category": t.category,
            "detectability": t.detectability.value,
        }
        for code, t in CAFT_TAXONOMY.items()
    })


@app.get("/api/memory")
async def api_memory(caft_code: str = "", limit: int = 5):
    """OpenViking memory panel: detector stats, past cases, FP rates.

    Returns real counts from the case ledger — no synthetic data.
    """
    if _context_store is None:
        return JSONResponse({"error": "no context store", "cases": [],
                             "detector_stats": {}, "fp_rates": {}})
    feedback = _context_store.get_feedback_summary()
    cases = _context_store.load_cases()

    # Filter by caft_code if specified
    if caft_code:
        cases = [c for c in cases if c.get("caft_code") == caft_code]

    return JSONResponse({
        "cases": cases[:limit],
        "total_cases": feedback.get("total_cases", 0),
        "status_counts": feedback.get("status_counts", {}),
        "detector_stats": feedback.get("detector_stats", {}),
        "fp_rates": feedback.get("fp_rates", {}),
    })


# ── Comparison mode state ───────────────────────────────────────────────
_compare_mode: bool = False
_engine_b: Optional[MonitorEngine] = None
_current_state_b_json: str = '{"type":"waiting"}'
_event_count_b: int = 0
_stream_ended_b: bool = False


# ── Ingestion thread ────────────────────────────────────────────────────

def _ingest_thread_b(
    stream: IO[str],
    engine: MonitorEngine,
    delay: float = 0.0,
):
    """Background thread for trace B in comparison mode."""
    global _current_state_b_json, _event_count_b, _stream_ended_b

    for line in stream:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        engine.push_raw(data)
        state = engine.state
        _current_state_b_json = json.dumps(_dashboard_to_dict(state))
        _event_count_b = state.total_events

        if delay > 0:
            time.sleep(delay)

    _stream_ended_b = True


_CLAUDE_CODE_RAW_TYPES = {"user", "assistant", "file-history-snapshot", "system"}


def _is_claude_code_raw(data: dict) -> bool:
    """Check if a JSON object is from a raw Claude Code session JSONL file.

    Raw CC sessions have lines with type in {user, assistant,
    file-history-snapshot, system} and wrapper fields like sessionId,
    parentUuid, message, or snapshot.
    """
    if not isinstance(data, dict):
        return False
    t = data.get("type")
    if t not in _CLAUDE_CODE_RAW_TYPES:
        return False
    # Any of these wrapper fields indicate raw CC format
    return bool(
        "message" in data
        or "timestamp" in data
        or "sessionId" in data
        or "snapshot" in data
        or "parentUuid" in data
    )


def _replay_trace_events(
    events,
    engine: MonitorEngine,
    delay: float = 0.0,
):
    """Push pre-parsed TraceEvents through engine and update dashboard state."""
    global _current_state_json, _event_count, _stream_ended

    for event in events:
        engine.push(event)
        state = engine.state

        current_phase = (
            state.hta_state.current_phase.label
            if state.hta_state
            else "idle"
        )
        _agent_phases["M"] = current_phase

        state_dict = _dashboard_to_dict(state)

        if _engine is not None and _engine.decision_trace is not None:
            state_dict["detector_snapshots"] = _engine.decision_trace.latest_snapshots(1)

        _current_state_json = json.dumps(state_dict)
        _event_count = state.total_events

        if delay > 0:
            time.sleep(delay)

    _stream_ended = True


def _ingest_thread(
    stream: IO[str],
    engine: MonitorEngine,
    delay: float = 0.0,
):
    """Background thread: read JSONL → push to MonitorEngine → update global state."""
    global _current_state_json, _event_count, _stream_ended, _next_agent_idx

    # Peek at the first non-empty line to detect raw Claude Code sessions
    first_line = None
    for line in stream:
        line = line.strip()
        if not line:
            continue
        try:
            first_data = json.loads(line)
        except json.JSONDecodeError:
            continue
        first_line = line
        break

    if first_line is None:
        _stream_ended = True
        return

    # Check if this is a raw Claude Code session file
    if _is_claude_code_raw(first_data):
        # Read remaining lines and pre-parse with ClaudeCodeExtractor
        from agentdiag.adapters.claude_code import ClaudeCodeExtractor
        import tempfile, os

        all_lines = [first_line]
        for line in stream:
            line = line.strip()
            if line:
                all_lines.append(line)

        # Write to temp file for ClaudeCodeExtractor (expects a file path)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as tmp:
            tmp.write("\n".join(all_lines) + "\n")
            tmp_path = tmp.name

        try:
            extractor = ClaudeCodeExtractor()
            events = extractor.parse_session(tmp_path)

            # Extract session metadata from raw lines
            first_ts, last_ts = extractor._scan_timestamps(Path(tmp_path))
            first_goal = next((e.goal_text for e in events if e.goal_text), "")

            # Derive session_id and project_dir from input_path or temp path
            input_path = _session_meta.get("_input_path", "")
            if input_path:
                ip = Path(input_path)
                sid = ip.stem
                pdir = ip.parent.name
            else:
                sid = ""
                pdir = ""

            _session_meta.update({
                "session_id": sid,
                "project_dir": pdir,
                "first_timestamp": first_ts or "",
                "last_timestamp": last_ts or "",
                "total_events": len(events),
                "user_goal": first_goal,
            })

            _replay_trace_events(events, engine, delay)
        finally:
            os.unlink(tmp_path)
        return

    # Standard path: pre-parsed TraceEvent JSONL or adapter-compatible format
    def _process_data(data):
        global _current_state_json, _event_count, _next_agent_idx

        # Handle scenario boundary markers (from showcase mode)
        if data.get("type") == "scenario_boundary":
            _current_state_json = json.dumps(data)
            _agent_phases.clear()
            _agent_active.clear()
            _agent_phases["M"] = "idle"
            _agent_active["M"] = True
            _next_agent_idx = 1
            return

        # Track multi-agent state
        agent_id = data.get("agent_id", "M")
        tool = data.get("tool", "")

        if tool == "Task":
            new_id = f"S{_next_agent_idx}"
            _next_agent_idx += 1
            _agent_phases[new_id] = "gathering"
            _agent_active[new_id] = True

        if data.get("type") == "tool_result" and agent_id.startswith("S"):
            _agent_active[agent_id] = False

        engine.push_raw(data)
        state = engine.state

        current_phase = (
            state.hta_state.current_phase.label
            if state.hta_state
            else "idle"
        )
        _agent_phases[agent_id] = current_phase

        # Feed status engine with event and any new diagnoses
        if _status_engine is not None:
            event_dict = {
                "step": data.get("step", _event_count),
                "event_type": data.get("type", "tool_call"),
                "tool_name": data.get("tool", ""),
                "target_path": data.get("target_path", ""),
                "symbol": f"tool:{data.get('tool', 'unknown')}",
            }
            # Convert CAFT diagnoses to anomaly-like dicts for status engine
            anomaly = None
            if state.diagnoses:
                latest = state.diagnoses[-1]
                if latest.at_step == data.get("step", -1):
                    anomaly = {
                        "signature": latest.failure_name or "unknown",
                        "step": latest.at_step,
                        "metrics": {},
                        "wickens_stage": "",
                    }
            # Check for gradual declines from cognitive tracker
            declines = None
            if _engine and _engine.cognitive_state:
                it = _engine.cognitive_state.to_dict().get("info_theoretic", {})
                bl = it.get("baseline", {}) if "baseline" in it else {}
                if isinstance(bl, dict):
                    declines = bl.get("gradual_declines")

            _status_engine.update(event_dict, anomaly, declines)

        state_dict = _dashboard_to_dict(state)

        if _engine is not None and _engine.decision_trace is not None:
            state_dict["detector_snapshots"] = _engine.decision_trace.latest_snapshots(1)

        _current_state_json = json.dumps(state_dict)
        _event_count = state.total_events

    # Process the first line we already read
    _process_data(first_data)
    if delay > 0:
        time.sleep(delay)

    # Process remaining lines
    for line in stream:
        line = line.strip()
        if not line:
            continue

        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        _process_data(data)

        if delay > 0:
            time.sleep(delay)

    _stream_ended = True


# ── Server entry point ──────────────────────────────────────────────────

def start_server(
    stream: IO[str],
    goal: str = "",
    port: int = 8080,
    host: str = "127.0.0.1",
    delay: float = 0.0,
    context_store=None,
    confirm: bool = False,
    decision_trace: bool = False,
    cognitive: bool = False,
    input_path: str = "",
):
    """Start the visualization server.

    Args:
        stream: JSONL input (stdin or file handle).
        goal: Goal text for HTA tracking.
        port: HTTP port.
        host: Bind address.
        delay: Seconds between events (for file replay pacing).
        context_store: Optional ContextStore for persistent diagnostics.
        confirm: Enable LLM confirmation of CAFT detections.
        decision_trace: Enable per-step decision trace recording.
        cognitive: Enable cognitive load monitoring.
        input_path: Path to the input file being analyzed (for session metadata).
    """
    global _engine, _context_store, _status_engine, _current_state_json, _event_count, _stream_ended
    global _agent_phases, _agent_active, _next_agent_idx, _session_meta

    # Reset global state
    _current_state_json = '{"type":"waiting"}'
    _event_count = 0
    _stream_ended = False
    _agent_phases = {"M": "idle"}
    _agent_active = {"M": True}
    _next_agent_idx = 1

    # Initialize status engine
    from agentdiag.status_engine import StatusEngine
    _status_engine = StatusEngine()

    # Seed session metadata from input_path (will be enriched during ingestion)
    _session_meta = {"_input_path": input_path}
    if input_path:
        ip = Path(input_path)
        _session_meta["session_id"] = ip.stem
        _session_meta["project_dir"] = ip.parent.name

    _context_store = context_store
    _engine = MonitorEngine(
        goal=goal, context_store=context_store, confirm=confirm,
        decision_trace=decision_trace, cognitive=cognitive,
    )

    if context_store is not None:
        _engine.start_context_session(goal=goal, source="visualize")

    # Start ingestion in background thread
    thread = threading.Thread(
        target=_ingest_thread,
        args=(stream, _engine, delay),
        daemon=True,
    )
    thread.start()

    # Suppress 'Event loop is closed' noise from httpx/anthropic client
    # cleanup that fires after uvicorn shuts down the event loop.
    # Must be installed before uvicorn starts so it covers shutdown too.
    _suppress_event_loop_closed_errors()

    print(f"agentdiag visualize")
    print(f"  http://{host}:{port}")
    print(f"  Waiting for events on {'stdin' if stream is sys.stdin else 'file'}...")
    print()

    try:
        uvicorn.run(app, host=host, port=port, log_level="warning")
    finally:
        if context_store is not None:
            _engine.end_context_session()


def start_compare_server(
    stream_a: IO[str],
    stream_b: IO[str],
    goal: str = "",
    port: int = 8080,
    host: str = "127.0.0.1",
    delay: float = 0.0,
):
    """Start the visualization server in comparison mode (two traces side-by-side).

    Args:
        stream_a: JSONL input for trace A.
        stream_b: JSONL input for trace B.
        goal: Goal text for HTA tracking.
        port: HTTP port.
        host: Bind address.
        delay: Seconds between events for pacing.
    """
    global _engine, _engine_b, _context_store
    global _current_state_json, _event_count, _stream_ended
    global _current_state_b_json, _event_count_b, _stream_ended_b
    global _compare_mode
    global _agent_phases, _agent_active, _next_agent_idx

    # Reset global state
    _compare_mode = True
    _current_state_json = '{"type":"waiting"}'
    _current_state_b_json = '{"type":"waiting"}'
    _event_count = 0
    _event_count_b = 0
    _stream_ended = False
    _stream_ended_b = False
    _agent_phases = {"M": "idle"}
    _agent_active = {"M": True}
    _next_agent_idx = 1
    _context_store = None

    _engine = MonitorEngine(goal=goal)
    _engine_b = MonitorEngine(goal=goal)

    # Start ingestion for both traces
    thread_a = threading.Thread(
        target=_ingest_thread,
        args=(stream_a, _engine, delay),
        daemon=True,
    )
    thread_b = threading.Thread(
        target=_ingest_thread_b,
        args=(stream_b, _engine_b, delay),
        daemon=True,
    )
    thread_a.start()
    thread_b.start()

    _suppress_event_loop_closed_errors()

    print(f"agentdiag compare")
    print(f"  http://{host}:{port}")
    print(f"  Comparing two traces side-by-side...")
    print()

    uvicorn.run(app, host=host, port=port, log_level="warning")


@app.websocket("/ws/compare")
async def websocket_compare(ws: WebSocket):
    """Push both trace states to the browser for comparison mode."""
    await ws.accept()
    last_a = -1
    last_b = -1

    try:
        while True:
            ca = _event_count
            cb = _event_count_b
            if ca != last_a or cb != last_b:
                last_a = ca
                last_b = cb
                state_a = json.loads(_current_state_json) if _current_state_json != '{"type":"waiting"}' else {"type": "waiting"}
                state_b = json.loads(_current_state_b_json) if _current_state_b_json != '{"type":"waiting"}' else {"type": "waiting"}
                await ws.send_json({
                    "mode": "compare",
                    "trace_a": state_a,
                    "trace_b": state_b,
                })

            both_ended = _stream_ended and _stream_ended_b
            if both_ended and ca == last_a and cb == last_b:
                state_a = json.loads(_current_state_json) if _current_state_json != '{"type":"waiting"}' else {"type": "waiting"}
                state_b = json.loads(_current_state_b_json) if _current_state_b_json != '{"type":"waiting"}' else {"type": "waiting"}
                await ws.send_json({
                    "mode": "compare",
                    "type": "stream_end",
                    "trace_a": state_a,
                    "trace_b": state_b,
                })
                break

            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


# ── Ablation server entry point ────────────────────────────────────────

def start_ablation_server(
    annotations_path: str,
    traces_root: str,
    output_dir: str,
    port: int = 8080,
    host: str = "127.0.0.1",
    split: str | None = None,
    splits_file: str | None = None,
    llm_provider: str | None = None,
    match_window: int = 5,
    bootstrap_n: int = 1000,
    skip_bootstrap: bool = False,
    detector_filter: set[str] | None = None,
    modes: list[str] | None = None,
    context_db: str | None = None,
    verbose: bool = False,
):
    """Start the web dashboard in ablation mode.

    Runs the ablation study in a background thread while serving the
    real-time dashboard. The ablation thread emits events via
    _ablation_progress_callback which updates global state polled by
    the WebSocket handler at 10Hz.
    """
    global _ablation_mode, _ablation_state, _ablation_events
    global _ablation_event_count, _ablation_complete, _context_store

    _ablation_mode = True
    _ablation_state = {"type": "ablation_waiting", "phase": "idle"}
    _ablation_events.clear()
    _ablation_event_count = 0
    _ablation_complete = False

    # Initialize context store for API endpoints
    if context_db:
        from agentdiag.context import get_context_store
        _context_store = get_context_store(context_db)

    # Use a startup event to delay the ablation thread until uvicorn is
    # actually listening. Without this, the ablation prints output before
    # the web UI is reachable.
    _server_ready = threading.Event()

    @app.on_event("startup")
    async def _on_startup():
        _server_ready.set()

    def _run():
        # Wait up to 10s for the server to start accepting connections
        _server_ready.wait(timeout=10)
        time.sleep(0.5)  # small extra delay for browser to connect
        from scripts.run_ablation import run_ablation as _run_ablation
        try:
            _run_ablation(
                annotations_path=Path(annotations_path),
                traces_root=Path(traces_root),
                output_dir=Path(output_dir),
                split=split,
                splits_file=Path(splits_file) if splits_file else None,
                llm_provider=llm_provider,
                match_window=match_window,
                bootstrap_n=bootstrap_n,
                skip_bootstrap=skip_bootstrap,
                detector_filter=detector_filter,
                modes=modes,
                verbose=verbose,
                context_db=context_db,
                progress_callback=_ablation_progress_callback,
            )
        except Exception as e:
            global _ablation_complete, _ablation_state
            _ablation_state = {
                "type": "ablation_complete",
                "phase": "error",
                "error": str(e),
                "reports": {},
                "comparison_table": "",
                "modes_completed": _ablation_state.get("modes_completed", []),
                "recent_events": _ablation_events[-50:],
            }
            _ablation_complete = True
            import traceback
            traceback.print_exc()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    _suppress_event_loop_closed_errors()

    print(f"agentdiag ablation dashboard")
    print(f"  http://{host}:{port}")
    print(f"  Ablation study will start when server is ready...")
    print()

    uvicorn.run(app, host=host, port=port, log_level="warning")


def _suppress_event_loop_closed_errors():
    """Silence 'Event loop is closed' RuntimeErrors from async client teardown.

    When uvicorn shuts down, httpx/anthropic AsyncClient objects may still
    have pending __del__ callbacks that try to close on the dead loop.
    Python surfaces these as 'Task exception was never retrieved' warnings
    and via sys.unraisablehook. This installs handlers to silence them.
    """
    import asyncio
    import logging

    # 1. Install a silent exception handler on the current event loop
    #    (catches "Task exception was never retrieved" warnings)
    try:
        loop = asyncio.get_event_loop()
        if not loop.is_closed():
            loop.set_exception_handler(lambda _loop, _ctx: None)
    except RuntimeError:
        pass

    # 2. Install an unraisable hook that silences "Event loop is closed"
    #    from __del__ methods of httpx/anthropic async transports
    _orig_hook = sys.unraisablehook

    def _quiet_hook(unraisable):
        if (
            isinstance(unraisable.exc_value, RuntimeError)
            and "Event loop is closed" in str(unraisable.exc_value)
        ):
            return  # swallow silently
        _orig_hook(unraisable)

    sys.unraisablehook = _quiet_hook

    # 3. Suppress asyncio "Task was destroyed but it is pending" log noise
    logging.getLogger("asyncio").setLevel(logging.CRITICAL)
