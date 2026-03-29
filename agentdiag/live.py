"""Live observation mode — tail Claude Code JSONL traces in real-time.

Finds active Claude Code session files, tails them for new events,
pipes through ClaudeCodeAdapter → UniversalMonitor, and serves the
visualization UI.

Usage::

    python -m agentdiag live                        # auto-detect current project
    python -m agentdiag live --session path/to.jsonl # specific session
    python -m agentdiag live --all-sessions          # watch all active sessions
    python -m agentdiag live --replay file.jsonl --speed 10
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Iterator, Optional


def _parse_ts(ts) -> Optional[float]:
    """Parse a timestamp value to epoch seconds."""
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, str):
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return dt.timestamp()
        except (ValueError, TypeError):
            return None
    return None


# ── Claude Code trace file discovery ──────────────────────────────────────

CLAUDE_PROJECTS_DIR = Path.home() / ".claude" / "projects"


def _project_hash(project_dir: str) -> str:
    """Convert a project directory to the Claude Code hash format.

    Claude Code uses the absolute path with '/' replaced by '-'.
    """
    abs_path = str(Path(project_dir).resolve())
    return abs_path.replace("/", "-")


def find_sessions(project_dir: str = ".") -> list[Path]:
    """Find Claude Code JSONL session files for a project.

    Returns files sorted by modification time (most recent first).
    """
    project_hash = _project_hash(project_dir)
    project_path = CLAUDE_PROJECTS_DIR / project_hash

    if not project_path.exists():
        return []

    sessions = list(project_path.glob("*.jsonl"))
    # Also check for subagent sessions
    for subdir in project_path.glob("*/subagents"):
        sessions.extend(subdir.glob("*.jsonl"))

    sessions.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return sessions


def find_all_sessions() -> list[Path]:
    """Find all Claude Code JSONL session files across all projects."""
    if not CLAUDE_PROJECTS_DIR.exists():
        return []

    sessions = list(CLAUDE_PROJECTS_DIR.glob("**/*.jsonl"))
    sessions.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return sessions


# ── File tailing ──────────────────────────────────────────────────────────

def _tail_jsonl(
    path: Path,
    *,
    from_start: bool = False,
    speed: float = 1.0,
) -> Iterator[dict]:
    """Tail a JSONL file, yielding new JSON objects as they appear.

    Args:
        path: Path to the JSONL file.
        from_start: If True (replay mode), read from beginning. Otherwise seek to end.
        speed: For replay mode, controls inter-event delay based on timestamps.
    """
    prev_ts: Optional[float] = None

    with open(path, "r") as f:
        if not from_start:
            f.seek(0, 2)  # seek to end for live mode

        while True:
            line = f.readline()
            if line.strip():
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # In replay mode, pace events by timestamp deltas
                if from_start and speed > 0:
                    ts = _parse_ts(data.get("timestamp"))
                    if ts is not None and prev_ts is not None:
                        delay = (ts - prev_ts) / speed
                        if 0 < delay < 5.0:
                            time.sleep(delay)
                    if ts is not None:
                        prev_ts = ts

                yield data
            else:
                if from_start and not f.readline():
                    # In replay mode, EOF means we're done
                    break
                time.sleep(0.5)  # poll interval for live mode


# ── Live → visualization bridge ──────────────────────────────────────────

class _LiveQueueStream:
    """Thread-safe bridge: live tailer writes lines, ingestion thread reads.

    Implements the file-like interface that start_server() expects.
    """

    def __init__(self) -> None:
        self._q: queue.Queue[Optional[str]] = queue.Queue()

    def write_event(self, data: dict) -> None:
        """Write an event dict as a JSONL line."""
        self._q.put(json.dumps(data, default=str) + "\n")

    def close(self) -> None:
        """Signal end of stream."""
        self._q.put(None)

    def __iter__(self):
        return self

    def __next__(self) -> str:
        item = self._q.get()
        if item is None:
            raise StopIteration
        return item

    def readline(self) -> str:
        try:
            item = self._q.get(timeout=1.0)
            if item is None:
                return ""
            return item
        except queue.Empty:
            return ""


# Persistent state for tool name tracking across messages.
# tool_use and tool_result arrive in separate JSONL entries,
# so we carry forward the last seen tool name.
_last_tool_names: list[str] = []


def _extract_trace_events_from_cc(raw: dict, step_counter: list[int]) -> list[dict]:
    """Extract TraceEvent-compatible dicts from a raw Claude Code JSONL entry.

    Raw CC entries have type in {user, assistant, file-history-snapshot, progress}.
    Tool usage is embedded in assistant messages with content type 'tool_use',
    and results come as user messages with content type 'tool_result'.

    tool_use blocks have a 'name' field.  tool_result blocks do NOT — they
    arrive in subsequent 'user' messages.  We track tool names from tool_use
    blocks and pair them with the next tool_result blocks.
    """
    global _last_tool_names

    msg_type = raw.get("type", "")
    message = raw.get("message", {})
    content = message.get("content", "")

    if not isinstance(content, list):
        return []

    events = []
    ts = _parse_ts(raw.get("timestamp")) or time.time()

    for block in content:
        block_type = block.get("type", "")

        if block_type == "tool_use":
            step_counter[0] += 1
            tool_name = block.get("name", "unknown")
            _last_tool_names.append(tool_name)

            # Extract target_path from tool input for read/write classification
            tool_input = block.get("input", {})
            target_path = (
                tool_input.get("file_path")
                or tool_input.get("path")
                or tool_input.get("pattern")  # Glob/Grep
                or tool_input.get("command", "")[:80]  # Bash — first 80 chars
            )

            events.append({
                "step": step_counter[0],
                "type": "tool_call",
                "tool": tool_name,
                "target_path": target_path if isinstance(target_path, str) else None,
                "latency_ms": 0.0,
                "success": True,
                "tokens_in": len(json.dumps(tool_input, default=str)) // 4,
                "tokens_out": 0,
                "timestamp": ts,
            })

        elif block_type == "tool_result":
            result_content = block.get("content", "")
            if isinstance(result_content, str):
                tokens_out = len(result_content) // 4
            elif isinstance(result_content, list):
                tokens_out = sum(len(json.dumps(c, default=str)) for c in result_content) // 4
            else:
                tokens_out = 0

            is_error = block.get("is_error", False)

            if events:
                # Pair with preceding tool_use in same message
                events[-1]["tokens_out"] = tokens_out
                events[-1]["success"] = not is_error
            elif _last_tool_names:
                # Pair with tool_use from previous message
                step_counter[0] += 1
                tool_name = _last_tool_names.pop(0)
                events.append({
                    "step": step_counter[0],
                    "type": "tool_call",
                    "tool": tool_name,
                    "latency_ms": 0.0,
                    "success": not is_error,
                    "tokens_in": 0,
                    "tokens_out": tokens_out,
                    "timestamp": ts,
                })
            else:
                # Truly orphan — no tool name available
                step_counter[0] += 1
                events.append({
                    "step": step_counter[0],
                    "type": "tool_call",
                    "tool": "unknown",
                    "latency_ms": 0.0,
                    "success": not is_error,
                    "tokens_in": 0,
                    "tokens_out": tokens_out,
                    "timestamp": ts,
                })

        elif block_type == "thinking":
            step_counter[0] += 1
            text = block.get("thinking", "")
            events.append({
                "step": step_counter[0],
                "type": "reasoning",
                "tool": None,
                "latency_ms": 0.0,
                "success": True,
                "tokens_in": 0,
                "tokens_out": len(text) // 4,
                "timestamp": ts,
            })

        elif block_type == "text":
            # Assistant text output — treat as reasoning/output
            text = block.get("text", "")
            if text.strip():
                step_counter[0] += 1
                events.append({
                    "step": step_counter[0],
                    "type": "output",
                    "tool": None,
                    "latency_ms": 0.0,
                    "success": True,
                    "tokens_in": 0,
                    "tokens_out": len(text) // 4,
                    "timestamp": ts,
                })

    return events


def _tail_and_feed(
    session_path: Path,
    stream: _LiveQueueStream,
    *,
    replay: bool = False,
    speed: float = 1.0,
) -> None:
    """Tail a session file and push TraceEvent-compatible dicts to the stream.

    For raw Claude Code JSONL, extracts tool_use/tool_result/thinking blocks
    and converts them to TraceEvent format for the visualization pipeline.

    Runs in a background thread.
    """
    step_counter = [0]

    try:
        for raw in _tail_jsonl(session_path, from_start=replay, speed=speed):
            events = _extract_trace_events_from_cc(raw, step_counter)
            for event in events:
                stream.write_event(event)
    except Exception as e:
        print(f"Error tailing {session_path}: {e}", file=sys.stderr)
    finally:
        stream.close()


# ── Server mode ───────────────────────────────────────────────────────────

def _run_server(
    session_path: Path,
    *,
    port: int = 8080,
    open_browser: bool = True,
    replay: bool = False,
    speed: float = 1.0,
    z_threshold: float = 3.0,
) -> None:
    """Run live observation with the visualization server."""
    try:
        from agentdiag.visualize import start_server
    except ImportError:
        print("Visualization requires: pip install uvicorn fastapi")
        sys.exit(1)

    stream = _LiveQueueStream()

    # Start tailing in background thread
    tailer = threading.Thread(
        target=_tail_and_feed,
        args=(session_path, stream),
        kwargs={"replay": replay, "speed": speed},
        daemon=True,
    )
    tailer.start()

    session_id = session_path.stem[:8]
    mode = "replay" if replay else "live"
    print(f"Live: {mode} mode on {session_path.name}")
    print(f"Live: serving at http://localhost:{port}")
    if replay:
        print(f"Live: replay speed {speed}x")
    else:
        print(f"Live: tailing for new events (poll every 500ms)")
    print()

    if open_browser:
        def _open():
            time.sleep(1.5)
            import webbrowser
            webbrowser.open(f"http://localhost:{port}")
        threading.Thread(target=_open, daemon=True).start()

    start_server(
        stream=stream,
        goal=f"Live observation: {session_path.name}",
        port=port,
        delay=0.0,  # pacing handled by tailer
        cognitive=True,
        input_path=str(session_path),
    )


# ── Multi-session mode ───────────────────────────────────────────────────

def _run_multi_server(
    session_paths: list[Path],
    *,
    port: int = 8080,
    open_browser: bool = True,
    z_threshold: float = 3.0,
) -> None:
    """Run live observation on multiple sessions.

    For now, picks the most recent session. Multi-session tabbed view
    is a future enhancement.
    """
    if not session_paths:
        print("No active sessions found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(session_paths)} session(s):")
    for i, p in enumerate(session_paths[:5]):
        age = time.time() - p.stat().st_mtime
        if age < 60:
            age_str = f"{age:.0f}s ago"
        elif age < 3600:
            age_str = f"{age / 60:.0f}m ago"
        else:
            age_str = f"{age / 3600:.1f}h ago"
        marker = " ← watching" if i == 0 else ""
        print(f"  {p.stem[:8]}  {age_str}  {p.stat().st_size / 1024:.0f}KB{marker}")

    print()
    _run_server(
        session_paths[0],
        port=port,
        open_browser=open_browser,
        z_threshold=z_threshold,
    )


# ── Validation logging ─────────────────────────────────────────────────────

def _keyboard_listener(
    vlog: "ValidationLog",
    stop_event: threading.Event,
) -> None:
    """Listen for keystrokes in the terminal to record human marks.

    Keys:
      s / S  → mark "struggling"  (agent appears stuck or confused)
      f / F  → mark "fine"        (agent is making normal progress)
      q / Q  → quit listener

    Uses raw terminal mode on Unix; falls back to simple input() on Windows.
    """
    import select

    print("Validation: press [s] = struggling, [f] = fine, [q] = quit listener")

    try:
        import tty
        import termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)

        try:
            while not stop_event.is_set():
                if select.select([sys.stdin], [], [], 0.5)[0]:
                    ch = sys.stdin.read(1)
                    if ch in ("s", "S"):
                        vlog.mark_struggling()
                        print(f"\r  ⚠ STRUGGLING marked at step {vlog._current_step}  ", end="", flush=True)
                    elif ch in ("f", "F"):
                        vlog.mark_fine()
                        print(f"\r  ✓ FINE marked at step {vlog._current_step}        ", end="", flush=True)
                    elif ch in ("q", "Q"):
                        print("\r  Validation listener stopped.                        ")
                        break
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    except (ImportError, OSError):
        # Windows or non-TTY: fall back to line-buffered input
        while not stop_event.is_set():
            try:
                line = input()
                ch = line.strip().lower()[:1]
                if ch == "s":
                    vlog.mark_struggling()
                    print(f"  STRUGGLING marked at step {vlog._current_step}")
                elif ch == "f":
                    vlog.mark_fine()
                    print(f"  FINE marked at step {vlog._current_step}")
                elif ch == "q":
                    print("  Validation listener stopped.")
                    break
            except EOFError:
                break


def _tail_and_feed_with_validation(
    session_path: Path,
    stream: _LiveQueueStream,
    vlog: "ValidationLog",
    *,
    replay: bool = False,
    speed: float = 1.0,
) -> None:
    """Tail + feed, updating the validation log's position on each event."""
    step_counter = [0]

    try:
        for raw in _tail_jsonl(session_path, from_start=replay, speed=speed):
            events = _extract_trace_events_from_cc(raw, step_counter)
            for event in events:
                stream.write_event(event)
                vlog.update_position(
                    step=event.get("step", 0),
                    event_count=step_counter[0],
                )
    except Exception as e:
        print(f"Error tailing {session_path}: {e}", file=sys.stderr)
    finally:
        stream.close()


def _run_server_with_validation(
    session_path: Path,
    validation_log_path: str,
    *,
    port: int = 8080,
    open_browser: bool = True,
    replay: bool = False,
    speed: float = 1.0,
    z_threshold: float = 3.0,
) -> None:
    """Run live observation with validation logging enabled."""
    try:
        from agentdiag.visualize import start_server
    except ImportError:
        print("Visualization requires: pip install uvicorn fastapi")
        sys.exit(1)

    from agentdiag.validation_log import ValidationLog

    vlog = ValidationLog(validation_log_path)
    vlog.start_session(
        goal=f"Validation: {session_path.name}",
        source=str(session_path),
    )

    stream = _LiveQueueStream()
    stop_event = threading.Event()

    # Start tailing with validation position tracking
    tailer = threading.Thread(
        target=_tail_and_feed_with_validation,
        args=(session_path, stream, vlog),
        kwargs={"replay": replay, "speed": speed},
        daemon=True,
    )
    tailer.start()

    # Start keyboard listener
    kb_thread = threading.Thread(
        target=_keyboard_listener,
        args=(vlog, stop_event),
        daemon=True,
    )
    kb_thread.start()

    session_id = session_path.stem[:8]
    mode = "replay" if replay else "live"
    print(f"Live+Validation: {mode} mode on {session_path.name}")
    print(f"Live+Validation: logging to {validation_log_path}")
    print(f"Live+Validation: serving at http://localhost:{port}")
    print()

    if open_browser:
        def _open():
            time.sleep(1.5)
            import webbrowser
            webbrowser.open(f"http://localhost:{port}")
        threading.Thread(target=_open, daemon=True).start()

    try:
        start_server(
            stream=stream,
            goal=f"Validation: {session_path.name}",
            port=port,
            delay=0.0,
            cognitive=True,
            input_path=str(session_path),
        )
    finally:
        stop_event.set()
        vlog.end_session(event_count=vlog._current_event_count)
        print(f"\nValidation log saved: {vlog.path}")
        print(f"  Human marks: {vlog.human_marks_count}")
        print(f"  System detections: {vlog.detection_count}")
        print(f"\nRun: python scripts/compare_validation.py {vlog.path}")


# ── Harness replay ────────────────────────────────────────────────────────

def _replay_harness_and_feed(
    result_path: Path,
    stream: _LiveQueueStream,
    *,
    speed: float = 1.0,
) -> None:
    """Replay a saved HarnessResult JSON through the visualization.

    Uses HarnessLogAdapter to reconstruct the ObservableEvent stream,
    then converts each event to TraceEvent-compatible dicts for the
    visualization pipeline.

    Runs in a background thread.
    """
    from agentdiag.adapters.harness_adapter import HarnessLogAdapter

    try:
        result_dict = json.loads(result_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"Error loading harness result: {e}", file=sys.stderr)
        stream.close()
        return

    adapter = HarnessLogAdapter()
    events = adapter.replay(result_dict, base_timestamp=time.time())

    prev_ts = None
    for event in events:
        # Pace events by timestamp deltas
        if speed > 0 and prev_ts is not None:
            delay = (event.timestamp - prev_ts) / speed
            if 0 < delay < 5.0:
                time.sleep(delay)
        prev_ts = event.timestamp

        event_dict = event.to_dict()
        trace_event = {
            "step": event.step,
            "type": event_dict.get("event_type", "tool_call"),
            "tool": event_dict.get("tool_name") or event_dict.get("event_type", "harness"),
            "latency_ms": event_dict.get("duration_ms", 0.0),
            "success": True,
            "tokens_in": event_dict.get("input_tokens", 0) or event_dict.get("token_count", 0) or 0,
            "tokens_out": event_dict.get("output_tokens", 0) or 0,
            "timestamp": event_dict.get("timestamp", time.time()),
            "goal_text": event_dict.get("symbol", ""),
        }
        stream.write_event(trace_event)

    stream.close()


def _run_harness_replay(
    result_path: Path,
    *,
    port: int = 8080,
    open_browser: bool = True,
    speed: float = 1.0,
) -> None:
    """Run harness replay with the visualization server."""
    try:
        from agentdiag.visualize import start_server
    except ImportError:
        print("Visualization requires: pip install uvicorn fastapi")
        sys.exit(1)

    # Load goal from the result file
    try:
        result_dict = json.loads(result_path.read_text(encoding="utf-8"))
        goal = result_dict.get("goal", "Harness replay")
    except Exception:
        goal = "Harness replay"

    stream = _LiveQueueStream()

    tailer = threading.Thread(
        target=_replay_harness_and_feed,
        args=(result_path, stream),
        kwargs={"speed": speed},
        daemon=True,
    )
    tailer.start()

    n_sprints = len(result_dict.get("sprints", []))
    passed = result_dict.get("overall_passed", False)
    print(f"Harness replay: {result_path.name}")
    print(f"  Goal: {goal}")
    print(f"  Sprints: {n_sprints}, Passed: {passed}")
    print(f"  Visualization: http://localhost:{port}")
    print(f"  Speed: {speed}x")
    print()

    if open_browser:
        def _open():
            time.sleep(1.5)
            import webbrowser
            webbrowser.open(f"http://localhost:{port}")
        threading.Thread(target=_open, daemon=True).start()

    start_server(
        stream=stream,
        goal=f"Replay: {goal}",
        port=port,
        delay=0.0,
        cognitive=True,
        input_path=str(result_path),
    )


# ── CLI ───────────────────────────────────────────────────────────────────

def _run_from_args(args: argparse.Namespace) -> None:
    """Execute live mode from a pre-parsed argparse Namespace.

    Called by __main__.py when 'live' subcommand is used.
    """
    # Harness replay mode
    if getattr(args, "replay_harness", None):
        harness_path = Path(args.replay_harness)
        if not harness_path.exists():
            print(f"Error: File not found: {harness_path}", file=sys.stderr)
            sys.exit(1)
        _run_harness_replay(
            harness_path,
            port=args.port,
            open_browser=not args.no_browser,
            speed=args.speed,
        )
        return

    # Resolve validation log path if requested
    validation_path = getattr(args, "log_validation", None)

    # Replay mode
    if getattr(args, "replay", None):
        replay_path = Path(args.replay)
        if not replay_path.exists():
            print(f"Error: File not found: {replay_path}", file=sys.stderr)
            sys.exit(1)
        if validation_path:
            _run_server_with_validation(
                replay_path, validation_path,
                port=args.port, open_browser=not args.no_browser,
                replay=True, speed=args.speed, z_threshold=args.z_threshold,
            )
        else:
            _run_server(
                replay_path,
                port=args.port, open_browser=not args.no_browser,
                replay=True, speed=args.speed, z_threshold=args.z_threshold,
            )
        return

    # Explicit session
    if getattr(args, "session", None):
        session_path = Path(args.session)
        if not session_path.exists():
            print(f"Error: File not found: {session_path}", file=sys.stderr)
            sys.exit(1)
        if validation_path:
            _run_server_with_validation(
                session_path, validation_path,
                port=args.port, open_browser=not args.no_browser,
                z_threshold=args.z_threshold,
            )
        else:
            _run_server(
                session_path,
                port=args.port, open_browser=not args.no_browser,
                z_threshold=args.z_threshold,
            )
        return

    # Auto-detect sessions
    project = getattr(args, "project", ".")
    if getattr(args, "all_sessions", False):
        sessions = find_all_sessions()
    else:
        sessions = find_sessions(project)

    if not sessions:
        project_hash = _project_hash(project)
        print("No active Claude Code sessions found.")
        print()
        print("Looked in:")
        print(f"  {CLAUDE_PROJECTS_DIR / project_hash}")
        print()
        print("Try one of:")
        print(f"  python -m agentdiag live --session <path-to-session.jsonl>")
        print(f"  python -m agentdiag live --all-sessions")
        print(f"  python -m agentdiag live --replay <past-session.jsonl>")
        print()
        print("Claude Code sessions are stored in:")
        print(f"  {CLAUDE_PROJECTS_DIR}/")
        sys.exit(1)

    if validation_path:
        _run_server_with_validation(
            sessions[0], validation_path,
            port=args.port, open_browser=not args.no_browser,
            z_threshold=args.z_threshold,
        )
    else:
        _run_multi_server(
            sessions,
            port=args.port,
            open_browser=not args.no_browser,
            z_threshold=args.z_threshold,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live agent observation — watch Claude Code sessions in real-time",
    )
    parser.add_argument(
        "--session", type=str, default=None,
        help="Path to specific JSONL trace file",
    )
    parser.add_argument(
        "--project", type=str, default=".",
        help="Project directory to find sessions for (default: cwd)",
    )
    parser.add_argument(
        "--all-sessions", action="store_true",
        help="Watch all active sessions (picks most recent)",
    )
    parser.add_argument(
        "--replay", type=str, default=None,
        help="Replay a past session JSONL file",
    )
    parser.add_argument(
        "--replay-harness", type=str, default=None,
        help="Replay a saved HarnessResult JSON file",
    )
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="Replay speed multiplier (default: 1.0)",
    )
    parser.add_argument(
        "--port", type=int, default=8080,
        help="Visualization server port (default: 8080)",
    )
    parser.add_argument(
        "--no-browser", action="store_true",
        help="Don't auto-open browser",
    )
    parser.add_argument(
        "--z-threshold", type=float, default=3.0,
        help="Anomaly detection sensitivity (default: 3.0)",
    )
    parser.add_argument(
        "--log-validation", type=str, default=None,
        help="Path to validation log JSONL (enables human mark mode)",
    )

    args = parser.parse_args()

    # Harness replay mode
    if args.replay_harness:
        harness_path = Path(args.replay_harness)
        if not harness_path.exists():
            print(f"Error: File not found: {harness_path}", file=sys.stderr)
            sys.exit(1)
        _run_harness_replay(
            harness_path,
            port=args.port,
            open_browser=not args.no_browser,
            speed=args.speed,
        )
        return

    validation_path = getattr(args, "log_validation", None)

    # Replay mode
    if args.replay:
        replay_path = Path(args.replay)
        if not replay_path.exists():
            print(f"Error: File not found: {replay_path}", file=sys.stderr)
            sys.exit(1)
        if validation_path:
            _run_server_with_validation(
                replay_path, validation_path,
                port=args.port, open_browser=not args.no_browser,
                replay=True, speed=args.speed, z_threshold=args.z_threshold,
            )
        else:
            _run_server(
                replay_path,
                port=args.port, open_browser=not args.no_browser,
                replay=True, speed=args.speed, z_threshold=args.z_threshold,
            )
        return

    # Explicit session
    if args.session:
        session_path = Path(args.session)
        if not session_path.exists():
            print(f"Error: File not found: {session_path}", file=sys.stderr)
            sys.exit(1)
        if validation_path:
            _run_server_with_validation(
                session_path, validation_path,
                port=args.port, open_browser=not args.no_browser,
                z_threshold=args.z_threshold,
            )
        else:
            _run_server(
                session_path,
                port=args.port, open_browser=not args.no_browser,
                z_threshold=args.z_threshold,
            )
        return

    # Auto-detect sessions
    if args.all_sessions:
        sessions = find_all_sessions()
    else:
        sessions = find_sessions(args.project)

    if not sessions:
        project_hash = _project_hash(args.project)
        print("No active Claude Code sessions found.")
        print()
        print("Looked in:")
        print(f"  {CLAUDE_PROJECTS_DIR / project_hash}")
        print()
        print("Try one of:")
        print(f"  python -m agentdiag live --session <path-to-session.jsonl>")
        print(f"  python -m agentdiag live --all-sessions")
        print(f"  python -m agentdiag live --replay <past-session.jsonl>")
        print()
        print("Claude Code sessions are stored in:")
        print(f"  {CLAUDE_PROJECTS_DIR}/")
        sys.exit(1)

    if validation_path:
        _run_server_with_validation(
            sessions[0], validation_path,
            port=args.port, open_browser=not args.no_browser,
            z_threshold=args.z_threshold,
        )
    else:
        _run_multi_server(
            sessions,
            port=args.port,
            open_browser=not args.no_browser,
            z_threshold=args.z_threshold,
        )


if __name__ == "__main__":
    main()
