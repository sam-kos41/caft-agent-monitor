"""Claude Code session log extractor.

Parses the JSONL files produced by Claude Code CLI sessions
(~/.claude/projects/*/SESSION_ID.jsonl) into TraceEvent sequences
suitable for HTA + CAFT analysis.

Claude Code session logs differ from the Anthropic Messages API format:
  - Each line is a self-contained event (user, assistant, progress, system)
  - Assistant messages contain content blocks (thinking, text, tool_use)
  - Tool results come back as user messages with tool_result content blocks
  - Timestamps are ISO 8601 strings, not epoch seconds
  - Token usage is in message.usage (input_tokens, output_tokens)

Usage:
    from agentdiag.adapters.claude_code import ClaudeCodeExtractor

    extractor = ClaudeCodeExtractor()
    sessions = extractor.discover("~/.claude/projects")
    for session in sessions:
        events = extractor.parse_session(session)
        # events is list[TraceEvent] ready for MonitorEngine
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from agentdiag.models import TraceEvent
from agentdiag.adapters.base import AdapterMeta


@dataclass
class SessionInfo:
    """Metadata about a discovered Claude Code session."""
    session_id: str
    path: Path
    project_dir: str
    line_count: int = 0
    first_timestamp: Optional[str] = None
    last_timestamp: Optional[str] = None
    event_count: int = 0

    @property
    def display_name(self) -> str:
        return f"{self.project_dir}/{self.session_id[:8]}"


@dataclass
class _PendingToolUse:
    """Internal: tracks a tool_use waiting for its tool_result."""
    tool_use_id: str
    tool_name: str
    input_hash: str
    timestamp_iso: str
    tokens_in: int
    tokens_out: int


class ClaudeCodeExtractor:
    """Extracts TraceEvent sequences from Claude Code session logs.

    This is NOT a TraceAdapter (which takes pre-loaded data). It operates
    on the filesystem, discovering and parsing session JSONL files.
    """

    meta = AdapterMeta(
        name="claude_code",
        version="1.0.0",
        supported_formats=["jsonl"],
    )

    def discover(
        self,
        root: str | Path,
        min_lines: int = 5,
    ) -> list[SessionInfo]:
        """Discover Claude Code session files under a root directory.

        Args:
            root: Path to search (e.g. ~/.claude/projects or a specific project).
            min_lines: Skip files with fewer lines than this.

        Returns:
            List of SessionInfo, sorted by most recent first.
        """
        root = Path(root).expanduser()
        sessions = []

        # Claude Code stores sessions as PROJECT_DIR/SESSION_ID.jsonl
        for jsonl_path in sorted(root.rglob("*.jsonl")):
            # Skip non-session files (e.g. output files in /tmp)
            if not self._looks_like_session(jsonl_path):
                continue

            line_count = sum(1 for _ in open(jsonl_path, "r", errors="replace"))
            if line_count < min_lines:
                continue

            # Session ID is the filename stem (UUID)
            session_id = jsonl_path.stem
            project_dir = jsonl_path.parent.name

            # Quick scan for first/last timestamps
            first_ts, last_ts = self._scan_timestamps(jsonl_path)

            sessions.append(SessionInfo(
                session_id=session_id,
                path=jsonl_path,
                project_dir=project_dir,
                line_count=line_count,
                first_timestamp=first_ts,
                last_timestamp=last_ts,
            ))

        # Sort by last timestamp (most recent first)
        sessions.sort(
            key=lambda s: s.last_timestamp or "",
            reverse=True,
        )
        return sessions

    def parse_session(self, session: SessionInfo | str | Path) -> list[TraceEvent]:
        """Parse a session file into TraceEvents.

        Args:
            session: SessionInfo, path string, or Path to a JSONL file.

        Returns:
            Ordered list of TraceEvents extracted from the session.
        """
        if isinstance(session, SessionInfo):
            path = session.path
        else:
            path = Path(session).expanduser()

        raw_events = self._load_raw_events(path)
        return self._extract_trace_events(raw_events)

    def parse_session_raw(self, session: SessionInfo | str | Path) -> list[dict]:
        """Load raw events without converting to TraceEvent.

        Useful for debugging or custom analysis.
        """
        if isinstance(session, SessionInfo):
            path = session.path
        else:
            path = Path(session).expanduser()
        return self._load_raw_events(path)

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _looks_like_session(self, path: Path) -> bool:
        """Check if a file looks like a Claude Code session log."""
        stem = path.stem
        # Session IDs are UUIDs (8-4-4-4-12)
        parts = stem.split("-")
        if len(parts) == 5:
            lengths = [len(p) for p in parts]
            if lengths == [8, 4, 4, 4, 12]:
                return True
        return False

    def _scan_timestamps(self, path: Path) -> tuple[Optional[str], Optional[str]]:
        """Quick scan to get first and last timestamps."""
        first_ts = None
        last_ts = None
        with open(path, "r", errors="replace") as f:
            for line in f:
                try:
                    d = json.loads(line.strip())
                    ts = d.get("timestamp")
                    if ts:
                        if first_ts is None:
                            first_ts = ts
                        last_ts = ts
                except (json.JSONDecodeError, AttributeError):
                    continue
        return first_ts, last_ts

    def _load_raw_events(self, path: Path) -> list[dict]:
        """Load all JSON events from a session file."""
        events = []
        with open(path, "r", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return events

    def _extract_trace_events(self, raw_events: list[dict]) -> list[TraceEvent]:
        """Convert raw session events into TraceEvent sequence.

        Strategy:
        - assistant events with tool_use → TraceEvent(type="tool_call")
        - assistant events with thinking → TraceEvent(type="planning")
        - assistant events with text → TraceEvent(type="reasoning")
        - user events with tool_result → update the corresponding tool_call
          with success/failure and latency
        - system events with turn_duration → metadata (skipped)
        - progress events → skipped (hook notifications)

        Tool_use and tool_result are correlated by tool_use_id to compute
        latency and success status.
        """
        trace_events: list[TraceEvent] = []
        step = 0

        # Pending tool_use blocks waiting for their result
        pending: dict[str, _PendingToolUse] = {}
        # Map from tool_use_id → TraceEvent index for back-patching
        tool_event_idx: dict[str, int] = {}

        # Track session start for relative timestamps
        session_start_iso: Optional[str] = None

        for raw in raw_events:
            event_type = raw.get("type")
            timestamp_iso = raw.get("timestamp")

            if session_start_iso is None and timestamp_iso:
                session_start_iso = timestamp_iso

            if event_type == "assistant":
                msg = raw.get("message", {})
                content_blocks = msg.get("content", [])
                usage = msg.get("usage", {})
                tokens_in = usage.get("input_tokens", 0) or 0
                tokens_out = usage.get("output_tokens", 0) or 0

                for block in content_blocks:
                    block_type = block.get("type")

                    if block_type == "tool_use":
                        step += 1
                        tool_name = block.get("name", "unknown")
                        tool_id = block.get("id", "")
                        input_data = block.get("input", {})
                        input_str = json.dumps(input_data, sort_keys=True)
                        input_hash = hashlib.md5(
                            input_str.encode()
                        ).hexdigest()[:12]

                        te = TraceEvent(
                            step=step,
                            type="tool_call",
                            tool=tool_name,
                            success=True,  # default, updated by tool_result
                            tokens_in=tokens_in,
                            tokens_out=tokens_out,
                            timestamp=self._iso_to_relative(
                                timestamp_iso, session_start_iso
                            ),
                            output_hash=input_hash,
                            goal_text=self._extract_goal_from_input(
                                tool_name, input_data
                            ),
                            input_hash=input_hash,
                        )
                        trace_events.append(te)
                        event_idx = len(trace_events) - 1

                        # Track pending for tool_result correlation
                        pending[tool_id] = _PendingToolUse(
                            tool_use_id=tool_id,
                            tool_name=tool_name,
                            input_hash=input_hash,
                            timestamp_iso=timestamp_iso or "",
                            tokens_in=tokens_in,
                            tokens_out=tokens_out,
                        )
                        tool_event_idx[tool_id] = event_idx

                    elif block_type == "thinking":
                        step += 1
                        thinking_text = block.get("thinking", "")
                        te = TraceEvent(
                            step=step,
                            type="planning",
                            tokens_out=tokens_out,
                            timestamp=self._iso_to_relative(
                                timestamp_iso, session_start_iso
                            ),
                            goal_text=self._extract_goal_from_thinking(
                                thinking_text
                            ),
                        )
                        trace_events.append(te)

                    elif block_type == "text":
                        text = block.get("text", "")
                        if not text.strip():
                            continue
                        step += 1
                        te = TraceEvent(
                            step=step,
                            type="reasoning",
                            tokens_out=tokens_out,
                            timestamp=self._iso_to_relative(
                                timestamp_iso, session_start_iso
                            ),
                            goal_text=self._extract_goal_from_thinking(text),
                        )
                        trace_events.append(te)

            elif event_type == "user":
                msg = raw.get("message", {})
                content = msg.get("content", "")

                if isinstance(content, str) and content.strip():
                    # User typed a message — context event
                    step += 1
                    te = TraceEvent(
                        step=step,
                        type="user_input",
                        timestamp=self._iso_to_relative(
                            timestamp_iso, session_start_iso
                        ),
                        goal_text=content[:200] if content else None,
                    )
                    trace_events.append(te)

                elif isinstance(content, list):
                    # Tool results
                    for block in content:
                        if block.get("type") != "tool_result":
                            continue

                        tool_use_id = block.get("tool_use_id", "")
                        is_error = block.get("is_error", False)
                        result_content = block.get("content", "")
                        if isinstance(result_content, list):
                            result_content = " ".join(
                                b.get("text", "")
                                for b in result_content
                                if isinstance(b, dict)
                            )

                        # Back-patch the corresponding tool_call event
                        if tool_use_id in tool_event_idx:
                            idx = tool_event_idx[tool_use_id]
                            te = trace_events[idx]
                            te.success = not is_error
                            if is_error:
                                te.error_message = str(result_content)[:500]

                            # Compute latency from tool_use → tool_result
                            p = pending.get(tool_use_id)
                            if p and timestamp_iso and p.timestamp_iso:
                                latency = self._compute_latency_ms(
                                    p.timestamp_iso, timestamp_iso
                                )
                                if latency is not None:
                                    te.latency_ms = latency

                            # Compute output hash from result
                            if isinstance(result_content, str) and result_content:
                                te.output_hash = hashlib.md5(
                                    result_content[:2000].encode()
                                ).hexdigest()[:12]

                        # Clean up pending
                        pending.pop(tool_use_id, None)

            # Skip progress, system, file-history-snapshot, queue-operation
            # They don't contribute to behavioral trace analysis

        return trace_events

    def _iso_to_relative(
        self,
        timestamp_iso: Optional[str],
        start_iso: Optional[str],
    ) -> Optional[float]:
        """Convert ISO timestamp to seconds from session start."""
        if not timestamp_iso or not start_iso:
            return None
        try:
            t = datetime.fromisoformat(timestamp_iso.replace("Z", "+00:00"))
            s = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
            return max(0.0, (t - s).total_seconds())
        except (ValueError, TypeError):
            return None

    def _compute_latency_ms(
        self,
        start_iso: str,
        end_iso: str,
    ) -> Optional[float]:
        """Compute latency between two ISO timestamps in milliseconds."""
        try:
            s = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
            e = datetime.fromisoformat(end_iso.replace("Z", "+00:00"))
            delta = (e - s).total_seconds() * 1000
            return max(0.0, delta)
        except (ValueError, TypeError):
            return None

    def _extract_goal_from_thinking(self, text: str) -> Optional[str]:
        """Extract goal/intent signal from thinking text.

        Takes the first sentence (up to 200 chars) as a goal indicator
        for drift detection.
        """
        if not text:
            return None
        # First sentence or first 200 chars
        text = text.strip()
        for delim in (".", "\n"):
            idx = text.find(delim)
            if 10 < idx < 200:
                return text[:idx + 1]
        return text[:200] if len(text) > 10 else None

    @staticmethod
    def _short_path(path_str: str) -> str:
        """Return just the filename, or parent/filename for generic names."""
        if not path_str:
            return ""
        p = Path(path_str)
        generic = {"__init__.py", "index.js", "index.ts", "mod.rs", "main.py",
                   "main.go", "main.rs", "setup.py", "conftest.py"}
        if p.name in generic and p.parent.name:
            return f"{p.parent.name}/{p.name}"
        return p.name

    def _extract_goal_from_input(
        self,
        tool_name: str,
        input_data: dict,
    ) -> Optional[str]:
        """Extract goal text from tool input when relevant."""
        if tool_name == "Task":
            desc = input_data.get("description", "")
            prompt = input_data.get("prompt", "")
            return desc[:200] if desc else (prompt[:200] if prompt else None)

        if tool_name in ("Read", "Write"):
            fp = input_data.get("file_path", "")
            if fp:
                return self._short_path(fp)
            return None

        if tool_name == "Edit":
            fp = input_data.get("file_path", "")
            old = input_data.get("old_string", "")
            new = input_data.get("new_string", "")
            if fp:
                old_lines = old.count("\n") + (1 if old else 0)
                new_lines = new.count("\n") + (1 if new else 0)
                return f"{self._short_path(fp)} (+{new_lines}/-{old_lines} lines)"
            return None

        if tool_name == "Bash":
            desc = input_data.get("description", "")
            cmd = input_data.get("command", "")
            text = desc if desc else cmd
            return text[:120] if text else None

        if tool_name == "Grep":
            pat = input_data.get("pattern", "")
            path = input_data.get("path", "")
            if pat:
                short = self._short_path(path) if path else ""
                return f"/{pat[:80]}/ in {short}" if short else f"/{pat[:80]}/"
            return None

        if tool_name == "Glob":
            pat = input_data.get("pattern", "")
            path = input_data.get("path", "")
            if pat:
                short = self._short_path(path) if path else ""
                return f"{pat[:80]} in {short}" if short else pat[:80]
            return None

        if tool_name == "WebSearch":
            q = input_data.get("query", "")
            return f'"{q[:120]}"' if q else None

        if tool_name == "WebFetch":
            url = input_data.get("url", "")
            return url[:120] if url else None

        return None


def discover_sessions(
    root: str | Path = "~/.claude/projects",
    min_lines: int = 5,
) -> list[SessionInfo]:
    """Convenience: discover all Claude Code sessions under a path."""
    return ClaudeCodeExtractor().discover(root, min_lines=min_lines)


def parse_session(session: SessionInfo | str | Path) -> list[TraceEvent]:
    """Convenience: parse a single session file into TraceEvents."""
    return ClaudeCodeExtractor().parse_session(session)
