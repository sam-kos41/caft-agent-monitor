"""Tests for the Claude Code session log extractor."""

import json
import tempfile
from pathlib import Path

import pytest

from agentdiag.adapters.claude_code import (
    ClaudeCodeExtractor,
    SessionInfo,
    discover_sessions,
    parse_session,
)
from agentdiag.models import TraceEvent


def _write_session(tmp_dir: Path, session_id: str, events: list[dict]) -> Path:
    """Write a list of dicts as JSONL to a file named like a session."""
    path = tmp_dir / f"{session_id}.jsonl"
    with open(path, "w") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")
    return path


# A minimal valid session ID (UUID format)
VALID_UUID = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
VALID_UUID_2 = "11111111-2222-3333-4444-555555555555"


def _make_assistant_tool_use(tool_name: str, tool_id: str, timestamp: str, **input_kw) -> dict:
    return {
        "type": "assistant",
        "timestamp": timestamp,
        "message": {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": tool_id,
                    "name": tool_name,
                    "input": input_kw,
                }
            ],
            "usage": {"input_tokens": 100, "output_tokens": 50},
        },
    }


def _make_tool_result(tool_id: str, content: str, timestamp: str, is_error: bool = False) -> dict:
    return {
        "type": "user",
        "timestamp": timestamp,
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": content,
                    "is_error": is_error,
                }
            ],
        },
    }


def _make_assistant_thinking(text: str, timestamp: str) -> dict:
    return {
        "type": "assistant",
        "timestamp": timestamp,
        "message": {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": text}
            ],
            "usage": {"input_tokens": 50, "output_tokens": 30},
        },
    }


def _make_assistant_text(text: str, timestamp: str) -> dict:
    return {
        "type": "assistant",
        "timestamp": timestamp,
        "message": {
            "role": "assistant",
            "content": [
                {"type": "text", "text": text}
            ],
            "usage": {"input_tokens": 20, "output_tokens": 10},
        },
    }


def _make_user_input(text: str, timestamp: str) -> dict:
    return {
        "type": "user",
        "timestamp": timestamp,
        "message": {
            "role": "user",
            "content": text,
        },
    }


class TestClaudeCodeExtractor:
    def test_discover_finds_uuid_sessions(self, tmp_path):
        _write_session(tmp_path, VALID_UUID, [{"type": "x"}] * 10)
        _write_session(tmp_path, VALID_UUID_2, [{"type": "y"}] * 10)
        # Non-UUID file should be skipped
        (tmp_path / "not-a-session.jsonl").write_text("test\n" * 10)

        sessions = ClaudeCodeExtractor().discover(tmp_path)
        assert len(sessions) == 2
        ids = {s.session_id for s in sessions}
        assert VALID_UUID in ids
        assert VALID_UUID_2 in ids

    def test_discover_skips_short_files(self, tmp_path):
        _write_session(tmp_path, VALID_UUID, [{"type": "x"}] * 3)
        sessions = ClaudeCodeExtractor().discover(tmp_path, min_lines=5)
        assert len(sessions) == 0

    def test_discover_extracts_timestamps(self, tmp_path):
        events = [
            {"type": "user", "timestamp": "2026-01-01T10:00:00Z"},
            {"type": "assistant", "timestamp": "2026-01-01T10:05:00Z"},
            {"type": "assistant", "timestamp": "2026-01-01T10:10:00Z"},
        ] + [{"type": "x"}] * 5  # pad to min_lines
        _write_session(tmp_path, VALID_UUID, events)
        sessions = ClaudeCodeExtractor().discover(tmp_path)
        assert sessions[0].first_timestamp == "2026-01-01T10:00:00Z"
        assert sessions[0].last_timestamp == "2026-01-01T10:10:00Z"

    def test_parse_tool_use_events(self, tmp_path):
        events = [
            _make_assistant_tool_use("Read", "t1", "2026-01-01T10:00:00Z", file_path="/foo.py"),
            _make_tool_result("t1", "contents of foo.py", "2026-01-01T10:00:05Z"),
        ]
        path = _write_session(tmp_path, VALID_UUID, events)

        traces = ClaudeCodeExtractor().parse_session(path)
        assert len(traces) == 1  # tool_use produces event, tool_result back-patches it
        assert traces[0].type == "tool_call"
        assert traces[0].tool == "Read"
        assert traces[0].success is True

    def test_parse_tool_use_with_error(self, tmp_path):
        events = [
            _make_assistant_tool_use("Bash", "t1", "2026-01-01T10:00:00Z", command="rm -rf /"),
            _make_tool_result("t1", "Permission denied", "2026-01-01T10:00:02Z", is_error=True),
        ]
        path = _write_session(tmp_path, VALID_UUID, events)

        traces = ClaudeCodeExtractor().parse_session(path)
        assert traces[0].success is False
        assert "Permission denied" in traces[0].error_message

    def test_parse_latency_computation(self, tmp_path):
        events = [
            _make_assistant_tool_use("Read", "t1", "2026-01-01T10:00:00.000Z", file_path="/a"),
            _make_tool_result("t1", "data", "2026-01-01T10:00:03.500Z"),
        ]
        path = _write_session(tmp_path, VALID_UUID, events)

        traces = ClaudeCodeExtractor().parse_session(path)
        assert 3400 < traces[0].latency_ms < 3600  # ~3500ms

    def test_parse_thinking_events(self, tmp_path):
        events = [
            _make_assistant_thinking("I need to analyze the code structure.", "2026-01-01T10:00:00Z"),
        ]
        path = _write_session(tmp_path, VALID_UUID, events)

        traces = ClaudeCodeExtractor().parse_session(path)
        assert len(traces) == 1
        assert traces[0].type == "planning"
        assert traces[0].goal_text is not None
        assert "analyze" in traces[0].goal_text

    def test_parse_text_events(self, tmp_path):
        events = [
            _make_assistant_text("Here is the implementation plan.", "2026-01-01T10:00:00Z"),
        ]
        path = _write_session(tmp_path, VALID_UUID, events)

        traces = ClaudeCodeExtractor().parse_session(path)
        assert len(traces) == 1
        assert traces[0].type == "reasoning"
        assert traces[0].goal_text is not None
        assert "implementation plan" in traces[0].goal_text

    def test_parse_user_input_events(self, tmp_path):
        events = [
            _make_user_input("Fix the login bug", "2026-01-01T10:00:00Z"),
        ]
        path = _write_session(tmp_path, VALID_UUID, events)

        traces = ClaudeCodeExtractor().parse_session(path)
        assert len(traces) == 1
        assert traces[0].type == "user_input"
        assert "login bug" in traces[0].goal_text

    def test_parse_empty_text_skipped(self, tmp_path):
        events = [
            _make_assistant_text("", "2026-01-01T10:00:00Z"),
            _make_assistant_text("   ", "2026-01-01T10:00:01Z"),
        ]
        path = _write_session(tmp_path, VALID_UUID, events)

        traces = ClaudeCodeExtractor().parse_session(path)
        assert len(traces) == 0

    def test_parse_relative_timestamps(self, tmp_path):
        events = [
            _make_assistant_text("First", "2026-01-01T10:00:00Z"),
            _make_assistant_text("Second", "2026-01-01T10:00:30Z"),
            _make_assistant_text("Third", "2026-01-01T10:01:00Z"),
        ]
        path = _write_session(tmp_path, VALID_UUID, events)

        traces = ClaudeCodeExtractor().parse_session(path)
        assert traces[0].timestamp == 0.0
        assert abs(traces[1].timestamp - 30.0) < 0.01
        assert abs(traces[2].timestamp - 60.0) < 0.01

    def test_parse_step_numbering(self, tmp_path):
        events = [
            _make_user_input("Hello", "2026-01-01T10:00:00Z"),
            _make_assistant_thinking("Plan.", "2026-01-01T10:00:01Z"),
            _make_assistant_tool_use("Read", "t1", "2026-01-01T10:00:02Z"),
            _make_tool_result("t1", "data", "2026-01-01T10:00:05Z"),
            _make_assistant_text("Done", "2026-01-01T10:00:06Z"),
        ]
        path = _write_session(tmp_path, VALID_UUID, events)

        traces = ClaudeCodeExtractor().parse_session(path)
        steps = [t.step for t in traces]
        assert steps == [1, 2, 3, 4]  # user_input, planning, tool_call, reasoning

    def test_parse_output_hash_from_result(self, tmp_path):
        events = [
            _make_assistant_tool_use("Read", "t1", "2026-01-01T10:00:00Z", file_path="/f"),
            _make_tool_result("t1", "file content here", "2026-01-01T10:00:05Z"),
        ]
        path = _write_session(tmp_path, VALID_UUID, events)

        traces = ClaudeCodeExtractor().parse_session(path)
        # output_hash should be updated from tool_result content
        assert traces[0].output_hash is not None

    def test_parse_task_tool_extracts_goal(self, tmp_path):
        events = [
            _make_assistant_tool_use(
                "Task", "t1", "2026-01-01T10:00:00Z",
                subagent_type="Explore",
                description="Find authentication code",
                prompt="Search for auth-related files",
            ),
            _make_tool_result("t1", "Found 3 files", "2026-01-01T10:00:10Z"),
        ]
        path = _write_session(tmp_path, VALID_UUID, events)

        traces = ClaudeCodeExtractor().parse_session(path)
        assert traces[0].goal_text == "Find authentication code"

    def test_parse_skips_progress_events(self, tmp_path):
        events = [
            {"type": "progress", "data": {"type": "hook_progress"}, "timestamp": "2026-01-01T10:00:00Z"},
            _make_assistant_text("Hello", "2026-01-01T10:00:01Z"),
        ]
        path = _write_session(tmp_path, VALID_UUID, events)

        traces = ClaudeCodeExtractor().parse_session(path)
        assert len(traces) == 1

    def test_parse_skips_system_events(self, tmp_path):
        events = [
            {"type": "system", "subtype": "turn_duration", "durationMs": 5000, "timestamp": "2026-01-01T10:00:00Z"},
            _make_assistant_text("Hello", "2026-01-01T10:00:01Z"),
        ]
        path = _write_session(tmp_path, VALID_UUID, events)

        traces = ClaudeCodeExtractor().parse_session(path)
        assert len(traces) == 1

    def test_parse_handles_malformed_json(self, tmp_path):
        path = tmp_path / f"{VALID_UUID}.jsonl"
        with open(path, "w") as f:
            f.write("not json\n")
            f.write(json.dumps(_make_assistant_text("OK", "2026-01-01T10:00:00Z")) + "\n")
            f.write("{broken\n")

        traces = ClaudeCodeExtractor().parse_session(path)
        assert len(traces) == 1

    def test_parse_full_lifecycle(self, tmp_path):
        """A realistic mini-session: user asks, agent thinks, reads, edits, responds."""
        events = [
            _make_user_input("Fix the bug in auth.py", "2026-01-01T10:00:00Z"),
            _make_assistant_thinking("I need to read auth.py first to understand the bug.", "2026-01-01T10:00:01Z"),
            _make_assistant_tool_use("Read", "t1", "2026-01-01T10:00:02Z", file_path="/auth.py"),
            _make_tool_result("t1", "def login(): pass", "2026-01-01T10:00:04Z"),
            _make_assistant_thinking("I see the issue. The login function is empty.", "2026-01-01T10:00:05Z"),
            _make_assistant_tool_use("Edit", "t2", "2026-01-01T10:00:06Z", file_path="/auth.py", old_string="pass", new_string="return True"),
            _make_tool_result("t2", "File edited", "2026-01-01T10:00:07Z"),
            _make_assistant_text("I fixed the login function.", "2026-01-01T10:00:08Z"),
        ]
        path = _write_session(tmp_path, VALID_UUID, events)

        traces = ClaudeCodeExtractor().parse_session(path)
        types = [t.type for t in traces]
        assert types == ["user_input", "planning", "tool_call", "planning", "tool_call", "reasoning"]
        assert traces[2].tool == "Read"
        assert traces[4].tool == "Edit"
        assert traces[2].success is True
        assert traces[4].success is True

    def test_parse_tokens_from_usage(self, tmp_path):
        events = [
            _make_assistant_tool_use("Read", "t1", "2026-01-01T10:00:00Z", file_path="/f"),
            _make_tool_result("t1", "data", "2026-01-01T10:00:01Z"),
        ]
        path = _write_session(tmp_path, VALID_UUID, events)

        traces = ClaudeCodeExtractor().parse_session(path)
        assert traces[0].tokens_in == 100
        assert traces[0].tokens_out == 50


    def test_parse_read_extracts_file_path(self, tmp_path):
        events = [
            _make_assistant_tool_use("Read", "t1", "2026-01-01T10:00:00Z",
                                     file_path="/a/b/detectors.py"),
            _make_tool_result("t1", "file contents", "2026-01-01T10:00:01Z"),
        ]
        path = _write_session(tmp_path, VALID_UUID, events)
        traces = ClaudeCodeExtractor().parse_session(path)
        assert traces[0].goal_text == "detectors.py"

    def test_parse_edit_extracts_file_and_delta(self, tmp_path):
        events = [
            _make_assistant_tool_use("Edit", "t1", "2026-01-01T10:00:00Z",
                                     file_path="/a/b/file.py",
                                     old_string="line1\nline2\nline3",
                                     new_string="new1\nnew2\nnew3\nnew4\nnew5"),
            _make_tool_result("t1", "ok", "2026-01-01T10:00:01Z"),
        ]
        path = _write_session(tmp_path, VALID_UUID, events)
        traces = ClaudeCodeExtractor().parse_session(path)
        assert "file.py" in traces[0].goal_text
        assert "+5/-3" in traces[0].goal_text

    def test_parse_bash_extracts_command(self, tmp_path):
        events = [
            _make_assistant_tool_use("Bash", "t1", "2026-01-01T10:00:00Z",
                                     command="pytest tests/ -x"),
            _make_tool_result("t1", "passed", "2026-01-01T10:00:02Z"),
        ]
        path = _write_session(tmp_path, VALID_UUID, events)
        traces = ClaudeCodeExtractor().parse_session(path)
        assert "pytest" in traces[0].goal_text

    def test_parse_grep_extracts_pattern(self, tmp_path):
        events = [
            _make_assistant_tool_use("Grep", "t1", "2026-01-01T10:00:00Z",
                                     pattern="login.*error", path="/src"),
            _make_tool_result("t1", "match", "2026-01-01T10:00:01Z"),
        ]
        path = _write_session(tmp_path, VALID_UUID, events)
        traces = ClaudeCodeExtractor().parse_session(path)
        assert "/login.*error/" in traces[0].goal_text
        assert "src" in traces[0].goal_text

    def test_parse_glob_extracts_pattern(self, tmp_path):
        events = [
            _make_assistant_tool_use("Glob", "t1", "2026-01-01T10:00:00Z",
                                     pattern="**/*.test.js", path="/tests"),
            _make_tool_result("t1", "files", "2026-01-01T10:00:01Z"),
        ]
        path = _write_session(tmp_path, VALID_UUID, events)
        traces = ClaudeCodeExtractor().parse_session(path)
        assert "**/*.test.js" in traces[0].goal_text
        assert "tests" in traces[0].goal_text

    def test_parse_read_generic_name_includes_parent(self, tmp_path):
        events = [
            _make_assistant_tool_use("Read", "t1", "2026-01-01T10:00:00Z",
                                     file_path="/a/b/caft/__init__.py"),
            _make_tool_result("t1", "content", "2026-01-01T10:00:01Z"),
        ]
        path = _write_session(tmp_path, VALID_UUID, events)
        traces = ClaudeCodeExtractor().parse_session(path)
        assert traces[0].goal_text == "caft/__init__.py"

    def test_parse_bash_prefers_description(self, tmp_path):
        events = [
            _make_assistant_tool_use("Bash", "t1", "2026-01-01T10:00:00Z",
                                     command="find . -name '*.py' | xargs wc -l",
                                     description="Count Python lines"),
            _make_tool_result("t1", "1234", "2026-01-01T10:00:01Z"),
        ]
        path = _write_session(tmp_path, VALID_UUID, events)
        traces = ClaudeCodeExtractor().parse_session(path)
        assert traces[0].goal_text == "Count Python lines"


class TestConvenienceFunctions:
    def test_discover_sessions(self, tmp_path):
        _write_session(tmp_path, VALID_UUID, [{"type": "x"}] * 10)
        sessions = discover_sessions(tmp_path)
        assert len(sessions) == 1

    def test_parse_session(self, tmp_path):
        events = [
            _make_assistant_text("Hello world", "2026-01-01T10:00:00Z"),
        ]
        path = _write_session(tmp_path, VALID_UUID, events)
        traces = parse_session(path)
        assert len(traces) == 1


class TestSessionInfo:
    def test_display_name(self):
        info = SessionInfo(
            session_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            path=Path("/some/path.jsonl"),
            project_dir="my-project",
        )
        assert info.display_name == "my-project/a1b2c3d4"
