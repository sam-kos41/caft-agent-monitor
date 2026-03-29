"""Tests for the evaluation pipeline."""

import json
from pathlib import Path

import pytest

from agentdiag.evaluate import (
    evaluate_claude_code,
    EvaluationReport,
    SessionResult,
    print_evaluation_report,
)


VALID_UUID = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
VALID_UUID_2 = "11111111-2222-3333-4444-555555555555"


def _make_session(tmp_path, session_id, events):
    path = tmp_path / f"{session_id}.jsonl"
    with open(path, "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    return path


def _make_assistant_tool_use(tool_name, tool_id, timestamp, **kw):
    return {
        "type": "assistant",
        "timestamp": timestamp,
        "message": {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": tool_id, "name": tool_name, "input": kw}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        },
    }


def _make_tool_result(tool_id, content, timestamp, is_error=False):
    return {
        "type": "user",
        "timestamp": timestamp,
        "message": {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": content, "is_error": is_error}],
        },
    }


def _make_thinking(text, timestamp):
    return {
        "type": "assistant",
        "timestamp": timestamp,
        "message": {
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": text}],
            "usage": {"input_tokens": 5, "output_tokens": 3},
        },
    }


def _make_text(text, timestamp):
    return {
        "type": "assistant",
        "timestamp": timestamp,
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "usage": {"input_tokens": 3, "output_tokens": 2},
        },
    }


def _build_healthy_session():
    """Build events that produce a clean session with good HTA progression."""
    events = [
        # Gathering
        _make_thinking("Let me read the files.", "2026-01-01T10:00:00Z"),
        _make_assistant_tool_use("Read", "t1", "2026-01-01T10:00:01Z", file_path="/a.py"),
        _make_tool_result("t1", "def foo(): pass", "2026-01-01T10:00:02Z"),
        _make_assistant_tool_use("Grep", "t2", "2026-01-01T10:00:03Z", pattern="foo"),
        _make_tool_result("t2", "a.py:1:def foo", "2026-01-01T10:00:04Z"),
        # Executing
        _make_assistant_tool_use("Edit", "t3", "2026-01-01T10:00:05Z", file_path="/a.py"),
        _make_tool_result("t3", "edited", "2026-01-01T10:00:06Z"),
        _make_assistant_tool_use("Write", "t4", "2026-01-01T10:00:07Z", file_path="/b.py"),
        _make_tool_result("t4", "written", "2026-01-01T10:00:08Z"),
        # Done
        _make_text("I have fixed the issue.", "2026-01-01T10:00:09Z"),
    ]
    return events


class TestEvaluateClaude:
    def test_evaluates_single_session(self, tmp_path):
        _make_session(tmp_path, VALID_UUID, _build_healthy_session())
        report = evaluate_claude_code(tmp_path)
        assert report.total_sessions == 1
        assert report.total_events > 0
        assert len(report.session_results) == 1

    def test_evaluates_multiple_sessions(self, tmp_path):
        _make_session(tmp_path, VALID_UUID, _build_healthy_session())
        _make_session(tmp_path, VALID_UUID_2, _build_healthy_session())
        report = evaluate_claude_code(tmp_path)
        assert report.total_sessions == 2

    def test_session_result_has_fields(self, tmp_path):
        _make_session(tmp_path, VALID_UUID, _build_healthy_session())
        report = evaluate_claude_code(tmp_path)
        r = report.session_results[0]
        assert r.session_id == VALID_UUID
        assert r.event_count > 0
        assert r.final_phase in ("idle", "gathering", "planning", "executing", "verifying", "delivering")
        assert 0.0 <= r.trust_score <= 1.0
        assert r.health in ("healthy", "degraded", "failing")

    def test_session_filter_by_id(self, tmp_path):
        _make_session(tmp_path, VALID_UUID, _build_healthy_session())
        _make_session(tmp_path, VALID_UUID_2, _build_healthy_session())
        report = evaluate_claude_code(tmp_path, session_id=VALID_UUID[:8])
        assert report.total_sessions == 1
        assert report.session_results[0].session_id == VALID_UUID

    def test_session_filter_no_match(self, tmp_path):
        _make_session(tmp_path, VALID_UUID, _build_healthy_session())
        with pytest.raises(ValueError, match="No session found"):
            evaluate_claude_code(tmp_path, session_id="zzzzzzz")

    def test_health_distribution(self, tmp_path):
        _make_session(tmp_path, VALID_UUID, _build_healthy_session())
        report = evaluate_claude_code(tmp_path)
        total = report.sessions_healthy + report.sessions_degraded + report.sessions_failing
        assert total == report.total_sessions

    def test_tool_counts(self, tmp_path):
        _make_session(tmp_path, VALID_UUID, _build_healthy_session())
        report = evaluate_claude_code(tmp_path)
        r = report.session_results[0]
        assert "Read" in r.tool_counts
        assert "Edit" in r.tool_counts

    def test_event_type_counts(self, tmp_path):
        _make_session(tmp_path, VALID_UUID, _build_healthy_session())
        report = evaluate_claude_code(tmp_path)
        r = report.session_results[0]
        assert "tool_call" in r.event_type_counts
        assert "planning" in r.event_type_counts

    def test_report_to_json(self, tmp_path):
        _make_session(tmp_path, VALID_UUID, _build_healthy_session())
        report = evaluate_claude_code(tmp_path)
        j = report.to_json()
        parsed = json.loads(j)
        assert parsed["dataset"] == "claude-code"
        assert parsed["total_sessions"] == 1
        assert "session_results" in parsed

    def test_report_to_dict(self, tmp_path):
        _make_session(tmp_path, VALID_UUID, _build_healthy_session())
        report = evaluate_claude_code(tmp_path)
        d = report.to_dict()
        assert isinstance(d, dict)
        assert d["total_sessions"] == 1

    def test_empty_directory(self, tmp_path):
        report = evaluate_claude_code(tmp_path)
        assert report.total_sessions == 0
        assert report.total_events == 0

    def test_print_report(self, tmp_path, capsys):
        _make_session(tmp_path, VALID_UUID, _build_healthy_session())
        report = evaluate_claude_code(tmp_path)
        print_evaluation_report(report)
        output = capsys.readouterr().out
        assert "EVALUATION REPORT" in output
        assert "HEALTH DISTRIBUTION" in output
        assert VALID_UUID[:10] in output
