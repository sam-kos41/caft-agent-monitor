"""Tests for the pilot evaluation pipeline."""

import json

import pytest

from agentdiag.pilot import (
    run_pilot,
    PilotResult,
    PilotReport,
    print_pilot_report,
    _classify_result,
)


VALID_UUID = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
VALID_UUID_2 = "11111111-2222-3333-4444-555555555555"


def _make_session(tmp_path, session_id, events):
    path = tmp_path / f"{session_id}.jsonl"
    with open(path, "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    return path


def _make_tool_use(tool, tool_id, ts, **kw):
    return {
        "type": "assistant", "timestamp": ts,
        "message": {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": tool_id, "name": tool, "input": kw}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        },
    }


def _make_result(tool_id, content, ts, is_error=False):
    return {
        "type": "user", "timestamp": ts,
        "message": {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_id,
                          "content": content, "is_error": is_error}],
        },
    }


def _make_text(text, ts):
    return {
        "type": "assistant", "timestamp": ts,
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "usage": {"input_tokens": 3, "output_tokens": 2},
        },
    }


def _build_session_events():
    return [
        _make_tool_use("Read", "t1", "2026-01-01T10:00:00Z", file_path="/a"),
        _make_result("t1", "data", "2026-01-01T10:00:02Z"),
        _make_tool_use("Grep", "t2", "2026-01-01T10:00:03Z", pattern="x"),
        _make_result("t2", "found", "2026-01-01T10:00:04Z"),
        _make_tool_use("Edit", "t3", "2026-01-01T10:00:05Z", file_path="/a"),
        _make_result("t3", "ok", "2026-01-01T10:00:06Z"),
        _make_tool_use("Write", "t4", "2026-01-01T10:00:07Z", file_path="/b"),
        _make_result("t4", "ok", "2026-01-01T10:00:08Z"),
        _make_text("Done.", "2026-01-01T10:00:09Z"),
    ] + [{"type": "x"}] * 5  # pad for min_lines


class TestRunPilot:
    def test_runs_on_single_session(self, tmp_path):
        _make_session(tmp_path, VALID_UUID, _build_session_events())
        report = run_pilot(tmp_path, n=5, min_lines=5)
        assert report.n_traces == 1
        assert report.n_parsed == 1

    def test_runs_on_multiple_sessions(self, tmp_path):
        _make_session(tmp_path, VALID_UUID, _build_session_events())
        _make_session(tmp_path, VALID_UUID_2, _build_session_events())
        report = run_pilot(tmp_path, n=5, min_lines=5)
        assert report.n_traces == 2

    def test_limits_to_n(self, tmp_path):
        for i in range(5):
            uuid = f"a{i}b2c3d4-e5f6-7890-abcd-ef1234567890"
            _make_session(tmp_path, uuid, _build_session_events())
        report = run_pilot(tmp_path, n=3, min_lines=5)
        assert report.n_traces == 3

    def test_empty_directory(self, tmp_path):
        report = run_pilot(tmp_path, n=5)
        assert report.n_traces == 0
        assert report.n_parsed == 0

    def test_result_has_timing(self, tmp_path):
        _make_session(tmp_path, VALID_UUID, _build_session_events())
        report = run_pilot(tmp_path, n=5, min_lines=5)
        assert report.total_ms > 0
        assert report.avg_ms_per_trace > 0

    def test_report_to_json(self, tmp_path):
        _make_session(tmp_path, VALID_UUID, _build_session_events())
        report = run_pilot(tmp_path, n=5, min_lines=5)
        j = report.to_json()
        parsed = json.loads(j)
        assert "n_traces" in parsed
        assert "results" in parsed

    def test_print_report(self, tmp_path, capsys):
        _make_session(tmp_path, VALID_UUID, _build_session_events())
        report = run_pilot(tmp_path, n=5, min_lines=5)
        print_pilot_report(report)
        out = capsys.readouterr().out
        assert "PILOT EVALUATION" in out


class TestClassifyResult:
    def test_classify_parser_issue(self):
        r = PilotResult(trace_num=1, session_id="x", project="p",
                        raw_lines=10, parse_ok=False)
        assert _classify_result(r) == "parser_issue"

    def test_classify_clean(self):
        r = PilotResult(trace_num=1, session_id="x", project="p",
                        raw_lines=10, parse_ok=True, hta_plausible=True)
        assert _classify_result(r) == "clean"

    def test_classify_real_multi_detector(self):
        r = PilotResult(trace_num=1, session_id="x", project="p",
                        raw_lines=10, parse_ok=True, hta_plausible=True,
                        detectors_fired=["step_repetition", "goal_drift"],
                        event_count=50, severities=["warning", "critical"])
        assert _classify_result(r) == "real"

    def test_classify_ambiguous_single_step_rep(self):
        r = PilotResult(trace_num=1, session_id="x", project="p",
                        raw_lines=10, parse_ok=True, hta_plausible=True,
                        detectors_fired=["step_repetition"],
                        event_count=50, severities=["warning"])
        assert _classify_result(r) == "ambiguous"
