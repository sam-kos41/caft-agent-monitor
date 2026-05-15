"""Tests for the deterministic programmatic signal extractor.

Each test builds a synthetic session and asserts the EXACT extracted
facts — the whole point of this module is determinism, so the tests
pin exact integers, not ranges.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentdiag.validation.signals import (
    extract_signals, signals_to_ratings, rate_with_signals, SessionSignals,
)


def _w(path: Path, events: list[dict]) -> None:
    with path.open("w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


def _u(text: str) -> dict:
    return {"type": "user", "message": {"role": "user", "content": text}}


def _bash(cmd: str) -> dict:
    return {"type": "assistant", "message": {"role": "assistant", "content": [
        {"type": "tool_use", "name": "Bash", "input": {"command": cmd}}]}}


def _toolresult(err: bool = False) -> dict:
    return {"type": "user", "message": {"role": "user", "content": [
        {"type": "tool_result", "content": "out", "is_error": err}]}}


def test_literal_loop_detection(tmp_path):
    p = tmp_path / "s.jsonl"
    ev = [_u("do it")]
    for _ in range(9):
        ev.append(_bash("git add page.tsx"))
    ev.append(_bash("ls"))
    _w(p, ev)
    s = extract_signals(p)
    assert s.literal_loop_max == 9
    assert "git add page.tsx" in s.literal_loop_tool
    assert s.n_tool_calls == 10


def test_different_files_not_a_loop(tmp_path):
    p = tmp_path / "s.jsonl"
    ev = [_u("edit files")]
    for i in range(8):
        ev.append({"type": "assistant", "message": {"role": "assistant",
                   "content": [{"type": "tool_use", "name": "Edit",
                   "input": {"file_path": f"file_{i}.py"}}]}})
    _w(p, ev)
    s = extract_signals(p)
    assert s.literal_loop_max == 1  # every call distinct -> not a loop


def test_error_retry_cycle(tmp_path):
    p = tmp_path / "s.jsonl"
    _w(p, [
        _u("run it"),
        _bash("python x.py"), _toolresult(err=True),
        _bash("python x.py"), _toolresult(err=True),
        _bash("python x.py"), _toolresult(err=False),
    ])
    s = extract_signals(p)
    assert s.error_retry_cycles == 2
    assert s.n_errors == 2


def test_user_reprompt_detected(tmp_path):
    p = tmp_path / "s.jsonl"
    _w(p, [
        _u("please add a dark mode toggle to the settings page"),
        _bash("ls"),
        _u("please add the dark mode toggle to the settings page again"),
    ])
    s = extract_signals(p)
    assert s.user_reprompts == 1


def test_frustration_and_correction(tmp_path):
    p = tmp_path / "s.jsonl"
    _w(p, [
        _u("build the parser"),
        _u("no, that's not what i asked for"),
        _u("this is fucking broken, undo it"),
    ])
    s = extract_signals(p)
    assert s.frustration_hits == 1
    assert s.correction_hits >= 1
    assert s.frustration_quotes


def test_git_undo_and_tasks(tmp_path):
    p = tmp_path / "s.jsonl"
    _w(p, [
        _u("go"),
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "tool_use", "name": "TaskCreate", "input": {}}]}},
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "tool_use", "name": "TaskUpdate",
             "input": {"status": "completed"}}]}},
        _bash("git reset --hard HEAD~1"),
    ])
    s = extract_signals(p)
    assert s.tasks_created == 1
    assert s.tasks_completed == 1
    assert s.git_undo == 1


def test_resolution_positive(tmp_path):
    p = tmp_path / "s.jsonl"
    _w(p, [_u("fix the bug"), _bash("pytest"), _u("perfect, that works thanks!")])
    assert extract_signals(p).resolution == "positive"


def test_resolution_negative(tmp_path):
    p = tmp_path / "s.jsonl"
    _w(p, [_u("fix it"), _bash("pytest"), _u("no that's still wrong")])
    assert extract_signals(p).resolution == "negative"


def test_resolution_none_when_log_ending(tmp_path):
    p = tmp_path / "s.jsonl"
    _w(p, [_u("run pipeline"),
           _u("INFO - PHASE A: extraction\nINFO - loading checkpoint")])
    assert extract_signals(p).resolution == "none"


# -------- signals_to_ratings mapping --------


def test_mapping_stuck_scale():
    s = SessionSignals(session_id="x", literal_loop_max=20)
    by = {r.dimension: r for r in signals_to_ratings(s)}
    assert by["stuck_in_loop"].value == 5
    assert by["stuck_in_loop"].confidence == "high"
    s2 = SessionSignals(session_id="x", literal_loop_max=1)
    by2 = {r.dimension: r for r in signals_to_ratings(s2)}
    assert by2["stuck_in_loop"].value == 1


def test_mapping_goal_drifted_always_abstains():
    s = SessionSignals(session_id="x", literal_loop_max=3)
    by = {r.dimension: r for r in signals_to_ratings(s)}
    assert by["goal_drifted"].value is None
    assert by["goal_drifted"].confidence == ""


def test_mapping_user_satisfied_from_resolution():
    pos = {r.dimension: r for r in signals_to_ratings(
        SessionSignals(session_id="x", resolution="positive"))}
    assert pos["user_satisfied"].value == 5
    neg = {r.dimension: r for r in signals_to_ratings(
        SessionSignals(session_id="x", resolution="negative"))}
    assert neg["user_satisfied"].value == 1
    non = {r.dimension: r for r in signals_to_ratings(
        SessionSignals(session_id="x", resolution="none"))}
    assert non["user_satisfied"].value is None  # abstain, not a guessed 3


def test_v2_abstains_on_non_observable_dims():
    """v2 narrowing: signal asserts only stuck + user_satisfied;
    overall_health / coherent_progress / goal_drifted all abstain."""
    by = {r.dimension: r for r in signals_to_ratings(
        SessionSignals(session_id="x", error_retry_cycles=4,
                       correction_hits=2, literal_loop_max=20))}
    assert by["overall_health"].value is None
    assert "retired" in by["overall_health"].reasoning
    assert by["coherent_progress"].value is None
    assert by["goal_drifted"].value is None
    # the two it can prove are still emitted
    assert by["stuck_in_loop"].value == 5
    assert by["stuck_in_loop"].confidence == "high"


def test_midsession_frustration_does_not_flip_satisfaction():
    """A session with mid-session frustration but a clean positive
    close must read user_satisfied=5 (recovery), not 1."""
    s = SessionSignals(session_id="x", frustration_hits=3,
                       resolution="positive")
    by = {r.dimension: r for r in signals_to_ratings(s)}
    assert by["user_satisfied"].value == 5


def test_rate_with_signals_round_trip(tmp_path):
    p = tmp_path / "s.jsonl"
    ev = [_u("go")] + [_bash("git add a")] * 16 + [_u("perfect thanks")]
    _w(p, ev)
    ratings = rate_with_signals(p)
    by = {r.dimension: r for r in ratings}
    assert by["stuck_in_loop"].value == 5      # 16x loop
    assert by["user_satisfied"].value == 5     # positive close
    assert all(r.rater_type == "signal" for r in ratings)
