"""Tests for the lead/lag analysis.

The literal-detector axis is deterministic and is pinned exactly.
CAFT's fire point depends on the real pipeline (baseline needs ~100
events to calibrate) so it is NOT asserted here — that is measured,
not unit-tested.
"""

from __future__ import annotations

import json
from pathlib import Path

from agentdiag.validation.leadlag import lead_lag, LeadLagResult


def _w(path: Path, events: list[dict]) -> None:
    with path.open("w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


def _bash(cmd: str) -> dict:
    return {"type": "assistant", "message": {"role": "assistant", "content": [
        {"type": "tool_use", "name": "Bash", "input": {"command": cmd}}]}}


def test_literal_loop_axis_detected(tmp_path):
    p = tmp_path / "s.jsonl"
    ev = [{"type": "user", "message": {"role": "user", "content": "go"}}]
    ev += [_bash("ls")] * 5          # 5 distinct-from-loop warmup
    ev += [_bash("git add x")] * 10  # the literal loop
    _w(p, ev)
    r = lead_lag(p, literal_threshold=8)
    assert isinstance(r, LeadLagResult)
    assert r.has_literal_loop is True
    assert r.t_literal is not None


def test_no_loop_not_eligible(tmp_path):
    p = tmp_path / "s.jsonl"
    ev = [{"type": "user", "message": {"role": "user", "content": "go"}}]
    for i in range(20):
        ev.append({"type": "assistant", "message": {"role": "assistant",
                   "content": [{"type": "tool_use", "name": "Edit",
                   "input": {"file_path": f"f{i}.py"}}]}})
    _w(p, ev)
    r = lead_lag(p, literal_threshold=8)
    assert r.has_literal_loop is False
    assert r.lead is None          # not eligible for the lead statistic
    assert "neither" in r.verdict() or "caft_only" in r.verdict()


def test_verdict_co_fire_logic():
    r = LeadLagResult(session_id="x", n_events=100, t_literal=50, t_caft=49,
                      caft_signature="mechanical_repetition",
                      has_literal_loop=True, caft_fired=True, lead=1)
    assert "co-fire" in r.verdict()
    r2 = LeadLagResult(session_id="x", n_events=100, t_literal=80, t_caft=40,
                       caft_signature="mechanical_repetition",
                       has_literal_loop=True, caft_fired=True, lead=40)
    assert "LED" in r2.verdict()
