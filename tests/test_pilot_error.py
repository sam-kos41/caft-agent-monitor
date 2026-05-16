"""Tests for Leg-4 error-recovery extraction (deterministic core)."""

from __future__ import annotations

from agentdiag.validation.pilot_error import (
    extract_error_one, ERROR_FEATURES, GATE_FEATURES, GATE_THRESHOLD,
    SEED, K_RECOVER, _ERR,
)


def _ai(cmd):
    return {"role": "ai", "text": f"thinking\n```\n{cmd}\n```"}


def _obs(text):
    return {"role": "user", "text": text}


def test_frozen_constants():
    assert len(ERROR_FEATURES) == 6
    assert GATE_FEATURES == ("error.strategy_change_rate",
                             "error.recovery_success_rate")
    assert GATE_THRESHOLD == 0.80 and SEED == 20260515 and K_RECOVER == 3


def test_error_regex_frozen_signatures():
    for s in ["Traceback (most recent call last)", "No such file",
              "command not found", "Exception:", "FAILED", "cannot open",
              "invalid syntax", "Permission denied"]:
        assert _ERR.search(s)
    assert not _ERR.search("file opened successfully")


def test_no_errors_zero_features():
    row = {"target": True, "trajectory": [
        _obs("ISSUE: bug"),
        _ai('open a.py'), _obs("[file shown]"),
        _ai('submit'), _obs("done")]}
    e = extract_error_one(row)["error"]
    assert e["error.n_episodes"] == 0.0
    assert e["error.strategy_change_rate"] == 0.0
    assert e["error.recovery_success_rate"] == 0.0
    assert e["error.terminal_unresolved"] == 0.0


def test_strategy_change_vs_identical_retry():
    # episode 1: fail open a.py -> respond DIFFERENT (open b.py) = change
    chg = {"target": True, "trajectory": [
        _obs("ISSUE"),
        _ai('open a.py'), _obs("Error: no such file a.py"),
        _ai('open b.py'), _obs("[ok]"), _ai('submit'), _obs("done")]}
    e1 = extract_error_one(chg)["error"]
    assert e1["error.n_episodes"] == 1.0
    assert e1["error.strategy_change_rate"] == 1.0

    # identical retry: fail open a.py -> respond SAME (open a.py) = no change
    same = {"target": False, "trajectory": [
        _obs("ISSUE"),
        _ai('open a.py'), _obs("Error: no such file a.py"),
        _ai('open a.py'), _obs("Error: no such file a.py")]}
    e2 = extract_error_one(same)["error"]
    assert e2["error.strategy_change_rate"] == 0.0


def test_recurrence_rate_repeated_failing_signature():
    row = {"target": False, "trajectory": [
        _obs("ISSUE"),
        _ai('open a.py'), _obs("Error: no such file"),
        _ai('open a.py'), _obs("Error: no such file"),  # same fail recurs
        _ai('open a.py'), _obs("Error: no such file")]}
    e = extract_error_one(row)["error"]
    assert e["error.n_episodes"] == 3.0
    # 2 of 3 episodes have a previously-seen failing signature
    assert abs(e["error.recurrence_rate"] - 2 / 3) < 1e-9


def test_recovery_success_within_K():
    # NB: success obs text must not contain the frozen-regex substring
    # 'error' (the pre-registered signature is deliberately simple;
    # its substring imprecision is a documented honest-scoping tradeoff
    # and is NOT altered post-registration).
    row = {"target": True, "trajectory": [
        _obs("ISSUE"),
        _ai('python x.py'), _obs("Traceback: boom"),
        _ai('edit x.py'), _obs("File updated successfully."),
        _ai('submit'), _obs("done")]}
    e = extract_error_one(row)["error"]
    assert e["error.n_episodes"] == 1.0
    assert e["error.recovery_success_rate"] == 1.0
    assert e["error.mean_latency_turns"] > 0.0


def test_terminal_unresolved_when_ends_on_error():
    row = {"target": False, "trajectory": [
        _obs("ISSUE"),
        _ai('python x.py'), _obs("Traceback: boom"),
        _ai('python x.py'), _obs("Traceback: boom again")]}  # ends in error
    e = extract_error_one(row)["error"]
    assert e["error.terminal_unresolved"] == 1.0

    ok = {"target": True, "trajectory": [
        _obs("ISSUE"),
        _ai('python x.py'), _obs("Traceback"),
        _ai('submit'), _obs("resolved")]}  # submit after, clean end
    assert extract_error_one(ok)["error"]["error.terminal_unresolved"] == 0.0


def test_extract_shape():
    row = {"target": True, "trajectory": [_obs("ISSUE"), _ai('open a'),
                                          _obs("ok")]}
    r = extract_error_one(row)
    assert set(r["error"].keys()) == set(ERROR_FEATURES)
    assert r["target"] is True
    assert isinstance(r["tool_counts"], dict)
