"""Tests for Leg-2 workload extraction (deterministic core)."""

from __future__ import annotations

from agentdiag.validation.pilot_workload import (
    extract_workload_one, _reasoning_text, WORKLOAD_FEATURES,
    GATE_FEATURES, GATE_THRESHOLD, SEED,
)


def test_reasoning_text_strips_action_blocks():
    t = "I will search the repo.\n```\nsearch_dir \"x\"\n```"
    assert _reasoning_text(t) == "I will search the repo."
    assert _reasoning_text("```\nopen a\n```") == ""
    assert _reasoning_text("pure thought") == "pure thought"


def test_feature_set_frozen_nine():
    assert len(WORKLOAD_FEATURES) == 9
    assert GATE_FEATURES == ("reasoning_len.mean", "context_cum.final")
    assert GATE_THRESHOLD == 0.80
    assert SEED == 20260515


_ROW = {
    "target": True,
    "trajectory": [
        {"role": "system", "text": "sys"},
        {"role": "user", "text": "ISSUE: bug"},
        {"role": "ai", "text": "Let me look.\n```\nopen a.py\n```"},
        {"role": "user", "text": "Error: no such file a.py"},
        {"role": "ai", "text": "It failed. Try another path here.\n"
                               "```\nopen src/a.py\n```"},
        {"role": "user", "text": "[file shown ok]"},
        {"role": "ai", "text": "Now fix.\n```\nedit 1:1\nx\nend_of_edit\n```"},
        {"role": "user", "text": "File updated."},
    ],
}


def test_extract_shape_and_keys():
    r = extract_workload_one(_ROW)
    assert set(r["workload"].keys()) == set(WORKLOAD_FEATURES)
    assert r["target"] is True


def test_error_recovery_episode_and_latency():
    r = extract_workload_one(_ROW)["workload"]
    # one error observation (turn idx 3) closed by the next ai turn (idx 4)
    assert r["error_recovery.n_episodes"] == 1.0
    assert r["error_recovery.mean_latency_turns"] == 1.0


def test_reasoning_len_counts_thought_only_not_action():
    r = extract_workload_one(_ROW)["workload"]
    # 3 ai turns; reasoning text excludes the fenced commands
    assert r["reasoning_len.total"] == float(
        len("Let me look.")
        + len("It failed. Try another path here.")
        + len("Now fix."))
    assert r["reasoning_len.max"] >= r["reasoning_len.mean"]


def test_context_cum_is_monotone_final_is_total_text():
    r = extract_workload_one(_ROW)["workload"]
    total = sum(len(t["text"]) for t in _ROW["trajectory"])
    assert r["context_cum.final"] == float(total)


def test_no_error_means_zero_episodes():
    row = {"target": False, "trajectory": [
        {"role": "user", "text": "ISSUE"},
        {"role": "ai", "text": "ok\n```\nsubmit\n```"},
        {"role": "user", "text": "done cleanly"}]}
    w = extract_workload_one(row)["workload"]
    assert w["error_recovery.n_episodes"] == 0.0
    assert w["error_recovery.mean_latency_turns"] == 0.0
