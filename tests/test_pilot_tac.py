"""Tests for agent-native TAC extraction (deterministic primary)."""

from __future__ import annotations

from agentdiag.validation.pilot_tac import (
    extract_tac_one, TAC_FEATURES, GATE_FEATURES, GATE_THRESHOLD, SEED,
)


def _ai(thought, cmd):
    return {"role": "ai", "text": f"{thought}\n```\n{cmd}\n```"}


def test_frozen_constants():
    assert len(TAC_FEATURES) == 6
    assert GATE_FEATURES == ("tac.mean", "tac.target_match_rate")
    assert GATE_THRESHOLD == 0.80 and SEED == 20260515


def test_perfectly_coherent_turn_scores_one():
    # thought says "open the dispatcher file"; action opens dispatcher.py
    row = {"target": True, "trajectory": [
        _ai("Let me open the dispatcher file to inspect it.",
            "open src/dispatcher.py")]}
    t = extract_tac_one(row)["tac"]
    assert t["tac.mean"] == 1.0          # verb_align=1, target_present=1
    assert t["tac.verb_align_rate"] == 1.0
    assert t["tac.target_match_rate"] == 1.0


def test_incoherent_turn_scores_zero():
    # thought talks about searching; action edits an unmentioned file
    row = {"target": False, "trajectory": [
        _ai("I should search the repository for the constant.",
            "edit unrelated/file.py")]}
    t = extract_tac_one(row)["tac"]
    # verb_align: action cat=mutate, thought has no mutate-lexicon word -> 0
    # target_present: 'file' basename token not in thought -> 0
    assert t["tac.mean"] == 0.0


def test_verb_align_only_partial_score():
    # thought has the mutate verb but not the target token
    row = {"target": True, "trajectory": [
        _ai("Now I will fix the bug.", "edit deep/path/module.py")]}
    t = extract_tac_one(row)["tac"]
    assert t["tac.mean"] == 0.5          # verb_align=1, target_present=0
    assert t["tac.verb_align_rate"] == 1.0
    assert t["tac.target_match_rate"] == 0.0


def test_targetless_action_not_penalized():
    # submit has no target -> target_present := verb_align
    coh = {"target": True, "trajectory": [
        _ai("I'm done, let me submit.", "submit")]}
    assert extract_tac_one(coh)["tac"]["tac.mean"] == 1.0  # verb_align=1
    incoh = {"target": False, "trajectory": [
        _ai("The weather is nice today.", "submit")]}
    # no submit-lexicon word in thought -> verb_align 0 -> tac 0
    assert extract_tac_one(incoh)["tac"]["tac.mean"] == 0.0


def test_aggregates_over_turns():
    row = {"target": True, "trajectory": [
        _ai("open the file dispatcher.py", "open dispatcher.py"),   # 1.0
        _ai("the weather is nice", "edit other.py"),                # 0.0
    ]}
    t = extract_tac_one(row)["tac"]
    assert t["tac.mean"] == 0.5
    assert t["tac.min"] == 0.0
    assert t["tac.final"] == 0.0


def test_extract_shape():
    row = {"target": True, "trajectory": [_ai("open a", "open a.py")]}
    r = extract_tac_one(row)
    assert set(r["tac"].keys()) == set(TAC_FEATURES)
    assert r["target"] is True
    assert isinstance(r["tool_counts"], dict)
