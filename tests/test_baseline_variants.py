"""Tests for the steelman baseline variants (pure functions only).

corpus_fire / changepoint_fire are deterministic over synthetic
per-step metric lists and are pinned exactly. The full replay path is
exercised by running it on the real corpus, not unit-tested.
"""

from __future__ import annotations

from agentdiag.validation.baseline_variants import (
    corpus_fire, changepoint_fire,
)


def test_corpus_fire_triggers_on_deviation():
    mean = {"action_mi": 1.0, "kl_divergence": 0.2}
    std = {"action_mi": 0.1, "kl_divergence": 0.05}
    per_step = [
        {"action_mi": 1.0, "kl_divergence": 0.2},   # normal
        {"action_mi": 1.01, "kl_divergence": 0.21},  # normal
        {"action_mi": 0.2, "kl_divergence": 0.9},    # both >2 SD -> fire
    ]
    assert corpus_fire(per_step, mean, std, z=2.0, min_metrics=2) == 2


def test_corpus_fire_needs_two_metrics():
    mean = {"action_mi": 1.0, "kl_divergence": 0.2}
    std = {"action_mi": 0.1, "kl_divergence": 0.05}
    per_step = [{"action_mi": 0.2, "kl_divergence": 0.2}]  # only 1 hot
    assert corpus_fire(per_step, mean, std, z=2.0, min_metrics=2) is None


def test_changepoint_detects_level_shift():
    # flat then a sustained jump in action_mi
    per_step = ([{"action_mi": 1.0, "compression_ratio": 1.0}] * 30
                + [{"action_mi": 0.1, "compression_ratio": 1.0}] * 30)
    cp = changepoint_fire(per_step, w=20, k=2.5)
    assert cp is not None
    assert 25 <= cp <= 35  # shift is at index 30


def test_changepoint_none_when_flat():
    per_step = [{"action_mi": 1.0, "compression_ratio": 1.0}] * 60
    assert changepoint_fire(per_step, w=20, k=2.5) is None
