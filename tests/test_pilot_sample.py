"""Tests for the pilot sampler's deterministic core (no network).

Pins exactly what amendment A1 promises: a fixed, stream-order-
independent selection reproducible from the seed alone.
"""

from __future__ import annotations

import random

from agentdiag.validation.pilot_sample import sel_key_int, _Reservoir, SEED


def test_sel_key_is_deterministic_and_seed_bound():
    a = sel_key_int("django__django-123", "swe-agent-llama-70b")
    b = sel_key_int("django__django-123", "swe-agent-llama-70b")
    assert a == b
    assert a != sel_key_int("django__django-124", "swe-agent-llama-70b")
    assert a != sel_key_int("django__django-123", "swe-agent-llama-8b")
    assert SEED == 20260515  # frozen


def test_reservoir_keeps_the_k_smallest_keys_order_independent():
    items = [(i, {"id": i}) for i in range(500)]
    forward = _Reservoir(cap=10)
    for k, r in items:
        forward.offer(k, r)
    rev = _Reservoir(cap=10)
    for k, r in reversed(items):
        rev.offer(k, r)
    shuf = _Reservoir(cap=10)
    s = items[:]
    random.Random(1).shuffle(s)
    for k, r in s:
        shuf.offer(k, r)
    want = list(range(10))  # the 10 smallest keys
    for res in (forward, rev, shuf):
        got = sorted(k for k, _ in res.rows())
        assert got == want


def test_reservoir_under_capacity_keeps_all():
    res = _Reservoir(cap=100)
    for k in (5, 1, 9, 3):
        res.offer(k, {"k": k})
    assert sorted(k for k, _ in res.rows()) == [1, 3, 5, 9]
