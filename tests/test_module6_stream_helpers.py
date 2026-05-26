"""Module 6 stream helpers — stream_simulator + latency sampling."""
from __future__ import annotations

import random
from unittest.mock import patch

from module6_evaluation.stream_helpers import (
    draw_latency_sample,
    push_latency_sample,
    stream_simulator,
)


# ── stream_simulator ───────────────────────────────────────────────────


def test_stream_simulator_yields_in_order():
    responses = [{"i": 0}, {"i": 1}, {"i": 2}]
    out = list(stream_simulator(responses, delay=0.0))
    assert out == responses


def test_stream_simulator_respects_delay():
    responses = [{"i": 0}, {"i": 1}, {"i": 2}]
    with patch("module6_evaluation.stream_helpers.time.sleep") as ms:
        list(stream_simulator(responses, delay=0.5))
    # 3 yields, 3 sleeps
    assert ms.call_count == 3
    ms.assert_called_with(0.5)


def test_stream_simulator_zero_delay_no_sleep():
    responses = [{"i": 0}]
    with patch("module6_evaluation.stream_helpers.time.sleep") as ms:
        list(stream_simulator(responses, delay=0.0))
    assert ms.call_count == 0


def test_stream_simulator_empty():
    assert list(stream_simulator([], delay=0.0)) == []


# ── draw_latency_sample ────────────────────────────────────────────────


def test_draw_latency_sample_above_floor():
    rng = random.Random(0)
    stats = {"mean_ms": 100.0, "std_ms": 0.0, "min_ms": 50.0}
    s = draw_latency_sample(stats, rng)
    assert s == 100.0


def test_draw_latency_sample_clipped_at_floor():
    rng = random.Random(0)
    # Negative mean with zero std → returns mean (100) bound by floor 50.
    stats = {"mean_ms": -10.0, "std_ms": 0.0, "min_ms": 5.0}
    s = draw_latency_sample(stats, rng)
    assert s == 5.0


def test_draw_latency_sample_zero_std_returns_mean():
    rng = random.Random(0)
    stats = {"mean_ms": 42.0, "std_ms": 0.0, "min_ms": 0.0}
    assert draw_latency_sample(stats, rng) == 42.0


def test_draw_latency_sample_seeded_reproducible():
    rng1 = random.Random(99)
    rng2 = random.Random(99)
    stats = {"mean_ms": 100.0, "std_ms": 20.0, "min_ms": 0.0}
    assert draw_latency_sample(stats, rng1) == draw_latency_sample(stats, rng2)


# ── push_latency_sample ────────────────────────────────────────────────


def test_push_latency_sample_returns_total():
    profile = {
        "detect": {"mean_ms": 50, "std_ms": 0, "min_ms": 0},
        "risk":   {"mean_ms": 30, "std_ms": 0, "min_ms": 0},
        "respond": {"mean_ms": 20, "std_ms": 0, "min_ms": 0},
    }
    out = push_latency_sample(profile, rng=random.Random(0))
    assert out["total_ms"] == 100
    assert out["detect"] == 50
    assert out["risk"] == 30
    assert out["respond"] == 20


def test_push_latency_sample_empty_returns_none():
    assert push_latency_sample({}, rng=random.Random(0)) is None


def test_push_latency_sample_malformed_returns_none():
    assert push_latency_sample(None, rng=random.Random(0)) is None
    assert push_latency_sample({"detect": "not a dict"},
                                rng=random.Random(0)) is None


def test_push_latency_sample_default_rng():
    # When rng is None, a fresh one is built — must still return a dict.
    profile = {"x": {"mean_ms": 10, "std_ms": 0, "min_ms": 0}}
    out = push_latency_sample(profile)
    assert out is not None
    assert out["total_ms"] == 10
