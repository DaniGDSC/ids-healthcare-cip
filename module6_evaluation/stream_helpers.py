"""Stream simulator + latency profile sampling helpers."""
from __future__ import annotations

import random
import time
from typing import Iterator


def stream_simulator(responses: list, delay: float = 1.0) -> Iterator[dict]:
    """Yield responses with a configurable inter-emission delay.

    Used by the Online Simulation page to replay alerts in near-real time.
    Caller is expected to ``yield`` between renders.
    """
    for r in responses:
        yield r
        if delay > 0:
            time.sleep(delay)


def draw_latency_sample(stage_stats: dict, rng: random.Random) -> float:
    """Sample a per-stage latency given mean/stddev/lower-clip stats.

    ``stage_stats`` shape: ``{"mean_ms": float, "std_ms": float, "min_ms": float}``.
    """
    mean = float(stage_stats.get("mean_ms", 0.0))
    std = float(stage_stats.get("std_ms", 0.0))
    floor = float(stage_stats.get("min_ms", 0.0))
    sample = rng.gauss(mean, std) if std > 0 else mean
    return max(floor, sample)


def push_latency_sample(profile: dict, rng: random.Random | None = None) -> dict | None:
    """Draw one end-to-end latency sample across every stage in ``profile``.

    Returns a dict mapping stage name → milliseconds, plus a ``total_ms``
    key summing all stages. Returns ``None`` when ``profile`` is empty or
    malformed.
    """
    if not profile or not isinstance(profile, dict):
        return None
    if rng is None:
        rng = random.Random()
    out: dict = {}
    total = 0.0
    for stage, stats in profile.items():
        if not isinstance(stats, dict):
            continue
        sample = draw_latency_sample(stats, rng)
        out[stage] = sample
        total += sample
    if not out:
        return None
    out["total_ms"] = total
    return out


__all__ = [
    "stream_simulator",
    "draw_latency_sample",
    "push_latency_sample",
]
