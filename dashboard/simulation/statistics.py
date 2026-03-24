"""Distribution fitting utilities for streaming statistics.

Computes rolling statistics, threat posture scores, and
latency percentiles for dashboard analytics panels.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Deque, Dict, List, Tuple


def weighted_risk_score(risk_counts: Dict[str, int]) -> float:
    """Compute weighted threat posture score (0-100).

    Weights: NORMAL=0, LOW=1, MEDIUM=2, HIGH=4, CRITICAL=8

    Args:
        risk_counts: Dictionary mapping risk levels to counts.

    Returns:
        Score between 0 (all NORMAL) and 100 (all CRITICAL).
    """
    weights = {
        "NORMAL": 0, "LOW": 1, "MEDIUM": 2,
        "HIGH": 4, "CRITICAL": 8,
    }
    total = sum(risk_counts.values())
    if total == 0:
        return 0.0

    score = sum(
        weights.get(level, 0) * count
        for level, count in risk_counts.items()
    )
    max_possible = 8 * total  # if all CRITICAL
    return round(score / max_possible * 100, 1)


def threat_level(score: float) -> Tuple[str, str]:
    """Classify threat posture score into named level.

    Args:
        score: Weighted risk score (0-100).

    Returns:
        Tuple of (level_name, color_hex).
    """
    if score < 20:
        return "SECURE", "#2ecc71"
    elif score < 40:
        return "ELEVATED", "#f39c12"
    elif score < 70:
        return "HIGH", "#e67e22"
    else:
        return "CRITICAL", "#e74c3c"


def k_schedule_display(hour: int) -> Tuple[float, str]:
    """Return the MAD multiplier and mode label for given hour.

    Args:
        hour: Hour of day (0-23).

    Returns:
        Tuple of (k_value, mode_label).
    """
    if 0 <= hour < 6:
        return 2.5, "Night mode"
    elif 6 <= hour < 22:
        return 3.0, "Day mode"
    else:
        return 3.5, "Evening mode"


def threshold_drift_status(
    dynamic_threshold: float,
    baseline_threshold: float,
) -> Tuple[str, str]:
    """Compute threshold drift status.

    Args:
        dynamic_threshold: Current adaptive threshold.
        baseline_threshold: Static baseline threshold.

    Returns:
        Tuple of (status_label, status_emoji).
    """
    if baseline_threshold == 0:
        return "UNKNOWN", ""

    drift_pct = abs(dynamic_threshold - baseline_threshold) / baseline_threshold * 100

    if drift_pct < 10:
        return "STABLE", "STABLE"
    elif drift_pct < 20:
        return "DRIFTING", "DRIFTING"
    else:
        return "FALLBACK ACTIVE", "FALLBACK ACTIVE"


class LatencyTracker:
    """Rolling latency statistics tracker.

    Maintains a circular buffer of latency measurements
    for p50/p95/p99 computation.
    """

    def __init__(self, maxlen: int = 1000) -> None:
        self._measurements: Deque[float] = deque(maxlen=maxlen)
        self._timestamps: Deque[float] = deque(maxlen=maxlen)

    def record(self, latency_ms: float) -> None:
        """Record a latency measurement."""
        self._measurements.append(latency_ms)
        self._timestamps.append(time.time())

    def p95(self) -> float:
        """Compute p95 latency in ms."""
        if not self._measurements:
            return 0.0
        sorted_vals = sorted(self._measurements)
        idx = int(len(sorted_vals) * 0.95)
        return sorted_vals[min(idx, len(sorted_vals) - 1)]

    def stats(self) -> Dict[str, float]:
        """Compute full latency statistics."""
        if not self._measurements:
            return {"min": 0, "max": 0, "p50": 0, "p95": 0, "p99": 0}

        sorted_vals = sorted(self._measurements)
        n = len(sorted_vals)
        return {
            "min": sorted_vals[0],
            "max": sorted_vals[-1],
            "p50": sorted_vals[int(n * 0.5)],
            "p95": sorted_vals[min(int(n * 0.95), n - 1)],
            "p99": sorted_vals[min(int(n * 0.99), n - 1)],
        }
