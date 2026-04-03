"""Unit tests for AlertFilter (non-NORMAL sample filtering)."""

from __future__ import annotations

import numpy as np

from src.phase5_explanation_engine.phase5.alert_filter import AlertFilter


def _make_assessments(
    n_normal: int = 10,
    n_low: int = 3,
    n_medium: int = 2,
    n_high: int = 5,
    n_critical: int = 1,
) -> list:
    """Create fake sample assessment dicts."""
    assessments = []
    idx = 0
    for _ in range(n_normal):
        assessments.append(
            {
                "sample_index": idx,
                "risk_level": "NORMAL",
                "anomaly_score": 0.1,
                "threshold": 0.2,
            }
        )
        idx += 1
    for level, count in [
        ("LOW", n_low),
        ("MEDIUM", n_medium),
        ("HIGH", n_high),
        ("CRITICAL", n_critical),
    ]:
        for _ in range(count):
            assessments.append(
                {
                    "sample_index": idx,
                    "risk_level": level,
                    "anomaly_score": 0.5,
                    "threshold": 0.2,
                }
            )
            idx += 1
    return assessments


class TestAlertFilter:
    def test_filter_removes_normal(self) -> None:
        filt = AlertFilter(max_samples=100)
        rng = np.random.default_rng(42)
        assessments = _make_assessments()
        filtered, _ = filt.filter(assessments, rng)
        levels = {s["risk_level"] for s in filtered}
        assert "NORMAL" not in levels

    def test_filter_preserves_non_normal(self) -> None:
        filt = AlertFilter(max_samples=100)
        rng = np.random.default_rng(42)
        assessments = _make_assessments()
        filtered, _ = filt.filter(assessments, rng)
        assert len(filtered) == 11  # 3+2+5+1

    def test_stratified_sampling(self) -> None:
        filt = AlertFilter(max_samples=5)
        rng = np.random.default_rng(42)
        assessments = _make_assessments(n_normal=0, n_low=10, n_medium=10, n_high=10, n_critical=10)
        filtered, _ = filt.filter(assessments, rng)
        assert len(filtered) <= 5

    def test_level_counts_correct(self) -> None:
        filt = AlertFilter(max_samples=100)
        rng = np.random.default_rng(42)
        assessments = _make_assessments()
        _, counts = filt.filter(assessments, rng)
        assert counts["LOW"] == 3
        assert counts["HIGH"] == 5
        assert counts["CRITICAL"] == 1

    def test_empty_input(self) -> None:
        filt = AlertFilter()
        rng = np.random.default_rng(42)
        filtered, counts = filt.filter([], rng)
        assert filtered == []
        assert all(v == 0 for v in counts.values())

    def test_all_normal(self) -> None:
        filt = AlertFilter()
        rng = np.random.default_rng(42)
        assessments = _make_assessments(n_normal=20, n_low=0, n_medium=0, n_high=0, n_critical=0)
        filtered, _ = filt.filter(assessments, rng)
        assert len(filtered) == 0

    def test_get_config(self) -> None:
        filt = AlertFilter(max_samples=50)
        assert filt.get_config() == {"max_samples": 50}
