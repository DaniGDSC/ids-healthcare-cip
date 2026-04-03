"""Unit tests for DynamicThresholdUpdater (rolling window + k(t))."""

from __future__ import annotations

import numpy as np

from src.phase4_risk_engine.phase4.base import BaseDetector
from src.phase4_risk_engine.phase4.config import KScheduleEntry
from src.phase4_risk_engine.phase4.dynamic_threshold import DynamicThresholdUpdater


def _make_k_schedule() -> list:
    return [
        KScheduleEntry(start_hour=0, end_hour=6, k=2.5),
        KScheduleEntry(start_hour=6, end_hour=22, k=3.0),
        KScheduleEntry(start_hour=22, end_hour=24, k=3.5),
    ]


class TestDynamicThresholdUpdater:
    """Test dynamic threshold computation."""

    def test_implements_base(self) -> None:
        assert issubclass(DynamicThresholdUpdater, BaseDetector)

    def test_update_returns_arrays(self) -> None:
        np.random.seed(42)
        scores = np.random.rand(500)
        baseline = {"baseline_threshold": 0.5}
        updater = DynamicThresholdUpdater(window_size=100, k_schedule=_make_k_schedule())
        thresholds, window_log = updater.update(scores, baseline)

        assert thresholds.shape == scores.shape
        assert len(window_log) > 0

    def test_k_for_hour_night(self) -> None:
        updater = DynamicThresholdUpdater(k_schedule=_make_k_schedule())
        assert updater.get_k_for_hour(2) == 2.5

    def test_k_for_hour_day(self) -> None:
        updater = DynamicThresholdUpdater(k_schedule=_make_k_schedule())
        assert updater.get_k_for_hour(12) == 3.0

    def test_k_for_hour_evening(self) -> None:
        updater = DynamicThresholdUpdater(k_schedule=_make_k_schedule())
        assert updater.get_k_for_hour(23) == 3.5

    def test_k_for_hour_default(self) -> None:
        updater = DynamicThresholdUpdater(k_schedule=[])
        assert updater.get_k_for_hour(12) == 3.0

    def test_window_log_entries(self) -> None:
        np.random.seed(42)
        scores = np.random.rand(300)
        baseline = {"baseline_threshold": 0.5}
        updater = DynamicThresholdUpdater(window_size=100, k_schedule=_make_k_schedule())
        _, window_log = updater.update(scores, baseline)

        assert len(window_log) > 0
        entry = window_log[0]
        assert "sample_index" in entry
        assert "hour" in entry
        assert "k_t" in entry
        assert "window_median" in entry
        assert "dynamic_threshold" in entry

    def test_get_config(self) -> None:
        updater = DynamicThresholdUpdater(window_size=100, k_schedule=_make_k_schedule())
        config = updater.get_config()
        assert config["window_size"] == 100
        assert len(config["k_schedule"]) == 3
