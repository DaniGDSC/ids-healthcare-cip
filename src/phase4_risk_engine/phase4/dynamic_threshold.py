"""Dynamic thresholding — rolling window Median + k(t)*MAD.

Adapts anomaly detection thresholds using time-of-day sensitivity
multipliers and a sliding window over recent anomaly scores.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

from .base import BaseDetector
from .config import KScheduleEntry

logger = logging.getLogger(__name__)


class DynamicThresholdUpdater(BaseDetector):
    """Rolling window Median + k(t)*MAD with time-of-day sensitivity.

    Args:
        window_size: Number of samples in the rolling window.
        k_schedule: List of KScheduleEntry with start_hour, end_hour, k.
    """

    def __init__(
        self,
        window_size: int = 100,
        k_schedule: List[KScheduleEntry] | None = None,
    ) -> None:
        self._window_size = window_size
        self._k_schedule = k_schedule or []

    def update(
        self,
        anomaly_scores: np.ndarray,
        baseline: Dict[str, Any],
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Compute dynamic thresholds using rolling Median + k(t)*MAD.

        Args:
            anomaly_scores: Model sigmoid outputs, shape (N,).
            baseline: Baseline config with baseline_threshold.

        Returns:
            Tuple of (dynamic_thresholds array, window_log list).
        """
        logger.info("── Dynamic thresholding ──")
        n_samples = len(anomaly_scores)
        dynamic_thresholds = np.full(n_samples, baseline["baseline_threshold"])
        window_log: List[Dict[str, Any]] = []

        # Simulate time progression (spread across 24 hours)
        hours = np.linspace(0, 24, n_samples, endpoint=False)

        for i in range(self._window_size, n_samples):
            window = anomaly_scores[i - self._window_size : i]
            w_median = float(np.median(window))
            w_mad = float(np.median(np.abs(window - w_median)))

            hour = int(hours[i]) % 24
            k_t = self.get_k_for_hour(hour)
            dyn_threshold = w_median + k_t * max(w_mad, 1e-8)
            dynamic_thresholds[i] = dyn_threshold

            # Log every Nth window for monitoring
            if i % self._window_size == 0:
                window_log.append(
                    {
                        "sample_index": int(i),
                        "hour": hour,
                        "k_t": k_t,
                        "window_median": round(w_median, 6),
                        "window_mad": round(w_mad, 6),
                        "dynamic_threshold": round(dyn_threshold, 6),
                    }
                )

        logger.info(
            "  Dynamic thresholds computed: %d samples, window_size=%d",
            n_samples,
            self._window_size,
        )
        return dynamic_thresholds, window_log

    def get_k_for_hour(self, hour: int) -> float:
        """Lookup sensitivity multiplier k(t) for given hour.

        Args:
            hour: Hour of day (0-23).

        Returns:
            k(t) value from schedule, or 3.0 as default fallback.
        """
        for entry in self._k_schedule:
            if entry.start_hour <= hour < entry.end_hour:
                return entry.k
        return 3.0

    def get_config(self) -> Dict[str, Any]:
        """Return dynamic threshold configuration."""
        return {
            "window_size": self._window_size,
            "k_schedule": [
                {
                    "start_hour": e.start_hour,
                    "end_hour": e.end_hour,
                    "k": e.k,
                }
                for e in self._k_schedule
            ],
        }
