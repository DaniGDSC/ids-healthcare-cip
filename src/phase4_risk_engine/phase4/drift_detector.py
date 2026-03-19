"""Concept drift detection — baseline-relative ratio detector.

Detects when dynamic thresholds diverge significantly from the
baseline, indicating a potential concept drift in the data stream.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from .base import BaseDetector

logger = logging.getLogger(__name__)


class ConceptDriftDetector(BaseDetector):
    """Detect concept drift via baseline-relative ratio.

    Drift is detected when ``|dynamic - baseline| / baseline`` exceeds
    the configured drift threshold.

    Args:
        drift_threshold: Ratio trigger (default 0.20 = 20%).
    """

    def __init__(self, drift_threshold: float = 0.20) -> None:
        self._drift_threshold = drift_threshold

    def detect(
        self,
        dynamic_threshold: float,
        baseline_threshold: float,
    ) -> bool:
        """Check if concept drift is present.

        Args:
            dynamic_threshold: Current dynamic threshold value.
            baseline_threshold: Static baseline threshold.

        Returns:
            True if drift_ratio exceeds drift_threshold.
        """
        ratio = self.compute_drift_ratio(dynamic_threshold, baseline_threshold)
        return ratio > self._drift_threshold

    def compute_drift_ratio(
        self,
        dynamic_threshold: float,
        baseline_threshold: float,
    ) -> float:
        """Compute drift ratio between dynamic and baseline thresholds.

        Args:
            dynamic_threshold: Current dynamic threshold value.
            baseline_threshold: Static baseline threshold.

        Returns:
            ``|dynamic - baseline| / baseline``.
        """
        return abs(dynamic_threshold - baseline_threshold) / max(baseline_threshold, 1e-8)

    def get_config(self) -> Dict[str, Any]:
        """Return drift detector configuration."""
        return {"drift_threshold": self._drift_threshold}
