"""Threshold fallback manager — lock/resume dynamic thresholds.

When concept drift is detected, locks thresholds to baseline.
Resumes dynamic thresholds after a configurable number of
consecutive stable windows below the recovery threshold.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

from .drift_detector import ConceptDriftDetector

logger = logging.getLogger(__name__)


class ThresholdFallbackManager:
    """Manage threshold fallback locking and recovery.

    Uses a ConceptDriftDetector to decide when to lock/unlock.

    Args:
        drift_detector: ConceptDriftDetector for drift checks.
        baseline_threshold: Static baseline threshold value.
        recovery_threshold: Ratio below which drift is considered recovered.
        recovery_windows: Consecutive stable windows required to resume.
    """

    def __init__(
        self,
        drift_detector: ConceptDriftDetector,
        baseline_threshold: float,
        recovery_threshold: float = 0.10,
        recovery_windows: int = 3,
    ) -> None:
        self._detector = drift_detector
        self._baseline_threshold = baseline_threshold
        self._recovery_threshold = recovery_threshold
        self._recovery_windows = recovery_windows
        self._locked = False

    @property
    def is_locked(self) -> bool:
        """Whether fallback is currently locked to baseline."""
        return self._locked

    def process(
        self,
        dynamic_thresholds: np.ndarray,
        window_size: int,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Apply fallback logic across all windows.

        Iterates over window boundaries, detects drift, and locks/unlocks
        thresholds accordingly.

        Args:
            dynamic_thresholds: Dynamic threshold array from updater.
            window_size: Window size for boundary detection.

        Returns:
            Tuple of (adjusted_thresholds, drift_events list).
        """
        logger.info("── Concept drift detection + fallback ──")
        adjusted = dynamic_thresholds.copy()
        drift_events: List[Dict[str, Any]] = []
        self._locked = False
        consecutive_stable = 0

        for i in range(window_size, len(adjusted)):
            if i % window_size != 0:
                if self._locked:
                    adjusted[i] = self._baseline_threshold
                continue

            drift_ratio = self._detector.compute_drift_ratio(
                float(adjusted[i]), self._baseline_threshold
            )

            if not self._locked and self._detector.detect(
                float(adjusted[i]), self._baseline_threshold
            ):
                self._locked = True
                consecutive_stable = 0
                logger.info(
                    "  Drift detected: ratio=%.4f at sample %d",
                    drift_ratio,
                    i,
                )
                drift_events.append(
                    {
                        "sample_index": int(i),
                        "drift_ratio": round(drift_ratio, 4),
                        "action": "FALLBACK_LOCKED",
                        "dynamic_threshold": round(float(adjusted[i]), 6),
                        "baseline_threshold": round(self._baseline_threshold, 6),
                    }
                )
                adjusted[i] = self._baseline_threshold

            elif self._locked:
                adjusted[i] = self._baseline_threshold
                if drift_ratio < self._recovery_threshold:
                    consecutive_stable += 1
                else:
                    consecutive_stable = 0

                if consecutive_stable >= self._recovery_windows:
                    self._locked = False
                    consecutive_stable = 0
                    logger.info(
                        "  Drift recovered: %d stable windows at sample %d",
                        self._recovery_windows,
                        i,
                    )
                    drift_events.append(
                        {
                            "sample_index": int(i),
                            "drift_ratio": round(drift_ratio, 4),
                            "action": "RESUMED_DYNAMIC",
                            "dynamic_threshold": round(float(dynamic_thresholds[i]), 6),
                            "baseline_threshold": round(self._baseline_threshold, 6),
                        }
                    )

        n_locked = int(np.sum(np.isclose(adjusted, self._baseline_threshold)))
        logger.info(
            "  Drift events: %d, locked samples: %d/%d",
            len(drift_events),
            n_locked,
            len(adjusted),
        )
        return adjusted, drift_events
