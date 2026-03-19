"""Risk scoring — classify samples into 5 risk levels.

Uses MAD-relative distance boundaries and delegates CRITICAL
detection to CrossModalFusionDetector.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

from .cross_modal import CrossModalFusionDetector
from .risk_level import RiskLevel

logger = logging.getLogger(__name__)


class RiskScorer:
    """Classify samples into 5 risk levels using MAD-relative boundaries.

    Risk levels::

        NORMAL:   distance < 0
        LOW:      0 <= distance < low_upper * MAD
        MEDIUM:   low_upper * MAD <= distance < medium_upper * MAD
        HIGH:     medium_upper * MAD <= distance < high_upper * MAD
        CRITICAL: distance >= high_upper * MAD AND cross-modal fusion

    Args:
        low_upper: MAD multiplier for LOW/MEDIUM boundary.
        medium_upper: MAD multiplier for MEDIUM/HIGH boundary.
        high_upper: MAD multiplier for HIGH/CRITICAL boundary.
        cross_modal: CrossModalFusionDetector for CRITICAL assessment.
    """

    def __init__(
        self,
        low_upper: float = 0.5,
        medium_upper: float = 1.0,
        high_upper: float = 2.0,
        cross_modal: CrossModalFusionDetector | None = None,
    ) -> None:
        self._low_upper = low_upper
        self._medium_upper = medium_upper
        self._high_upper = high_upper
        self._cross_modal = cross_modal

    def score(
        self,
        anomaly_scores: np.ndarray,
        thresholds: np.ndarray,
        mad: float,
        raw_features: np.ndarray,
        feature_names: List[str],
    ) -> List[Dict[str, Any]]:
        """Score all samples into risk levels.

        Args:
            anomaly_scores: Model sigmoid outputs, shape (N,).
            thresholds: Adjusted thresholds, shape (N,).
            mad: Median Absolute Deviation from baseline.
            raw_features: Raw feature values, shape (N, F).
            feature_names: List of all feature names.

        Returns:
            List of per-sample risk dicts with sample_index, anomaly_score,
            threshold, distance, risk_level.
        """
        logger.info("── Risk scoring ──")

        risk_results: List[Dict[str, Any]] = []
        level_counts: Dict[str, int] = {lvl.value: 0 for lvl in RiskLevel}

        for i in range(len(anomaly_scores)):
            score = float(anomaly_scores[i])
            threshold = float(thresholds[i])
            distance = score - threshold

            level = self.classify_single(
                distance=distance,
                mad=mad,
                feature_values=(raw_features[i] if i < len(raw_features) else None),
                feature_names=feature_names,
            )

            level_counts[level.value] += 1
            risk_results.append(
                {
                    "sample_index": i,
                    "anomaly_score": round(score, 6),
                    "threshold": round(threshold, 6),
                    "distance": round(distance, 6),
                    "risk_level": level.value,
                }
            )

        n_total = len(anomaly_scores)
        for lvl, count in level_counts.items():
            pct = count / n_total * 100 if n_total > 0 else 0
            logger.info("  %s: %d (%.1f%%)", lvl, count, pct)

        return risk_results

    def classify_single(
        self,
        distance: float,
        mad: float,
        feature_values: np.ndarray | None = None,
        feature_names: List[str] | None = None,
    ) -> RiskLevel:
        """Classify a single sample based on distance from threshold.

        Args:
            distance: anomaly_score - threshold.
            mad: Median Absolute Deviation from baseline.
            feature_values: Raw features for cross-modal check (optional).
            feature_names: Feature names for cross-modal check (optional).

        Returns:
            RiskLevel enum value.
        """
        low_bound = self._low_upper * mad
        medium_bound = self._medium_upper * mad
        high_bound = self._high_upper * mad

        if distance < 0:
            return RiskLevel.NORMAL
        if distance < low_bound:
            return RiskLevel.LOW
        if distance < medium_bound:
            return RiskLevel.MEDIUM
        if distance < high_bound:
            return RiskLevel.HIGH

        # Check cross-modal fusion for CRITICAL
        if (
            self._cross_modal is not None
            and feature_values is not None
            and feature_names is not None
            and self._cross_modal.detect(feature_values, feature_names)
        ):
            return RiskLevel.CRITICAL

        return RiskLevel.HIGH
