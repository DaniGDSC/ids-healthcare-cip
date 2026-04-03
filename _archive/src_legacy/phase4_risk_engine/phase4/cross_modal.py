"""Cross-modal fusion detection — biometric + network anomaly detector.

CRITICAL risk requires BOTH biometric AND network modalities to show
anomalous values simultaneously (values beyond sigma_threshold standard
deviations in both modalities).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

from .base import BaseDetector

logger = logging.getLogger(__name__)


class CrossModalFusionDetector(BaseDetector):
    """Detect cross-modal anomalies (biometric + network simultaneously).

    Args:
        biometric_columns: Names of biometric feature columns.
        sigma_threshold: Standard deviation threshold for anomaly (default 2.0).
    """

    def __init__(
        self,
        biometric_columns: List[str],
        sigma_threshold: float = 2.0,
    ) -> None:
        self._biometric_columns = biometric_columns
        self._sigma_threshold = sigma_threshold

    def detect(
        self,
        feature_values: np.ndarray,
        feature_names: List[str],
    ) -> bool:
        """Check if both biometric and network modalities are anomalous.

        Args:
            feature_values: Raw feature values for a single sample, shape (F,).
            feature_names: List of all feature names matching feature_values.

        Returns:
            True if both biometric AND network modalities exceed threshold.
        """
        bio_indices = [
            feature_names.index(c) for c in self._biometric_columns if c in feature_names
        ]
        net_indices = [i for i in range(len(feature_names)) if i not in bio_indices]

        bio_vals = feature_values[bio_indices] if bio_indices else np.array([])
        net_vals = feature_values[net_indices] if net_indices else np.array([])

        bio_anomaly = (
            bool(np.any(np.abs(bio_vals) > self._sigma_threshold)) if len(bio_vals) > 0 else False
        )
        net_anomaly = (
            bool(np.any(np.abs(net_vals) > self._sigma_threshold)) if len(net_vals) > 0 else False
        )

        return bio_anomaly and net_anomaly

    def get_config(self) -> Dict[str, Any]:
        """Return cross-modal detector configuration."""
        return {
            "biometric_columns": list(self._biometric_columns),
            "sigma_threshold": self._sigma_threshold,
        }
