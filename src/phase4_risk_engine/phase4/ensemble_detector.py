"""Ensemble anomaly detector — combines multiple diverse detectors.

Voting strategy: max-risk (conservative, safety-first for medical).
If ANY detector flags an anomaly and the main model says NORMAL/LOW,
escalate to MEDIUM for investigation.

Detectors:
  1. IsolationForest (statistical outlier detection)
  2. AutoencoderDetector (reconstruction error)

Both are unsupervised — trained on benign-only flows, no labels needed.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

from .autoencoder_detector import AutoencoderDetector

logger = logging.getLogger(__name__)


class EnsembleDetector:
    """Combines IsolationForest + Autoencoder for diverse anomaly detection.

    Args:
        contamination: Expected attack fraction for IsolationForest.
        autoencoder_dim: Latent dimension for autoencoder.
    """

    def __init__(
        self,
        contamination: float = 0.10,
        autoencoder_dim: int = 8,
    ) -> None:
        self._iforest = IsolationForest(
            contamination=contamination, random_state=42, n_estimators=100,
        )
        self._autoencoder = AutoencoderDetector(encoding_dim=autoencoder_dim)
        self._fitted = False

    def fit(self, X_normal: np.ndarray) -> Dict[str, Any]:
        """Fit all ensemble members on benign-only features.

        Args:
            X_normal: Benign flows, shape (N, 24).

        Returns:
            Fitting summary.
        """
        if len(X_normal) < 20:
            logger.warning("Too few benign samples for ensemble: %d", len(X_normal))
            return {"fitted": False, "reason": "insufficient_samples"}

        # IsolationForest
        self._iforest.fit(X_normal)

        # Autoencoder
        ae_summary = self._autoencoder.build_and_train(X_normal, epochs=30)

        self._fitted = True
        logger.info(
            "Ensemble fitted: %d benign samples, IF + AE (%d params)",
            len(X_normal), ae_summary.get("params", 0),
        )

        return {
            "fitted": True,
            "samples": len(X_normal),
            "autoencoder": ae_summary,
        }

    def score(self, features: np.ndarray) -> Dict[str, Any]:
        """Score a single flow with all detectors.

        Args:
            features: Raw feature vector, shape (24,).

        Returns:
            Dict with per-detector results and ensemble decision.
        """
        if not self._fitted:
            return {"ensemble_flag": False, "detectors": {}}

        x = features.reshape(1, -1)

        # IsolationForest: -1 = anomaly, 1 = normal
        if_score = -float(self._iforest.decision_function(x)[0])
        if_flag = if_score > 0  # positive = more anomalous

        # Autoencoder
        ae_result = self._autoencoder.score(features)
        ae_flag = ae_result["autoencoder_flag"]

        # Voting: any detector flags → ensemble flags
        n_flagged = sum([if_flag, ae_flag])
        ensemble_flag = n_flagged >= 1

        return {
            "ensemble_flag": ensemble_flag,
            "detectors_flagged": n_flagged,
            "total_detectors": 2,
            "isolation_forest": {
                "flag": if_flag,
                "score": round(if_score, 4),
            },
            "autoencoder": {
                "flag": ae_flag,
                "error": ae_result.get("reconstruction_error", 0),
            },
        }

    @property
    def fitted(self) -> bool:
        return self._fitted

    def get_config(self) -> Dict[str, Any]:
        return {
            "fitted": self._fitted,
            "detectors": ["IsolationForest", "Autoencoder"],
            "autoencoder": self._autoencoder.get_config(),
        }
