"""Attention-based anomaly detector — novelty/zero-day detection.

Computes L2 distance of attention context vectors from the Normal
baseline. Catches novel attacks that the binary classifier misses
because they produce unusual attention patterns even when the sigmoid
output is low (classifier has never seen this attack type).

This runs PARALLEL to the classification-based scoring — if either
path flags an anomaly, the sample is escalated.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import tensorflow as tf

from src.common.base import BaseDetector

logger = logging.getLogger(__name__)


class AttentionAnomalyDetector(BaseDetector):
    """Detect novel attacks via attention vector distance from Normal baseline.

    Extracts the detection backbone (everything before the classification
    head) and computes per-sample L2 norm of the 128-D context vector.
    Samples whose magnitude exceeds the Normal baseline threshold are
    flagged as attention-anomalous — regardless of the classifier output.

    Args:
        baseline_median: Median L2 norm of Normal attention vectors.
        baseline_mad: MAD of Normal attention vector L2 norms.
        mad_multiplier: k for threshold = median + k * MAD.
    """

    def __init__(
        self,
        baseline_median: float,
        baseline_mad: float,
        mad_multiplier: float = 3.0,
    ) -> None:
        self._median = baseline_median
        self._mad = baseline_mad
        self._k = mad_multiplier
        self._threshold = baseline_median + mad_multiplier * baseline_mad

    def compute_scores(
        self,
        model: tf.keras.Model,
        X_windows: np.ndarray,
    ) -> np.ndarray:
        """Compute per-sample attention anomaly scores.

        Extracts the backbone output (attention context vector) from the
        full classification model and returns L2 norms.

        Args:
            model: Full classification model (backbone + head).
            X_windows: Windowed input, shape (N, timesteps, features).

        Returns:
            L2 norms of attention vectors, shape (N,).
        """
        # Extract backbone: input → attention layer (before dense_head)
        # The attention layer output is the input to "dense_head"
        dense_head = model.get_layer("dense_head")
        backbone_output = dense_head.input
        backbone = tf.keras.Model(model.input, backbone_output, name="backbone_extractor")

        context_vectors = backbone.predict(X_windows, verbose=0)
        magnitudes = np.linalg.norm(context_vectors, axis=1)

        n_anomalous = int(np.sum(magnitudes > self._threshold))
        logger.info(
            "  Attention anomaly: %d/%d samples above threshold (%.4f)",
            n_anomalous, len(magnitudes), self._threshold,
        )
        return magnitudes

    def classify(self, magnitudes: np.ndarray) -> np.ndarray:
        """Classify samples as attention-anomalous or not.

        Args:
            magnitudes: L2 norms from compute_scores(), shape (N,).

        Returns:
            Boolean array, True = attention-anomalous, shape (N,).
        """
        return magnitudes > self._threshold

    def distances(self, magnitudes: np.ndarray) -> np.ndarray:
        """Compute distance above threshold (0 if below).

        Args:
            magnitudes: L2 norms from compute_scores(), shape (N,).

        Returns:
            Distance above threshold (clipped at 0), shape (N,).
        """
        return np.maximum(magnitudes - self._threshold, 0.0)

    def get_config(self) -> Dict[str, Any]:
        return {
            "baseline_median": self._median,
            "baseline_mad": self._mad,
            "mad_multiplier": self._k,
            "threshold": self._threshold,
        }
