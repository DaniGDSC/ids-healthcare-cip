"""Autoencoder anomaly detector — reconstruction-based.

Trains on benign-only flows. At inference, high reconstruction error
indicates anomaly. Provides per-feature error for explainability.

Complements the CNN-BiLSTM classifier: detects ANY anomaly (not just
known attack types) because attacks reconstruct poorly.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class AutoencoderDetector:
    """Anomaly detection via reconstruction error.

    Args:
        encoding_dim: Latent space dimension.
        threshold_percentile: Percentile of benign errors for anomaly threshold.
    """

    def __init__(self, encoding_dim: int = 8, threshold_percentile: float = 95.0) -> None:
        self._encoding_dim = encoding_dim
        self._threshold_pct = threshold_percentile
        self._model = None
        self._threshold: float = 0.0
        self._fitted = False

    def build_and_train(self, X_normal: np.ndarray, epochs: int = 50) -> Dict[str, Any]:
        """Build and train autoencoder on benign feature vectors.

        Args:
            X_normal: Benign flows, shape (N, 24).
            epochs: Training epochs.

        Returns:
            Training summary dict.
        """
        import tensorflow as tf

        n_features = X_normal.shape[1]

        inp = tf.keras.Input(shape=(n_features,))
        encoded = tf.keras.layers.Dense(16, activation="relu")(inp)
        encoded = tf.keras.layers.Dense(self._encoding_dim, activation="relu")(encoded)
        decoded = tf.keras.layers.Dense(16, activation="relu")(encoded)
        decoded = tf.keras.layers.Dense(n_features, activation="linear")(decoded)

        self._model = tf.keras.Model(inp, decoded, name="anomaly_autoencoder")
        self._model.compile(optimizer="adam", loss="mse")

        self._model.fit(
            X_normal, X_normal,
            epochs=epochs, batch_size=32, verbose=0,
            validation_split=0.1,
        )

        # Set threshold at percentile of benign reconstruction errors
        recon = self._model.predict(X_normal, verbose=0)
        errors = np.mean((X_normal - recon) ** 2, axis=1)
        self._threshold = float(np.percentile(errors, self._threshold_pct))
        self._fitted = True

        logger.info(
            "Autoencoder trained: %d samples, threshold=%.6f (P%.0f)",
            len(X_normal), self._threshold, self._threshold_pct,
        )

        return {
            "samples": len(X_normal),
            "threshold": self._threshold,
            "mean_error": float(np.mean(errors)),
            "max_error": float(np.max(errors)),
            "params": self._model.count_params(),
        }

    def score(self, features: np.ndarray) -> Dict[str, Any]:
        """Score a single flow by reconstruction error.

        Args:
            features: Raw feature vector, shape (24,).

        Returns:
            Dict with error, flag, and per-feature errors.
        """
        if not self._fitted or self._model is None:
            return {"autoencoder_flag": False, "reconstruction_error": 0.0}

        x = features.reshape(1, -1)
        recon = self._model.predict(x, verbose=0)
        per_feature_error = (x - recon).ravel() ** 2
        total_error = float(np.mean(per_feature_error))

        return {
            "autoencoder_flag": total_error > self._threshold,
            "reconstruction_error": round(total_error, 6),
            "threshold": round(self._threshold, 6),
            "per_feature_error": per_feature_error.tolist(),
        }

    @property
    def fitted(self) -> bool:
        return self._fitted

    def get_config(self) -> Dict[str, Any]:
        return {
            "fitted": self._fitted,
            "encoding_dim": self._encoding_dim,
            "threshold": self._threshold,
            "threshold_percentile": self._threshold_pct,
        }
