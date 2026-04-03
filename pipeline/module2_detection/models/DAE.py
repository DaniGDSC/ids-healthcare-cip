"""Denoising Autoencoder (DAE) backbone for novelty-based intrusion detection.

Track B detector: trained on benign-only traffic, flags anomalies by
reconstruction error exceeding a threshold.

Architecture:
  Input (n_features)
    → Winsorize + MinMax [0,1] per-feature normalisation
    → Dropout(noise_rate)                     [denoising corruption]
    → Dense(encoder, relu)                    [encoder]
    → Dense(bottleneck, relu)                 [bottleneck, < n_features]
    → Dense(decoder, relu)                    [decoder]
    → Dense(n_features, sigmoid)              [reconstruction in [0,1]]

Anomaly scoring:
  - Per-sample MSE between normalised input and reconstruction
  - Threshold set at percentile of benign training errors
  - Samples above threshold classified as attack
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class DAEDetector:
    """Denoising Autoencoder for novelty-based anomaly detection.

    Trained on benign-only data.  At inference, high reconstruction
    error indicates an anomaly (attack).

    Args:
        encoding_dims: Hidden layer sizes [encoder, bottleneck, decoder].
            The bottleneck (middle) dimension must be < n_features to
            force compression; a ValueError is raised otherwise.
        noise_rate: Dropout rate applied to input during training
            (denoising corruption).
        epochs: Training epochs.
        batch_size: Mini-batch size.
        learning_rate: Adam learning rate.
        threshold_percentile: Percentile of benign training errors
            used as the anomaly threshold.
        clip_percentile: Winsorize features at this lower/upper
            percentile before MinMax scaling (default 1/99).
        random_state: Seed for reproducibility.
    """

    def __init__(
        self,
        encoding_dims: List[int] | None = None,
        noise_rate: float = 0.1,
        epochs: int = 100,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        threshold_percentile: float = 95.0,
        clip_percentile: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self._encoding_dims = encoding_dims or [16, 8, 16]
        self._noise_rate = noise_rate
        self._epochs = epochs
        self._batch_size = batch_size
        self._lr = learning_rate
        self._threshold_pct = threshold_percentile
        self._clip_pct = clip_percentile
        self._random_state = random_state

        self._model = None
        self._threshold: float = 0.0
        self._train_errors: np.ndarray | None = None
        self._history: Dict[str, List[float]] = {}
        self._test_metrics: Dict[str, float] = {}

        # Per-feature normalisation params (fit on benign train)
        self._clip_lo: np.ndarray | None = None
        self._clip_hi: np.ndarray | None = None
        self._feat_min: np.ndarray | None = None
        self._feat_scale: np.ndarray | None = None

        # Per-feature inverse-variance weights (fit after training)
        self._feat_weights: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Feature-wise normalisation (winsorize + MinMax to [0, 1])
    # ------------------------------------------------------------------

    def _fit_normaliser(self, X: np.ndarray) -> None:
        """Compute per-feature clip bounds and MinMax params from benign data."""
        self._clip_lo = np.percentile(X, self._clip_pct, axis=0)
        self._clip_hi = np.percentile(X, 100.0 - self._clip_pct, axis=0)
        X_clipped = np.clip(X, self._clip_lo, self._clip_hi)
        self._feat_min = X_clipped.min(axis=0)
        feat_max = X_clipped.max(axis=0)
        self._feat_scale = feat_max - self._feat_min
        self._feat_scale[self._feat_scale == 0] = 1.0  # constant features

    def _normalise(self, X: np.ndarray) -> np.ndarray:
        """Winsorize and MinMax-scale features to [0, 1]."""
        X_clipped = np.clip(X, self._clip_lo, self._clip_hi)
        return (X_clipped - self._feat_min) / self._feat_scale

    def _fit_feature_weights(self, X_norm: np.ndarray) -> None:
        """Compute inverse-variance feature weights from benign reconstruction.

        Features the model reconstructs tightly (low error variance) get
        high weight — deviations on those features are strong anomaly
        signals.  Features with high error variance are noisy and get
        down-weighted.  Weights are normalised to sum to 1.
        """
        recon = self._model.predict(X_norm, verbose=0)
        per_feat_var = np.var((X_norm - recon) ** 2, axis=0)
        # Inverse variance; floor at 1e-12 to avoid division by zero
        inv_var = 1.0 / np.maximum(per_feat_var, 1e-12)
        self._feat_weights = inv_var / inv_var.sum()

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------

    def _build_model(self, n_features: int):
        """Build Keras autoencoder with dropout noise.

        Validates that the bottleneck dimension is strictly less than
        n_features to enforce an information bottleneck.
        """
        import keras
        from keras import layers

        dims = self._encoding_dims
        if dims[1] >= n_features:
            raise ValueError(
                f"Bottleneck dim ({dims[1]}) must be < n_features "
                f"({n_features}) to force compression."
            )

        inputs = layers.Input(shape=(n_features,))
        x = layers.Dropout(self._noise_rate)(inputs)  # denoising corruption

        # Encoder
        x = layers.Dense(dims[0], activation="relu")(x)

        # Bottleneck
        x = layers.Dense(dims[1], activation="relu")(x)

        # Decoder
        x = layers.Dense(dims[2], activation="relu")(x)

        # Reconstruction — sigmoid to match [0, 1] normalised input
        outputs = layers.Dense(n_features, activation="sigmoid")(x)

        model = keras.Model(inputs, outputs, name="DAE")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self._lr),
            loss="mse",
        )
        return model

    # ------------------------------------------------------------------
    # Train (benign-only)
    # ------------------------------------------------------------------

    def fit(
        self,
        X_benign: np.ndarray,
        validation_split: float = 0.1,
    ) -> DAEDetector:
        """Train autoencoder on benign-only data.

        Args:
            X_benign: Scaled benign training features (y=0 only).
            validation_split: Fraction held for early stopping.

        Returns:
            self
        """
        import keras

        t0 = time.perf_counter()

        # Reproducibility
        rng = np.random.RandomState(self._random_state)
        np.random.seed(self._random_state)
        try:
            import tensorflow as tf
            tf.random.set_seed(self._random_state)
        except ImportError:
            pass

        # Shuffle benign data before validation split so the held-out
        # slice is representative (Keras validation_split takes the
        # last N rows without shuffling first).
        shuffle_idx = rng.permutation(len(X_benign))
        X_benign = X_benign[shuffle_idx]

        # Fit per-feature normaliser on the training portion only
        n_val = int(len(X_benign) * validation_split)
        n_train = len(X_benign) - n_val
        self._fit_normaliser(X_benign[:n_train] if n_val > 0 else X_benign)
        X_norm = self._normalise(X_benign)

        n_features = X_norm.shape[1]
        self._model = self._build_model(n_features)

        callbacks = []
        if validation_split > 0:
            callbacks.append(keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
            ))

        history = self._model.fit(
            X_norm, X_norm,  # autoencoder: input == target
            epochs=self._epochs,
            batch_size=self._batch_size,
            validation_split=validation_split if validation_split > 0 else 0.0,
            callbacks=callbacks,
            verbose=0,
        )
        self._history = {k: [float(v) for v in vs] for k, vs in history.history.items()}

        # Compute inverse-variance feature weights from benign training portion
        self._fit_feature_weights(X_norm[:n_train] if n_val > 0 else X_norm)

        # Compute weighted reconstruction errors on normalised training set
        recon = self._model.predict(X_norm, verbose=0)
        self._train_errors = self._weighted_mse(X_norm, recon)

        # Set threshold at configured percentile of benign errors
        self._threshold = float(np.percentile(self._train_errors, self._threshold_pct))

        elapsed = time.perf_counter() - t0
        actual_epochs = len(self._history.get("loss", []))
        final_loss = self._history["loss"][-1] if self._history.get("loss") else 0.0

        logger.info(
            "DAE fit: %d benign samples, %d features, %d epochs (early stop), "
            "loss=%.6f, threshold=%.6f (p%.0f), %.1fs",
            len(X_benign), n_features, actual_epochs,
            final_loss, self._threshold, self._threshold_pct, elapsed,
        )
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def _weighted_mse(self, X_norm: np.ndarray, recon: np.ndarray) -> np.ndarray:
        """Per-sample weighted MSE using inverse-variance feature weights."""
        sq_err = (X_norm - recon) ** 2  # (n_samples, n_features)
        return sq_err @ self._feat_weights  # (n_samples,)

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Per-sample weighted MSE reconstruction error on normalised features."""
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X_norm = self._normalise(X)
        recon = self._model.predict(X_norm, verbose=0)
        return self._weighted_mse(X_norm, recon)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Binary anomaly prediction: 1=attack (above threshold)."""
        errors = self.reconstruction_error(X)
        return (errors > self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Anomaly score normalized to [0, 1] range.

        Uses min-max scaling relative to benign training error
        distribution.  Values > 1.0 are clipped.
        """
        errors = self.reconstruction_error(X)
        if self._train_errors is not None and len(self._train_errors) > 0:
            e_min = float(self._train_errors.min())
            e_max = float(np.percentile(self._train_errors, 99))
            span = e_max - e_min if e_max > e_min else 1.0
            scores = (errors - e_min) / span
        else:
            scores = errors / (self._threshold if self._threshold > 0 else 1.0)
        return np.clip(scores, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate on mixed test set (benign + attack).

        Args:
            X_test: Scaled test features.
            y_test: Binary labels (0=benign, 1=attack).

        Returns:
            Dict of evaluation metrics.
        """
        from sklearn.metrics import (
            classification_report,
            f1_score,
            fbeta_score,
            roc_auc_score,
        )

        y_pred = self.predict(X_test)
        errors = self.reconstruction_error(X_test)

        metrics = {
            "attack_f1": float(f1_score(y_test, y_pred, pos_label=1)),
            "attack_f2": float(fbeta_score(y_test, y_pred, beta=2, pos_label=1)),
            "weighted_f1": float(f1_score(y_test, y_pred, average="weighted")),
            "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
            "auc_roc": float(roc_auc_score(y_test, errors)) if len(np.unique(y_test)) > 1 else float("nan"),
            "threshold": self._threshold,
            "mean_benign_error": float(errors[y_test == 0].mean()) if (y_test == 0).any() else float("nan"),
            "mean_attack_error": float(errors[y_test == 1].mean()) if (y_test == 1).any() else float("nan"),
        }
        self._test_metrics = metrics

        logger.info(
            "DAE eval: attack_f1=%.4f, attack_f2=%.4f, AUC=%.4f, "
            "benign_err=%.6f, attack_err=%.6f",
            metrics["attack_f1"], metrics["attack_f2"], metrics["auc_roc"],
            metrics["mean_benign_error"], metrics["mean_attack_error"],
        )
        logger.info("\n%s", classification_report(
            y_test, y_pred, target_names=["Normal", "Attack"], digits=4,
        ))
        return metrics

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def get_report(self) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "model_type": "Denoising Autoencoder (DAE)",
            "architecture": {
                "encoding_dims": self._encoding_dims,
                "noise_rate": self._noise_rate,
                "learning_rate": self._lr,
                "clip_percentile": self._clip_pct,
            },
            "training": {
                "epochs_run": len(self._history.get("loss", [])),
                "final_loss": self._history["loss"][-1] if self._history.get("loss") else None,
                "final_val_loss": self._history["val_loss"][-1] if self._history.get("val_loss") else None,
            },
            "threshold": self._threshold,
            "threshold_percentile": self._threshold_pct,
            "test_metrics": self._test_metrics,
        }
        if self._feat_weights is not None:
            report["feature_weights"] = self._feat_weights.tolist()
        return report

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def model(self):
        return self._model

    @property
    def train_errors(self) -> np.ndarray | None:
        return self._train_errors
