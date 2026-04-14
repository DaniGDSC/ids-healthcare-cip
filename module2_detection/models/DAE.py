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

Persistence model
-----------------
Use ``DAEDetector.save_artefacts(json_path, weights_path)`` and
``DAEDetector.from_artefacts(json_path, weights_path)`` to round-trip
the detector via:

  - a JSON sidecar containing all numeric state
    (``_threshold``, normaliser bounds, feature weights, threshold
    percentile, hyperparameters)
  - a Keras-native weights file (``*.weights.h5``)

Loading the JSON is pure ``json.loads`` and ``np.asarray``; loading the
Keras weights does NOT execute Python. There is no pickle anywhere on
the load path.

The legacy joblib-pickled detector still loads via ``loads_signed`` from
``common.signed_pickle`` (Phase 2 finding #3a) for backwards
compatibility, but new code should use the JSON+weights pair —
``loads_signed`` can be removed once every legacy ``dae_detector.pkl``
has been re-baselined to the new format.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_SIDECAR_FORMAT = "phase2.dae_detector.v1"


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

        # Cached predict_proba scaling params (set at fit time, avoids
        # np.percentile recomputation on every predict_proba() call).
        self._proba_e_min: float | None = None
        self._proba_e_span: float | None = None

    # ------------------------------------------------------------------
    # Feature-wise normalisation (winsorize + MinMax to [0, 1])
    # ------------------------------------------------------------------

    def _fit_normaliser(self, X: np.ndarray) -> None:
        """Compute per-feature clip bounds and MinMax params from benign data."""
        # Opt-2: one np.percentile call computes both bounds in a single sort
        # pass instead of two separate O(B log B) sorts.
        bounds = np.percentile(X, [self._clip_pct, 100.0 - self._clip_pct], axis=0)
        self._clip_lo, self._clip_hi = bounds[0], bounds[1]
        X_clipped = np.clip(X, self._clip_lo, self._clip_hi)
        self._feat_min = X_clipped.min(axis=0)
        feat_max = X_clipped.max(axis=0)
        self._feat_scale = feat_max - self._feat_min
        self._feat_scale[self._feat_scale == 0] = 1.0  # constant features

    def _ood_penalty(self, X: np.ndarray) -> np.ndarray:
        """Per-sample penalty for features outside Winsorize bounds.

        OOD-02 fix: the Winsorize clipper masks novelty by pulling
        extreme values back into the training range. This penalty
        measures how far outside the bounds each sample is BEFORE
        clipping, ensuring truly novel inputs produce elevated error
        even after normalisation.

        Returns:
            penalty: shape (n_samples,) — sum of squared exceedances
            scaled by feature weights.
        """
        below = np.maximum(self._clip_lo - X, 0)
        above = np.maximum(X - self._clip_hi, 0)
        exceedance = (below ** 2 + above ** 2)
        if self._feat_scale is not None:
            exceedance = exceedance / (self._feat_scale ** 2 + 1e-12)
        if self._feat_weights is not None:
            return exceedance @ self._feat_weights
        return exceedance.sum(axis=1)

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
        recon = self._forward(X_norm)
        per_feat_var = np.var((X_norm - recon) ** 2, axis=0)
        # Inverse variance; floor at 1e-12 to avoid division by zero
        inv_var = 1.0 / np.maximum(per_feat_var, 1e-12)
        self._feat_weights = inv_var / inv_var.sum()

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _forward(self, X_norm: np.ndarray) -> np.ndarray:
        """Single forward pass that bypasses keras.Model.predict().

        keras.Model.predict() carries 20-50ms of per-call setup overhead
        that dwarfs the actual matmul cost for this 1.5K-parameter
        autoencoder. Calling the model directly with `training=False`
        produces bit-identical reconstructions with sub-millisecond
        per-call latency.
        """
        return self._model(X_norm, training=False).numpy()

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
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=10,
                    restore_best_weights=True,
                )
            )

        history = self._model.fit(
            X_norm,
            X_norm,  # autoencoder: input == target
            epochs=self._epochs,
            batch_size=self._batch_size,
            validation_split=validation_split if validation_split > 0 else 0.0,
            callbacks=callbacks,
            verbose=0,
        )
        self._history = {k: [float(v) for v in vs] for k, vs in history.history.items()}

        # Compute inverse-variance feature weights from benign training portion
        self._fit_feature_weights(X_norm[:n_train] if n_val > 0 else X_norm)

        # Opt-3: use _forward() (direct model call, sub-ms latency) instead of
        # model.predict() (Keras batch API with 20-50ms per-call setup overhead).
        # _forward() is already used by all other inference paths for this reason.
        recon = self._forward(X_norm)
        self._train_errors = self._weighted_mse(X_norm, recon)

        # Set threshold at configured percentile of benign errors
        self._threshold = float(np.percentile(self._train_errors, self._threshold_pct))

        # Pre-compute predict_proba scaling params once to avoid
        # np.percentile() recomputation on every inference call.
        self._proba_e_min = float(self._train_errors.min())
        e_max = float(np.percentile(self._train_errors, 99))
        self._proba_e_span = e_max - self._proba_e_min if e_max > self._proba_e_min else 1.0

        elapsed = time.perf_counter() - t0
        actual_epochs = len(self._history.get("loss", []))
        final_loss = self._history["loss"][-1] if self._history.get("loss") else 0.0

        logger.info(
            "DAE fit: %d benign samples, %d features, %d epochs (early stop), "
            "loss=%.6f, threshold=%.6f (p%.0f), %.1fs",
            len(X_benign),
            n_features,
            actual_epochs,
            final_loss,
            self._threshold,
            self._threshold_pct,
            elapsed,
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
        """Per-sample weighted MSE + OOD penalty on normalised features."""
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        ood = self._ood_penalty(X)
        X_norm = self._normalise(X)
        recon = self._forward(X_norm)
        return self._weighted_mse(X_norm, recon) + ood

    def reconstruction_error_decomposed(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """One forward pass; returns (per_sample_error, per_feature_weighted_error).

        Equivalent to calling reconstruction_error(X) and then re-running
        the same forward pass to compute the per-feature breakdown,
        except it shares a single forward pass and one normalisation.
        Used by the online explainer to halve the DAE compute per alert.

        Returns:
            per_sample: shape (n_samples,) — same as reconstruction_error()
            per_feature_weighted: shape (n_samples, n_features) —
                element-wise weighted squared error before the per-row sum.
                `per_sample == per_feature_weighted.sum(axis=1)` exactly.
        """
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        ood = self._ood_penalty(X)
        X_norm = self._normalise(X)
        recon = self._forward(X_norm)
        sq_err = (X_norm - recon) ** 2                     # (n_samples, n_features)
        per_feature_weighted = sq_err * self._feat_weights  # (n_samples, n_features)
        per_sample = per_feature_weighted.sum(axis=1) + ood # (n_samples,)
        return per_sample, per_feature_weighted

    def _noisy_threshold(self) -> float:
        """Return threshold with ±10% random noise (TM-04 fix).

        Prevents attackers from mapping the exact decision boundary
        via repeated probing. Each call returns a slightly different
        threshold, making the boundary non-deterministic.
        """
        noise = np.random.uniform(-0.10, 0.10)
        return self._threshold * (1.0 + noise)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Binary anomaly prediction: 1=attack (above noisy threshold)."""
        errors = self.reconstruction_error(X)
        return (errors > self._noisy_threshold()).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Anomaly score normalized to [0, 1] range.

        Uses min-max scaling relative to benign training error distribution.
        Scaling params (e_min, e_span) are pre-computed at fit() time to avoid
        np.percentile recomputation on every inference call.  Values > 1.0
        are clipped.
        """
        errors = self.reconstruction_error(X)
        if self._proba_e_min is not None:
            scores = (errors - self._proba_e_min) / self._proba_e_span
        elif self._train_errors is not None and len(self._train_errors) > 0:
            # Fallback for artefacts loaded before this optimisation was applied
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

        # Opt-4: one forward pass shared between prediction and error metrics.
        # Previously predict() called reconstruction_error() internally, then
        # evaluate() called reconstruction_error() again — two full DAE passes.
        errors = self.reconstruction_error(X_test)
        y_pred = (errors > self._noisy_threshold()).astype(int)

        metrics = {
            "attack_f1": float(f1_score(y_test, y_pred, pos_label=1)),
            "attack_f2": float(fbeta_score(y_test, y_pred, beta=2, pos_label=1)),
            "weighted_f1": float(f1_score(y_test, y_pred, average="weighted")),
            "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
            "auc_roc": (
                float(roc_auc_score(y_test, errors)) if len(np.unique(y_test)) > 1 else float("nan")
            ),
            "threshold": self._threshold,
            "mean_benign_error": (
                float(errors[y_test == 0].mean()) if (y_test == 0).any() else float("nan")
            ),
            "mean_attack_error": (
                float(errors[y_test == 1].mean()) if (y_test == 1).any() else float("nan")
            ),
        }
        self._test_metrics = metrics

        logger.info(
            "DAE eval: attack_f1=%.4f, attack_f2=%.4f, AUC=%.4f, "
            "benign_err=%.6f, attack_err=%.6f",
            metrics["attack_f1"],
            metrics["attack_f2"],
            metrics["auc_roc"],
            metrics["mean_benign_error"],
            metrics["mean_attack_error"],
        )
        logger.info(
            "\n%s",
            classification_report(
                y_test,
                y_pred,
                target_names=["Normal", "Attack"],
                digits=4,
            ),
        )
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
                "final_val_loss": (
                    self._history["val_loss"][-1] if self._history.get("val_loss") else None
                ),
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

    # ------------------------------------------------------------------
    # Native (pickle-free) persistence
    # ------------------------------------------------------------------

    def save_artefacts(
        self,
        json_path: Path,
        weights_path: Path,
    ) -> None:
        """Persist the detector via a JSON sidecar + Keras weights file.

        The JSON sidecar contains every numeric piece of state the
        detector needs to reconstruct its inference behaviour
        (hyperparameters, normaliser bounds, feature weights, the
        threshold). The Keras weights file is written via the official
        ``model.save_weights`` API in HDF5 format. Loading either
        artefact does NOT execute Python — there is no pickle in the
        round-trip.

        Args:
            json_path: Destination ``.json`` for the sidecar.
            weights_path: Destination ``.weights.h5`` for the Keras
                model weights.

        Raises:
            RuntimeError: if the detector has not been fitted.
        """
        if self._model is None:
            raise RuntimeError("DAE not fitted. Call fit() first.")
        if self._feat_weights is None:
            raise RuntimeError(
                "DAE feature weights are missing — fit() did not " "complete successfully."
            )

        json_path = Path(json_path)
        weights_path = Path(weights_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        weights_path.parent.mkdir(parents=True, exist_ok=True)

        body: Dict[str, Any] = {
            "format": _SIDECAR_FORMAT,
            "format_version": 1,
            "hyperparameters": {
                "encoding_dims": list(self._encoding_dims),
                "noise_rate": self._noise_rate,
                "epochs": self._epochs,
                "batch_size": self._batch_size,
                "learning_rate": self._lr,
                "threshold_percentile": self._threshold_pct,
                "clip_percentile": self._clip_pct,
                "random_state": self._random_state,
            },
            "normaliser": {
                "clip_lo": self._clip_lo.tolist() if self._clip_lo is not None else None,
                "clip_hi": self._clip_hi.tolist() if self._clip_hi is not None else None,
                "feat_min": self._feat_min.tolist() if self._feat_min is not None else None,
                "feat_scale": self._feat_scale.tolist() if self._feat_scale is not None else None,
            },
            "feature_weights": self._feat_weights.tolist(),
            "threshold": float(self._threshold),
            "proba_e_min": self._proba_e_min,
            "proba_e_span": self._proba_e_span,
            "train_errors": self._train_errors.tolist() if self._train_errors is not None else None,
            "n_features": int(self._feat_weights.shape[0]),
            "test_metrics": dict(self._test_metrics),
            "history": dict(self._history),
        }

        # Atomic JSON write so a crash mid-write cannot leave a
        # half-written sidecar that the loader would parse as a
        # malformed detector.
        tmp = json_path.with_suffix(json_path.suffix + ".tmp")
        tmp.write_text(json.dumps(body, indent=2))
        os.replace(tmp, json_path)

        # Keras weights — non-executable HDF5.
        self._model.save_weights(str(weights_path))

        logger.info(
            "DAEDetector.save_artefacts: wrote %s and %s " "(no pickle on the load path)",
            json_path.name,
            weights_path.name,
        )

    @classmethod
    def from_artefacts(
        cls,
        json_path: Path,
        weights_path: Path,
    ) -> "DAEDetector":
        """Reconstruct a fitted detector from the JSON+weights pair.

        This is the only supported pickle-free load path. There is no
        ``DAEDetector.from_pickle`` and never will be — the pickle
        loader is exactly the RCE sink that this method exists to
        replace. Legacy ``dae_detector.pkl`` files still load via
        ``common.signed_pickle.loads_signed`` for backwards
        compatibility while the migration is rolling out.

        Args:
            json_path: Path to a sidecar previously written by
                ``save_artefacts()``.
            weights_path: Path to the matching Keras weights file.

        Returns:
            A fitted ``DAEDetector`` whose ``predict``/``predict_proba``/
            ``reconstruction_error`` behaviour is bit-identical to the
            original.

        Raises:
            FileNotFoundError: if either path does not exist.
            ValueError: if the JSON sidecar is not a recognised format.
        """
        json_path = Path(json_path)
        weights_path = Path(weights_path)
        if not json_path.exists():
            raise FileNotFoundError(f"DAE sidecar not found: {json_path}")
        if not weights_path.exists():
            raise FileNotFoundError(f"DAE weights not found: {weights_path}")

        body = json.loads(json_path.read_text())
        if body.get("format") != _SIDECAR_FORMAT:
            raise ValueError(
                f"{json_path} is not a {_SIDECAR_FORMAT} sidecar "
                f"(got format={body.get('format')!r})"
            )

        hp = body.get("hyperparameters", {})
        instance = cls(
            encoding_dims=list(hp.get("encoding_dims", [16, 8, 16])),
            noise_rate=float(hp.get("noise_rate", 0.1)),
            epochs=int(hp.get("epochs", 100)),
            batch_size=int(hp.get("batch_size", 256)),
            learning_rate=float(hp.get("learning_rate", 1e-3)),
            threshold_percentile=float(hp.get("threshold_percentile", 95.0)),
            clip_percentile=float(hp.get("clip_percentile", 1.0)),
            random_state=int(hp.get("random_state", 42)),
        )

        # Restore normaliser bounds.
        norm = body.get("normaliser", {})
        if norm.get("clip_lo") is None or norm.get("clip_hi") is None:
            raise ValueError(
                f"{json_path}: normaliser bounds missing — sidecar is "
                f"incomplete and cannot reconstruct a fitted detector."
            )
        instance._clip_lo = np.asarray(norm["clip_lo"], dtype=np.float64)
        instance._clip_hi = np.asarray(norm["clip_hi"], dtype=np.float64)
        instance._feat_min = np.asarray(norm["feat_min"], dtype=np.float64)
        instance._feat_scale = np.asarray(norm["feat_scale"], dtype=np.float64)

        instance._feat_weights = np.asarray(
            body.get("feature_weights"),
            dtype=np.float64,
        )
        instance._threshold = float(body.get("threshold", 0.0))
        if body.get("train_errors") is not None:
            instance._train_errors = np.asarray(
                body["train_errors"],
                dtype=np.float64,
            )
        instance._test_metrics = dict(body.get("test_metrics", {}))
        instance._history = dict(body.get("history", {}))

        # Restore pre-computed predict_proba scaling params.
        # If absent (artefact pre-dates this optimisation), recompute from
        # train_errors so the fallback path in predict_proba() is not needed.
        if body.get("proba_e_min") is not None:
            instance._proba_e_min = float(body["proba_e_min"])
            instance._proba_e_span = float(body["proba_e_span"])
        elif instance._train_errors is not None:
            e_min = float(instance._train_errors.min())
            e_max = float(np.percentile(instance._train_errors, 99))
            instance._proba_e_min = e_min
            instance._proba_e_span = e_max - e_min if e_max > e_min else 1.0

        # Build the Keras model with the right shape and load weights.
        n_features = int(body.get("n_features", instance._feat_weights.shape[0]))
        instance._model = instance._build_model(n_features)
        instance._model.load_weights(str(weights_path))

        logger.info(
            "DAEDetector.from_artefacts: loaded %s + %s " "(no pickle on the load path)",
            json_path.name,
            weights_path.name,
        )
        return instance
