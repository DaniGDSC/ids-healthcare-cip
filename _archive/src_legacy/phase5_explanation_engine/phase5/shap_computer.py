"""SHAPComputer — compute SHAP feature attributions for anomaly explanation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from src.phase2_detection_engine.phase2.reshaper import DataReshaper

logger = logging.getLogger(__name__)

from dashboard.streaming.feature_aligner import N_FEATURES as _N_FEATURES  # canonical: 24


class SHAPComputer:
    """Compute SHAP feature attributions for anomaly explanation.

    Uses GradientExplainer as primary method, integrated gradients
    as fallback.

    Args:
        n_background: Number of background samples for SHAP baseline.
        timesteps: Sliding window timestep count.
        stride: Sliding window stride.
        label_column: Name of label column in data.
        n_integration_steps: Steps for integrated gradients fallback.
    """

    def __init__(
        self,
        n_background: int = 100,
        timesteps: int = 20,
        stride: int = 1,
        label_column: str = "Label",
        n_integration_steps: int = 50,
    ) -> None:
        self._n_background = n_background
        self._timesteps = timesteps
        self._stride = stride
        self._label_column = label_column
        self._n_integration_steps = n_integration_steps

    def prepare_background(
        self,
        train_path: Path,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Prepare SHAP background from Normal training samples.

        Args:
            train_path: Path to train_phase1.parquet.
            rng: Random number generator.

        Returns:
            Background data array, shape (n_background, timesteps, n_features).
        """
        logger.info("── Preparing SHAP background data ──")
        train_df = pd.read_parquet(train_path)
        feature_names = [c for c in train_df.columns if c != self._label_column]

        normal_mask = train_df[self._label_column] == 0
        X_normal = train_df.loc[normal_mask, feature_names].values.astype(np.float32)

        reshaper = DataReshaper(timesteps=self._timesteps, stride=self._stride)
        y_dummy = np.zeros(len(X_normal), dtype=np.int32)
        X_windows, _ = reshaper.reshape(X_normal, y_dummy)

        indices = rng.choice(len(X_windows), self._n_background, replace=False)
        background = X_windows[indices]
        logger.info(
            "  Background data: %s from %d Normal windows",
            background.shape,
            len(X_windows),
        )
        return background

    def prepare_explanation_data(
        self,
        test_path: Path,
        sample_indices: List[int],
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare windowed test data for SHAP explanation.

        Args:
            test_path: Path to test_phase1.parquet.
            sample_indices: Indices of samples to explain.

        Returns:
            Tuple of (X_explain, X_all_windows, feature_names).
        """
        test_df = pd.read_parquet(test_path)
        feature_names = [c for c in test_df.columns if c != self._label_column]

        X_test = test_df[feature_names].values.astype(np.float32)
        y_test = test_df[self._label_column].values.astype(np.int32)

        reshaper = DataReshaper(timesteps=self._timesteps, stride=self._stride)
        X_windows, _ = reshaper.reshape(X_test, y_test)

        valid = [i for i in sample_indices if i < len(X_windows)]
        X_explain = X_windows[valid]

        logger.info(
            "  Explanation data: %s (%d samples)",
            X_explain.shape,
            len(valid),
        )
        return X_explain, X_windows, feature_names

    def compute(
        self,
        model: tf.keras.Model,
        background: np.ndarray,
        X_explain: np.ndarray,
    ) -> np.ndarray:
        """Compute SHAP values. Falls back to integrated gradients.

        Args:
            model: Loaded Keras classification model.
            background: Background data, shape (B, T, F).
            X_explain: Samples to explain, shape (N, T, F).

        Returns:
            SHAP values array, shape (N, T, F).
        """
        logger.info("── Computing SHAP values ──")
        logger.info(
            "  Explaining %d samples against %d background",
            len(X_explain),
            len(background),
        )

        try:
            import shap

            explainer = shap.GradientExplainer(model, background)
            shap_vals = explainer.shap_values(X_explain)

            if isinstance(shap_vals, list):
                shap_vals = shap_vals[0]

            shap_vals = np.array(shap_vals)
            if shap_vals.ndim == 4 and shap_vals.shape[-1] == 1:
                shap_vals = shap_vals.squeeze(-1)
            logger.info(
                "  SHAP computed via GradientExplainer: %s",
                shap_vals.shape,
            )
            return shap_vals

        except Exception as e:
            logger.warning(
                "  GradientExplainer failed (%s), using gradient attribution",
                e,
            )
            return self._gradient_attribution(model, background, X_explain)

    def _gradient_attribution(
        self,
        model: tf.keras.Model,
        background: np.ndarray,
        X_explain: np.ndarray,
    ) -> np.ndarray:
        """Integrated gradients fallback for feature attribution.

        Args:
            model: Loaded Keras model.
            background: Background data for baseline.
            X_explain: Samples to explain.

        Returns:
            Attribution values, shape (N, T, F).
        """
        n_steps = self._n_integration_steps
        logger.info("  Computing integrated gradients (%d steps)", n_steps)
        baseline_ref = np.mean(background, axis=0, keepdims=True)
        attributions = np.zeros_like(X_explain)

        batch_size = 32
        for start in range(0, len(X_explain), batch_size):
            end = min(start + batch_size, len(X_explain))
            batch = X_explain[start:end]

            batch_attr = np.zeros_like(batch)
            for step in range(n_steps):
                alpha = step / n_steps
                interpolated = baseline_ref + alpha * (batch - baseline_ref)
                inp = tf.constant(interpolated, dtype=tf.float32)

                with tf.GradientTape() as tape:
                    tape.watch(inp)
                    pred = model(inp, training=False)
                grads = tape.gradient(pred, inp).numpy()
                batch_attr += grads / n_steps

            attributions[start:end] = batch_attr * (batch - baseline_ref)

        logger.info("  Integrated gradients computed: %s", attributions.shape)
        return attributions

    def get_config(self) -> Dict[str, Any]:
        """Return SHAP computer configuration."""
        return {
            "n_background": self._n_background,
            "timesteps": self._timesteps,
            "stride": self._stride,
            "label_column": self._label_column,
            "n_integration_steps": self._n_integration_steps,
        }
