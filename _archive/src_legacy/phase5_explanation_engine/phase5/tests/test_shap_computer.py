"""Unit tests for SHAPComputer (SHAP feature attribution)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src.phase5_explanation_engine.phase5.shap_computer import SHAPComputer


def _make_dummy_model(timesteps: int = 5, n_features: int = 3) -> tf.keras.Model:
    """Build a minimal model for testing gradient attribution."""
    inp = tf.keras.Input(shape=(timesteps, n_features))
    x = tf.keras.layers.Flatten()(inp)
    x = tf.keras.layers.Dense(8, activation="relu")(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inp, out)


class TestSHAPComputer:
    def test_gradient_attribution_shape(self) -> None:
        model = _make_dummy_model(timesteps=5, n_features=3)
        computer = SHAPComputer(timesteps=5, stride=1, n_integration_steps=5)
        bg = np.random.randn(10, 5, 3).astype(np.float32)
        X = np.random.randn(4, 5, 3).astype(np.float32)
        attr = computer._gradient_attribution(model, bg, X)
        assert attr.shape == (4, 5, 3)

    def test_gradient_attribution_non_zero(self) -> None:
        model = _make_dummy_model(timesteps=5, n_features=3)
        computer = SHAPComputer(timesteps=5, stride=1, n_integration_steps=10)
        bg = np.zeros((10, 5, 3), dtype=np.float32)
        X = np.ones((2, 5, 3), dtype=np.float32) * 2.0
        attr = computer._gradient_attribution(model, bg, X)
        assert np.any(np.abs(attr) > 1e-10)

    def test_compute_fallback_to_ig(self) -> None:
        """When shap is not importable, should fall back to IG."""
        model = _make_dummy_model(timesteps=5, n_features=3)
        computer = SHAPComputer(timesteps=5, stride=1, n_integration_steps=5)
        bg = np.random.randn(10, 5, 3).astype(np.float32)
        X = np.random.randn(2, 5, 3).astype(np.float32)
        # compute() tries GradientExplainer first, falls back on error
        result = computer.compute(model, bg, X)
        assert result.shape == (2, 5, 3)

    def test_get_config(self) -> None:
        computer = SHAPComputer(
            n_background=50,
            timesteps=10,
            stride=2,
            label_column="Target",
            n_integration_steps=25,
        )
        cfg = computer.get_config()
        assert cfg["n_background"] == 50
        assert cfg["timesteps"] == 10
        assert cfg["stride"] == 2
        assert cfg["label_column"] == "Target"
        assert cfg["n_integration_steps"] == 25

    def test_gradient_batch_processing(self) -> None:
        """Test that batched processing gives same shape for large input."""
        model = _make_dummy_model(timesteps=5, n_features=3)
        computer = SHAPComputer(timesteps=5, stride=1, n_integration_steps=3)
        bg = np.random.randn(5, 5, 3).astype(np.float32)
        X = np.random.randn(50, 5, 3).astype(np.float32)
        attr = computer._gradient_attribution(model, bg, X)
        assert attr.shape == (50, 5, 3)

    def test_prepare_background(self, tmp_path: Path) -> None:
        """Test background preparation from parquet file."""
        n_rows = 200
        rng = np.random.default_rng(42)
        data = {f"f{i}": rng.standard_normal(n_rows).astype(np.float32) for i in range(5)}
        data["Label"] = np.zeros(n_rows, dtype=np.int32)
        df = pd.DataFrame(data)
        parquet_path = tmp_path / "train.parquet"
        df.to_parquet(parquet_path)

        computer = SHAPComputer(n_background=10, timesteps=5, stride=1)
        bg = computer.prepare_background(parquet_path, rng)
        assert bg.shape[0] == 10
        assert bg.shape[1] == 5
        assert bg.shape[2] == 5

    def test_prepare_explanation_data(self, tmp_path: Path) -> None:
        """Test explanation data preparation from parquet file."""
        n_rows = 100
        rng = np.random.default_rng(42)
        data = {f"f{i}": rng.standard_normal(n_rows).astype(np.float32) for i in range(5)}
        data["Label"] = np.zeros(n_rows, dtype=np.int32)
        df = pd.DataFrame(data)
        parquet_path = tmp_path / "test.parquet"
        df.to_parquet(parquet_path)

        computer = SHAPComputer(timesteps=5, stride=1)
        X_explain, X_all, feature_names = computer.prepare_explanation_data(parquet_path, [0, 1, 2])
        assert X_explain.shape[0] == 3
        assert X_explain.shape[1] == 5
        assert X_explain.shape[2] == 5
        assert len(feature_names) == 5
        assert "Label" not in feature_names

    def test_prepare_explanation_data_clamps_indices(self, tmp_path: Path) -> None:
        """Out-of-range indices are filtered out."""
        n_rows = 30
        rng = np.random.default_rng(42)
        data = {f"f{i}": rng.standard_normal(n_rows).astype(np.float32) for i in range(3)}
        data["Label"] = np.zeros(n_rows, dtype=np.int32)
        df = pd.DataFrame(data)
        parquet_path = tmp_path / "test.parquet"
        df.to_parquet(parquet_path)

        computer = SHAPComputer(timesteps=5, stride=1)
        X_explain, _, _ = computer.prepare_explanation_data(parquet_path, [0, 1, 9999])
        assert X_explain.shape[0] == 2  # 9999 is out of range
