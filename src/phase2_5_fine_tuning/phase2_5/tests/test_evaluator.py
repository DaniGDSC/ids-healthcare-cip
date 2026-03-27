"""Tests for QuickEvaluator — fast train+eval cycle."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from src.phase2_5_fine_tuning.phase2_5.config import QuickTrainConfig
from src.phase2_5_fine_tuning.phase2_5.evaluator import QuickEvaluator


@pytest.fixture
def evaluator() -> QuickEvaluator:
    qt = QuickTrainConfig(
        epochs=1,
        validation_split=0.2,
        early_stopping_patience=1,
        dense_units=16,
        dense_activation="relu",
        head_dropout_rate=0.1,
    )
    return QuickEvaluator(qt, n_features=5, random_state=42)


@pytest.fixture
def dummy_data():
    """Small dummy dataset for quick tests."""
    rng = np.random.RandomState(42)
    X_train = rng.randn(100, 5).astype(np.float32)
    y_train = rng.randint(0, 2, size=100)
    X_test = rng.randn(30, 5).astype(np.float32)
    y_test = rng.randint(0, 2, size=30)
    return X_train, y_train, X_test, y_test


class TestQuickEvaluator:
    """Validate quick evaluator produces expected result structure."""

    def test_evaluate_config_returns_metrics(self, evaluator, dummy_data) -> None:
        X_train, y_train, X_test, y_test = dummy_data
        hp = {
            "cnn_filters_1": 8,
            "cnn_filters_2": 16,
            "cnn_kernel_size": 3,
            "bilstm_units_1": 8,
            "bilstm_units_2": 4,
            "dropout_rate": 0.1,
            "attention_units": 8,
            "timesteps": 5,
            "batch_size": 32,
            "learning_rate": 0.01,
        }

        result = evaluator.evaluate_config(hp, X_train, y_train, X_test, y_test)

        assert "metrics" in result
        assert "hyperparameters" in result
        assert "duration_seconds" in result
        metrics = result["metrics"]
        assert "f1_score" in metrics
        assert "auc_roc" in metrics
        assert "accuracy" in metrics
        assert 0.0 <= metrics["f1_score"] <= 1.0
        assert 0.0 <= metrics["auc_roc"] <= 1.0

    def test_evaluate_ablation_variant_returns_variant_name(
        self, evaluator, dummy_data
    ) -> None:
        X_train, y_train, X_test, y_test = dummy_data
        hp = {
            "cnn_filters_1": 8,
            "cnn_filters_2": 16,
            "cnn_kernel_size": 3,
            "bilstm_units_1": 8,
            "bilstm_units_2": 4,
            "dropout_rate": 0.1,
            "attention_units": 8,
            "timesteps": 5,
            "batch_size": 32,
            "learning_rate": 0.01,
        }

        result = evaluator.evaluate_ablation_variant(
            variant_name="no_attention",
            base_hp=hp,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            remove="attention",
        )

        assert result["variant"] == "no_attention"
        assert "metrics" in result
        assert result["modifications"]["remove"] == "attention"

    def test_build_variant_no_cnn2(self, evaluator) -> None:
        hp = {
            "cnn_filters_1": 8,
            "cnn_filters_2": 16,
            "cnn_kernel_size": 3,
            "bilstm_units_1": 8,
            "bilstm_units_2": 4,
            "dropout_rate": 0.1,
            "attention_units": 8,
        }
        model = evaluator._build_variant_model(hp, timesteps=5, remove="cnn2")
        layer_names = [l.name for l in model.layers]
        assert "conv1" in layer_names
        assert "conv2" not in layer_names

    def test_build_variant_no_bilstm2(self, evaluator) -> None:
        hp = {
            "cnn_filters_1": 8,
            "cnn_filters_2": 16,
            "cnn_kernel_size": 3,
            "bilstm_units_1": 8,
            "bilstm_units_2": 4,
            "dropout_rate": 0.1,
            "attention_units": 8,
        }
        model = evaluator._build_variant_model(hp, timesteps=5, remove="bilstm2")
        layer_names = [l.name for l in model.layers]
        assert "bilstm1" in layer_names
        assert "bilstm2" not in layer_names

    def test_build_variant_unidirectional(self, evaluator) -> None:
        hp = {
            "cnn_filters_1": 8,
            "cnn_filters_2": 16,
            "cnn_kernel_size": 3,
            "bilstm_units_1": 8,
            "bilstm_units_2": 4,
            "dropout_rate": 0.1,
            "attention_units": 8,
        }
        model = evaluator._build_variant_model(
            hp, timesteps=5, replace="bilstm_to_lstm"
        )
        layer_names = [l.name for l in model.layers]
        assert "lstm1" in layer_names
        assert "bilstm1" not in layer_names
