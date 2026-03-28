"""Tests for QuickEvaluator — two-stage train+eval cycle."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from src.phase2_5_fine_tuning.phase2_5.config import QuickTrainConfig
from src.phase2_5_fine_tuning.phase2_5.evaluator import QuickEvaluator


@pytest.fixture
def evaluator() -> QuickEvaluator:
    qt = QuickTrainConfig(
        epochs=1, validation_split=0.2, early_stopping_patience=1,
        dense_units=16, dense_activation="relu", head_dropout_rate=0.1,
    )
    return QuickEvaluator(
        qt, n_features=5, random_state=42,
        model_architecture={
            "cnn_filters_1": 8, "cnn_filters_2": 16,
            "bilstm_units_1": 8, "bilstm_units_2": 4,
            "attention_units": 8, "head_dense_units": 16,
            "dropout_rate": 0.1, "timesteps": 5,
        },
    )


@pytest.fixture
def dummy_data():
    rng = np.random.RandomState(42)
    X_train = rng.randn(100, 5).astype(np.float32)
    y_train = rng.randint(0, 2, size=100)
    X_test = rng.randn(30, 5).astype(np.float32)
    y_test = rng.randint(0, 2, size=30)
    return X_train, y_train, X_test, y_test


class TestQuickEvaluator:
    def test_evaluate_two_stage_returns_metrics(self, evaluator, dummy_data) -> None:
        Xtr, ytr, Xte, yte = dummy_data
        hp = {"head_lr": 0.01, "finetune_lr": 0.0001, "cw_attack": 2.0,
              "head_epochs": 1, "ft_epochs": 1}

        result = evaluator.evaluate_two_stage(hp, Xtr, ytr, Xtr, ytr, Xte, yte)

        assert "metrics" in result
        assert "optimal_threshold" in result
        m = result["metrics"]
        assert "attack_f1" in m
        assert "auc_roc" in m
        assert 0.0 <= m["attack_f1"] <= 1.0

    def test_evaluate_ablation_variant(self, evaluator, dummy_data) -> None:
        Xtr, ytr, Xte, yte = dummy_data

        result = evaluator.evaluate_ablation_variant(
            variant_name="no_attention",
            base_hp={},
            X_train_smote=Xtr,
            y_train_smote=ytr,
            X_test=Xte,
            y_test=yte,
            remove="attention",
        )

        assert result["variant"] == "no_attention"
        assert "metrics" in result
        assert result["modifications"]["remove"] == "attention"

    def test_build_variant_no_cnn2(self, evaluator) -> None:
        model = evaluator._build_variant_model({}, timesteps=5, remove="cnn2")
        layer_names = [l.name for l in model.layers]
        assert "conv1" in layer_names
        assert "conv2" not in layer_names

    def test_build_variant_no_bilstm2(self, evaluator) -> None:
        model = evaluator._build_variant_model({}, timesteps=5, remove="bilstm2")
        layer_names = [l.name for l in model.layers]
        assert "bilstm1" in layer_names
        assert "bilstm2" not in layer_names

    def test_build_variant_unidirectional(self, evaluator) -> None:
        model = evaluator._build_variant_model({}, timesteps=5, replace="bilstm_to_lstm")
        layer_names = [l.name for l in model.layers]
        assert "lstm1" in layer_names
        assert "bilstm1" not in layer_names
