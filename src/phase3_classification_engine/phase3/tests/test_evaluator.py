"""Tests for ModelEvaluator — classification metrics computation."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from src.phase3_classification_engine.phase3.evaluator import ModelEvaluator


def _make_binary_model() -> tf.keras.Model:
    """Model that always outputs 0.9 (predicts class 1)."""
    inp = tf.keras.Input(shape=(3,))
    # Use a Dense layer with fixed weights to always output ~0.9
    x = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(inp)
    model = tf.keras.Model(inp, x)
    # Set weights so output is high for any input
    model.get_layer("output").set_weights(
        [np.zeros((3, 1), dtype=np.float32), np.array([3.0], dtype=np.float32)]
    )
    return model


class TestModelEvaluator:
    """Validate metric computation on test set."""

    def test_metrics_keys(self) -> None:
        model = _make_binary_model()
        X_test = np.ones((20, 3), dtype=np.float32)
        y_test = np.ones(20, dtype=np.int32)

        evaluator = ModelEvaluator(threshold=0.5)
        result = evaluator.evaluate(model, X_test, y_test)

        expected_keys = {
            "accuracy",
            "f1_score",
            "precision",
            "recall",
            "auc_roc",
            "confusion_matrix",
            "classification_report",
            "threshold",
            "test_samples",
        }
        assert expected_keys.issubset(result.keys())

    def test_perfect_binary_predictions(self) -> None:
        model = _make_binary_model()
        # All class 1 → model predicts high → perfect
        X_test = np.ones((20, 3), dtype=np.float32)
        y_test = np.ones(20, dtype=np.int32)

        evaluator = ModelEvaluator(threshold=0.5)
        result = evaluator.evaluate(model, X_test, y_test)
        assert result["accuracy"] == 1.0

    def test_confusion_matrix_shape(self) -> None:
        model = _make_binary_model()
        X_test = np.ones((20, 3), dtype=np.float32)
        y_test = np.array([0] * 10 + [1] * 10, dtype=np.int32)

        evaluator = ModelEvaluator(threshold=0.5)
        result = evaluator.evaluate(model, X_test, y_test)
        cm = result["confusion_matrix"]
        assert len(cm) == 2
        assert len(cm[0]) == 2

    def test_test_samples_count(self) -> None:
        model = _make_binary_model()
        X_test = np.ones((30, 3), dtype=np.float32)
        y_test = np.ones(30, dtype=np.int32)

        evaluator = ModelEvaluator(threshold=0.5)
        result = evaluator.evaluate(model, X_test, y_test)
        assert result["test_samples"] == 30
