"""Tests for ClassificationTrainer — training phase execution."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from src.phase3_classification_engine.phase3.config import TrainingPhaseConfig
from src.phase3_classification_engine.phase3.trainer import ClassificationTrainer
from src.phase3_classification_engine.phase3.unfreezer import ProgressiveUnfreezer


@pytest.fixture()
def simple_model() -> tf.keras.Model:
    """Minimal model for training tests."""
    inp = tf.keras.Input(shape=(5,))
    x = tf.keras.layers.Dense(4, activation="relu", name="dense_head")(inp)
    x = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)
    return tf.keras.Model(inp, x)


@pytest.fixture()
def tiny_data() -> tuple:
    """Small random dataset for training tests."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, 5).astype(np.float32)
    y = np.array([0] * 80 + [1] * 20, dtype=np.int32)
    return X, y


class TestClassificationTrainer:
    """Validate training phase execution."""

    def test_train_phase_returns_history(
        self,
        simple_model: tf.keras.Model,
        tiny_data: tuple,
        tmp_path: Path,
    ) -> None:
        X, y = tiny_data
        trainer = ClassificationTrainer(
            batch_size=32,
            validation_split=0.2,
            early_stopping_patience=1,
            reduce_lr_patience=1,
            reduce_lr_factor=0.5,
        )
        phase_cfg = TrainingPhaseConfig(
            name="Test Phase",
            epochs=2,
            learning_rate=0.01,
            frozen=[],
        )
        result = trainer.train_phase(simple_model, phase_cfg, X, y, "binary_crossentropy", tmp_path)
        assert "phase" in result
        assert "epochs_run" in result
        assert "final_train_loss" in result
        assert "final_val_loss" in result
        assert result["epochs_run"] > 0

    def test_train_all_phases(
        self,
        simple_model: tf.keras.Model,
        tiny_data: tuple,
        tmp_path: Path,
    ) -> None:
        X, y = tiny_data
        trainer = ClassificationTrainer(
            batch_size=32,
            validation_split=0.2,
            early_stopping_patience=1,
            reduce_lr_patience=1,
            reduce_lr_factor=0.5,
        )
        phases = [
            TrainingPhaseConfig(name="Phase A", epochs=1, learning_rate=0.01, frozen=[]),
            TrainingPhaseConfig(name="Phase B", epochs=1, learning_rate=0.001, frozen=[]),
        ]
        unfreezer = ProgressiveUnfreezer()
        histories = trainer.train_all_phases(
            simple_model, phases, unfreezer, X, y, "binary_crossentropy", tmp_path
        )
        assert len(histories) == 2
        assert histories[0]["phase"] == "Phase A"
        assert histories[1]["phase"] == "Phase B"
