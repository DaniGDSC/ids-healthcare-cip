"""Tests for ProgressiveUnfreezer — layer freeze/unfreeze logic."""

from __future__ import annotations

import pytest
import tensorflow as tf

from src.phase3_classification_engine.phase3.unfreezer import ProgressiveUnfreezer


@pytest.fixture()
def detection_like_model() -> tf.keras.Model:
    """Build a small model with the same layer names as the detection model."""
    inp = tf.keras.Input(shape=(20, 5), name="input")

    x = tf.keras.layers.Conv1D(8, 3, padding="same", name="conv1")(inp)
    x = tf.keras.layers.MaxPooling1D(2, name="pool1")(x)
    x = tf.keras.layers.Conv1D(16, 3, padding="same", name="conv2")(x)
    x = tf.keras.layers.MaxPooling1D(2, name="pool2")(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(8, return_sequences=True), name="bilstm1"
    )(x)
    x = tf.keras.layers.Dropout(0.1, name="drop1")(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(4, return_sequences=True), name="bilstm2"
    )(x)
    x = tf.keras.layers.Dropout(0.1, name="drop2")(x)

    x = tf.keras.layers.GlobalAveragePooling1D(name="attention")(x)

    # Classification head
    x = tf.keras.layers.Dense(4, activation="relu", name="dense_head")(x)
    x = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)

    return tf.keras.Model(inp, x)


class TestProgressiveUnfreezer:
    """Validate freeze/unfreeze behaviour per phase."""

    def test_freeze_all_detection(self, detection_like_model: tf.keras.Model) -> None:
        unfreezer = ProgressiveUnfreezer()
        frozen_groups = ["cnn", "bilstm1", "bilstm2", "attention"]
        unfreezer.apply_phase(detection_like_model, frozen_groups)

        # Detection layers should be frozen
        for layer in detection_like_model.layers:
            if layer.name in (
                "conv1",
                "pool1",
                "conv2",
                "pool2",
                "bilstm1",
                "drop1",
                "bilstm2",
                "drop2",
                "attention",
            ):
                assert not layer.trainable, f"{layer.name} should be frozen"
            elif layer.name != "input":
                assert layer.trainable, f"{layer.name} should be trainable"

    def test_unfreeze_attention(self, detection_like_model: tf.keras.Model) -> None:
        unfreezer = ProgressiveUnfreezer()
        unfreezer.apply_phase(detection_like_model, ["cnn", "bilstm1", "bilstm2"])

        assert detection_like_model.get_layer("attention").trainable
        assert not detection_like_model.get_layer("conv1").trainable
        assert not detection_like_model.get_layer("bilstm1").trainable
        assert not detection_like_model.get_layer("bilstm2").trainable

    def test_unfreeze_bilstm2_attention(self, detection_like_model: tf.keras.Model) -> None:
        unfreezer = ProgressiveUnfreezer()
        unfreezer.apply_phase(detection_like_model, ["cnn", "bilstm1"])

        assert detection_like_model.get_layer("attention").trainable
        assert detection_like_model.get_layer("bilstm2").trainable
        assert detection_like_model.get_layer("drop2").trainable
        assert not detection_like_model.get_layer("conv1").trainable
        assert not detection_like_model.get_layer("bilstm1").trainable
        assert not detection_like_model.get_layer("drop1").trainable

    def test_returns_param_counts(self, detection_like_model: tf.keras.Model) -> None:
        unfreezer = ProgressiveUnfreezer()
        trainable, frozen = unfreezer.apply_phase(
            detection_like_model, ["cnn", "bilstm1", "bilstm2", "attention"]
        )
        assert trainable > 0  # head layers
        assert frozen > 0  # detection layers

    def test_get_config(self) -> None:
        unfreezer = ProgressiveUnfreezer()
        cfg = unfreezer.get_config()
        assert "cnn" in cfg
        assert "bilstm1" in cfg
        assert "bilstm2" in cfg
        assert "attention" in cfg
