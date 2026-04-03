"""Tests for AutoClassificationHead — binary/multiclass detection."""

from __future__ import annotations

import tensorflow as tf

from src.phase3_classification_engine.phase3.base import BaseClassificationHead
from src.phase3_classification_engine.phase3.head import AutoClassificationHead


class TestAutoClassificationHead:
    """Validate classification head build and configuration."""

    def test_implements_base(self) -> None:
        assert issubclass(AutoClassificationHead, BaseClassificationHead)

    def test_binary_head_shape(self) -> None:
        head = AutoClassificationHead(dense_units=32, dropout_rate=0.2)
        inp = tf.keras.Input(shape=(128,))
        out = head.build(inp, n_classes=2)
        model = tf.keras.Model(inp, out)
        assert model.output_shape == (None, 1)

    def test_binary_head_activation(self) -> None:
        head = AutoClassificationHead(dense_units=32)
        inp = tf.keras.Input(shape=(128,))
        out = head.build(inp, n_classes=2)
        model = tf.keras.Model(inp, out)
        # Last layer should use sigmoid
        last_layer = model.layers[-1]
        cfg = last_layer.get_config()
        assert cfg["activation"] == "sigmoid"

    def test_multiclass_head_shape(self) -> None:
        head = AutoClassificationHead(dense_units=32)
        inp = tf.keras.Input(shape=(128,))
        out = head.build(inp, n_classes=4)
        model = tf.keras.Model(inp, out)
        assert model.output_shape == (None, 4)

    def test_multiclass_head_activation(self) -> None:
        head = AutoClassificationHead(dense_units=32)
        inp = tf.keras.Input(shape=(128,))
        out = head.build(inp, n_classes=4)
        model = tf.keras.Model(inp, out)
        last_layer = model.layers[-1]
        cfg = last_layer.get_config()
        assert cfg["activation"] == "softmax"

    def test_get_loss_binary(self) -> None:
        head = AutoClassificationHead()
        assert head.get_loss(2) == "binary_crossentropy"

    def test_get_loss_multiclass(self) -> None:
        head = AutoClassificationHead()
        assert head.get_loss(4) == "categorical_crossentropy"

    def test_get_config(self) -> None:
        head = AutoClassificationHead(dense_units=64, dropout_rate=0.3)
        cfg = head.get_config()
        assert cfg["dense_units"] == 64
        assert cfg["dropout_rate"] == 0.3
        assert "dense_activation" in cfg
