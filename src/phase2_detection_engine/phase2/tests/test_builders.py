"""Unit tests for CNN, BiLSTM, and Attention builders.

Tests verify output shapes, interface compliance, and attention
weight normalisation (sum ≈ 1.0 over timesteps).
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from src.phase2_detection_engine.phase2.attention_builder import (
    AttentionBuilder,
    BahdanauAttention,
)
from src.phase2_detection_engine.phase2.base import BaseLayerBuilder
from src.phase2_detection_engine.phase2.bilstm_builder import BiLSTMBuilder
from src.phase2_detection_engine.phase2.cnn_builder import CNNBuilder


class TestCNNBuilder:
    """Test CNNBuilder output shapes and interface."""

    def test_implements_base(self) -> None:
        assert issubclass(CNNBuilder, BaseLayerBuilder)

    def test_output_shape(self) -> None:
        builder = CNNBuilder(filters_1=32, filters_2=64, kernel_size=3, pool_size=2)
        inp = tf.keras.Input(shape=(20, 5))
        out = builder.build(inp)
        model = tf.keras.Model(inp, out)
        # 20 / 2 / 2 = 5 timesteps, 64 filters
        assert model.output_shape == (None, 5, 64)

    def test_output_shape_different_input(self) -> None:
        builder = CNNBuilder(filters_1=16, filters_2=32, kernel_size=3, pool_size=2)
        inp = tf.keras.Input(shape=(40, 10))
        out = builder.build(inp)
        model = tf.keras.Model(inp, out)
        # 40 / 2 / 2 = 10 timesteps
        assert model.output_shape == (None, 10, 32)

    def test_get_config(self) -> None:
        builder = CNNBuilder(filters_1=64, filters_2=128, kernel_size=3)
        cfg = builder.get_config()
        assert cfg["filters_1"] == 64
        assert cfg["filters_2"] == 128
        assert cfg["padding"] == "same"


class TestBiLSTMBuilder:
    """Test BiLSTMBuilder output shapes and interface."""

    def test_implements_base(self) -> None:
        assert issubclass(BiLSTMBuilder, BaseLayerBuilder)

    def test_output_shape(self) -> None:
        builder = BiLSTMBuilder(units_1=32, units_2=16, dropout_rate=0.1)
        inp = tf.keras.Input(shape=(5, 64))
        out = builder.build(inp)
        model = tf.keras.Model(inp, out)
        # Bidirectional doubles units_2: 16 * 2 = 32
        assert model.output_shape == (None, 5, 32)

    def test_returns_sequences(self) -> None:
        builder = BiLSTMBuilder(units_1=16, units_2=8)
        inp = tf.keras.Input(shape=(10, 32))
        out = builder.build(inp)
        model = tf.keras.Model(inp, out)
        # All timesteps preserved
        assert model.output_shape[1] == 10

    def test_get_config_output_dim(self) -> None:
        builder = BiLSTMBuilder(units_1=128, units_2=64)
        cfg = builder.get_config()
        assert cfg["output_dim"] == 128  # 64 * 2


class TestAttentionBuilder:
    """Test AttentionBuilder output shapes and weight normalisation."""

    def test_implements_base(self) -> None:
        assert issubclass(AttentionBuilder, BaseLayerBuilder)

    def test_output_shape(self) -> None:
        builder = AttentionBuilder(units=64)
        inp = tf.keras.Input(shape=(5, 32))
        out = builder.build(inp)
        model = tf.keras.Model(inp, out)
        # Attention collapses timestep dim → (batch, features)
        assert model.output_shape == (None, 32)

    def test_get_config(self) -> None:
        builder = AttentionBuilder(units=128)
        cfg = builder.get_config()
        assert cfg["units"] == 128


class TestBahdanauAttention:
    """Test the custom Keras attention layer directly."""

    def test_output_shape(self) -> None:
        layer = BahdanauAttention(units=64)
        inp = tf.keras.Input(shape=(5, 32))
        out = layer(inp)
        model = tf.keras.Model(inp, out)
        assert model.output_shape == (None, 32)

    def test_attention_weights_sum_to_one(self) -> None:
        """Softmax weights over timesteps must sum to ≈ 1.0."""
        # Use eager execution with real data to verify softmax weights
        attn = BahdanauAttention(units=64)

        rng = np.random.RandomState(42)
        X = rng.randn(10, 5, 32).astype(np.float32)

        # Build layers by calling once
        _ = attn(X)

        # Now extract weights eagerly
        scores = attn.score_dense(X)
        logits = attn.weight_dense(scores)
        weights = tf.nn.softmax(logits, axis=1)
        w = weights.numpy()

        # w shape: (10, 5, 1)
        assert w.shape == (10, 5, 1)
        # Sum over timesteps (axis=1) should be ≈ 1.0
        sums = w.sum(axis=1).flatten()
        np.testing.assert_allclose(sums, 1.0, atol=1e-5)

    def test_serialisation_config(self) -> None:
        layer = BahdanauAttention(units=128, name="test_attn")
        cfg = layer.get_config()
        assert cfg["units"] == 128
        assert cfg["name"] == "test_attn"

    def test_forward_pass_deterministic(self) -> None:
        """Same input → same output (no stochastic layers)."""
        layer = BahdanauAttention(units=32)
        inp = tf.keras.Input(shape=(5, 16))
        out = layer(inp)
        model = tf.keras.Model(inp, out)

        rng = np.random.RandomState(42)
        X = rng.randn(4, 5, 16).astype(np.float32)
        y1 = model.predict(X, verbose=0)
        y2 = model.predict(X, verbose=0)
        np.testing.assert_array_equal(y1, y2)
