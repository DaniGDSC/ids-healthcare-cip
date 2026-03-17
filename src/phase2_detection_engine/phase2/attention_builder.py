"""Bahdanau attention builder — third stage.

Contains the custom ``BahdanauAttention`` Keras layer (registered for
serialisation) and the ``AttentionBuilder`` wrapper.

Data flow::

    Input: (batch, 5, 128) — from BiLSTM
    Dense(128, tanh) → score: (batch, 5, 128)
    Dense(1, no_bias) → logits: (batch, 5, 1)
    softmax(axis=1) → weights: (batch, 5, 1)
    x * weights → weighted: (batch, 5, 128)
    GlobalAvgPool1D → context: (batch, 128)
"""

from __future__ import annotations

from typing import Any, Dict

import tensorflow as tf

from .base import BaseLayerBuilder


@tf.keras.utils.register_keras_serializable(package="phase2")
class BahdanauAttention(tf.keras.layers.Layer):
    """Additive (Bahdanau) attention over a temporal sequence.

    Uses ``tf.nn.softmax`` (NOT ``Dense(activation='softmax')``) for
    proper normalisation over the timestep axis.

    Input shape:  (batch, timesteps, features)
    Output shape: (batch, features)
    """

    def __init__(self, units: int = 128, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.units = units
        self.score_dense = tf.keras.layers.Dense(
            units, activation="tanh", name="score",
        )
        self.weight_dense = tf.keras.layers.Dense(
            1, use_bias=False, name="attn_weights",
        )
        self.pool = tf.keras.layers.GlobalAveragePooling1D(name="context_pool")

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass: compute attention-weighted context vector."""
        scores = self.score_dense(x)
        weights = self.weight_dense(scores)
        weights = tf.nn.softmax(weights, axis=1)
        weighted = x * weights
        return self.pool(weighted)

    def get_config(self) -> Dict[str, Any]:
        """Serialisation support for model saving."""
        return {**super().get_config(), "units": self.units}


class AttentionBuilder(BaseLayerBuilder):
    """Build BahdanauAttention block via Keras functional API.

    Args:
        units: Hidden units in the attention score layer.
    """

    def __init__(self, units: int = 128) -> None:
        self._units = units

    def build(self, input_tensor: tf.Tensor) -> tf.Tensor:
        """Apply attention to input tensor.

        Args:
            input_tensor: Shape (batch, timesteps, features).

        Returns:
            Tensor of shape (batch, features) — context vector.
        """
        return BahdanauAttention(
            units=self._units, name="attention",
        )(input_tensor)

    def get_config(self) -> Dict[str, Any]:
        """Return attention hyperparameters for the report."""
        return {"units": self._units}
