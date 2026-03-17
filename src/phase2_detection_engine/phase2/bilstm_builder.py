"""BiLSTM temporal encoder builder — second stage.

Data flow::

    Input: (batch, 5, 128) — from CNN
    BiLSTM(128, return_seq=True) → (batch, 5, 256)
    Dropout(0.3)
    BiLSTM(64, return_seq=True) → (batch, 5, 128)
    Dropout(0.3)
"""

from __future__ import annotations

from typing import Any, Dict

import tensorflow as tf

from .base import BaseLayerBuilder


class BiLSTMBuilder(BaseLayerBuilder):
    """Build BiLSTM block via Keras functional API.

    Args:
        units_1: Units in the first BiLSTM layer (doubled by bidirectional).
        units_2: Units in the second BiLSTM layer (doubled by bidirectional).
        dropout_rate: Dropout rate after each BiLSTM layer.
    """

    def __init__(
        self,
        units_1: int = 128,
        units_2: int = 64,
        dropout_rate: float = 0.3,
    ) -> None:
        self._units_1 = units_1
        self._units_2 = units_2
        self._dropout_rate = dropout_rate

    def build(self, input_tensor: tf.Tensor) -> tf.Tensor:
        """Apply BiLSTM block to input tensor.

        Args:
            input_tensor: Shape (batch, timesteps, features).

        Returns:
            Tensor of shape (batch, timesteps, units_2 * 2).
        """
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(self._units_1, return_sequences=True),
            name="bilstm1",
        )(input_tensor)
        x = tf.keras.layers.Dropout(self._dropout_rate, name="drop1")(x)

        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(self._units_2, return_sequences=True),
            name="bilstm2",
        )(x)
        x = tf.keras.layers.Dropout(self._dropout_rate, name="drop2")(x)
        return x

    def get_config(self) -> Dict[str, Any]:
        """Return BiLSTM hyperparameters for the report."""
        return {
            "units_1": self._units_1,
            "units_2": self._units_2,
            "output_dim": self._units_2 * 2,
            "dropout_rate": self._dropout_rate,
        }
