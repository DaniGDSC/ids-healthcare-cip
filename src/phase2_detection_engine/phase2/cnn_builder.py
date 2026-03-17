"""CNN feature extractor builder — first stage.

Data flow::

    Input: (batch, 20, 29)
    Conv1D(64, k=3, relu, same) → (batch, 20, 64)
    MaxPool(2) → (batch, 10, 64)
    Conv1D(128, k=3, relu, same) → (batch, 10, 128)
    MaxPool(2) → (batch, 5, 128)
"""

from __future__ import annotations

from typing import Any, Dict

import tensorflow as tf

from .base import BaseLayerBuilder

CNN_PADDING: str = "same"


class CNNBuilder(BaseLayerBuilder):
    """Build CNN block via Keras functional API.

    Args:
        filters_1: Filters in the first Conv1D layer.
        filters_2: Filters in the second Conv1D layer.
        kernel_size: Kernel size for both Conv1D layers.
        activation: Activation function for Conv1D layers.
        pool_size: Pool size for MaxPooling1D layers.
    """

    def __init__(
        self,
        filters_1: int = 64,
        filters_2: int = 128,
        kernel_size: int = 3,
        activation: str = "relu",
        pool_size: int = 2,
    ) -> None:
        self._filters_1 = filters_1
        self._filters_2 = filters_2
        self._kernel_size = kernel_size
        self._activation = activation
        self._pool_size = pool_size

    def build(self, input_tensor: tf.Tensor) -> tf.Tensor:
        """Apply CNN block to input tensor.

        Args:
            input_tensor: Shape (batch, timesteps, features).

        Returns:
            Tensor of shape (batch, timesteps // pool^2, filters_2).
        """
        x = tf.keras.layers.Conv1D(
            self._filters_1, self._kernel_size,
            activation=self._activation,
            padding=CNN_PADDING, name="conv1",
        )(input_tensor)
        x = tf.keras.layers.MaxPooling1D(self._pool_size, name="pool1")(x)

        x = tf.keras.layers.Conv1D(
            self._filters_2, self._kernel_size,
            activation=self._activation,
            padding=CNN_PADDING, name="conv2",
        )(x)
        x = tf.keras.layers.MaxPooling1D(self._pool_size, name="pool2")(x)
        return x

    def get_config(self) -> Dict[str, Any]:
        """Return CNN hyperparameters for the report."""
        return {
            "filters_1": self._filters_1,
            "filters_2": self._filters_2,
            "kernel_size": self._kernel_size,
            "activation": self._activation,
            "pool_size": self._pool_size,
            "padding": CNN_PADDING,
        }
