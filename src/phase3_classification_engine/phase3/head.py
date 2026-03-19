"""Auto classification head — Dense→Dropout→Dense.

Detects binary vs multiclass from ``n_classes`` and selects the
appropriate activation function and loss.
"""

from __future__ import annotations

from typing import Any, Dict

import tensorflow as tf

from .base import BaseClassificationHead


class AutoClassificationHead(BaseClassificationHead):
    """Dense→Dropout→Dense head with automatic binary/multiclass detection.

    Args:
        dense_units: Hidden units in the intermediate Dense layer.
        dense_activation: Activation for the intermediate Dense layer.
        dropout_rate: Dropout rate between Dense layers.
    """

    def __init__(
        self,
        dense_units: int = 64,
        dense_activation: str = "relu",
        dropout_rate: float = 0.3,
    ) -> None:
        self._dense_units = dense_units
        self._dense_activation = dense_activation
        self._dropout_rate = dropout_rate

    def build(self, input_tensor: tf.Tensor, n_classes: int) -> tf.Tensor:
        """Attach classification layers to *input_tensor*.

        Args:
            input_tensor: Output tensor from the detection backbone (batch, 128).
            n_classes: Number of target classes.

        Returns:
            Output Keras tensor with shape (batch, 1) for binary
            or (batch, n_classes) for multiclass.
        """
        output_units, activation = self._head_params(n_classes)

        x = tf.keras.layers.Dense(
            self._dense_units,
            activation=self._dense_activation,
            name="dense_head",
        )(input_tensor)
        x = tf.keras.layers.Dropout(
            self._dropout_rate,
            name="drop_head",
        )(x)
        x = tf.keras.layers.Dense(
            output_units,
            activation=activation,
            name="output",
        )(x)
        return x

    def get_loss(self, n_classes: int) -> str:
        """Return the appropriate loss function for *n_classes*.

        Args:
            n_classes: Number of target classes.

        Returns:
            ``"binary_crossentropy"`` for binary, else
            ``"categorical_crossentropy"``.
        """
        return "binary_crossentropy" if n_classes == 2 else "categorical_crossentropy"

    def get_config(self) -> Dict[str, Any]:
        """Return head hyperparameters for the pipeline report."""
        return {
            "dense_units": self._dense_units,
            "dense_activation": self._dense_activation,
            "dropout_rate": self._dropout_rate,
        }

    # ── private ───────────────────────────────────────────────────

    @staticmethod
    def _head_params(n_classes: int) -> tuple:
        """Return (output_units, activation) for the given class count."""
        if n_classes == 2:
            return 1, "sigmoid"
        return n_classes, "softmax"
