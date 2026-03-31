"""Learnable per-feature weighting layer.

Scales each input feature by a trainable weight before CNN processing.
Allows the model to learn which features matter most — biometric features
can receive higher weight than network features if clinically important.

Initializes to ones (no change from baseline) and adapts during training.
"""

from __future__ import annotations

import tensorflow as tf


class FeatureWeightLayer(tf.keras.layers.Layer):
    """Learnable per-feature scaling.

    Applied before CNN: input (batch, timesteps, features) is element-wise
    multiplied by a trainable weight vector of shape (features,).
    """

    def build(self, input_shape: tf.TensorShape) -> None:
        self.w = self.add_weight(
            name="feature_weights",
            shape=(input_shape[-1],),
            initializer="ones",
            trainable=True,
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return x * self.w

    def get_config(self):
        config = super().get_config()
        return config
