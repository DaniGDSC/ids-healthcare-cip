"""Detection model assembler — chains builders into a Keras Model.

Implements the Open/Closed Principle: new builders can be added
without modifying the assembler.  The model outputs a context vector
(weighted sum) — there is NO classification head.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import tensorflow as tf

from .base import BaseLayerBuilder

logger = logging.getLogger(__name__)


class DetectionModelAssembler:
    """Assemble detection model from BaseLayerBuilder components.

    Args:
        timesteps: Input window length.
        n_features: Number of input features per timestep.
        builders: Ordered list of BaseLayerBuilder instances.
        model_name: Name for the Keras model.
    """

    def __init__(
        self,
        timesteps: int,
        n_features: int,
        builders: List[BaseLayerBuilder],
        model_name: str = "detection_engine",
    ) -> None:
        self._timesteps = timesteps
        self._n_features = n_features
        self._builders = builders
        self._model_name = model_name

    def assemble(self) -> tf.keras.Model:
        """Build the Keras Model by chaining all builders.

        Returns:
            Keras Model with input shape (batch, timesteps, n_features)
            and output shape (batch, context_dim).

        Raises:
            ValueError: If no builders are provided.
        """
        if not self._builders:
            raise ValueError("No builders provided to assembler.")

        inp = tf.keras.Input(
            shape=(self._timesteps, self._n_features), name="input",
        )

        x = inp
        for builder in self._builders:
            x = builder.build(x)

        model = tf.keras.Model(inp, x, name=self._model_name)
        logger.info(
            "Model assembled: %s (params=%d)",
            self._model_name, model.count_params(),
        )
        return model

    def get_config(self) -> Dict[str, Any]:
        """Return assembler configuration for the report."""
        return {
            "timesteps": self._timesteps,
            "n_features": self._n_features,
            "model_name": self._model_name,
            "builders": [
                {"type": type(b).__name__, "config": b.get_config()}
                for b in self._builders
            ],
        }
