"""Base layer builder interface for Phase 2 detection engine.

All model component builders implement ``BaseLayerBuilder``, satisfying
the Liskov Substitution Principle — any builder can be swapped in the
assembler without changing the orchestrator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import tensorflow as tf


class BaseLayerBuilder(ABC):
    """Abstract base for all detection model component builders.

    Subclasses must implement ``build()`` and ``get_config()``.
    The assembler chains builders via the Keras functional API.
    """

    @abstractmethod
    def build(self, input_tensor: tf.Tensor) -> tf.Tensor:
        """Apply this component's layers to the input tensor.

        Args:
            input_tensor: Keras tensor from the previous stage.

        Returns:
            Output Keras tensor.
        """

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return component hyperparameters for the pipeline report.

        Returns:
            Dict with component-specific configuration.
        """
