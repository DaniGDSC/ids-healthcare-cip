"""Abstract base class for classification heads.

Subclasses must implement ``build()`` and ``get_config()``.
New head types (e.g., multi-head attention classifier) can be
added without modifying existing code (Open/Closed Principle).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import tensorflow as tf


class BaseClassificationHead(ABC):
    """Interface for all classification heads.

    Subclasses attach classification layers on top of a detection
    backbone's output tensor.
    """

    @abstractmethod
    def build(self, input_tensor: tf.Tensor, n_classes: int) -> tf.Tensor:
        """Attach classification layers to *input_tensor*.

        Args:
            input_tensor: Output tensor from the detection backbone.
            n_classes: Number of target classes.

        Returns:
            Output Keras tensor with classification predictions.
        """

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return head hyperparameters for the pipeline report.

        Returns:
            Dict with component-specific configuration.
        """
