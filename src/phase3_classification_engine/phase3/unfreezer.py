"""Progressive unfreezer — manage freeze/unfreeze of detection backbone layers.

Layer groups are mapped to detection model layer names.  Training phases
are defined in ``config/phase3_config.yaml``, never hardcoded.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import tensorflow as tf

logger = logging.getLogger(__name__)

_LAYER_GROUPS: Dict[str, List[str]] = {
    "cnn": ["conv1", "pool1", "conv2", "pool2"],
    "bilstm1": ["bilstm1", "drop1"],
    "bilstm2": ["bilstm2", "drop2"],
    "attention": ["attention"],
}


class ProgressiveUnfreezer:
    """Manage freeze/unfreeze of detection model layer groups.

    Uses module-level ``_LAYER_GROUPS`` mapping to translate group
    names (from config) to concrete Keras layer names.
    """

    def apply_phase(self, model: tf.keras.Model, frozen_groups: List[str]) -> Tuple[int, int]:
        """Set trainable flags for a single unfreezing phase.

        Args:
            model: Full classification model (detection backbone + head).
            frozen_groups: List of group names to freeze.

        Returns:
            Tuple of (trainable_params, frozen_params).
        """
        frozen_names: set = set()
        for group in frozen_groups:
            frozen_names.update(_LAYER_GROUPS.get(group, []))

        for layer in model.layers:
            if layer.name in frozen_names:
                layer.trainable = False
            elif layer.name not in ("input",):
                layer.trainable = True

        trainable = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
        frozen = sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)
        logger.info("  Trainable: %d, Frozen: %d", trainable, frozen)
        return trainable, frozen

    def get_config(self) -> Dict[str, Any]:
        """Return layer group mapping for the pipeline report."""
        return dict(_LAYER_GROUPS)
