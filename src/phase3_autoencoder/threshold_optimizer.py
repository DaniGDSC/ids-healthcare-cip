"""Optimize anomaly detection threshold."""

import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class ThresholdOptimizer:
    """Optimize reconstruction error threshold for anomaly detection."""

    def __init__(self, method: str = 'percentile', percentile: float = 95.0, n_std: float = 3.0):
        self.method = method
        self.percentile = percentile
        self.n_std = n_std

    def compute_threshold(self, errors: np.ndarray) -> float:
        if self.method == 'percentile':
            threshold = np.percentile(errors, self.percentile)
        elif self.method == 'std':
            threshold = np.mean(errors) + self.n_std * np.std(errors)
        elif self.method == 'mad':
            median = np.median(errors)
            mad = np.median(np.abs(errors - median))
            threshold = median + self.n_std * mad
        else:
            raise ValueError(f"Unknown threshold method: {self.method}")

        logger.info(f"Threshold ({self.method}): {threshold:.6f}")
        return threshold