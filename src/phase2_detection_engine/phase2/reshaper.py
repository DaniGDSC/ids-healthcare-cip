"""Sliding window reshaper for temporal detection.

Converts flat feature matrices into 3-D tensors suitable for
1-D convolution: ``(N, F) → (N_win, timesteps, F)``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class DataReshaper:
    """Create sliding windows from flat feature matrices.

    Args:
        timesteps: Window length (number of consecutive events).
        stride: Step between consecutive windows.
    """

    def __init__(self, timesteps: int, stride: int = 1) -> None:
        if timesteps < 2:
            raise ValueError(f"timesteps must be >= 2, got {timesteps}")
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        self._timesteps = timesteps
        self._stride = stride

    def reshape(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sliding windows: (N, F) → (N_win, timesteps, F).

        The label for each window is the label of the **last** sample
        in the window (standard IDS temporal windowing convention).

        Args:
            X: 2-D array of shape (n_samples, n_features).
            y: 1-D array of shape (n_samples,).

        Returns:
            Tuple of (X_windows, y_windows).

        Raises:
            ValueError: If n_samples < timesteps.
        """
        n_samples, n_features = X.shape
        if n_samples < self._timesteps:
            raise ValueError(
                f"Cannot create windows: n_samples={n_samples} "
                f"< timesteps={self._timesteps}"
            )

        n_windows = (n_samples - self._timesteps) // self._stride + 1

        starts = np.arange(0, n_windows * self._stride, self._stride)
        offsets = np.arange(self._timesteps)
        indices = starts[:, None] + offsets[None, :]

        X_windows = X[indices].astype(np.float32)
        y_windows = y[indices[:, -1]]

        logger.info(
            "Reshape: (%d, %d) → %s  [timesteps=%d, stride=%d]",
            n_samples, n_features, X_windows.shape,
            self._timesteps, self._stride,
        )
        return X_windows, y_windows

    def get_config(self) -> Dict[str, Any]:
        """Return reshape configuration for the report."""
        return {"timesteps": self._timesteps, "stride": self._stride}
