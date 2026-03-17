"""SMOTE balancer — applies oversampling to training set ONLY.

Applied BEFORE scaling so synthetic points are generated in the
original feature space, not in a normalised space where inter-feature
distances are distorted.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import numpy as np
from imblearn.over_sampling import SMOTE

logger = logging.getLogger(__name__)


class SMOTEBalancer:
    """Oversample minority class using SMOTE.

    Args:
        strategy: SMOTE sampling strategy (``"auto"`` balances 1:1).
        k_neighbors: Number of nearest neighbours for interpolation.
        random_state: Seed for reproducibility.
    """

    def __init__(
        self,
        strategy: str = "auto",
        k_neighbors: int = 5,
        random_state: int = 42,
    ) -> None:
        self._strategy = strategy
        self._k = k_neighbors
        self._random_state = random_state
        self._stats: Dict[str, Any] = {}

    def resample(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resample the training set to balance class distribution.

        Args:
            X_train: Training feature matrix.
            y_train: Training labels.

        Returns:
            Tuple of (X_resampled, y_resampled).
        """
        n_before = len(X_train)
        counts_before = dict(zip(*np.unique(y_train, return_counts=True)))

        smote = SMOTE(
            sampling_strategy=self._strategy,
            k_neighbors=self._k,
            random_state=self._random_state,
        )
        X_res, y_res = smote.fit_resample(X_train, y_train)

        n_after = len(X_res)
        counts_after = dict(zip(*np.unique(y_res, return_counts=True)))

        self._stats = {
            "samples_before": n_before,
            "samples_after": n_after,
            "synthetic_added": n_after - n_before,
            "attack_rate_before": round(float(y_train.mean()), 4),
            "attack_rate_after": round(float(y_res.mean()), 4),
            "class_counts_before": {int(k): int(v) for k, v in counts_before.items()},
            "class_counts_after": {int(k): int(v) for k, v in counts_after.items()},
            "k_neighbors": self._k,
        }
        logger.info(
            "SMOTEBalancer: %d → %d (+%d synthetic)",
            n_before, n_after, n_after - n_before,
        )
        return X_res, y_res

    def get_report(self) -> Dict[str, Any]:
        return dict(self._stats)
