"""Stratified train/test splitter — Single Responsibility.

Produces a 70/30 split preserving class balance via
``StratifiedShuffleSplit``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

logger = logging.getLogger(__name__)


class DataSplitter:
    """Stratified train/test split preserving class balance.

    Args:
        test_ratio: Fraction of samples for the test partition.
        random_state: Seed for reproducibility.
        label_column: Name of the binary label column.
    """

    def __init__(
        self,
        test_ratio: float = 0.30,
        random_state: int = 42,
        label_column: str = "Label",
    ) -> None:
        self._test_ratio = test_ratio
        self._random_state = random_state
        self._label_col = label_column
        self._stats: Dict[str, Any] = {}

    def split(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Split the DataFrame into stratified train/test partitions.

        Args:
            df: Preprocessed DataFrame with numeric features + label.

        Returns:
            Tuple of (X_train, X_test, y_train, y_test, feature_names).

        Raises:
            ValueError: If the label column is not found.
        """
        if self._label_col not in df.columns:
            raise ValueError(f"Label column '{self._label_col}' not found.")

        y = df[self._label_col].values
        X_df = df.drop(columns=[self._label_col]).select_dtypes(include=[np.number])
        feature_names = X_df.columns.tolist()
        X = X_df.values

        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=self._test_ratio,
            random_state=self._random_state,
        )
        train_idx, test_idx = next(sss.split(X, y))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        self._stats = {
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "train_ratio": round(1 - self._test_ratio, 2),
            "test_ratio": self._test_ratio,
            "stratified": True,
            "train_attack_rate": round(float(y_train.mean()), 4),
            "test_attack_rate": round(float(y_test.mean()), 4),
        }
        logger.info(
            "DataSplitter: train=%d (attack=%.1f%%) | test=%d (attack=%.1f%%)",
            len(X_train), y_train.mean() * 100,
            len(X_test), y_test.mean() * 100,
        )
        return X_train, X_test, y_train, y_test, feature_names

    def get_report(self) -> Dict[str, Any]:
        return dict(self._stats)
