"""Variance filter — drops zero/near-zero variance features.

Applied after redundancy elimination, before train/test split.
Features with unique_count <= 1 (constant) are always dropped.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import pandas as pd

from .base import BaseTransformer

logger = logging.getLogger(__name__)


class VarianceFilter(BaseTransformer):
    """Drop features with zero or near-zero variance.

    Args:
        max_unique: Features with unique_count <= this are dropped.
        label_column: Label column name (never dropped).
    """

    def __init__(
        self,
        max_unique: int = 1,
        label_column: str = "Label",
    ) -> None:
        self._max_unique = max_unique
        self._label_col = label_column
        self._dropped: List[str] = []

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop features with too few unique values.

        Args:
            df: DataFrame after redundancy elimination.

        Returns:
            DataFrame with zero-variance features removed.
        """
        exclude = {self._label_col, "Attack Category"}
        numeric_cols = [
            c for c in df.select_dtypes(include=["number"]).columns
            if c not in exclude
        ]

        cols_to_drop = [
            c for c in numeric_cols
            if df[c].nunique() <= self._max_unique
        ]

        df = df.drop(columns=cols_to_drop, errors="ignore")
        self._dropped = cols_to_drop
        logger.info(
            "VarianceFilter: dropped %d features (unique ≤ %d): %s",
            len(cols_to_drop), self._max_unique, cols_to_drop,
        )
        return df

    def get_report(self) -> Dict[str, Any]:
        return {
            "max_unique": self._max_unique,
            "columns_dropped": self._dropped,
            "n_dropped": len(self._dropped),
        }
