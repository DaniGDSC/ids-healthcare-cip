"""Redundancy remover — reads Phase 0 high_correlations.csv.

Does NOT recompute the correlation matrix.  For each high-correlation
pair, drops ``feature_b`` (the secondary feature).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import pandas as pd

from .base import BaseTransformer

logger = logging.getLogger(__name__)


class RedundancyRemover(BaseTransformer):
    """Drop redundant features identified by Phase 0 correlation analysis.

    Args:
        corr_df: Phase 0 high-correlation pairs DataFrame
                 (columns: feature_a, feature_b, correlation).
        threshold: Minimum |r| to consider a pair redundant.
        label_column: Label column name (never dropped).
    """

    def __init__(
        self,
        corr_df: pd.DataFrame,
        threshold: float = 0.95,
        label_column: str = "Label",
    ) -> None:
        self._corr_df = corr_df
        self._threshold = threshold
        self._label_col = label_column
        self._dropped: List[str] = []

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop one feature from each high-correlation pair.

        Args:
            df: DataFrame after missing value handling.

        Returns:
            DataFrame with redundant features removed.
        """
        high = self._corr_df[self._corr_df["correlation"].abs() >= self._threshold]
        cols_to_drop: List[str] = []

        for _, row in high.iterrows():
            candidate = row["feature_b"]
            if (
                candidate in df.columns
                and candidate != self._label_col
                and candidate not in cols_to_drop
            ):
                cols_to_drop.append(candidate)

        df = df.drop(columns=cols_to_drop, errors="ignore")
        self._dropped = cols_to_drop
        logger.info(
            "RedundancyRemover: dropped %d features (|r| ≥ %.2f): %s",
            len(cols_to_drop), self._threshold, cols_to_drop,
        )
        return df

    def get_report(self) -> Dict[str, Any]:
        return {
            "threshold": self._threshold,
            "columns_dropped": self._dropped,
            "n_dropped": len(self._dropped),
        }
