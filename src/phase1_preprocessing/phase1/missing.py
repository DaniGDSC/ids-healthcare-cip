"""Context-aware missing value handler — Single Responsibility.

Biometric features: forward-fill (sensor dropout assumption).
Network features:  row-wise dropna (corrupted packet assumption).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .base import BaseTransformer

logger = logging.getLogger(__name__)


class MissingValueHandler(BaseTransformer):
    """Handle missing values with domain-appropriate strategies.

    Args:
        biometric_columns: Column names for biometric sensor features.
        label_column: Label column name (excluded from network set).
        biometric_strategy: Imputation strategy for biometrics.
        network_strategy: Handling strategy for network features.
    """

    def __init__(
        self,
        biometric_columns: List[str],
        label_column: str = "Label",
        biometric_strategy: str = "ffill",
        network_strategy: str = "dropna",
    ) -> None:
        self._bio_cols = biometric_columns
        self._label_col = label_column
        self._bio_strategy = biometric_strategy
        self._net_strategy = network_strategy
        self._stats: Dict[str, int] = {}

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply context-aware missing value handling.

        Args:
            df: HIPAA-sanitized DataFrame.

        Returns:
            DataFrame with missing values handled.
        """
        bio_cols = [c for c in self._bio_cols if c in df.columns]
        exclude = set(bio_cols) | {self._label_col, "Attack Category"}
        net_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in exclude
        ]

        # Biometric: forward-fill + backward-fill
        bio_filled = 0
        if bio_cols:
            bio_filled = int(df[bio_cols].isna().sum().sum())
            if self._bio_strategy == "ffill":
                df[bio_cols] = df[bio_cols].ffill().bfill()

        # Network: drop rows with NaN
        rows_before = len(df)
        net_missing = int(df[net_cols].isna().sum().sum()) if net_cols else 0
        if net_cols and self._net_strategy == "dropna":
            df = df.dropna(subset=net_cols)
        rows_dropped = rows_before - len(df)

        self._stats = {
            "biometric_cells_filled": bio_filled,
            "network_cells_missing": net_missing,
            "rows_dropped": rows_dropped,
            "rows_remaining": len(df),
        }
        logger.info(
            "MissingValueHandler: %d bio cells filled, %d rows dropped",
            bio_filled, rows_dropped,
        )
        return df

    def get_report(self) -> Dict[str, Any]:
        return dict(self._stats)
