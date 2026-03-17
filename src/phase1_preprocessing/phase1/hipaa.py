"""HIPAA Safe Harbor sanitizer — Single Responsibility.

Drops network identifier / PHI columns from the DataFrame.
Delegates to the existing ``HIPAACompliance`` class via composition.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import pandas as pd

from .base import BaseTransformer

logger = logging.getLogger(__name__)


class HIPAASanitizer(BaseTransformer):
    """Drop HIPAA-sensitive identifier columns.

    Args:
        columns: Column names to remove (IP, MAC, port, direction, flags).
    """

    def __init__(self, columns: List[str]) -> None:
        self._columns = columns
        self._dropped: List[str] = []

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop PHI columns from the DataFrame.

        Args:
            df: Raw input DataFrame.

        Returns:
            DataFrame with identifier columns removed.
        """
        present = [c for c in self._columns if c in df.columns]
        df = df.drop(columns=present)
        self._dropped = present
        logger.info(
            "HIPAASanitizer: dropped %d columns: %s",
            len(present), present,
        )
        return df

    def get_report(self) -> Dict[str, Any]:
        return {
            "columns_requested": self._columns,
            "columns_dropped": self._dropped,
            "n_dropped": len(self._dropped),
        }
