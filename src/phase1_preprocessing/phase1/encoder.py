"""Categorical encoder — label-encodes categorical columns and parses
string-typed numeric columns.

Applied after HIPAA sanitization, before missing value handling.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .base import BaseTransformer

logger = logging.getLogger(__name__)


class CategoricalEncoder(BaseTransformer):
    """Encode remaining categorical columns to numeric.

    - Columns listed in ``label_encode`` are integer label-encoded.
    - Columns listed in ``parse_numeric`` are coerced to numeric via
      ``pd.to_numeric``; non-parseable values are replaced with
      ``sentinel`` (default -1).

    Args:
        label_encode: Column names to label-encode.
        parse_numeric: Column names to coerce from string to numeric.
        sentinel: Value for non-parseable strings (default -1).
    """

    def __init__(
        self,
        label_encode: List[str] | None = None,
        parse_numeric: List[str] | None = None,
        sentinel: int = -1,
    ) -> None:
        self._label_encode = label_encode or []
        self._parse_numeric = parse_numeric or []
        self._sentinel = sentinel
        self._encoders: Dict[str, LabelEncoder] = {}
        self._report_data: Dict[str, Any] = {}

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        label_encoded: Dict[str, int] = {}
        parsed: Dict[str, int] = {}

        # Label-encode categorical columns
        for col in self._label_encode:
            if col not in df.columns:
                continue
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self._encoders[col] = le
            label_encoded[col] = len(le.classes_)
            logger.info(
                "CategoricalEncoder: label-encoded '%s' (%d classes: %s)",
                col, len(le.classes_), list(le.classes_),
            )

        # Parse string columns to numeric with sentinel for failures
        for col in self._parse_numeric:
            if col not in df.columns:
                continue
            n_before = df[col].notna().sum()
            df[col] = pd.to_numeric(df[col], errors="coerce")
            n_coerced = n_before - df[col].notna().sum()
            df[col] = df[col].fillna(self._sentinel)
            parsed[col] = int(n_coerced)
            logger.info(
                "CategoricalEncoder: parsed '%s' to numeric "
                "(%d non-parseable → sentinel=%d)",
                col, n_coerced, self._sentinel,
            )

        self._report_data = {
            "label_encoded": label_encoded,
            "parsed_numeric": parsed,
            "sentinel": self._sentinel,
        }
        return df

    def get_report(self) -> Dict[str, Any]:
        return dict(self._report_data)
