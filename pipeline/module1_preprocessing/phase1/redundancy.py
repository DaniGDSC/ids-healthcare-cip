"""Redundancy remover — reads Phase 0 high_correlations.csv.

Does NOT recompute the correlation matrix.  For each high-correlation
pair, drops ``feature_b`` (the secondary feature).
Labels are already separated at this point.

Hardening (security review finding #14)
---------------------------------------
- The constructor validates that ``corr_df`` exposes the required
  schema (``feature_a``, ``feature_b``, ``correlation``) before any
  rows are read. An attacker who can write to
  ``high_correlations.csv`` cannot smuggle a different schema past
  this check.
- The transform refuses to drop any column whose name is in
  ``protected_columns`` (the label columns by default). Without this,
  an attacker could neutralise the model by listing ``Label`` as
  ``feature_b`` in the correlations CSV.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence

import pandas as pd

from .base import BaseTransformer

logger = logging.getLogger(__name__)

_REQUIRED_CORR_COLUMNS = frozenset({"feature_a", "feature_b", "correlation"})


class RedundancyRemover(BaseTransformer):
    """Drop redundant features identified by Phase 0 correlation analysis.

    Args:
        corr_df: Phase 0 high-correlation pairs DataFrame
                 (columns: feature_a, feature_b, correlation).
        threshold: Minimum |r| to consider a pair redundant.
        protected_columns: Column names that are never dropped, even
            if listed as ``feature_b`` in the correlations file.
            Defaults to ``("Label", "Attack Category")``.

    Raises:
        ValueError: If ``corr_df`` lacks the required schema.
    """

    def __init__(
        self,
        corr_df: pd.DataFrame,
        threshold: float = 0.95,
        protected_columns: Sequence[str] = ("Label", "Attack Category"),
    ) -> None:
        missing = _REQUIRED_CORR_COLUMNS - set(corr_df.columns)
        if missing:
            raise ValueError(
                f"Phase 0 correlations file is missing required columns "
                f"{sorted(missing)}; refusing to apply redundancy removal "
                f"to a corr_df with an unexpected schema."
            )
        self._corr_df = corr_df
        self._threshold = threshold
        self._protected = frozenset(protected_columns)
        self._dropped: List[str] = []
        self._refused: List[str] = []

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop one feature from each high-correlation pair.

        Args:
            df: Feature-only DataFrame (labels already separated).

        Returns:
            DataFrame with redundant features removed.
        """
        high = self._corr_df[self._corr_df["correlation"].abs() >= self._threshold]
        cols_to_drop: List[str] = []
        refused: List[str] = []

        for _, row in high.iterrows():
            candidate = row["feature_b"]
            if candidate in self._protected:
                # Refuse to drop a protected column even if the corr
                # file lists it. Log loudly so an operator notices any
                # tampered correlations CSV. See finding #14.
                refused.append(candidate)
                logger.error(
                    "RedundancyRemover: REFUSED to drop protected column "
                    "'%s' listed in Phase 0 correlations file. This "
                    "may indicate a tampered high_correlations.csv.",
                    candidate,
                )
                continue
            if candidate in df.columns and candidate not in cols_to_drop:
                cols_to_drop.append(candidate)

        df = df.drop(columns=cols_to_drop, errors="ignore")
        self._dropped = cols_to_drop
        self._refused = refused
        logger.info(
            "RedundancyRemover: dropped %d features (|r| ≥ %.2f): %s",
            len(cols_to_drop), self._threshold, cols_to_drop,
        )
        return df

    def get_report(self) -> Dict[str, Any]:
        return {
            "threshold":          self._threshold,
            "columns_dropped":    self._dropped,
            "columns_refused":    self._refused,
            "n_dropped":          len(self._dropped),
            "n_refused_protected": len(self._refused),
        }
