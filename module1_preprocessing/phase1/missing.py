"""Context-aware missing value handler — Single Responsibility.

Biometric features
------------------
Default strategy is **median imputation**. The previous default
(``ffill``) was unsafe for the WUSTL-EHMS-2020 layout because the CSV
concatenates capture sessions across patients with no session-grouping
column: a NaN at the start of patient B's session would get
back-filled with the last valid value from patient A, silently
cross-contaminating biometrics between unrelated patients. The
``ffill`` mode is still selectable, but only when an explicit
``session_column`` is provided (so ``ffill`` happens *within* a
patient's session, never across the boundary). Calling ``ffill``
without a session column raises ``ValueError`` rather than producing a
quietly poisoned dataset.

Network features
----------------
Default strategy is **dropna**. The previous default (``fill_zero``)
conflated "missing" with "zero traffic," which an attacker can exploit
by inducing capture loss in their attack flows: zero-flow looks like
benign quiescence, which the model learns to ignore. ``fill_zero``
remains selectable for operators who explicitly accept that risk.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .base import BaseTransformer

logger = logging.getLogger(__name__)


_BIO_STRATEGIES = frozenset({"median", "ffill"})
_NET_STRATEGIES = frozenset({"dropna", "fill_zero"})


class MissingValueHandler(BaseTransformer):
    """Handle missing values with domain-appropriate strategies.

    Args:
        biometric_columns: Column names for biometric sensor features.
        label_column: Label column name (excluded from network set).
        biometric_strategy: Imputation strategy for biometrics; one of
            ``{"median", "ffill"}``. Default is ``"median"``. ``ffill``
            requires ``session_column``.
        network_strategy: Handling strategy for network features; one
            of ``{"dropna", "fill_zero"}``. Default is ``"dropna"``.
        session_column: Optional column name that groups rows belonging
            to the same patient/capture-session. Required when
            ``biometric_strategy="ffill"``. Used to ``groupby`` so
            forward-fill never crosses a session boundary.
    """

    def __init__(
        self,
        biometric_columns: List[str],
        label_column: str = "Label",
        multi_label_column: str = "Attack Category",
        biometric_strategy: str = "median",
        network_strategy: str = "dropna",
        session_column: str | None = None,
    ) -> None:
        if biometric_strategy not in _BIO_STRATEGIES:
            raise ValueError(
                f"biometric_strategy must be one of {sorted(_BIO_STRATEGIES)}, "
                f"got {biometric_strategy!r}"
            )
        if network_strategy not in _NET_STRATEGIES:
            raise ValueError(
                f"network_strategy must be one of {sorted(_NET_STRATEGIES)}, "
                f"got {network_strategy!r}"
            )
        if biometric_strategy == "ffill" and not session_column:
            raise ValueError(
                "biometric_strategy='ffill' requires a session_column so "
                "forward-fill cannot cross patient/session boundaries. "
                "Either provide session_column or switch to "
                "biometric_strategy='median'."
            )
        if network_strategy == "fill_zero":
            logger.warning(
                "MissingValueHandler: network_strategy='fill_zero' "
                "conflates missing-flow with zero-flow, which an "
                "attacker can use to mask attack traffic by inducing "
                "capture loss. Prefer 'dropna' unless you have a "
                "documented reason."
            )

        self._bio_cols = biometric_columns
        self._label_col = label_column
        self._multi_label_col = multi_label_column
        self._bio_strategy = biometric_strategy
        self._net_strategy = network_strategy
        self._session_col = session_column
        self._stats: Dict[str, int] = {}

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply context-aware missing value handling.

        Args:
            df: HIPAA-sanitized DataFrame.

        Returns:
            DataFrame with missing values handled.
        """
        bio_cols = [c for c in self._bio_cols if c in df.columns]
        exclude = set(bio_cols) | {self._label_col, self._multi_label_col}
        if self._session_col:
            exclude.add(self._session_col)
        net_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]

        # ── Biometric imputation ──
        bio_filled = 0
        if bio_cols:
            bio_filled = int(df[bio_cols].isna().sum().sum())
            if self._bio_strategy == "median":
                # Per-column median over all non-NaN values. Median is
                # robust to outliers (consistent with the RobustScaler
                # choice in §4.1.6) and patient-safe by construction:
                # the imputed value never depends on a different
                # patient's reading at a temporal boundary.
                medians = df[bio_cols].median(skipna=True)
                df[bio_cols] = df[bio_cols].fillna(medians)
            elif self._bio_strategy == "ffill":
                # Per-session forward+backward fill. The session_column
                # constructor check guarantees this branch never runs
                # without a grouping column, so cross-patient leakage
                # is structurally impossible.
                grouped = df.groupby(self._session_col, sort=False)[bio_cols]
                df[bio_cols] = grouped.transform(lambda s: s.ffill().bfill())

        # ── Network handling ──
        rows_before = len(df)
        net_missing = int(df[net_cols].isna().sum().sum()) if net_cols else 0
        net_filled = 0
        if net_cols and self._net_strategy == "fill_zero":
            net_filled = net_missing
            df[net_cols] = df[net_cols].fillna(0)
        elif net_cols and self._net_strategy == "dropna":
            df = df.dropna(subset=net_cols)
        rows_dropped = rows_before - len(df)

        self._stats = {
            "biometric_strategy": self._bio_strategy,
            "biometric_cells_filled": bio_filled,
            "network_strategy": self._net_strategy,
            "network_cells_missing": net_missing,
            "network_cells_filled_zero": net_filled,
            "rows_dropped": rows_dropped,
            "rows_remaining": len(df),
        }
        logger.info(
            "MissingValueHandler: bio=%s (%d cells filled), net=%s "
            "(%d cells zeroed, %d rows dropped)",
            self._bio_strategy,
            bio_filled,
            self._net_strategy,
            net_filled,
            rows_dropped,
        )
        return df

    def get_report(self) -> Dict[str, Any]:
        return dict(self._stats)
