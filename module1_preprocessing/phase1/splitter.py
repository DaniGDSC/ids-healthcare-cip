"""Stratified train/val/test splitter — Single Responsibility.

Produces a stratified train/test split (default 70/30) preserving class
balance via ``StratifiedShuffleSplit``. When ``val_ratio > 0``, also
carves a held-out validation slice off the training side for use by
the cascaded DAE in Module 2 (closes GAP-L1-2 / GAP-L1-1: replaces
OOF probas with validation-set probas to eliminate train-inference
skew on the joint feature-prediction space).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

logger = logging.getLogger(__name__)


@dataclass
class SplitOutput:
    """Container for a 3-way stratified split. ``val_*`` arrays are
    empty when ``val_ratio == 0`` (backward-compatible 2-way mode)."""

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    y_multi_train: np.ndarray
    y_multi_val: np.ndarray
    y_multi_test: np.ndarray


class DataSplitter:
    """Stratified train/val/test split preserving class balance.

    Args:
        test_ratio: Fraction of total samples for the test partition.
        val_ratio: Fraction of the *training* partition (post-test-split)
            held out as a validation set. ``0.0`` (default) preserves the
            legacy 2-way behaviour. ``0.20`` produces a ~14% global
            validation split (since 0.20 × 0.70 ≈ 0.14).
        random_state: Seed for reproducibility.
        label_column: Name of the binary label column.
        multi_label_column: Name of the multi-class label column.
    """

    def __init__(
        self,
        test_ratio: float = 0.30,
        val_ratio: float = 0.0,
        random_state: int = 42,
        label_column: str = "Label",
        multi_label_column: str = "Attack Category",
    ) -> None:
        self._test_ratio = test_ratio
        self._val_ratio = val_ratio
        self._random_state = random_state
        self._label_col = label_column
        self._multi_label_col = multi_label_column
        self._stats: Dict[str, Any] = {}

    def split(self, df: pd.DataFrame) -> SplitOutput:
        """Split the DataFrame into stratified train/val/test partitions.

        Returns:
            ``SplitOutput`` dataclass with X_*, y_*, y_multi_* arrays.
            When ``val_ratio == 0`` the val arrays are empty.

        Raises:
            ValueError: If the label column is not found.
        """
        if self._label_col not in df.columns:
            raise ValueError(f"Label column '{self._label_col}' not found.")

        y = df[self._label_col].values

        # Extract multi-class labels if present
        has_multi = self._multi_label_col in df.columns
        if has_multi:
            y_multi = df[self._multi_label_col].values

        drop_cols = [self._label_col]
        if has_multi:
            drop_cols.append(self._multi_label_col)
        X_df = df.drop(columns=drop_cols).select_dtypes(include=[np.number])
        feature_names = X_df.columns.tolist()
        X = X_df.values

        # Stratify on y_multi (Attack Category) if available for finer balance
        stratify_on = y_multi if has_multi else y

        # ── Step 1: train+val vs test ──
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=self._test_ratio,
            random_state=self._random_state,
        )
        trainval_idx, test_idx = next(sss.split(X, stratify_on))

        X_test, y_test = X[test_idx], y[test_idx]
        y_multi_test = y_multi[test_idx] if has_multi else np.array([], dtype=object)

        # ── Step 2 (optional): split train+val into train and val ──
        empty = np.array([], dtype=object)
        if self._val_ratio > 0.0:
            stratify_inner = (y_multi[trainval_idx]
                              if has_multi else y[trainval_idx])
            sss_inner = StratifiedShuffleSplit(
                n_splits=1,
                test_size=self._val_ratio,
                random_state=self._random_state,
            )
            inner_train_idx, inner_val_idx = next(
                sss_inner.split(X[trainval_idx], stratify_inner)
            )
            train_idx = trainval_idx[inner_train_idx]
            val_idx = trainval_idx[inner_val_idx]
            X_val = X[val_idx]
            y_val = y[val_idx]
            y_multi_val = y_multi[val_idx] if has_multi else empty
        else:
            train_idx = trainval_idx
            val_idx = np.array([], dtype=np.int64)
            X_val = np.empty((0, X.shape[1]), dtype=X.dtype)
            y_val = np.array([], dtype=y.dtype)
            y_multi_val = empty

        X_train = X[train_idx]
        y_train = y[train_idx]
        y_multi_train = y_multi[train_idx] if has_multi else empty

        self._stats = {
            "train_samples": int(len(X_train)),
            "val_samples": int(len(X_val)),
            "test_samples": int(len(X_test)),
            "train_ratio_global": round(len(X_train) / len(X), 4),
            "val_ratio_global": round(len(X_val) / len(X), 4) if len(X_val) else 0.0,
            "test_ratio_global": round(len(X_test) / len(X), 4),
            "val_ratio_within_trainval": self._val_ratio,
            "stratified": True,
            "train_attack_rate": round(float(y_train.mean()), 4) if len(y_train) else 0.0,
            "val_attack_rate": round(float(y_val.mean()), 4) if len(y_val) else 0.0,
            "test_attack_rate": round(float(y_test.mean()), 4) if len(y_test) else 0.0,
        }
        if len(X_val) > 0:
            logger.info(
                "DataSplitter: train=%d (atk=%.1f%%) | val=%d (atk=%.1f%%) | test=%d (atk=%.1f%%)",
                len(X_train), y_train.mean() * 100,
                len(X_val), y_val.mean() * 100,
                len(X_test), y_test.mean() * 100,
            )
        else:
            logger.info(
                "DataSplitter: train=%d (atk=%.1f%%) | test=%d (atk=%.1f%%)",
                len(X_train), y_train.mean() * 100,
                len(X_test), y_test.mean() * 100,
            )
        return SplitOutput(
            X_train=X_train, X_val=X_val, X_test=X_test,
            y_train=y_train, y_val=y_val, y_test=y_test,
            feature_names=feature_names,
            y_multi_train=y_multi_train,
            y_multi_val=y_multi_val,
            y_multi_test=y_multi_test,
        )

    def get_report(self) -> Dict[str, Any]:
        return dict(self._stats)
