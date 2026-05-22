"""4-way stratified train / val / test / demo splitter — Strategy 1.

Produces 4 disjoint stratified partitions preserving class balance via
sequential ``StratifiedShuffleSplit`` calls on the multi-class label
(``Attack Category``). ARCHITECTURE.md Step [1] / "Strategy 1 — Frozen
Test + Demo Pool":

* ``train`` (60%) — Track A + Track B model fitting
* ``val``   (15%) — threshold calibration / DAE cascade input probas
* ``test``  (15%) — frozen, paper metrics only (M-metrics)
* ``demo``  (10%) — frozen, dashboard alerts + Phase 2 user study

The four ratios MUST sum to 1.0. The split is deterministic in
``random_state`` and reproducible byte-for-byte across runs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

logger = logging.getLogger(__name__)


@dataclass
class SplitOutput:
    """Container for the 4-way stratified split."""

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    X_demo: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    y_demo: np.ndarray
    feature_names: List[str]
    y_multi_train: np.ndarray
    y_multi_val: np.ndarray
    y_multi_test: np.ndarray
    y_multi_demo: np.ndarray


class DataSplitter:
    """Stratified 4-way split preserving class balance.

    Implements Strategy 1 (Frozen Test + Demo Pool) via three
    sequential ``StratifiedShuffleSplit`` calls. The split is
    deterministic in ``random_state``: the same seed always produces
    byte-identical splits.

    Args:
        train_ratio: Global fraction of samples for ``train`` (model fit).
        val_ratio:   Global fraction for ``val`` (calibration).
        test_ratio:  Global fraction for ``test`` (frozen, paper metrics).
        demo_ratio:  Global fraction for ``demo`` (frozen, dashboard).
        random_state: Seed for reproducibility.
        label_column: Name of the binary label column.
        multi_label_column: Name of the multi-class label column
            (stratification target when present; falls back to binary
            label otherwise).

    The four ratios MUST sum to 1.0 (validated at construction time).
    """

    def __init__(
        self,
        train_ratio: float = 0.60,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        demo_ratio: float = 0.10,
        random_state: int = 42,
        label_column: str = "Label",
        multi_label_column: str = "Attack Category",
    ) -> None:
        total = round(train_ratio + val_ratio + test_ratio + demo_ratio, 6)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"DataSplitter ratios must sum to 1.0, got "
                f"train={train_ratio} + val={val_ratio} + test={test_ratio} "
                f"+ demo={demo_ratio} = {total}"
            )
        self._train_ratio = train_ratio
        self._val_ratio = val_ratio
        self._test_ratio = test_ratio
        self._demo_ratio = demo_ratio
        self._random_state = random_state
        self._label_col = label_column
        self._multi_label_col = multi_label_column
        self._stats: Dict[str, Any] = {}

    def split(self, df: pd.DataFrame) -> SplitOutput:
        """4-way stratified split → ``SplitOutput`` dataclass.

        The split is implemented as 3 sequential
        ``StratifiedShuffleSplit`` calls, each stratified on the
        multi-class label (``Attack Category``) when available, else
        on the binary label. This preserves attack-category proportions
        within ±2% across all 4 partitions.

        Sequence:
            (1) split → demo (10%) vs rest (90%)
            (2) split rest → test (15% absolute = 16.67% of rest) vs trainval
            (3) split trainval → val (15% absolute = 20% of trainval) vs train

        Raises:
            ValueError: If the label column is not found.
        """
        if self._label_col not in df.columns:
            raise ValueError(f"Label column '{self._label_col}' not found.")

        y = df[self._label_col].values

        has_multi = self._multi_label_col in df.columns
        y_multi = df[self._multi_label_col].values if has_multi else None

        drop_cols = [self._label_col]
        if has_multi:
            drop_cols.append(self._multi_label_col)
        X_df = df.drop(columns=drop_cols).select_dtypes(include=[np.number])
        feature_names = X_df.columns.tolist()
        X = X_df.values

        stratify_full = y_multi if has_multi else y

        # ── Step (1): demo (10%) vs rest (90%) ──
        sss_demo = StratifiedShuffleSplit(
            n_splits=1,
            test_size=self._demo_ratio,
            random_state=self._random_state,
        )
        rest_idx, demo_idx = next(sss_demo.split(X, stratify_full))

        # ── Step (2): test (15% absolute) vs trainval (75%) ──
        # test_size relative to rest = test_global / (1 - demo_global)
        test_size_rel = self._test_ratio / (1.0 - self._demo_ratio)
        sss_test = StratifiedShuffleSplit(
            n_splits=1,
            test_size=test_size_rel,
            random_state=self._random_state,
        )
        stratify_rest = (y_multi[rest_idx] if has_multi else y[rest_idx])
        inner_trainval_idx, inner_test_idx = next(
            sss_test.split(X[rest_idx], stratify_rest)
        )
        trainval_idx = rest_idx[inner_trainval_idx]
        test_idx = rest_idx[inner_test_idx]

        # ── Step (3): val (15% absolute) vs train (60%) ──
        # val_size relative to trainval = val_global / (train_global + val_global)
        val_size_rel = self._val_ratio / (self._train_ratio + self._val_ratio)
        sss_val = StratifiedShuffleSplit(
            n_splits=1,
            test_size=val_size_rel,
            random_state=self._random_state,
        )
        stratify_trainval = (
            y_multi[trainval_idx] if has_multi else y[trainval_idx]
        )
        inner_train_idx, inner_val_idx = next(
            sss_val.split(X[trainval_idx], stratify_trainval)
        )
        train_idx = trainval_idx[inner_train_idx]
        val_idx = trainval_idx[inner_val_idx]

        # ── Materialise partitions ──
        X_train, y_train = X[train_idx], y[train_idx]
        X_val,   y_val   = X[val_idx],   y[val_idx]
        X_test,  y_test  = X[test_idx],  y[test_idx]
        X_demo,  y_demo  = X[demo_idx],  y[demo_idx]

        empty_obj = np.array([], dtype=object)
        if has_multi:
            y_multi_train = y_multi[train_idx]
            y_multi_val = y_multi[val_idx]
            y_multi_test = y_multi[test_idx]
            y_multi_demo = y_multi[demo_idx]
        else:
            y_multi_train = empty_obj
            y_multi_val = empty_obj
            y_multi_test = empty_obj
            y_multi_demo = empty_obj

        n_total = len(X)
        self._stats = {
            "train_samples": int(len(X_train)),
            "val_samples":   int(len(X_val)),
            "test_samples":  int(len(X_test)),
            "demo_samples":  int(len(X_demo)),
            "train_ratio_global": round(len(X_train) / n_total, 4),
            "val_ratio_global":   round(len(X_val)   / n_total, 4),
            "test_ratio_global":  round(len(X_test)  / n_total, 4),
            "demo_ratio_global":  round(len(X_demo)  / n_total, 4),
            "stratified": True,
            "stratify_target": (
                self._multi_label_col if has_multi else self._label_col
            ),
            "train_attack_rate": round(float(y_train.mean()), 4),
            "val_attack_rate":   round(float(y_val.mean()),   4),
            "test_attack_rate":  round(float(y_test.mean()),  4),
            "demo_attack_rate":  round(float(y_demo.mean()),  4),
        }
        logger.info(
            "DataSplitter: train=%d (atk=%.1f%%) | val=%d (atk=%.1f%%) "
            "| test=%d (atk=%.1f%%) | demo=%d (atk=%.1f%%)",
            len(X_train), y_train.mean() * 100,
            len(X_val), y_val.mean() * 100,
            len(X_test), y_test.mean() * 100,
            len(X_demo), y_demo.mean() * 100,
        )
        return SplitOutput(
            X_train=X_train, X_val=X_val, X_test=X_test, X_demo=X_demo,
            y_train=y_train, y_val=y_val, y_test=y_test, y_demo=y_demo,
            feature_names=feature_names,
            y_multi_train=y_multi_train,
            y_multi_val=y_multi_val,
            y_multi_test=y_multi_test,
            y_multi_demo=y_multi_demo,
        )

    def get_report(self) -> Dict[str, Any]:
        return dict(self._stats)
