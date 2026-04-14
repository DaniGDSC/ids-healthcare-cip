"""Shared data-loading utility for Track A and Track B tuning scripts.

A single canonical ``load_data`` implementation used by all four
run_*.py scripts so changes (e.g. new label columns, dtype policy) only
need to be made in one place.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DROP_CANDIDATES = ("Label", "Attack Category")


def load_data(
    train_path: Path,
    test_path: Path,
    label_col: str = "Label",
) -> tuple:
    """Load train/test parquet files and split into feature matrices and labels.

    Drops ``label_col`` and ``"Attack Category"`` (if present) from both
    splits.  Features are cast to ``float32`` for memory and speed.

    Args:
        train_path: Path to the training parquet file.
        test_path:  Path to the test parquet file.
        label_col:  Name of the binary label column (default ``"Label"``).

    Returns:
        ``(X_train, X_test, y_train, y_test, feat_names)``
    """
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    drop_cols = [c for c in _DROP_CANDIDATES if c in train_df.columns]

    y_train = train_df[label_col].values
    y_test = test_df[label_col].values

    X_train = train_df.drop(columns=drop_cols).values.astype(np.float32)
    X_test = test_df.drop(columns=drop_cols).values.astype(np.float32)

    feat_names = [c for c in train_df.columns if c not in drop_cols]

    logger.info(
        "Data loaded: train=%d×%d (attack=%.1f%%), test=%d×%d (attack=%.1f%%)",
        *X_train.shape, y_train.mean() * 100,
        *X_test.shape, y_test.mean() * 100,
    )
    return X_train, X_test, y_train, y_test, feat_names


def load_data_dae(
    train_path: Path,
    test_path: Path,
    label_col: str = "Label",
) -> tuple:
    """Load data for DAE (Track B) training.

    Same as ``load_data`` but additionally extracts the benign-only subset
    required by the DAE fit protocol.

    Returns:
        ``(X_benign, X_train, X_test, y_train, y_test, feat_names)``
    """
    X_train, X_test, y_train, y_test, feat_names = load_data(
        train_path, test_path, label_col=label_col,
    )

    benign_mask = y_train == 0
    X_benign = X_train[benign_mask]

    logger.info(
        "DAE split: benign=%d, attack=%d, test=%d",
        benign_mask.sum(), (~benign_mask).sum(), len(y_test),
    )
    return X_benign, X_train, X_test, y_train, y_test, feat_names
