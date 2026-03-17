"""Phase 2 dataset loader.

Pipeline
--------
Parquet file
    ↓
pandas.read_parquet() → DataFrame
    ↓
.values → NumPy array
    ↓
tf.convert_to_tensor()
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import tensorflow as tf

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"
_REPORT_PATH = _PROCESSED_DIR / "phase1_report.json"


class DataSplit(NamedTuple):
    X: tf.Tensor          # float32, shape (n_samples, n_features)
    y: tf.Tensor          # int32,   shape (n_samples,)
    feature_names: list   # list[str]


def load(split: str = "train") -> DataSplit:
    """Load a Phase 1 Parquet split as TensorFlow tensors.

    Parameters
    ----------
    split : {"train", "test"}

    Returns
    -------
    DataSplit(X, y, feature_names)
        X : float32 tensor (n_samples, n_features)
        y : int32   tensor (n_samples,)
    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    report = json.loads(_REPORT_PATH.read_text())
    feature_names: list = report["output"]["feature_names"]

    parquet_path = Path(report["output"][f"{split}_parquet"])

    # Parquet → DataFrame
    df: pd.DataFrame = pd.read_parquet(parquet_path)

    # DataFrame → NumPy
    X_np: np.ndarray = np.asarray(df[feature_names], dtype=np.float32)
    y_np: np.ndarray = np.asarray(df["Label"], dtype=np.int32)

    # NumPy → TensorFlow
    X = tf.convert_to_tensor(X_np, dtype=tf.float32)
    y = tf.convert_to_tensor(y_np, dtype=tf.int32)

    return DataSplit(X=X, y=y, feature_names=feature_names)


def load_both() -> tuple:
    """Return (train, test) DataSplit tuple in one call."""
    return load("train"), load("test")
