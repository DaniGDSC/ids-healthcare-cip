"""Regression tests for the 3-way stratified splitter (GAP-L1-2).

Verifies that the new ``val_ratio`` parameter:
- preserves backward-compat (val_ratio=0 → 2-way split, empty val arrays)
- produces disjoint train / val / test row sets
- preserves stratification (val attack-rate within tolerance of test)
- is deterministic at fixed seed
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from module1_preprocessing.phase1.splitter import DataSplitter


def _make_df(n_rows: int = 1000, attack_rate: float = 0.10,
             n_features: int = 10, seed: int = 0) -> pd.DataFrame:
    """Synthesize a binary-labelled DF with two attack categories."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_rows, n_features)).astype(np.float32)
    y = (rng.random(n_rows) < attack_rate).astype(int)
    cats = np.where(y == 1,
                    rng.choice(["Spoofing", "Data Alteration"], size=n_rows),
                    "normal")
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    df["Label"] = y
    df["Attack Category"] = cats
    return df


def test_default_split_is_2way_backward_compat() -> None:
    """val_ratio=0 produces empty val arrays; train+test sizes unchanged."""
    df = _make_df(1000)
    sp = DataSplitter(test_ratio=0.30, val_ratio=0.0, random_state=42)
    out = sp.split(df)
    assert len(out.X_train) == 700
    assert len(out.X_test) == 300
    assert len(out.X_val) == 0
    assert len(out.y_val) == 0
    assert len(out.y_multi_val) == 0


def test_3way_split_sizes_match_ratios() -> None:
    """val_ratio=0.20 (within trainval) → ~14% global val, 56% train, 30% test."""
    df = _make_df(1000)
    sp = DataSplitter(test_ratio=0.30, val_ratio=0.20, random_state=42)
    out = sp.split(df)
    assert len(out.X_test) == 300              # 30% of 1000
    assert len(out.X_val) == int(700 * 0.20)   # 20% of 700 = 140
    assert len(out.X_train) == 700 - len(out.X_val)
    # Global ratios
    total = len(out.X_train) + len(out.X_val) + len(out.X_test)
    assert total == 1000


def test_3way_splits_are_pairwise_disjoint_by_content() -> None:
    """No row repeats across train/val/test (uses synthetic uniqueness via row hash)."""
    df = _make_df(1000)
    df["row_id"] = np.arange(1000)  # unique tag survives split as a feature
    sp = DataSplitter(test_ratio=0.30, val_ratio=0.20, random_state=42)
    out = sp.split(df)
    # row_id is in feature_names; locate its index
    row_id_col = out.feature_names.index("row_id")
    train_ids = set(out.X_train[:, row_id_col].astype(int))
    val_ids = set(out.X_val[:, row_id_col].astype(int))
    test_ids = set(out.X_test[:, row_id_col].astype(int))
    assert len(train_ids & val_ids) == 0
    assert len(train_ids & test_ids) == 0
    assert len(val_ids & test_ids) == 0
    # Union covers every row.
    assert len(train_ids | val_ids | test_ids) == 1000


def test_3way_split_preserves_attack_rate_stratification() -> None:
    """Val and test attack-rates should be within 1.5pp of the global rate."""
    df = _make_df(2000, attack_rate=0.12, seed=7)
    sp = DataSplitter(test_ratio=0.30, val_ratio=0.20, random_state=42)
    out = sp.split(df)
    global_rate = df["Label"].mean()
    assert abs(out.y_train.mean() - global_rate) < 0.015
    assert abs(out.y_val.mean() - global_rate) < 0.015
    assert abs(out.y_test.mean() - global_rate) < 0.015


def test_3way_split_deterministic_at_fixed_seed() -> None:
    """Two runs with same seed produce identical splits."""
    df = _make_df(500)
    sp1 = DataSplitter(test_ratio=0.30, val_ratio=0.20, random_state=42)
    sp2 = DataSplitter(test_ratio=0.30, val_ratio=0.20, random_state=42)
    out1 = sp1.split(df)
    out2 = sp2.split(df)
    np.testing.assert_array_equal(out1.X_train, out2.X_train)
    np.testing.assert_array_equal(out1.X_val, out2.X_val)
    np.testing.assert_array_equal(out1.X_test, out2.X_test)


def test_split_report_includes_val_stats_when_val_ratio_positive() -> None:
    """get_report() exposes val_samples + val_attack_rate when val_ratio>0."""
    df = _make_df(1000)
    sp = DataSplitter(test_ratio=0.30, val_ratio=0.20, random_state=42)
    sp.split(df)
    rep = sp.get_report()
    assert rep["val_samples"] > 0
    assert "val_attack_rate" in rep
    assert rep["val_ratio_within_trainval"] == 0.20


def test_split_report_val_ratio_zero_when_disabled() -> None:
    """Backward-compat report shape: val_samples=0, val_ratio_global=0."""
    df = _make_df(1000)
    sp = DataSplitter(test_ratio=0.30, val_ratio=0.0, random_state=42)
    sp.split(df)
    rep = sp.get_report()
    assert rep["val_samples"] == 0
    assert rep["val_ratio_global"] == 0.0
