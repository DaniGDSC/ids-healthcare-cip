"""Regression tests for the 4-way stratified splitter (Strategy 1).

ARCHITECTURE.md Step [1] — Frozen Test + Demo Pool. The splitter
produces 4 disjoint partitions whose ratios sum to 1.0:

    train (60%) | val (15%) | test (15%) | demo (10%)

Invariants checked here:

* I1  Default ratios sum to 1.0 (60/15/15/10 by default).
* I2  Sample sizes match ratios exactly (deterministic in random_state).
* I3  All 4 splits are pairwise-disjoint and their union covers every
      row (no row repeats; no row dropped).
* I4  Stratification preserves attack-rate within ±2pp of the global
      rate across all 4 partitions.
* I5  Two splits at the same seed produce byte-identical partitions
      (split is reproducible).
* I6  The report exposes per-split sample counts + attack rates +
      stratify_target.
* I7  Ratios that don't sum to 1.0 are rejected at construction time.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from module1_preprocessing.splitter import DataSplitter


def _make_df(n_rows: int = 1000, attack_rate: float = 0.10,
             n_features: int = 10, seed: int = 0) -> pd.DataFrame:
    """Synthesize a binary-labelled DF with two attack categories."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_rows, n_features)).astype(np.float32)
    y = (rng.random(n_rows) < attack_rate).astype(int)
    cats = np.where(
        y == 1,
        rng.choice(["Spoofing", "Data Alteration"], size=n_rows),
        "normal",
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    df["Label"] = y
    df["Attack Category"] = cats
    return df


# ── I1 + I2: default ratios + size match ──────────────────────────────


def test_default_ratios_are_60_15_15_10() -> None:
    """4-way default produces train=60%, val=15%, test=15%, demo=10%."""
    df = _make_df(1000)
    sp = DataSplitter(random_state=42)
    out = sp.split(df)
    # Allow ±1 row stratification rounding tolerance.
    assert abs(len(out.X_train) - 600) <= 1
    assert abs(len(out.X_val)   - 150) <= 1
    assert abs(len(out.X_test)  - 150) <= 1
    assert abs(len(out.X_demo)  - 100) <= 1
    assert (
        len(out.X_train) + len(out.X_val) + len(out.X_test) + len(out.X_demo)
        == 1000
    )


def test_custom_ratios_match_partition_sizes() -> None:
    df = _make_df(2000)
    sp = DataSplitter(
        train_ratio=0.50, val_ratio=0.20, test_ratio=0.20, demo_ratio=0.10,
        random_state=42,
    )
    out = sp.split(df)
    assert abs(len(out.X_train) - 1000) <= 2
    assert abs(len(out.X_val)   -  400) <= 2
    assert abs(len(out.X_test)  -  400) <= 2
    assert abs(len(out.X_demo)  -  200) <= 2


# ── I3: pairwise disjoint + total coverage ────────────────────────────


def test_4way_splits_are_pairwise_disjoint_by_content() -> None:
    """No row repeats across train/val/test/demo; union covers every row."""
    df = _make_df(1000)
    df["row_id"] = np.arange(1000)
    sp = DataSplitter(random_state=42)
    out = sp.split(df)
    row_id_col = out.feature_names.index("row_id")
    train_ids = set(out.X_train[:, row_id_col].astype(int))
    val_ids   = set(out.X_val[:, row_id_col].astype(int))
    test_ids  = set(out.X_test[:, row_id_col].astype(int))
    demo_ids  = set(out.X_demo[:, row_id_col].astype(int))
    # Pairwise disjoint
    assert len(train_ids & val_ids) == 0
    assert len(train_ids & test_ids) == 0
    assert len(train_ids & demo_ids) == 0
    assert len(val_ids & test_ids) == 0
    assert len(val_ids & demo_ids) == 0
    assert len(test_ids & demo_ids) == 0
    # Union covers every row
    assert len(train_ids | val_ids | test_ids | demo_ids) == 1000


# ── I4: stratification preserved across all 4 partitions ──────────────


def test_4way_split_preserves_attack_rate_stratification() -> None:
    """Each split's attack-rate within ±2pp of the global rate."""
    df = _make_df(4000, attack_rate=0.125, seed=7)
    sp = DataSplitter(random_state=42)
    out = sp.split(df)
    global_rate = df["Label"].mean()
    for label, y in (
        ("train", out.y_train),
        ("val",   out.y_val),
        ("test",  out.y_test),
        ("demo",  out.y_demo),
    ):
        assert abs(y.mean() - global_rate) < 0.02, (
            f"{label} attack_rate={y.mean():.4f} drifted >2pp from "
            f"global={global_rate:.4f}"
        )


# ── I5: deterministic at fixed seed ───────────────────────────────────


def test_4way_split_deterministic_at_fixed_seed() -> None:
    df = _make_df(500)
    sp1 = DataSplitter(random_state=42)
    sp2 = DataSplitter(random_state=42)
    out1 = sp1.split(df)
    out2 = sp2.split(df)
    np.testing.assert_array_equal(out1.X_train, out2.X_train)
    np.testing.assert_array_equal(out1.X_val,   out2.X_val)
    np.testing.assert_array_equal(out1.X_test,  out2.X_test)
    np.testing.assert_array_equal(out1.X_demo,  out2.X_demo)


# ── I6: report shape ──────────────────────────────────────────────────


def test_split_report_exposes_4way_stats() -> None:
    df = _make_df(1000)
    sp = DataSplitter(random_state=42)
    sp.split(df)
    rep = sp.get_report()
    for k in (
        "train_samples", "val_samples", "test_samples", "demo_samples",
        "train_ratio_global", "val_ratio_global",
        "test_ratio_global",  "demo_ratio_global",
        "train_attack_rate",  "val_attack_rate",
        "test_attack_rate",   "demo_attack_rate",
        "stratified", "stratify_target",
    ):
        assert k in rep, f"missing {k!r} in get_report() output"
    assert rep["stratified"] is True
    assert rep["stratify_target"] == "Attack Category"


# ── I7: bad ratios rejected at construction ───────────────────────────


def test_ratios_not_summing_to_one_raise() -> None:
    with pytest.raises(ValueError):
        DataSplitter(
            train_ratio=0.50, val_ratio=0.30, test_ratio=0.30, demo_ratio=0.10,
        )
    with pytest.raises(ValueError):
        DataSplitter(
            train_ratio=0.40, val_ratio=0.10, test_ratio=0.10, demo_ratio=0.10,
        )
