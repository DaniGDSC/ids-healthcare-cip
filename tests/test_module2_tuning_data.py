"""tuning/_data.py tests — load_data + leakage guard."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from module2_detection.tuning._data import (
    _FORBIDDEN_TRAINING_PARQUETS,
    _assert_no_demo_leakage,
    load_data,
    load_data_dae,
)


def _make_parquet(path: Path, n: int = 50, attack_frac: float = 0.3) -> None:
    rng = np.random.default_rng(0)
    n_attack = int(n * attack_frac)
    n_benign = n - n_attack
    df = pd.DataFrame({
        "f1": rng.normal(size=n).astype(np.float32),
        "f2": rng.normal(size=n).astype(np.float32),
        "Label": [0] * n_benign + [1] * n_attack,
        "Attack Category": ["normal"] * n_benign + ["recon"] * n_attack,
        "row_id": np.arange(n, dtype=np.int64),
        "device_class": ["patient_monitor"] * n,
    })
    df.to_parquet(path, index=False)


# ── Leakage guard ────────────────────────────────────────────────────


def test_demo_phase1_parquet_is_forbidden():
    assert "demo_phase1.parquet" in _FORBIDDEN_TRAINING_PARQUETS


def test_assert_no_demo_leakage_blocks_demo(tmp_path):
    with pytest.raises(RuntimeError, match="must not load demo_phase1.parquet"):
        _assert_no_demo_leakage(tmp_path / "demo_phase1.parquet")


def test_assert_no_demo_leakage_allows_train(tmp_path):
    _assert_no_demo_leakage(tmp_path / "train_phase1.parquet")  # no raise


def test_assert_no_demo_leakage_allows_test(tmp_path):
    _assert_no_demo_leakage(tmp_path / "test_phase1.parquet")  # no raise


def test_load_data_refuses_demo_as_train(tmp_path):
    demo = tmp_path / "demo_phase1.parquet"
    test = tmp_path / "test_phase1.parquet"
    _make_parquet(demo)
    _make_parquet(test)
    with pytest.raises(RuntimeError, match="must not load demo_phase1"):
        load_data(demo, test)


def test_load_data_refuses_demo_as_test(tmp_path):
    train = tmp_path / "train_phase1.parquet"
    demo = tmp_path / "demo_phase1.parquet"
    _make_parquet(train)
    _make_parquet(demo)
    with pytest.raises(RuntimeError, match="must not load demo_phase1"):
        load_data(train, demo)


# ── Happy path ────────────────────────────────────────────────────────


def test_load_data_returns_expected_shapes(tmp_path):
    train = tmp_path / "train_phase1.parquet"
    test = tmp_path / "test_phase1.parquet"
    _make_parquet(train, n=80, attack_frac=0.25)
    _make_parquet(test, n=20, attack_frac=0.25)
    X_train, X_test, y_train, y_test, feat_names = load_data(train, test)
    assert X_train.shape == (80, 2)
    assert X_test.shape == (20, 2)
    assert len(y_train) == 80
    assert feat_names == ["f1", "f2"]


def test_load_data_drops_label_and_attack_category(tmp_path):
    train = tmp_path / "train_phase1.parquet"
    test = tmp_path / "test_phase1.parquet"
    _make_parquet(train)
    _make_parquet(test)
    _, _, _, _, feat_names = load_data(train, test)
    assert "Label" not in feat_names
    assert "Attack Category" not in feat_names
    assert "row_id" not in feat_names
    assert "device_class" not in feat_names


def test_load_data_features_cast_to_float32(tmp_path):
    train = tmp_path / "train_phase1.parquet"
    test = tmp_path / "test_phase1.parquet"
    _make_parquet(train)
    _make_parquet(test)
    X_train, X_test, _, _, _ = load_data(train, test)
    assert X_train.dtype == np.float32
    assert X_test.dtype == np.float32


# ── load_data_dae extension ──────────────────────────────────────────


def test_load_data_dae_returns_benign_subset(tmp_path):
    train = tmp_path / "train_phase1.parquet"
    test = tmp_path / "test_phase1.parquet"
    _make_parquet(train, n=100, attack_frac=0.3)
    _make_parquet(test, n=20)
    X_benign, X_train, X_test, y_train, y_test, _ = load_data_dae(train, test)
    # X_benign is the y_train==0 subset; with 30% attack, 70% benign → 70 samples
    assert len(X_benign) == 70
    assert (y_train[y_train == 0]).shape[0] == 70
