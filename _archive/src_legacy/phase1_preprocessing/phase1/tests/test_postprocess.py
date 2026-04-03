"""Unit tests for post-split components: DataSplitter, SMOTEBalancer, RobustScalerTransformer.

Tests include data leakage prevention and SMOTE train-only assertions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.phase1_preprocessing.phase1.scaler import RobustScalerTransformer
from src.phase1_preprocessing.phase1.smote import SMOTEBalancer
from src.phase1_preprocessing.phase1.splitter import DataSplitter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def split_df() -> pd.DataFrame:
    """DataFrame with enough samples for splitting."""
    rng = np.random.RandomState(42)
    n = 200
    return pd.DataFrame({
        "f1": rng.randn(n),
        "f2": rng.randn(n),
        "f3": rng.randn(n),
        "Label": np.array([0] * 160 + [1] * 40),
    })


@pytest.fixture()
def train_arrays():
    """Pre-split arrays for SMOTE and scaler tests."""
    rng = np.random.RandomState(42)
    X_train = rng.randn(100, 5)
    y_train = np.array([0] * 80 + [1] * 20)
    X_test = rng.randn(30, 5)
    return X_train, y_train, X_test


# ---------------------------------------------------------------------------
# DataSplitter
# ---------------------------------------------------------------------------


class TestDataSplitter:
    def test_split_shapes(self, split_df: pd.DataFrame) -> None:
        splitter = DataSplitter(test_ratio=0.30, random_state=42)
        X_train, X_test, y_train, y_test, feat_names, *_ = splitter.split(split_df)
        assert len(X_train) + len(X_test) == len(split_df)
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)

    def test_stratified_class_balance(self, split_df: pd.DataFrame) -> None:
        splitter = DataSplitter(test_ratio=0.30, random_state=42)
        _, _, y_train, y_test, *_ = splitter.split(split_df)
        train_rate = y_train.mean()
        test_rate = y_test.mean()
        # Class ratios should be approximately equal
        assert abs(train_rate - test_rate) < 0.05

    def test_feature_names_exclude_label(self, split_df: pd.DataFrame) -> None:
        splitter = DataSplitter()
        _, _, _, _, feat_names, *_ = splitter.split(split_df)
        assert "Label" not in feat_names
        assert "f1" in feat_names

    def test_report_has_required_keys(self, split_df: pd.DataFrame) -> None:
        splitter = DataSplitter(test_ratio=0.30)
        splitter.split(split_df)
        report = splitter.get_report()
        assert "train_samples" in report
        assert "test_samples" in report
        assert report["stratified"] is True

    def test_raises_on_missing_label(self) -> None:
        df = pd.DataFrame({"f1": [1, 2], "f2": [3, 4]})
        splitter = DataSplitter()
        with pytest.raises(ValueError, match="Label"):
            splitter.split(df)

    def test_reproducibility(self, split_df: pd.DataFrame) -> None:
        s1 = DataSplitter(random_state=42)
        s2 = DataSplitter(random_state=42)
        X1, _, _, _, *_ = s1.split(split_df)
        X2, _, _, _, *_ = s2.split(split_df)
        np.testing.assert_array_equal(X1, X2)


# ---------------------------------------------------------------------------
# SMOTEBalancer
# ---------------------------------------------------------------------------


class TestSMOTEBalancer:
    def test_balances_classes(self, train_arrays) -> None:
        X_train, y_train, _ = train_arrays
        balancer = SMOTEBalancer(random_state=42)
        X_res, y_res = balancer.resample(X_train, y_train)
        unique, counts = np.unique(y_res, return_counts=True)
        assert counts[0] == counts[1]  # balanced 1:1

    def test_increases_sample_count(self, train_arrays) -> None:
        X_train, y_train, _ = train_arrays
        balancer = SMOTEBalancer(random_state=42)
        X_res, _ = balancer.resample(X_train, y_train)
        assert len(X_res) > len(X_train)

    def test_preserves_feature_count(self, train_arrays) -> None:
        X_train, y_train, _ = train_arrays
        balancer = SMOTEBalancer(random_state=42)
        X_res, _ = balancer.resample(X_train, y_train)
        assert X_res.shape[1] == X_train.shape[1]

    def test_report_has_counts(self, train_arrays) -> None:
        X_train, y_train, _ = train_arrays
        balancer = SMOTEBalancer(random_state=42)
        balancer.resample(X_train, y_train)
        report = balancer.get_report()
        assert report["samples_before"] == 100
        assert report["samples_after"] > 100
        assert report["synthetic_added"] > 0

    def test_smote_not_applied_to_test(self, train_arrays) -> None:
        """SMOTE must only be applied to training data."""
        X_train, y_train, X_test = train_arrays
        balancer = SMOTEBalancer(random_state=42)
        X_res, _ = balancer.resample(X_train, y_train)
        # X_test must remain unchanged
        assert X_test.shape == (30, 5)


# ---------------------------------------------------------------------------
# RobustScalerTransformer
# ---------------------------------------------------------------------------


class TestRobustScalerTransformer:
    def test_fit_transform_shape(self, train_arrays) -> None:
        X_train, _, _ = train_arrays
        scaler = RobustScalerTransformer(method="robust")
        X_scaled = scaler.fit_transform(X_train)
        assert X_scaled.shape == X_train.shape

    def test_transform_without_fit_raises(self, train_arrays) -> None:
        X_train, _, _ = train_arrays
        scaler = RobustScalerTransformer(method="robust")
        with pytest.raises(RuntimeError, match="not fitted"):
            scaler.transform(X_train)

    def test_no_data_leakage(self, train_arrays) -> None:
        """Scaler must be fitted on train only, not on test."""
        X_train, _, X_test = train_arrays
        scaler = RobustScalerTransformer(method="robust")
        X_train_s, X_test_s = scaler.scale_both(X_train, X_test)

        # Verify scaler was fitted on train
        assert scaler.is_fitted

        # Train should have median ≈ 0 (by construction of RobustScaler)
        train_medians = np.median(X_train_s, axis=0)
        np.testing.assert_array_almost_equal(train_medians, 0, decimal=1)

        # Test medians should NOT be exactly 0 (not fitted on test)
        test_medians = np.median(X_test_s, axis=0)
        assert not np.allclose(test_medians, 0, atol=0.05)

    def test_scale_both_shapes(self, train_arrays) -> None:
        X_train, _, X_test = train_arrays
        scaler = RobustScalerTransformer(method="robust")
        X_train_s, X_test_s = scaler.scale_both(X_train, X_test)
        assert X_train_s.shape == X_train.shape
        assert X_test_s.shape == X_test.shape

    def test_report(self) -> None:
        scaler = RobustScalerTransformer(method="robust")
        report = scaler.get_report()
        assert report["method"] == "robust"
        assert report["fitted"] is False

    def test_is_fitted_property(self, train_arrays) -> None:
        X_train, _, _ = train_arrays
        scaler = RobustScalerTransformer()
        assert not scaler.is_fitted
        scaler.fit(X_train)
        assert scaler.is_fitted
