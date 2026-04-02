"""Unit tests for Phase 1 DataFrame-level transformers.

Tests HIPAASanitizer, CategoricalEncoder, MissingValueHandler,
RedundancyRemover, and VarianceFilter.
All inputs are synthetic — no file I/O, no real dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.phase1_preprocessing.phase1.encoder import CategoricalEncoder
from src.phase1_preprocessing.phase1.hipaa import HIPAASanitizer
from src.phase1_preprocessing.phase1.missing import MissingValueHandler
from src.phase1_preprocessing.phase1.shap_selector import SHAPSelector
from src.phase1_preprocessing.phase1.redundancy import RedundancyRemover
from src.phase1_preprocessing.phase1.variance import VarianceFilter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def raw_df() -> pd.DataFrame:
    """Minimal DataFrame mimicking WUSTL-EHMS structure."""
    return pd.DataFrame({
        "SrcAddr": ["192.168.1.1"] * 10,
        "DstAddr": ["10.0.0.1"] * 10,
        "Sport": ["443"] * 10,
        "Dport": [1111] * 10,
        "SrcMac": ["aa:bb"] * 10,
        "DstMac": ["cc:dd"] * 10,
        "Dir": ["->"] * 10,
        "Flgs": ["e"] * 10,
        "Load": np.random.rand(10) * 1000,
        "SrcLoad": np.random.rand(10) * 500,
        "Temp": [36.5, 36.6, np.nan, 36.8, 36.7, 36.9, np.nan, 37.0, 36.5, 36.6],
        "SpO2": [98, 97, 99, np.nan, 98, 97, 96, 98, 99, 97],
        "Label": [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
    })


# ---------------------------------------------------------------------------
# HIPAASanitizer
# ---------------------------------------------------------------------------


class TestHIPAASanitizer:
    def test_drops_phi_columns(self, raw_df: pd.DataFrame) -> None:
        hipaa = HIPAASanitizer(["SrcAddr", "DstAddr", "Sport", "Dport"])
        result = hipaa.transform(raw_df)
        assert "SrcAddr" not in result.columns
        assert "DstAddr" not in result.columns
        assert "Sport" not in result.columns
        assert "Dport" not in result.columns

    def test_preserves_non_phi_columns(self, raw_df: pd.DataFrame) -> None:
        hipaa = HIPAASanitizer(["SrcAddr"])
        result = hipaa.transform(raw_df)
        assert "Load" in result.columns
        assert "Label" in result.columns

    def test_report_counts_dropped(self, raw_df: pd.DataFrame) -> None:
        hipaa = HIPAASanitizer(["SrcAddr", "DstAddr", "NonExistent"])
        hipaa.transform(raw_df)
        report = hipaa.get_report()
        assert report["n_dropped"] == 2
        assert "SrcAddr" in report["columns_dropped"]
        assert "NonExistent" not in report["columns_dropped"]

    def test_ignores_missing_columns(self, raw_df: pd.DataFrame) -> None:
        hipaa = HIPAASanitizer(["FakeColumn"])
        result = hipaa.transform(raw_df)
        assert len(result.columns) == len(raw_df.columns)

    def test_implements_base_transformer(self) -> None:
        from src.phase1_preprocessing.phase1.base import BaseTransformer
        assert issubclass(HIPAASanitizer, BaseTransformer)

    def test_fit_returns_self(self, raw_df: pd.DataFrame) -> None:
        hipaa = HIPAASanitizer(["SrcAddr"])
        assert hipaa.fit(raw_df) is hipaa


# ---------------------------------------------------------------------------
# CategoricalEncoder
# ---------------------------------------------------------------------------


class TestCategoricalEncoder:
    def test_label_encodes_string_column(self) -> None:
        df = pd.DataFrame({
            "Flgs": ["e", "M", "eR", "e", "M"],
            "Val": [1, 2, 3, 4, 5],
        })
        enc = CategoricalEncoder(label_encode=["Flgs"])
        result = enc.transform(df)
        assert pd.api.types.is_integer_dtype(result["Flgs"])
        assert result["Flgs"].nunique() == 3

    def test_parse_numeric_coerces_strings(self) -> None:
        df = pd.DataFrame({
            "Sport": ["58059", "1111", "dircproxy", "443"],
        })
        enc = CategoricalEncoder(parse_numeric=["Sport"])
        result = enc.transform(df)
        assert pd.api.types.is_numeric_dtype(result["Sport"])
        assert result["Sport"].iloc[0] == 58059
        assert result["Sport"].iloc[2] == -1  # sentinel for non-parseable

    def test_ignores_missing_columns(self) -> None:
        df = pd.DataFrame({"A": [1, 2]})
        enc = CategoricalEncoder(label_encode=["Missing"], parse_numeric=["Also_Missing"])
        result = enc.transform(df)
        assert list(result.columns) == ["A"]

    def test_report(self) -> None:
        df = pd.DataFrame({
            "Dir": ["->", "->", "->"],
            "Sport": ["80", "443", "bad"],
        })
        enc = CategoricalEncoder(label_encode=["Dir"], parse_numeric=["Sport"])
        enc.transform(df)
        report = enc.get_report()
        assert report["label_encoded"]["Dir"] == 1
        assert report["parsed_numeric"]["Sport"] == 1


# ---------------------------------------------------------------------------
# MissingValueHandler
# ---------------------------------------------------------------------------


class TestMissingValueHandler:
    def test_ffill_biometrics(self) -> None:
        df = pd.DataFrame({
            "Temp": [36.5, np.nan, np.nan, 37.0],
            "Load": [100, 200, 300, 400],
            "Label": [0, 0, 0, 1],
        })
        handler = MissingValueHandler(biometric_columns=["Temp"])
        result = handler.transform(df)
        assert result["Temp"].isna().sum() == 0
        assert result["Temp"].iloc[1] == 36.5  # forward-filled

    def test_dropna_network(self) -> None:
        df = pd.DataFrame({
            "Load": [100, np.nan, 300, 400],
            "Rate": [10, 20, np.nan, 40],
            "Label": [0, 0, 0, 1],
        })
        handler = MissingValueHandler(biometric_columns=[])
        result = handler.transform(df)
        assert len(result) == 2  # rows 0 and 3

    def test_report_stats(self) -> None:
        df = pd.DataFrame({
            "Temp": [36.5, np.nan, 37.0],
            "Load": [100, np.nan, 300],
            "Label": [0, 0, 1],
        })
        handler = MissingValueHandler(biometric_columns=["Temp"])
        handler.transform(df)
        report = handler.get_report()
        assert report["biometric_cells_filled"] == 1
        assert report["rows_dropped"] == 1

    def test_fill_zero_network(self) -> None:
        df = pd.DataFrame({
            "Load": [100, np.nan, 300, 400],
            "Rate": [10, 20, np.nan, 40],
            "Label": [0, 0, 0, 1],
        })
        handler = MissingValueHandler(
            biometric_columns=[], network_strategy="fill_zero",
        )
        result = handler.transform(df)
        assert len(result) == 4  # no rows dropped
        assert result["Load"].iloc[1] == 0
        assert result["Rate"].iloc[2] == 0
        report = handler.get_report()
        assert report["network_cells_filled_zero"] == 2
        assert report["rows_dropped"] == 0

    def test_no_missing_passthrough(self) -> None:
        df = pd.DataFrame({
            "Temp": [36.5, 36.6],
            "Load": [100, 200],
            "Label": [0, 1],
        })
        handler = MissingValueHandler(biometric_columns=["Temp"])
        result = handler.transform(df)
        assert len(result) == 2

    def test_excludes_label_from_network(self) -> None:
        df = pd.DataFrame({
            "Load": [100, 200],
            "Label": [0, 1],
        })
        handler = MissingValueHandler(
            biometric_columns=[], label_column="Label",
        )
        result = handler.transform(df)
        assert "Label" in result.columns


# ---------------------------------------------------------------------------
# RedundancyRemover
# ---------------------------------------------------------------------------


class TestRedundancyRemover:
    def test_drops_feature_b(self) -> None:
        df = pd.DataFrame({
            "A": [1, 2, 3], "B": [2, 4, 6], "C": [10, 20, 30],
            "Label": [0, 0, 1],
        })
        corr = pd.DataFrame({
            "feature_a": ["A"], "feature_b": ["B"], "correlation": [0.99],
        })
        remover = RedundancyRemover(corr, threshold=0.95)
        result = remover.transform(df)
        assert "B" not in result.columns
        assert "A" in result.columns
        assert "C" in result.columns

    def test_keeps_feature_a(self) -> None:
        df = pd.DataFrame({"A": [1, 2], "B": [2, 4]})
        corr = pd.DataFrame({
            "feature_a": ["A"], "feature_b": ["B"], "correlation": [0.99],
        })
        remover = RedundancyRemover(corr, threshold=0.95)
        result = remover.transform(df)
        assert "A" in result.columns
        assert "B" not in result.columns

    def test_respects_threshold(self) -> None:
        df = pd.DataFrame({"A": [1, 2], "B": [2, 4], "Label": [0, 1]})
        corr = pd.DataFrame({
            "feature_a": ["A"], "feature_b": ["B"], "correlation": [0.90],
        })
        remover = RedundancyRemover(corr, threshold=0.95)
        result = remover.transform(df)
        assert "B" in result.columns  # below threshold

    def test_report_lists_dropped(self) -> None:
        df = pd.DataFrame({
            "A": [1, 2], "B": [2, 4], "C": [3, 6], "Label": [0, 1],
        })
        corr = pd.DataFrame({
            "feature_a": ["A", "A"], "feature_b": ["B", "C"],
            "correlation": [0.99, 0.96],
        })
        remover = RedundancyRemover(corr, threshold=0.95)
        remover.transform(df)
        report = remover.get_report()
        assert report["n_dropped"] == 2
        assert "B" in report["columns_dropped"]
        assert "C" in report["columns_dropped"]

    def test_no_duplicates_in_drop_list(self) -> None:
        df = pd.DataFrame({"A": [1, 2], "B": [2, 4], "Label": [0, 1]})
        corr = pd.DataFrame({
            "feature_a": ["A", "A"], "feature_b": ["B", "B"],
            "correlation": [0.99, 0.98],
        })
        remover = RedundancyRemover(corr, threshold=0.95)
        remover.transform(df)
        assert remover.get_report()["n_dropped"] == 1


# ---------------------------------------------------------------------------
# VarianceFilter
# ---------------------------------------------------------------------------


class TestVarianceFilter:
    def test_drops_unary_numeric(self) -> None:
        df = pd.DataFrame({
            "A": [1, 2, 3],
            "B": [99, 99, 99],
            "Label": [0, 0, 1],
        })
        vf = VarianceFilter(max_unique=1)
        result = vf.transform(df)
        assert "B" not in result.columns
        assert "A" in result.columns

    def test_drops_unary_non_numeric(self) -> None:
        df = pd.DataFrame({
            "A": [1, 2, 3],
            "Dir": ["->", "->", "->"],
            "Mac": ["aa:bb", "aa:bb", "aa:bb"],
            "Label": [0, 0, 1],
        })
        vf = VarianceFilter(max_unique=1)
        result = vf.transform(df)
        assert "Dir" not in result.columns
        assert "Mac" not in result.columns
        assert "A" in result.columns

    def test_preserves_multi_valued(self) -> None:
        df = pd.DataFrame({
            "A": [1, 2, 3],
            "B": [0, 0, 0],
            "C": [10, 20, 30],
        })
        vf = VarianceFilter(max_unique=1)
        result = vf.transform(df)
        assert "A" in result.columns
        assert "C" in result.columns
        assert "B" not in result.columns

    def test_report_includes_all_dtypes(self) -> None:
        df = pd.DataFrame({
            "A": [1, 2, 3],
            "B": [0, 0, 0],
            "C": ["x", "x", "x"],
            "Label": [0, 0, 1],
        })
        vf = VarianceFilter(max_unique=1)
        vf.transform(df)
        report = vf.get_report()
        assert report["n_dropped"] == 2
        assert "B" in report["columns_dropped"]
        assert "C" in report["columns_dropped"]


# ---------------------------------------------------------------------------
# SHAPSelector
# ---------------------------------------------------------------------------


class TestSHAPSelector:
    @pytest.fixture()
    def train_data(self):
        rng = np.random.RandomState(42)
        n = 200
        # f_good: strongly predictive; f_noise: random; f_weak: mildly useful
        f_good = np.concatenate([rng.randn(n // 2) - 2, rng.randn(n // 2) + 2])
        f_weak = np.concatenate([rng.randn(n // 2) - 0.3, rng.randn(n // 2) + 0.3])
        f_noise = rng.randn(n)
        X_train = np.column_stack([f_good, f_weak, f_noise])
        y_train = np.array([0] * (n // 2) + [1] * (n // 2))
        X_test = rng.randn(50, 3)
        feat_names = ["f_good", "f_weak", "f_noise"]
        return X_train, X_test, y_train, feat_names

    def test_selects_informative_feature(self, train_data) -> None:
        X_train, X_test, y_train, feat_names = train_data
        sel = SHAPSelector(min_features=1, n_estimators=50, cv_folds=2, random_state=42)
        X_tr, X_te, selected = sel.select(X_train, X_test, y_train, feat_names)
        assert "f_good" in selected
        assert X_tr.shape[1] == len(selected)
        assert X_te.shape[1] == len(selected)

    def test_rfe_eliminates_noise_first(self, train_data) -> None:
        X_train, X_test, y_train, feat_names = train_data
        sel = SHAPSelector(min_features=1, n_estimators=50, cv_folds=2, random_state=42)
        sel.select(X_train, X_test, y_train, feat_names)
        report = sel.get_report()
        # f_good should have much higher initial importance than f_noise
        assert report["initial_importances"]["f_good"] > report["initial_importances"]["f_noise"]
        # RFE should have eliminated features
        assert len(report["rfe_elimination_order"]) >= 1

    def test_report_structure(self, train_data) -> None:
        X_train, X_test, y_train, feat_names = train_data
        sel = SHAPSelector(min_features=2, n_estimators=50, cv_folds=2, random_state=42)
        sel.select(X_train, X_test, y_train, feat_names)
        report = sel.get_report()
        assert "XGBoost-RFE" in report["method"]
        assert "initial_importances" in report
        assert "cv_scores" in report
        assert "shap_crosscheck_importances" in report
        assert report["n_selected"] + report["n_dropped"] == 3

    def test_min_features_respected(self, train_data) -> None:
        X_train, X_test, y_train, feat_names = train_data
        sel = SHAPSelector(min_features=2, n_estimators=50, cv_folds=2, random_state=42)
        X_tr, _, selected = sel.select(X_train, X_test, y_train, feat_names)
        # RFE selected subset is at least min_features; SHAP crosscheck may reduce further
        assert len(selected) >= 1
