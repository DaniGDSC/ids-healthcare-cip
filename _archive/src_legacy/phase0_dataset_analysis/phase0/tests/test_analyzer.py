"""Unit tests for StatisticsAnalyzer, CorrelationAnalyzer, and OutlierAnalyzer.

Strategy
--------
- All DataFrames are built in-memory — no file I/O.
- Each test asserts a single behaviour; edge cases are explicit.
- Fixtures are shared via pytest parametrize where useful.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.phase0_dataset_analysis.phase0.analyzer import (
    CorrelationAnalyzer,
    OutlierAnalyzer,
    StatisticsAnalyzer,
)
from src.phase0_dataset_analysis.phase0.config import Phase0Config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_config(**overrides) -> Phase0Config:
    """Build a Phase0Config with sensible defaults, allowing overrides."""
    defaults = dict(
        data_path=Path("data/fake.csv"),
        output_dir=Path("results/fake"),
        label_column="Label",
        required_columns=["Label"],
        leakage_columns=[],
        network_feature_count=35,
        biometric_feature_count=8,
        correlation_threshold=0.95,
        head_rows=3,
        missing_value_warn_pct=5.0,
        outlier_iqr_multiplier=1.5,
        top_variance_k=5,
        random_state=42,
        train_ratio=0.70,
        test_ratio=0.30,
        stats_report_file="stats.json",
        high_correlations_file="corr.csv",
        correlation_matrix_file="matrix.parquet",
        quality_report_file="report_section_quality.md",
    )
    defaults.update(overrides)
    return Phase0Config(**defaults)


@pytest.fixture()
def config() -> Phase0Config:
    """Minimal valid Phase0Config."""
    return _make_config()


@pytest.fixture()
def balanced_df() -> pd.DataFrame:
    """50 / 50 Normal / Attack DataFrame with two numeric features."""
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame(
        {
            "Label":    [0] * 50 + [1] * 50,
            "Feature1": rng.normal(0.0, 1.0, n),
            "Feature2": rng.normal(5.0, 2.0, n),
        }
    )


@pytest.fixture()
def imbalanced_df() -> pd.DataFrame:
    """87.5 / 12.5 Normal / Attack imbalance (mirrors WUSTL-EHMS split)."""
    n_normal, n_attack = 875, 125
    return pd.DataFrame(
        {
            "Label":    [0] * n_normal + [1] * n_attack,
            "Feature1": list(range(n_normal + n_attack)),
        }
    )


# ---------------------------------------------------------------------------
# StatisticsAnalyzer.descriptive_stats()
# ---------------------------------------------------------------------------


class TestDescriptiveStats:
    def test_returns_all_numeric_columns(
        self, config: Phase0Config, balanced_df: pd.DataFrame
    ) -> None:
        """descriptive_stats() covers every numeric column."""
        result = StatisticsAnalyzer(balanced_df, config).descriptive_stats()
        # Label, Feature1, Feature2 are numeric
        assert set(result.keys()) == {"Label", "Feature1", "Feature2"}

    def test_keys_per_feature(
        self, config: Phase0Config, balanced_df: pd.DataFrame
    ) -> None:
        """Each feature entry contains exactly mean, median, std, min, max."""
        result = StatisticsAnalyzer(balanced_df, config).descriptive_stats()
        for feature_stats in result.values():
            assert set(feature_stats.keys()) == {"mean", "median", "std", "min", "max"}

    def test_correct_values_for_known_data(self, config: Phase0Config) -> None:
        """Computed stats match manual calculation for a deterministic series."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "Label": [0] * 5})
        result = StatisticsAnalyzer(df, config).descriptive_stats()["x"]

        assert result["mean"]   == pytest.approx(3.0, abs=1e-5)
        assert result["median"] == pytest.approx(3.0, abs=1e-5)
        assert result["min"]    == pytest.approx(1.0, abs=1e-5)
        assert result["max"]    == pytest.approx(5.0, abs=1e-5)

    def test_skips_non_numeric_columns(self, config: Phase0Config) -> None:
        """Non-numeric columns are not present in the result."""
        df = pd.DataFrame({"Label": [0, 1], "Category": ["a", "b"]})
        result = StatisticsAnalyzer(df, config).descriptive_stats()
        assert "Category" not in result

    def test_excludes_nan_from_computation(self, config: Phase0Config) -> None:
        """NaN values do not bias mean or min/max."""
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0], "Label": [0, 0, 1]})
        result = StatisticsAnalyzer(df, config).descriptive_stats()["x"]
        assert result["mean"] == pytest.approx(2.0, abs=1e-5)
        assert result["min"]  == pytest.approx(1.0, abs=1e-5)

    def test_values_rounded_to_six_places(self, config: Phase0Config) -> None:
        """All returned float values have at most 6 decimal places."""
        df = pd.DataFrame({"x": [1 / 3, 2 / 3], "Label": [0, 1]})
        result = StatisticsAnalyzer(df, config).descriptive_stats()["x"]
        for v in result.values():
            assert round(v, 6) == v


# ---------------------------------------------------------------------------
# StatisticsAnalyzer.missing_values()
# ---------------------------------------------------------------------------


class TestMissingValues:
    def test_empty_when_no_missing(
        self, config: Phase0Config, balanced_df: pd.DataFrame
    ) -> None:
        """missing_values() returns an empty dict for a complete DataFrame."""
        result = StatisticsAnalyzer(balanced_df, config).missing_values()
        assert result == {}

    def test_detects_missing_column(self, config: Phase0Config) -> None:
        """missing_values() reports the one column that has NaN values."""
        df = pd.DataFrame(
            {"Label": [0, 1, 0], "x": [1.0, np.nan, 3.0]}
        )
        result = StatisticsAnalyzer(df, config).missing_values()
        assert "x" in result
        assert result["x"]["count"] == 1
        assert result["x"]["percentage"] == pytest.approx(100 / 3, abs=0.01)

    def test_omits_complete_columns(self, config: Phase0Config) -> None:
        """missing_values() does not include features with zero NaN values."""
        df = pd.DataFrame({"Label": [0, 1], "complete": [1.0, 2.0], "partial": [np.nan, 1.0]})
        result = StatisticsAnalyzer(df, config).missing_values()
        assert "complete" not in result
        assert "partial" in result

    def test_warns_above_threshold(
        self,
        config: Phase0Config,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """missing_values() logs WARNING when missing% > missing_value_warn_pct."""
        import logging

        # 60% missing — above the 5% threshold
        df = pd.DataFrame({"Label": [0] * 10, "x": [np.nan] * 6 + [1.0] * 4})
        with caplog.at_level(logging.WARNING):
            StatisticsAnalyzer(df, config).missing_values()

        assert any(r.levelname == "WARNING" and "x" in r.message for r in caplog.records)

    def test_no_warning_below_threshold(
        self,
        config: Phase0Config,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """missing_values() does not warn when missing% <= missing_value_warn_pct."""
        import logging

        # 1 / 100 = 1% missing — below the 5% threshold
        data = [np.nan] + [1.0] * 99
        df = pd.DataFrame({"Label": [0] * 100, "x": data})
        with caplog.at_level(logging.WARNING):
            StatisticsAnalyzer(df, config).missing_values()

        assert not any(r.levelname == "WARNING" for r in caplog.records)


# ---------------------------------------------------------------------------
# StatisticsAnalyzer.class_distribution()
# ---------------------------------------------------------------------------


class TestClassDistribution:
    def test_balanced_split(
        self, config: Phase0Config, balanced_df: pd.DataFrame
    ) -> None:
        """50/50 split is reported correctly."""
        result = StatisticsAnalyzer(balanced_df, config).class_distribution()
        assert result["Normal"]["count"]      == 50
        assert result["Attack"]["count"]      == 50
        assert result["Normal"]["percentage"] == pytest.approx(50.0, abs=0.01)
        assert result["Attack"]["percentage"] == pytest.approx(50.0, abs=0.01)

    def test_imbalanced_split(
        self, config: Phase0Config, imbalanced_df: pd.DataFrame
    ) -> None:
        """87.5/12.5 imbalance is reported with correct percentages."""
        result = StatisticsAnalyzer(imbalanced_df, config).class_distribution()
        assert result["Normal"]["count"]      == 875
        assert result["Attack"]["count"]      == 125
        assert result["Normal"]["percentage"] == pytest.approx(87.5, abs=0.01)
        assert result["Attack"]["percentage"] == pytest.approx(12.5, abs=0.01)

    def test_raises_when_label_missing(self, config: Phase0Config) -> None:
        """class_distribution() raises KeyError if label column is absent."""
        df = pd.DataFrame({"Feature1": [0, 1, 0]})
        with pytest.raises(KeyError, match="Label"):
            StatisticsAnalyzer(df, config).class_distribution()

    def test_logs_error_before_raising(
        self, config: Phase0Config, caplog: pytest.LogCaptureFixture
    ) -> None:
        """class_distribution() logs ERROR before raising KeyError."""
        import logging

        df = pd.DataFrame({"x": [1, 2]})
        with caplog.at_level(logging.ERROR):
            with pytest.raises(KeyError):
                StatisticsAnalyzer(df, config).class_distribution()

        assert any(r.levelname == "ERROR" for r in caplog.records)

    def test_attack_absent_returns_zero(self, config: Phase0Config) -> None:
        """class_distribution() returns count=0 for Attack when none exist."""
        df = pd.DataFrame({"Label": [0, 0, 0]})
        result = StatisticsAnalyzer(df, config).class_distribution()
        assert result["Attack"]["count"] == 0
        assert result["Attack"]["percentage"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# CorrelationAnalyzer.correlation_matrix()
# ---------------------------------------------------------------------------


class TestCorrelationMatrix:
    def test_returns_square_dataframe(
        self, config: Phase0Config, balanced_df: pd.DataFrame
    ) -> None:
        """correlation_matrix() returns a square DataFrame of numeric features."""
        matrix = CorrelationAnalyzer(balanced_df, config).correlation_matrix()
        n_numeric = len(balanced_df.select_dtypes(include="number").columns)
        assert matrix.shape == (n_numeric, n_numeric)

    def test_diagonal_is_one(
        self, config: Phase0Config, balanced_df: pd.DataFrame
    ) -> None:
        """All diagonal entries are exactly 1.0 (self-correlation)."""
        matrix = CorrelationAnalyzer(balanced_df, config).correlation_matrix()
        for col in matrix.columns:
            assert matrix.loc[col, col] == pytest.approx(1.0, abs=1e-10)

    def test_matrix_is_cached(
        self, config: Phase0Config, balanced_df: pd.DataFrame
    ) -> None:
        """correlation_matrix() computes once; the same object is returned twice."""
        analyzer = CorrelationAnalyzer(balanced_df, config)
        first  = analyzer.correlation_matrix()
        second = analyzer.correlation_matrix()
        assert first is second  # identity check — not a recompute

    def test_perfect_correlation_detected(self, config: Phase0Config) -> None:
        """A perfectly collinear pair produces r = 1.0."""
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.0]})
        matrix = CorrelationAnalyzer(df, config).correlation_matrix()
        assert matrix.loc["a", "b"] == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# CorrelationAnalyzer.high_correlation_pairs()
# ---------------------------------------------------------------------------


class TestHighCorrelationPairs:
    def test_finds_high_correlation_pair(self, config: Phase0Config) -> None:
        """Perfectly correlated pair (r=1.0) is returned above threshold 0.95."""
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.0], "Label": [0, 1, 0]})
        pairs = CorrelationAnalyzer(df, config).high_correlation_pairs()
        feature_pairs = {(p[0], p[1]) for p in pairs}
        assert ("a", "b") in feature_pairs

    def test_returns_empty_when_no_pairs_above_threshold(
        self, config: Phase0Config
    ) -> None:
        """Returns empty list when no pair exceeds the threshold."""
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "a": rng.normal(size=200),
                "b": rng.normal(size=200),   # independent → |r| << 0.95
            }
        )
        pairs = CorrelationAnalyzer(df, config).high_correlation_pairs()
        assert pairs == []

    def test_upper_triangle_only(self, config: Phase0Config) -> None:
        """Each pair is reported exactly once (no (b, a) duplicate of (a, b))."""
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.0]})
        pairs = CorrelationAnalyzer(df, config).high_correlation_pairs()
        seen = set()
        for col_a, col_b, _ in pairs:
            key = tuple(sorted([col_a, col_b]))
            assert key not in seen, f"Pair {key} reported more than once"
            seen.add(key)

    def test_sorted_by_descending_absolute_r(self, config: Phase0Config) -> None:
        """Pairs are ordered from highest to lowest |r|."""
        # a–b perfect correlation, a–c near-perfect
        df = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0],
                "b": [2.0, 4.0, 6.0, 8.0, 10.0],   # r = 1.0
                "c": [1.1, 1.9, 3.1, 3.9, 5.1],     # r ≈ 0.9997
            }
        )
        pairs = CorrelationAnalyzer(df, config).high_correlation_pairs()
        absolute_rs = [abs(p[2]) for p in pairs]
        assert absolute_rs == sorted(absolute_rs, reverse=True)

    def test_skips_nan_correlations(self, config: Phase0Config) -> None:
        """Zero-variance features (NaN correlation) do not crash the method."""
        df = pd.DataFrame(
            {
                "constant": [1.0, 1.0, 1.0],   # zero variance → NaN corr
                "a":        [1.0, 2.0, 3.0],
            }
        )
        # Should not raise, just skip the NaN entry
        pairs = CorrelationAnalyzer(df, config).high_correlation_pairs()
        assert isinstance(pairs, list)


# ---------------------------------------------------------------------------
# OutlierAnalyzer.outlier_report()
# ---------------------------------------------------------------------------


class TestOutlierReport:
    def test_returns_entry_per_numeric_column(
        self, config: Phase0Config, balanced_df: pd.DataFrame
    ) -> None:
        """outlier_report() returns one entry per numeric feature."""
        report = OutlierAnalyzer(balanced_df, config).outlier_report()
        n_numeric = len(balanced_df.select_dtypes(include="number").columns)
        assert len(report) == n_numeric

    def test_entry_keys(
        self, config: Phase0Config, balanced_df: pd.DataFrame
    ) -> None:
        """Each report entry contains all expected keys."""
        report = OutlierAnalyzer(balanced_df, config).outlier_report()
        expected_keys = {
            "feature", "q1", "q3", "iqr", "lower_bound",
            "upper_bound", "outlier_count", "outlier_pct", "total",
        }
        for entry in report:
            assert set(entry.keys()) == expected_keys

    def test_detects_known_outlier(self, config: Phase0Config) -> None:
        """An extreme value outside the IQR fence is counted as an outlier."""
        # IQR for [1,2,3,4,5] = Q3(4) - Q1(2) = 2
        # Fence: [2-3, 4+3] = [-1, 7]  → 100 is an outlier
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]})
        report = OutlierAnalyzer(df, config).outlier_report()
        x_entry = [r for r in report if r["feature"] == "x"][0]
        assert x_entry["outlier_count"] >= 1

    def test_no_outliers_for_uniform_data(self, config: Phase0Config) -> None:
        """A constant series has zero outliers."""
        df = pd.DataFrame({"x": [5.0] * 100})
        report = OutlierAnalyzer(df, config).outlier_report()
        assert report[0]["outlier_count"] == 0

    def test_sorted_by_descending_outlier_pct(
        self, config: Phase0Config
    ) -> None:
        """Report entries are sorted from highest to lowest outlier %."""
        df = pd.DataFrame({
            "no_outliers": [1.0] * 100,
            "some_outliers": [1.0] * 95 + [1000.0] * 5,
        })
        report = OutlierAnalyzer(df, config).outlier_report()
        pcts = [r["outlier_pct"] for r in report]
        assert pcts == sorted(pcts, reverse=True)

    def test_iqr_bounds_correct(self, config: Phase0Config) -> None:
        """IQR bounds are computed as Q1 - 1.5*IQR and Q3 + 1.5*IQR."""
        # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] → Q1=2.25, Q3=6.75, IQR=4.5
        df = pd.DataFrame({"x": list(range(10))})
        report = OutlierAnalyzer(df, config).outlier_report()
        x_entry = report[0]
        expected_lower = x_entry["q1"] - 1.5 * x_entry["iqr"]
        expected_upper = x_entry["q3"] + 1.5 * x_entry["iqr"]
        assert x_entry["lower_bound"] == pytest.approx(expected_lower, abs=1e-4)
        assert x_entry["upper_bound"] == pytest.approx(expected_upper, abs=1e-4)

    def test_custom_multiplier(self) -> None:
        """A higher multiplier produces wider fences and fewer outliers."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 20.0]})
        narrow = OutlierAnalyzer(df, _make_config(outlier_iqr_multiplier=1.0)).outlier_report()
        wide = OutlierAnalyzer(df, _make_config(outlier_iqr_multiplier=3.0)).outlier_report()
        narrow_count = narrow[0]["outlier_count"]
        wide_count = wide[0]["outlier_count"]
        assert wide_count <= narrow_count
