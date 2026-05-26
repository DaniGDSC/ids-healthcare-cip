"""StatisticsAnalyzer + CorrelationAnalyzer + OutlierAnalyzer tests.

PHI Safe-Harbor invariant: biometric features publish aggregate-only
statistics (mean/std, outlier_count/pct) — never min/max/median/quantiles.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from module0_analysis import (
    CorrelationAnalyzer,
    OutlierAnalyzer,
    Phase0Config,
    StatisticsAnalyzer,
)


def _config(label_col="Label") -> Phase0Config:
    return Phase0Config(
        data_path=Path("data.csv"),
        output_dir=Path("out"),
        label_column=label_col,
        required_columns=[label_col],
        leakage_columns=[],
        network_feature_count=0,
        biometric_feature_count=0,
        correlation_threshold=0.95,
        missing_value_warn_pct=5.0,
        outlier_iqr_multiplier=1.5,
        top_variance_k=5,
        random_state=42,
        train_ratio=0.7,
        test_ratio=0.3,
        stats_report_file="s.json",
        high_correlations_file="c.csv",
        correlation_matrix_file="m.parquet",
        quality_report_file="r.md",
    )


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame({
        "Label": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        "Pulse_Rate": [70, 72, 75, 80, 85, 90, 95, 100, 110, 200],   # biometric + outlier
        "Temp": [36.5, 36.6, 36.8, 37.0, 37.2, 37.5, 37.8, 38.0, 38.5, 39.0],  # biometric
        "Dur": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],   # network, no outliers
        "TotPkts": [10, 20, 10, 20, 10, 20, 10, 20, 10, 20],          # bimodal network
    })


# ── StatisticsAnalyzer ────────────────────────────────────────────────


def test_descriptive_stats_biometric_excludes_minmax(df):
    """PHI invariant — biometric features publish only mean/std."""
    a = StatisticsAnalyzer(df, _config())
    stats = a.descriptive_stats()
    assert set(stats["Pulse_Rate"]) == {"mean", "std"}
    assert set(stats["Temp"]) == {"mean", "std"}
    assert "min" not in stats["Pulse_Rate"]
    assert "max" not in stats["Pulse_Rate"]
    assert "median" not in stats["Pulse_Rate"]


def test_descriptive_stats_network_includes_full_set(df):
    a = StatisticsAnalyzer(df, _config())
    stats = a.descriptive_stats()
    assert set(stats["Dur"]) == {"mean", "median", "std", "min", "max"}
    assert set(stats["TotPkts"]) == {"mean", "median", "std", "min", "max"}


def test_missing_values_reports_only_affected(df):
    """Features with zero missing must be omitted from the report."""
    a = StatisticsAnalyzer(df, _config())
    out = a.missing_values()
    assert out == {}, "Synthetic data has no missing values"


def test_missing_values_warns_above_threshold(df, caplog):
    import logging
    df2 = df.copy()
    df2.loc[df2.index[:5], "Dur"] = np.nan  # 50% missing
    a = StatisticsAnalyzer(df2, _config())
    caplog.set_level(logging.WARNING)
    out = a.missing_values()
    assert "Dur" in out
    assert out["Dur"]["percentage"] == 50.0
    assert any("50.00% missing" in r.message for r in caplog.records)


def test_class_distribution_counts(df):
    a = StatisticsAnalyzer(df, _config())
    dist = a.class_distribution()
    assert dist["Normal"]["count"] == 5
    assert dist["Attack"]["count"] == 5
    assert dist["Normal"]["percentage"] == 50.0


def test_class_distribution_raises_on_missing_label_col(df):
    a = StatisticsAnalyzer(df, _config(label_col="NotPresent"))
    with pytest.raises(KeyError, match="NotPresent"):
        a.class_distribution()


# ── CorrelationAnalyzer ───────────────────────────────────────────────


def test_correlation_matrix_is_cached(df):
    a = CorrelationAnalyzer(df, _config())
    m1 = a.correlation_matrix()
    m2 = a.correlation_matrix()
    assert m1 is m2  # exact same object — lazy cache hit


def test_high_correlation_pairs_threshold_filter(df):
    """Make a deliberately collinear pair and ensure detection."""
    df2 = df.copy()
    df2["Dur_x2"] = df2["Dur"] * 2 + 0.01  # near-perfect linear
    a = CorrelationAnalyzer(df2, _config())
    pairs = a.high_correlation_pairs()
    # Find the (Dur, Dur_x2) pair
    sorted_pairs = [(min(p[0], p[1]), max(p[0], p[1])) for p in pairs]
    assert ("Dur", "Dur_x2") in sorted_pairs


def test_high_correlation_pairs_sorted_by_abs_value(df):
    df2 = df.copy()
    df2["A"] = df2["Dur"] * 2
    df2["B"] = df2["Dur"] * -3
    df2["C"] = df2["TotPkts"] * 0.5
    a = CorrelationAnalyzer(df2, _config())
    pairs = a.high_correlation_pairs()
    abs_vals = [abs(p[2]) for p in pairs]
    assert abs_vals == sorted(abs_vals, reverse=True)


# ── OutlierAnalyzer ───────────────────────────────────────────────────


def test_outlier_report_biometric_excludes_quantiles(df):
    """PHI invariant — biometric features publish only counts."""
    a = OutlierAnalyzer(df, _config())
    report = a.outlier_report()
    bio = next(r for r in report if r["feature"] == "Pulse_Rate")
    assert set(bio.keys()) == {"feature", "outlier_count", "outlier_pct", "total"}
    assert "q1" not in bio
    assert "q3" not in bio
    assert "lower_bound" not in bio
    assert "upper_bound" not in bio


def test_outlier_report_network_includes_quantiles(df):
    a = OutlierAnalyzer(df, _config())
    report = a.outlier_report()
    net = next(r for r in report if r["feature"] == "Dur")
    assert {"q1", "q3", "iqr", "lower_bound", "upper_bound"}.issubset(net.keys())


def test_outlier_report_detects_known_outlier(df):
    """Pulse_Rate = 200 should be an obvious outlier."""
    a = OutlierAnalyzer(df, _config())
    report = a.outlier_report()
    bio = next(r for r in report if r["feature"] == "Pulse_Rate")
    assert bio["outlier_count"] >= 1


def test_outlier_report_sorted_by_outlier_pct_desc(df):
    a = OutlierAnalyzer(df, _config())
    report = a.outlier_report()
    pcts = [r["outlier_pct"] for r in report]
    assert pcts == sorted(pcts, reverse=True)
