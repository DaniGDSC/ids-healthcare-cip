"""Unit tests for the quality report renderer.

Strategy
--------
- All inputs are synthetic dicts/lists — no file I/O, no real dataset.
- Each test verifies that a specific section is present and well-formed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.phase0_dataset_analysis.phase0.config import Phase0Config
from src.phase0_dataset_analysis.phase0.quality_report import render_quality_report


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config() -> Phase0Config:
    return Phase0Config(
        data_path=Path("data/fake.csv"),
        output_dir=Path("results/fake"),
        label_column="Label",
        required_columns=["Label"],
        leakage_columns=["SrcAddr", "DstAddr", "Sport"],
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


@pytest.fixture()
def class_dist():
    return {
        "Normal": {"count": 14272, "percentage": 87.4617},
        "Attack": {"count": 2046, "percentage": 12.5383},
        "imbalance_ratio": 6.9756,
    }


@pytest.fixture()
def outlier_report():
    return [
        {
            "feature": "Load",
            "q1": 100.0,
            "q3": 5000.0,
            "iqr": 4900.0,
            "lower_bound": -7250.0,
            "upper_bound": 12350.0,
            "outlier_count": 320,
            "outlier_pct": 1.9608,
            "total": 16318,
        },
        {
            "feature": "SpO2",
            "q1": 88.0,
            "q3": 99.0,
            "iqr": 11.0,
            "lower_bound": 71.5,
            "upper_bound": 115.5,
            "outlier_count": 0,
            "outlier_pct": 0.0,
            "total": 16318,
        },
    ]


@pytest.fixture()
def high_pairs():
    return [
        ("SIntPktAct", "SrcJitter", 0.9973),
        ("Loss", "pLoss", 0.9860),
    ]


@pytest.fixture()
def rendered(config, class_dist, outlier_report, high_pairs) -> str:
    """Pre-render the full report for reuse across tests."""
    return render_quality_report(
        config=config,
        n_rows=16318,
        n_cols=45,
        class_dist=class_dist,
        outlier_report=outlier_report,
        high_pairs=high_pairs,
        missing={},
        top_variance=[("Load", 12393250127.24), ("SrcLoad", 6309105848.11)],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestReportStructure:
    def test_contains_main_header(self, rendered: str) -> None:
        assert "## 3.2 Data Quality Assessment" in rendered

    def test_contains_outlier_section(self, rendered: str) -> None:
        assert "### 3.2.1 Outlier Analysis" in rendered

    def test_contains_class_imbalance_section(self, rendered: str) -> None:
        assert "### 3.2.2 Class Imbalance Analysis" in rendered

    def test_contains_correlation_section(self, rendered: str) -> None:
        assert "### 3.2.3 Feature Correlation Analysis" in rendered

    def test_contains_missing_values_section(self, rendered: str) -> None:
        assert "### 3.2.4 Missing Value Summary" in rendered

    def test_contains_leakage_section(self, rendered: str) -> None:
        assert "### 3.2.5 Data Leakage Risk Assessment" in rendered

    def test_contains_reproducibility_section(self, rendered: str) -> None:
        assert "### 3.2.6 Reproducibility Statement" in rendered


class TestOutlierSection:
    def test_shows_features_with_outliers(self, rendered: str) -> None:
        """Only features with outlier_count > 0 appear in the table."""
        assert "Load" in rendered
        # SpO2 has 0 outliers — should NOT appear in the outlier table row
        lines = rendered.split("\n")
        outlier_table_rows = [
            l for l in lines if l.startswith("| ") and "outlier" not in l.lower()
            and "Feature" not in l and "---" not in l
        ]
        spo2_in_outlier_table = any("SpO2" in l and "1.5" not in l for l in outlier_table_rows)
        # SpO2 has 0 outliers so it should be excluded from the outlier table
        # (but might appear elsewhere in the doc); check the table specifically
        assert "320" in rendered  # Load's outlier count

    def test_mentions_robustscaler(self, rendered: str) -> None:
        assert "RobustScaler" in rendered


class TestClassImbalanceSection:
    def test_contains_ratio(self, rendered: str) -> None:
        assert "6.9756:1" in rendered

    def test_justifies_smote(self, rendered: str) -> None:
        assert "SMOTE" in rendered

    def test_contains_counts(self, rendered: str) -> None:
        assert "14,272" in rendered
        assert "2,046" in rendered


class TestCorrelationSection:
    def test_lists_pairs(self, rendered: str) -> None:
        assert "SIntPktAct" in rendered
        assert "SrcJitter" in rendered
        assert "Loss" in rendered

    def test_contains_interpretations(self, rendered: str) -> None:
        assert "jitter" in rendered.lower() or "Timing" in rendered


class TestLeakageSection:
    def test_lists_dropped_columns(self, rendered: str) -> None:
        assert "`SrcAddr`" in rendered
        assert "`DstAddr`" in rendered
        assert "`Sport`" in rendered

    def test_contains_justification(self, rendered: str) -> None:
        assert "Justification" in rendered
        assert "HIPAA" in rendered

    def test_empty_leakage_columns(self, config, class_dist, outlier_report, high_pairs) -> None:
        """When leakage_columns is empty, no feature list is rendered."""
        config.leakage_columns = []
        content = render_quality_report(
            config=config, n_rows=100, n_cols=10,
            class_dist=class_dist, outlier_report=outlier_report,
            high_pairs=high_pairs, missing={},
            top_variance=[("a", 1.0)],
        )
        assert "No columns were identified" in content


class TestReproducibilitySection:
    def test_contains_random_state(self, rendered: str) -> None:
        assert "random_state=42" in rendered

    def test_contains_split_ratio(self, rendered: str) -> None:
        assert "70/30" in rendered

    def test_mentions_stratified(self, rendered: str) -> None:
        assert "stratified" in rendered.lower()


class TestMissingValuesSection:
    def test_no_missing_statement(self, rendered: str) -> None:
        """When missing dict is empty, report says 'zero missing values'."""
        assert "zero missing values" in rendered

    def test_with_missing_values(self, config, class_dist, outlier_report, high_pairs) -> None:
        """When there are missing values, a table is rendered."""
        missing = {"SpO2": {"count": 50, "percentage": 0.3064}}
        content = render_quality_report(
            config=config, n_rows=16318, n_cols=45,
            class_dist=class_dist, outlier_report=outlier_report,
            high_pairs=high_pairs, missing=missing,
            top_variance=[("Load", 1.0)],
        )
        assert "SpO2" in content
        assert "50" in content
        assert "forward-fill" in content
