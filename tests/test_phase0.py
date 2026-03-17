"""Tests for Phase 0: Dataset Analysis."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.phase0_dataset_analysis.dataset_profiler import DatasetProfiler
from src.phase0_dataset_analysis.report_generator import ReportGenerator
from src.phase0_dataset_analysis.wustl_ehms_schema import WUSTLEHMSSchema


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Minimal synthetic DataFrame that resembles WUSTL-EHMS structure."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame(
        {
            "feature_a": np.random.normal(0, 1, n),
            "feature_b": np.random.normal(0, 1, n),
            # Highly correlated pair
            "feature_c": np.random.normal(0, 1, n),
            "feature_d": None,  # will be set below
            # Column with missing values
            "feature_e": np.where(np.arange(n) % 5 == 0, np.nan, np.random.rand(n)),
            "label": (["Normal"] * 140) + (["Attack"] * 60),
        }
    )
    # Make feature_d highly correlated with feature_c
    df["feature_d"] = df["feature_c"] * 0.99 + np.random.normal(0, 0.01, n)
    return df


@pytest.fixture
def profiler(sample_df: pd.DataFrame) -> DatasetProfiler:
    return DatasetProfiler(
        sample_df,
        label_col="label",
        correlation_threshold=0.90,
        outlier_iqr_multiplier=1.5,
        top_n_correlations=10,
    )


@pytest.fixture
def full_report(profiler: DatasetProfiler) -> dict:
    return profiler.run_full_profile()


# ---------------------------------------------------------------------------
# DatasetProfiler tests
# ---------------------------------------------------------------------------

class TestProfileShape:
    def test_correct_row_count(self, profiler: DatasetProfiler, sample_df: pd.DataFrame):
        shape = profiler.profile_shape()
        assert shape["n_rows"] == len(sample_df)

    def test_correct_col_count(self, profiler: DatasetProfiler, sample_df: pd.DataFrame):
        shape = profiler.profile_shape()
        assert shape["n_columns"] == len(sample_df.columns)


class TestProfileMissing:
    def test_detects_missing(self, profiler: DatasetProfiler):
        missing = profiler.profile_missing()
        assert missing["feature_e"]["missing_count"] > 0

    def test_zero_missing_for_complete_col(self, profiler: DatasetProfiler):
        missing = profiler.profile_missing()
        assert missing["feature_a"]["missing_count"] == 0

    def test_missing_pct_in_range(self, profiler: DatasetProfiler):
        missing = profiler.profile_missing()
        for info in missing.values():
            assert 0.0 <= info["missing_pct"] <= 100.0


class TestProfileClassBalance:
    def test_correct_n_classes(self, profiler: DatasetProfiler):
        balance = profiler.profile_class_balance()
        assert balance["n_classes"] == 2

    def test_correct_counts(self, profiler: DatasetProfiler):
        balance = profiler.profile_class_balance()
        dist = balance["distribution"]
        assert dist["Normal"]["count"] == 140
        assert dist["Attack"]["count"] == 60

    def test_pct_sums_to_100(self, profiler: DatasetProfiler):
        balance = profiler.profile_class_balance()
        total_pct = sum(v["pct"] for v in balance["distribution"].values())
        assert abs(total_pct - 100.0) < 0.01


class TestProfileCorrelations:
    def test_finds_high_correlation_pair(self, profiler: DatasetProfiler):
        pairs = profiler.profile_correlations()
        feature_pairs = {(p["feature_a"], p["feature_b"]) for p in pairs}
        # feature_c and feature_d should appear (corr ≈ 0.99)
        assert ("feature_c", "feature_d") in feature_pairs or \
               ("feature_d", "feature_c") in feature_pairs

    def test_all_above_threshold(self, profiler: DatasetProfiler):
        pairs = profiler.profile_correlations()
        for pair in pairs:
            assert pair["correlation"] >= profiler.correlation_threshold

    def test_max_pairs_respected(self, profiler: DatasetProfiler):
        pairs = profiler.profile_correlations()
        assert len(pairs) <= profiler.top_n_correlations


class TestProfileOutliers:
    def test_returns_all_numeric_non_label(self, profiler: DatasetProfiler, sample_df: pd.DataFrame):
        outliers = profiler.profile_outliers()
        numeric_non_label = [
            c for c in sample_df.select_dtypes(include=[np.number]).columns
            if c != "label"
        ]
        for col in numeric_non_label:
            assert col in outliers

    def test_outlier_count_non_negative(self, profiler: DatasetProfiler):
        outliers = profiler.profile_outliers()
        for info in outliers.values():
            assert info["n_outliers"] >= 0


class TestFullProfile:
    def test_report_has_all_keys(self, full_report: dict):
        expected_keys = {
            "shape", "dtypes", "missing", "descriptive_stats",
            "class_balance", "high_correlations", "outliers",
            "summary", "iomt_characteristics",
        }
        assert expected_keys.issubset(full_report.keys())

    def test_summary_has_correct_row_count(self, full_report: dict):
        assert full_report["summary"]["n_rows"] == 200


# ---------------------------------------------------------------------------
# ReportGenerator tests
# ---------------------------------------------------------------------------

class TestReportGenerator:
    def test_json_file_created(self, full_report: dict):
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = ReportGenerator(Path(tmpdir))
            path = gen.save_json(full_report)
            assert path.exists()
            with open(path) as f:
                data = json.load(f)
            assert "shape" in data

    def test_descriptive_stats_csv_created(self, full_report: dict):
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = ReportGenerator(Path(tmpdir))
            path = gen.save_csv_stats(full_report)
            assert path.exists()
            assert path.stat().st_size > 0

    def test_class_balance_csv_created(self, full_report: dict):
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = ReportGenerator(Path(tmpdir))
            path = gen.save_class_balance_csv(full_report)
            assert path.exists()
            lines = path.read_text().strip().splitlines()
            # header + 2 class rows
            assert len(lines) == 3

    def test_correlation_csv_created(self, full_report: dict):
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = ReportGenerator(Path(tmpdir))
            path = gen.save_correlation_csv(full_report)
            assert path.exists()

    def test_save_all_creates_seven_files(self, full_report: dict):
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = ReportGenerator(Path(tmpdir))
            paths = gen.save_all(full_report)
            assert len(paths) == 7
            for p in paths.values():
                assert p.exists()


# ---------------------------------------------------------------------------
# WUSTLEHMSSchema tests
# ---------------------------------------------------------------------------

class TestWUSTLEHMSSchema:
    def test_total_feature_count(self):
        assert len(WUSTLEHMSSchema.ALL_FEATURES) == 43  # 35 + 8

    def test_expected_columns_includes_label(self):
        cols = WUSTLEHMSSchema.expected_columns()
        assert WUSTLEHMSSchema.LABEL_COLUMN in cols

    def test_validate_columns_missing(self):
        result = WUSTLEHMSSchema.validate_columns(["dur", "HR", "label"])
        # Most expected columns should be in 'missing'
        assert len(result["missing"]) > 30

    def test_validate_columns_extra(self):
        extra = WUSTLEHMSSchema.expected_columns() + ["unexpected_col"]
        result = WUSTLEHMSSchema.validate_columns(extra)
        assert "unexpected_col" in result["extra"]

    def test_validate_all_present(self):
        result = WUSTLEHMSSchema.validate_columns(WUSTLEHMSSchema.expected_columns())
        assert result["missing"] == []
        assert result["extra"] == []


# ---------------------------------------------------------------------------
# IoMT thesis characteristics tests
# ---------------------------------------------------------------------------

class TestIoMTCharacteristics:
    """Tests for the 5 IoMT thesis evidence methods."""

    def test_returns_all_five_characteristics(self, profiler: DatasetProfiler):
        iomt = profiler.profile_iomt_characteristics()
        for key in [
            "medical_footprint",
            "extreme_imbalance",
            "temporal_dynamics",
            "feature_redundancy",
            "attack_vector_profiling",
        ]:
            assert key in iomt, f"Missing IoMT characteristic: {key}"

    def test_medical_footprint_counts(self, profiler: DatasetProfiler):
        mf = profiler.profile_iomt_characteristics()["medical_footprint"]
        # No WUSTL-EHMS cols in sample_df, so biometric/network counts should be 0
        assert "biometric_count" in mf
        assert "network_count" in mf
        assert isinstance(mf["biometric_count"], int)

    def test_extreme_imbalance_ratio(self, profiler: DatasetProfiler):
        ei = profiler.profile_iomt_characteristics()["extreme_imbalance"]
        ratio = ei["imbalance_ratio"]
        # 140 Normal / 60 Attack ≈ 2.33
        assert ratio is not None
        assert abs(ratio - (140 / 60)) < 0.01

    def test_feature_redundancy_pair_count(self, profiler: DatasetProfiler):
        fr = profiler.profile_iomt_characteristics()["feature_redundancy"]
        # feature_c and feature_d are highly correlated → at least 1 pair
        assert fr["n_high_correlation_pairs"] >= 1

    def test_attack_vector_top_features(self, profiler: DatasetProfiler):
        av = profiler.profile_iomt_characteristics()["attack_vector_profiling"]
        top = av["top_outlier_features"]
        assert len(top) <= 3
        for item in top:
            assert "feature" in item
            assert "n_outliers" in item

    def test_iomt_summary_csv_created(self, full_report: dict):
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = ReportGenerator(Path(tmpdir))
            path = gen.save_iomt_summary_csv(full_report)
            assert path.exists()
            lines = path.read_text().strip().splitlines()
            # header + 5 characteristic rows
            assert len(lines) == 6
            assert "characteristic" in lines[0]


# ---------------------------------------------------------------------------
# Phase0Pipeline smoke test (no real CSV needed)
# ---------------------------------------------------------------------------

class TestPhase0PipelineSmoke:
    def test_run_with_temp_csv(self, sample_df: pd.DataFrame):
        """End-to-end: write sample_df to a temp CSV, run the pipeline."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.phase0_dataset_analysis.run_phase0 import Phase0Pipeline
        import logging

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            # Write sample CSV
            csv_path = tmp / "test_data.csv"
            sample_df.to_csv(csv_path, index=False)

            config = {
                "data": {
                    "input_dir": str(tmp),
                    "output_dir": str(tmp / "out"),
                    "file_pattern": "*.csv",
                    "label_column": "label",
                },
                "profiling": {
                    "correlation_threshold": 0.90,
                    "outlier_iqr_multiplier": 1.5,
                    "top_n_correlations": 10,
                },
            }
            logger = logging.getLogger("test_phase0")
            pipeline = Phase0Pipeline(config, logger)
            report = pipeline.run()

            # Check outputs
            out_dir = tmp / "out"
            assert (out_dir / "analysis_report.json").exists()
            assert (out_dir / "descriptive_stats.csv").exists()
            assert (out_dir / "class_balance.csv").exists()
            assert (out_dir / "iomt_thesis_evidence.csv").exists()
            assert report["summary"]["n_rows"] == 200
            assert "iomt_characteristics" in report
