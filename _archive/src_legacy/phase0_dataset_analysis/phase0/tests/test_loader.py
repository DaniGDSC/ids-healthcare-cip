"""Unit tests for DataLoader.

Strategy
--------
- No real file I/O: ``pd.read_csv`` is patched; ``Path.exists`` is patched.
- Each test exercises a single behaviour of DataLoader.
- Fixtures build a minimal ``Phase0Config`` and a toy DataFrame.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.phase0_dataset_analysis.phase0.config import Phase0Config
from src.phase0_dataset_analysis.phase0.loader import DataLoader


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config() -> Phase0Config:
    """Minimal valid Phase0Config with two required columns."""
    return Phase0Config(
        data_path=Path("data/fake.csv"),
        output_dir=Path("results/fake"),
        label_column="Label",
        required_columns=["Label", "Attack Category"],
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


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Toy DataFrame matching the required schema."""
    return pd.DataFrame(
        {
            "Label":           [0, 1, 0, 1, 0],
            "Attack Category": ["normal", "DoS", "normal", "FDI", "normal"],
            "SrcBytes":        [100, 200, 150, 300, 120],
            "DstBytes":        [50, 80, 60, 90, 55],
        }
    )


# ---------------------------------------------------------------------------
# DataLoader.load()
# ---------------------------------------------------------------------------


class TestLoad:
    def test_load_returns_dataframe(
        self, config: Phase0Config, sample_df: pd.DataFrame
    ) -> None:
        """load() returns the DataFrame produced by pd.read_csv."""
        with (
            patch.object(Path, "exists", return_value=True),
            patch("pandas.read_csv", return_value=sample_df) as mock_read,
        ):
            loader = DataLoader(config)
            result = loader.load()

        mock_read.assert_called_once_with(config.data_path, low_memory=False)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == sample_df.shape

    def test_load_raises_when_file_missing(self, config: Phase0Config) -> None:
        """load() raises FileNotFoundError when the CSV does not exist."""
        with patch.object(Path, "exists", return_value=False):
            loader = DataLoader(config)
            with pytest.raises(FileNotFoundError, match="Dataset not found"):
                loader.load()

    def test_load_logs_shape(
        self,
        config: Phase0Config,
        sample_df: pd.DataFrame,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """load() emits an INFO log containing row and column counts."""
        import logging

        with (
            patch.object(Path, "exists", return_value=True),
            patch("pandas.read_csv", return_value=sample_df),
            caplog.at_level(logging.INFO),
        ):
            DataLoader(config).load()

        assert any("5" in r.message and "4" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# DataLoader.validate()
# ---------------------------------------------------------------------------


class TestValidate:
    def test_validate_passes_when_all_columns_present(
        self, config: Phase0Config, sample_df: pd.DataFrame
    ) -> None:
        """validate() returns None silently when all required columns exist."""
        loader = DataLoader(config)
        loader.validate(sample_df)  # must not raise

    def test_validate_raises_on_missing_column(
        self, config: Phase0Config
    ) -> None:
        """validate() raises KeyError listing the absent column."""
        df_missing = pd.DataFrame({"Label": [0, 1]})  # no 'Attack Category'
        loader = DataLoader(config)

        with pytest.raises(KeyError, match="Attack Category"):
            loader.validate(df_missing)

    def test_validate_raises_on_all_missing(
        self, config: Phase0Config
    ) -> None:
        """validate() raises KeyError when the DataFrame has no required columns."""
        empty_df = pd.DataFrame({"SrcBytes": [100, 200]})
        loader = DataLoader(config)

        with pytest.raises(KeyError):
            loader.validate(empty_df)

    def test_validate_logs_success(
        self,
        config: Phase0Config,
        sample_df: pd.DataFrame,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """validate() emits an INFO log confirming schema is valid."""
        import logging

        with caplog.at_level(logging.INFO):
            DataLoader(config).validate(sample_df)

        assert any("validation passed" in r.message for r in caplog.records)

    def test_validate_logs_error_before_raising(
        self, config: Phase0Config, caplog: pytest.LogCaptureFixture
    ) -> None:
        """validate() logs ERROR before raising KeyError."""
        import logging

        bad_df = pd.DataFrame({"X": [1, 2]})
        with caplog.at_level(logging.ERROR):
            with pytest.raises(KeyError):
                DataLoader(config).validate(bad_df)

        assert any(r.levelname == "ERROR" for r in caplog.records)


# ---------------------------------------------------------------------------
# DataLoader.overview()
# ---------------------------------------------------------------------------


class TestOverview:
    def test_overview_logs_shape(
        self,
        config: Phase0Config,
        sample_df: pd.DataFrame,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """overview() emits INFO logs containing shape, dtypes, and head rows."""
        import logging

        with caplog.at_level(logging.INFO):
            DataLoader(config).overview(sample_df)

        messages = " ".join(r.message for r in caplog.records)
        assert "5" in messages        # row count
        assert "4" in messages        # column count
        assert "head_rows" not in messages  # should show the value, not the key

    def test_overview_respects_head_rows(
        self, config: Phase0Config, caplog: pytest.LogCaptureFixture
    ) -> None:
        """overview() passes config.head_rows to DataFrame.head()."""
        import logging

        large_df = pd.DataFrame({"Label": range(100), "Attack Category": range(100)})
        mock_head = MagicMock(return_value=large_df.head(config.head_rows))
        large_df.head = mock_head  # type: ignore[method-assign]

        with caplog.at_level(logging.INFO):
            DataLoader(config).overview(large_df)

        mock_head.assert_called_once_with(config.head_rows)
