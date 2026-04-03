"""Integration tests for the Phase 1 pipeline and config validation.

Tests config validation, artifact reader, report rendering, and
the full pipeline contract (using mocked I/O).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.phase1_preprocessing.phase1.artifact_reader import Phase0ArtifactReader
from src.phase1_preprocessing.phase1.config import Phase1Config
from src.phase1_preprocessing.phase1.report import render_preprocessing_report


# ---------------------------------------------------------------------------
# Phase1Config
# ---------------------------------------------------------------------------


class TestPhase1Config:
    """Test pydantic config validation."""

    def _make_config(self, **overrides) -> Phase1Config:
        defaults = dict(
            input_dir=Path("data/raw"),
            output_dir=Path("data/processed"),
            id_removal_columns=["SrcAddr", "DstAddr"],
            biometric_columns=["Temp", "SpO2"],
            phase0_corr_file=Path("results/high_correlations.csv"),
        )
        defaults.update(overrides)
        return Phase1Config(**defaults)

    def test_valid_config(self) -> None:
        cfg = self._make_config()
        assert cfg.train_ratio == 0.70
        assert cfg.random_state == 42

    def test_ratios_must_sum_to_one(self) -> None:
        with pytest.raises(Exception):
            self._make_config(train_ratio=0.50, test_ratio=0.30)

    def test_threshold_range(self) -> None:
        with pytest.raises(Exception):
            self._make_config(correlation_threshold=1.5)

    def test_k_neighbors_positive(self) -> None:
        with pytest.raises(Exception):
            self._make_config(smote_k_neighbors=0)

    def test_from_yaml(self, tmp_path: Path) -> None:
        yaml_content = """
data:
  input_dir: "data/raw/WUSTL-EHMS"
  output_dir: "data/processed"
  label_column: "Label"
identifier_removal:
  enabled: true
  remove_columns: ["SrcAddr"]
cleaning:
  biometric_columns: ["Temp"]
  biometric_strategy: "ffill"
  network_strategy: "fill_zero"
correlation_removal:
  threshold: 0.95
  phase0_corr_file: "results/high_correlations.csv"
splitting:
  train_ratio: 0.70
  test_ratio: 0.30
  random_state: 42
track_a:
  smote:
    enabled: true
    sampling_strategy: "auto"
    k_neighbors: 5
normalization:
  method: "robust"
output:
  train_parquet: "train.parquet"
"""
        p = tmp_path / "config.yaml"
        p.write_text(yaml_content)
        cfg = Phase1Config.from_yaml(p)
        assert cfg.id_removal_columns == ["SrcAddr"]
        assert cfg.scaling_method == "robust"


# ---------------------------------------------------------------------------
# Phase0ArtifactReader
# ---------------------------------------------------------------------------


class TestPhase0ArtifactReader:
    def test_read_stats(self, tmp_path: Path) -> None:
        stats = '{"descriptive_statistics": {"A": {"mean": 1.0}}, "missing_values": {}}'
        (tmp_path / "stats.json").write_text(stats)
        reader = Phase0ArtifactReader(
            tmp_path, Path("stats.json"),
            Path("corr.csv"), Path("integrity.json"),
        )
        data = reader.read_stats()
        assert "A" in data["descriptive_statistics"]

    def test_read_correlations(self, tmp_path: Path) -> None:
        csv = "feature_a,feature_b,correlation\nA,B,0.99\n"
        (tmp_path / "corr.csv").write_text(csv)
        reader = Phase0ArtifactReader(
            tmp_path, Path("stats.json"),
            Path("corr.csv"), Path("integrity.json"),
        )
        df = reader.read_correlations()
        assert len(df) == 1
        assert df.iloc[0]["feature_a"] == "A"

    def test_stats_missing_raises(self, tmp_path: Path) -> None:
        reader = Phase0ArtifactReader(
            tmp_path, Path("missing.json"),
            Path("corr.csv"), Path("integrity.json"),
        )
        with pytest.raises(FileNotFoundError):
            reader.read_stats()

    def test_correlations_missing_raises(self, tmp_path: Path) -> None:
        reader = Phase0ArtifactReader(
            tmp_path, Path("stats.json"),
            Path("missing.csv"), Path("integrity.json"),
        )
        with pytest.raises(FileNotFoundError):
            reader.read_correlations()

    def test_integrity_no_baseline_returns_hash(self, tmp_path: Path) -> None:
        fake_file = tmp_path / "data.csv"
        fake_file.write_text("a,b\n1,2\n")
        reader = Phase0ArtifactReader(
            tmp_path, Path("s.json"),
            Path("c.csv"), Path("integrity.json"),
        )
        digest = reader.verify_integrity(fake_file)
        assert len(digest) == 64  # SHA-256 hex


# ---------------------------------------------------------------------------
# Report Rendering
# ---------------------------------------------------------------------------


class TestReportRendering:
    @pytest.fixture()
    def sample_report(self) -> dict:
        return {
            "ingestion": {"files_loaded": 1, "raw_rows": 16318, "raw_columns": 45},
            "hipaa": {
                "columns_requested": ["SrcAddr", "DstAddr"],
                "columns_dropped": ["SrcAddr", "DstAddr"],
                "n_dropped": 2,
            },
            "missing_values": {
                "biometric_cells_filled": 50,
                "rows_dropped": 10,
                "rows_remaining": 16308,
            },
            "redundancy": {
                "threshold": 0.95,
                "columns_dropped": ["SrcJitter", "pLoss"],
                "n_dropped": 2,
            },
            "split": {
                "train_samples": 11400,
                "test_samples": 4918,
                "train_ratio": 0.70,
                "test_ratio": 0.30,
            },
            "smote": {
                "samples_before": 11400,
                "samples_after": 19900,
                "synthetic_added": 8500,
                "attack_rate_before": 0.125,
                "attack_rate_after": 0.5,
                "class_counts_before": {0: 9975, 1: 1425},
                "class_counts_after": {0: 9975, 1: 9925},
                "k_neighbors": 5,
            },
            "scaling": {"method": "robust"},
            "output": {"n_features": 29, "feature_names": ["f"] * 29},
            "elapsed_seconds": 0.15,
            "random_state": 42,
        }

    def test_contains_pipeline_steps_table(self, sample_report: dict) -> None:
        md = render_preprocessing_report(sample_report)
        assert "| Step | Input Shape | Output Shape | Notes |" in md

    def test_contains_smote_table(self, sample_report: dict) -> None:
        md = render_preprocessing_report(sample_report)
        assert "Before SMOTE" in md
        assert "After SMOTE" in md
        assert "Normal (0)" in md
        assert "Attack (1)" in md

    def test_contains_feature_reduction_table(self, sample_report: dict) -> None:
        md = render_preprocessing_report(sample_report)
        assert "Feature Reduction Summary" in md
        assert "HIPAA identifiers" in md
        assert "Redundancy" in md

    def test_contains_scaling_justification(self, sample_report: dict) -> None:
        md = render_preprocessing_report(sample_report)
        assert "RobustScaler" in md
        assert "outlier" in md.lower()

    def test_contains_leakage_prevention(self, sample_report: dict) -> None:
        md = render_preprocessing_report(sample_report)
        assert "fitted exclusively on training set" in md
        assert "information leakage" in md

    def test_contains_all_sections(self, sample_report: dict) -> None:
        md = render_preprocessing_report(sample_report)
        assert "### 4.1.1" in md
        assert "### 4.1.2" in md
        assert "### 4.1.3" in md
        assert "### 4.1.4" in md
        assert "### 4.1.5" in md
        assert "### 4.1.6" in md
        assert "### 4.1.7" in md

    def test_smote_counts_match(self, sample_report: dict) -> None:
        md = render_preprocessing_report(sample_report)
        assert "9,975" in md  # Normal count
        assert "11,400" in md  # Before total
