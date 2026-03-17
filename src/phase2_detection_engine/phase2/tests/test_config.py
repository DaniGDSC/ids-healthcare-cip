"""Unit tests for Phase2Config (pydantic validation + from_yaml)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.phase2_detection_engine.phase2.config import Phase2Config


class TestPhase2Config:
    """Validate Phase2Config construction and field validators."""

    @staticmethod
    def _make_config(**overrides) -> Phase2Config:
        """Create a Phase2Config with sensible defaults."""
        defaults = dict(
            train_parquet=Path("data/processed/train_phase1.parquet"),
            test_parquet=Path("data/processed/test_phase1.parquet"),
            metadata_file=Path("data/processed/preprocessing_metadata.json"),
            report_file=Path("data/processed/phase1_report.json"),
            label_column="Label",
            timesteps=20,
            stride=1,
            cnn_filters_1=64,
            cnn_filters_2=128,
            cnn_kernel_size=3,
            cnn_activation="relu",
            cnn_pool_size=2,
            bilstm_units_1=128,
            bilstm_units_2=64,
            dropout_rate=0.3,
            attention_units=128,
            output_dir=Path("data/phase2"),
            model_file="detection_model.weights.h5",
            attention_parquet="attention_output.parquet",
            report_json="detection_report.json",
            random_state=42,
        )
        defaults.update(overrides)
        return Phase2Config(**defaults)

    def test_valid_defaults(self) -> None:
        cfg = self._make_config()
        assert cfg.timesteps == 20
        assert cfg.stride == 1
        assert cfg.cnn_filters_1 == 64
        assert cfg.dropout_rate == 0.3

    def test_timesteps_minimum(self) -> None:
        with pytest.raises(Exception, match="timesteps"):
            self._make_config(timesteps=1)

    def test_timesteps_valid_boundary(self) -> None:
        cfg = self._make_config(timesteps=2)
        assert cfg.timesteps == 2

    def test_stride_minimum(self) -> None:
        with pytest.raises(Exception, match="stride"):
            self._make_config(stride=0)

    def test_dropout_lower_bound(self) -> None:
        cfg = self._make_config(dropout_rate=0.0)
        assert cfg.dropout_rate == 0.0

    def test_dropout_upper_bound(self) -> None:
        with pytest.raises(Exception, match="dropout"):
            self._make_config(dropout_rate=1.0)

    def test_dropout_negative(self) -> None:
        with pytest.raises(Exception, match="dropout"):
            self._make_config(dropout_rate=-0.1)

    def test_units_positive(self) -> None:
        with pytest.raises(Exception):
            self._make_config(cnn_filters_1=0)

    def test_pool_size_positive(self) -> None:
        with pytest.raises(Exception):
            self._make_config(cnn_pool_size=0)

    def test_model_file_extension(self) -> None:
        with pytest.raises(Exception, match="weights.h5"):
            self._make_config(model_file="model.h5")

    def test_model_file_valid(self) -> None:
        cfg = self._make_config(model_file="my_model.weights.h5")
        assert cfg.model_file == "my_model.weights.h5"


class TestPhase2ConfigFromYaml:
    """Test YAML loading and mapping."""

    def test_from_yaml(self, tmp_path: Path) -> None:
        yaml_content = {
            "data": {
                "train_parquet": "data/processed/train_phase1.parquet",
                "test_parquet": "data/processed/test_phase1.parquet",
                "metadata_file": "data/processed/preprocessing_metadata.json",
                "report_file": "data/processed/phase1_report.json",
                "label_column": "Label",
            },
            "reshape": {"timesteps": 20, "stride": 1},
            "cnn": {
                "filters_1": 64,
                "filters_2": 128,
                "kernel_size": 3,
                "activation": "relu",
                "pool_size": 2,
            },
            "bilstm": {"units_1": 128, "units_2": 64, "dropout_rate": 0.3},
            "attention": {"units": 128},
            "output": {
                "output_dir": "data/phase2",
                "model_file": "detection_model.weights.h5",
                "attention_parquet": "attention_output.parquet",
                "report_file": "detection_report.json",
            },
            "random_state": 42,
        }
        yaml_path = tmp_path / "test_config.yaml"
        yaml_path.write_text(yaml.dump(yaml_content), encoding="utf-8")

        cfg = Phase2Config.from_yaml(yaml_path)
        assert cfg.timesteps == 20
        assert cfg.cnn_filters_2 == 128
        assert cfg.bilstm_units_2 == 64
        assert cfg.attention_units == 128
        assert cfg.random_state == 42
