"""Tests for Phase3Config — pydantic validation and YAML loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

from src.phase3_classification_engine.phase3.config import Phase3Config, TrainingPhaseConfig


def _make_phase(**overrides: Any) -> TrainingPhaseConfig:
    defaults = {
        "name": "Test Phase",
        "epochs": 5,
        "learning_rate": 0.001,
        "frozen": ["cnn"],
    }
    defaults.update(overrides)
    return TrainingPhaseConfig(**defaults)


def _make_config(**overrides: Any) -> Phase3Config:
    defaults: Dict[str, Any] = {
        "phase2_dir": Path("data/phase2"),
        "phase1_train": Path("data/processed/train_phase1.parquet"),
        "phase1_test": Path("data/processed/test_phase1.parquet"),
        "phase2_metadata": Path("data/phase2/detection_metadata.json"),
        "training_phases": [_make_phase()],
        "output_dir": Path("data/phase3"),
    }
    defaults.update(overrides)
    return Phase3Config(**defaults)


class TestPhase3Config:
    """Validate Phase3Config construction and field validators."""

    def test_valid_defaults(self) -> None:
        cfg = _make_config()
        assert cfg.dense_units == 64
        assert cfg.random_state == 42
        assert cfg.threshold == 0.5

    def test_units_positive(self) -> None:
        with pytest.raises(Exception, match="dense_units"):
            _make_config(dense_units=0)

    def test_dropout_negative(self) -> None:
        with pytest.raises(Exception, match="head_dropout_rate"):
            _make_config(head_dropout_rate=-0.1)

    def test_dropout_too_high(self) -> None:
        with pytest.raises(Exception, match="head_dropout_rate"):
            _make_config(head_dropout_rate=1.0)

    def test_threshold_too_high(self) -> None:
        with pytest.raises(Exception, match="threshold"):
            _make_config(threshold=1.5)

    def test_threshold_zero(self) -> None:
        with pytest.raises(Exception, match="threshold"):
            _make_config(threshold=0.0)

    def test_empty_phases(self) -> None:
        with pytest.raises(Exception, match="[Aa]t least one"):
            _make_config(training_phases=[])

    def test_weights_extension(self) -> None:
        with pytest.raises(Exception, match="weights.h5"):
            _make_config(model_file="bad.h5")


class TestPhase3ConfigFromYaml:
    """Test YAML loading."""

    def test_from_yaml(self, tmp_path: Path) -> None:
        yaml_content = """\
data:
  phase2_dir: "data/phase2"
  phase1_train: "data/processed/train_phase1.parquet"
  phase1_test: "data/processed/test_phase1.parquet"
  phase2_metadata: "data/phase2/detection_metadata.json"
  label_column: "Label"
classification_head:
  dense_units: 32
  dense_activation: "relu"
  dropout_rate: 0.2
training:
  phases:
    - name: "Phase A"
      epochs: 3
      learning_rate: 0.001
      frozen: ["cnn", "bilstm1"]
  batch_size: 128
  validation_split: 0.1
callbacks:
  early_stopping_patience: 2
  reduce_lr_patience: 1
  reduce_lr_factor: 0.3
evaluation:
  threshold: 0.4
output:
  output_dir: "data/phase3"
  model_file: "test.weights.h5"
random_state: 123
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml_content)

        cfg = Phase3Config.from_yaml(config_file)
        assert cfg.dense_units == 32
        assert cfg.head_dropout_rate == 0.2
        assert cfg.batch_size == 128
        assert cfg.threshold == 0.4
        assert cfg.random_state == 123
        assert len(cfg.training_phases) == 1
        assert cfg.training_phases[0].name == "Phase A"
