"""Tests for Phase2_5Config — pydantic validation and YAML loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

from src.phase2_5_fine_tuning.phase2_5.config import (
    AblationVariantConfig,
    Phase2_5Config,
    QuickTrainConfig,
    SearchSpaceConfig,
)


def _make_config(**overrides: Any) -> Phase2_5Config:
    defaults: Dict[str, Any] = {
        "phase1_train": Path("data/processed/train_phase1.parquet"),
        "phase1_test": Path("data/processed/test_phase1.parquet"),
        "phase2_config": Path("config/phase2_config.yaml"),
        "output_dir": Path("data/phase2_5"),
    }
    defaults.update(overrides)
    return Phase2_5Config(**defaults)


class TestPhase2_5Config:
    """Validate Phase2_5Config construction and field validators."""

    def test_valid_defaults(self) -> None:
        cfg = _make_config()
        assert cfg.search_strategy == "random"
        assert cfg.max_trials == 20
        assert cfg.search_metric == "f1_score"
        assert cfg.search_direction == "maximize"
        assert cfg.random_state == 42

    def test_invalid_strategy(self) -> None:
        with pytest.raises(Exception, match="search_strategy"):
            _make_config(search_strategy="bayesian")

    def test_invalid_direction(self) -> None:
        with pytest.raises(Exception, match="search_direction"):
            _make_config(search_direction="ascending")

    def test_invalid_metric(self) -> None:
        with pytest.raises(Exception, match="search_metric"):
            _make_config(search_metric="loss")

    def test_max_trials_zero(self) -> None:
        with pytest.raises(Exception, match="max_trials"):
            _make_config(max_trials=0)

    def test_grid_strategy(self) -> None:
        cfg = _make_config(search_strategy="grid")
        assert cfg.search_strategy == "grid"


class TestSearchSpaceConfig:
    """Validate SearchSpaceConfig defaults."""

    def test_defaults(self) -> None:
        space = SearchSpaceConfig()
        assert 64 in space.cnn_filters_1
        assert 0.3 in space.dropout_rate
        assert 20 in space.timesteps


class TestQuickTrainConfig:
    """Validate QuickTrainConfig defaults."""

    def test_defaults(self) -> None:
        qt = QuickTrainConfig()
        assert qt.epochs == 3
        assert qt.validation_split == 0.2
        assert qt.dense_units == 64


class TestAblationVariantConfig:
    """Validate AblationVariantConfig."""

    def test_remove_variant(self) -> None:
        v = AblationVariantConfig(
            name="no_attention",
            description="Remove attention",
            remove="attention",
        )
        assert v.remove == "attention"
        assert v.replace is None

    def test_replace_variant(self) -> None:
        v = AblationVariantConfig(
            name="unidirectional",
            description="Replace BiLSTM",
            replace="bilstm_to_lstm",
        )
        assert v.replace == "bilstm_to_lstm"

    def test_override_variant(self) -> None:
        v = AblationVariantConfig(
            name="low_dropout",
            description="Low dropout",
            override={"dropout_rate": 0.1},
        )
        assert v.override == {"dropout_rate": 0.1}


class TestPhase2_5ConfigFromYaml:
    """Test YAML loading."""

    def test_from_yaml(self, tmp_path: Path) -> None:
        yaml_content = """\
data:
  phase1_train: "data/processed/train_phase1.parquet"
  phase1_test: "data/processed/test_phase1.parquet"
  phase2_config: "config/phase2_config.yaml"
  label_column: "Label"
search:
  strategy: "random"
  max_trials: 10
  metric: "auc_roc"
  direction: "maximize"
  space:
    cnn_filters_1: [32, 64]
    cnn_filters_2: [128]
    cnn_kernel_size: [3]
    bilstm_units_1: [128]
    bilstm_units_2: [64]
    dropout_rate: [0.3]
    attention_units: [128]
    timesteps: [20]
    batch_size: [256]
    learning_rate: [0.001]
quick_train:
  epochs: 2
  validation_split: 0.15
  classification_head:
    dense_units: 32
    dense_activation: "relu"
    dropout_rate: 0.2
ablation:
  baseline: "full"
  variants:
    - name: "no_attention"
      description: "Remove attention"
      remove: "attention"
output:
  output_dir: "data/phase2_5"
  tuning_results_file: "tuning_results.json"
random_state: 99
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml_content)

        cfg = Phase2_5Config.from_yaml(config_file)
        assert cfg.search_strategy == "random"
        assert cfg.max_trials == 10
        assert cfg.search_metric == "auc_roc"
        assert cfg.random_state == 99
        assert cfg.quick_train.epochs == 2
        assert cfg.quick_train.dense_units == 32
        assert len(cfg.ablation_variants) == 1
        assert cfg.ablation_variants[0].name == "no_attention"
        assert 32 in cfg.search_space.cnn_filters_1
