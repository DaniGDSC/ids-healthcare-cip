"""Tests for Phase2_5Config — pydantic validation and YAML loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

from src.phase2_5_fine_tuning.phase2_5.config import (
    AblationVariantConfig,
    MultiSeedConfig,
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
    def test_valid_defaults(self) -> None:
        cfg = _make_config()
        assert cfg.search_metric == "attack_f1"
        assert cfg.max_trials == 30
        assert cfg.search_direction == "maximize"
        assert cfg.random_state == 42

    def test_invalid_direction(self) -> None:
        with pytest.raises(Exception, match="search_direction"):
            _make_config(search_direction="ascending")

    def test_invalid_metric(self) -> None:
        with pytest.raises(Exception, match="search_metric"):
            _make_config(search_metric="loss")

    def test_max_trials_zero(self) -> None:
        with pytest.raises(Exception, match="max_trials"):
            _make_config(max_trials=0)


class TestSearchSpaceConfig:
    def test_defaults(self) -> None:
        space = SearchSpaceConfig()
        assert space.head_lr_low == 5e-4
        assert space.head_lr_high == 5e-3
        assert space.cw_attack_low == 1.0
        assert 5 in space.head_epochs
        assert 3 in space.ft_epochs


class TestMultiSeedConfig:
    def test_defaults(self) -> None:
        ms = MultiSeedConfig()
        assert ms.enabled is False
        assert ms.top_k == 3
        assert len(ms.seeds) == 5


class TestQuickTrainConfig:
    def test_defaults(self) -> None:
        qt = QuickTrainConfig()
        assert qt.epochs == 5
        assert qt.dense_units == 64


class TestAblationVariantConfig:
    def test_remove_variant(self) -> None:
        v = AblationVariantConfig(name="no_attention", description="Remove attention", remove="attention")
        assert v.remove == "attention"

    def test_override_variant(self) -> None:
        v = AblationVariantConfig(name="low_dropout", description="Low dropout", override={"dropout_rate": 0.1})
        assert v.override == {"dropout_rate": 0.1}


class TestPhase2_5ConfigFromYaml:
    def test_from_yaml(self, tmp_path: Path) -> None:
        yaml_content = """\
data:
  phase1_train: "data/processed/train_phase1.parquet"
  phase1_test: "data/processed/test_phase1.parquet"
  phase2_config: "config/phase2_config.yaml"
search:
  max_trials: 15
  metric: "auc_roc"
  direction: "maximize"
  space:
    head_lr_low: 0.001
    head_lr_high: 0.01
    finetune_lr_low: 0.00001
    finetune_lr_high: 0.001
    cw_attack_low: 1.5
    cw_attack_high: 4.0
    head_epochs: [3, 5]
    ft_epochs: [1, 2]
quick_train:
  epochs: 2
  classification_head:
    dense_units: 32
multi_seed:
  enabled: true
  top_k: 2
  seeds: [42, 99]
ablation:
  baseline: "full"
  variants:
    - name: "no_attention"
      description: "Remove attention"
      remove: "attention"
output:
  output_dir: "data/phase2_5"
random_state: 99
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml_content)

        cfg = Phase2_5Config.from_yaml(config_file)
        assert cfg.max_trials == 15
        assert cfg.search_metric == "auc_roc"
        assert cfg.random_state == 99
        assert cfg.quick_train.epochs == 2
        assert cfg.multi_seed.enabled is True
        assert cfg.search_space.head_lr_low == 0.001
        assert cfg.search_space.cw_attack_high == 4.0
        assert len(cfg.ablation_variants) == 1
