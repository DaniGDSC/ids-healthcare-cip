"""Tests for AblationRunner — component ablation studies."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

import numpy as np

from src.phase2_5_fine_tuning.phase2_5.ablation import AblationRunner
from src.phase2_5_fine_tuning.phase2_5.config import AblationVariantConfig, Phase2_5Config


def _make_config(**overrides: Any) -> Phase2_5Config:
    defaults: Dict[str, Any] = {
        "phase1_train": Path("data/processed/train_phase1.parquet"),
        "phase1_test": Path("data/processed/test_phase1.parquet"),
        "phase2_config": Path("config/phase2_config.yaml"),
        "ablation_variants": [
            AblationVariantConfig(name="no_attention", description="Remove attention", remove="attention"),
            AblationVariantConfig(name="low_dropout", description="Low dropout", override={"dropout_rate": 0.1}),
        ],
    }
    defaults.update(overrides)
    return Phase2_5Config(**defaults)


def _fake_result(variant: str, af1: float) -> Dict[str, Any]:
    return {
        "variant": variant,
        "metrics": {"accuracy": 0.9, "f1_score": 0.9, "auc_roc": 0.95,
                     "attack_f1": af1, "attack_recall": 0.5, "attack_precision": 0.5,
                     "macro_f1": 0.7, "threshold": 0.3},
        "total_params": 200, "detection_params": 100,
        "optimal_threshold": 0.3, "duration_seconds": 1.0,
    }


class TestAblationRunner:
    def test_runs_baseline_and_variants(self) -> None:
        config = _make_config()
        evaluator = MagicMock()
        evaluator.evaluate_ablation_variant.side_effect = [
            _fake_result("baseline", af1=0.50),
            _fake_result("no_attention", af1=0.42),
            _fake_result("low_dropout", af1=0.48),
        ]

        runner = AblationRunner(config, evaluator)
        dummy = np.zeros((10, 29), dtype=np.float32)
        dummy_y = np.zeros(10)

        results = runner.run({"head_lr": 0.001}, dummy, dummy_y, dummy, dummy_y)

        assert "baseline" in results
        assert len(results["variants"]) == 2

    def test_handles_failed_variant(self) -> None:
        config = _make_config()
        evaluator = MagicMock()
        evaluator.evaluate_ablation_variant.side_effect = [
            _fake_result("baseline", af1=0.50),
            RuntimeError("OOM"),
            _fake_result("low_dropout", af1=0.48),
        ]

        runner = AblationRunner(config, evaluator)
        dummy = np.zeros((10, 29), dtype=np.float32)
        dummy_y = np.zeros(10)

        results = runner.run({"head_lr": 0.001}, dummy, dummy_y, dummy, dummy_y)
        failed = [v for v in results["variants"] if v.get("status") == "failed"]
        assert len(failed) == 1
