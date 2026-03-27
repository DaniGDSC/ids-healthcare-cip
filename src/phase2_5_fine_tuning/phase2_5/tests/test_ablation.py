"""Tests for AblationRunner — component ablation studies."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.phase2_5_fine_tuning.phase2_5.ablation import AblationRunner
from src.phase2_5_fine_tuning.phase2_5.config import (
    AblationVariantConfig,
    Phase2_5Config,
)


def _make_config(**overrides: Any) -> Phase2_5Config:
    defaults: Dict[str, Any] = {
        "phase1_train": Path("data/processed/train_phase1.parquet"),
        "phase1_test": Path("data/processed/test_phase1.parquet"),
        "phase2_config": Path("config/phase2_config.yaml"),
        "ablation_variants": [
            AblationVariantConfig(
                name="no_attention",
                description="Remove attention",
                remove="attention",
            ),
            AblationVariantConfig(
                name="low_dropout",
                description="Low dropout",
                override={"dropout_rate": 0.1},
            ),
        ],
    }
    defaults.update(overrides)
    return Phase2_5Config(**defaults)


def _fake_result(variant: str, f1: float, auc: float = 0.95) -> Dict[str, Any]:
    return {
        "variant": variant,
        "metrics": {
            "accuracy": 0.9,
            "f1_score": f1,
            "precision": 0.9,
            "recall": 0.9,
            "auc_roc": auc,
        },
        "detection_params": 100,
        "total_params": 200,
        "epochs_run": 3,
        "final_val_loss": 0.1,
        "duration_seconds": 1.0,
    }


class TestAblationRunner:
    """Validate ablation study runner."""

    def test_runs_baseline_and_variants(self) -> None:
        config = _make_config()
        evaluator = MagicMock()

        # Baseline call
        evaluator.evaluate_config.return_value = _fake_result("baseline", f1=0.90)
        # Variant calls
        evaluator.evaluate_ablation_variant.side_effect = [
            _fake_result("no_attention", f1=0.82),
            _fake_result("low_dropout", f1=0.88),
        ]

        runner = AblationRunner(config, evaluator)
        base_hp = {"cnn_filters_1": 64, "timesteps": 20}
        dummy = np.zeros((10, 29), dtype=np.float32)
        dummy_y = np.zeros(10)

        results = runner.run(base_hp, dummy, dummy_y, dummy, dummy_y)

        assert "baseline" in results
        assert results["baseline"]["metrics"]["f1_score"] == 0.90
        assert len(results["variants"]) == 2

    def test_comparison_table_has_deltas(self) -> None:
        config = _make_config()
        evaluator = MagicMock()
        evaluator.evaluate_config.return_value = _fake_result("baseline", f1=0.90, auc=0.95)
        evaluator.evaluate_ablation_variant.side_effect = [
            _fake_result("no_attention", f1=0.82, auc=0.88),
            _fake_result("low_dropout", f1=0.88, auc=0.93),
        ]

        runner = AblationRunner(config, evaluator)
        dummy = np.zeros((10, 29), dtype=np.float32)
        dummy_y = np.zeros(10)

        results = runner.run({"timesteps": 20}, dummy, dummy_y, dummy, dummy_y)
        comparison = results["comparison"]

        # Baseline row has delta 0
        assert comparison[0]["delta_f1"] == 0.0
        # no_attention has negative delta
        assert comparison[1]["delta_f1"] < 0
        # All rows present
        assert len(comparison) == 3

    def test_handles_failed_variant(self) -> None:
        config = _make_config()
        evaluator = MagicMock()
        evaluator.evaluate_config.return_value = _fake_result("baseline", f1=0.90)
        evaluator.evaluate_ablation_variant.side_effect = [
            RuntimeError("OOM"),
            _fake_result("low_dropout", f1=0.88),
        ]

        runner = AblationRunner(config, evaluator)
        dummy = np.zeros((10, 29), dtype=np.float32)
        dummy_y = np.zeros(10)

        results = runner.run({"timesteps": 20}, dummy, dummy_y, dummy, dummy_y)

        failed = [v for v in results["variants"] if v.get("status") == "failed"]
        assert len(failed) == 1
        assert failed[0]["variant"] == "no_attention"
