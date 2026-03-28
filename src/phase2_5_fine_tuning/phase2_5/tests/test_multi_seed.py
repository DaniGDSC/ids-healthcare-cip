"""Tests for MultiSeedValidator — confidence interval computation."""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock

import numpy as np

from src.phase2_5_fine_tuning.phase2_5.config import MultiSeedConfig
from src.phase2_5_fine_tuning.phase2_5.multi_seed import MultiSeedValidator


def _fake_result(f1: float, seed: int) -> Dict[str, Any]:
    return {
        "hyperparameters": {"lr": 0.001},
        "metrics": {
            "accuracy": 0.9,
            "f1_score": f1,
            "precision": 0.9,
            "recall": 0.9,
            "auc_roc": 0.95,
        },
        "detection_params": 100,
        "total_params": 200,
        "epochs_run": 5,
        "final_val_loss": 0.1,
        "duration_seconds": 1.0,
    }


class TestMultiSeedValidator:
    """Validate multi-seed confidence intervals."""

    def test_disabled_returns_empty(self) -> None:
        config = MultiSeedConfig(enabled=False)
        evaluator = MagicMock()
        validator = MultiSeedValidator(config, evaluator)

        result = validator.validate(
            {"metric": "f1_score", "direction": "maximize", "trials": []},
            np.zeros((10, 5)), np.zeros(10),
            np.zeros((5, 5)), np.zeros(5),
        )
        assert result["enabled"] is False
        assert result["configs"] == []

    def test_validates_top_k(self) -> None:
        config = MultiSeedConfig(enabled=True, top_k=1, seeds=[42, 99], full_epochs=5)
        evaluator = MagicMock()
        evaluator.evaluate_config.side_effect = [
            _fake_result(f1=0.91, seed=42),
            _fake_result(f1=0.89, seed=99),
        ]
        validator = MultiSeedValidator(config, evaluator)

        tuning_results = {
            "metric": "f1_score",
            "direction": "maximize",
            "trials": [
                {
                    "status": "completed",
                    "hyperparameters": {"lr": 0.001},
                    "metrics": {"f1_score": 0.90},
                },
            ],
        }
        dummy = np.zeros((10, 5), dtype=np.float32)
        dummy_y = np.zeros(10)

        result = validator.validate(tuning_results, dummy, dummy_y, dummy, dummy_y)

        assert result["enabled"] is True
        assert len(result["configs"]) == 1
        stats = result["configs"][0]["statistics"]
        assert stats["n_seeds"] == 2
        assert 0.89 <= stats["mean"] <= 0.91

    def test_handles_seed_failure(self) -> None:
        config = MultiSeedConfig(enabled=True, top_k=1, seeds=[42, 99], full_epochs=5)
        evaluator = MagicMock()
        evaluator.evaluate_config.side_effect = [
            _fake_result(f1=0.90, seed=42),
            RuntimeError("OOM"),
        ]
        validator = MultiSeedValidator(config, evaluator)

        tuning_results = {
            "metric": "f1_score",
            "direction": "maximize",
            "trials": [
                {
                    "status": "completed",
                    "hyperparameters": {"lr": 0.001},
                    "metrics": {"f1_score": 0.90},
                },
            ],
        }
        dummy = np.zeros((10, 5), dtype=np.float32)
        dummy_y = np.zeros(10)

        result = validator.validate(tuning_results, dummy, dummy_y, dummy, dummy_y)

        seed_results = result["configs"][0]["seed_results"]
        assert seed_results[0]["status"] == "completed"
        assert seed_results[1]["status"] == "failed"
        assert result["configs"][0]["statistics"]["n_seeds"] == 1
