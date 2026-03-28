"""Tests for HyperparameterTuner — Bayesian TPE search."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.phase2_5_fine_tuning.phase2_5.config import Phase2_5Config
from src.phase2_5_fine_tuning.phase2_5.tuner import HyperparameterTuner


def _make_config(**overrides: Any) -> Phase2_5Config:
    defaults: Dict[str, Any] = {
        "phase1_train": Path("data/processed/train_phase1.parquet"),
        "phase1_test": Path("data/processed/test_phase1.parquet"),
        "phase2_config": Path("config/phase2_config.yaml"),
        "max_trials": 2,
        "search_metric": "attack_f1",
        "search_direction": "maximize",
    }
    defaults.update(overrides)
    return Phase2_5Config(**defaults)


def _fake_result(hp: Dict, af1: float) -> Dict[str, Any]:
    return {
        "hyperparameters": hp,
        "metrics": {
            "accuracy": 0.9, "f1_score": 0.9, "auc_roc": 0.95,
            "attack_recall": 0.5, "attack_precision": 0.5, "attack_f1": af1,
            "macro_f1": 0.7, "threshold": 0.3,
        },
        "optimal_threshold": 0.3,
        "total_params": 200,
        "duration_seconds": 1.0,
    }


class TestHyperparameterTuner:
    def test_selects_best_maximize(self) -> None:
        config = _make_config(max_trials=2)
        evaluator = MagicMock()
        evaluator.evaluate_two_stage.side_effect = [
            _fake_result({"head_lr": 0.001}, af1=0.35),
            _fake_result({"head_lr": 0.003}, af1=0.50),
        ]
        space = MagicMock()
        space.suggest.side_effect = [{"head_lr": 0.001}, {"head_lr": 0.003}]

        tuner = HyperparameterTuner(config, evaluator, space)
        dummy = np.zeros((10, 29), dtype=np.float32)
        dummy_y = np.zeros(10)

        results = tuner.run(dummy, dummy_y, dummy, dummy_y, dummy, dummy_y)

        assert results["best_score"] == 0.50
        assert results["completed_trials"] == 2

    def test_all_failed_raises(self) -> None:
        config = _make_config(max_trials=1)
        evaluator = MagicMock()
        evaluator.evaluate_two_stage.side_effect = RuntimeError("fail")
        space = MagicMock()
        space.suggest.return_value = {"head_lr": 0.001}

        tuner = HyperparameterTuner(config, evaluator, space)
        dummy = np.zeros((10, 29), dtype=np.float32)
        dummy_y = np.zeros(10)

        # Optuna catches and re-raises the original error
        with pytest.raises(RuntimeError):
            tuner.run(dummy, dummy_y, dummy, dummy_y, dummy, dummy_y)

    def test_optuna_study_stored(self) -> None:
        config = _make_config(max_trials=1)
        evaluator = MagicMock()
        evaluator.evaluate_two_stage.return_value = _fake_result({"head_lr": 0.001}, af1=0.4)
        space = MagicMock()
        space.suggest.return_value = {"head_lr": 0.001}

        tuner = HyperparameterTuner(config, evaluator, space)
        dummy = np.zeros((10, 29), dtype=np.float32)
        dummy_y = np.zeros(10)

        tuner.run(dummy, dummy_y, dummy, dummy_y, dummy, dummy_y)
        assert tuner.optuna_study is not None
