"""Tests for HyperparameterTuner — search orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.phase2_5_fine_tuning.phase2_5.config import Phase2_5Config, SearchSpaceConfig
from src.phase2_5_fine_tuning.phase2_5.search_space import SearchSpace
from src.phase2_5_fine_tuning.phase2_5.tuner import HyperparameterTuner


def _make_config(**overrides: Any) -> Phase2_5Config:
    defaults: Dict[str, Any] = {
        "phase1_train": Path("data/processed/train_phase1.parquet"),
        "phase1_test": Path("data/processed/test_phase1.parquet"),
        "phase2_config": Path("config/phase2_config.yaml"),
        "search_strategy": "random",
        "max_trials": 2,
        "search_metric": "f1_score",
        "search_direction": "maximize",
    }
    defaults.update(overrides)
    return Phase2_5Config(**defaults)


def _fake_eval_result(hp: Dict[str, Any], f1: float) -> Dict[str, Any]:
    return {
        "hyperparameters": hp,
        "metrics": {
            "accuracy": 0.9,
            "f1_score": f1,
            "precision": 0.9,
            "recall": 0.9,
            "auc_roc": 0.95,
        },
        "detection_params": 100,
        "total_params": 200,
        "epochs_run": 3,
        "final_val_loss": 0.1,
        "duration_seconds": 1.0,
    }


class TestHyperparameterTuner:
    """Validate tuner orchestration and result selection."""

    def test_selects_best_maximize(self) -> None:
        config = _make_config(max_trials=2)
        evaluator = MagicMock()
        evaluator.evaluate_config.side_effect = [
            _fake_eval_result({"lr": 0.001}, f1=0.85),
            _fake_eval_result({"lr": 0.0001}, f1=0.92),
        ]

        small_space = SearchSpaceConfig(
            cnn_filters_1=[64],
            cnn_filters_2=[128],
            cnn_kernel_size=[3],
            bilstm_units_1=[128],
            bilstm_units_2=[64],
            dropout_rate=[0.3],
            attention_units=[128],
            timesteps=[20],
            batch_size=[256],
            learning_rate=[0.001, 0.0001],
        )
        space = SearchSpace(small_space, random_state=42)

        tuner = HyperparameterTuner(config, evaluator, space)
        dummy = np.zeros((10, 29), dtype=np.float32)
        dummy_y = np.zeros(10)

        results = tuner.run(dummy, dummy_y, dummy, dummy_y)

        assert results["best_score"] == 0.92
        assert results["completed_trials"] == 2
        assert results["failed_trials"] == 0

    def test_handles_failed_trial(self) -> None:
        config = _make_config(max_trials=2)
        evaluator = MagicMock()
        evaluator.evaluate_config.side_effect = [
            RuntimeError("OOM"),
            _fake_eval_result({"lr": 0.001}, f1=0.80),
        ]

        small_space = SearchSpaceConfig(
            cnn_filters_1=[64],
            cnn_filters_2=[128],
            cnn_kernel_size=[3],
            bilstm_units_1=[128],
            bilstm_units_2=[64],
            dropout_rate=[0.3],
            attention_units=[128],
            timesteps=[20],
            batch_size=[256],
            learning_rate=[0.001, 0.0001],
        )
        space = SearchSpace(small_space, random_state=42)

        tuner = HyperparameterTuner(config, evaluator, space)
        dummy = np.zeros((10, 29), dtype=np.float32)
        dummy_y = np.zeros(10)

        results = tuner.run(dummy, dummy_y, dummy, dummy_y)

        assert results["completed_trials"] == 1
        assert results["failed_trials"] == 1
        assert results["best_score"] == 0.80

    def test_all_failed_raises(self) -> None:
        config = _make_config(max_trials=1)
        evaluator = MagicMock()
        evaluator.evaluate_config.side_effect = RuntimeError("fail")

        small_space = SearchSpaceConfig(
            cnn_filters_1=[64],
            cnn_filters_2=[128],
            cnn_kernel_size=[3],
            bilstm_units_1=[128],
            bilstm_units_2=[64],
            dropout_rate=[0.3],
            attention_units=[128],
            timesteps=[20],
            batch_size=[256],
            learning_rate=[0.001],
        )
        space = SearchSpace(small_space, random_state=42)

        tuner = HyperparameterTuner(config, evaluator, space)
        dummy = np.zeros((10, 29), dtype=np.float32)
        dummy_y = np.zeros(10)

        with pytest.raises(RuntimeError, match="All trials failed"):
            tuner.run(dummy, dummy_y, dummy, dummy_y)

    def test_minimize_direction(self) -> None:
        config = _make_config(
            max_trials=2,
            search_metric="f1_score",
            search_direction="minimize",
        )
        evaluator = MagicMock()
        evaluator.evaluate_config.side_effect = [
            _fake_eval_result({"lr": 0.001}, f1=0.85),
            _fake_eval_result({"lr": 0.0001}, f1=0.70),
        ]

        small_space = SearchSpaceConfig(
            cnn_filters_1=[64],
            cnn_filters_2=[128],
            cnn_kernel_size=[3],
            bilstm_units_1=[128],
            bilstm_units_2=[64],
            dropout_rate=[0.3],
            attention_units=[128],
            timesteps=[20],
            batch_size=[256],
            learning_rate=[0.001, 0.0001],
        )
        space = SearchSpace(small_space, random_state=42)

        tuner = HyperparameterTuner(config, evaluator, space)
        dummy = np.zeros((10, 29), dtype=np.float32)
        dummy_y = np.zeros(10)

        results = tuner.run(dummy, dummy_y, dummy, dummy_y)
        assert results["best_score"] == 0.70
