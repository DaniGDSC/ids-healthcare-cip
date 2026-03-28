"""Tests for parameter importance analysis."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.phase2_5_fine_tuning.phase2_5.importance import (
    compute_importance,
    compute_importance_grouped,
)


def _make_trials():
    """Create fake trial results for importance testing."""
    return [
        {
            "status": "completed",
            "hyperparameters": {"lr": 0.001, "dropout": 0.3, "filters": 64},
            "metrics": {"f1_score": 0.90},
        },
        {
            "status": "completed",
            "hyperparameters": {"lr": 0.001, "dropout": 0.5, "filters": 64},
            "metrics": {"f1_score": 0.85},
        },
        {
            "status": "completed",
            "hyperparameters": {"lr": 0.0001, "dropout": 0.3, "filters": 128},
            "metrics": {"f1_score": 0.92},
        },
        {
            "status": "completed",
            "hyperparameters": {"lr": 0.0001, "dropout": 0.5, "filters": 128},
            "metrics": {"f1_score": 0.88},
        },
        {
            "status": "failed",
            "hyperparameters": {"lr": 0.01, "dropout": 0.1, "filters": 32},
        },
    ]


class TestGroupedImportance:
    """Validate grouped variance importance analysis."""

    def test_returns_all_params(self) -> None:
        result = compute_importance_grouped(_make_trials(), "f1_score")
        assert "lr" in result
        assert "dropout" in result
        assert "filters" in result

    def test_importances_sum_to_one(self) -> None:
        result = compute_importance_grouped(_make_trials(), "f1_score")
        total = sum(result.values())
        assert abs(total - 1.0) < 1e-6

    def test_sorted_descending(self) -> None:
        result = compute_importance_grouped(_make_trials(), "f1_score")
        scores = list(result.values())
        assert scores == sorted(scores, reverse=True)

    def test_too_few_trials(self) -> None:
        result = compute_importance_grouped(
            [{"status": "completed", "hyperparameters": {"x": 1}, "metrics": {"f1_score": 0.9}}],
            "f1_score",
        )
        assert result == {}


class TestComputeImportance:
    """Validate the combined importance function."""

    def test_uses_grouped_without_study(self) -> None:
        tuner = MagicMock()
        tuner.optuna_study = None
        result = compute_importance(tuner, _make_trials(), "f1_score")
        assert result["method"] == "grouped_variance"
        assert "importances" in result
        assert len(result["importances"]) > 0
