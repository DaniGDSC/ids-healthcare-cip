"""Tests for SearchSpace — Optuna suggestion."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.phase2_5_fine_tuning.phase2_5.config import SearchSpaceConfig
from src.phase2_5_fine_tuning.phase2_5.search_space import SearchSpace


class TestSearchSpace:
    def test_suggest_calls_trial_methods(self) -> None:
        space = SearchSpace(SearchSpaceConfig())
        trial = MagicMock()
        trial.suggest_float.return_value = 0.001
        trial.suggest_categorical.return_value = 5

        hp = space.suggest(trial)

        assert "head_lr" in hp
        assert "finetune_lr" in hp
        assert "cw_attack" in hp
        assert "head_epochs" in hp
        assert "ft_epochs" in hp
        assert len(hp) == 5

    def test_suggest_float_log_scale_for_lr(self) -> None:
        space = SearchSpace(SearchSpaceConfig())
        trial = MagicMock()
        trial.suggest_float.return_value = 0.001
        trial.suggest_categorical.return_value = 5

        space.suggest(trial)

        # head_lr and finetune_lr should use log=True
        float_calls = trial.suggest_float.call_args_list
        for call in float_calls:
            if call.args[0] in ("head_lr", "finetune_lr"):
                assert call.kwargs.get("log") is True

    def test_suggest_categorical_for_epochs(self) -> None:
        space = SearchSpace(SearchSpaceConfig(head_epochs=[3, 7], ft_epochs=[2, 4]))
        trial = MagicMock()
        trial.suggest_float.return_value = 0.001
        trial.suggest_categorical.return_value = 3

        space.suggest(trial)

        cat_calls = trial.suggest_categorical.call_args_list
        cat_names = [c.args[0] for c in cat_calls]
        assert "head_epochs" in cat_names
        assert "ft_epochs" in cat_names

    def test_custom_ranges(self) -> None:
        space = SearchSpace(SearchSpaceConfig(head_lr_low=0.01, head_lr_high=0.1))
        trial = MagicMock()
        trial.suggest_float.return_value = 0.05
        trial.suggest_categorical.return_value = 5

        space.suggest(trial)

        # Verify head_lr range
        head_lr_call = [c for c in trial.suggest_float.call_args_list if c.args[0] == "head_lr"][0]
        assert head_lr_call.args[1] == 0.01
        assert head_lr_call.args[2] == 0.1
