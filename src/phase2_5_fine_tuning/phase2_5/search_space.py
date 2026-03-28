"""Hyperparameter search space — Optuna trial suggestion.

Simplified to 5 core parameters identified by importance analysis:
  - head_lr (65.7%)
  - cw_attack (24.7%)
  - head_epochs (4.8%)
  - ft_epochs (2.4%)
  - finetune_lr (2.4%)
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from .config import SearchSpaceConfig

logger = logging.getLogger(__name__)


class SearchSpace:
    """Suggest hyperparameters using Optuna trial objects.

    Args:
        space: Search space configuration.
        random_state: Seed (unused, kept for API compatibility).
    """

    def __init__(self, space: SearchSpaceConfig, random_state: int = 42) -> None:
        self._space = space

    def suggest(self, trial: Any) -> Dict[str, Any]:
        """Suggest hyperparameters using an Optuna trial.

        Args:
            trial: ``optuna.trial.Trial`` instance.

        Returns:
            Dict with the 5 core hyperparameters.
        """
        return {
            "head_lr": trial.suggest_float(
                "head_lr", self._space.head_lr_low, self._space.head_lr_high, log=True,
            ),
            "finetune_lr": trial.suggest_float(
                "finetune_lr", self._space.finetune_lr_low, self._space.finetune_lr_high, log=True,
            ),
            "cw_attack": trial.suggest_float(
                "cw_attack", self._space.cw_attack_low, self._space.cw_attack_high,
            ),
            "head_epochs": trial.suggest_categorical("head_epochs", self._space.head_epochs),
            "ft_epochs": trial.suggest_categorical("ft_epochs", self._space.ft_epochs),
        }
