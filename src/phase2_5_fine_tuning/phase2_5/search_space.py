"""Hyperparameter search space — generates candidate configurations.

Supports grid search (exhaustive), random search (sampled), and provides
Optuna trial suggestion for Bayesian TPE search.
"""

from __future__ import annotations

import itertools
import logging
from typing import Any, Dict, List, Union

import numpy as np

from .config import ContinuousRange, SearchSpaceConfig

logger = logging.getLogger(__name__)

# Parameters that use ContinuousRange for log-scale/float sampling
_CONTINUOUS_PARAMS = {"dropout_rate", "learning_rate", "phase_a_lr", "phase_b_lr", "phase_c_lr"}
# Parameters that are always categorical int
_CATEGORICAL_INT_PARAMS = {
    "cnn_filters_1", "cnn_filters_2", "cnn_kernel_size",
    "bilstm_units_1", "bilstm_units_2", "attention_units",
    "timesteps", "batch_size", "unfreezing_epochs",
}


class SearchSpace:
    """Generate hyperparameter configurations from a search space.

    Args:
        space: Search space configuration with value lists per parameter.
        random_state: Seed for reproducible random sampling.
    """

    def __init__(self, space: SearchSpaceConfig, random_state: int = 42) -> None:
        self._space = space
        self._rng = np.random.RandomState(random_state)

    def grid(self) -> List[Dict[str, Any]]:
        """Generate all combinations (Cartesian product).

        Only works with categorical (list) parameters.  Continuous ranges
        are discretised to their [low, mid, high] for grid mode.

        Returns:
            List of hyperparameter dicts.
        """
        param_dict = self._as_categorical_dict()
        keys = sorted(param_dict.keys())
        values = [param_dict[k] for k in keys]
        combos = list(itertools.product(*values))
        configs = [dict(zip(keys, combo)) for combo in combos]
        logger.info("Grid search: %d total combinations", len(configs))
        return configs

    def random(self, max_trials: int) -> List[Dict[str, Any]]:
        """Sample random configurations without replacement.

        Args:
            max_trials: Maximum number of configurations to sample.

        Returns:
            List of hyperparameter dicts (up to max_trials).
        """
        all_configs = self.grid()
        n = min(max_trials, len(all_configs))
        indices = self._rng.choice(len(all_configs), size=n, replace=False)
        sampled = [all_configs[i] for i in sorted(indices)]
        logger.info("Random search: sampled %d of %d combinations", n, len(all_configs))
        return sampled

    def suggest_optuna(self, trial: Any) -> Dict[str, Any]:
        """Suggest hyperparameters using an Optuna trial object.

        Supports ContinuousRange with log-scale and categorical lists.

        Args:
            trial: ``optuna.trial.Trial`` instance.

        Returns:
            Hyperparameter dict suggested by the trial.
        """
        hp: Dict[str, Any] = {}
        space_dict = self._as_raw_dict()

        for name, values in space_dict.items():
            if isinstance(values, ContinuousRange):
                if values.log:
                    hp[name] = trial.suggest_float(name, values.low, values.high, log=True)
                else:
                    hp[name] = trial.suggest_float(name, values.low, values.high)
            elif all(isinstance(v, int) for v in values):
                hp[name] = trial.suggest_categorical(name, values)
            else:
                hp[name] = trial.suggest_categorical(name, values)

        return hp

    def total_combinations(self) -> int:
        """Return total number of grid combinations."""
        param_dict = self._as_categorical_dict()
        total = 1
        for values in param_dict.values():
            total *= len(values)
        return total

    def _as_raw_dict(self) -> Dict[str, Union[List[Any], ContinuousRange]]:
        """Return raw space dict preserving ContinuousRange types."""
        return {
            "cnn_filters_1": self._space.cnn_filters_1,
            "cnn_filters_2": self._space.cnn_filters_2,
            "cnn_kernel_size": self._space.cnn_kernel_size,
            "bilstm_units_1": self._space.bilstm_units_1,
            "bilstm_units_2": self._space.bilstm_units_2,
            "dropout_rate": self._space.dropout_rate,
            "attention_units": self._space.attention_units,
            "timesteps": self._space.timesteps,
            "batch_size": self._space.batch_size,
            "learning_rate": self._space.learning_rate,
            "phase_a_lr": self._space.phase_a_lr,
            "phase_b_lr": self._space.phase_b_lr,
            "phase_c_lr": self._space.phase_c_lr,
            "unfreezing_epochs": self._space.unfreezing_epochs,
        }

    def _as_categorical_dict(self) -> Dict[str, List[Any]]:
        """Convert all parameters to categorical lists for grid/random.

        ContinuousRange is discretised to [low, midpoint, high].
        """
        raw = self._as_raw_dict()
        result: Dict[str, List[Any]] = {}
        for name, values in raw.items():
            if isinstance(values, ContinuousRange):
                mid = (values.low + values.high) / 2.0
                if values.log:
                    mid = (values.low * values.high) ** 0.5  # geometric mean
                result[name] = [round(values.low, 6), round(mid, 6), round(values.high, 6)]
            else:
                result[name] = list(values)
        return result
