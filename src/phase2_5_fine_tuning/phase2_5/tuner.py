"""Hyperparameter tuner — Bayesian TPE search via Optuna.

Orchestrates the two-stage training strategy:
  Stage 1: Train head on SMOTE balanced data (frozen backbone)
  Stage 2: Fine-tune attention+BiLSTM2+head on imbalanced data with class_weight

Selects the best configuration by attack_f1 on the imbalanced test set.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

from .config import Phase2_5Config
from .evaluator import QuickEvaluator
from .search_space import SearchSpace

logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """Bayesian TPE hyperparameter search for the two-stage training strategy.

    Args:
        config: Phase 2.5 configuration.
        evaluator: QuickEvaluator for fast train+eval.
        search_space: SearchSpace for Optuna suggestion.
    """

    def __init__(
        self,
        config: Phase2_5Config,
        evaluator: QuickEvaluator,
        search_space: SearchSpace,
    ) -> None:
        self._config = config
        self._evaluator = evaluator
        self._search_space = search_space
        self._optuna_study = None

    def run(
        self,
        X_train_smote: np.ndarray,
        y_train_smote: np.ndarray,
        X_train_orig: np.ndarray,
        y_train_orig: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """Execute Bayesian TPE search.

        Args:
            X_train_smote: SMOTE-balanced training features (for head training).
            y_train_smote: SMOTE-balanced training labels.
            X_train_orig: Original imbalanced training features (for fine-tuning).
            y_train_orig: Original imbalanced training labels.
            X_test: Test features (imbalanced, for attack_f1 evaluation).
            y_test: Test labels.

        Returns:
            Dict with all trial results, best config, and summary.
        """
        import optuna

        metric = self._config.search_metric
        direction = self._config.search_direction
        max_trials = self._config.max_trials

        logger.info("═══ Bayesian TPE Search (Optuna) ═══")
        logger.info("  Metric:     %s (%s)", metric, direction)
        logger.info("  Max trials: %d", max_trials)

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=self._config.random_state),
            study_name="phase2_5_tuning",
        )

        evaluator = self._evaluator
        search_space = self._search_space

        def objective(trial: optuna.Trial) -> float:
            hp = search_space.suggest(trial)

            result = evaluator.evaluate_two_stage(
                hp, X_train_smote, y_train_smote,
                X_train_orig, y_train_orig, X_test, y_test,
            )

            trial.set_user_attr("result", result)
            return result["metrics"][metric]

        study.optimize(objective, n_trials=max_trials)
        self._optuna_study = study

        # Collect results
        trials: List[Dict[str, Any]] = []
        for i, trial in enumerate(study.trials):
            if trial.state == optuna.trial.TrialState.COMPLETE:
                result = trial.user_attrs.get("result", {})
                result["trial_index"] = i
                result["status"] = "completed"
                trials.append(result)
            else:
                trials.append({
                    "trial_index": i,
                    "hyperparameters": trial.params,
                    "status": "failed",
                    "error": str(trial.user_attrs.get("error", "unknown")),
                })

        completed = [t for t in trials if t.get("status") == "completed"]
        if not completed:
            raise RuntimeError("All trials failed — no valid configuration found")

        if direction == "maximize":
            best = max(completed, key=lambda t: t["metrics"][metric])
        else:
            best = min(completed, key=lambda t: t["metrics"][metric])

        best_score = best["metrics"][metric]
        logger.info("═══ Best: trial %d, %s=%.4f ═══", best["trial_index"], metric, best_score)

        return {
            "strategy": "bayesian_tpe",
            "metric": metric,
            "direction": direction,
            "total_trials": max_trials,
            "completed_trials": len(completed),
            "failed_trials": len(trials) - len(completed),
            "trials": trials,
            "best_trial_index": best["trial_index"],
            "best_config": best["hyperparameters"],
            "best_metrics": best["metrics"],
            "best_score": best_score,
        }

    @property
    def optuna_study(self) -> Any:
        """Return the Optuna study object for importance analysis."""
        return self._optuna_study
