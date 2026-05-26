"""Shared Track A detector base class.

Template-method pattern for the three Track A classifiers
(GradientBoosting, RandomForest, DecisionTree). Each concrete class
only needs to declare:

  - ``PARAM_SPACE``    — sklearn-style classifier__* search space
  - ``DEFAULT_N_ITER`` — sensible default for random search
  - ``MODEL_TYPE``     — string label for the report
  - ``_classifier()``  — fresh classifier instance (with random_state wired)

Everything else (SMOTE-in-CV pipeline, RandomizedSearchCV,
out-of-fold threshold optimisation, evaluate, report) lives here so a
fix in the shared logic propagates to all three detectors.

Replaces ~840 LOC of triplicated code across the previous
``XGBoost.py`` / ``RandomForest.py`` / ``DecisionTree.py`` classes.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, List

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import (
    classification_report,
    f1_score,
    fbeta_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
)

from ._threshold import find_optimal_threshold as _find_optimal_threshold_shared

logger = logging.getLogger(__name__)


class BaseTrackADetector(ABC):
    """Shared CV-tuning + threshold logic for Track A detectors.

    Args:
        n_iter:          Number of random parameter samples for search.
                         Defaults to ``cls.DEFAULT_N_ITER``.
        cv_folds:        Stratified CV folds.
        scoring:         Metric for RandomizedSearchCV.
        smote_strategy:  SMOTE sampling strategy ("auto" for 1:1).
        smote_k:         SMOTE k-neighbors.
        random_state:    Seed for reproducibility.
    """

    PARAM_SPACE: ClassVar[Dict[str, List[Any]]] = {}
    DEFAULT_N_ITER: ClassVar[int] = 50
    MODEL_TYPE: ClassVar[str] = "BaseTrackADetector"
    LOG_NAME: ClassVar[str] = "TrackA"

    def __init__(
        self,
        n_iter: int | None = None,
        cv_folds: int = 5,
        scoring: str = "f1_weighted",
        smote_strategy: str = "auto",
        smote_k: int = 5,
        random_state: int = 42,
    ) -> None:
        self._n_iter = n_iter if n_iter is not None else self.DEFAULT_N_ITER
        self._cv_folds = cv_folds
        self._scoring = scoring
        self._smote_strategy = smote_strategy
        self._smote_k = smote_k
        self._random_state = random_state

        self._best_pipeline: ImbPipeline | None = None
        self._best_params: Dict[str, Any] = {}
        self._cv_results: Dict[str, Any] = {}
        self._optimal_threshold: float = 0.5
        self._test_metrics: Dict[str, float] = {}

    # ── subclass extension point ───────────────────────────────────────

    @abstractmethod
    def _classifier(self) -> Any:
        """Instantiate a fresh sklearn classifier with random_state wired in."""

    # ── pipeline ───────────────────────────────────────────────────────

    def _build_pipeline(self) -> ImbPipeline:
        return ImbPipeline(
            [
                (
                    "smote",
                    SMOTE(
                        sampling_strategy=self._smote_strategy,
                        k_neighbors=self._smote_k,
                        random_state=self._random_state,
                    ),
                ),
                ("classifier", self._classifier()),
            ]
        )

    # ── train ──────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> BaseTrackADetector:
        """Run RandomizedSearchCV with SMOTE-in-CV, then OOF threshold opt."""
        t0 = time.perf_counter()

        pipeline = self._build_pipeline()
        cv = StratifiedKFold(
            n_splits=self._cv_folds,
            shuffle=True,
            random_state=self._random_state,
        )

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=self.PARAM_SPACE,
            n_iter=self._n_iter,
            cv=cv,
            scoring=self._scoring,
            random_state=self._random_state,
            n_jobs=-1,
            verbose=1,
            return_train_score=True,
        )
        search.fit(X_train, y_train)

        self._best_pipeline = search.best_estimator_
        self._best_params = search.best_params_
        self._cv_results = {
            "best_score": float(search.best_score_),
            "best_rank": int(search.best_index_) + 1,
            "n_candidates": self._n_iter,
            "n_folds": self._cv_folds,
        }

        # ── Optimal threshold via OUT-OF-FOLD probabilities ──
        # See finding #2 — resubstitution probas on training rows are
        # pinned to ~0/~1 for boosting/bagging models, making any
        # threshold optimised against them meaningless. cross_val_predict
        # gives the validation-fold probability distribution an unseen
        # sample would actually receive, which is what we should
        # optimise the threshold against.
        oof_proba = cross_val_predict(
            self._build_pipeline(),
            X_train,
            y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=-1,
        )[:, 1]
        self._optimal_threshold = self._find_optimal_threshold(y_train, oof_proba)

        elapsed = time.perf_counter() - t0
        self._cv_results["elapsed_seconds"] = round(elapsed, 1)

        logger.info(
            "%s fit: best CV %s=%.4f, threshold=%.3f, %.1fs",
            self.LOG_NAME,
            self._scoring,
            search.best_score_,
            self._optimal_threshold,
            elapsed,
        )
        return self

    # ── predict ────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba >= self._optimal_threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._best_pipeline is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._best_pipeline.predict_proba(X)[:, 1]

    # ── evaluate ───────────────────────────────────────────────────────

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)

        metrics = {
            "attack_f1": float(f1_score(y_test, y_pred, pos_label=1)),
            "attack_f2": float(fbeta_score(y_test, y_pred, beta=2, pos_label=1)),
            "weighted_f1": float(f1_score(y_test, y_pred, average="weighted")),
            "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
            "auc_roc": (
                float(roc_auc_score(y_test, y_proba))
                if len(np.unique(y_test)) > 1
                else float("nan")
            ),
            "optimal_threshold": self._optimal_threshold,
        }
        self._test_metrics = metrics

        logger.info(
            "%s eval: attack_f1=%.4f, attack_f2=%.4f, AUC=%.4f",
            self.LOG_NAME,
            metrics["attack_f1"],
            metrics["attack_f2"],
            metrics["auc_roc"],
        )
        logger.info(
            "\n%s",
            classification_report(
                y_test,
                y_pred,
                target_names=["Normal", "Attack"],
                digits=4,
            ),
        )
        return metrics

    # ── threshold ──────────────────────────────────────────────────────

    @staticmethod
    def _find_optimal_threshold(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        beta: float = 2.0,
    ) -> float:
        """Delegate to the precision_recall_curve-based shared utility."""
        return _find_optimal_threshold_shared(y_true, y_proba, beta=beta)

    # ── report / properties ────────────────────────────────────────────

    def get_report(self) -> Dict[str, Any]:
        return {
            "model_type": self.MODEL_TYPE,
            "best_params": self._best_params,
            "cv_results": self._cv_results,
            "optimal_threshold": self._optimal_threshold,
            "test_metrics": self._test_metrics,
        }

    @property
    def best_params(self) -> Dict[str, Any]:
        return dict(self._best_params)

    @property
    def optimal_threshold(self) -> float:
        return self._optimal_threshold

    @property
    def pipeline(self) -> ImbPipeline | None:
        return self._best_pipeline
