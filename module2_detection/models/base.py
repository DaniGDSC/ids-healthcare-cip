"""Abstract base class for Track A detectors (XGB / RF / DT).

Centralises the SMOTE-in-CV training loop, threshold optimisation,
prediction, evaluation, and reporting that were previously triplicated
across ``XGBoost.py``, ``RandomForest.py``, and ``DecisionTree.py``.

Each subclass provides:
  - ``MODEL_NAME``  : human-readable label used in log lines
  - ``MODEL_TYPE``  : descriptive string written into ``get_report()``
  - ``PARAM_SPACE`` : RandomizedSearchCV distribution
  - ``DEFAULT_N_ITER`` : default search budget (was 25 / 40 / 50)
  - ``_make_classifier()`` : the configured sklearn estimator

The shared ``_build_pipeline`` template wraps ``_make_classifier`` in a
``SMOTE → classifier`` ``imblearn.Pipeline`` so the SMOTE configuration
is owned in exactly one place and cannot drift between subclasses. A
subclass that needs a fundamentally different pipeline shape may
override ``_build_pipeline`` directly.

The public surface (constructor signature, ``fit / predict / predict_proba /
evaluate / get_report``, properties ``best_params / optimal_threshold /
pipeline``) is preserved bit-identical to the pre-refactor classes so the
three tuning runners and any pickle artefacts continue to work.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, List

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator
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


class BaseDetector(ABC):
    """Shared SMOTE-in-CV detector implementation.

    Subclasses must define ``MODEL_NAME``, ``MODEL_TYPE``, ``PARAM_SPACE``,
    ``DEFAULT_N_ITER`` (class-level constants) and the ``_make_classifier``
    factory.  The shared ``_build_pipeline`` template assembles the
    SMOTE → classifier pipeline; override it only if a subclass needs a
    fundamentally different pipeline shape.
    """

    MODEL_NAME: ClassVar[str] = "BaseDetector"
    MODEL_TYPE: ClassVar[str] = "BaseDetector"
    PARAM_SPACE: ClassVar[Dict[str, List[Any]]] = {}
    DEFAULT_N_ITER: ClassVar[int] = 25

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

    @abstractmethod
    def _make_classifier(self) -> BaseEstimator:
        """Return a fresh, configured sklearn classifier instance.

        The returned estimator is wrapped by ``_build_pipeline`` into an
        ``imblearn.Pipeline`` whose first step is SMOTE.  Subclasses
        therefore only own the classifier-specific kwargs; the SMOTE
        configuration lives in one place on the base class.
        """

    def _build_pipeline(self) -> ImbPipeline:
        """Return a fresh ``SMOTE → classifier`` ``imblearn.Pipeline``.

        Template method: subclasses customise the pipeline by overriding
        ``_make_classifier``.  Override this method directly only when a
        subclass needs a fundamentally different pipeline shape.
        """
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
                ("classifier", self._make_classifier()),
            ]
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> "BaseDetector":
        """Run RandomizedSearchCV with SMOTE-in-CV, then OOF threshold tuning.

        Args:
            X_train: Scaled training features.
            y_train: Binary training labels (1 = attack).

        Returns:
            self
        """
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

        # Threshold via OUT-OF-FOLD probabilities (Phase 2 security review,
        # finding #2): resubstitution probas on tree-based models are pinned
        # to ~0/~1 and yield a meaningless F2-optimal threshold. cross_val_predict
        # gives the validation-fold view a production sample would actually face.
        oof_proba = cross_val_predict(
            self._build_pipeline(),
            X_train,
            y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=-1,
        )[:, 1]
        self._optimal_threshold = self._find_optimal_threshold(
            y_train,
            oof_proba,
        )

        elapsed = time.perf_counter() - t0
        self._cv_results["elapsed_seconds"] = round(elapsed, 1)

        logger.info(
            "%s fit: best CV %s=%.4f, threshold=%.3f, %.1fs",
            self.MODEL_NAME,
            self._scoring,
            search.best_score_,
            self._optimal_threshold,
            elapsed,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the OOF-tuned threshold."""
        proba = self.predict_proba(X)
        return (proba >= self._optimal_threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(attack) for each sample."""
        if self._best_pipeline is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._best_pipeline.predict_proba(X)[:, 1]

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate on a held-out test set.

        Args:
            X_test: Scaled test features.
            y_test: Binary test labels.

        Returns:
            Dict of attack_f1 / attack_f2 / weighted_f1 / macro_f1 / auc_roc /
            optimal_threshold.
        """
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
            self.MODEL_NAME,
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

    @staticmethod
    def _find_optimal_threshold(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        beta: float = 2.0,
        n_thresholds: int = 200,  # kept for API compatibility, unused
    ) -> float:
        """F-beta-optimal threshold via ``precision_recall_curve`` (O(N log N))."""
        return _find_optimal_threshold_shared(y_true, y_proba, beta=beta)

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
