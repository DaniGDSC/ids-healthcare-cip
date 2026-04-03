"""Decision Tree detection backbone for IoMT intrusion detection.

Supports:
  - RandomizedSearchCV with literature-backed hyperparameter space
  - SMOTE inside the CV pipeline (imblearn.Pipeline)
  - class_weight='balanced' for residual imbalance handling
  - Threshold optimization on attack-class F2

References:
  Nzuva et al. (2024) — DT achieved F1=0.950 on CIC-IDS2017
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import (
    classification_report,
    f1_score,
    fbeta_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger(__name__)

# ── Literature-backed hyperparameter search space ────────────────────────
PARAM_SPACE: Dict[str, List[Any]] = {
    "classifier__max_depth": [3, 5, 7, 10, 15, None],
    "classifier__min_samples_split": [2, 5, 10, 20, 50],
    "classifier__min_samples_leaf": [1, 2, 5, 10, 20],
    "classifier__max_features": ["sqrt", "log2", None],
    "classifier__criterion": ["gini", "entropy"],
    "classifier__class_weight": ["balanced", None],
    "classifier__splitter": ["best", "random"],
}


class DecisionTreeDetector:
    """Decision Tree detector with SMOTE-in-CV.

    Args:
        n_iter: Number of random parameter samples for search.
        cv_folds: Stratified CV folds.
        scoring: Metric for RandomizedSearchCV.
        smote_strategy: SMOTE sampling strategy ("auto" for 1:1).
        smote_k: SMOTE k-neighbors.
        random_state: Seed for reproducibility.
    """

    def __init__(
        self,
        n_iter: int = 25,
        cv_folds: int = 5,
        scoring: str = "f1_weighted",
        smote_strategy: str = "auto",
        smote_k: int = 5,
        random_state: int = 42,
    ) -> None:
        self._n_iter = n_iter
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

    # ------------------------------------------------------------------
    # Build pipeline
    # ------------------------------------------------------------------

    def _build_pipeline(self) -> ImbPipeline:
        """SMOTE + DecisionTree inside an imblearn pipeline."""
        return ImbPipeline([
            ("smote", SMOTE(
                sampling_strategy=self._smote_strategy,
                k_neighbors=self._smote_k,
                random_state=self._random_state,
            )),
            ("classifier", DecisionTreeClassifier(
                random_state=self._random_state,
            )),
        ])

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> DecisionTreeDetector:
        """Run RandomizedSearchCV with SMOTE-in-CV.

        Args:
            X_train: Scaled training features.
            y_train: Binary training labels.

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
            param_distributions=PARAM_SPACE,
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

        # Optimal threshold on training predictions
        y_proba = search.best_estimator_.predict_proba(X_train)[:, 1]
        self._optimal_threshold = self._find_optimal_threshold(y_train, y_proba)

        elapsed = time.perf_counter() - t0
        self._cv_results["elapsed_seconds"] = round(elapsed, 1)

        logger.info(
            "DecisionTree fit: best CV %s=%.4f, threshold=%.3f, %.1fs",
            self._scoring, search.best_score_,
            self._optimal_threshold, elapsed,
        )
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using optimal threshold."""
        proba = self.predict_proba(X)
        return (proba >= self._optimal_threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(attack) for each sample."""
        if self._best_pipeline is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._best_pipeline.predict_proba(X)[:, 1]

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate on held-out test set.

        Args:
            X_test: Scaled test features.
            y_test: Binary test labels.

        Returns:
            Dict of evaluation metrics.
        """
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)

        metrics = {
            "attack_f1": float(f1_score(y_test, y_pred, pos_label=1)),
            "attack_f2": float(fbeta_score(y_test, y_pred, beta=2, pos_label=1)),
            "weighted_f1": float(f1_score(y_test, y_pred, average="weighted")),
            "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
            "auc_roc": float(roc_auc_score(y_test, y_proba)) if len(np.unique(y_test)) > 1 else float("nan"),
            "optimal_threshold": self._optimal_threshold,
        }
        self._test_metrics = metrics

        logger.info(
            "DecisionTree eval: attack_f1=%.4f, attack_f2=%.4f, AUC=%.4f",
            metrics["attack_f1"], metrics["attack_f2"], metrics["auc_roc"],
        )
        logger.info("\n%s", classification_report(
            y_test, y_pred, target_names=["Normal", "Attack"], digits=4,
        ))
        return metrics

    # ------------------------------------------------------------------
    # Threshold optimization
    # ------------------------------------------------------------------

    @staticmethod
    def _find_optimal_threshold(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        beta: float = 2.0,
        n_thresholds: int = 200,
    ) -> float:
        """Find threshold that maximizes F-beta on attack class."""
        thresholds = np.linspace(0.05, 0.95, n_thresholds)
        best_score = 0.0
        best_t = 0.5
        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            score = fbeta_score(y_true, y_pred, beta=beta, pos_label=1)
            if score > best_score:
                best_score = score
                best_t = float(t)
        return best_t

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def get_report(self) -> Dict[str, Any]:
        return {
            "model_type": "DecisionTreeClassifier",
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
