"""Random Forest detection backbone for IoMT intrusion detection.

Supports:
  - RandomizedSearchCV with literature-backed hyperparameter space
  - SMOTE inside the CV pipeline (imblearn.Pipeline)
  - class_weight='balanced' for residual imbalance handling
  - Threshold optimization on attack-class F2
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List

from sklearn.ensemble import RandomForestClassifier

from .base import BaseDetector

# Literature-backed hyperparameter search space
PARAM_SPACE: Dict[str, List[Any]] = {
    "classifier__n_estimators": [100, 200, 300, 500, 700],
    "classifier__max_depth": [10, 15, 20, 30, None],
    "classifier__min_samples_split": [2, 5, 10, 20],
    "classifier__min_samples_leaf": [1, 2, 4, 8],
    "classifier__max_features": ["sqrt", "log2", 0.3, 0.5],
    "classifier__class_weight": ["balanced", "balanced_subsample", None],
    "classifier__criterion": ["gini", "entropy"],
}


class RandomForestDetector(BaseDetector):
    """Random Forest detector with SMOTE-in-CV.

    Args:
        n_iter: Number of random parameter samples for search.
        cv_folds: Stratified CV folds.
        scoring: Metric for RandomizedSearchCV.
        smote_strategy: SMOTE sampling strategy ("auto" for 1:1).
        smote_k: SMOTE k-neighbors.
        random_state: Seed for reproducibility.
    """

    MODEL_NAME: ClassVar[str] = "RandomForest"
    MODEL_TYPE: ClassVar[str] = "RandomForestClassifier"
    PARAM_SPACE: ClassVar[Dict[str, List[Any]]] = PARAM_SPACE
    DEFAULT_N_ITER: ClassVar[int] = 40

    def _make_classifier(self) -> RandomForestClassifier:
        return RandomForestClassifier(
            bootstrap=True,
            random_state=self._random_state,
            n_jobs=-1,
        )
