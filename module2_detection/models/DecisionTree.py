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

from typing import Any, ClassVar, Dict, List

from sklearn.tree import DecisionTreeClassifier

from .base import BaseDetector

# Literature-backed hyperparameter search space
PARAM_SPACE: Dict[str, List[Any]] = {
    "classifier__max_depth": [3, 5, 7, 10, 15, None],
    "classifier__min_samples_split": [2, 5, 10, 20, 50],
    "classifier__min_samples_leaf": [1, 2, 5, 10, 20],
    "classifier__max_features": ["sqrt", "log2", None],
    "classifier__criterion": ["gini", "entropy"],
    "classifier__class_weight": ["balanced", None],
    "classifier__splitter": ["best", "random"],
}


class DecisionTreeDetector(BaseDetector):
    """Decision Tree detector with SMOTE-in-CV.

    Args:
        n_iter: Number of random parameter samples for search.
        cv_folds: Stratified CV folds.
        scoring: Metric for RandomizedSearchCV.
        smote_strategy: SMOTE sampling strategy ("auto" for 1:1).
        smote_k: SMOTE k-neighbors.
        random_state: Seed for reproducibility.
    """

    MODEL_NAME: ClassVar[str] = "DecisionTree"
    MODEL_TYPE: ClassVar[str] = "DecisionTreeClassifier"
    PARAM_SPACE: ClassVar[Dict[str, List[Any]]] = PARAM_SPACE
    DEFAULT_N_ITER: ClassVar[int] = 25

    def _make_classifier(self) -> DecisionTreeClassifier:
        return DecisionTreeClassifier(random_state=self._random_state)
