"""Decision Tree detection backbone for IoMT intrusion detection.

Thin subclass of ``BaseTrackADetector`` — only the classifier choice
and the hyperparameter search space differ from the shared base.

References:
  Nzuva et al. (2024) — DT achieved F1=0.950 on CIC-IDS2017
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List

from sklearn.tree import DecisionTreeClassifier

from ._base_detector import BaseTrackADetector

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


class DecisionTreeDetector(BaseTrackADetector):
    """sklearn DecisionTreeClassifier wrapped in SMOTE-in-CV."""

    PARAM_SPACE: ClassVar[Dict[str, List[Any]]] = PARAM_SPACE
    DEFAULT_N_ITER: ClassVar[int] = 25
    MODEL_TYPE: ClassVar[str] = "DecisionTreeClassifier"
    LOG_NAME: ClassVar[str] = "DecisionTree"

    def _classifier(self) -> Any:
        return DecisionTreeClassifier(random_state=self._random_state)
