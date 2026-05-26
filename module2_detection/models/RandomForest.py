"""Random Forest detection backbone for IoMT intrusion detection.

Thin subclass of ``BaseTrackADetector`` — only the classifier choice
and the hyperparameter search space differ from the shared base.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List

from sklearn.ensemble import RandomForestClassifier

from ._base_detector import BaseTrackADetector

# ── Literature-backed hyperparameter search space ────────────────────────
PARAM_SPACE: Dict[str, List[Any]] = {
    "classifier__n_estimators": [100, 200, 300, 500, 700],
    "classifier__max_depth": [10, 15, 20, 30, None],
    "classifier__min_samples_split": [2, 5, 10, 20],
    "classifier__min_samples_leaf": [1, 2, 4, 8],
    "classifier__max_features": ["sqrt", "log2", 0.3, 0.5],
    "classifier__class_weight": ["balanced", "balanced_subsample", None],
    "classifier__criterion": ["gini", "entropy"],
}


class RandomForestDetector(BaseTrackADetector):
    """sklearn RandomForestClassifier wrapped in SMOTE-in-CV."""

    PARAM_SPACE: ClassVar[Dict[str, List[Any]]] = PARAM_SPACE
    DEFAULT_N_ITER: ClassVar[int] = 40
    MODEL_TYPE: ClassVar[str] = "RandomForestClassifier"
    LOG_NAME: ClassVar[str] = "RandomForest"

    def _classifier(self) -> Any:
        return RandomForestClassifier(
            bootstrap=True,
            random_state=self._random_state,
            n_jobs=-1,
        )
