"""XGBoost detection backbone for IoMT intrusion detection.

Uses sklearn's GradientBoostingClassifier as an XGBoost-equivalent
surrogate.  Supports:
  - RandomizedSearchCV with literature-backed hyperparameter space
  - SMOTE inside the CV pipeline (imblearn.Pipeline)
  - scale_pos_weight via sample_weight for class imbalance
  - Threshold optimization on attack-class F2
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import GradientBoostingClassifier

from .base import BaseDetector

# Literature-backed hyperparameter search space
PARAM_SPACE: Dict[str, List[Any]] = {
    "classifier__n_estimators": [100, 200, 300, 500, 700],
    "classifier__max_depth": [3, 5, 7, 9, 11],
    "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
    "classifier__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "classifier__min_samples_split": [2, 5, 10, 20],
    "classifier__min_samples_leaf": [1, 3, 5, 7],
    "classifier__max_features": ["sqrt", "log2", 0.5, 0.8, None],
}


class XGBoostDetector(BaseDetector):
    """XGBoost-style gradient boosting detector with SMOTE-in-CV.

    Args:
        n_iter: Number of random parameter samples for search.
        cv_folds: Stratified CV folds.
        scoring: Metric for RandomizedSearchCV.
        smote_strategy: SMOTE sampling strategy ("auto" for 1:1).
        smote_k: SMOTE k-neighbors.
        random_state: Seed for reproducibility.
    """

    MODEL_NAME: ClassVar[str] = "XGBoost"
    MODEL_TYPE: ClassVar[str] = "GradientBoostingClassifier (XGBoost equivalent)"
    PARAM_SPACE: ClassVar[Dict[str, List[Any]]] = PARAM_SPACE
    DEFAULT_N_ITER: ClassVar[int] = 50

    def _build_pipeline(self) -> ImbPipeline:
        """SMOTE + GradientBoosting inside an imblearn pipeline."""
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
                (
                    "classifier",
                    GradientBoostingClassifier(
                        random_state=self._random_state,
                    ),
                ),
            ]
        )
