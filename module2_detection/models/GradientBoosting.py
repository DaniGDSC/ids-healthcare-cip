"""Gradient Boosting detection backbone for IoMT intrusion detection.

This class wraps sklearn's ``GradientBoostingClassifier`` as an
XGBoost-equivalent surrogate. We use sklearn rather than the ``xgboost``
library so the project has zero extra C/C++ dependencies and the
training stack matches the reproducibility profile of the rest of the
detectors. Performance is within a few F2 points of native xgboost on
this corpus; the trade-off is disclosed in the manuscript benchmark
section.

Historical name
---------------
This class used to be called ``XGBoostDetector`` — the new name is
``GradientBoostingDetector`` so the class identity matches the underlying
implementation. ``XGBoostDetector`` is retained as a compatibility alias
in ``XGBoost.py``.

Supports (inherited from ``BaseTrackADetector``):
  - RandomizedSearchCV with literature-backed hyperparameter space
  - SMOTE inside the CV pipeline (imblearn.Pipeline)
  - Threshold optimization on attack-class F2 via OUT-OF-FOLD probas
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List

from sklearn.ensemble import GradientBoostingClassifier

from ._base_detector import BaseTrackADetector

# ── Literature-backed hyperparameter search space ────────────────────────
PARAM_SPACE: Dict[str, List[Any]] = {
    "classifier__n_estimators": [100, 200, 300, 500, 700],
    "classifier__max_depth": [3, 5, 7, 9, 11],
    "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
    "classifier__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "classifier__min_samples_split": [2, 5, 10, 20],
    "classifier__min_samples_leaf": [1, 3, 5, 7],
    "classifier__max_features": ["sqrt", "log2", 0.5, 0.8, None],
}


class GradientBoostingDetector(BaseTrackADetector):
    """sklearn GradientBoostingClassifier wrapped in SMOTE-in-CV.

    Args inherited from ``BaseTrackADetector``. See that class for the
    full pipeline contract.
    """

    PARAM_SPACE: ClassVar[Dict[str, List[Any]]] = PARAM_SPACE
    DEFAULT_N_ITER: ClassVar[int] = 50
    MODEL_TYPE: ClassVar[str] = (
        "GradientBoostingClassifier (XGBoost-equivalent surrogate)"
    )
    LOG_NAME: ClassVar[str] = "GradientBoosting"

    def _classifier(self) -> Any:
        return GradientBoostingClassifier(random_state=self._random_state)
