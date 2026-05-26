"""Module 2 — Detection.

Public API
----------
GradientBoostingDetector / RandomForestDetector / DecisionTreeDetector
    Track A supervised classifiers (SMOTE-in-CV + OOF threshold opt).
DAEDetector
    Track B denoising autoencoder (benign-only novelty detector).
XGBoostDetector
    Deprecated alias for GradientBoostingDetector.

CLI entry points
----------------
``python -m module2_detection.module2_train_models``
    Final-fit Track A + Track B, evaluate on frozen test split.
``python -m module2_detection.tuning.run_xgboost`` (+ run_random_forest, run_decision_tree)
    RandomizedSearchCV hyperparameter search per detector.
``python -m module2_detection.tuning.run_dae``
    DAE grid search on train-only validation slice (finding #1 fix).
"""

from .models import (
    DAEDetector,
    DecisionTreeDetector,
    GradientBoostingDetector,
    RandomForestDetector,
    XGBoostDetector,
)

__all__ = [
    "GradientBoostingDetector",
    "RandomForestDetector",
    "DecisionTreeDetector",
    "DAEDetector",
    "XGBoostDetector",
]
