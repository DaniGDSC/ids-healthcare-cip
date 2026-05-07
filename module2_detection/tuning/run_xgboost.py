#!/usr/bin/env python3
"""Run XGBoost fine-tuning pipeline.

Loads Phase 1 preprocessed data, runs RandomizedSearchCV with SMOTE-in-CV,
optimizes the decision threshold on attack-class F2, evaluates on the
held-out test set, and persists all artifacts.

Usage:
    python module2_detection/tuning/run_xgboost.py
    python module2_detection/tuning/run_xgboost.py --n-iter 50 --cv-folds 5
"""

from __future__ import annotations

from module2_detection.models.XGBoost import XGBoostDetector
from module2_detection.tuning._runner import run_tuning


def main() -> None:
    run_tuning(
        detector_cls=XGBoostDetector,
        display_name="XGBoost",
        output_subdir="xgboost",
        report_filename="xgboost_report.json",
        default_n_iter=50,
    )


if __name__ == "__main__":
    main()
