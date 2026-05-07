#!/usr/bin/env python3
"""Run Random Forest fine-tuning pipeline.

Loads Phase 1 preprocessed data, runs RandomizedSearchCV with SMOTE-in-CV,
optimizes the decision threshold on attack-class F2, evaluates on the
held-out test set, and persists all artifacts.

Usage:
    python module2_detection/tuning/run_random_forest.py
    python module2_detection/tuning/run_random_forest.py --n-iter 40 --cv-folds 5
"""

from __future__ import annotations

from module2_detection.models.RandomForest import RandomForestDetector
from module2_detection.tuning._runner import run_tuning


def main() -> None:
    run_tuning(
        detector_cls=RandomForestDetector,
        display_name="Random Forest",
        output_subdir="random_forest",
        report_filename="random_forest_report.json",
        default_n_iter=40,
    )


if __name__ == "__main__":
    main()
