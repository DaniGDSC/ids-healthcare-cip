#!/usr/bin/env python3
"""Run Decision Tree fine-tuning pipeline.

Loads Phase 1 preprocessed data, runs RandomizedSearchCV with SMOTE-in-CV,
optimizes the decision threshold on attack-class F2, evaluates on the
held-out test set, and persists all artifacts.

Usage:
    python module2_detection/tuning/run_decision_tree.py
    python module2_detection/tuning/run_decision_tree.py --n-iter 25 --cv-folds 5
"""

from __future__ import annotations

from module2_detection.models.DecisionTree import DecisionTreeDetector
from module2_detection.tuning._runner import run_tuning


def main() -> None:
    run_tuning(
        detector_cls=DecisionTreeDetector,
        display_name="Decision Tree",
        output_subdir="decision_tree",
        report_filename="decision_tree_report.json",
        default_n_iter=25,
    )


if __name__ == "__main__":
    main()
