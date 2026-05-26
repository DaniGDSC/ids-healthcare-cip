#!/usr/bin/env python3
"""Run Decision Tree fine-tuning pipeline.

Delegates entirely to the shared ``_runner.run_track_a_tuning``.

Usage:
    python -m module2_detection.tuning.run_decision_tree
    python -m module2_detection.tuning.run_decision_tree --n-iter 25 --cv-folds 5
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from module2_detection.models.DecisionTree import DecisionTreeDetector
from module2_detection.tuning._runner import run_track_a_tuning


if __name__ == "__main__":
    run_track_a_tuning(
        detector_class=DecisionTreeDetector,
        output_subdir="decision_tree",
        report_filename="decision_tree_report.json",
        description="Decision Tree fine-tuning for IoMT intrusion detection",
        default_n_iter=25,
    )
