#!/usr/bin/env python3
"""Run Random Forest fine-tuning pipeline.

Delegates entirely to the shared ``_runner.run_track_a_tuning``.

Usage:
    python -m module2_detection.tuning.run_random_forest
    python -m module2_detection.tuning.run_random_forest --n-iter 40 --cv-folds 5
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from module2_detection.models.RandomForest import RandomForestDetector
from module2_detection.tuning._runner import run_track_a_tuning


if __name__ == "__main__":
    run_track_a_tuning(
        detector_class=RandomForestDetector,
        output_subdir="random_forest",
        report_filename="random_forest_report.json",
        description="Random Forest fine-tuning for IoMT intrusion detection",
        default_n_iter=40,
    )
