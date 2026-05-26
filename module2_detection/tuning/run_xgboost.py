#!/usr/bin/env python3
"""Run gradient-boosting (XGBoost-equivalent) fine-tuning pipeline.

Delegates entirely to the shared ``_runner.run_track_a_tuning``. The
output directory is still named ``xgboost`` for backward compatibility
with downstream artefact-loading code.

Usage:
    python -m module2_detection.tuning.run_xgboost
    python -m module2_detection.tuning.run_xgboost --n-iter 50 --cv-folds 5
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from module2_detection.models.GradientBoosting import GradientBoostingDetector
from module2_detection.tuning._runner import run_track_a_tuning


if __name__ == "__main__":
    run_track_a_tuning(
        detector_class=GradientBoostingDetector,
        output_subdir="xgboost",
        report_filename="xgboost_report.json",
        description="Gradient-boosting (XGBoost-equivalent) fine-tuning for IoMT IDS",
        default_n_iter=50,
    )
