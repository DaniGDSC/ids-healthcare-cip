"""Classification exporter — save model weights, metrics, CM, and history.

Follows the same pattern as ``DetectionExporter`` in Phase 2.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import tensorflow as tf

logger = logging.getLogger(__name__)


class ClassificationExporter:
    """Export Phase 3 classification artifacts.

    Args:
        output_dir: Directory for all output files.
    """

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir

    def export_model_weights(self, model: tf.keras.Model, filename: str) -> Path:
        """Save model weights (Keras 3 ``.weights.h5`` format).

        Returns:
            Absolute path to the written file.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)
        path = self._output_dir / filename
        model.save_weights(str(path))
        logger.info("  Saved model weights: %s", path.name)
        return path

    def export_metrics(self, metrics_report: Dict[str, Any], filename: str) -> Path:
        """Write metrics report as JSON.

        Returns:
            Absolute path to the written file.
        """
        path = self._output_dir / filename
        with open(path, "w") as f:
            json.dump(metrics_report, f, indent=2)
        logger.info("  Saved metrics: %s", path.name)
        return path

    def export_confusion_matrix(
        self,
        cm: List[List[int]],
        labels: List[str],
        filename: str,
    ) -> Path:
        """Write confusion matrix as CSV.

        Returns:
            Absolute path to the written file.
        """
        cm_arr = np.array(cm)
        cm_df = pd.DataFrame(cm_arr, index=labels, columns=labels)
        cm_df.index.name = "Actual"
        path = self._output_dir / filename
        cm_df.to_csv(path)
        logger.info("  Saved confusion matrix: %s", path.name)
        return path

    def export_history(self, histories: List[Dict[str, Any]], filename: str) -> Path:
        """Write training history as JSON.

        Returns:
            Absolute path to the written file.
        """
        path = self._output_dir / filename
        with open(path, "w") as f:
            json.dump(histories, f, indent=2)
        logger.info("  Saved training history: %s", path.name)
        return path

    @staticmethod
    def build_metrics_report(
        metrics: Dict[str, Any],
        model: tf.keras.Model,
        hw_info: Dict[str, str],
        duration_s: float,
        git_commit: str,
    ) -> Dict[str, Any]:
        """Build the metrics report dict (pure computation, no I/O).

        Returns:
            Structured metrics report for JSON serialisation.
        """
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "pipeline": "phase3_classification",
            "git_commit": git_commit,
            "hardware": hw_info,
            "duration_seconds": round(duration_s, 2),
            "metrics": metrics,
            "model_summary": {
                "name": model.name,
                "total_params": model.count_params(),
                "layers": len(model.layers),
            },
        }
