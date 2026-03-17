"""Preprocessing artifact exporter — Single Responsibility.

Writes Parquet splits, scaler pickle, and JSON report.
All file I/O is isolated in this class.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PreprocessingExporter:
    """Export Phase 1 preprocessing artifacts.

    Args:
        output_dir: Directory for Parquet and JSON output.
        scaler_dir: Directory for the scaler pickle.
        label_column: Label column name in output Parquet.
    """

    def __init__(
        self,
        output_dir: Path,
        scaler_dir: Path,
        label_column: str = "Label",
    ) -> None:
        self._output_dir = output_dir
        self._scaler_dir = scaler_dir
        self._label_col = label_column

    def export_parquet(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        filename: str,
    ) -> Path:
        """Export a scaled partition as a Parquet file.

        Args:
            X: Scaled feature matrix.
            y: Label array.
            feature_names: Ordered column names.
            filename: Output file name.

        Returns:
            Absolute path to the written file.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)
        path = self._output_dir / filename
        df = pd.DataFrame(X, columns=feature_names)
        df[self._label_col] = y
        df.to_parquet(path, index=False)
        logger.info("Exported %s: %d rows × %d cols", path.name, *df.shape)
        return path

    def export_scaler(self, scaler: Any, filename: str) -> Path:
        """Persist the fitted scaler to disk.

        Args:
            scaler: Fitted scaler object (e.g. RobustScalerTransformer).
            filename: Output file name.

        Returns:
            Absolute path to the written file.
        """
        self._scaler_dir.mkdir(parents=True, exist_ok=True)
        path = self._scaler_dir / filename
        if hasattr(scaler, "save"):
            scaler.save(path)
        else:
            joblib.dump(scaler, path)
        logger.info("Exported scaler: %s", path.name)
        return path

    def export_report(self, report: Dict[str, Any], filename: str) -> Path:
        """Write the pipeline report as JSON.

        Args:
            report: Complete pipeline report dict.
            filename: Output file name.

        Returns:
            Absolute path to the written file.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)
        path = self._output_dir / filename
        path.write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )
        logger.info("Exported report: %s", path.name)
        return path
