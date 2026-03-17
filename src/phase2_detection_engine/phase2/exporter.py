"""Detection artifact exporter — Single Responsibility.

Writes model weights, attention vectors (Parquet), and JSON report.
All file I/O is isolated in this class.
"""

from __future__ import annotations

import json
import logging
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

logger = logging.getLogger(__name__)

_LABEL_COLUMN: str = "Label"


class DetectionExporter:
    """Export Phase 2 detection artifacts.

    Args:
        output_dir: Directory for model weights, parquet, and JSON.
        label_column: Label column name in output Parquet.
    """

    def __init__(
        self,
        output_dir: Path,
        label_column: str = _LABEL_COLUMN,
    ) -> None:
        self._output_dir = output_dir
        self._label_col = label_column

    def export_model_weights(
        self, model: tf.keras.Model, filename: str,
    ) -> Path:
        """Save model weights (Keras 3 .weights.h5 format).

        Args:
            model: Built Keras model (no classification head).
            filename: Output file name (must end with .weights.h5).

        Returns:
            Absolute path to the written file.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)
        path = self._output_dir / filename
        model.save_weights(str(path))
        logger.info("Exported weights: %s", path.name)
        return path

    def export_attention_vectors(
        self,
        train_context: np.ndarray,
        test_context: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        filename: str,
    ) -> Path:
        """Save attention-weighted context vectors as Parquet.

        Args:
            train_context: (n_train, context_dim) float32.
            test_context: (n_test, context_dim) float32.
            y_train: (n_train,) int32 labels.
            y_test: (n_test,) int32 labels.
            filename: Output file name.

        Returns:
            Absolute path to the written file.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)
        path = self._output_dir / filename

        context_dim = train_context.shape[1]
        context_cols = [f"attn_{i}" for i in range(context_dim)]

        train_df = pd.DataFrame(train_context, columns=context_cols)
        train_df[self._label_col] = y_train
        train_df["split"] = "train"

        test_df = pd.DataFrame(test_context, columns=context_cols)
        test_df[self._label_col] = y_test
        test_df["split"] = "test"

        combined = pd.concat([train_df, test_df], ignore_index=True)
        combined.to_parquet(path, index=False)
        logger.info(
            "Exported attention output: %s  shape=%s",
            path.name, combined.shape,
        )
        return path

    def export_report(
        self, report: Dict[str, Any], filename: str,
    ) -> Path:
        """Write the detection report as JSON.

        Args:
            report: Complete detection report dict.
            filename: Output file name.

        Returns:
            Absolute path to the written file.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)
        path = self._output_dir / filename
        path.write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8",
        )
        logger.info("Exported report: %s", path.name)
        return path

    @staticmethod
    def build_report(
        model: tf.keras.Model,
        config_dict: Dict[str, Any],
        feature_names: List[str],
        train_context: np.ndarray,
        test_context: np.ndarray,
        train_windows_shape: Tuple[int, ...],
        test_windows_shape: Tuple[int, ...],
        elapsed: float,
    ) -> Dict[str, Any]:
        """Build the detection report dict (pure computation, no I/O).

        Args:
            model: The detection Keras model.
            config_dict: Assembler + reshape config for the report.
            feature_names: Input feature names from Phase 1.
            train_context: Train context vectors.
            test_context: Test context vectors.
            train_windows_shape: Shape of train sliding windows.
            test_windows_shape: Shape of test sliding windows.
            elapsed: Pipeline wall-clock time in seconds.

        Returns:
            Report dict ready for JSON serialisation.
        """
        layers_info: List[Dict[str, Any]] = []
        for layer in model.layers:
            if layer.name == "input":
                continue
            info: Dict[str, Any] = {
                "name": layer.name,
                "type": type(layer).__name__,
            }
            if hasattr(layer, "output") and layer.output is not None:
                info["output_shape"] = str(tuple(layer.output.shape))
            info["params"] = int(layer.count_params())
            layers_info.append(info)

        return {
            "phase": "Phase 2 — Detection Engine",
            "model_name": model.name,
            "architecture": "CNN→BiLSTM→Attention",
            "total_parameters": int(model.count_params()),
            "trainable_parameters": int(
                sum(
                    tf.keras.backend.count_params(w)
                    for w in model.trainable_weights
                )
            ),
            "layers": layers_info,
            "hyperparameters": config_dict,
            "input_features": feature_names,
            "n_features": len(feature_names),
            "output_dim": int(train_context.shape[1]),
            "shapes": {
                "train_windows": str(train_windows_shape),
                "test_windows": str(test_windows_shape),
                "train_context": str(train_context.shape),
                "test_context": str(test_context.shape),
            },
            "elapsed_seconds": round(elapsed, 3),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "environment": {
                "python": platform.python_version(),
                "tensorflow": tf.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "platform": platform.platform(),
            },
        }
