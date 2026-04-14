"""Preprocessing artifact exporter — Single Responsibility.

Writes Parquet splits, scaler pickle, and JSON report.
All file I/O is isolated in this class.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PreprocessingExporter:
    """Export Phase 1 preprocessing artifacts.

    Args:
        output_dir: Directory for Parquet and JSON output.
        scaler_dir: Directory for the scaler pickle.
        label_column: Label column name in output Parquet.
        multi_label_column: Multi-class label column name in output Parquet.
    """

    def __init__(
        self,
        output_dir: Path,
        scaler_dir: Path,
        label_column: str = "Label",
        multi_label_column: str = "Attack Category",
    ) -> None:
        self._output_dir = output_dir
        self._scaler_dir = scaler_dir
        self._label_col = label_column
        self._multi_label_col = multi_label_column

    def export_parquet(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        filename: str,
        y_multi: np.ndarray | None = None,
    ) -> Path:
        """Export a scaled partition as a Parquet file (atomic write).

        Writes to ``<filename>.tmp`` first and then ``os.replace`` so a
        crash mid-write cannot leave a half-written Parquet file that
        the next stage would parse as truncated. ``os.replace`` is
        atomic on POSIX and Windows for same-filesystem moves.

        Args:
            X: Scaled feature matrix.
            y: Binary label array.
            feature_names: Ordered column names.
            filename: Output file name.
            y_multi: Optional multi-class label array.

        Returns:
            Absolute path to the written file.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)
        path = self._output_dir / filename
        df = pd.DataFrame(X, columns=feature_names)
        df[self._label_col] = y
        if y_multi is not None and len(y_multi) > 0:
            df[self._multi_label_col] = y_multi
        tmp = path.with_suffix(path.suffix + ".tmp")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
        logger.info("Exported %s: %d rows × %d cols", path.name, *df.shape)
        return path

    def export_scaler(self, scaler: Any, filename: str) -> Path:
        """Persist the fitted scaler to disk as a JSON sidecar.

        ``scaler`` must implement ``.save(path)`` (i.e. be a
        ``RobustScalerTransformer``). The exporter no longer falls back
        to ``joblib.dump`` — pickling the scaler is the RCE sink that
        the JSON sidecar exists to eliminate, so an unsupported scaler
        type is a hard error rather than a silent pickle.

        If *filename* still ends in ``.pkl`` (legacy config), the actual
        file written is the ``.json`` sibling. The returned path
        reflects the real on-disk location.

        Args:
            scaler: Fitted scaler object exposing ``.save(path)``.
            filename: Output file name from config.

        Returns:
            Absolute path to the written sidecar.
        """
        if not hasattr(scaler, "save"):
            raise TypeError(
                f"export_scaler refuses to pickle a {type(scaler).__name__} "
                f"— wrap it in RobustScalerTransformer (which implements "
                f"a JSON-only save) instead. Pickling fitted estimators "
                f"is an RCE sink at every load site."
            )
        self._scaler_dir.mkdir(parents=True, exist_ok=True)
        path = self._scaler_dir / filename
        scaler.save(path)
        # ``RobustScalerTransformer.save`` rewrites .pkl → .json on disk;
        # reflect that in the returned path so callers point at the real
        # file rather than a phantom .pkl.
        if path.suffix == ".pkl":
            path = path.with_suffix(".json")
        logger.info("Exported scaler sidecar: %s", path.name)
        return path

    def export_report(self, report: Dict[str, Any], filename: str) -> Path:
        """Write the pipeline report as JSON (atomic, strict serialisation).

        ``default=str`` is intentionally NOT used: a value that doesn't
        round-trip through ``json.dumps`` is a bug at the producer that
        should fail loudly here, not be coerced to a ``repr()`` string
        that looks legitimate in the JSON. The only known offender (the
        unsigned integrity baseline) was removed in finding #1.

        Args:
            report: Complete pipeline report dict.
            filename: Output file name.

        Returns:
            Absolute path to the written file.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)
        path = self._output_dir / filename
        try:
            payload = json.dumps(report, indent=2)
        except TypeError as exc:
            raise TypeError(
                f"Phase 1 report contains a non-JSON-serialisable value "
                f"(detail: {exc}). Fix the producer; this exporter no "
                f"longer silently coerces with default=str."
            ) from exc
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, path)
        logger.info("Exported report: %s", path.name)
        return path
