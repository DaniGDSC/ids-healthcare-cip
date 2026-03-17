"""Phase 1 artifact reader — Dependency Inversion.

Reads Phase 1 parquet outputs, verifies SHA-256 integrity against
``preprocessing_metadata.json``, and extracts feature names from
``phase1_report.json``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_HASH_CHUNK: int = 65_536


class Phase1ArtifactReader:
    """Read and verify Phase 1 preprocessing artifacts.

    Args:
        project_root: Absolute path to the project root directory.
        train_parquet: Relative path to train_phase1.parquet.
        test_parquet: Relative path to test_phase1.parquet.
        metadata_file: Relative path to preprocessing_metadata.json.
        report_file: Relative path to phase1_report.json.
        label_column: Name of the binary label column.
    """

    def __init__(
        self,
        project_root: Path,
        train_parquet: Path,
        test_parquet: Path,
        metadata_file: Path,
        report_file: Path,
        label_column: str = "Label",
    ) -> None:
        self._root = project_root
        self._train_path = project_root / train_parquet
        self._test_path = project_root / test_parquet
        self._metadata_path = project_root / metadata_file
        self._report_path = project_root / report_file
        self._label_col = label_column

    def load_and_verify(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Load parquet files, verify SHA-256, separate X and y.

        Returns:
            Tuple of (X_train, y_train, X_test, y_test, feature_names).

        Raises:
            FileNotFoundError: If any artifact file is missing.
            ValueError: If SHA-256 hash does not match expected value.
        """
        for label, path in [
            ("train parquet", self._train_path),
            ("test parquet", self._test_path),
            ("metadata", self._metadata_path),
            ("report", self._report_path),
        ]:
            if not path.exists():
                raise FileNotFoundError(f"Phase 1 {label} not found: {path}")

        metadata = json.loads(self._metadata_path.read_text(encoding="utf-8"))
        hashes = metadata["artifact_hashes"]

        for name, fpath in [
            ("train_phase1.parquet", self._train_path),
            ("test_phase1.parquet", self._test_path),
        ]:
            actual = self._compute_sha256(fpath)
            expected = hashes[name]["sha256"]
            if actual != expected:
                raise ValueError(
                    f"SHA-256 mismatch: {name} " f"({actual[:16]}… ≠ {expected[:16]}…)"
                )
            logger.info("SHA-256 verified: %s", name)

        report = json.loads(self._report_path.read_text(encoding="utf-8"))
        feature_names: List[str] = report["output"]["feature_names"]

        train_df = pd.read_parquet(self._train_path)
        test_df = pd.read_parquet(self._test_path)

        X_train = train_df[feature_names].values.astype(np.float32)
        y_train = train_df[self._label_col].values.astype(np.int32)
        X_test = test_df[feature_names].values.astype(np.float32)
        y_test = test_df[self._label_col].values.astype(np.int32)

        logger.info(
            "Loaded — train: %s (%d features), test: %s",
            X_train.shape,
            len(feature_names),
            X_test.shape,
        )
        return X_train, y_train, X_test, y_test, feature_names

    @staticmethod
    def _compute_sha256(path: Path) -> str:
        """Compute SHA-256 hex digest of a file."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(_HASH_CHUNK):
                h.update(chunk)
        return h.hexdigest()
