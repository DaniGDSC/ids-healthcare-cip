"""Phase 2 artifact reader — load and SHA-256-verify detection outputs.

Reads ``detection_model.weights.h5``, ``attention_output.parquet``,
and ``detection_metadata.json``.  Also loads Phase 1 parquets for
classification training.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_HASH_CHUNK: int = 65_536


class Phase2ArtifactReader:
    """Load and verify Phase 2 detection artifacts via SHA-256.

    Args:
        project_root: Absolute path to the project root directory.
        phase2_dir: Relative path to the Phase 2 output directory.
        metadata_file: Relative path to ``detection_metadata.json``.
        label_column: Name of the binary label column.
    """

    def __init__(
        self,
        project_root: Path,
        phase2_dir: Path,
        metadata_file: Path,
        label_column: str = "Label",
    ) -> None:
        self._root = project_root
        self._phase2_dir = project_root / phase2_dir
        self._metadata_path = project_root / metadata_file
        self._label_col = label_column

    def load_and_verify(self) -> Tuple[Path, Dict[str, Any]]:
        """Load metadata and verify all Phase 2 artifact SHA-256 hashes.

        Returns:
            Tuple of (weights_path, metadata_dict).

        Raises:
            FileNotFoundError: If metadata or artifact files are missing.
            ValueError: If SHA-256 hash does not match expected value.
        """
        logger.info("── Phase 2 artifact verification ──")

        if not self._metadata_path.exists():
            raise FileNotFoundError(f"Missing: {self._metadata_path}")

        with open(self._metadata_path) as f:
            metadata = json.load(f)

        for artifact_name, hash_info in metadata["artifact_hashes"].items():
            artifact_path = self._phase2_dir / artifact_name
            if not artifact_path.exists():
                raise FileNotFoundError(f"Missing: {artifact_path}")

            expected = hash_info["sha256"]
            actual = self._compute_sha256(artifact_path)
            if actual != expected:
                raise ValueError(
                    f"SHA-256 mismatch for {artifact_name}: "
                    f"expected={expected[:16]}…, actual={actual[:16]}…"
                )
            logger.info("  ✓ SHA-256 verified: %s", artifact_name)

        weights_path = self._phase2_dir / "detection_model.weights.h5"
        logger.info("  All Phase 2 artifacts verified.")
        return weights_path, metadata

    def load_phase1_data(
        self,
        train_path: Path,
        test_path: Path,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Load Phase 1 train/test parquets and separate features/labels.

        Args:
            train_path: Absolute path to train_phase1.parquet.
            test_path: Absolute path to test_phase1.parquet.

        Returns:
            (X_train, y_train, X_test, y_test, feature_names).
        """
        logger.info("── Loading Phase 1 data ──")

        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)

        feature_names = [c for c in train_df.columns if c != self._label_col]

        y_train = train_df[self._label_col].values
        X_train = train_df[feature_names].values.astype(np.float32)

        y_test = test_df[self._label_col].values
        X_test = test_df[feature_names].values.astype(np.float32)

        logger.info("  Train: %s, Test: %s", X_train.shape, X_test.shape)
        dist = dict(zip(*np.unique(y_train, return_counts=True)))
        logger.info("  Label distribution (train): %s", dist)
        return X_train, y_train, X_test, y_test, feature_names

    @staticmethod
    def _compute_sha256(path: Path) -> str:
        """Compute SHA-256 hex digest of a file."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(_HASH_CHUNK):
                h.update(chunk)
        return h.hexdigest()
