"""Phase 0 artifact reader — Dependency Inversion.

Reads pre-computed Phase 0 outputs (stats, correlations, integrity hash)
so Phase 1 never recomputes what Phase 0 already produced.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd

logger = logging.getLogger(__name__)

_HASH_CHUNK: int = 65_536


class Phase0ArtifactReader:
    """Read Phase 0 analysis artifacts for Phase 1 consumption.

    Args:
        project_root: Absolute path to the project root directory.
        stats_file: Relative path to ``stats_report.json``.
        corr_file: Relative path to ``high_correlations.csv``.
        integrity_file: Relative path to ``dataset_integrity.json``.
    """

    def __init__(
        self,
        project_root: Path,
        stats_file: Path,
        corr_file: Path,
        integrity_file: Path,
    ) -> None:
        self._root = project_root
        self._stats_path = project_root / stats_file
        self._corr_path = project_root / corr_file
        self._integrity_path = project_root / integrity_file

    def read_stats(self) -> Dict[str, Any]:
        """Read descriptive statistics and class distribution from Phase 0.

        Returns:
            Dict with ``descriptive_statistics``, ``missing_values``,
            ``class_distribution``.

        Raises:
            FileNotFoundError: If the stats file is missing.
        """
        if not self._stats_path.exists():
            raise FileNotFoundError(f"Phase 0 stats not found: {self._stats_path}")
        data = json.loads(self._stats_path.read_text(encoding="utf-8"))
        logger.info("Phase 0 stats: %d features, missing=%s",
                     len(data.get("descriptive_statistics", {})),
                     "none" if not data.get("missing_values") else "present")
        return data

    def read_correlations(self) -> pd.DataFrame:
        """Read high-correlation feature pairs from Phase 0.

        Returns:
            DataFrame with columns ``feature_a``, ``feature_b``,
            ``correlation``.

        Raises:
            FileNotFoundError: If the correlations file is missing.
        """
        if not self._corr_path.exists():
            raise FileNotFoundError(
                f"Phase 0 correlations not found: {self._corr_path}"
            )
        df = pd.read_csv(self._corr_path)
        logger.info("Phase 0 correlations: %d pairs loaded", len(df))
        return df

    def verify_integrity(self, dataset_path: Path) -> str:
        """Verify dataset SHA-256 against the Phase 0 baseline.

        Args:
            dataset_path: Path to the raw CSV file.

        Returns:
            Verified SHA-256 hex digest.

        Raises:
            ValueError: If the hash does not match.
        """
        current = self._compute_sha256(dataset_path)

        if not self._integrity_path.exists():
            logger.warning("No integrity baseline — skipping verification.")
            return current

        metadata = json.loads(self._integrity_path.read_text(encoding="utf-8"))
        stored = None
        for key, val in metadata.items():
            if Path(key).name == dataset_path.name:
                stored = val["sha256"]
                break

        if stored is None:
            logger.warning("Dataset not in integrity file.")
            return current

        if current != stored:
            raise ValueError(
                f"INTEGRITY VIOLATION: expected {stored[:16]}…, "
                f"got {current[:16]}…"
            )
        logger.info("Integrity verified: sha256=%s…", current[:16])
        return current

    @staticmethod
    def _compute_sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(_HASH_CHUNK):
                h.update(chunk)
        return h.hexdigest()
