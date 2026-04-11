"""Phase 0 artifact reader — Dependency Inversion.

Reads pre-computed Phase 0 outputs (stats, correlations) so Phase 1
never recomputes what Phase 0 already produced.

Integrity verification used to live here as a parallel, fail-open
re-implementation that quietly skipped the check on a missing baseline
and ignored the signed-metadata format. It has been removed: Phase 1
now delegates integrity to ``phase0.security.IntegrityVerifier``,
which refuses to run without a signed baseline. See
``PreprocessingPipeline._ingest_with_integrity`` for the new wiring.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd

logger = logging.getLogger(__name__)


class Phase0ArtifactReader:
    """Read Phase 0 analysis artifacts for Phase 1 consumption.

    Args:
        project_root: Absolute path to the project root directory.
        stats_file: Relative path to ``stats_report.json``.
        corr_file: Relative path to ``high_correlations.csv``.
        integrity_file: Relative path to the signed baseline; kept on
            the constructor only so callers don't have to update yet,
            but the reader no longer touches the file. The pipeline
            consumes the baseline directly via ``IntegrityVerifier``.
    """

    def __init__(
        self,
        project_root: Path,
        stats_file: Path,
        corr_file: Path,
        integrity_file: Path | None = None,
    ) -> None:
        self._root = project_root
        self._stats_path = project_root / stats_file
        self._corr_path = project_root / corr_file
        # Retained for backwards-compatible constructor signatures only.
        self._integrity_path = (
            project_root / integrity_file if integrity_file is not None else None
        )

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
