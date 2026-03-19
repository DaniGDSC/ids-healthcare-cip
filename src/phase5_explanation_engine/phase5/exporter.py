"""ExplanationExporter — export Phase 5 explanation artifacts."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_HASH_CHUNK: int = 65_536


class ExplanationExporter:
    """Export Phase 5 explanation artifacts.

    Args:
        output_dir: Directory for all output files.
    """

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir

    def export_shap_values(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
        filename: str,
    ) -> Tuple[Path, str]:
        """Export aggregated SHAP values as parquet.

        Returns:
            Tuple of (path, sha256_hash).
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)
        per_sample_shap = np.mean(np.abs(shap_values), axis=1)  # (N, F)
        shap_df = pd.DataFrame(per_sample_shap, columns=feature_names)
        path = self._output_dir / filename
        shap_df.to_parquet(path, index=False)
        sha = self._compute_sha256(path)
        logger.info("  Saved: %s (%s)", filename, shap_df.shape)
        return path, sha

    def export_explanation_report(
        self,
        enriched_samples: List[Dict[str, Any]],
        importance_df: pd.DataFrame,
        level_counts: Dict[str, int],
        top_k: int,
        git_commit: str,
        filename: str,
    ) -> Tuple[Path, str]:
        """Export explanation report as JSON.

        Returns:
            Tuple of (path, sha256_hash).
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "pipeline": "phase5_explanation",
            "git_commit": git_commit,
            "total_explained": len(enriched_samples),
            "risk_level_counts": level_counts,
            "feature_importance": importance_df.head(top_k)[
                ["feature", "mean_abs_shap", "rank"]
            ].to_dict(orient="records"),
            "explanations": enriched_samples,
        }
        path = self._output_dir / filename
        path.write_text(json.dumps(report, indent=2))
        sha = self._compute_sha256(path)
        logger.info(
            "  Saved: %s (%d explanations)",
            filename,
            len(enriched_samples),
        )
        return path, sha

    def export_metadata(
        self,
        enriched_samples: List[Dict[str, Any]],
        level_counts: Dict[str, int],
        chart_files: List[str],
        hw_info: Dict[str, str],
        duration_s: float,
        git_commit: str,
        artifact_hashes: Dict[str, str],
        background_samples: int,
        filename: str,
    ) -> Path:
        """Export explanation metadata as JSON.

        Returns:
            Path to saved metadata file.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "pipeline": "phase5_explanation",
            "git_commit": git_commit,
            "hardware": hw_info,
            "duration_seconds": round(duration_s, 2),
            "samples_explained": len(enriched_samples),
            "shap_method": ("GradientExplainer (integrated gradients fallback)"),
            "background_samples": background_samples,
            "risk_level_counts": level_counts,
            "charts_generated": chart_files,
            "artifact_hashes": {
                k: {"sha256": v, "algorithm": "SHA-256"} for k, v in artifact_hashes.items()
            },
        }
        path = self._output_dir / filename
        path.write_text(json.dumps(metadata, indent=2))
        logger.info("  Saved: %s", filename)
        return path

    @staticmethod
    def _compute_sha256(path: Path) -> str:
        """Compute SHA-256 hex digest of a file."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(_HASH_CHUNK):
                h.update(chunk)
        return h.hexdigest()

    def get_config(self) -> Dict[str, Any]:
        """Return exporter configuration."""
        return {"output_dir": str(self._output_dir)}
