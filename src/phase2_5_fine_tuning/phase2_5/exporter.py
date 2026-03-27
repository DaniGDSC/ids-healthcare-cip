"""Tuning exporter — save search results, ablation results, and best config.

Follows the same pattern as ``ClassificationExporter`` in Phase 3.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


class TuningExporter:
    """Export Phase 2.5 tuning and ablation artifacts.

    Args:
        output_dir: Directory for all output files.
    """

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir

    def export_tuning_results(
        self, results: Dict[str, Any], filename: str
    ) -> Path:
        """Write hyperparameter search results as JSON."""
        return self._write_json(results, filename, "tuning results")

    def export_ablation_results(
        self, results: Dict[str, Any], filename: str
    ) -> Path:
        """Write ablation study results as JSON."""
        return self._write_json(results, filename, "ablation results")

    def export_best_config(
        self, best_config: Dict[str, Any], filename: str
    ) -> Path:
        """Write the best hyperparameter configuration as JSON."""
        return self._write_json(best_config, filename, "best config")

    def export_json(
        self, data: Dict[str, Any], filename: str
    ) -> Path:
        """Write any dict as JSON."""
        return self._write_json(data, filename, filename)

    def export_report(
        self, report: Dict[str, Any], filename: str
    ) -> Path:
        """Write the pipeline report as JSON."""
        return self._write_json(report, filename, "report")

    def _write_json(
        self, data: Dict[str, Any], filename: str, label: str
    ) -> Path:
        """Write data as JSON to the output directory."""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        path = self._output_dir / filename
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info("  Saved %s: %s", label, path.name)
        return path

    @staticmethod
    def build_report(
        tuning_results: Dict[str, Any],
        ablation_results: Dict[str, Any],
        importance_results: Dict[str, Any],
        multi_seed_results: Dict[str, Any],
        hw_info: Dict[str, str],
        duration_s: float,
        git_commit: str,
    ) -> Dict[str, Any]:
        """Build the pipeline report dict (pure computation, no I/O)."""
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "pipeline": "phase2_5_fine_tuning",
            "git_commit": git_commit,
            "hardware": hw_info,
            "duration_seconds": round(duration_s, 2),
            "tuning_summary": {
                "strategy": tuning_results.get("strategy"),
                "metric": tuning_results.get("metric"),
                "total_trials": tuning_results.get("total_trials"),
                "completed_trials": tuning_results.get("completed_trials"),
                "pruned_trials": tuning_results.get("pruned_trials", 0),
                "best_score": tuning_results.get("best_score"),
                "best_config": tuning_results.get("best_config"),
            },
            "importance_summary": {
                "method": importance_results.get("method"),
                "top_3": dict(list(importance_results.get("importances", {}).items())[:3]),
            },
            "multi_seed_summary": {
                "enabled": multi_seed_results.get("enabled", False),
                "configs": [
                    {
                        "rank": c["rank"],
                        "mean": c.get("statistics", {}).get("mean"),
                        "std": c.get("statistics", {}).get("std"),
                    }
                    for c in multi_seed_results.get("configs", [])
                ],
            },
            "ablation_summary": {
                "n_variants": len(ablation_results.get("variants", [])),
                "comparison": ablation_results.get("comparison", []),
            },
        }
