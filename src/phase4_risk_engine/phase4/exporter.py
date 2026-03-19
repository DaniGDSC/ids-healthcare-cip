"""Risk-adaptive exporter — save baseline, threshold, risk, and drift artifacts.

Follows the same pattern as ClassificationExporter in Phase 3.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .risk_level import RiskLevel

logger = logging.getLogger(__name__)


class RiskAdaptiveExporter:
    """Export Phase 4 risk-adaptive artifacts.

    Args:
        output_dir: Directory for all output files.
    """

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir

    def export_baseline(self, baseline: Dict[str, Any], filename: str) -> Path:
        """Export immutable baseline config as JSON.

        Returns:
            Absolute path to the written file.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)
        path = self._output_dir / filename
        with open(path, "w") as f:
            json.dump(baseline, f, indent=2)
        logger.info("  Saved baseline config: %s", path.name)
        return path

    def export_threshold_config(
        self,
        baseline: Dict[str, Any],
        window_log: List[Dict[str, Any]],
        config: Dict[str, Any],
        filename: str,
    ) -> Path:
        """Export current threshold configuration as JSON.

        Args:
            baseline: Baseline config dict.
            window_log: Dynamic threshold window log.
            config: k_schedule and window_size from config.
            filename: Output filename.

        Returns:
            Absolute path to the written file.
        """
        threshold_cfg = {
            "baseline_threshold": baseline["baseline_threshold"],
            "current_dynamic_threshold": (
                window_log[-1]["dynamic_threshold"]
                if window_log
                else baseline["baseline_threshold"]
            ),
            "k_schedule": config["k_schedule"],
            "window_size": config["window_size"],
            "window_log_entries": len(window_log),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        path = self._output_dir / filename
        with open(path, "w") as f:
            json.dump(threshold_cfg, f, indent=2)
        logger.info("  Saved threshold config: %s", path.name)
        return path

    def export_risk_report(
        self,
        risk_results: List[Dict[str, Any]],
        baseline: Dict[str, Any],
        metrics: Dict[str, Any],
        hw_info: Dict[str, str],
        duration_s: float,
        git_commit: str,
        filename: str,
    ) -> Path:
        """Export risk report with per-sample assessments and summary.

        Returns:
            Absolute path to the written file.
        """
        level_counts = self.build_risk_summary(risk_results)

        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "pipeline": "phase4_risk_adaptive",
            "git_commit": git_commit,
            "hardware": hw_info,
            "duration_seconds": round(duration_s, 2),
            "baseline": {
                "median": baseline["median"],
                "mad": baseline["mad"],
                "threshold": baseline["baseline_threshold"],
                "n_normal_samples": baseline["n_normal_samples"],
            },
            "phase3_metrics": {
                "accuracy": metrics.get("accuracy", 0),
                "f1_score": metrics.get("f1_score", 0),
                "auc_roc": metrics.get("auc_roc", 0),
            },
            "risk_distribution": level_counts,
            "total_samples": len(risk_results),
            "sample_assessments": risk_results,
        }
        path = self._output_dir / filename
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(
            "  Saved risk report: %s (%d samples)",
            path.name,
            len(risk_results),
        )
        return path

    def export_drift_log(
        self,
        drift_events: List[Dict[str, Any]],
        filename: str,
    ) -> Path:
        """Export drift events as CSV.

        Returns:
            Absolute path to the written file.
        """
        if drift_events:
            df = pd.DataFrame(drift_events)
        else:
            df = pd.DataFrame(
                columns=[
                    "sample_index",
                    "drift_ratio",
                    "action",
                    "dynamic_threshold",
                    "baseline_threshold",
                ]
            )
        self._output_dir.mkdir(parents=True, exist_ok=True)
        path = self._output_dir / filename
        df.to_csv(path, index=False)
        logger.info(
            "  Saved drift log: %s (%d events)",
            path.name,
            len(drift_events),
        )
        return path

    @staticmethod
    def build_risk_summary(
        risk_results: List[Dict[str, Any]],
    ) -> Dict[str, int]:
        """Build risk level distribution counts.

        Args:
            risk_results: Per-sample risk assessment list.

        Returns:
            Dict mapping risk level strings to counts.
        """
        counts: Dict[str, int] = {lvl.value: 0 for lvl in RiskLevel}
        for r in risk_results:
            lvl = r["risk_level"]
            counts[lvl] = counts.get(lvl, 0) + 1
        return counts
