"""Risk-adaptive pipeline — orchestrates Phase 4 end-to-end.

Components are injected via constructor (Dependency Inversion).
Reuses Phase 2 SOLID components for data reshaping.

Usage::

    python -m src.phase4_risk_engine.phase4.pipeline
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import tensorflow as tf

from src.phase2_detection_engine.phase2.reshaper import DataReshaper

from .artifact_reader import Phase3ArtifactReader
from .baseline import BaselineComputer
from .config import Phase4Config
from .cross_modal import CrossModalFusionDetector
from .drift_detector import ConceptDriftDetector
from .dynamic_threshold import DynamicThresholdUpdater
from .exporter import RiskAdaptiveExporter
from .fallback_manager import ThresholdFallbackManager
from .report import render_risk_adaptive_report
from .risk_scorer import RiskScorer

logger = logging.getLogger(__name__)

PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]


def _get_git_commit() -> str:
    """Get current git commit hash for model versioning."""
    try:
        result = subprocess.run(  # noqa: S603, S607
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(PROJECT_ROOT),
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


def _detect_hardware() -> Dict[str, str]:
    """Detect GPU/CPU availability and return hardware info dict."""
    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        device_name = gpus[0].name
        logger.info("  GPU detected: %s", device_name)
        info = {"device": f"GPU: {device_name}", "cuda": "available"}
    else:
        cpu_info = platform.processor() or platform.machine()
        logger.info("  CPU fallback: %s", cpu_info)
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        info = {"device": f"CPU: {cpu_info}", "cuda": "N/A (CPU execution)"}

    info["tensorflow"] = tf.__version__
    info["python"] = platform.python_version()
    info["platform"] = platform.platform()
    return info


class RiskAdaptivePipeline:
    """Orchestrate Phase 4: verify → baseline → threshold → drift → score → export.

    All components are injected via the constructor (Dependency Inversion).

    Args:
        config: Validated Phase 4 configuration.
        artifact_reader: Phase 3/2 artifact reader.
        baseline_computer: Baseline computer (Median + MAD).
        threshold_updater: Dynamic threshold updater.
        risk_scorer: Risk scorer (5-level + cross-modal).
        exporter: Risk-adaptive exporter.
        project_root: Absolute project root for path resolution.
    """

    def __init__(
        self,
        config: Phase4Config,
        artifact_reader: Phase3ArtifactReader,
        baseline_computer: BaselineComputer,
        threshold_updater: DynamicThresholdUpdater,
        risk_scorer: RiskScorer,
        exporter: RiskAdaptiveExporter,
        project_root: Path,
    ) -> None:
        self._config = config
        self._reader = artifact_reader
        self._baseline = baseline_computer
        self._threshold = threshold_updater
        self._scorer = risk_scorer
        self._exporter = exporter
        self._root = project_root

    def run(self) -> Dict[str, Any]:
        """Execute all pipeline steps and return the summary dict.

        Returns:
            Risk report summary dict.
        """
        t0 = time.time()
        cfg = self._config

        logger.info("═══════════════════════════════════════════════════")
        logger.info("  Phase 4 Risk-Adaptive Engine (SOLID)")
        logger.info("═══════════════════════════════════════════════════")

        # 1. Reproducibility seeds
        np.random.seed(cfg.random_state)  # noqa: NPY002
        tf.random.set_seed(cfg.random_state)
        os.environ["TF_DETERMINISTIC_OPS"] = "1"

        # 2. Hardware detection
        hw_info = _detect_hardware()

        # 3. Verify Phase 3 + Phase 2 artifacts (SHA-256)
        _, p3_metadata = self._reader.verify_phase3()
        _, p2_metadata = self._reader.verify_phase2()

        # 4. Load attention output + Phase 1 data
        attn_df = self._reader.load_attention_output()
        train_path = self._root / cfg.phase1_train
        test_path = self._root / cfg.phase1_test
        X_train, y_train, X_test, y_test, feature_names = self._reader.load_phase1_data(
            train_path, test_path
        )

        # Load Phase 3 metrics (DO NOT recompute)
        p3_metrics_path = self._root / cfg.phase3_dir / "metrics_report.json"
        p3_metrics = json.loads(p3_metrics_path.read_text())["metrics"]

        # 5. Rebuild model and predict anomaly scores
        model = self._reader.rebuild_model(p2_metadata, p3_metadata)
        hp = p2_metadata["hyperparameters"]
        reshaper = DataReshaper(timesteps=hp["timesteps"], stride=hp["stride"])
        X_test_w, y_test_w = reshaper.reshape(X_test, y_test)
        logger.info("  Test data reshaped: %s", X_test_w.shape)

        logger.info("── Computing anomaly scores ──")
        anomaly_scores = model.predict(X_test_w, verbose=0).flatten()
        logger.info(
            "  Anomaly scores: min=%.4f, max=%.4f, mean=%.4f",
            float(anomaly_scores.min()),
            float(anomaly_scores.max()),
            float(anomaly_scores.mean()),
        )

        # 6. Compute baseline (Median + MAD)
        baseline = self._baseline.compute(attn_df)

        # 7. Dynamic thresholding (rolling window)
        dynamic_thresholds, window_log = self._threshold.update(anomaly_scores, baseline)

        # 8. Concept drift detection + fallback
        drift_detector = ConceptDriftDetector(drift_threshold=cfg.drift_threshold)
        fallback = ThresholdFallbackManager(
            drift_detector=drift_detector,
            baseline_threshold=baseline["baseline_threshold"],
            recovery_threshold=cfg.recovery_threshold,
            recovery_windows=cfg.recovery_windows,
        )
        adjusted_thresholds, drift_events = fallback.process(dynamic_thresholds, cfg.window_size)

        # 9. Risk scoring (5-level + cross-modal)
        raw_test_features = X_test[: len(anomaly_scores)]
        risk_results = self._scorer.score(
            anomaly_scores=anomaly_scores,
            thresholds=adjusted_thresholds,
            mad=baseline["mad"],
            raw_features=raw_test_features,
            feature_names=feature_names,
        )

        # 10. Export 4 artifacts
        logger.info("── Exporting artifacts ──")
        duration_s = time.time() - t0
        git_commit = _get_git_commit()

        self._exporter.export_baseline(baseline, cfg.baseline_file)

        # Build threshold config dict for export
        threshold_export_cfg = {
            "k_schedule": [
                {
                    "start_hour": e.start_hour,
                    "end_hour": e.end_hour,
                    "k": e.k,
                }
                for e in cfg.k_schedule
            ],
            "window_size": cfg.window_size,
        }
        self._exporter.export_threshold_config(
            baseline, window_log, threshold_export_cfg, cfg.threshold_file
        )
        self._exporter.export_risk_report(
            risk_results,
            baseline,
            p3_metrics,
            hw_info,
            duration_s,
            git_commit,
            cfg.risk_report_file,
        )
        self._exporter.export_drift_log(drift_events, cfg.drift_log_file)

        # 11. Generate report
        report_md = render_risk_adaptive_report(
            baseline=baseline,
            risk_results=risk_results,
            drift_events=drift_events,
            window_log=window_log,
            config=cfg,
            hw_info=hw_info,
            duration_s=duration_s,
            p3_metrics=p3_metrics,
            git_commit=git_commit,
        )
        report_dir = self._root / "results" / "phase0_analysis"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "report_section_risk_adaptive.md"
        with open(report_path, "w") as f:
            f.write(report_md)
        logger.info("  Report saved: %s", report_path.name)

        # Summary
        level_counts: Dict[str, int] = {}
        for r in risk_results:
            lvl = r["risk_level"]
            level_counts[lvl] = level_counts.get(lvl, 0) + 1

        self._log_summary(baseline, drift_events, level_counts, duration_s)

        return {
            "baseline": baseline,
            "risk_distribution": level_counts,
            "drift_events": len(drift_events),
            "duration_s": duration_s,
        }

    @staticmethod
    def _log_summary(
        baseline: Dict[str, Any],
        drift_events: list,
        level_counts: Dict[str, int],
        duration_s: float,
    ) -> None:
        """Log a concise pipeline summary."""
        sep = "═" * 51
        logger.info(sep)
        logger.info("  Phase 4 Risk-Adaptive Engine — %.2fs", duration_s)
        logger.info(
            "  Baseline threshold: %.6f",
            baseline["baseline_threshold"],
        )
        logger.info("  Drift events: %d", len(drift_events))
        logger.info("  Risk distribution: %s", level_counts)
        logger.info(sep)


# ===================================================================
# Entry Point
# ===================================================================


def main() -> None:
    """Entry point for ``python -m src.phase4_risk_engine.phase4.pipeline``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    config_path = PROJECT_ROOT / "config" / "phase4_config.yaml"
    config = Phase4Config.from_yaml(config_path)

    reader = Phase3ArtifactReader(
        project_root=PROJECT_ROOT,
        phase3_dir=config.phase3_dir,
        phase3_metadata=config.phase3_metadata,
        phase2_dir=config.phase2_dir,
        phase2_metadata=config.phase2_metadata,
        label_column=config.label_column,
    )

    baseline_computer = BaselineComputer(mad_multiplier=config.mad_multiplier)

    threshold_updater = DynamicThresholdUpdater(
        window_size=config.window_size,
        k_schedule=config.k_schedule,
    )

    cross_modal = CrossModalFusionDetector(biometric_columns=config.biometric_columns)

    scorer = RiskScorer(
        low_upper=config.low_upper,
        medium_upper=config.medium_upper,
        high_upper=config.high_upper,
        cross_modal=cross_modal,
    )

    exporter = RiskAdaptiveExporter(output_dir=PROJECT_ROOT / config.output_dir)

    pipeline = RiskAdaptivePipeline(
        config=config,
        artifact_reader=reader,
        baseline_computer=baseline_computer,
        threshold_updater=threshold_updater,
        risk_scorer=scorer,
        exporter=exporter,
        project_root=PROJECT_ROOT,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
