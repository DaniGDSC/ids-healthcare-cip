"""Tuning pipeline orchestrator — chains all Phase 2.5 components.

Nine-step pipeline:
  Load → Search Space → Hyperparameter Tuning → Importance Analysis →
  Multi-Seed Validation → Ablation → Export → Report → Summary
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import tensorflow as tf

from .ablation import AblationRunner
from .config import Phase2_5Config
from .evaluator import QuickEvaluator
from .exporter import TuningExporter
from .importance import compute_importance
from .multi_seed import MultiSeedValidator
from .report import render_tuning_report
from .search_space import SearchSpace
from .tuner import HyperparameterTuner

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
        cuda_version = getattr(tf.sysconfig, "get_build_info", lambda: {})()
        info = {
            "device": f"GPU: {device_name}",
            "cuda": cuda_version.get("cuda_version", "N/A"),
        }
    else:
        cpu_info = platform.processor() or platform.machine()
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        info = {"device": f"CPU: {cpu_info}", "cuda": "N/A (CPU execution)"}

    info["tensorflow"] = tf.__version__
    info["python"] = platform.python_version()
    info["platform"] = platform.platform()
    return info


class TuningPipeline:
    """Orchestrate Phase 2.5: load, search, importance, multi-seed, ablate, export.

    All components are injected via the constructor (Dependency Inversion).

    Args:
        config: Validated Phase 2.5 configuration.
        tuner: HyperparameterTuner for search.
        multi_seed_validator: MultiSeedValidator for confidence intervals.
        ablation_runner: AblationRunner for ablation studies.
        exporter: TuningExporter for artifact export.
        project_root: Absolute project root for path resolution.
    """

    def __init__(
        self,
        config: Phase2_5Config,
        tuner: HyperparameterTuner,
        multi_seed_validator: MultiSeedValidator,
        ablation_runner: AblationRunner,
        exporter: TuningExporter,
        project_root: Path,
    ) -> None:
        self._config = config
        self._tuner = tuner
        self._multi_seed = multi_seed_validator
        self._ablation = ablation_runner
        self._exporter = exporter
        self._root = project_root

    def run(self) -> Dict[str, Any]:
        """Execute all pipeline steps and return the report dict."""
        t0 = time.time()
        cfg = self._config

        logger.info("═══════════════════════════════════════════════════")
        logger.info("  Phase 2.5 Fine-Tuning & Ablation (SOLID)")
        logger.info("═══════════════════════════════════════════════════")

        # 1. Reproducibility seeds
        np.random.seed(cfg.random_state)  # noqa: NPY002
        tf.random.set_seed(cfg.random_state)
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        logger.info("  Random state: %d", cfg.random_state)

        # 2. Hardware detection
        hw_info = _detect_hardware()

        # 3. Load Phase 1 data
        X_train, y_train, X_test, y_test = self._load_phase1_data()

        # 4. Hyperparameter search (grid / random / bayesian)
        tuning_results = self._tuner.run(X_train, y_train, X_test, y_test)

        # 5. Parameter importance analysis
        importance_results = compute_importance(
            self._tuner,
            tuning_results.get("trials", []),
            tuning_results.get("metric", "f1_score"),
        )

        # 6. Multi-seed validation of top-K configs
        multi_seed_results = self._multi_seed.validate(
            tuning_results, X_train, y_train, X_test, y_test
        )

        # 7. Ablation study (using best config as baseline)
        base_hp = tuning_results["best_config"]
        ablation_results = self._ablation.run(
            base_hp, X_train, y_train, X_test, y_test
        )

        duration_s = time.time() - t0
        git_commit = _get_git_commit()

        # 8. Export artifacts
        logger.info("── Exporting artifacts ──")
        output_dir = self._root / cfg.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        self._exporter.export_tuning_results(
            tuning_results, cfg.tuning_results_file
        )
        self._exporter.export_ablation_results(
            ablation_results, cfg.ablation_results_file
        )
        self._exporter.export_best_config(
            tuning_results["best_config"], cfg.best_config_file
        )
        self._exporter.export_json(
            importance_results, cfg.importance_file
        )
        self._exporter.export_json(
            multi_seed_results, cfg.multi_seed_file
        )

        report_dict = TuningExporter.build_report(
            tuning_results, ablation_results, importance_results,
            multi_seed_results, hw_info, duration_s, git_commit,
        )
        self._exporter.export_report(report_dict, cfg.report_file)

        # 9. Generate markdown report
        report_md = render_tuning_report(
            tuning_results=tuning_results,
            ablation_results=ablation_results,
            importance_results=importance_results,
            multi_seed_results=multi_seed_results,
            hw_info=hw_info,
            duration_s=duration_s,
            git_commit=git_commit,
        )
        report_dir = self._root / "results" / "phase0_analysis"
        report_path = report_dir / "report_section_tuning.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            f.write(report_md)
        logger.info("  Report saved: %s", report_path.name)

        logger.info("═══════════════════════════════════════════════════")
        logger.info("  Phase 2.5 complete — %.2fs", duration_s)
        logger.info("═══════════════════════════════════════════════════")

        return report_dict

    def _load_phase1_data(self):
        """Load Phase 1 train/test parquets."""
        cfg = self._config
        label_col = cfg.label_column

        train_path = self._root / cfg.phase1_train
        test_path = self._root / cfg.phase1_test

        logger.info("── Loading Phase 1 data ──")
        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)

        feature_names = [c for c in train_df.columns if c != label_col]

        X_train = train_df[feature_names].values.astype(np.float32)
        y_train = train_df[label_col].values
        X_test = test_df[feature_names].values.astype(np.float32)
        y_test = test_df[label_col].values

        logger.info("  Train: %s, Test: %s", X_train.shape, X_test.shape)
        return X_train, y_train, X_test, y_test


# ===================================================================
# Entry Point
# ===================================================================


def main() -> None:
    """Entry point for ``python -m src.phase2_5_fine_tuning.phase2_5.pipeline``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    config_path = PROJECT_ROOT / "config" / "phase2_5_config.yaml"
    config = Phase2_5Config.from_yaml(config_path)

    search_space = SearchSpace(config.search_space, config.random_state)
    evaluator = QuickEvaluator(config.quick_train, random_state=config.random_state)

    output_dir = PROJECT_ROOT / config.output_dir
    pipeline = TuningPipeline(
        config=config,
        tuner=HyperparameterTuner(config, evaluator, search_space),
        multi_seed_validator=MultiSeedValidator(config.multi_seed, evaluator),
        ablation_runner=AblationRunner(config, evaluator),
        exporter=TuningExporter(output_dir),
        project_root=PROJECT_ROOT,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
