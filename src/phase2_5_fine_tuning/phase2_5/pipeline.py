"""Tuning pipeline orchestrator — chains all Phase 2.5 components.

Pipeline:
  Load data → Bayesian TPE Search → Importance → Multi-Seed → Ablation → Export → Report
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
from sklearn.model_selection import train_test_split

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
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, cwd=str(PROJECT_ROOT),
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


def _detect_hardware() -> Dict[str, str]:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        info = {"device": f"GPU: {gpus[0].name}", "cuda": "available"}
    else:
        info = {"device": f"CPU: {platform.processor() or platform.machine()}", "cuda": "N/A"}
    info["tensorflow"] = tf.__version__
    info["python"] = platform.python_version()
    info["platform"] = platform.platform()
    return info


class TuningPipeline:
    """Orchestrate Phase 2.5: load, search, importance, multi-seed, ablate, export.

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
        logger.info("  Phase 2.5 Fine-Tuning & Ablation")
        logger.info("═══════════════════════════════════════════════════")

        np.random.seed(cfg.random_state)
        tf.random.set_seed(cfg.random_state)
        os.environ["TF_DETERMINISTIC_OPS"] = "1"

        hw_info = _detect_hardware()

        # Load SMOTE train + original imbalanced train + imbalanced test
        Xs, ys, Xo, yo, Xt, yt = self._load_data()

        # Bayesian TPE search
        tuning_results = self._tuner.run(Xs, ys, Xo, yo, Xt, yt)

        # Parameter importance
        importance_results = compute_importance(
            self._tuner, tuning_results.get("trials", []), cfg.search_metric,
        )

        # Multi-seed validation (disabled by default)
        multi_seed_results = self._multi_seed.validate(
            tuning_results, Xs, ys, Xt, yt,
        )

        # Ablation study
        ablation_results = self._ablation.run(
            tuning_results["best_config"], Xs, ys, Xt, yt,
        )

        duration_s = time.time() - t0
        git_commit = _get_git_commit()

        # Export
        output_dir = self._root / cfg.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        self._exporter.export_tuning_results(tuning_results, cfg.tuning_results_file)
        self._exporter.export_ablation_results(ablation_results, cfg.ablation_results_file)
        self._exporter.export_best_config(tuning_results["best_config"], cfg.best_config_file)
        self._exporter.export_json(importance_results, cfg.importance_file)
        self._exporter.export_json(multi_seed_results, cfg.multi_seed_file)

        report_dict = TuningExporter.build_report(
            tuning_results, ablation_results, importance_results,
            multi_seed_results, hw_info, duration_s, git_commit,
        )
        self._exporter.export_report(report_dict, cfg.report_file)

        # Markdown report
        report_md = render_tuning_report(
            tuning_results, ablation_results, importance_results,
            multi_seed_results, hw_info, duration_s, git_commit,
        )
        report_path = self._root / "results" / "phase0_analysis" / "report_section_tuning.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            f.write(report_md)

        logger.info("  Phase 2.5 complete — %.2fs", duration_s)
        return report_dict

    def _load_data(self):
        """Load SMOTE train + original imbalanced train + imbalanced test."""
        import joblib

        cfg = self._config
        label_col = cfg.label_column

        # SMOTE-balanced train
        train_df = pd.read_parquet(self._root / cfg.phase1_train)
        test_df = pd.read_parquet(self._root / cfg.phase1_test)
        feature_names = [c for c in train_df.columns if c != label_col]

        Xs = train_df[feature_names].values.astype(np.float32)
        ys = train_df[label_col].values
        Xt = test_df[feature_names].values.astype(np.float32)
        yt = test_df[label_col].values

        # Original imbalanced train
        raw = pd.read_csv(self._root / "data" / "raw" / "wustl-ehms-2020_with_attacks_categories.csv")
        hipaa = ["SrcAddr", "DstAddr", "Sport", "Dport", "SrcMac", "DstMac", "Dir", "Flgs"]
        corr = ["SrcJitter", "pLoss", "Rate", "DstJitter", "Loss", "TotPkts"]
        df = raw.drop(columns=hipaa).drop(columns=[c for c in corr if c in raw.columns])
        if "Attack Category" in df.columns:
            df = df.drop(columns=["Attack Category"])
        df = df.dropna()

        y_all = df[label_col].values
        X_all = df[feature_names].values.astype(np.float32)
        Xo_raw, _, yo, _ = train_test_split(X_all, y_all, test_size=0.3, random_state=42, stratify=y_all)

        scaler = joblib.load(self._root / "models" / "scalers" / "robust_scaler.pkl")
        Xo = scaler.transform(Xo_raw).astype(np.float32)

        logger.info("  SMOTE train: %s, Original train: %s, Test: %s", Xs.shape, Xo.shape, Xt.shape)
        return Xs, ys, Xo, yo, Xt, yt


# ===================================================================
# Entry Point
# ===================================================================

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    config_path = PROJECT_ROOT / "config" / "phase2_5_config.yaml"
    config = Phase2_5Config.from_yaml(config_path)

    weights_path = str(PROJECT_ROOT / "data" / "phase3" / "classification_model.weights.h5")
    search_space = SearchSpace(config.search_space, config.random_state)
    evaluator = QuickEvaluator(
        config.quick_train, random_state=config.random_state,
        pretrained_weights_path=weights_path,
    )

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
