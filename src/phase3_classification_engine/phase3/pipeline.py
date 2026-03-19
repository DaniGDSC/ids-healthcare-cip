"""Classification pipeline — orchestrates Phase 3 end-to-end.

Components are injected via constructor (Dependency Inversion).
Reuses Phase 2 SOLID components for model rebuild and reshaping.

Usage::

    python -m src.phase3_classification_engine.phase3.pipeline
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
import tensorflow as tf

# ── Phase 2 SOLID components (reused, NOT reimplemented) ──────────
from src.phase2_detection_engine.phase2.assembler import DetectionModelAssembler
from src.phase2_detection_engine.phase2.attention_builder import (  # noqa: F401
    AttentionBuilder,
    BahdanauAttention,
)
from src.phase2_detection_engine.phase2.bilstm_builder import BiLSTMBuilder
from src.phase2_detection_engine.phase2.cnn_builder import CNNBuilder
from src.phase2_detection_engine.phase2.reshaper import DataReshaper

from .artifact_reader import Phase2ArtifactReader
from .base import BaseClassificationHead
from .config import Phase3Config
from .evaluator import ModelEvaluator
from .exporter import ClassificationExporter
from .report import render_classification_report
from .trainer import ClassificationTrainer
from .unfreezer import ProgressiveUnfreezer

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
        logger.info("  Training on GPU: %s", device_name)
        cuda_version = getattr(tf.sysconfig, "get_build_info", lambda: {})()
        info = {
            "device": f"GPU: {device_name}",
            "cuda": cuda_version.get("cuda_version", "N/A"),
        }
    else:
        cpu_info = platform.processor() or platform.machine()
        logger.info("  CPU fallback: %s", cpu_info)
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        info = {"device": f"CPU: {cpu_info}", "cuda": "N/A (CPU execution)"}

    info["tensorflow"] = tf.__version__
    info["python"] = platform.python_version()
    info["platform"] = platform.platform()
    return info


class ClassificationPipeline:
    """Orchestrate Phase 3: load → verify → reshape → build → train → eval → export.

    All components are injected via the constructor (Dependency Inversion).

    Args:
        config: Validated Phase 3 configuration.
        artifact_reader: Phase 2 artifact reader.
        head: Classification head builder.
        unfreezer: Progressive unfreezer.
        trainer: Classification trainer.
        evaluator: Model evaluator.
        exporter: Classification exporter.
        project_root: Absolute project root for path resolution.
    """

    def __init__(
        self,
        config: Phase3Config,
        artifact_reader: Phase2ArtifactReader,
        head: BaseClassificationHead,
        unfreezer: ProgressiveUnfreezer,
        trainer: ClassificationTrainer,
        evaluator: ModelEvaluator,
        exporter: ClassificationExporter,
        project_root: Path,
    ) -> None:
        self._config = config
        self._reader = artifact_reader
        self._head = head
        self._unfreezer = unfreezer
        self._trainer = trainer
        self._evaluator = evaluator
        self._exporter = exporter
        self._root = project_root

    def run(self) -> Dict[str, Any]:
        """Execute all pipeline steps and return the metrics dict.

        Returns:
            Evaluation metrics dict.
        """
        t0 = time.time()
        cfg = self._config

        logger.info("═══════════════════════════════════════════════════")
        logger.info("  Phase 3 Classification Engine (SOLID)")
        logger.info("═══════════════════════════════════════════════════")

        # 1. Reproducibility seeds
        np.random.seed(cfg.random_state)  # noqa: NPY002
        tf.random.set_seed(cfg.random_state)
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        logger.info("  Random state: %d", cfg.random_state)

        # 2. Hardware detection
        hw_info = _detect_hardware()

        # 3. Verify Phase 2 artifacts (SHA-256)
        weights_path, metadata = self._reader.load_and_verify()

        # 4. Load Phase 1 data
        train_path = self._root / cfg.phase1_train
        test_path = self._root / cfg.phase1_test
        X_train, y_train, X_test, y_test, _ = self._reader.load_phase1_data(train_path, test_path)

        # 5. Reshape (sliding windows) — reuse Phase 2 DataReshaper
        hp = metadata["hyperparameters"]
        reshaper = DataReshaper(timesteps=hp["timesteps"], stride=hp["stride"])
        X_train_w, y_train_w = reshaper.reshape(X_train, y_train)
        X_test_w, y_test_w = reshaper.reshape(X_test, y_test)

        # 6. Rebuild detection model (Phase 2 builders)
        detection_model = self._rebuild_detection_model(metadata, weights_path)
        detection_params = detection_model.count_params()

        # 7. Auto classification head
        n_classes = len(np.unique(y_train_w))
        logger.info("  Auto-detected %d classes", n_classes)

        # 8. Build full model
        output_tensor = self._head.build(detection_model.output, n_classes)
        full_model = tf.keras.Model(
            detection_model.input, output_tensor, name="classification_engine"
        )
        loss = self._head.get_loss(n_classes)

        total_params = full_model.count_params()
        head_params = total_params - detection_params
        logger.info("  Total params: %d (head: %d)", total_params, head_params)

        # 9. Create output directory
        output_dir = self._root / cfg.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # 10. Progressive unfreezing training
        histories = self._trainer.train_all_phases(
            model=full_model,
            phases=cfg.training_phases,
            unfreezer=self._unfreezer,
            X_train=X_train_w,
            y_train=y_train_w,
            loss=loss,
            output_dir=output_dir,
        )

        # 11. Evaluate on test set
        metrics = self._evaluator.evaluate(full_model, X_test_w, y_test_w)

        duration_s = time.time() - t0
        git_commit = _get_git_commit()

        # 12. Export artifacts
        logger.info("── Exporting artifacts ──")
        metrics_report = ClassificationExporter.build_metrics_report(
            metrics, full_model, hw_info, duration_s, git_commit
        )
        self._exporter.export_model_weights(full_model, cfg.model_file)
        self._exporter.export_metrics(metrics_report, cfg.metrics_file)

        cm = metrics["confusion_matrix"]
        n = len(cm)
        labels = ["Normal", "Attack"] if n == 2 else [str(i) for i in range(n)]
        self._exporter.export_confusion_matrix(cm, labels, cfg.confusion_matrix_file)
        self._exporter.export_history(histories, cfg.history_file)

        # 13. Generate report
        report_md = render_classification_report(
            model=full_model,
            metrics=metrics,
            histories=histories,
            config=cfg,
            hw_info=hw_info,
            duration_s=duration_s,
            detection_params=detection_params,
            git_commit=git_commit,
        )
        report_dir = self._root / "results" / "phase0_analysis"
        report_path = report_dir / "report_section_classification.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            f.write(report_md)
        logger.info("  Report saved: %s", report_path.name)

        logger.info("═══════════════════════════════════════════════════")
        logger.info("  Phase 3 complete — %.2fs", duration_s)
        logger.info("═══════════════════════════════════════════════════")

        return metrics

    @staticmethod
    def _rebuild_detection_model(metadata: Dict[str, Any], weights_path: Path) -> tf.keras.Model:
        """Rebuild Phase 2 detection model architecture and load weights."""
        logger.info("── Rebuilding detection model ──")

        hp = metadata["hyperparameters"]
        builders = [
            CNNBuilder(
                filters_1=hp["cnn_filters_1"],
                filters_2=hp["cnn_filters_2"],
                kernel_size=hp["cnn_kernel_size"],
                activation=hp["cnn_activation"],
                pool_size=hp["cnn_pool_size"],
            ),
            BiLSTMBuilder(
                units_1=hp["bilstm_units_1"],
                units_2=hp["bilstm_units_2"],
                dropout_rate=hp["dropout_rate"],
            ),
            AttentionBuilder(units=hp["attention_units"]),
        ]

        assembler = DetectionModelAssembler(
            timesteps=hp["timesteps"],
            n_features=29,
            builders=builders,
        )
        detection_model = assembler.assemble()
        detection_model.load_weights(str(weights_path))
        logger.info("  Loaded weights from %s", weights_path.name)
        logger.info("  Detection model params: %d", detection_model.count_params())
        return detection_model


# ===================================================================
# Entry Point
# ===================================================================


def main() -> None:
    """Entry point for ``python -m src.phase3_classification_engine.phase3.pipeline``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    config_path = PROJECT_ROOT / "config" / "phase3_config.yaml"
    config = Phase3Config.from_yaml(config_path)

    reader = Phase2ArtifactReader(
        project_root=PROJECT_ROOT,
        phase2_dir=config.phase2_dir,
        metadata_file=config.phase2_metadata,
        label_column=config.label_column,
    )

    from .head import AutoClassificationHead

    head = AutoClassificationHead(
        dense_units=config.dense_units,
        dense_activation=config.dense_activation,
        dropout_rate=config.head_dropout_rate,
    )

    pipeline = ClassificationPipeline(
        config=config,
        artifact_reader=reader,
        head=head,
        unfreezer=ProgressiveUnfreezer(),
        trainer=ClassificationTrainer(
            batch_size=config.batch_size,
            validation_split=config.validation_split,
            early_stopping_patience=config.early_stopping_patience,
            reduce_lr_patience=config.reduce_lr_patience,
            reduce_lr_factor=config.reduce_lr_factor,
        ),
        evaluator=ModelEvaluator(threshold=config.threshold),
        exporter=ClassificationExporter(PROJECT_ROOT / config.output_dir),
        project_root=PROJECT_ROOT,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
