"""Detection pipeline orchestrator — chains all components.

Six-step pipeline: Load → Reshape → Build → Forward → Export → Report.
Fails fast if Phase 1 artifacts are missing or corrupt.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import tensorflow as tf

from .artifact_reader import Phase1ArtifactReader
from .assembler import DetectionModelAssembler
from .attention_builder import AttentionBuilder
from .bilstm_builder import BiLSTMBuilder
from .cnn_builder import CNNBuilder
from .config import Phase2Config
from .exporter import DetectionExporter
from .report import render_detection_report
from .reshaper import DataReshaper

logger = logging.getLogger(__name__)

_PREDICT_BATCH_SIZE: int = 256


class DetectionPipeline:
    """Six-step detection pipeline for WUSTL-EHMS-2020.

    Components are injected via ``Phase2Config`` and
    ``Phase1ArtifactReader`` (Dependency Inversion).

    Args:
        config: Validated Phase 2 configuration.
        artifact_reader: Reader for Phase 1 preprocessing artifacts.
        project_root: Absolute project root for path resolution.
    """

    def __init__(
        self,
        config: Phase2Config,
        artifact_reader: Phase1ArtifactReader,
        project_root: Path,
    ) -> None:
        self._config = config
        self._reader = artifact_reader
        self._root = project_root

    def run(self) -> Dict[str, Any]:
        """Execute all pipeline steps and return the report dict.

        Returns:
            Detection report dict (also saved to disk).
        """
        t0 = time.perf_counter()
        cfg = self._config

        # Reproducibility
        tf.random.set_seed(cfg.random_state)
        np.random.seed(cfg.random_state)

        logger.info("Phase 2 Detection Engine — starting")

        # ── Step 1: Load & Verify ──
        X_train, y_train, X_test, y_test, feature_names = (
            self._reader.load_and_verify()
        )
        n_features = len(feature_names)

        # ── Step 2: Reshape ──
        reshaper = DataReshaper(cfg.timesteps, cfg.stride)
        X_train_w, y_train_w = reshaper.reshape(X_train, y_train)
        X_test_w, y_test_w = reshaper.reshape(X_test, y_test)

        # ── Step 3: Build Model ──
        builders = [
            CNNBuilder(
                filters_1=cfg.cnn_filters_1,
                filters_2=cfg.cnn_filters_2,
                kernel_size=cfg.cnn_kernel_size,
                activation=cfg.cnn_activation,
                pool_size=cfg.cnn_pool_size,
            ),
            BiLSTMBuilder(
                units_1=cfg.bilstm_units_1,
                units_2=cfg.bilstm_units_2,
                dropout_rate=cfg.dropout_rate,
            ),
            AttentionBuilder(units=cfg.attention_units),
        ]

        assembler = DetectionModelAssembler(
            timesteps=cfg.timesteps,
            n_features=n_features,
            builders=builders,
        )
        model = assembler.assemble()
        model.summary(print_fn=logger.info)

        # ── Step 4: Forward Pass ──
        batch_size = cfg.predict_batch_size
        train_context = model.predict(
            X_train_w, batch_size=batch_size, verbose=0,
        )
        test_context = model.predict(
            X_test_w, batch_size=batch_size, verbose=0,
        )
        logger.info("Train context: %s", train_context.shape)
        logger.info("Test context:  %s", test_context.shape)

        # ── Step 5: Export ──
        elapsed = time.perf_counter() - t0
        output_dir = self._root / cfg.output_dir
        exporter = DetectionExporter(output_dir, cfg.label_column)

        exporter.export_model_weights(model, cfg.model_file)
        exporter.export_attention_vectors(
            train_context, test_context,
            y_train_w, y_test_w,
            cfg.attention_parquet,
        )

        # Build report dict
        hp_dict = self._build_hyperparameters(cfg, reshaper, assembler)
        report = DetectionExporter.build_report(
            model=model,
            config_dict=hp_dict,
            feature_names=feature_names,
            train_context=train_context,
            test_context=test_context,
            train_windows_shape=X_train_w.shape,
            test_windows_shape=X_test_w.shape,
            elapsed=elapsed,
        )
        exporter.export_report(report, cfg.report_json)

        # ── Step 6: Thesis Report ──
        md = render_detection_report(report)
        md_path = (
            self._root / "results" / "phase0_analysis"
            / "report_section_detection.md"
        )
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(md, encoding="utf-8")
        logger.info("Thesis report → %s", md_path)

        self._log_summary(report)
        return report

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_hyperparameters(
        cfg: Phase2Config,
        reshaper: DataReshaper,
        assembler: DetectionModelAssembler,
    ) -> Dict[str, Any]:
        """Collect hyperparameters from config and components."""
        return {
            "timesteps": cfg.timesteps,
            "stride": cfg.stride,
            "cnn_filters_1": cfg.cnn_filters_1,
            "cnn_filters_2": cfg.cnn_filters_2,
            "cnn_kernel_size": cfg.cnn_kernel_size,
            "cnn_activation": cfg.cnn_activation,
            "cnn_pool_size": cfg.cnn_pool_size,
            "bilstm_units_1": cfg.bilstm_units_1,
            "bilstm_units_2": cfg.bilstm_units_2,
            "dropout_rate": cfg.dropout_rate,
            "attention_units": cfg.attention_units,
            "random_state": cfg.random_state,
        }

    @staticmethod
    def _log_summary(report: Dict[str, Any]) -> None:
        """Log a concise pipeline summary."""
        sep = "=" * 72
        shapes = report.get("shapes", {})
        logger.info("")
        logger.info(sep)
        logger.info("PHASE 2 — DETECTION ENGINE SUMMARY")
        logger.info(sep)
        logger.info("  Architecture : %s", report.get("architecture", "—"))
        logger.info("  Parameters   : %d", report.get("total_parameters", 0))
        logger.info("  Output dim   : %d", report.get("output_dim", 0))
        logger.info("  Train windows: %s", shapes.get("train_windows", "—"))
        logger.info("  Test windows : %s", shapes.get("test_windows", "—"))
        logger.info("  Train context: %s", shapes.get("train_context", "—"))
        logger.info("  Test context : %s", shapes.get("test_context", "—"))
        logger.info(
            "  Elapsed      : %.2f s",
            report.get("elapsed_seconds", 0),
        )
        logger.info(sep)


# ======================================================================
# Entry Point
# ======================================================================

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent.parent


def main() -> None:
    """Run the Phase 2 detection pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config_path = PROJECT_ROOT / "config" / "phase2_config.yaml"
    config = Phase2Config.from_yaml(config_path)

    reader = Phase1ArtifactReader(
        project_root=PROJECT_ROOT,
        train_parquet=config.train_parquet,
        test_parquet=config.test_parquet,
        metadata_file=config.metadata_file,
        report_file=config.report_file,
        label_column=config.label_column,
    )

    pipeline = DetectionPipeline(config, reader, PROJECT_ROOT)
    pipeline.run()


if __name__ == "__main__":
    main()
