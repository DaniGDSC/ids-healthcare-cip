"""Preprocessing pipeline orchestrator — chains all transformers.

Each step logs input → output shape.  Fails fast if Phase 0
artifacts are missing.
"""

from __future__ import annotations

import glob as globmod
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml

from .artifact_reader import Phase0ArtifactReader
from .config import Phase1Config
from .exporter import PreprocessingExporter
from .hipaa import HIPAASanitizer
from .missing import MissingValueHandler
from .redundancy import RedundancyRemover
from .report import render_preprocessing_report
from .scaler import RobustScalerTransformer
from .smote import SMOTEBalancer
from .splitter import DataSplitter

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """Seven-step preprocessing pipeline for WUSTL-EHMS-2020.

    Components are injected via ``Phase1Config`` and
    ``Phase0ArtifactReader`` (Dependency Inversion).

    Args:
        config: Validated Phase 1 configuration.
        artifact_reader: Reader for Phase 0 analysis artifacts.
        project_root: Absolute project root for path resolution.
    """

    def __init__(
        self,
        config: Phase1Config,
        artifact_reader: Phase0ArtifactReader,
        project_root: Path,
    ) -> None:
        self._config = config
        self._reader = artifact_reader
        self._root = project_root
        self._report: Dict[str, Any] = {}

    def run(self) -> Dict[str, Any]:
        """Execute all pipeline steps and return the report dict."""
        t0 = time.perf_counter()
        cfg = self._config

        # ── Verify integrity ──
        data_dir = self._root / cfg.input_dir
        csv_files = sorted(data_dir.glob(cfg.file_pattern))
        if csv_files:
            sha = self._reader.verify_integrity(csv_files[0])
            self._report["integrity"] = {"sha256": sha, "verified": True}

        # ── Step 1: Ingest ──
        df = self._ingest(data_dir, cfg.file_pattern)

        # ── Step 2: HIPAA ──
        hipaa = HIPAASanitizer(cfg.hipaa_columns)
        df = hipaa.transform(df)
        self._report["hipaa"] = hipaa.get_report()

        # ── Step 3: Missing values ──
        handler = MissingValueHandler(
            biometric_columns=cfg.biometric_columns,
            label_column=cfg.label_column,
            biometric_strategy=cfg.biometric_strategy,
            network_strategy=cfg.network_strategy,
        )
        df = handler.transform(df)
        self._report["missing_values"] = handler.get_report()

        # ── Step 4: Redundancy (Phase 0 artifact) ──
        corr_df = self._reader.read_correlations()
        remover = RedundancyRemover(
            corr_df, cfg.correlation_threshold, cfg.label_column,
        )
        df = remover.transform(df)
        self._report["redundancy"] = remover.get_report()

        # ── Step 5: Stratified split ──
        splitter = DataSplitter(
            test_ratio=cfg.test_ratio,
            random_state=cfg.random_state,
            label_column=cfg.label_column,
        )
        X_train, X_test, y_train, y_test, feat_names = splitter.split(df)
        self._report["split"] = splitter.get_report()

        # ── Step 6: SMOTE (train only) ──
        balancer = SMOTEBalancer(
            strategy=cfg.smote_strategy,
            k_neighbors=cfg.smote_k_neighbors,
            random_state=cfg.random_state,
        )
        X_train, y_train = balancer.resample(X_train, y_train)
        self._report["smote"] = balancer.get_report()

        # ── Step 7: Robust scaling ──
        scaler = RobustScalerTransformer(method=cfg.scaling_method)
        X_train_s, X_test_s = scaler.scale_both(X_train, X_test)
        self._report["scaling"] = scaler.get_report()

        # ── Export ──
        elapsed = time.perf_counter() - t0
        self._report["elapsed_seconds"] = round(elapsed, 2)
        self._report["random_state"] = cfg.random_state

        output_dir = self._root / cfg.output_dir
        scaler_dir = self._root / "models" / "scalers"
        exporter = PreprocessingExporter(
            output_dir, scaler_dir, cfg.label_column,
        )

        exporter.export_parquet(X_train_s, y_train, feat_names, cfg.train_parquet)
        exporter.export_parquet(X_test_s, y_test, feat_names, cfg.test_parquet)
        exporter.export_scaler(scaler, cfg.scaler_file)

        self._report["output"] = {
            "feature_names": feat_names,
            "n_features": len(feat_names),
        }
        exporter.export_report(self._report, cfg.report_file)

        # ── Thesis report ──
        md = render_preprocessing_report(self._report)
        md_path = self._root / "results" / "phase0_analysis" / "report_section_preprocessing.md"
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(md, encoding="utf-8")
        logger.info("Thesis report → %s", md_path)

        self._log_summary()
        return self._report

    def get_report(self) -> Dict[str, Any]:
        """Return the accumulated pipeline report."""
        return dict(self._report)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _ingest(self, data_dir: Path, file_pattern: str) -> pd.DataFrame:
        csv_files = sorted(data_dir.glob(file_pattern))
        if not csv_files:
            raise FileNotFoundError(
                f"No files matching '{file_pattern}' in {data_dir}."
            )
        frames: List[pd.DataFrame] = []
        for path in csv_files:
            df = pd.read_csv(path, low_memory=False)
            logger.info("  Loaded %s: %d × %d", path.name, *df.shape)
            frames.append(df)

        combined = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
        self._report["ingestion"] = {
            "files_loaded": len(csv_files),
            "raw_rows": int(combined.shape[0]),
            "raw_columns": int(combined.shape[1]),
        }
        logger.info("Ingestion: %d rows × %d cols", *combined.shape)
        return combined

    def _log_summary(self) -> None:
        sep = "=" * 72
        ing = self._report.get("ingestion", {})
        hip = self._report.get("hipaa", {})
        mv = self._report.get("missing_values", {})
        red = self._report.get("redundancy", {})
        spl = self._report.get("split", {})
        smt = self._report.get("smote", {})
        out = self._report.get("output", {})

        logger.info("")
        logger.info(sep)
        logger.info("PHASE 1 — PREPROCESSING SUMMARY")
        logger.info(sep)
        logger.info("  Ingestion   : %d files → %d × %d",
                     ing.get("files_loaded", 0), ing.get("raw_rows", 0),
                     ing.get("raw_columns", 0))
        logger.info("  HIPAA       : %d columns dropped", hip.get("n_dropped", 0))
        logger.info("  Missing     : %d bio cells filled, %d rows dropped",
                     mv.get("biometric_cells_filled", 0), mv.get("rows_dropped", 0))
        logger.info("  Redundancy  : %d features (|r| ≥ %.2f)",
                     red.get("n_dropped", 0), red.get("threshold", 0))
        logger.info("  Split       : train=%d, test=%d",
                     spl.get("train_samples", 0), spl.get("test_samples", 0))
        logger.info("  SMOTE       : %d → %d (+%d)",
                     smt.get("samples_before", 0), smt.get("samples_after", 0),
                     smt.get("synthetic_added", 0))
        logger.info("  Features    : %d", out.get("n_features", 0))
        logger.info("  Elapsed     : %.2f s", self._report.get("elapsed_seconds", 0))
        logger.info(sep)


# ======================================================================
# Entry Point
# ======================================================================

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent.parent


def main() -> None:
    """Run the Phase 1 preprocessing pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config_path = PROJECT_ROOT / "config" / "phase1_config.yaml"
    config = Phase1Config.from_yaml(config_path)

    reader = Phase0ArtifactReader(
        project_root=PROJECT_ROOT,
        stats_file=config.phase0_stats_file,
        corr_file=config.phase0_corr_file,
        integrity_file=config.phase0_integrity_file,
    )

    pipeline = PreprocessingPipeline(config, reader, PROJECT_ROOT)
    pipeline.run()


if __name__ == "__main__":
    main()
