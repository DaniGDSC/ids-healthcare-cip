"""Preprocessing pipeline orchestrator.

Pipeline (matches canonical diagram):
  Step 1:  Identifier sanitization (remove MAC/address columns)
  Step 2:  Encode non-numeric features
  Step 3:  Data cleaning (missing data, outliers)
  Step 4a: Remove unary (zero-variance) features
  Step 4b: Correlation-based redundancy check
  ═══════ LEAKAGE BARRIER ═══════
  Step 5:  Train–test split (stratified 70/30)
  Step 6:  Scaling (fit on train, transform test)
  ═══════ DUAL-TRACK BRANCH ═══════
  Track A: Supervised — SMOTE inside CV pipeline (config exported, not applied)
  Track B: Novelty — benign-only training subset exported
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .artifact_reader import Phase0ArtifactReader
from .config import Phase1Config
from .encoder import CategoricalEncoder
from .exporter import PreprocessingExporter
from .hipaa import HIPAASanitizer
from .missing import MissingValueHandler
from .redundancy import RedundancyRemover
from .report import render_preprocessing_report
from .scaler import RobustScalerTransformer
from .splitter import DataSplitter
from .variance import VarianceFilter

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """Preprocessing pipeline for WUSTL-EHMS-2020.

    Outputs scaled train/test sets ready for dual-track modelling:
      - Track A (supervised): X_train, y_train + SMOTE config
      - Track B (novelty): X_train_benign for autoencoder training

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

        # ── Verify & ingest ──
        self._verify_integrity(cfg)
        df = self._ingest(self._root / cfg.input_dir, cfg.file_pattern)

        # ── Pre-split transforms (Steps 1–4) ──
        df, y_binary, y_multi = self._pre_split_transforms(df, cfg)

        # ══════════════════ LEAKAGE BARRIER ═══════════════════════════

        feat_names = df.columns.tolist()
        X = df.values.astype(np.float32)

        # ── Step 5: Train–test split ──
        splitter = DataSplitter(
            test_ratio=cfg.test_ratio,
            random_state=cfg.random_state,
            label_column=cfg.label_column,
            multi_label_column=cfg.multi_label_column,
        )
        # Reassemble DataFrame with labels for DataSplitter
        split_df = pd.DataFrame(X, columns=feat_names)
        split_df[cfg.label_column] = y_binary
        if y_multi is not None:
            split_df[cfg.multi_label_column] = y_multi

        X_train, X_test, y_train, y_test, feat_names, y_multi_train, y_multi_test = (
            splitter.split(split_df)
        )
        self._report["split"] = splitter.get_report()

        # ── Step 6: Scaling (fit on TRAIN, transform TEST) ──
        scaler = RobustScalerTransformer(method=cfg.scaling_method)
        X_train, X_test = scaler.scale_both(X_train, X_test)
        self._report["scaling"] = scaler.get_report()

        # ── Dual-track branch & export ──
        self._report["elapsed_seconds"] = round(time.perf_counter() - t0, 2)
        self._report["random_state"] = cfg.random_state
        self._build_track_reports(y_train, cfg)
        self._export(
            X_train, X_test, y_train, y_test,
            y_multi_train, y_multi_test,
            feat_names, scaler, cfg,
        )

        self._log_summary()
        return self._report

    def get_report(self) -> Dict[str, Any]:
        return dict(self._report)

    # ------------------------------------------------------------------
    # Pre-split transforms (Steps 1–4)
    # ------------------------------------------------------------------

    def _pre_split_transforms(
        self,
        df: pd.DataFrame,
        cfg: Phase1Config,
    ) -> Tuple[pd.DataFrame, np.ndarray, "np.ndarray | None"]:
        """Steps 1–4: sanitize, encode, clean, filter."""

        # Step 1: Identifier sanitization
        sanitizer = HIPAASanitizer(cfg.id_removal_columns)
        df = sanitizer.transform(df)
        self._report["identifier_removal"] = sanitizer.get_report()

        # Separate labels before feature transforms
        y_binary = df[cfg.label_column].values
        has_multi = cfg.multi_label_column in df.columns
        y_multi = df[cfg.multi_label_column].values if has_multi else None
        label_cols = [cfg.label_column]
        if has_multi:
            label_cols.append(cfg.multi_label_column)
        df = df.drop(columns=label_cols)

        self._report["label_separation"] = {
            "y_binary_column": cfg.label_column,
            "y_multi_column": cfg.multi_label_column if has_multi else None,
            "n_samples": len(y_binary),
        }

        # Step 2: Encode non-numeric features
        encoder = CategoricalEncoder(
            label_encode=cfg.label_encode_columns,
            parse_numeric=cfg.parse_numeric_columns,
            sentinel=cfg.parse_numeric_sentinel,
        )
        df = encoder.transform(df)
        self._report["encoding"] = encoder.get_report()

        # Step 3: Data cleaning
        handler = MissingValueHandler(
            biometric_columns=cfg.biometric_columns,
            label_column=cfg.label_column,
            biometric_strategy=cfg.biometric_strategy,
            network_strategy=cfg.network_strategy,
        )
        df = handler.transform(df)
        self._report["cleaning"] = handler.get_report()

        # Step 4a: Variance filtering
        if cfg.variance_enabled:
            var_filter = VarianceFilter(max_unique=cfg.variance_max_unique)
            df = var_filter.transform(df)
            self._report["variance"] = var_filter.get_report()

        # Step 4b: Correlation-based redundancy
        if cfg.correlation_enabled:
            corr_df = self._reader.read_correlations()
            remover = RedundancyRemover(corr_df, cfg.correlation_threshold)
            df = remover.transform(df)
            self._report["redundancy"] = remover.get_report()

        return df, y_binary, y_multi

    # ------------------------------------------------------------------
    # Dual-track reports
    # ------------------------------------------------------------------

    def _build_track_reports(
        self,
        y_train: np.ndarray,
        cfg: Phase1Config,
    ) -> None:
        """Populate Track A / Track B report sections."""
        benign_mask = y_train == 0
        self._report["track_b"] = {
            "enabled": cfg.track_b_enabled,
            "benign_train_samples": int(benign_mask.sum()),
            "attack_train_samples": int((~benign_mask).sum()),
        }
        logger.info(
            "Track B — Benign-only train: %d samples (%.1f%% of train)",
            benign_mask.sum(), benign_mask.mean() * 100,
        )
        self._report["track_a"] = {
            "smote_enabled": cfg.smote_enabled,
            "smote_strategy": cfg.smote_strategy,
            "smote_k_neighbors": cfg.smote_k_neighbors,
            "note": "SMOTE applied inside CV pipeline, not during preprocessing",
        }

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _export(
        self,
        X_train_s: np.ndarray,
        X_test_s: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        y_multi_train: np.ndarray,
        y_multi_test: np.ndarray,
        feat_names: List[str],
        scaler: RobustScalerTransformer,
        cfg: Phase1Config,
    ) -> None:
        """Write all pipeline artifacts to disk."""
        output_dir = self._root / cfg.output_dir
        scaler_dir = self._root / "models" / "scalers"
        exporter = PreprocessingExporter(
            output_dir, scaler_dir, cfg.label_column, cfg.multi_label_column,
        )

        exporter.export_parquet(
            X_train_s, y_train, feat_names, cfg.train_parquet,
            y_multi=y_multi_train,
        )
        exporter.export_parquet(
            X_test_s, y_test, feat_names, cfg.test_parquet,
            y_multi=y_multi_test,
        )
        if cfg.track_b_enabled:
            benign_mask = y_train == 0
            exporter.export_parquet(
                X_train_s[benign_mask],
                np.zeros(int(benign_mask.sum()), dtype=int),
                feat_names,
                cfg.train_benign_parquet,
            )

        exporter.export_scaler(scaler, cfg.scaler_file)

        self._report["output"] = {
            "feature_names": feat_names,
            "n_features": len(feat_names),
        }
        exporter.export_report(self._report, cfg.report_file)

        md = render_preprocessing_report(self._report)
        md_path = self._root / "results" / "phase0_analysis" / "report_section_preprocessing.md"
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(md, encoding="utf-8")
        logger.info("Thesis report → %s", md_path)

    # ------------------------------------------------------------------
    # Ingestion & integrity
    # ------------------------------------------------------------------

    def _verify_integrity(self, cfg: Phase1Config) -> None:
        data_dir = self._root / cfg.input_dir
        csv_files = sorted(data_dir.glob(cfg.file_pattern))
        if csv_files:
            sha = self._reader.verify_integrity(csv_files[0])
            self._report["integrity"] = {"sha256": sha, "verified": True}

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

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _log_summary(self) -> None:
        sep = "=" * 72
        ing = self._report.get("ingestion", {})
        idr = self._report.get("identifier_removal", {})
        cl = self._report.get("cleaning", {})
        var = self._report.get("variance", {})
        red = self._report.get("redundancy", {})
        spl = self._report.get("split", {})
        tb = self._report.get("track_b", {})
        out = self._report.get("output", {})

        logger.info("")
        logger.info(sep)
        logger.info("PHASE 1 — PREPROCESSING SUMMARY")
        logger.info(sep)
        logger.info("  Ingestion    : %d files → %d × %d",
                     ing.get("files_loaded", 0), ing.get("raw_rows", 0),
                     ing.get("raw_columns", 0))
        logger.info("  Identifiers  : %d columns dropped", idr.get("n_dropped", 0))
        logger.info("  Cleaning     : %d bio cells filled, %d rows dropped",
                     cl.get("biometric_cells_filled", 0), cl.get("rows_dropped", 0))
        logger.info("  Variance     : %d features dropped",
                     var.get("n_dropped", 0))
        logger.info("  Redundancy   : %d features dropped (|r| ≥ %.2f)",
                     red.get("n_dropped", 0), red.get("threshold", 0))
        logger.info("  Split        : train=%d, test=%d",
                     spl.get("train_samples", 0), spl.get("test_samples", 0))
        logger.info("  Track A      : SMOTE inside CV pipeline")
        logger.info("  Track B      : %d benign-only samples",
                     tb.get("benign_train_samples", 0))
        logger.info("  Features     : %d", out.get("n_features", 0))
        logger.info("  Elapsed      : %.2f s", self._report.get("elapsed_seconds", 0))
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
