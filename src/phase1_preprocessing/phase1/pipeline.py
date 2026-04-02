"""Preprocessing pipeline orchestrator.

Pipeline (matches canonical diagram):
  Step 1:  Identifier sanitization (remove MAC/address columns)
  Step 2:  Encode non-numeric features
  Step 3:  Data cleaning (missing data, outliers)
  Step 4a: Remove unary (zero-variance) features
  Step 4b: Correlation-based redundancy check
  ═══════ LEAKAGE BARRIER ═══════
  Step 5a: Train–test split (stratified 70/30)
  Step 5b: RFECV + SHAP validation (train only)
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
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from .artifact_reader import Phase0ArtifactReader
from .config import Phase1Config
from .encoder import CategoricalEncoder
from .exporter import PreprocessingExporter
from .hipaa import HIPAASanitizer
from .missing import MissingValueHandler
from .redundancy import RedundancyRemover
from .report import render_preprocessing_report
from .scaler import RobustScalerTransformer
from .shap_selector import SHAPSelector
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

        # ── Verify integrity ──
        data_dir = self._root / cfg.input_dir
        csv_files = sorted(data_dir.glob(cfg.file_pattern))
        if csv_files:
            sha = self._reader.verify_integrity(csv_files[0])
            self._report["integrity"] = {"sha256": sha, "verified": True}

        # ── Ingest ──
        df = self._ingest(data_dir, cfg.file_pattern)

        # ══════════════════════════════════════════════════════════════
        # Step 1: Identifier sanitization
        # ══════════════════════════════════════════════════════════════
        sanitizer = HIPAASanitizer(cfg.id_removal_columns)
        df = sanitizer.transform(df)
        self._report["identifier_removal"] = sanitizer.get_report()

        # Separate labels before any feature transforms
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

        # ══════════════════════════════════════════════════════════════
        # Step 2: Encode non-numeric features
        # ══════════════════════════════════════════════════════════════
        encoder = CategoricalEncoder(
            label_encode=cfg.label_encode_columns,
            parse_numeric=cfg.parse_numeric_columns,
            sentinel=cfg.parse_numeric_sentinel,
        )
        df = encoder.transform(df)
        self._report["encoding"] = encoder.get_report()

        # ══════════════════════════════════════════════════════════════
        # Step 3: Data cleaning
        # ══════════════════════════════════════════════════════════════
        handler = MissingValueHandler(
            biometric_columns=cfg.biometric_columns,
            label_column=cfg.label_column,
            biometric_strategy=cfg.biometric_strategy,
            network_strategy=cfg.network_strategy,
        )
        df = handler.transform(df)
        self._report["cleaning"] = handler.get_report()

        # ══════════════════════════════════════════════════════════════
        # Step 4a: Remove unary (zero-variance) features
        # ══════════════════════════════════════════════════════════════
        if cfg.variance_enabled:
            var_filter = VarianceFilter(max_unique=cfg.variance_max_unique)
            df = var_filter.transform(df)
            self._report["variance"] = var_filter.get_report()

        # ══════════════════════════════════════════════════════════════
        # Step 4b: Correlation-based redundancy check
        # ══════════════════════════════════════════════════════════════
        if cfg.correlation_enabled:
            corr_df = self._reader.read_correlations()
            remover = RedundancyRemover(corr_df, cfg.correlation_threshold)
            df = remover.transform(df)
            self._report["redundancy"] = remover.get_report()

        # ══════════════════ LEAKAGE BARRIER ═══════════════════════════

        feat_names = df.columns.tolist()
        X = df.values.astype(np.float32)

        # ══════════════════════════════════════════════════════════════
        # Step 5a: Train–test split
        # ══════════════════════════════════════════════════════════════
        stratify_on = y_multi if (cfg.stratify and y_multi is not None) else y_binary

        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=cfg.test_ratio,
            random_state=cfg.random_state,
        )
        train_idx, test_idx = next(sss.split(X, stratify_on))

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_binary[train_idx], y_binary[test_idx]
        y_multi_train = y_multi[train_idx] if y_multi is not None else np.array([], dtype=object)
        y_multi_test = y_multi[test_idx] if y_multi is not None else np.array([], dtype=object)

        self._report["split"] = {
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "train_ratio": round(1 - cfg.test_ratio, 2),
            "test_ratio": cfg.test_ratio,
            "stratified": cfg.stratify,
            "stratify_column": cfg.multi_label_column if y_multi is not None else cfg.label_column,
            "train_attack_rate": round(float(y_train.mean()), 4),
            "test_attack_rate": round(float(y_test.mean()), 4),
        }
        logger.info(
            "Step 5a — Split: train=%d (attack=%.1f%%) | test=%d (attack=%.1f%%)",
            len(X_train), y_train.mean() * 100,
            len(X_test), y_test.mean() * 100,
        )

        # ══════════════════════════════════════════════════════════════
        # Step 5b: RFECV + SHAP validation (train only)
        # ══════════════════════════════════════════════════════════════
        if cfg.shap_enabled:
            shap_sel = SHAPSelector(
                min_features=cfg.shap_min_features,
                n_estimators=cfg.shap_n_estimators,
                cv_folds=cfg.shap_cv_folds,
                shap_threshold=cfg.shap_threshold,
                random_state=cfg.random_state,
            )
            X_train, X_test, feat_names = shap_sel.select(
                X_train, X_test, y_train, feat_names,
            )
            self._report["shap_selection"] = shap_sel.get_report()

        # ══════════════════════════════════════════════════════════════
        # Step 6: Scaling (fit on TRAIN, transform TEST)
        # ══════════════════════════════════════════════════════════════
        scaler = RobustScalerTransformer(method=cfg.scaling_method)
        X_train_s, X_test_s = scaler.scale_both(X_train, X_test)
        self._report["scaling"] = scaler.get_report()

        # ══════════════════ DUAL-TRACK BRANCH ═════════════════════════

        # ══════════════════════════════════════════════════════════════
        # Track B: Extract benign-only training subset for novelty detection
        # ══════════════════════════════════════════════════════════════
        benign_mask = y_train == 0
        X_train_benign = X_train_s[benign_mask]

        self._report["track_b"] = {
            "enabled": cfg.track_b_enabled,
            "benign_train_samples": int(benign_mask.sum()),
            "attack_train_samples": int((~benign_mask).sum()),
        }
        logger.info(
            "Track B — Benign-only train: %d samples (%.1f%% of train)",
            benign_mask.sum(), benign_mask.mean() * 100,
        )

        # ══════════════════════════════════════════════════════════════
        # Track A: SMOTE config (applied inside CV pipeline, not here)
        # ══════════════════════════════════════════════════════════════
        self._report["track_a"] = {
            "smote_enabled": cfg.smote_enabled,
            "smote_strategy": cfg.smote_strategy,
            "smote_k_neighbors": cfg.smote_k_neighbors,
            "note": "SMOTE applied inside CV pipeline, not during preprocessing",
        }

        # ══════════════════════════════════════════════════════════════
        # Export
        # ══════════════════════════════════════════════════════════════
        elapsed = time.perf_counter() - t0
        self._report["elapsed_seconds"] = round(elapsed, 2)
        self._report["random_state"] = cfg.random_state

        output_dir = self._root / cfg.output_dir
        scaler_dir = self._root / "models" / "scalers"
        exporter = PreprocessingExporter(
            output_dir, scaler_dir, cfg.label_column, cfg.multi_label_column,
        )

        # Train set (scaled, pre-SMOTE — SMOTE applied inside CV)
        exporter.export_parquet(
            X_train_s, y_train, feat_names, cfg.train_parquet,
            y_multi=y_multi_train,
        )
        # Test set (scaled, untouched)
        exporter.export_parquet(
            X_test_s, y_test, feat_names, cfg.test_parquet,
            y_multi=y_multi_test,
        )
        # Track B: benign-only train
        if cfg.track_b_enabled:
            exporter.export_parquet(
                X_train_benign,
                np.zeros(len(X_train_benign), dtype=int),
                feat_names,
                cfg.train_benign_parquet,
            )

        exporter.export_scaler(scaler, cfg.scaler_file)

        # Selected features list
        sel_path = output_dir / "selected_features.json"
        sel_path.write_text(json.dumps(feat_names, indent=2), encoding="utf-8")

        self._report["output"] = {
            "feature_names": feat_names,
            "n_features": len(feat_names),
        }
        exporter.export_report(self._report, cfg.report_file)

        # Thesis report
        md = render_preprocessing_report(self._report)
        md_path = self._root / "results" / "phase0_analysis" / "report_section_preprocessing.md"
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(md, encoding="utf-8")
        logger.info("Thesis report → %s", md_path)

        self._log_summary()
        return self._report

    def get_report(self) -> Dict[str, Any]:
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
        idr = self._report.get("identifier_removal", {})
        cl = self._report.get("cleaning", {})
        var = self._report.get("variance", {})
        red = self._report.get("redundancy", {})
        spl = self._report.get("split", {})
        shp = self._report.get("shap_selection", {})
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
        logger.info("  RFECV+SHAP   : %d → %d features",
                     shp.get("n_selected", 0) + shp.get("n_dropped", 0),
                     shp.get("n_selected", 0))
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
