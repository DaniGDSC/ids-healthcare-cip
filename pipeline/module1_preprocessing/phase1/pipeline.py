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

import io
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from pipeline.module0_analysis.phase0.security import (
    IntegrityError,
    IntegrityVerifier,
    PathValidator,
)

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

# Hard cap on a single CSV file before pandas attempts to parse it.
# 200 MB comfortably accommodates the WUSTL-EHMS-2020 dataset (~25 MB)
# and pads for future captures, while preventing a hostile or accidental
# multi-gigabyte file from OOMing the host. See finding #11 in the
# Phase 1 security review.
_MAX_INPUT_BYTES: int = 200 * 1024 * 1024

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
        # Captured during _pre_split_transforms so _export can persist
        # the deterministic mappings as a JSON sidecar (finding #9).
        self._encoder: CategoricalEncoder | None = None

    def run(self) -> Dict[str, Any]:
        """Execute all pipeline steps and return the report dict."""
        t0 = time.perf_counter()
        cfg = self._config

        # ── Verify & ingest (single in-memory pass per file) ──
        # Each CSV in the input directory is hashed against the SIGNED
        # Phase 0 baseline AND parsed from the same in-memory bytes.
        # The previous flow hashed only the first CSV via a fail-open
        # reader and re-opened the file for pd.read_csv, leaving a
        # TOCTOU window AND a multi-file bypass — both closed below.
        df = self._ingest_with_integrity(cfg)

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

        # Step 2: Encode non-numeric features. The encoder builds
        # deterministic alphabetical mappings (NOT order-dependent
        # LabelEncoder codes) and is captured here so the JSON sidecar
        # can be persisted in _export — that sidecar is the only thing
        # that lets downstream inference reproduce the same integer
        # codes for unseen samples without re-fitting.
        encoder = CategoricalEncoder(
            label_encode=cfg.label_encode_columns,
            parse_numeric=cfg.parse_numeric_columns,
            sentinel=cfg.parse_numeric_sentinel,
        )
        df = encoder.transform(df)
        self._encoder = encoder
        self._report["encoding"] = encoder.get_report()

        # Step 3: Data cleaning. The handler refuses ffill without a
        # session_column (closes the cross-patient leakage hole) and
        # warns on fill_zero (closes the attacker-induced capture-loss
        # masking hole). See missing.py for the threat model.
        handler = MissingValueHandler(
            biometric_columns=cfg.biometric_columns,
            label_column=cfg.label_column,
            biometric_strategy=cfg.biometric_strategy,
            network_strategy=cfg.network_strategy,
            session_column=cfg.session_column,
        )
        df = handler.transform(df)
        self._report["cleaning"] = handler.get_report()

        # Step 4a: Variance filtering
        if cfg.variance_enabled:
            var_filter = VarianceFilter(max_unique=cfg.variance_max_unique)
            df = var_filter.transform(df)
            self._report["variance"] = var_filter.get_report()

        # Step 4b: Correlation-based redundancy. The remover refuses
        # to drop the binary or multi-class label even if the Phase 0
        # correlations CSV lists them as feature_b — closes the
        # tampered-corr-file attack documented in finding #14.
        if cfg.correlation_enabled:
            corr_df = self._reader.read_correlations()
            remover = RedundancyRemover(
                corr_df,
                cfg.correlation_threshold,
                protected_columns=(cfg.label_column, cfg.multi_label_column),
            )
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

        # Persist the deterministic categorical-encoder mappings as a
        # JSON sidecar so downstream inference can reproduce the same
        # integer codes for unseen samples without re-fitting (which
        # would otherwise drift silently from the training codes).
        if self._encoder is not None:
            encoder_path = output_dir / cfg.encoder_file
            self._encoder.save(encoder_path)
            self._report["encoder_sidecar"] = str(encoder_path.name)

        self._report["output"] = {
            "feature_names": feat_names,
            "n_features": len(feat_names),
        }
        exporter.export_report(self._report, cfg.report_file)

        md = render_preprocessing_report(self._report)
        # Phase 1 output goes under results/phase1_preprocessing/, NOT
        # results/phase0_analysis/. The previous path crossed module
        # boundaries and broke the Phase 0 biometric-leak regression
        # guard, which scans phase0_analysis/ for biometric column
        # names — a Phase 1 file with biometric column names in a
        # table header would have tripped it. See finding #20.
        md_path = (
            self._root / "results" / "phase1_preprocessing"
            / "report_section_preprocessing.md"
        )
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(md, encoding="utf-8")
        logger.info("Thesis report → %s", md_path)

    # ------------------------------------------------------------------
    # Ingestion & integrity (single in-memory pass per file)
    # ------------------------------------------------------------------

    def _ingest_with_integrity(self, cfg: Phase1Config) -> pd.DataFrame:
        """Verify and parse every input CSV in one in-memory pass.

        Discipline enforced here:
          1. ``input_dir`` is resolved through ``PathValidator`` so any
             configured directory that escapes the workspace is rejected
             before we touch the filesystem.
          2. ``file_pattern`` is restricted to a basename glob (no path
             separators) so a hostile YAML can't traverse out via
             ``../**``.
          3. Each matching file's size is checked against
             ``_MAX_INPUT_BYTES`` before any read.
          4. **Every** matching CSV is verified against the SIGNED
             Phase 0 baseline via ``IntegrityVerifier.verify_and_read``,
             not just the first one. This closes the multi-file bypass
             where an attacker dropped an extra CSV that sorted after
             the legitimate file (the old code only hashed
             ``csv_files[0]`` but ``concat``'d them all).
          5. The verified bytes are parsed with
             ``pd.read_csv(io.BytesIO(data))`` so the parser sees the
             exact buffer that was hashed — no second open, no TOCTOU.
          6. ``self._report["integrity"]`` is populated only after a
             successful verification AND only with the per-file digests
             that actually validated. The previous code hard-coded
             ``verified: True`` regardless.
        """
        validator = PathValidator(self._root)
        data_dir = validator.validate_input_path(cfg.input_dir)

        # Refuse globs that traverse out of the input directory.
        if "/" in cfg.file_pattern or "\\" in cfg.file_pattern:
            raise ValueError(
                f"Phase 1 file_pattern must be a basename glob, "
                f"got {cfg.file_pattern!r}"
            )

        csv_files = sorted(data_dir.glob(cfg.file_pattern))
        if not csv_files:
            raise FileNotFoundError(
                f"No files matching '{cfg.file_pattern}' in {data_dir}."
            )

        # Stand up the hardened verifier against the same metadata
        # directory Phase 0 uses. The verifier itself refuses to run
        # without an existing signed baseline (no auto-bootstrap), so
        # we cannot reach the parse step on a missing or forged file.
        verifier = IntegrityVerifier(
            metadata_dir=self._root / "pipeline/module0_analysis/phase0",
        )

        frames: List[pd.DataFrame] = []
        per_file_integrity: List[Dict[str, Any]] = []
        for path in csv_files:
            size = path.stat().st_size
            if size > _MAX_INPUT_BYTES:
                raise ValueError(
                    f"Refusing to read {path.name}: {size} bytes exceeds "
                    f"the {_MAX_INPUT_BYTES}-byte safety cap. Pin "
                    f"input_dir / file_pattern to the expected file or "
                    f"raise _MAX_INPUT_BYTES if this is intentional."
                )

            # ONE read: verify against the signed baseline AND get the
            # bytes the parser will see. IntegrityError propagates and
            # aborts the pipeline.
            try:
                data, digest = verifier.verify_and_read(path)
            except IntegrityError:
                # Make sure the failure leaves a visible mark in the
                # report rather than a half-populated dict.
                self._report["integrity"] = {
                    "verified":   False,
                    "failure_at": path.name,
                }
                raise

            df = pd.read_csv(io.BytesIO(data), low_memory=False)
            logger.info("  Loaded %s: %d × %d (sha256=%s…)",
                        path.name, df.shape[0], df.shape[1], digest[:16])
            frames.append(df)
            per_file_integrity.append({
                "file":   path.name,
                "sha256": digest,
                "rows":   int(df.shape[0]),
            })

        combined = (
            pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
        )
        self._report["ingestion"] = {
            "files_loaded": len(csv_files),
            "raw_rows":     int(combined.shape[0]),
            "raw_columns":  int(combined.shape[1]),
        }
        # Only populated when every file actually verified — see (6).
        self._report["integrity"] = {
            "verified":         True,
            "n_files_verified": len(per_file_integrity),
            "files":            per_file_integrity,
        }
        logger.info(
            "Ingestion: %d rows × %d cols across %d verified file(s)",
            combined.shape[0], combined.shape[1], len(csv_files),
        )
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
    """Run the Phase 1 preprocessing pipeline.

    Looks up the YAML at the canonical in-package location first
    (``pipeline/module1_preprocessing/phase1_config.yaml``) and falls
    back to the legacy ``config/phase1_config.yaml`` only if the
    in-package file is missing. The previous default-path-only lookup
    crashed unless an operator first ``cp``-ed the YAML into
    ``config/`` — see finding #21.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    in_package = PROJECT_ROOT / "pipeline/module1_preprocessing/phase1_config.yaml"
    legacy = PROJECT_ROOT / "config" / "phase1_config.yaml"
    config_path = in_package if in_package.exists() else legacy
    config = Phase1Config.from_yaml(config_path)

    # The artifact reader no longer touches the integrity file —
    # IntegrityVerifier owns that responsibility (finding #1).
    reader = Phase0ArtifactReader(
        project_root=PROJECT_ROOT,
        stats_file=config.phase0_stats_file,
        corr_file=config.phase0_corr_file,
    )

    pipeline = PreprocessingPipeline(config, reader, PROJECT_ROOT)
    pipeline.run()


if __name__ == "__main__":
    main()
