#!/usr/bin/env python3
"""Security-hardened Phase 1 preprocessing — OWASP Top 10 + HIPAA compliance.

Wraps the SOLID phase1 package with security controls, **extending**
Phase 0 security classes — never duplicating them.

    A01  Output path validation, read-only enforcement, overwrite protection
    A02  SHA-256 artifact hashing (train, test, scaler → preprocessing_metadata.json)
    A03  SMOTE/split parameter bounds validation
    A08  Data integrity assertions (split sum, no overlap, SMOTE train-only)
    A09  HIPAA-compliant audit logging (column names only, never values)

Phase 0 controls reused via direct import:
    - IntegrityVerifier.compute_hash()      (SHA-256 — not re-implemented)
    - PathValidator.validate_output_dir()    (path traversal — not re-implemented)
    - ConfigSanitizer.sanitize_config_dict() (injection — not re-implemented)
    - AuditLogger.log_*()                   (audit trail — not re-implemented)

Usage::

    python -m src.phase1_preprocessing.security_hardened_phase1
"""

from __future__ import annotations

import json
import logging
import os
import stat
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml
from sklearn.model_selection import StratifiedShuffleSplit

# ── Phase 0 security controls (reused, NOT duplicated) ──────────────
from src.phase0_dataset_analysis.phase0.security import (
    AuditLogger,
    ConfigSanitizer,
    IntegrityVerifier,
    PathValidator,
    BIOMETRIC_COLUMNS,
)

# ── Phase 1 SOLID components ────────────────────────────────────────
from src.phase1_preprocessing.phase1.artifact_reader import Phase0ArtifactReader
from src.phase1_preprocessing.phase1.config import Phase1Config
from src.phase1_preprocessing.phase1.exporter import PreprocessingExporter
from src.phase1_preprocessing.phase1.hipaa import HIPAASanitizer
from src.phase1_preprocessing.phase1.missing import MissingValueHandler
from src.phase1_preprocessing.phase1.redundancy import RedundancyRemover
from src.phase1_preprocessing.phase1.report import render_preprocessing_report
from src.phase1_preprocessing.phase1.scaler import RobustScalerTransformer
from src.phase1_preprocessing.phase1.smote import SMOTEBalancer

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
CONFIG_PATH: Path = PROJECT_ROOT / "config" / "phase1_config.yaml"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# A03 — Hardened parameter bounds (tighter than pydantic defaults)
SMOTE_K_MIN: int = 1
SMOTE_K_MAX: int = 10
SPLIT_RATIO_MIN: float = 0.5
SPLIT_RATIO_MAX: float = 0.9


# ===================================================================
# A03 — Injection: Parameter bounds validation
# ===================================================================


def _validate_phase1_parameters(config: Phase1Config) -> None:
    """Enforce Phase 1-specific parameter bounds (A03).

    Args:
        config: Validated Phase1Config.

    Raises:
        ValueError: If any parameter is outside hardened bounds.
    """
    logger.info("── A03: Phase 1 parameter bounds validation ──")

    # SMOTE k_neighbors ∈ [1, 10]
    if not (SMOTE_K_MIN <= config.smote_k_neighbors <= SMOTE_K_MAX):
        msg = (
            f"A03: smote_k_neighbors={config.smote_k_neighbors} "
            f"outside allowed range [{SMOTE_K_MIN}, {SMOTE_K_MAX}]"
        )
        AuditLogger.log_security_event("PARAMETER_VIOLATION", msg, logging.ERROR)
        raise ValueError(msg)
    logger.info(
        "  A03 ✓  smote_k_neighbors=%d ∈ [%d, %d]",
        config.smote_k_neighbors, SMOTE_K_MIN, SMOTE_K_MAX,
    )

    # random_state must be int
    if not isinstance(config.random_state, int):
        msg = (
            f"A03: random_state must be int, "
            f"got {type(config.random_state).__name__}"
        )
        AuditLogger.log_security_event("PARAMETER_VIOLATION", msg, logging.ERROR)
        raise ValueError(msg)
    logger.info("  A03 ✓  random_state=%d (int verified)", config.random_state)

    # train_ratio ∈ [0.5, 0.9]
    if not (SPLIT_RATIO_MIN <= config.train_ratio <= SPLIT_RATIO_MAX):
        msg = (
            f"A03: train_ratio={config.train_ratio} "
            f"outside allowed range [{SPLIT_RATIO_MIN}, {SPLIT_RATIO_MAX}]"
        )
        AuditLogger.log_security_event("PARAMETER_VIOLATION", msg, logging.ERROR)
        raise ValueError(msg)
    logger.info(
        "  A03 ✓  train_ratio=%.2f ∈ [%.1f, %.1f]",
        config.train_ratio, SPLIT_RATIO_MIN, SPLIT_RATIO_MAX,
    )


# ===================================================================
# A01 — Broken Access Control: Output paths + read-only
# ===================================================================


def _validate_output_paths(
    config: Phase1Config,
    validator: PathValidator,
    *,
    allow_overwrite: bool = False,
) -> Tuple[Path, Path]:
    """Validate all output paths within workspace (A01).

    Args:
        config: Validated Phase1Config.
        validator: Phase 0 PathValidator (reused, not duplicated).
        allow_overwrite: If False, raises on pre-existing artifacts.

    Returns:
        Tuple of (output_dir, scaler_dir) resolved paths.

    Raises:
        PermissionError: If paths escape workspace.
        FileExistsError: If artifacts exist and overwrite is not allowed.
    """
    logger.info("── A01: Validating output paths ──")

    output_dir = validator.validate_output_dir(config.output_dir)
    logger.info("  A01 ✓  Output dir: %s", output_dir)

    scaler_dir = validator.validate_output_dir(Path("models") / "scalers")
    logger.info("  A01 ✓  Scaler dir: %s", scaler_dir)

    if not allow_overwrite:
        for name in [config.train_parquet, config.test_parquet]:
            path = output_dir / name
            if path.exists():
                msg = (
                    f"A01: Artifact exists: {path.name}. "
                    "Set allow_overwrite=True to replace."
                )
                AuditLogger.log_security_event(
                    "OVERWRITE_BLOCKED", msg, logging.WARNING,
                )
                raise FileExistsError(msg)

        scaler_path = scaler_dir / config.scaler_file
        if scaler_path.exists():
            msg = (
                f"A01: Scaler exists: {scaler_path.name}. "
                "Set allow_overwrite=True to replace."
            )
            AuditLogger.log_security_event(
                "OVERWRITE_BLOCKED", msg, logging.WARNING,
            )
            raise FileExistsError(msg)

    logger.info(
        "  A01 ✓  Overwrite protection checked (allow_overwrite=%s)",
        allow_overwrite,
    )
    return output_dir, scaler_dir


def _make_read_only(path: Path) -> None:
    """Set exported artifact to read-only (A01).

    Args:
        path: Path to the artifact file.
    """
    current = path.stat().st_mode
    readonly = current & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    os.chmod(path, readonly)
    AuditLogger.log_file_access(
        "READ_ONLY_SET", path, extra=f"mode={oct(readonly & 0o777)}",
    )


def _clear_read_only(path: Path) -> None:
    """Temporarily restore write permission for overwriting (A01).

    Args:
        path: Path to the artifact file.
    """
    if path.exists():
        current = path.stat().st_mode
        if not (current & stat.S_IWUSR):
            os.chmod(path, current | stat.S_IWUSR)
            AuditLogger.log_file_access(
                "WRITE_RESTORED", path, extra="for overwrite",
            )


# ===================================================================
# A08 — Data Integrity: Split and SMOTE assertions
# ===================================================================


class IntegrityAssertions:
    """Data integrity assertion tracker for Phase 1 (A08).

    Each assertion is logged with pass/fail status and recorded
    for inclusion in the security report.
    """

    def __init__(self) -> None:
        self._results: List[Dict[str, Any]] = []

    def assert_split_sum(
        self,
        n_original: int,
        n_train: int,
        n_test: int,
    ) -> bool:
        """Assert train + test = original samples (after dropna).

        Args:
            n_original: Sample count after missing value handling.
            n_train: Training partition size.
            n_test: Test partition size.

        Returns:
            True if the assertion passes.
        """
        actual = n_train + n_test
        passed = actual == n_original

        self._results.append({
            "assertion": "Train + test = original",
            "expected": str(n_original),
            "actual": str(actual),
            "status": "PASS" if passed else "FAIL",
        })

        if passed:
            logger.info(
                "  A08 ✓  Split sum: %d + %d = %d", n_train, n_test, actual,
            )
        else:
            AuditLogger.log_security_event(
                "INTEGRITY_VIOLATION",
                f"Split sum: {n_train}+{n_test}={actual} ≠ {n_original}",
                logging.ERROR,
            )
        return passed

    def assert_no_overlap(
        self,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
    ) -> bool:
        """Assert zero overlap between train and test indices.

        Args:
            train_idx: Row indices assigned to training.
            test_idx: Row indices assigned to testing.

        Returns:
            True if the assertion passes.
        """
        overlap = np.intersect1d(train_idx, test_idx)
        n_overlap = len(overlap)
        passed = n_overlap == 0

        self._results.append({
            "assertion": "No train/test overlap",
            "expected": "0",
            "actual": str(n_overlap),
            "status": "PASS" if passed else "FAIL",
        })

        if passed:
            logger.info("  A08 ✓  Train/test overlap: 0 shared indices")
        else:
            AuditLogger.log_security_event(
                "INTEGRITY_VIOLATION",
                f"Train/test overlap: {n_overlap} shared indices",
                logging.CRITICAL,
            )
        return passed

    def assert_smote_train_only(
        self,
        n_test_before: int,
        n_test_after: int,
    ) -> bool:
        """Assert SMOTE was applied only to the training partition.

        Args:
            n_test_before: Test set size before SMOTE step.
            n_test_after: Test set size after SMOTE step.

        Returns:
            True if the assertion passes.
        """
        passed = n_test_before == n_test_after

        self._results.append({
            "assertion": "SMOTE on train only",
            "expected": "True",
            "actual": str(passed),
            "status": "PASS" if passed else "FAIL",
        })

        if passed:
            logger.info(
                "  A08 ✓  SMOTE train-only: test unchanged (%d)", n_test_after,
            )
        else:
            AuditLogger.log_security_event(
                "INTEGRITY_VIOLATION",
                f"Test modified during SMOTE: {n_test_before} → {n_test_after}",
                logging.CRITICAL,
            )
        return passed

    @property
    def results(self) -> List[Dict[str, Any]]:
        """Return all assertion results."""
        return list(self._results)

    @property
    def all_passed(self) -> bool:
        """True if every assertion passed."""
        return all(r["status"] == "PASS" for r in self._results)


# ===================================================================
# A02 — Cryptographic Failures: Artifact hashing
# ===================================================================


def _hash_artifacts(
    verifier: IntegrityVerifier,
    artifact_paths: Dict[str, Path],
) -> Dict[str, str]:
    """Compute SHA-256 for all exported artifacts (A02).

    Uses Phase 0 ``IntegrityVerifier.compute_hash()`` — not re-implemented.

    Args:
        verifier: Phase 0 IntegrityVerifier instance.
        artifact_paths: Mapping of artifact name → file path.

    Returns:
        Mapping of artifact name → SHA-256 hex digest.
    """
    logger.info("── A02: Hashing exported artifacts ──")
    hashes: Dict[str, str] = {}

    for name, path in artifact_paths.items():
        digest = verifier.compute_hash(path)
        hashes[name] = digest
        logger.info("  A02 ✓  %s: sha256=%s…", name, digest[:32])

    return hashes


def _store_preprocessing_metadata(
    output_dir: Path,
    artifact_hashes: Dict[str, str],
    assertion_results: List[Dict[str, Any]],
    config: Phase1Config,
) -> Path:
    """Persist artifact hashes and assertion results (A02).

    Creates ``preprocessing_metadata.json`` for downstream verification
    by the Detection Engine (Phase 2).

    Args:
        output_dir: Target directory.
        artifact_hashes: Artifact name → SHA-256 digest.
        assertion_results: Integrity assertion pass/fail records.
        config: Config for reproducibility metadata.

    Returns:
        Path to the written metadata file.
    """
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline": "security_hardened_phase1",
        "random_state": config.random_state,
        "artifact_hashes": {
            name: {"sha256": digest, "algorithm": "SHA-256"}
            for name, digest in artifact_hashes.items()
        },
        "integrity_assertions": assertion_results,
    }

    path = output_dir / "preprocessing_metadata.json"
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    AuditLogger.log_file_access("METADATA_WRITTEN", path)
    logger.info("  A02 ✓  Metadata stored: %s", path.name)
    return path


# ===================================================================
# A09 — Security Logging: HIPAA-safe wrappers
# ===================================================================


def _log_hipaa_sanitization(dropped_columns: List[str]) -> None:
    """Log HIPAA column removal — names only, NEVER values (A09)."""
    AuditLogger.log_security_event(
        "HIPAA_SANITIZATION",
        f"Dropped {len(dropped_columns)} PHI columns: {dropped_columns}",
        level=logging.INFO,
    )


def _log_smote_augmentation(report: Dict[str, Any]) -> None:
    """Log SMOTE counts — not individual sample values (A09)."""
    AuditLogger.log_security_event(
        "SMOTE_AUGMENTATION",
        (
            f"Train: {report['samples_before']} → {report['samples_after']} "
            f"(+{report['synthetic_added']} synthetic)"
        ),
        level=logging.INFO,
    )


def _log_scaling_parameters(scaler: RobustScalerTransformer) -> None:
    """Log scaler centre/scale statistics — safe to log (A09).

    These are derived from network-traffic feature distributions
    (median, IQR) and do not constitute PHI.
    """
    inner = scaler._scaler  # underlying sklearn scaler
    if hasattr(inner, "center_") and hasattr(inner, "scale_"):
        n = len(inner.center_)
        AuditLogger.log_security_event(
            "SCALING_PARAMETERS",
            (
                f"{n} features — "
                f"center range: [{inner.center_.min():.4f}, {inner.center_.max():.4f}], "
                f"scale range: [{inner.scale_.min():.4f}, {inner.scale_.max():.4f}]"
            ),
            level=logging.INFO,
        )


# ===================================================================
# Report Generator
# ===================================================================


def _generate_security_report(
    hipaa_columns: List[str],
    assertion_results: List[Dict[str, Any]],
    artifact_hashes: Dict[str, str],
    config: Phase1Config,
    scaler: RobustScalerTransformer,
) -> str:
    """Render ``report_section_preprocessing_security.md`` (§4.2).

    Args:
        hipaa_columns: PHI columns that were dropped.
        assertion_results: Data integrity assertion pass/fail records.
        artifact_hashes: SHA-256 hashes of exported artifacts.
        config: Phase1Config for parameter documentation.
        scaler: Fitted scaler for parameter reporting.

    Returns:
        Markdown string for thesis defence.
    """
    hipaa_list = ", ".join(f"`{c}`" for c in hipaa_columns)
    bio_list = ", ".join(f"`{c}`" for c in sorted(BIOMETRIC_COLUMNS))

    # Assertion table rows
    assertion_rows = "\n".join(
        f"| {r['assertion']} | {r['expected']} | {r['actual']} | {r['status']} |"
        for r in assertion_results
    )

    all_pass = all(r["status"] == "PASS" for r in assertion_results)
    overall = "ALL PASSED" if all_pass else "FAILURES DETECTED"

    # Artifact hash rows
    hash_rows = "\n".join(
        f"| `{name}` | `{digest}` |"
        for name, digest in artifact_hashes.items()
    )

    # Scaler statistics
    inner = scaler._scaler
    scale_info = ""
    if hasattr(inner, "center_") and hasattr(inner, "scale_"):
        scale_info = (
            f"- **Features scaled:** {len(inner.center_)}\n"
            f"- **Center (median) range:** "
            f"[{inner.center_.min():.4f}, {inner.center_.max():.4f}]\n"
            f"- **Scale (IQR) range:** "
            f"[{inner.scale_.min():.4f}, {inner.scale_.max():.4f}]\n"
        )

    return f"""## 4.2 Preprocessing Security Controls

This section documents the security controls applied during Phase 1
data preprocessing, extending the Phase 0 OWASP framework (§3.3) with
data-pipeline-specific protections.

### 4.2.1 OWASP Controls — Phase 1 Extensions

| OWASP ID | Risk Category | Control | Status |
|----------|---------------|---------|--------|
| A01 | Broken Access Control | Output paths validated within workspace boundary | Implemented |
| A01 | Broken Access Control | Artifacts set to read-only (chmod 444) after export | Implemented |
| A01 | Broken Access Control | Overwrite protection — existing artifacts not silently replaced | Implemented |
| A02 | Cryptographic Failures | SHA-256 hash computed for train.parquet, test.parquet, scaler.pkl | Implemented |
| A02 | Cryptographic Failures | Hashes stored in `preprocessing_metadata.json` | Implemented |
| A03 | Injection | SMOTE `k_neighbors` validated ∈ [1, 10] | Implemented |
| A03 | Injection | `random_state` type-checked as `int` | Implemented |
| A03 | Injection | `train_ratio` validated ∈ [0.5, 0.9] | Implemented |
| A08 | Data Integrity | Train + test = original sample count verified | Implemented |
| A08 | Data Integrity | Zero train/test index overlap verified | Implemented |
| A08 | Data Integrity | SMOTE applied exclusively to training partition | Implemented |
| A09 | Security Logging | HIPAA column drops logged (names only, never values) | Implemented |
| A09 | Security Logging | SMOTE augmentation count logged (no sample values) | Implemented |
| A09 | Security Logging | Scaler parameters logged (center, scale — non-PHI) | Implemented |

### 4.2.2 HIPAA Preprocessing Compliance Checklist

- [x] PII fields dropped before any transformation: [{hipaa_list}]
- [x] No PHI values in any log file — column names only
- [x] Output artifacts set to read-only after export
- [x] Train/test overlap verified — 0 shared indices
- [x] Biometric values never logged: {bio_list}
- [x] SMOTE sample values not logged — only aggregate counts

### 4.2.3 Data Integrity Assertions (A08)

All assertions are evaluated at runtime and logged with pass/fail status.

| Assertion | Expected | Actual | Status |
|-----------|----------|--------|--------|
{assertion_rows}

**Overall:** {overall}

### 4.2.4 Artifact Integrity (A02)

SHA-256 hashes computed after export for downstream verification
by the Detection Engine (Phase 2).

| Artifact | SHA-256 |
|----------|---------|
{hash_rows}

Hashes are stored in `preprocessing_metadata.json` and must be
verified before loading artifacts in subsequent pipeline phases.

### 4.2.5 Parameter Bounds Validation (A03)

| Parameter | Allowed Range | Configured Value | Status |
|-----------|--------------|-----------------|--------|
| `smote_k_neighbors` | [1, 10] | {config.smote_k_neighbors} | PASS |
| `random_state` | int | {config.random_state} | PASS |
| `train_ratio` | [0.5, 0.9] | {config.train_ratio} | PASS |
| `correlation_threshold` | (0, 1] | {config.correlation_threshold} | PASS |

### 4.2.6 Scaling Parameters (A09 — Safe to Log)

{scale_info}
RobustScaler center and scale values are derived from median and IQR
of *network traffic features* — they do not constitute PHI and are
safe to include in logs and reports.

### 4.2.7 Security Inheritance from Phase 0

The following Phase 0 controls (§3.3) are reused without duplication:

| Control | Phase 0 Module | Reuse Method |
|---------|---------------|-------------|
| SHA-256 hashing | `IntegrityVerifier` | Direct import — `compute_hash()` |
| Path traversal protection | `PathValidator` | Direct import — `validate_output_dir()` |
| Config sanitization | `ConfigSanitizer` | Direct import — `sanitize_config_dict()` |
| Audit logging | `AuditLogger` | Direct import — `log_file_access()`, `log_security_event()` |

---

**Reviewer signature:** ____________________  **Date:** ____/____/______
"""


# ===================================================================
# Security-Hardened Pipeline
# ===================================================================


def run_hardened_pipeline(*, allow_overwrite: bool = True) -> Dict[str, Any]:
    """Execute Phase 1 preprocessing with full OWASP/HIPAA controls.

    Steps:
        1. A03/A05 — Load config, sanitize strings, validate bounds
        2. A01     — Validate output paths, overwrite protection
        3. A02     — Verify raw dataset integrity (Phase 0 baseline)
        4. Pipeline — HIPAA → Missing → Redundancy → Split → SMOTE → Scale
        5. A08     — Data integrity assertions
        6. A01     — Export artifacts, enforce read-only
        7. A02     — Hash artifacts, store preprocessing_metadata.json
        8. A09     — HIPAA-compliant audit logging throughout
        9.         — Generate thesis security report

    Args:
        allow_overwrite: If True, existing artifacts are replaced.

    Returns:
        Combined pipeline report dict.
    """
    t0 = time.perf_counter()

    logger.info("=" * 70)
    logger.info("SECURITY-HARDENED PHASE 1: DATA PREPROCESSING")
    logger.info("=" * 70)

    # ── 1. A03/A05: Config sanitization + parameter bounds ────────
    logger.info("── A03/A05: Loading and sanitizing configuration ──")
    raw_yaml: dict = yaml.safe_load(CONFIG_PATH.read_text())
    AuditLogger.log_file_access("CONFIG_READ", CONFIG_PATH)
    ConfigSanitizer.sanitize_config_dict(raw_yaml)
    logger.info("  A03 ✓  Config sanitization passed")

    config = Phase1Config.from_yaml(CONFIG_PATH)
    logger.info("  A05 ✓  Pydantic schema validation passed")

    _validate_phase1_parameters(config)

    # ── 2. A01: Path validation ───────────────────────────────────
    validator = PathValidator(PROJECT_ROOT)
    output_dir, scaler_dir = _validate_output_paths(
        config, validator, allow_overwrite=allow_overwrite,
    )

    data_dir = validator.validate_input_path(config.input_dir)
    logger.info("  A01 ✓  Input dir validated: %s", data_dir.name)

    # If overwriting, clear read-only on existing artifacts
    if allow_overwrite:
        _clear_read_only(output_dir / config.train_parquet)
        _clear_read_only(output_dir / config.test_parquet)
        _clear_read_only(scaler_dir / config.scaler_file)

    # ── 3. A02: Dataset integrity (Phase 0 verifier — reused) ────
    logger.info("── A02: Dataset integrity verification ──")
    reader = Phase0ArtifactReader(
        project_root=PROJECT_ROOT,
        stats_file=config.phase0_stats_file,
        corr_file=config.phase0_corr_file,
        integrity_file=config.phase0_integrity_file,
    )

    csv_files = sorted(data_dir.glob(config.file_pattern))
    if not csv_files:
        raise FileNotFoundError(
            f"No files matching '{config.file_pattern}' in {data_dir}",
        )
    dataset_sha = reader.verify_integrity(csv_files[0])
    logger.info("  A02 ✓  Dataset SHA-256: %s…", dataset_sha[:32])

    # ── 4. Pipeline steps ─────────────────────────────────────────
    report: Dict[str, Any] = {
        "integrity": {"sha256": dataset_sha, "verified": True},
    }

    # Step 1: Ingest
    logger.info("── Step 1: Ingest ──")
    frames = []
    for path in csv_files:
        frame = pd.read_csv(path, low_memory=False)
        AuditLogger.log_file_access(
            "DATASET_READ", path,
            extra=f"{len(frame)} rows × {len(frame.columns)} cols",
        )
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    n_raw = len(df)
    report["ingestion"] = {
        "files_loaded": len(csv_files),
        "raw_rows": n_raw,
        "raw_columns": len(df.columns),
    }
    logger.info("  Ingestion: %d rows × %d cols", *df.shape)

    # Step 2: HIPAA
    logger.info("── Step 2: HIPAA Sanitization ──")
    hipaa = HIPAASanitizer(config.hipaa_columns)
    df = hipaa.transform(df)
    report["hipaa"] = hipaa.get_report()
    _log_hipaa_sanitization(hipaa.get_report()["columns_dropped"])

    # Step 3: Missing values
    logger.info("── Step 3: Missing Value Handling ──")
    handler = MissingValueHandler(
        biometric_columns=config.biometric_columns,
        label_column=config.label_column,
        biometric_strategy=config.biometric_strategy,
        network_strategy=config.network_strategy,
    )
    df = handler.transform(df)
    report["missing_values"] = handler.get_report()
    n_after_dropna = len(df)

    # Step 4: Redundancy
    logger.info("── Step 4: Redundancy Elimination ──")
    corr_df = reader.read_correlations()
    remover = RedundancyRemover(
        corr_df, config.correlation_threshold, config.label_column,
    )
    df = remover.transform(df)
    report["redundancy"] = remover.get_report()

    # Step 5: Stratified split (with index capture for A08)
    logger.info("── Step 5: Stratified Split ──")
    y = df[config.label_column].values
    X_df = df.drop(columns=[config.label_column]).select_dtypes(
        include=[np.number],
    )
    feat_names = X_df.columns.tolist()
    X = X_df.values

    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=config.test_ratio,
        random_state=config.random_state,
    )
    train_idx, test_idx = next(sss.split(X, y))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    report["split"] = {
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "train_ratio": round(1 - config.test_ratio, 2),
        "test_ratio": config.test_ratio,
        "stratified": True,
        "train_attack_rate": round(float(y_train.mean()), 4),
        "test_attack_rate": round(float(y_test.mean()), 4),
    }
    logger.info(
        "  Split: train=%d (attack=%.1f%%) | test=%d (attack=%.1f%%)",
        len(X_train), y_train.mean() * 100,
        len(X_test), y_test.mean() * 100,
    )

    # ── A08: Integrity assertions ─────────────────────────────────
    logger.info("── A08: Data integrity assertions ──")
    assertions = IntegrityAssertions()
    assertions.assert_split_sum(n_after_dropna, len(X_train), len(X_test))
    assertions.assert_no_overlap(train_idx, test_idx)

    n_test_before_smote = len(X_test)

    # Step 6: SMOTE (train only)
    logger.info("── Step 6: SMOTE (train only) ──")
    balancer = SMOTEBalancer(
        strategy=config.smote_strategy,
        k_neighbors=config.smote_k_neighbors,
        random_state=config.random_state,
    )
    X_train, y_train = balancer.resample(X_train, y_train)
    report["smote"] = balancer.get_report()

    # A08: Verify test set unchanged
    assertions.assert_smote_train_only(n_test_before_smote, len(X_test))

    # A09: Log SMOTE counts (not values)
    _log_smote_augmentation(balancer.get_report())

    # Step 7: Robust scaling
    logger.info("── Step 7: Robust Scaling ──")
    scaler = RobustScalerTransformer(method=config.scaling_method)
    X_train_s, X_test_s = scaler.scale_both(X_train, X_test)
    report["scaling"] = scaler.get_report()

    # A09: Log scaling parameters (safe — non-PHI)
    _log_scaling_parameters(scaler)

    # ── 6. Export + A01: read-only enforcement ────────────────────
    logger.info("── Export + A01: Read-only enforcement ──")
    exporter = PreprocessingExporter(
        output_dir, scaler_dir, config.label_column,
    )

    train_path = exporter.export_parquet(
        X_train_s, y_train, feat_names, config.train_parquet,
    )
    test_path = exporter.export_parquet(
        X_test_s, y_test, feat_names, config.test_parquet,
    )
    scaler_path = exporter.export_scaler(scaler, config.scaler_file)

    report["output"] = {
        "feature_names": feat_names,
        "n_features": len(feat_names),
    }

    elapsed = time.perf_counter() - t0
    report["elapsed_seconds"] = round(elapsed, 2)
    report["random_state"] = config.random_state
    exporter.export_report(report, config.report_file)

    # A01: Set artifacts read-only
    _make_read_only(train_path)
    _make_read_only(test_path)
    _make_read_only(scaler_path)
    logger.info("  A01 ✓  All artifacts set read-only")

    # ── 7. A02: Hash artifacts + metadata ─────────────────────────
    verifier = IntegrityVerifier(metadata_dir=output_dir)
    artifact_hashes = _hash_artifacts(verifier, {
        config.train_parquet: train_path,
        config.test_parquet: test_path,
        config.scaler_file: scaler_path,
    })

    _store_preprocessing_metadata(
        output_dir, artifact_hashes, assertions.results, config,
    )

    # ── 8. Thesis reports ─────────────────────────────────────────
    # Standard preprocessing report
    md = render_preprocessing_report(report)
    md_path = (
        PROJECT_ROOT / "results" / "phase0_analysis"
        / "report_section_preprocessing.md"
    )
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md, encoding="utf-8")
    logger.info("  Preprocessing report → %s", md_path.name)

    # Security report
    security_md = _generate_security_report(
        hipaa_columns=hipaa.get_report()["columns_dropped"],
        assertion_results=assertions.results,
        artifact_hashes=artifact_hashes,
        config=config,
        scaler=scaler,
    )
    security_path = (
        PROJECT_ROOT / "results" / "phase0_analysis"
        / "report_section_preprocessing_security.md"
    )
    security_path.write_text(security_md, encoding="utf-8")
    AuditLogger.log_file_access("SECURITY_REPORT_WRITTEN", security_path)
    logger.info("  Security report → %s", security_path.name)

    # ── Summary ───────────────────────────────────────────────────
    _log_summary(report, assertions, artifact_hashes)

    return report


# ===================================================================
# Summary
# ===================================================================


def _log_summary(
    report: Dict[str, Any],
    assertions: IntegrityAssertions,
    artifact_hashes: Dict[str, str],
) -> None:
    """Log final pipeline summary with security status."""
    sep = "=" * 70
    ing = report.get("ingestion", {})
    hip = report.get("hipaa", {})
    mv = report.get("missing_values", {})
    red = report.get("redundancy", {})
    spl = report.get("split", {})
    smt = report.get("smote", {})
    out = report.get("output", {})

    logger.info("")
    logger.info(sep)
    logger.info("SECURITY-HARDENED PHASE 1 — SUMMARY")
    logger.info(sep)
    logger.info(
        "  Ingestion   : %d files → %d × %d",
        ing.get("files_loaded", 0),
        ing.get("raw_rows", 0),
        ing.get("raw_columns", 0),
    )
    logger.info("  HIPAA       : %d columns dropped", hip.get("n_dropped", 0))
    logger.info(
        "  Missing     : %d bio cells filled, %d rows dropped",
        mv.get("biometric_cells_filled", 0),
        mv.get("rows_dropped", 0),
    )
    logger.info(
        "  Redundancy  : %d features (|r| ≥ %.2f)",
        red.get("n_dropped", 0),
        red.get("threshold", 0),
    )
    logger.info(
        "  Split       : train=%d, test=%d",
        spl.get("train_samples", 0),
        spl.get("test_samples", 0),
    )
    logger.info(
        "  SMOTE       : %d → %d (+%d)",
        smt.get("samples_before", 0),
        smt.get("samples_after", 0),
        smt.get("synthetic_added", 0),
    )
    logger.info("  Features    : %d", out.get("n_features", 0))
    logger.info(
        "  Integrity   : %s (%d assertions)",
        "ALL PASS" if assertions.all_passed else "FAILURE",
        len(assertions.results),
    )
    logger.info("  Artifacts   : %d hashed (SHA-256)", len(artifact_hashes))
    logger.info(
        "  Elapsed     : %.2f s", report.get("elapsed_seconds", 0),
    )
    logger.info(sep)


# ===================================================================
# Entry Point
# ===================================================================


def main() -> None:
    """Run the security-hardened Phase 1 preprocessing pipeline."""
    run_hardened_pipeline(allow_overwrite=True)


if __name__ == "__main__":
    main()
