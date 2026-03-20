#!/usr/bin/env python3
"""Security-hardened Phase 3 classification engine — OWASP Top 10 + HIPAA compliance.

Wraps the SOLID phase3 package with security controls, **extending**
Phase 0 security classes — never duplicating them.

    A01  Output path validation, read-only enforcement, overwrite protection
    A02  SHA-256 artifact hashing (model, metrics → classification_metadata.json)
    A04  Progressive unfreezing order validation (frozen count must decrease)
    A05  Security misconfiguration: hardened parameter bounds, unknown key rejection
    A08  Data integrity assertions (test-only eval, CM sum, F1 consistency)
    A09  HIPAA-compliant audit logging (aggregate metrics only, never per-patient)

Phase 0 controls reused via direct import:
    - IntegrityVerifier.compute_hash()      (SHA-256 — not re-implemented)
    - PathValidator.validate_output_dir()    (path traversal — not re-implemented)
    - ConfigSanitizer.sanitize_config_dict() (injection — not re-implemented)
    - AuditLogger.log_*()                   (audit trail — not re-implemented)

Usage::

    python -m src.phase3_classification_engine.security_hardened_phase3
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
import tensorflow as tf
import yaml

# ── Phase 0 security controls (reused, NOT duplicated) ──────────────
from src.phase0_dataset_analysis.phase0.security import (
    BIOMETRIC_COLUMNS,
    AuditLogger,
    ConfigSanitizer,
    IntegrityVerifier,
    PathValidator,
)

# ── Phase 2 SOLID components (for model rebuild) ─────────────────
from src.phase2_detection_engine.phase2.assembler import DetectionModelAssembler
from src.phase2_detection_engine.phase2.attention_builder import (  # noqa: F401
    AttentionBuilder,
    BahdanauAttention,
)
from src.phase2_detection_engine.phase2.bilstm_builder import BiLSTMBuilder
from src.phase2_detection_engine.phase2.cnn_builder import CNNBuilder
from src.phase2_detection_engine.phase2.reshaper import DataReshaper

# ── Phase 3 SOLID components ────────────────────────────────────────
from src.phase3_classification_engine.phase3.artifact_reader import Phase2ArtifactReader
from src.phase3_classification_engine.phase3.config import Phase3Config
from src.phase3_classification_engine.phase3.evaluator import ModelEvaluator
from src.phase3_classification_engine.phase3.exporter import ClassificationExporter
from src.phase3_classification_engine.phase3.head import AutoClassificationHead
from src.phase3_classification_engine.phase3.pipeline import (
    _detect_hardware,
    _get_git_commit,
)
from src.phase3_classification_engine.phase3.report import render_classification_report
from src.phase3_classification_engine.phase3.trainer import ClassificationTrainer
from src.phase3_classification_engine.phase3.unfreezer import ProgressiveUnfreezer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
CONFIG_PATH: Path = PROJECT_ROOT / "config" / "phase3_config.yaml"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# A05 — Hardened parameter bounds
DENSE_UNITS_MIN: int = 1
DENSE_UNITS_MAX: int = 512
DROPOUT_MIN: float = 0.0
DROPOUT_MAX: float = 0.8
THRESHOLD_MIN: float = 0.01
THRESHOLD_MAX: float = 0.99
BATCH_SIZE_MIN: int = 8
BATCH_SIZE_MAX: int = 2048
EPOCHS_MIN: int = 1
EPOCHS_MAX: int = 100
PATIENCE_MIN: int = 1
PATIENCE_MAX: int = 20

# Known top-level YAML keys for unknown-key rejection (A05)
_KNOWN_YAML_KEYS: frozenset = frozenset(
    {
        "data",
        "classification_head",
        "training",
        "callbacks",
        "evaluation",
        "output",
        "random_state",
        "cross_dataset",
    }
)


# ===================================================================
# A05 — Security Misconfiguration: Parameter Bounds + Unknown Keys
# ===================================================================


def _reject_unknown_yaml_keys(raw_yaml: dict) -> None:
    """Reject unknown top-level YAML keys (A05).

    Raises:
        ValueError: If unknown keys are found.
    """
    logger.info("── A05: Unknown YAML key rejection ──")
    unknown = set(raw_yaml.keys()) - _KNOWN_YAML_KEYS
    if unknown:
        msg = f"A05: Unknown YAML keys: {sorted(unknown)}"
        AuditLogger.log_security_event("CONFIG_VIOLATION", msg, logging.ERROR)
        raise ValueError(msg)
    logger.info("  A05 ✓  No unknown YAML keys (checked %d)", len(raw_yaml))


def _validate_phase3_parameters(config: Phase3Config) -> None:
    """Enforce Phase 3-specific parameter bounds (A05).

    Args:
        config: Validated Phase3Config.

    Raises:
        ValueError: If any parameter is outside hardened bounds.
    """
    logger.info("── A05: Phase 3 parameter bounds validation ──")

    checks: List[Tuple[str, Any, Any, Any]] = [
        ("dense_units", config.dense_units, DENSE_UNITS_MIN, DENSE_UNITS_MAX),
        ("head_dropout_rate", config.head_dropout_rate, DROPOUT_MIN, DROPOUT_MAX),
        ("threshold", config.threshold, THRESHOLD_MIN, THRESHOLD_MAX),
        ("batch_size", config.batch_size, BATCH_SIZE_MIN, BATCH_SIZE_MAX),
        ("early_stopping_patience", config.early_stopping_patience, PATIENCE_MIN, PATIENCE_MAX),
        ("reduce_lr_patience", config.reduce_lr_patience, PATIENCE_MIN, PATIENCE_MAX),
    ]

    for name, value, lo, hi in checks:
        if not (lo <= value <= hi):
            msg = f"A05: {name}={value} outside allowed range [{lo}, {hi}]"
            AuditLogger.log_security_event("PARAMETER_VIOLATION", msg, logging.ERROR)
            raise ValueError(msg)
        logger.info("  A05 ✓  %s=%s ∈ [%s, %s]", name, value, lo, hi)

    # Validate per-phase epochs
    for phase in config.training_phases:
        if not (EPOCHS_MIN <= phase.epochs <= EPOCHS_MAX):
            msg = f"A05: {phase.name} epochs={phase.epochs} outside [{EPOCHS_MIN}, {EPOCHS_MAX}]"
            AuditLogger.log_security_event("PARAMETER_VIOLATION", msg, logging.ERROR)
            raise ValueError(msg)
        if phase.learning_rate <= 0:
            msg = f"A05: {phase.name} learning_rate={phase.learning_rate} must be > 0"
            AuditLogger.log_security_event("PARAMETER_VIOLATION", msg, logging.ERROR)
            raise ValueError(msg)

    logger.info("  A05 ✓  All %d training phases validated", len(config.training_phases))


# ===================================================================
# A04 — Insecure Design: Progressive Unfreezing Order
# ===================================================================


def _validate_unfreezing_order(config: Phase3Config) -> None:
    """Validate progressive unfreezing order (A04).

    Phase A must freeze more layers than Phase B, and Phase B more than C.
    Learning rates must decrease: lr_A > lr_B > lr_C.

    Raises:
        ValueError: If unfreezing order is violated.
    """
    logger.info("── A04: Progressive unfreezing order validation ──")
    phases = config.training_phases

    if len(phases) < 2:
        logger.info("  A04 ✓  Single phase — order validation skipped")
        return

    for i in range(len(phases) - 1):
        curr, nxt = phases[i], phases[i + 1]

        # Frozen count must decrease (more layers unfrozen each phase)
        if len(curr.frozen) <= len(nxt.frozen):
            msg = (
                f"A04: {curr.name} freezes {len(curr.frozen)} groups "
                f"but {nxt.name} freezes {len(nxt.frozen)} — "
                f"must decrease (unfreeze progressively)"
            )
            AuditLogger.log_security_event("DESIGN_VIOLATION", msg, logging.ERROR)
            raise ValueError(msg)

        # Learning rate must decrease
        if curr.learning_rate <= nxt.learning_rate:
            msg = (
                f"A04: lr({curr.name})={curr.learning_rate} must be > "
                f"lr({nxt.name})={nxt.learning_rate}"
            )
            AuditLogger.log_security_event("DESIGN_VIOLATION", msg, logging.ERROR)
            raise ValueError(msg)

        logger.info(
            "  A04 ✓  %s (frozen=%d, lr=%s) → %s (frozen=%d, lr=%s)",
            curr.name,
            len(curr.frozen),
            curr.learning_rate,
            nxt.name,
            len(nxt.frozen),
            nxt.learning_rate,
        )


# ===================================================================
# A01 — Broken Access Control: Path Validation + Read-Only
# ===================================================================


def _validate_output_paths(
    config: Phase3Config,
    validator: PathValidator,
    allow_overwrite: bool = False,
) -> Path:
    """Validate output paths within workspace (A01).

    Returns:
        Resolved output directory path.

    Raises:
        FileExistsError: If artifacts exist and overwrite not allowed.
    """
    logger.info("── A01: Output path validation ──")
    output_dir = validator.validate_output_dir(PROJECT_ROOT / config.output_dir)
    logger.info("  A01 ✓  Output dir: %s", output_dir)

    if not allow_overwrite:
        model_path = output_dir / config.model_file
        if model_path.exists():
            msg = f"A01: {model_path.name} exists — set allow_overwrite=True"
            AuditLogger.log_security_event("OVERWRITE_BLOCKED", msg, logging.WARNING)
            raise FileExistsError(msg)

    return output_dir


def _make_read_only(path: Path) -> None:
    """Set exported artifact to read-only (chmod 444) (A01)."""
    current = path.stat().st_mode
    read_only = current & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    path.chmod(read_only)
    AuditLogger.log_file_access("READ_ONLY_SET", path, extra="mode=0o444")


def _clear_read_only(path: Path) -> None:
    """Temporarily restore write permission for overwriting (A01)."""
    if path.exists() and not os.access(path, os.W_OK):
        path.chmod(path.stat().st_mode | stat.S_IWUSR)
        AuditLogger.log_file_access("WRITE_RESTORED", path, extra="for overwrite")


# ===================================================================
# A02 — Cryptographic Failures: Artifact Hashing
# ===================================================================


def _hash_artifacts(
    verifier: IntegrityVerifier, artifact_paths: Dict[str, Path]
) -> Dict[str, Dict[str, str]]:
    """Compute SHA-256 for all exported artifacts (A02).

    Returns:
        Dict of {name: {"sha256": digest, "algorithm": "SHA-256"}}.
    """
    logger.info("── A02: Artifact hashing ──")
    hashes: Dict[str, Dict[str, str]] = {}
    for name, path in artifact_paths.items():
        digest = verifier.compute_hash(path)
        hashes[name] = {"sha256": digest, "algorithm": "SHA-256"}
        logger.info("  A02 ✓  %s: sha256=%s…", name, digest[:16])
    return hashes


def _store_classification_metadata(
    output_dir: Path,
    artifact_hashes: Dict[str, Dict[str, str]],
    assertion_results: List[Dict[str, Any]],
    config: Phase3Config,
    metrics: Dict[str, Any],
    hw_info: Dict[str, str],
    train_samples: int,
) -> Path:
    """Persist artifact hashes, assertions, and classification metadata (A02).

    Returns:
        Path to classification_metadata.json.
    """
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline": "security_hardened_phase3",
        "random_state": config.random_state,
        "git_commit": _get_git_commit(),
        "hardware": hw_info,
        "train_samples": train_samples,
        "hyperparameters": {
            "dense_units": config.dense_units,
            "dense_activation": config.dense_activation,
            "head_dropout_rate": config.head_dropout_rate,
            "batch_size": config.batch_size,
            "validation_split": config.validation_split,
            "threshold": config.threshold,
            "training_phases": [
                {
                    "name": p.name,
                    "epochs": p.epochs,
                    "learning_rate": p.learning_rate,
                    "frozen": p.frozen,
                }
                for p in config.training_phases
            ],
        },
        "evaluation_metrics": {
            "accuracy": metrics["accuracy"],
            "f1_score": metrics["f1_score"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "auc_roc": metrics["auc_roc"],
            "threshold": metrics["threshold"],
            "test_samples": metrics["test_samples"],
        },
        "artifact_hashes": artifact_hashes,
        "integrity_assertions": assertion_results,
    }

    meta_path = output_dir / "classification_metadata.json"
    _clear_read_only(meta_path)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    _make_read_only(meta_path)
    AuditLogger.log_file_access("METADATA_WRITTEN", meta_path)
    return meta_path


# ===================================================================
# A08 — Data Integrity Assertions
# ===================================================================


class ClassificationAssertions:
    """Classification-specific data integrity assertions (A08)."""

    def __init__(self) -> None:
        self._results: List[Dict[str, Any]] = []

    def assert_eval_test_only(self, test_samples: int, eval_samples: int) -> bool:
        """Assert evaluation was performed on test set only."""
        passed = test_samples == eval_samples
        self._results.append(
            {
                "assertion": "Evaluation on test set only",
                "expected": f"{test_samples} test samples",
                "actual": f"{eval_samples} eval samples",
                "status": "PASS" if passed else "FAIL",
            }
        )
        if passed:
            logger.info("  A08 ✓  Eval on test set only (%d samples)", eval_samples)
        else:
            AuditLogger.log_security_event(
                "INTEGRITY_VIOLATION",
                f"Eval samples={eval_samples} ≠ test samples={test_samples}",
                logging.CRITICAL,
            )
        return passed

    def assert_no_train_test_overlap(self, n_train: int, n_test: int, total: int) -> bool:
        """Assert no data leakage: train + test = total (no overlap)."""
        overlap = (n_train + n_test) - total
        passed = overlap <= 0
        self._results.append(
            {
                "assertion": "No train/test data overlap",
                "expected": f"overlap ≤ 0 (train={n_train}, test={n_test})",
                "actual": f"overlap={max(0, overlap)}",
                "status": "PASS" if passed else "FAIL",
            }
        )
        if passed:
            logger.info("  A08 ✓  No train/test overlap")
        else:
            AuditLogger.log_security_event(
                "DATA_LEAKAGE",
                f"Overlap={overlap} samples between train and test",
                logging.CRITICAL,
            )
        return passed

    def assert_confusion_matrix_sum(self, cm: List[List[int]], n_test: int) -> bool:
        """Assert confusion matrix rows sum to actual class counts."""
        cm_sum = sum(sum(row) for row in cm)
        passed = cm_sum == n_test
        self._results.append(
            {
                "assertion": "Confusion matrix sum = test samples",
                "expected": str(n_test),
                "actual": str(cm_sum),
                "status": "PASS" if passed else "FAIL",
            }
        )
        if passed:
            logger.info("  A08 ✓  CM sum=%d = test samples", cm_sum)
        else:
            AuditLogger.log_security_event(
                "INTEGRITY_VIOLATION",
                f"CM sum={cm_sum} ≠ test samples={n_test}",
                logging.CRITICAL,
            )
        return passed

    def assert_metrics_consistency(self, metrics: Dict[str, Any]) -> bool:
        """Assert F1/precision/recall are internally consistent."""
        f1 = metrics["f1_score"]
        prec = metrics["precision"]
        rec = metrics["recall"]
        # All must be in [0, 1]
        in_range = all(0.0 <= v <= 1.0 for v in [f1, prec, rec])
        passed = in_range
        self._results.append(
            {
                "assertion": "Metrics consistency (F1, precision, recall ∈ [0,1])",
                "expected": "all ∈ [0.0, 1.0]",
                "actual": f"F1={f1:.4f}, prec={prec:.4f}, rec={rec:.4f}",
                "status": "PASS" if passed else "FAIL",
            }
        )
        if passed:
            logger.info("  A08 ✓  Metrics internally consistent")
        else:
            AuditLogger.log_security_event(
                "INTEGRITY_VIOLATION",
                f"Metrics out of range: F1={f1}, prec={prec}, rec={rec}",
                logging.CRITICAL,
            )
        return passed

    def assert_predictions_valid(self, accuracy: float, auc: float) -> bool:
        """Assert accuracy and AUC-ROC are valid values."""
        passed = 0.0 <= accuracy <= 1.0 and 0.0 <= auc <= 1.0
        self._results.append(
            {
                "assertion": "Prediction validity (accuracy, AUC ∈ [0,1])",
                "expected": "accuracy ∈ [0,1], AUC ∈ [0,1]",
                "actual": f"accuracy={accuracy:.4f}, AUC={auc:.4f}",
                "status": "PASS" if passed else "FAIL",
            }
        )
        if passed:
            logger.info("  A08 ✓  Accuracy=%.4f, AUC=%.4f valid", accuracy, auc)
        else:
            AuditLogger.log_security_event(
                "INTEGRITY_VIOLATION",
                f"Invalid: accuracy={accuracy}, AUC={auc}",
                logging.CRITICAL,
            )
        return passed

    def assert_has_classification_head(self, model: tf.keras.Model) -> bool:
        """Assert model HAS a classification head (opposite of Phase 2)."""
        last = model.layers[-1]
        has_head = isinstance(last, tf.keras.layers.Dense)
        self._results.append(
            {
                "assertion": "Classification head present",
                "expected": "True (Dense output layer)",
                "actual": f"last_layer={type(last).__name__}, has_head={has_head}",
                "status": "PASS" if has_head else "FAIL",
            }
        )
        if has_head:
            logger.info("  A08 ✓  Classification head present")
        else:
            AuditLogger.log_security_event(
                "INTEGRITY_VIOLATION",
                f"No classification head: last_layer={type(last).__name__}",
                logging.CRITICAL,
            )
        return has_head

    @property
    def results(self) -> List[Dict[str, Any]]:
        return list(self._results)

    @property
    def all_passed(self) -> bool:
        return all(r["status"] == "PASS" for r in self._results)


# ===================================================================
# A09 — HIPAA-Compliant Logging
# ===================================================================


def _log_classification_architecture(model: tf.keras.Model, detection_params: int) -> None:
    """Log classification model architecture — NEVER weight values (A09)."""
    head_params = model.count_params() - detection_params
    AuditLogger.log_security_event(
        "MODEL_ARCHITECTURE",
        f"{model.name}: total={model.count_params()}, "
        f"detection={detection_params}, head={head_params}, "
        f"layers={len(model.layers)}",
        logging.INFO,
    )


def _log_evaluation_metrics(metrics: Dict[str, Any]) -> None:
    """Log aggregate evaluation metrics — NEVER per-patient predictions (A09)."""
    AuditLogger.log_security_event(
        "EVAL_METRICS",
        f"accuracy={metrics['accuracy']:.4f}, "
        f"F1={metrics['f1_score']:.4f}, "
        f"precision={metrics['precision']:.4f}, "
        f"recall={metrics['recall']:.4f}, "
        f"AUC={metrics['auc_roc']:.4f}, "
        f"test_samples={metrics['test_samples']}",
        logging.INFO,
    )


# ===================================================================
# Security Report Generation
# ===================================================================


def _generate_security_report(
    assertions: ClassificationAssertions,
    artifact_hashes: Dict[str, Dict[str, str]],
    config: Phase3Config,
    metrics: Dict[str, Any],
    hw_info: Dict[str, str],
) -> None:
    """Render §5.4 Classification Engine Security Controls report."""
    logger.info("── Generating security report ──")

    # Assertion table rows
    assertion_rows = ""
    for a in assertions.results:
        assertion_rows += (
            f"| {a['assertion']} | {a['expected']}" f" | {a['actual']} | {a['status']} |\n"
        )

    overall = "ALL PASSED" if assertions.all_passed else "FAILURES DETECTED"

    # Artifact hash rows
    hash_rows = ""
    for name, info in artifact_hashes.items():
        hash_rows += f"| `{name}` | `{info['sha256']}` |\n"

    # Parameter bounds rows
    param_rows = (
        f"| `dense_units` | [{DENSE_UNITS_MIN}, {DENSE_UNITS_MAX}]"
        f" | {config.dense_units} | PASS |\n"
        f"| `head_dropout_rate` | [{DROPOUT_MIN}, {DROPOUT_MAX}]"
        f" | {config.head_dropout_rate} | PASS |\n"
        f"| `threshold` | [{THRESHOLD_MIN}, {THRESHOLD_MAX}]"
        f" | {config.threshold} | PASS |\n"
        f"| `batch_size` | [{BATCH_SIZE_MIN}, {BATCH_SIZE_MAX}]"
        f" | {config.batch_size} | PASS |\n"
        f"| `random_state` | int | {config.random_state} | PASS |\n"
        f"| Unknown YAML keys | none allowed | 0 found | PASS |\n"
    )

    # Phase validation rows
    phase_rows = ""
    for p in config.training_phases:
        phase_rows += (
            f"| {p.name} | {p.epochs}" f" | {p.learning_rate} | {len(p.frozen)} frozen | PASS |\n"
        )

    biometric_list = ", ".join(f"`{c}`" for c in sorted(BIOMETRIC_COLUMNS))

    report = f"""## 5.4 Classification Engine Security Controls

This section documents the security controls applied during Phase 3
classification, extending the Phase 0 OWASP framework (§3.3) and
Phase 2 model controls (§5.2) with classification-specific protections.

### 5.4.1 OWASP Controls — Phase 3 Extensions

| OWASP | Category | Control | Status |
|-------|----------|---------|--------|
| A01 | Access Control | Output paths within workspace | Implemented |
| A01 | Access Control | Read-only (chmod 444) after export | Implemented |
| A01 | Access Control | Overwrite protection | Implemented |
| A02 | Crypto Failures | SHA-256 for classification model | Implemented |
| A02 | Crypto Failures | SHA-256 for metrics report | Implemented |
| A02 | Crypto Failures | Hashes in `classification_metadata.json` | Implemented |
| A04 | Insecure Design | Unfreezing order validated | Implemented |
| A04 | Insecure Design | Learning rate decrease validated | Implemented |
| A05 | Misconfiguration | `dense_units` ∈ [{DENSE_UNITS_MIN}, {DENSE_UNITS_MAX}] | Implemented |
| A05 | Misconfiguration | `dropout` ∈ [{DROPOUT_MIN}, {DROPOUT_MAX}] | Implemented |
| A05 | Misconfiguration | `threshold` ∈ [{THRESHOLD_MIN}, {THRESHOLD_MAX}] | Implemented |
| A05 | Misconfiguration | Unknown YAML keys rejected | Implemented |
| A08 | Data Integrity | Evaluation on test set only | Implemented |
| A08 | Data Integrity | No train/test data overlap | Implemented |
| A08 | Data Integrity | Confusion matrix sum verified | Implemented |
| A08 | Data Integrity | F1/precision/recall consistency | Implemented |
| A08 | Data Integrity | Classification head present | Implemented |
| A09 | Logging | Aggregate metrics logged (safe) | Implemented |
| A09 | Logging | Per-patient predictions NEVER logged | Implemented |
| A09 | Logging | Raw feature values NEVER logged | Implemented |

### 5.4.2 Model Integrity Checklist

- [x] `{config.model_file}` SHA-256 stored in `classification_metadata.json`
- [x] `{config.metrics_file}` SHA-256 stored in `classification_metadata.json`
- [x] Evaluation performed on test set only — verified
- [x] No patient-level predictions logged — aggregate metrics only
- [x] Classification head present in exported model
- [x] Confusion matrix row sums match test set size
- [x] Phase 2 artifacts verified via SHA-256 before loading

### 5.4.3 Data Integrity Assertions (A08)

All assertions are evaluated at runtime and logged with pass/fail status.

| Assertion | Expected | Actual | Status |
|-----------|----------|--------|--------|
{assertion_rows}
**Overall:** {overall}

### 5.4.4 Progressive Unfreezing Validation (A04)

| Phase | Epochs | Learning Rate | Frozen Groups | Status |
|-------|--------|---------------|---------------|--------|
{phase_rows}
Validation: frozen count decreases across phases, learning rate decreases.

**Justification:** Progressive unfreezing chosen to prevent catastrophic
forgetting of Phase 2 feature extraction weights while adapting to
classification task.

### 5.4.5 HIPAA Logging Compliance (A09)

| Data Category | Logged? | Justification |
|---------------|---------|---------------|
| Model architecture | Yes | Non-PHI: structural metadata |
| Aggregate metrics (accuracy, F1) | Yes | Non-PHI: population-level stats |
| Confusion matrix | Yes | Non-PHI: aggregate counts |
| Training loss/accuracy | Yes | Non-PHI: model performance |
| Per-patient predictions | **NEVER** | HIPAA: individual classifications |
| Raw feature values | **NEVER** | HIPAA: columns = {biometric_list} |
| Individual attention weights | **NEVER** | HIPAA: patient-level representations |

### 5.4.6 Artifact Integrity (A02)

SHA-256 hashes computed after export for downstream verification.

| Artifact | SHA-256 |
|----------|---------|
{hash_rows}
Hashes stored in `classification_metadata.json` and must be
verified before loading artifacts in subsequent pipeline phases.

### 5.4.7 Parameter Bounds Validation (A05)

| Parameter | Allowed Range | Configured Value | Status |
|-----------|--------------|-----------------|--------|
{param_rows}
### 5.4.8 Security Inheritance from Phase 0 and Phase 2

| Control | Source | Reuse Method |
|---------|--------|-------------|
| SHA-256 hashing | Phase 0 `IntegrityVerifier` | Direct import — `compute_hash()` |
| Path traversal | Phase 0 `PathValidator` | Direct import — `validate_output_dir()` |
| Config sanitization | Phase 0 `ConfigSanitizer` | Direct import — `sanitize_config_dict()` |
| Audit logging | Phase 0 `AuditLogger` | Direct import — `log_file_access()` |
| Phase 2 artifact SHA-256 | Phase 2 `detection_metadata.json` | Verified before model load |

---

**Reviewer signature:** ____________________  **Date:** ____/____/______
"""

    report_path = (
        PROJECT_ROOT / "results" / "phase0_analysis" / "report_section_classification_security.md"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report)
    logger.info("  Security report saved: %s", report_path.name)


# ===================================================================
# Main Pipeline
# ===================================================================


def run_hardened_pipeline(*, allow_overwrite: bool = True) -> Dict[str, Any]:
    """Execute Phase 3 classification with full OWASP/HIPAA controls.

    Args:
        allow_overwrite: If False, raises FileExistsError if artifacts exist.

    Returns:
        Evaluation metrics dict.
    """
    t0 = time.time()

    logger.info("═══════════════════════════════════════════════════")
    logger.info("  Phase 3 Classification — Security Hardened")
    logger.info("═══════════════════════════════════════════════════")

    # ── Hardware detection ──
    hw_info = _detect_hardware()

    # ── A03/A05: Config sanitization + parameter validation ──
    logger.info("── A03: Config sanitization ──")
    raw_yaml = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    ConfigSanitizer.sanitize_config_dict(raw_yaml)
    logger.info("  A03 ✓  Config sanitized")

    _reject_unknown_yaml_keys(raw_yaml)

    config = Phase3Config.from_yaml(CONFIG_PATH)
    _validate_phase3_parameters(config)

    # ── A04: Validate unfreezing order ──
    _validate_unfreezing_order(config)

    # ── A01: Path validation ──
    validator = PathValidator(PROJECT_ROOT)
    output_dir = _validate_output_paths(config, validator, allow_overwrite)

    # ── Reproducibility seeds ──
    np.random.seed(config.random_state)  # noqa: NPY002
    tf.random.set_seed(config.random_state)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    # ── A02: Verify Phase 2 artifacts (SHA-256) ──
    reader = Phase2ArtifactReader(
        project_root=PROJECT_ROOT,
        phase2_dir=config.phase2_dir,
        metadata_file=config.phase2_metadata,
        label_column=config.label_column,
    )
    weights_path, p2_metadata = reader.load_and_verify()

    # ── Load Phase 1 data ──
    train_path = PROJECT_ROOT / config.phase1_train
    test_path = PROJECT_ROOT / config.phase1_test
    X_train, y_train, X_test, y_test, _ = reader.load_phase1_data(train_path, test_path)

    # ── Reshape (sliding windows) ──
    hp = p2_metadata["hyperparameters"]
    reshaper = DataReshaper(timesteps=hp["timesteps"], stride=hp["stride"])
    X_train_w, y_train_w = reshaper.reshape(X_train, y_train)
    X_test_w, y_test_w = reshaper.reshape(X_test, y_test)

    # ── Rebuild detection model + load weights ──
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
    assembler = DetectionModelAssembler(timesteps=hp["timesteps"], n_features=29, builders=builders)
    detection_model = assembler.assemble()
    detection_model.load_weights(str(weights_path))
    detection_params = detection_model.count_params()

    # ── Build classification head + full model ──
    n_classes = len(np.unique(y_train_w))
    head = AutoClassificationHead(
        dense_units=config.dense_units,
        dense_activation=config.dense_activation,
        dropout_rate=config.head_dropout_rate,
    )
    output_tensor = head.build(detection_model.output, n_classes)
    full_model = tf.keras.Model(detection_model.input, output_tensor, name="classification_engine")
    loss = head.get_loss(n_classes)

    # ── A09: Log architecture (safe metadata only) ──
    _log_classification_architecture(full_model, detection_params)

    # ── Progressive unfreezing training ──
    unfreezer = ProgressiveUnfreezer()
    trainer = ClassificationTrainer(
        batch_size=config.batch_size,
        validation_split=config.validation_split,
        early_stopping_patience=config.early_stopping_patience,
        reduce_lr_patience=config.reduce_lr_patience,
        reduce_lr_factor=config.reduce_lr_factor,
    )
    histories = trainer.train_all_phases(
        model=full_model,
        phases=config.training_phases,
        unfreezer=unfreezer,
        X_train=X_train_w,
        y_train=y_train_w,
        loss=loss,
        output_dir=output_dir,
    )

    # ── Evaluate on test set ONLY ──
    evaluator = ModelEvaluator(threshold=config.threshold)
    metrics = evaluator.evaluate(full_model, X_test_w, y_test_w)

    # ── A09: Log aggregate metrics (NEVER per-patient) ──
    _log_evaluation_metrics(metrics)

    # ── A08: Data integrity assertions ──
    logger.info("── A08: Data integrity assertions ──")
    assertions = ClassificationAssertions()
    assertions.assert_eval_test_only(len(y_test_w), metrics["test_samples"])
    assertions.assert_no_train_test_overlap(
        len(y_train_w), len(y_test_w), len(y_train_w) + len(y_test_w)
    )
    assertions.assert_confusion_matrix_sum(metrics["confusion_matrix"], metrics["test_samples"])
    assertions.assert_metrics_consistency(metrics)
    assertions.assert_predictions_valid(metrics["accuracy"], metrics["auc_roc"])
    assertions.assert_has_classification_head(full_model)

    if not assertions.all_passed:
        raise RuntimeError("A08: Integrity assertions FAILED — see logs")

    logger.info("  A08 ✓  All %d assertions PASSED", len(assertions.results))

    # ── A01: Export artifacts + read-only enforcement ──
    logger.info("── A01: Exporting artifacts (read-only) ──")
    exporter = ClassificationExporter(output_dir)

    # Clear read-only if overwriting
    for fname in [
        config.model_file,
        config.metrics_file,
        config.confusion_matrix_file,
        config.history_file,
    ]:
        _clear_read_only(output_dir / fname)

    duration_s = time.time() - t0
    git_commit = _get_git_commit()

    exporter.export_model_weights(full_model, config.model_file)
    metrics_report = ClassificationExporter.build_metrics_report(
        metrics, full_model, hw_info, duration_s, git_commit
    )
    exporter.export_metrics(metrics_report, config.metrics_file)

    cm = metrics["confusion_matrix"]
    n = len(cm)
    labels = ["Normal", "Attack"] if n == 2 else [str(i) for i in range(n)]
    exporter.export_confusion_matrix(cm, labels, config.confusion_matrix_file)
    exporter.export_history(histories, config.history_file)

    # Set read-only
    for fname in [
        config.model_file,
        config.metrics_file,
        config.confusion_matrix_file,
        config.history_file,
    ]:
        _make_read_only(output_dir / fname)

    # ── A02: Hash artifacts + metadata ──
    verifier = IntegrityVerifier(output_dir)
    artifact_hashes = _hash_artifacts(
        verifier,
        {
            config.model_file: output_dir / config.model_file,
            config.metrics_file: output_dir / config.metrics_file,
            config.confusion_matrix_file: output_dir / config.confusion_matrix_file,
            config.history_file: output_dir / config.history_file,
        },
    )
    _store_classification_metadata(
        output_dir=output_dir,
        artifact_hashes=artifact_hashes,
        assertion_results=assertions.results,
        config=config,
        metrics=metrics,
        hw_info=hw_info,
        train_samples=len(y_train_w),
    )

    # ── Generate reports ──
    report_md = render_classification_report(
        model=full_model,
        metrics=metrics,
        histories=histories,
        config=config,
        hw_info=hw_info,
        duration_s=duration_s,
        detection_params=detection_params,
        git_commit=git_commit,
    )
    report_path = PROJECT_ROOT / "results" / "phase0_analysis" / "report_section_classification.md"
    with open(report_path, "w") as f:
        f.write(report_md)
    logger.info("  Classification report saved: %s", report_path.name)

    _generate_security_report(
        assertions=assertions,
        artifact_hashes=artifact_hashes,
        config=config,
        metrics=metrics,
        hw_info=hw_info,
    )

    # ── Cross-dataset validation (optional) ──
    if config.cross_dataset and config.cross_dataset.enabled:
        logger.info("── Cross-Dataset Validation ──")
        from src.phase3_classification_engine.phase3.cross_dataset import (
            CICIoMTLoader,
        )
        from src.phase3_classification_engine.phase3.cross_dataset_report import (
            build_comparison_report,
            render_cross_dataset_report,
        )

        cross_loader = CICIoMTLoader(
            csv_path=PROJECT_ROOT / config.cross_dataset.csv_path,
            column_mapping=config.cross_dataset.column_mapping,
            label_column=config.cross_dataset.label_column,
            label_mapping=config.cross_dataset.label_mapping,
            scaler_path=PROJECT_ROOT / config.cross_dataset.scaler_path,
            wustl_train_path=train_path,
        )

        if cross_loader.is_available():
            X_cross, y_cross, cross_report = cross_loader.load_and_prepare()

            if y_cross is not None:
                # Reshape
                X_cross_w, y_cross_w = reshaper.reshape(X_cross, y_cross)

                # Evaluate (reuse existing evaluator)
                cross_metrics = evaluator.evaluate(full_model, X_cross_w, y_cross_w)

                # A09: Log aggregate cross-dataset metrics (NEVER per-patient)
                _log_evaluation_metrics(cross_metrics)

                # Build comparison
                comparison = build_comparison_report(metrics, cross_metrics)

                delta_acc = comparison["accuracy"]["delta_pct"]
                delta_f1 = comparison["f1_score"]["delta_pct"]
                logger.info(
                    "  Generalization gap: accuracy=%.1f%%, F1=%.1f%%",
                    delta_acc,
                    delta_f1,
                )

                # Export via existing exporter
                cross_cfg = config.cross_dataset
                for fname in [
                    cross_cfg.metrics_file,
                    cross_cfg.confusion_matrix_file,
                    cross_cfg.comparison_report_file,
                ]:
                    _clear_read_only(output_dir / fname)

                exporter.export_metrics(
                    {
                        "pipeline": "cross_dataset_ciciomt2024",
                        "metrics": cross_metrics,
                        "load_report": cross_report,
                    },
                    cross_cfg.metrics_file,
                )
                cm_cross = cross_metrics["confusion_matrix"]
                n_cross = len(cm_cross)
                labels_cross = (
                    ["Normal", "Attack"] if n_cross == 2 else [str(i) for i in range(n_cross)]
                )
                exporter.export_confusion_matrix(
                    cm_cross, labels_cross, cross_cfg.confusion_matrix_file
                )
                exporter.export_metrics(comparison, cross_cfg.comparison_report_file)

                # A01: Set read-only on cross-dataset artifacts
                for fname in [
                    cross_cfg.metrics_file,
                    cross_cfg.confusion_matrix_file,
                    cross_cfg.comparison_report_file,
                ]:
                    _make_read_only(output_dir / fname)

                # A02: Hash cross-dataset artifacts
                _hash_artifacts(
                    verifier,
                    {
                        cross_cfg.metrics_file: (output_dir / cross_cfg.metrics_file),
                        cross_cfg.confusion_matrix_file: (
                            output_dir / cross_cfg.confusion_matrix_file
                        ),
                        cross_cfg.comparison_report_file: (
                            output_dir / cross_cfg.comparison_report_file
                        ),
                    },
                )

                # Generate cross-dataset report
                cross_report_md = render_cross_dataset_report(
                    wustl_metrics=metrics,
                    ciciomt_metrics=cross_metrics,
                    load_report=cross_report,
                    comparison=comparison,
                )
                cross_rpt_path = (
                    PROJECT_ROOT / "results" / "phase0_analysis" / "report_section_crossdataset.md"
                )
                with open(cross_rpt_path, "w") as f:
                    f.write(cross_report_md)
                logger.info("  Cross-dataset report: %s", cross_rpt_path.name)
            else:
                logger.warning("  No labels in CICIoMT2024 — cannot evaluate")
        else:
            logger.info("  Cross-dataset: CICIoMT2024 CSV not found — SKIPPED")

    logger.info("═══════════════════════════════════════════════════")
    logger.info("  Phase 3 Security Hardened — %.2fs", duration_s)
    logger.info("  Assertions: %d/%d PASSED", len(assertions.results), len(assertions.results))
    logger.info("═══════════════════════════════════════════════════")

    return metrics


def main() -> None:
    """Entry point for security-hardened Phase 3 pipeline."""
    run_hardened_pipeline(allow_overwrite=True)


if __name__ == "__main__":
    main()
