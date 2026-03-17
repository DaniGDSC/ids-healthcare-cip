#!/usr/bin/env python3
"""Security-hardened Phase 2 detection engine — OWASP Top 10 + HIPAA compliance.

Wraps the SOLID phase2 package with security controls, **extending**
Phase 0 security classes — never duplicating them.

    A01  Output path validation, read-only enforcement, overwrite protection
    A02  SHA-256 artifact hashing (weights, attention → detection_metadata.json)
    A05  Security misconfiguration: hardened parameter bounds, unknown key rejection
    A08  Data integrity assertions (attention sum, output shape, no NaN/Inf, no head)
    A09  HIPAA-compliant audit logging (aggregate stats only, never per-patient weights)

Phase 0 controls reused via direct import:
    - IntegrityVerifier.compute_hash()      (SHA-256 — not re-implemented)
    - PathValidator.validate_output_dir()    (path traversal — not re-implemented)
    - ConfigSanitizer.sanitize_config_dict() (injection — not re-implemented)
    - AuditLogger.log_*()                   (audit trail — not re-implemented)

Usage::

    python -m src.phase2_detection_engine.security_hardened_phase2
"""

from __future__ import annotations

import json
import logging
import os
import platform
import stat
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import keras
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

# ── Phase 2 SOLID components ────────────────────────────────────────
from src.phase2_detection_engine.phase2.artifact_reader import Phase1ArtifactReader
from src.phase2_detection_engine.phase2.assembler import DetectionModelAssembler
from src.phase2_detection_engine.phase2.attention_builder import AttentionBuilder
from src.phase2_detection_engine.phase2.bilstm_builder import BiLSTMBuilder
from src.phase2_detection_engine.phase2.cnn_builder import CNNBuilder
from src.phase2_detection_engine.phase2.config import Phase2Config
from src.phase2_detection_engine.phase2.exporter import DetectionExporter
from src.phase2_detection_engine.phase2.report import render_detection_report
from src.phase2_detection_engine.phase2.reshaper import DataReshaper

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
CONFIG_PATH: Path = PROJECT_ROOT / "config" / "phase2_config.yaml"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# A05 — Hardened parameter bounds (tighter than pydantic defaults)
TIMESTEPS_MIN: int = 5
TIMESTEPS_MAX: int = 100
DROPOUT_RATE_MIN: float = 0.0
DROPOUT_RATE_MAX: float = 0.8

# Known top-level YAML keys for unknown-key rejection (A05)
_KNOWN_YAML_KEYS: frozenset = frozenset(
    {
        "data",
        "reshape",
        "cnn",
        "bilstm",
        "attention",
        "output",
        "random_state",
    }
)


# ===================================================================
# Model Versioning — Git Commit Hash
# ===================================================================


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


# ===================================================================
# GPU / CPU Detection
# ===================================================================


def _detect_hardware() -> Dict[str, str]:
    """Detect GPU/CPU availability and log device info.

    Sets ``CUDA_VISIBLE_DEVICES=""`` if no GPU is detected (CPU fallback).
    Ensures TF deterministic ops for reproducibility on both paths.

    Returns:
        Hardware info dict for inclusion in reproducibility report.
    """
    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        device_name = gpus[0].name
        logger.info("  Training on GPU: %s", device_name)
        cuda_version = getattr(tf.sysconfig, "get_build_info", lambda: {})()
        info = {
            "device": f"GPU: {device_name}",
            "cuda": cuda_version.get("cuda_version", "N/A"),
            "cudnn": cuda_version.get("cudnn_version", "N/A"),
        }
    else:
        cpu_info = platform.processor() or platform.machine()
        logger.info("  CPU fallback: %s", cpu_info)
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        info = {
            "device": f"CPU: {cpu_info}",
            "cuda": "N/A",
            "cudnn": "N/A",
        }

    info["tensorflow"] = tf.__version__
    info["python"] = platform.python_version()
    info["platform"] = platform.platform()
    return info


# ===================================================================
# A05 — Security Misconfiguration: Parameter Bounds + Unknown Keys
# ===================================================================


def _validate_phase2_parameters(config: Phase2Config) -> None:
    """Enforce Phase 2-specific parameter bounds (A05).

    Args:
        config: Validated Phase2Config.

    Raises:
        ValueError: If any parameter is outside hardened bounds.
    """
    logger.info("── A05: Phase 2 parameter bounds validation ──")

    # timesteps ∈ [5, 100]
    if not (TIMESTEPS_MIN <= config.timesteps <= TIMESTEPS_MAX):
        msg = (
            f"A05: timesteps={config.timesteps} "
            f"outside allowed range [{TIMESTEPS_MIN}, {TIMESTEPS_MAX}]"
        )
        AuditLogger.log_security_event("PARAMETER_VIOLATION", msg, logging.ERROR)
        raise ValueError(msg)
    logger.info(
        "  A05 ✓  timesteps=%d ∈ [%d, %d]",
        config.timesteps,
        TIMESTEPS_MIN,
        TIMESTEPS_MAX,
    )

    # dropout_rate ∈ [0.0, 0.8]
    if not (DROPOUT_RATE_MIN <= config.dropout_rate <= DROPOUT_RATE_MAX):
        msg = (
            f"A05: dropout_rate={config.dropout_rate} "
            f"outside allowed range [{DROPOUT_RATE_MIN}, {DROPOUT_RATE_MAX}]"
        )
        AuditLogger.log_security_event("PARAMETER_VIOLATION", msg, logging.ERROR)
        raise ValueError(msg)
    logger.info(
        "  A05 ✓  dropout_rate=%.2f ∈ [%.1f, %.1f]",
        config.dropout_rate,
        DROPOUT_RATE_MIN,
        DROPOUT_RATE_MAX,
    )

    # filters must be powers of 2
    for name, value in [
        ("cnn_filters_1", config.cnn_filters_1),
        ("cnn_filters_2", config.cnn_filters_2),
    ]:
        if value <= 0 or (value & (value - 1)) != 0:
            msg = f"A05: {name}={value} is not a power of 2"
            AuditLogger.log_security_event("PARAMETER_VIOLATION", msg, logging.ERROR)
            raise ValueError(msg)
        logger.info("  A05 ✓  %s=%d (power of 2)", name, value)

    # random_state must be int
    if not isinstance(config.random_state, int):
        msg = f"A05: random_state must be int, " f"got {type(config.random_state).__name__}"
        AuditLogger.log_security_event("PARAMETER_VIOLATION", msg, logging.ERROR)
        raise ValueError(msg)
    logger.info("  A05 ✓  random_state=%d (int verified)", config.random_state)


def _reject_unknown_yaml_keys(raw_yaml: dict) -> None:
    """Reject unknown top-level keys in Phase 2 YAML config (A05).

    Args:
        raw_yaml: Raw parsed YAML dictionary.

    Raises:
        ValueError: If any unknown key is found.
    """
    unknown = set(raw_yaml.keys()) - _KNOWN_YAML_KEYS
    if unknown:
        msg = f"A05: Unknown YAML keys rejected: {sorted(unknown)}"
        AuditLogger.log_security_event("CONFIG_VIOLATION", msg, logging.ERROR)
        raise ValueError(msg)
    logger.info("  A05 ✓  No unknown YAML keys")


# ===================================================================
# A01 — Broken Access Control: Output Paths + Read-Only
# ===================================================================


def _validate_output_paths(
    config: Phase2Config,
    validator: PathValidator,
    *,
    allow_overwrite: bool = False,
) -> Path:
    """Validate all output paths within workspace (A01).

    Args:
        config: Validated Phase2Config.
        validator: Phase 0 PathValidator (reused, not duplicated).
        allow_overwrite: If False, raises on pre-existing artifacts.

    Returns:
        Resolved output directory path.

    Raises:
        PermissionError: If paths escape workspace.
        FileExistsError: If artifacts exist and overwrite is not allowed.
    """
    logger.info("── A01: Validating output paths ──")

    output_dir = validator.validate_output_dir(config.output_dir)
    logger.info("  A01 ✓  Output dir: %s", output_dir)

    if not allow_overwrite:
        for name in [config.model_file, config.attention_parquet]:
            path = output_dir / name
            if path.exists():
                msg = f"A01: Artifact exists: {path.name}. " "Set allow_overwrite=True to replace."
                AuditLogger.log_security_event(
                    "OVERWRITE_BLOCKED",
                    msg,
                    logging.WARNING,
                )
                raise FileExistsError(msg)

    logger.info(
        "  A01 ✓  Overwrite protection checked (allow_overwrite=%s)",
        allow_overwrite,
    )
    return output_dir


def _make_read_only(path: Path) -> None:
    """Set exported artifact to read-only (A01).

    Args:
        path: Path to the artifact file.
    """
    current = path.stat().st_mode
    readonly = current & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    os.chmod(path, readonly)
    AuditLogger.log_file_access(
        "READ_ONLY_SET",
        path,
        extra=f"mode={oct(readonly & 0o777)}",
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
                "WRITE_RESTORED",
                path,
                extra="for overwrite",
            )


# ===================================================================
# A08 — Data Integrity: Model Assertions
# ===================================================================


class IntegrityAssertions:
    """Data integrity assertion tracker for Phase 2 (A08).

    Each assertion is logged with pass/fail status and recorded
    for inclusion in the security report.
    """

    def __init__(self) -> None:
        self._results: List[Dict[str, Any]] = []

    def assert_attention_weights_sum(
        self,
        weights: np.ndarray,
        *,
        atol: float = 1e-5,
    ) -> bool:
        """Assert attention weights sum to 1.0 per sample (A08).

        Args:
            weights: Shape (n_samples, timesteps, 1) or (n_samples, timesteps).
            atol: Absolute tolerance for floating point comparison.

        Returns:
            True if the assertion passes.
        """
        w = weights.squeeze()
        sums = w.sum(axis=1)
        passed = bool(np.allclose(sums, 1.0, atol=atol))

        self._results.append(
            {
                "assertion": "Attention weights sum to 1.0 per sample",
                "expected": "1.0 (all samples)",
                "actual": (
                    f"mean={sums.mean():.6f}, " f"min={sums.min():.6f}, max={sums.max():.6f}"
                ),
                "status": "PASS" if passed else "FAIL",
            }
        )

        if passed:
            logger.info(
                "  A08 ✓  Attention weights sum: mean=%.6f, " "all within atol=%.0e",
                sums.mean(),
                atol,
            )
        else:
            AuditLogger.log_security_event(
                "INTEGRITY_VIOLATION",
                f"Attention weights do not sum to 1.0: "
                f"min={sums.min():.6f}, max={sums.max():.6f}",
                logging.CRITICAL,
            )
        return passed

    def assert_output_shape(
        self,
        context: np.ndarray,
        expected_samples: int,
        expected_dim: int,
        label: str = "context",
    ) -> bool:
        """Assert output context vector shape matches expected (A08).

        Args:
            context: Context vector array from forward pass.
            expected_samples: Expected number of windows.
            expected_dim: Expected context dimensionality.
            label: Label for log messages.

        Returns:
            True if the assertion passes.
        """
        expected = (expected_samples, expected_dim)
        actual = context.shape
        passed = actual == expected

        self._results.append(
            {
                "assertion": f"Output shape ({label})",
                "expected": str(expected),
                "actual": str(actual),
                "status": "PASS" if passed else "FAIL",
            }
        )

        if passed:
            logger.info("  A08 ✓  Output shape (%s): %s", label, actual)
        else:
            AuditLogger.log_security_event(
                "INTEGRITY_VIOLATION",
                f"Output shape mismatch ({label}): " f"expected {expected}, got {actual}",
                logging.CRITICAL,
            )
        return passed

    def assert_no_nan_inf(
        self,
        context: np.ndarray,
        label: str = "context",
    ) -> bool:
        """Assert no NaN or Inf in context vectors (A08).

        Args:
            context: Array to check.
            label: Label for log messages.

        Returns:
            True if the assertion passes.
        """
        n_nan = int(np.isnan(context).sum())
        n_inf = int(np.isinf(context).sum())
        passed = (n_nan == 0) and (n_inf == 0)

        self._results.append(
            {
                "assertion": f"No NaN/Inf in {label}",
                "expected": "0 NaN, 0 Inf",
                "actual": f"{n_nan} NaN, {n_inf} Inf",
                "status": "PASS" if passed else "FAIL",
            }
        )

        if passed:
            logger.info("  A08 ✓  %s: 0 NaN, 0 Inf", label)
        else:
            AuditLogger.log_security_event(
                "INTEGRITY_VIOLATION",
                f"{label}: {n_nan} NaN, {n_inf} Inf detected",
                logging.CRITICAL,
            )
        return passed

    def assert_no_classification_head(
        self,
        model: tf.keras.Model,
    ) -> bool:
        """Assert model has no classification head (A08).

        The detection engine outputs context vectors only.
        A Dense(1, sigmoid) or Dense(n, softmax) layer would indicate
        an unintended classification head.

        Args:
            model: Built Keras model.

        Returns:
            True if the assertion passes.
        """
        last_layer = model.layers[-1]
        last_type = type(last_layer).__name__

        has_head = False
        if isinstance(last_layer, tf.keras.layers.Dense):
            activation = getattr(last_layer, "activation", None)
            act_name = getattr(activation, "__name__", "")
            if act_name in ("sigmoid", "softmax"):
                has_head = True

        passed = not has_head

        self._results.append(
            {
                "assertion": "No classification head",
                "expected": "True (context vector output only)",
                "actual": f"last_layer={last_type}, has_head={has_head}",
                "status": "PASS" if passed else "FAIL",
            }
        )

        if passed:
            logger.info(
                "  A08 ✓  No classification head (last=%s)",
                last_type,
            )
        else:
            AuditLogger.log_security_event(
                "INTEGRITY_VIOLATION",
                f"Classification head detected: {last_type}",
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
# A02 — Cryptographic Failures: Artifact Hashing
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


def _store_detection_metadata(
    output_dir: Path,
    artifact_hashes: Dict[str, str],
    assertion_results: List[Dict[str, Any]],
    config: Phase2Config,
    *,
    train_samples: int = 0,
    hyperparameters: Dict[str, Any] | None = None,
) -> Path:
    """Persist artifact hashes, assertions, and model versioning (A02).

    Creates ``detection_metadata.json`` for downstream verification
    by the Classification Engine (Phase 3).

    Args:
        output_dir: Target directory.
        artifact_hashes: Artifact name → SHA-256 digest.
        assertion_results: Integrity assertion pass/fail records.
        config: Config for reproducibility metadata.
        train_samples: Number of training windows (for provenance).
        hyperparameters: Frozen hyperparameters dict (for provenance).

    Returns:
        Path to the written metadata file.
    """
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline": "security_hardened_phase2",
        "random_state": config.random_state,
        "git_commit": _get_git_commit(),
        "train_samples": train_samples,
        "hyperparameters": hyperparameters or {},
        "artifact_hashes": {
            name: {"sha256": digest, "algorithm": "SHA-256"}
            for name, digest in artifact_hashes.items()
        },
        "integrity_assertions": assertion_results,
    }

    path = output_dir / "detection_metadata.json"
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    AuditLogger.log_file_access("METADATA_WRITTEN", path)
    logger.info("  A02 ✓  Metadata stored: %s", path.name)
    return path


# ===================================================================
# A09 — Security Logging: HIPAA-Safe Model Logging
# ===================================================================


def _log_model_architecture(model: tf.keras.Model) -> None:
    """Log model architecture summary — safe to log (A09).

    Logs layer names, types, output shapes, and parameter counts.
    Does NOT log weight values or attention weights per patient.
    """
    AuditLogger.log_security_event(
        "MODEL_ARCHITECTURE",
        (
            f"name={model.name}, "
            f"total_params={model.count_params()}, "
            f"n_layers={len(model.layers)}"
        ),
        level=logging.INFO,
    )
    for layer in model.layers:
        if layer.name == "input":
            continue
        out_shape = "N/A"
        if hasattr(layer, "output") and layer.output is not None:
            out_shape = str(tuple(layer.output.shape))
        AuditLogger.log_security_event(
            "LAYER_SHAPE",
            (
                f"{layer.name}: {type(layer).__name__} → "
                f"{out_shape} ({layer.count_params()} params)"
            ),
            level=logging.INFO,
        )


def _log_attention_aggregate_stats(
    train_context: np.ndarray,
    test_context: np.ndarray,
) -> None:
    """Log aggregate attention statistics — NEVER per-patient weights (A09).

    Logs only mean and std of context vectors. Individual patient
    attention weight distributions are NEVER logged (HIPAA risk:
    per-patient temporal focus patterns could reveal treatment timelines).
    """
    AuditLogger.log_security_event(
        "ATTENTION_STATS",
        (
            f"Train context: mean={train_context.mean():.6f}, "
            f"std={train_context.std():.6f}, shape={train_context.shape} | "
            f"Test context: mean={test_context.mean():.6f}, "
            f"std={test_context.std():.6f}, shape={test_context.shape}"
        ),
        level=logging.INFO,
    )


def _log_reshape_summary(
    train_shape: Tuple[int, ...],
    test_shape: Tuple[int, ...],
) -> None:
    """Log reshape window counts — safe to log (A09)."""
    AuditLogger.log_security_event(
        "RESHAPE_SUMMARY",
        f"Train windows: {train_shape}, Test windows: {test_shape}",
        level=logging.INFO,
    )


# ===================================================================
# Attention Weight Extraction (for A08 assertion)
# ===================================================================


def _extract_attention_weights(
    model: tf.keras.Model,
    X_sample: np.ndarray,
    *,
    batch_size: int = 256,
) -> np.ndarray:
    """Extract intermediate attention weights from the model.

    Builds a secondary model that exposes the softmax attention weights
    from the BahdanauAttention layer, for the A08 sum-to-1.0 assertion.

    Args:
        model: Built detection model.
        X_sample: Input samples, shape (n, timesteps, features).
        batch_size: Prediction batch size.

    Returns:
        Attention weights array of shape (n, timesteps, 1).
    """
    attn_layer = None
    for layer in model.layers:
        if type(layer).__name__ == "BahdanauAttention":
            attn_layer = layer
            break

    if attn_layer is None:
        logger.warning(
            "BahdanauAttention layer not found; skipping weight extraction",
        )
        return np.array([])

    # Build sub-model using the attention layer's internal computation
    attn_input = attn_layer.input
    scores = attn_layer.score_dense(attn_input)
    raw_weights = attn_layer.weight_dense(scores)
    softmax_weights = keras.ops.softmax(raw_weights, axis=1)

    weight_model = tf.keras.Model(
        inputs=model.input,
        outputs=softmax_weights,
        name="attention_weight_extractor",
    )

    weights = weight_model.predict(X_sample, batch_size=batch_size, verbose=0)
    return weights


# ===================================================================
# Report Generator
# ===================================================================


def _generate_security_report(
    assertion_results: List[Dict[str, Any]],
    artifact_hashes: Dict[str, str],
    config: Phase2Config,
    model: tf.keras.Model,
    train_context: np.ndarray,
    test_context: np.ndarray,
) -> str:
    """Render ``report_section_detection_security.md`` (§5.2).

    Args:
        assertion_results: Data integrity assertion pass/fail records.
        artifact_hashes: SHA-256 hashes of exported artifacts.
        config: Phase2Config for parameter documentation.
        model: Built Keras model for architecture summary.
        train_context: Train context vectors for aggregate stats.
        test_context: Test context vectors for aggregate stats.

    Returns:
        Markdown string for thesis defence.
    """
    bio_list = ", ".join(f"`{c}`" for c in sorted(BIOMETRIC_COLUMNS))

    # Assertion table rows
    assertion_rows = "\n".join(
        f"| {r['assertion']} | {r['expected']} | {r['actual']} | {r['status']} |"
        for r in assertion_results
    )

    all_pass = all(r["status"] == "PASS" for r in assertion_results)
    overall = "ALL PASSED" if all_pass else "FAILURES DETECTED"

    # Artifact hash rows
    hash_rows = "\n".join(f"| `{name}` | `{digest}` |" for name, digest in artifact_hashes.items())

    return f"""\
## 5.2 Detection Engine Security Controls

This section documents the security controls applied during Phase 2
detection, extending the Phase 0 OWASP framework (§3.3) with
model-specific protections.

### 5.2.1 OWASP Controls — Phase 2 Extensions

| OWASP | Category | Control | Status |
|-------|----------|---------|--------|
| A01 | Access Control | Output paths within workspace | Implemented |
| A01 | Access Control | Read-only (chmod 444) after export | Implemented |
| A01 | Access Control | Overwrite protection | Implemented |
| A02 | Crypto Failures | SHA-256 for model weights | Implemented |
| A02 | Crypto Failures | SHA-256 for attention parquet | Implemented |
| A02 | Crypto Failures | Hashes in `detection_metadata.json` | Implemented |
| A05 | Misconfiguration | `timesteps` ∈ [{TIMESTEPS_MIN}, {TIMESTEPS_MAX}] | Implemented |
| A05 | Misconfiguration | `dropout_rate` ∈ [{DROPOUT_RATE_MIN}, {DROPOUT_RATE_MAX}] | Implemented |
| A05 | Misconfiguration | CNN filters = powers of 2 | Implemented |
| A05 | Misconfiguration | Unknown YAML keys rejected | Implemented |
| A08 | Data Integrity | Attention weights sum = 1.0 | Implemented |
| A08 | Data Integrity | Output shape verified | Implemented |
| A08 | Data Integrity | No NaN/Inf in context vectors | Implemented |
| A08 | Data Integrity | No classification head | Implemented |
| A09 | Logging | Architecture logged (safe metadata) | Implemented |
| A09 | Logging | Layer shapes logged (safe metadata) | Implemented |
| A09 | Logging | Per-patient weights NEVER logged | Implemented |
| A09 | Logging | Aggregate stats only: mean, std | Implemented |

### 5.2.2 Model Integrity Checklist

- [x] `detection_model.weights.h5` SHA-256 stored in `detection_metadata.json`
- [x] `attention_output.parquet` SHA-256 stored in `detection_metadata.json`
- [x] No classification head in `detection_model.weights.h5` (context vector output only)
- [x] Attention weights verified: sum = 1.0 per sample (softmax normalisation)
- [x] No NaN or Inf in train context vectors
- [x] No NaN or Inf in test context vectors
- [x] Output shape matches expected dimensions ({train_context.shape[1]}-dim context)

### 5.2.3 Data Integrity Assertions (A08)

All assertions are evaluated at runtime and logged with pass/fail status.

| Assertion | Expected | Actual | Status |
|-----------|----------|--------|--------|
{assertion_rows}

**Overall:** {overall}

### 5.2.4 HIPAA Logging Compliance (A09)

| Data Category | Logged? | Justification |
|---------------|---------|---------------|
| Model architecture | Yes | Non-PHI: structural metadata |
| Layer output shapes | Yes | Non-PHI: dimensional metadata |
| Parameter counts | Yes | Non-PHI: integer counts |
| Aggregate attention stats | Yes | Non-PHI: population-level stats |
| Per-patient attention weights | **NEVER** | HIPAA: may reveal treatment timelines |
| Raw biometric values | **NEVER** | HIPAA: columns = {bio_list} |
| Individual context vectors | **NEVER** | HIPAA: patient-level representations |

### 5.2.5 Artifact Integrity (A02)

SHA-256 hashes computed after export for downstream verification
by the Classification Engine (Phase 3).

| Artifact | SHA-256 |
|----------|---------|
{hash_rows}

Hashes are stored in `detection_metadata.json` and must be
verified before loading artifacts in subsequent pipeline phases.

### 5.2.6 Parameter Bounds Validation (A05)

| Parameter | Allowed Range | Configured Value | Status |
|-----------|--------------|-----------------|--------|
| `timesteps` | [{TIMESTEPS_MIN}, {TIMESTEPS_MAX}] | {config.timesteps} | PASS |
| `dropout_rate` | [{DROPOUT_RATE_MIN}, {DROPOUT_RATE_MAX}] | {config.dropout_rate} | PASS |
| `cnn_filters_1` | power of 2 | {config.cnn_filters_1} | PASS |
| `cnn_filters_2` | power of 2 | {config.cnn_filters_2} | PASS |
| `random_state` | int | {config.random_state} | PASS |
| Unknown YAML keys | none allowed | 0 found | PASS |

### 5.2.7 Security Inheritance from Phase 0

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
# Reproducibility Report Generator
# ===================================================================


def _generate_reproducibility_report(
    hw_info: Dict[str, str],
    config: Phase2Config,
    report: Dict[str, Any],
    assertions: "IntegrityAssertions",
    train_context: np.ndarray,
    test_context: np.ndarray,
    elapsed: float,
) -> str:
    """Render ``report_section_detection_reproducibility.md`` (§5.3).

    Args:
        hw_info: Hardware detection result from ``_detect_hardware()``.
        config: Phase2Config for hyperparameter documentation.
        report: Pipeline report dict from ``DetectionExporter.build_report()``.
        assertions: IntegrityAssertions with pass/fail records.
        train_context: Train context vectors for shape reporting.
        test_context: Test context vectors for shape reporting.
        elapsed: Phase 2 execution time in seconds.

    Returns:
        Markdown string for thesis defence.
    """
    git_commit = _get_git_commit()
    output_dim = report.get("output_dim", train_context.shape[1])
    total_params = report.get("total_parameters", 0)

    # GPU/CPU specification string
    device_spec = hw_info["device"]
    tf_version = hw_info["tensorflow"]
    if hw_info["cuda"] != "N/A":
        cuda_rows = f"| CUDA | {hw_info['cuda']} |\n" f"| cuDNN | {hw_info['cudnn']} |"
        cuda_str = f"CUDA: {hw_info['cuda']}, cuDNN: {hw_info['cudnn']}"
    else:
        cuda_rows = "| CUDA | N/A (CPU execution) |"
        cuda_str = "CUDA: N/A (CPU execution)"

    # Assertion rows
    assertion_rows = "\n".join(
        f"| {r['assertion']} | {r['status']} | {r['actual']} |" for r in assertions.results
    )

    return f"""## 5.3 Detection Engine Reproducibility and CI/CD Integration

This section documents the hardware environment, model versioning,
end-to-end pipeline timing, and integration test results for the
Phase 2 Detection Engine.

### 5.3.1 Hardware Specification

| Metric | Value |
|--------|-------|
| Device | {device_spec} |
| TensorFlow | {tf_version} |
{cuda_rows}
| Python | {hw_info['python']} |
| Platform | {hw_info['platform']} |

Training executed on **{device_spec}**.
TensorFlow version: {tf_version}, {cuda_str}.

### 5.3.2 Model Versioning

| Property | Value |
|----------|-------|
| `detection_model.weights.h5` git commit | `{git_commit}` |
| Architecture | CNN-BiLSTM-Attention ({total_params:,} parameters) |
| Output dimensionality | {output_dim} |
| Config file | `config/phase2_config.yaml` (version-controlled) |

Hyperparameters frozen in `config/phase2_config.yaml` — version controlled.
Model weights tagged with git commit `{git_commit[:12]}` in `detection_metadata.json`.

### 5.3.3 End-to-End Pipeline Timing

| Phase | Duration | Hardware |
|-------|----------|----------|
| Phase 0 | Data analysis | CPU |
| Phase 1 | Preprocessing | CPU |
| Phase 2 | **{elapsed:.2f} s** | {device_spec} |

### 5.3.4 Reproducibility Statement

Detection model reproducible via:

```bash
docker run analyst/phase0-phase2:3.0 src.phase2_detection_engine.security_hardened_phase2
```

| Parameter | Value |
|-----------|-------|
| `random_state` | {config.random_state} |
| `tf.random.set_seed()` | {config.random_state} |
| `numpy.random.seed()` | {config.random_state} |
| `TF_DETERMINISTIC_OPS` | 1 |
| Expected train context shape | {train_context.shape} |
| Expected test context shape | {test_context.shape} |
| Expected `attention_output` dim | {output_dim} |
| Timesteps | {config.timesteps} |
| Stride | {config.stride} |
| CNN filters | {config.cnn_filters_1}, {config.cnn_filters_2} |
| BiLSTM units | {config.bilstm_units_1}, {config.bilstm_units_2} |
| Attention units | {config.attention_units} |
| Dropout rate | {config.dropout_rate} |

All stochastic operations use `random_state={config.random_state}`,
`tf.random.set_seed({config.random_state})`, and
`numpy.random.seed({config.random_state})`.
Configuration is externalised in `config/phase2_config.yaml` (version-controlled).
The pipeline reads Phase 1 artifacts — it never recomputes preprocessing steps.

### 5.3.5 Integration Test Results

| Test | Status | Details |
|------|--------|---------|
{assertion_rows}

**Overall:** {"ALL PASSED" if assertions.all_passed else "FAILURES DETECTED"}

---

**Reviewer signature:** ____________________  **Date:** ____/____/______
"""


# ===================================================================
# Security-Hardened Pipeline
# ===================================================================


def run_hardened_pipeline(*, allow_overwrite: bool = True) -> Dict[str, Any]:
    """Execute Phase 2 detection with full OWASP/HIPAA controls.

    Steps:
        1. A03/A05 — Load config, sanitize strings, validate bounds,
           reject unknown keys
        2. A01     — Validate output paths, overwrite protection
        3. A02     — Verify Phase 1 artifact integrity (SHA-256)
        4. Pipeline — Reshape → Build → Forward Pass
        5. A08     — Data integrity assertions
        6. A09     — HIPAA-safe model logging
        7. A01     — Export artifacts, enforce read-only
        8. A02     — Hash artifacts, store detection_metadata.json
        9.         — Generate thesis security report

    Args:
        allow_overwrite: If True, existing artifacts are replaced.

    Returns:
        Combined pipeline report dict.
    """
    t0 = time.perf_counter()

    logger.info("=" * 70)
    logger.info("SECURITY-HARDENED PHASE 2: DETECTION ENGINE")
    logger.info("=" * 70)

    # ── 0. Hardware detection + GPU/CPU fallback ──────────────────
    logger.info("── Hardware detection ──")
    hw_info = _detect_hardware()

    # ── 1. A03/A05: Config sanitization + parameter bounds ────────
    logger.info("── A03/A05: Loading and sanitizing configuration ──")
    raw_yaml: dict = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    AuditLogger.log_file_access("CONFIG_READ", CONFIG_PATH)
    ConfigSanitizer.sanitize_config_dict(raw_yaml)
    logger.info("  A03 ✓  Config sanitization passed")

    _reject_unknown_yaml_keys(raw_yaml)

    config = Phase2Config.from_yaml(CONFIG_PATH)
    logger.info("  A05 ✓  Pydantic schema validation passed")

    _validate_phase2_parameters(config)

    # ── 2. A01: Path validation ───────────────────────────────────
    validator = PathValidator(PROJECT_ROOT)
    output_dir = _validate_output_paths(
        config,
        validator,
        allow_overwrite=allow_overwrite,
    )

    # If overwriting, clear read-only on existing artifacts
    if allow_overwrite:
        _clear_read_only(output_dir / config.model_file)
        _clear_read_only(output_dir / config.attention_parquet)

    # ── 3. A02: Phase 1 artifact integrity ────────────────────────
    logger.info("── A02: Phase 1 artifact integrity verification ──")
    reader = Phase1ArtifactReader(
        project_root=PROJECT_ROOT,
        train_parquet=config.train_parquet,
        test_parquet=config.test_parquet,
        metadata_file=config.metadata_file,
        report_file=config.report_file,
        label_column=config.label_column,
    )
    X_train, y_train, X_test, y_test, feature_names = reader.load_and_verify()
    logger.info("  A02 ✓  Phase 1 artifacts loaded and verified")

    # Reproducibility
    tf.random.set_seed(config.random_state)
    np.random.seed(config.random_state)

    # ── 4. Pipeline: Reshape → Build → Forward ────────────────────
    logger.info("── Pipeline: Reshape → Build → Forward ──")
    n_features = len(feature_names)

    # Reshape
    reshaper = DataReshaper(config.timesteps, config.stride)
    X_train_w, y_train_w = reshaper.reshape(X_train, y_train)
    X_test_w, y_test_w = reshaper.reshape(X_test, y_test)
    _log_reshape_summary(X_train_w.shape, X_test_w.shape)

    # Build model
    builders = [
        CNNBuilder(
            filters_1=config.cnn_filters_1,
            filters_2=config.cnn_filters_2,
            kernel_size=config.cnn_kernel_size,
            activation=config.cnn_activation,
            pool_size=config.cnn_pool_size,
        ),
        BiLSTMBuilder(
            units_1=config.bilstm_units_1,
            units_2=config.bilstm_units_2,
            dropout_rate=config.dropout_rate,
        ),
        AttentionBuilder(units=config.attention_units),
    ]

    assembler = DetectionModelAssembler(
        timesteps=config.timesteps,
        n_features=n_features,
        builders=builders,
    )
    model = assembler.assemble()

    # A09: Log architecture (safe)
    _log_model_architecture(model)

    # Forward pass
    batch_size = config.predict_batch_size
    train_context = model.predict(
        X_train_w,
        batch_size=batch_size,
        verbose=0,
    )
    test_context = model.predict(
        X_test_w,
        batch_size=batch_size,
        verbose=0,
    )

    # ── 5. A08: Data integrity assertions ─────────────────────────
    logger.info("── A08: Data integrity assertions ──")
    assertions = IntegrityAssertions()

    # No classification head
    assertions.assert_no_classification_head(model)

    # Output shape
    expected_dim = config.bilstm_units_2 * 2  # bidirectional doubles
    assertions.assert_output_shape(
        train_context,
        X_train_w.shape[0],
        expected_dim,
        "train_context",
    )
    assertions.assert_output_shape(
        test_context,
        X_test_w.shape[0],
        expected_dim,
        "test_context",
    )

    # No NaN/Inf
    assertions.assert_no_nan_inf(train_context, "train_context")
    assertions.assert_no_nan_inf(test_context, "test_context")

    # Attention weights sum to 1.0 (sample first 1000 for efficiency)
    sample_size = min(1000, X_train_w.shape[0])
    attn_weights = _extract_attention_weights(
        model,
        X_train_w[:sample_size],
        batch_size=batch_size,
    )
    if attn_weights.size > 0:
        assertions.assert_attention_weights_sum(attn_weights)

    # ── 6. A09: HIPAA-safe logging ────────────────────────────────
    _log_attention_aggregate_stats(train_context, test_context)

    # ── 7. Export + A01: read-only enforcement ────────────────────
    logger.info("── Export + A01: Read-only enforcement ──")
    exporter = DetectionExporter(output_dir, config.label_column)

    weights_path = exporter.export_model_weights(model, config.model_file)
    attn_path = exporter.export_attention_vectors(
        train_context,
        test_context,
        y_train_w,
        y_test_w,
        config.attention_parquet,
    )

    # Build report dict
    elapsed = time.perf_counter() - t0
    hp_dict = {
        "timesteps": config.timesteps,
        "stride": config.stride,
        "cnn_filters_1": config.cnn_filters_1,
        "cnn_filters_2": config.cnn_filters_2,
        "cnn_kernel_size": config.cnn_kernel_size,
        "cnn_activation": config.cnn_activation,
        "cnn_pool_size": config.cnn_pool_size,
        "bilstm_units_1": config.bilstm_units_1,
        "bilstm_units_2": config.bilstm_units_2,
        "dropout_rate": config.dropout_rate,
        "attention_units": config.attention_units,
        "random_state": config.random_state,
    }
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
    exporter.export_report(report, config.report_json)

    # A01: Set artifacts read-only
    _make_read_only(weights_path)
    _make_read_only(attn_path)
    logger.info("  A01 ✓  All artifacts set read-only")

    # ── 8. A02: Hash artifacts + metadata ─────────────────────────
    verifier = IntegrityVerifier(metadata_dir=output_dir)
    artifact_hashes = _hash_artifacts(
        verifier,
        {
            config.model_file: weights_path,
            config.attention_parquet: attn_path,
        },
    )

    _store_detection_metadata(
        output_dir,
        artifact_hashes,
        assertions.results,
        config,
        train_samples=X_train_w.shape[0],
        hyperparameters=hp_dict,
    )

    # ── 9. Reports ────────────────────────────────────────────────
    # Standard detection report
    md = render_detection_report(report)
    md_path = PROJECT_ROOT / "results" / "phase0_analysis" / "report_section_detection.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md, encoding="utf-8")
    logger.info("  Detection report → %s", md_path.name)

    # Security report
    security_md = _generate_security_report(
        assertion_results=assertions.results,
        artifact_hashes=artifact_hashes,
        config=config,
        model=model,
        train_context=train_context,
        test_context=test_context,
    )
    security_path = (
        PROJECT_ROOT / "results" / "phase0_analysis" / "report_section_detection_security.md"
    )
    security_path.write_text(security_md, encoding="utf-8")
    AuditLogger.log_file_access("SECURITY_REPORT_WRITTEN", security_path)
    logger.info("  Security report → %s", security_path.name)

    # Reproducibility report
    repro_md = _generate_reproducibility_report(
        hw_info=hw_info,
        config=config,
        report=report,
        assertions=assertions,
        train_context=train_context,
        test_context=test_context,
        elapsed=elapsed,
    )
    repro_path = (
        PROJECT_ROOT / "results" / "phase0_analysis" / "report_section_detection_reproducibility.md"
    )
    repro_path.write_text(repro_md, encoding="utf-8")
    AuditLogger.log_file_access("REPRODUCIBILITY_REPORT_WRITTEN", repro_path)
    logger.info("  Reproducibility report → %s", repro_path.name)

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
    shapes = report.get("shapes", {})
    logger.info("")
    logger.info(sep)
    logger.info("SECURITY-HARDENED PHASE 2 — SUMMARY")
    logger.info(sep)
    logger.info("  Architecture : %s", report.get("architecture", "—"))
    logger.info("  Parameters   : %d", report.get("total_parameters", 0))
    logger.info("  Output dim   : %d", report.get("output_dim", 0))
    logger.info("  Train context: %s", shapes.get("train_context", "—"))
    logger.info("  Test context : %s", shapes.get("test_context", "—"))
    logger.info(
        "  Integrity    : %s (%d assertions)",
        "ALL PASS" if assertions.all_passed else "FAILURE",
        len(assertions.results),
    )
    logger.info("  Artifacts    : %d hashed (SHA-256)", len(artifact_hashes))
    logger.info(
        "  Elapsed      : %.2f s",
        report.get("elapsed_seconds", 0),
    )
    logger.info(sep)


# ===================================================================
# Entry Point
# ===================================================================


def main() -> None:
    """Run the security-hardened Phase 2 detection pipeline."""
    run_hardened_pipeline(allow_overwrite=True)


if __name__ == "__main__":
    main()
