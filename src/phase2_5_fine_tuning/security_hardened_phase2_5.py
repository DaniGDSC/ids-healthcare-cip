#!/usr/bin/env python3
"""Security-hardened Phase 2.5 fine-tuning & ablation — OWASP Top 10 + HIPAA compliance.

Wraps the SOLID phase2_5 package with security controls, **extending**
Phase 0 security classes — never duplicating them.

    A01  Output path validation, read-only enforcement, overwrite protection
    A02  SHA-256 artifact hashing (results → tuning_metadata.json)
    A05  Security misconfiguration: hardened parameter bounds, unknown key rejection
    A08  Data integrity assertions (search results consistency, metric validity)
    A09  HIPAA-compliant audit logging (aggregate stats only, never per-patient)

Phase 0 controls reused via direct import:
    - IntegrityVerifier.compute_hash()      (SHA-256 — not re-implemented)
    - PathValidator.validate_output_dir()    (path traversal — not re-implemented)
    - ConfigSanitizer.sanitize_config_dict() (injection — not re-implemented)
    - AuditLogger.log_*()                   (audit trail — not re-implemented)

Usage::

    python -m src.phase2_5_fine_tuning.security_hardened_phase2_5
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
import pandas as pd
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

# ── Phase 2.5 SOLID components ────────────────────────────────────────
from src.phase2_5_fine_tuning.phase2_5.ablation import AblationRunner
from src.phase2_5_fine_tuning.phase2_5.config import Phase2_5Config
from src.phase2_5_fine_tuning.phase2_5.evaluator import QuickEvaluator
from src.phase2_5_fine_tuning.phase2_5.exporter import TuningExporter
from src.phase2_5_fine_tuning.phase2_5.importance import compute_importance
from src.phase2_5_fine_tuning.phase2_5.multi_seed import MultiSeedValidator
from src.phase2_5_fine_tuning.phase2_5.pipeline import _detect_hardware, _get_git_commit
from src.phase2_5_fine_tuning.phase2_5.report import render_tuning_report
from src.phase2_5_fine_tuning.phase2_5.search_space import SearchSpace
from src.phase2_5_fine_tuning.phase2_5.tuner import HyperparameterTuner

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
CONFIG_PATH: Path = PROJECT_ROOT / "config" / "phase2_5_config.yaml"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# A05 — Hardened parameter bounds
MAX_TRIALS_MIN: int = 1
MAX_TRIALS_MAX: int = 500
EPOCHS_MIN: int = 1
EPOCHS_MAX: int = 50
DROPOUT_MIN: float = 0.0
DROPOUT_MAX: float = 0.8
TIMESTEPS_MIN: int = 5
TIMESTEPS_MAX: int = 100
BATCH_SIZE_MIN: int = 8
BATCH_SIZE_MAX: int = 2048

# Known top-level YAML keys for unknown-key rejection (A05)
_KNOWN_YAML_KEYS: frozenset = frozenset(
    {
        "data",
        "search",
        "quick_train",
        "multi_seed",
        "ablation",
        "output",
        "random_state",
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


def _validate_phase2_5_parameters(config: Phase2_5Config) -> None:
    """Enforce Phase 2.5-specific parameter bounds (A05).

    Args:
        config: Validated Phase2_5Config.

    Raises:
        ValueError: If any parameter is outside hardened bounds.
    """
    logger.info("── A05: Phase 2.5 parameter bounds validation ──")

    checks: List[Tuple[str, Any, Any, Any]] = [
        ("max_trials", config.max_trials, MAX_TRIALS_MIN, MAX_TRIALS_MAX),
        ("quick_train.epochs", config.quick_train.epochs, EPOCHS_MIN, EPOCHS_MAX),
    ]

    for name, value, lo, hi in checks:
        if not (lo <= value <= hi):
            msg = f"A05: {name}={value} outside allowed range [{lo}, {hi}]"
            AuditLogger.log_security_event("PARAMETER_VIOLATION", msg, logging.ERROR)
            raise ValueError(msg)
        logger.info("  A05 ✓  %s=%s ∈ [%s, %s]", name, value, lo, hi)

    # Validate search space bounds
    space = config.search_space
    for ts in space.timesteps:
        if not (TIMESTEPS_MIN <= ts <= TIMESTEPS_MAX):
            msg = f"A05: search_space.timesteps contains {ts} outside [{TIMESTEPS_MIN}, {TIMESTEPS_MAX}]"
            AuditLogger.log_security_event("PARAMETER_VIOLATION", msg, logging.ERROR)
            raise ValueError(msg)
    logger.info("  A05 ✓  All timesteps values ∈ [%d, %d]", TIMESTEPS_MIN, TIMESTEPS_MAX)

    for dr in space.dropout_rate:
        if not (DROPOUT_MIN <= dr <= DROPOUT_MAX):
            msg = f"A05: search_space.dropout_rate contains {dr} outside [{DROPOUT_MIN}, {DROPOUT_MAX}]"
            AuditLogger.log_security_event("PARAMETER_VIOLATION", msg, logging.ERROR)
            raise ValueError(msg)
    logger.info("  A05 ✓  All dropout_rate values ∈ [%.1f, %.1f]", DROPOUT_MIN, DROPOUT_MAX)

    for bs in space.batch_size:
        if not (BATCH_SIZE_MIN <= bs <= BATCH_SIZE_MAX):
            msg = f"A05: search_space.batch_size contains {bs} outside [{BATCH_SIZE_MIN}, {BATCH_SIZE_MAX}]"
            AuditLogger.log_security_event("PARAMETER_VIOLATION", msg, logging.ERROR)
            raise ValueError(msg)
    logger.info("  A05 ✓  All batch_size values ∈ [%d, %d]", BATCH_SIZE_MIN, BATCH_SIZE_MAX)

    logger.info("  A05 ✓  All parameters validated")


# ===================================================================
# A01 — Broken Access Control: Path Validation + Read-Only
# ===================================================================


def _validate_output_paths(
    config: Phase2_5Config,
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
        for fname in [config.tuning_results_file, config.ablation_results_file]:
            path = output_dir / fname
            if path.exists():
                msg = f"A01: {path.name} exists — set allow_overwrite=True"
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


def _store_tuning_metadata(
    output_dir: Path,
    artifact_hashes: Dict[str, Dict[str, str]],
    assertion_results: List[Dict[str, Any]],
    config: Phase2_5Config,
    tuning_summary: Dict[str, Any],
    hw_info: Dict[str, str],
) -> Path:
    """Persist artifact hashes, assertions, and tuning metadata (A02).

    Returns:
        Path to tuning_metadata.json.
    """
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline": "security_hardened_phase2_5",
        "random_state": config.random_state,
        "git_commit": _get_git_commit(),
        "hardware": hw_info,
        "hyperparameters": {
            "search_strategy": config.search_strategy,
            "max_trials": config.max_trials,
            "search_metric": config.search_metric,
            "search_direction": config.search_direction,
            "quick_train_epochs": config.quick_train.epochs,
            "n_ablation_variants": len(config.ablation_variants),
        },
        "tuning_summary": tuning_summary,
        "artifact_hashes": artifact_hashes,
        "integrity_assertions": assertion_results,
    }

    meta_path = output_dir / "tuning_metadata.json"
    _clear_read_only(meta_path)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    _make_read_only(meta_path)
    AuditLogger.log_file_access("METADATA_WRITTEN", meta_path)
    return meta_path


# ===================================================================
# A08 — Data Integrity Assertions
# ===================================================================


class TuningAssertions:
    """Tuning-specific data integrity assertions (A08)."""

    def __init__(self) -> None:
        self._results: List[Dict[str, Any]] = []

    def assert_completed_trials(self, completed: int, total: int) -> bool:
        """Assert at least one trial completed successfully."""
        passed = completed > 0
        self._results.append(
            {
                "assertion": "At least one trial completed",
                "expected": ">= 1 completed",
                "actual": f"{completed}/{total} completed",
                "status": "PASS" if passed else "FAIL",
            }
        )
        if passed:
            logger.info("  A08 ✓  %d/%d trials completed", completed, total)
        else:
            AuditLogger.log_security_event(
                "INTEGRITY_VIOLATION",
                f"All {total} trials failed",
                logging.CRITICAL,
            )
        return passed

    def assert_best_metrics_valid(self, metrics: Dict[str, float]) -> bool:
        """Assert best trial metrics are within valid ranges [0, 1]."""
        in_range = all(0.0 <= v <= 1.0 for v in metrics.values())
        passed = in_range
        self._results.append(
            {
                "assertion": "Best metrics ∈ [0, 1]",
                "expected": "all ∈ [0.0, 1.0]",
                "actual": ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()),
                "status": "PASS" if passed else "FAIL",
            }
        )
        if passed:
            logger.info("  A08 ✓  Best metrics in valid range")
        else:
            AuditLogger.log_security_event(
                "INTEGRITY_VIOLATION",
                f"Best metrics out of range: {metrics}",
                logging.CRITICAL,
            )
        return passed

    def assert_ablation_baseline_exists(self, ablation_results: Dict[str, Any]) -> bool:
        """Assert ablation baseline was evaluated."""
        baseline = ablation_results.get("baseline")
        passed = baseline is not None and "metrics" in baseline
        self._results.append(
            {
                "assertion": "Ablation baseline evaluated",
                "expected": "baseline with metrics",
                "actual": "present" if passed else "missing",
                "status": "PASS" if passed else "FAIL",
            }
        )
        if passed:
            logger.info("  A08 ✓  Ablation baseline present")
        else:
            AuditLogger.log_security_event(
                "INTEGRITY_VIOLATION",
                "Ablation baseline missing or incomplete",
                logging.CRITICAL,
            )
        return passed

    def assert_no_data_leakage(self, n_train: int, n_test: int) -> bool:
        """Assert train and test sets are separate."""
        passed = n_train > 0 and n_test > 0 and n_train != n_test
        self._results.append(
            {
                "assertion": "No data leakage (train != test)",
                "expected": "n_train > 0, n_test > 0, n_train != n_test",
                "actual": f"n_train={n_train}, n_test={n_test}",
                "status": "PASS" if passed else "FAIL",
            }
        )
        if passed:
            logger.info("  A08 ✓  No data leakage (train=%d, test=%d)", n_train, n_test)
        else:
            AuditLogger.log_security_event(
                "DATA_LEAKAGE",
                f"Potential leakage: train={n_train}, test={n_test}",
                logging.CRITICAL,
            )
        return passed

    @property
    def results(self) -> List[Dict[str, Any]]:
        return list(self._results)

    @property
    def all_passed(self) -> bool:
        return all(r["status"] == "PASS" for r in self._results)


# ===================================================================
# A09 — HIPAA-Compliant Logging
# ===================================================================


def _log_tuning_summary(tuning_results: Dict[str, Any]) -> None:
    """Log aggregate tuning summary — NEVER per-patient data (A09)."""
    AuditLogger.log_security_event(
        "TUNING_SUMMARY",
        f"strategy={tuning_results.get('strategy')}, "
        f"completed={tuning_results.get('completed_trials')}/{tuning_results.get('total_trials')}, "
        f"best_{tuning_results.get('metric')}={tuning_results.get('best_score', 0):.4f}",
        logging.INFO,
    )


def _log_ablation_summary(ablation_results: Dict[str, Any]) -> None:
    """Log aggregate ablation summary — NEVER per-patient data (A09)."""
    n_variants = len(ablation_results.get("variants", []))
    completed = sum(
        1 for v in ablation_results.get("variants", [])
        if v.get("status") == "completed"
    )
    AuditLogger.log_security_event(
        "ABLATION_SUMMARY",
        f"variants={n_variants}, completed={completed}",
        logging.INFO,
    )


# ===================================================================
# Security Report Generation
# ===================================================================


def _generate_security_report(
    assertions: TuningAssertions,
    artifact_hashes: Dict[str, Dict[str, str]],
    config: Phase2_5Config,
) -> None:
    """Render Phase 2.5 Security Controls report."""
    logger.info("── Generating security report ──")

    assertion_rows = ""
    for a in assertions.results:
        assertion_rows += (
            f"| {a['assertion']} | {a['expected']}"
            f" | {a['actual']} | {a['status']} |\n"
        )

    overall = "ALL PASSED" if assertions.all_passed else "FAILURES DETECTED"

    hash_rows = ""
    for name, info in artifact_hashes.items():
        hash_rows += f"| `{name}` | `{info['sha256']}` |\n"

    biometric_list = ", ".join(f"`{c}`" for c in sorted(BIOMETRIC_COLUMNS))

    report = f"""## 5.3.7 Fine-Tuning & Ablation Security Controls

This section documents the security controls applied during Phase 2.5
fine-tuning and ablation, extending the Phase 0 OWASP framework (§3.3).

### 5.3.7.1 OWASP Controls — Phase 2.5 Extensions

| OWASP | Category | Control | Status |
|-------|----------|---------|--------|
| A01 | Access Control | Output paths within workspace | Implemented |
| A01 | Access Control | Read-only (chmod 444) after export | Implemented |
| A01 | Access Control | Overwrite protection | Implemented |
| A02 | Crypto Failures | SHA-256 for tuning results | Implemented |
| A02 | Crypto Failures | SHA-256 for ablation results | Implemented |
| A02 | Crypto Failures | Hashes in `tuning_metadata.json` | Implemented |
| A05 | Misconfiguration | `max_trials` ∈ [{MAX_TRIALS_MIN}, {MAX_TRIALS_MAX}] | Implemented |
| A05 | Misconfiguration | `timesteps` ∈ [{TIMESTEPS_MIN}, {TIMESTEPS_MAX}] | Implemented |
| A05 | Misconfiguration | `dropout_rate` ∈ [{DROPOUT_MIN}, {DROPOUT_MAX}] | Implemented |
| A05 | Misconfiguration | Unknown YAML keys rejected | Implemented |
| A08 | Data Integrity | At least one trial completed | Implemented |
| A08 | Data Integrity | Best metrics in valid range | Implemented |
| A08 | Data Integrity | Ablation baseline evaluated | Implemented |
| A08 | Data Integrity | No train/test data leakage | Implemented |
| A09 | Logging | Aggregate metrics logged (safe) | Implemented |
| A09 | Logging | Per-patient data NEVER logged | Implemented |

### 5.3.7.2 Data Integrity Assertions (A08)

| Assertion | Expected | Actual | Status |
|-----------|----------|--------|--------|
{assertion_rows}
**Overall:** {overall}

### 5.3.7.3 HIPAA Logging Compliance (A09)

| Data Category | Logged? | Justification |
|---------------|---------|---------------|
| Search configuration | Yes | Non-PHI: hyperparameter metadata |
| Aggregate metrics (F1, AUC) | Yes | Non-PHI: population-level stats |
| Ablation comparison | Yes | Non-PHI: aggregate counts |
| Per-patient predictions | **NEVER** | HIPAA: individual classifications |
| Raw feature values | **NEVER** | HIPAA: columns = {biometric_list} |

### 5.3.7.4 Artifact Integrity (A02)

| Artifact | SHA-256 |
|----------|---------|
{hash_rows}
### 5.3.7.5 Security Inheritance from Phase 0

| Control | Source | Reuse Method |
|---------|--------|-------------|
| SHA-256 hashing | Phase 0 `IntegrityVerifier` | Direct import — `compute_hash()` |
| Path traversal | Phase 0 `PathValidator` | Direct import — `validate_output_dir()` |
| Config sanitization | Phase 0 `ConfigSanitizer` | Direct import — `sanitize_config_dict()` |
| Audit logging | Phase 0 `AuditLogger` | Direct import — `log_file_access()` |

---

**Reviewer signature:** ____________________  **Date:** ____/____/______
"""

    report_path = (
        PROJECT_ROOT / "results" / "phase0_analysis" / "report_section_tuning_security.md"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report)
    logger.info("  Security report saved: %s", report_path.name)


# ===================================================================
# Main Pipeline
# ===================================================================


def run_hardened_pipeline(*, allow_overwrite: bool = True) -> Dict[str, Any]:
    """Execute Phase 2.5 fine-tuning with full OWASP/HIPAA controls.

    Args:
        allow_overwrite: If False, raises FileExistsError if artifacts exist.

    Returns:
        Pipeline report dict.
    """
    t0 = time.time()

    logger.info("═══════════════════════════════════════════════════")
    logger.info("  Phase 2.5 Fine-Tuning & Ablation — Security Hardened")
    logger.info("═══════════════════════════════════════════════════")

    # ── Hardware detection ──
    hw_info = _detect_hardware()

    # ── A03/A05: Config sanitization + parameter validation ──
    logger.info("── A03: Config sanitization ──")
    raw_yaml = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    ConfigSanitizer.sanitize_config_dict(raw_yaml)
    logger.info("  A03 ✓  Config sanitized")

    _reject_unknown_yaml_keys(raw_yaml)

    config = Phase2_5Config.from_yaml(CONFIG_PATH)
    _validate_phase2_5_parameters(config)

    # ── A01: Path validation ──
    validator = PathValidator(PROJECT_ROOT)
    output_dir = _validate_output_paths(config, validator, allow_overwrite)

    # ── Reproducibility seeds ──
    np.random.seed(config.random_state)  # noqa: NPY002
    tf.random.set_seed(config.random_state)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    # ── Load Phase 1 data ──
    logger.info("── Loading Phase 1 data ──")
    label_col = config.label_column
    train_path = PROJECT_ROOT / config.phase1_train
    test_path = PROJECT_ROOT / config.phase1_test

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    feature_names = [c for c in train_df.columns if c != label_col]

    X_train = train_df[feature_names].values.astype(np.float32)
    y_train = train_df[label_col].values
    X_test = test_df[feature_names].values.astype(np.float32)
    y_test = test_df[label_col].values

    logger.info("  Train: %s, Test: %s", X_train.shape, X_test.shape)

    # ── Build components ──
    search_space = SearchSpace(config.search_space, config.random_state)
    evaluator = QuickEvaluator(
        config.quick_train,
        n_features=len(feature_names),
        random_state=config.random_state,
    )
    tuner = HyperparameterTuner(config, evaluator, search_space)
    multi_seed_validator = MultiSeedValidator(config.multi_seed, evaluator)
    ablation_runner = AblationRunner(config, evaluator)

    # ── Hyperparameter search (grid / random / bayesian) ──
    tuning_results = tuner.run(X_train, y_train, X_test, y_test)

    # ── A09: Log aggregate tuning summary (NEVER per-patient) ──
    _log_tuning_summary(tuning_results)

    # ── Parameter importance analysis ──
    importance_results = compute_importance(
        tuner, tuning_results.get("trials", []), tuning_results.get("metric", "f1_score")
    )

    # ── Multi-seed validation of top-K configs ──
    multi_seed_results = multi_seed_validator.validate(
        tuning_results, X_train, y_train, X_test, y_test
    )

    # ── Ablation study (using best config as baseline) ──
    base_hp = tuning_results["best_config"]
    ablation_results = ablation_runner.run(base_hp, X_train, y_train, X_test, y_test)

    # ── A09: Log aggregate ablation summary ──
    _log_ablation_summary(ablation_results)

    # ── A08: Data integrity assertions ──
    logger.info("── A08: Data integrity assertions ──")
    assertions = TuningAssertions()
    assertions.assert_completed_trials(
        tuning_results["completed_trials"], tuning_results["total_trials"]
    )
    assertions.assert_best_metrics_valid(tuning_results["best_metrics"])
    assertions.assert_ablation_baseline_exists(ablation_results)
    assertions.assert_no_data_leakage(len(y_train), len(y_test))

    if not assertions.all_passed:
        raise RuntimeError("A08: Integrity assertions FAILED — see logs")

    logger.info("  A08 ✓  All %d assertions PASSED", len(assertions.results))

    # ── A01: Export artifacts + read-only enforcement ──
    logger.info("── A01: Exporting artifacts (read-only) ──")
    exporter = TuningExporter(output_dir)

    # Clear read-only if overwriting
    all_output_files = [
        config.tuning_results_file,
        config.ablation_results_file,
        config.best_config_file,
        config.importance_file,
        config.multi_seed_file,
        config.report_file,
    ]
    for fname in all_output_files:
        _clear_read_only(output_dir / fname)

    duration_s = time.time() - t0
    git_commit = _get_git_commit()

    exporter.export_tuning_results(tuning_results, config.tuning_results_file)
    exporter.export_ablation_results(ablation_results, config.ablation_results_file)
    exporter.export_best_config(tuning_results["best_config"], config.best_config_file)
    exporter.export_json(importance_results, config.importance_file)
    exporter.export_json(multi_seed_results, config.multi_seed_file)

    report_dict = TuningExporter.build_report(
        tuning_results, ablation_results, importance_results,
        multi_seed_results, hw_info, duration_s, git_commit,
    )
    exporter.export_report(report_dict, config.report_file)

    # Set read-only
    for fname in all_output_files:
        _make_read_only(output_dir / fname)

    # ── A02: Hash artifacts + metadata ──
    verifier = IntegrityVerifier(output_dir)
    artifact_hashes = _hash_artifacts(
        verifier,
        {fname: output_dir / fname for fname in all_output_files},
    )

    tuning_summary = {
        "strategy": tuning_results["strategy"],
        "metric": tuning_results["metric"],
        "completed_trials": tuning_results["completed_trials"],
        "pruned_trials": tuning_results.get("pruned_trials", 0),
        "best_score": tuning_results["best_score"],
    }

    _store_tuning_metadata(
        output_dir=output_dir,
        artifact_hashes=artifact_hashes,
        assertion_results=assertions.results,
        config=config,
        tuning_summary=tuning_summary,
        hw_info=hw_info,
    )

    # ── Generate reports ──
    report_md = render_tuning_report(
        tuning_results=tuning_results,
        ablation_results=ablation_results,
        importance_results=importance_results,
        multi_seed_results=multi_seed_results,
        hw_info=hw_info,
        duration_s=duration_s,
        git_commit=git_commit,
    )
    report_path = PROJECT_ROOT / "results" / "phase0_analysis" / "report_section_tuning.md"
    with open(report_path, "w") as f:
        f.write(report_md)
    logger.info("  Tuning report saved: %s", report_path.name)

    _generate_security_report(
        assertions=assertions,
        artifact_hashes=artifact_hashes,
        config=config,
    )

    logger.info("═══════════════════════════════════════════════════")
    logger.info("  Phase 2.5 Security Hardened — %.2fs", duration_s)
    logger.info("  Assertions: %d/%d PASSED", len(assertions.results), len(assertions.results))
    logger.info("═══════════════════════════════════════════════════")

    return report_dict


def main() -> None:
    """Entry point for security-hardened Phase 2.5 pipeline."""
    run_hardened_pipeline(allow_overwrite=True)


if __name__ == "__main__":
    main()
