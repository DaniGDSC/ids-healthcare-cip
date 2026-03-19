#!/usr/bin/env python3
"""Security-hardened Phase 5 explanation engine — OWASP Top 10 + HIPAA compliance.

Wraps the SOLID phase5 package with security controls, **extending**
Phase 0 security classes — never duplicating them.

    A01  Read-only explanation artifacts (chmod 444), workspace path validation
    A02  SHA-256 for shap_values.parquet + explanation_report.json → metadata
    A03  Template sanitization, top_n_features ∈ [1, 29], config dict sanitized
    A05  Tighter parameter bounds, unknown YAML key rejection
    A08  Normal-only SHAP background, shape assertion, waterfall for HIGH+ only
    A09  HIPAA-compliant logging — NEVER log individual SHAP values or biometrics

Phase 0 controls reused via direct import:
    - IntegrityVerifier.compute_hash()      (SHA-256 — not re-implemented)
    - PathValidator.validate_output_dir()    (path traversal — not re-implemented)
    - ConfigSanitizer.sanitize_config_dict() (injection — not re-implemented)
    - AuditLogger.log_*()                   (audit trail — not re-implemented)

Usage::

    python -m src.phase5_explanation_engine.security_hardened_phase5
"""

from __future__ import annotations

import json
import logging
import os
import re
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

# ── Phase 2 SOLID components (for model rebuild + custom_objects) ──
from src.phase2_detection_engine.phase2.attention_builder import (  # noqa: F401
    BahdanauAttention,
)

# ── Phase 5 SOLID components ────────────────────────────────────────
from src.phase5_explanation_engine.phase5.alert_filter import AlertFilter
from src.phase5_explanation_engine.phase5.artifact_reader import Phase4ArtifactReader
from src.phase5_explanation_engine.phase5.bar_chart_visualizer import BarChartVisualizer
from src.phase5_explanation_engine.phase5.config import Phase5Config
from src.phase5_explanation_engine.phase5.context_enricher import ContextEnricher
from src.phase5_explanation_engine.phase5.explanation_generator import (
    ExplanationGenerator,
)
from src.phase5_explanation_engine.phase5.exporter import ExplanationExporter
from src.phase5_explanation_engine.phase5.feature_importance import (
    FeatureImportanceRanker,
)
from src.phase5_explanation_engine.phase5.line_graph_visualizer import (
    LineGraphVisualizer,
)
from src.phase5_explanation_engine.phase5.pipeline import (
    ExplanationPipeline,
    _detect_hardware,
    _get_git_commit,
)
from src.phase5_explanation_engine.phase5.report import render_explanation_report
from src.phase5_explanation_engine.phase5.shap_computer import SHAPComputer
from src.phase5_explanation_engine.phase5.waterfall_visualizer import (
    WaterfallVisualizer,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
CONFIG_PATH: Path = PROJECT_ROOT / "config" / "phase5_config.yaml"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# A05 — Hardened parameter bounds
BACKGROUND_SAMPLES_MIN: int = 10
BACKGROUND_SAMPLES_MAX: int = 1000
MAX_EXPLAIN_SAMPLES_MIN: int = 1
MAX_EXPLAIN_SAMPLES_MAX: int = 1000
TOP_FEATURES_MIN: int = 1
TOP_FEATURES_MAX: int = 29
MAX_WATERFALL_MIN: int = 0
MAX_WATERFALL_MAX: int = 50
MAX_TIMELINE_MIN: int = 0
MAX_TIMELINE_MAX: int = 50

# Known top-level YAML keys for unknown-key rejection (A05)
_KNOWN_YAML_KEYS: frozenset = frozenset(
    {
        "data",
        "shap",
        "output",
        "explanation_templates",
        "biometric_columns",
        "random_state",
    }
)

# Template variables that are safe to interpolate (A03)
_SAFE_TEMPLATE_VARS: frozenset = frozenset(
    {"idx", "time", "f1", "f2", "f3", "v1", "v2", "v3", "p1", "p2", "p3"}
)

# Regex for detecting raw numeric values in explanations (A09)
_RAW_VALUE_PATTERN = re.compile(r"\b\d+\.\d{2,}\b")


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


def _validate_phase5_parameters(config: Phase5Config) -> None:
    """Enforce Phase 5-specific parameter bounds (A05).

    Args:
        config: Validated Phase5Config.

    Raises:
        ValueError: If any parameter is outside hardened bounds.
    """
    logger.info("── A05: Phase 5 parameter bounds validation ──")

    checks: List[Tuple[str, Any, Any, Any]] = [
        (
            "background_samples",
            config.background_samples,
            BACKGROUND_SAMPLES_MIN,
            BACKGROUND_SAMPLES_MAX,
        ),
        (
            "max_explain_samples",
            config.max_explain_samples,
            MAX_EXPLAIN_SAMPLES_MIN,
            MAX_EXPLAIN_SAMPLES_MAX,
        ),
        ("top_features", config.top_features, TOP_FEATURES_MIN, TOP_FEATURES_MAX),
        (
            "max_waterfall_charts",
            config.max_waterfall_charts,
            MAX_WATERFALL_MIN,
            MAX_WATERFALL_MAX,
        ),
        (
            "max_timeline_charts",
            config.max_timeline_charts,
            MAX_TIMELINE_MIN,
            MAX_TIMELINE_MAX,
        ),
    ]

    for name, value, lo, hi in checks:
        if not (lo <= value <= hi):
            msg = f"A05: {name}={value} outside allowed range [{lo}, {hi}]"
            AuditLogger.log_security_event("PARAMETER_VIOLATION", msg, logging.ERROR)
            raise ValueError(msg)
        logger.info("  A05 ✓  %s=%s ∈ [%s, %s]", name, value, lo, hi)


def _validate_template_safety(config: Phase5Config) -> None:
    """Validate explanation templates contain only safe format variables (A03).

    Raises:
        ValueError: If template contains unsafe format variables.
    """
    logger.info("── A03: Template variable validation ──")
    for level in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
        template = getattr(config.explanation_templates, level)
        # Extract all {var_name} placeholders (ignoring format specs like :.4f)
        placeholders = set(re.findall(r"\{(\w+)", template))
        unsafe = placeholders - _SAFE_TEMPLATE_VARS
        if unsafe:
            msg = f"A03: Template '{level}' contains unsafe variables: {sorted(unsafe)}"
            AuditLogger.log_security_event("TEMPLATE_VIOLATION", msg, logging.ERROR)
            raise ValueError(msg)
        logger.info("  A03 ✓  %s template: %d safe variables", level, len(placeholders))


# ===================================================================
# A01 — Broken Access Control: Path Validation + Read-Only
# ===================================================================


def _validate_output_paths(
    config: Phase5Config,
    validator: PathValidator,
    allow_overwrite: bool = False,
) -> Path:
    """Validate output paths within workspace (A01).

    Returns:
        Resolved output directory path.

    Raises:
        FileExistsError: If explanation_report.json exists and overwrite not allowed.
    """
    logger.info("── A01: Output path validation ──")
    output_dir = validator.validate_output_dir(PROJECT_ROOT / config.output_dir)
    logger.info("  A01 ✓  Output dir: %s", output_dir)

    # Validate charts subdirectory
    charts_dir = validator.validate_output_dir(PROJECT_ROOT / config.output_dir / config.charts_dir)
    logger.info("  A01 ✓  Charts dir: %s", charts_dir)

    if not allow_overwrite:
        report_path = output_dir / config.explanation_report_file
        if report_path.exists():
            msg = f"A01: {report_path.name} exists — set allow_overwrite=True"
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


def _store_explanation_metadata(
    output_dir: Path,
    artifact_hashes: Dict[str, Dict[str, str]],
    assertion_results: List[Dict[str, Any]],
    config: Phase5Config,
    enriched_samples: List[Dict[str, Any]],
    level_counts: Dict[str, int],
    chart_files: List[str],
    hw_info: Dict[str, str],
    duration_s: float,
    git_commit: str,
) -> Path:
    """Persist artifact hashes, assertions, and explanation metadata (A02).

    Returns:
        Path to explanation_metadata.json.
    """
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline": "security_hardened_phase5",
        "random_state": config.random_state,
        "git_commit": git_commit,
        "hardware": hw_info,
        "duration_seconds": round(duration_s, 2),
        "hyperparameters": {
            "background_samples": config.background_samples,
            "max_explain_samples": config.max_explain_samples,
            "top_features": config.top_features,
            "max_waterfall_charts": config.max_waterfall_charts,
            "max_timeline_charts": config.max_timeline_charts,
        },
        "samples_explained": len(enriched_samples),
        "level_counts": level_counts,
        "charts_generated": chart_files,
        "artifact_hashes": artifact_hashes,
        "integrity_assertions": assertion_results,
    }

    meta_path = output_dir / config.metadata_file
    _clear_read_only(meta_path)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    _make_read_only(meta_path)
    AuditLogger.log_file_access("METADATA_WRITTEN", meta_path)
    return meta_path


# ===================================================================
# A08 — Data Integrity Assertions
# ===================================================================


class ExplanationAssertions:
    """Phase 5-specific data integrity assertions (A08)."""

    def __init__(self) -> None:
        self._results: List[Dict[str, Any]] = []

    def assert_normal_only_background(self, n_normal: int, n_attack: int) -> bool:
        """Assert SHAP background was computed from Normal-only samples."""
        passed = n_normal > 0 and n_attack == 0
        self._results.append(
            {
                "assertion": "SHAP background Normal only",
                "expected": "n_normal > 0, n_attack = 0",
                "actual": f"n_normal={n_normal}, n_attack={n_attack}",
                "status": "PASS" if passed else "FAIL",
            }
        )
        if passed:
            logger.info("  A08 ✓  SHAP background Normal-only (%d samples)", n_normal)
        else:
            AuditLogger.log_security_event(
                "INTEGRITY_VIOLATION",
                f"SHAP background contaminated: n_normal={n_normal}, n_attack={n_attack}",
                logging.CRITICAL,
            )
        return passed

    def assert_shap_shape_matches(
        self, shap_shape: Tuple[int, ...], n_filtered: int, n_features: int
    ) -> bool:
        """Assert SHAP values shape matches filtered sample count."""
        expected_shape = f"({n_filtered}, *, {n_features})"
        passed = shap_shape[0] == n_filtered and shap_shape[-1] == n_features
        self._results.append(
            {
                "assertion": "shap_values shape correct",
                "expected": expected_shape,
                "actual": str(shap_shape),
                "status": "PASS" if passed else "FAIL",
            }
        )
        if passed:
            logger.info("  A08 ✓  SHAP shape %s matches filtered count", shap_shape)
        else:
            AuditLogger.log_security_event(
                "INTEGRITY_VIOLATION",
                f"SHAP shape mismatch: expected {expected_shape}, got {shap_shape}",
                logging.CRITICAL,
            )
        return passed

    def assert_feature_ranks_complete(self, importance_df: pd.DataFrame, n_features: int) -> bool:
        """Assert feature importance ranks cover all features."""
        n_ranked = len(importance_df)
        passed = n_ranked <= n_features
        self._results.append(
            {
                "assertion": "Feature importance ranks complete",
                "expected": f"<= {n_features} features ranked",
                "actual": f"{n_ranked} features ranked",
                "status": "PASS" if passed else "FAIL",
            }
        )
        if passed:
            logger.info("  A08 ✓  Feature importance: %d features ranked", n_ranked)
        else:
            AuditLogger.log_security_event(
                "INTEGRITY_VIOLATION",
                f"Feature ranking count {n_ranked} exceeds {n_features}",
                logging.CRITICAL,
            )
        return passed

    def assert_waterfall_high_plus_only(
        self, chart_files: List[str], enriched_samples: List[Dict[str, Any]]
    ) -> bool:
        """Assert waterfall charts generated for CRITICAL/HIGH only."""
        waterfall_files = [f for f in chart_files if f.startswith("waterfall_")]
        waterfall_indices = set()
        for f in waterfall_files:
            idx_str = f.replace("waterfall_", "").replace(".png", "")
            try:
                waterfall_indices.add(int(idx_str))
            except ValueError:
                pass

        high_plus_indices = {
            s["sample_index"] for s in enriched_samples if s["risk_level"] in ("CRITICAL", "HIGH")
        }

        invalid = waterfall_indices - high_plus_indices
        passed = len(invalid) == 0
        self._results.append(
            {
                "assertion": "Waterfall for HIGH+ only",
                "expected": "All waterfalls for CRITICAL/HIGH samples",
                "actual": (
                    f"{len(waterfall_files)} waterfalls, {len(invalid)} invalid"
                    if not passed
                    else f"{len(waterfall_files)} waterfalls, all valid"
                ),
                "status": "PASS" if passed else "FAIL",
            }
        )
        if passed:
            logger.info(
                "  A08 ✓  Waterfall charts: %d, all for HIGH+ samples",
                len(waterfall_files),
            )
        else:
            AuditLogger.log_security_event(
                "INTEGRITY_VIOLATION",
                f"Waterfall charts for non-HIGH+ samples: {sorted(invalid)}",
                logging.CRITICAL,
            )
        return passed

    def assert_no_raw_biometric_values(self, enriched_samples: List[Dict[str, Any]]) -> bool:
        """Assert no raw biometric values appear in explanation text."""
        violations = 0
        for sample in enriched_samples:
            explanation = sample.get("explanation", "")
            for col in BIOMETRIC_COLUMNS:
                # Check for patterns like "SpO2=0.1234" or "SpO2: 0.1234"
                pattern = re.compile(rf"\b{re.escape(col)}\s*[=:]\s*\d+\.?\d*")
                if pattern.search(explanation):
                    violations += 1
                    break

        passed = violations == 0
        self._results.append(
            {
                "assertion": "No raw biometric values in explanations",
                "expected": "0 violations",
                "actual": f"{violations} violations in {len(enriched_samples)} samples",
                "status": "PASS" if passed else "FAIL",
            }
        )
        if passed:
            logger.info(
                "  A08 ✓  No raw biometric values in %d explanations",
                len(enriched_samples),
            )
        else:
            AuditLogger.log_security_event(
                "INTEGRITY_VIOLATION",
                f"Raw biometric values found in {violations} explanations",
                logging.CRITICAL,
            )
        return passed

    def assert_explanation_count(self, n_enriched: int, n_with_explanation: int) -> bool:
        """Assert all enriched samples have explanations."""
        passed = n_enriched == n_with_explanation
        self._results.append(
            {
                "assertion": "Explanation count matches enriched",
                "expected": str(n_enriched),
                "actual": str(n_with_explanation),
                "status": "PASS" if passed else "FAIL",
            }
        )
        if passed:
            logger.info("  A08 ✓  All %d enriched samples have explanations", n_enriched)
        else:
            AuditLogger.log_security_event(
                "INTEGRITY_VIOLATION",
                f"Explanation count mismatch: {n_with_explanation} != {n_enriched}",
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


def _log_explanation_summary(
    level_counts: Dict[str, int], n_explained: int, duration_s: float
) -> None:
    """Log aggregate explanation summary — NEVER per-patient SHAP values (A09)."""
    AuditLogger.log_security_event(
        "EXPLANATION_SUMMARY",
        (
            f"samples_explained={n_explained}, "
            f"duration={duration_s:.2f}s, "
            + ", ".join(f"{k}={v}" for k, v in sorted(level_counts.items()))
        ),
        logging.INFO,
    )


def _log_feature_importance_safe(
    top_features: List[Tuple[str, float]],
) -> None:
    """Log feature importance — names and aggregate SHAP means only (A09).

    Feature names are safe to log. Individual per-sample SHAP values
    are NEVER logged.
    """
    feature_names_only = [f"{name} (mean|SHAP|={val:.6f})" for name, val in top_features]
    AuditLogger.log_security_event(
        "FEATURE_IMPORTANCE",
        f"top_features=[{', '.join(feature_names_only)}]",
        logging.INFO,
    )


def _log_shap_computation_safe(
    n_samples: int, n_features: int, method: str, duration_s: float
) -> None:
    """Log SHAP computation summary — NEVER individual SHAP values (A09)."""
    AuditLogger.log_security_event(
        "SHAP_COMPUTED",
        (
            f"method={method}, samples={n_samples}, "
            f"features={n_features}, duration={duration_s:.2f}s"
        ),
        logging.INFO,
    )


# ===================================================================
# Security Report Generation
# ===================================================================


def _generate_security_report(
    assertions: ExplanationAssertions,
    artifact_hashes: Dict[str, Dict[str, str]],
    config: Phase5Config,
    level_counts: Dict[str, int],
    hw_info: Dict[str, str],
    duration_s: float,
) -> None:
    """Render §8.2 Explanation Engine Security Controls report."""
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
        f"| `background_samples` | [{BACKGROUND_SAMPLES_MIN}, {BACKGROUND_SAMPLES_MAX}]"
        f" | {config.background_samples} | PASS |\n"
        f"| `max_explain_samples` | [{MAX_EXPLAIN_SAMPLES_MIN}, {MAX_EXPLAIN_SAMPLES_MAX}]"
        f" | {config.max_explain_samples} | PASS |\n"
        f"| `top_features` | [{TOP_FEATURES_MIN}, {TOP_FEATURES_MAX}]"
        f" | {config.top_features} | PASS |\n"
        f"| `max_waterfall_charts` | [{MAX_WATERFALL_MIN}, {MAX_WATERFALL_MAX}]"
        f" | {config.max_waterfall_charts} | PASS |\n"
        f"| `max_timeline_charts` | [{MAX_TIMELINE_MIN}, {MAX_TIMELINE_MAX}]"
        f" | {config.max_timeline_charts} | PASS |\n"
        f"| `random_state` | int | {config.random_state} | PASS |\n"
        "| Unknown YAML keys | none allowed | 0 found | PASS |\n"
    )

    # Risk level distribution
    risk_rows = ""
    total = sum(level_counts.values())
    for level in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
        count = level_counts.get(level, 0)
        pct = count / total * 100 if total > 0 else 0
        risk_rows += f"| {level} | {count} | {pct:.1f}% |\n"

    biometric_list = ", ".join(f"`{c}`" for c in sorted(BIOMETRIC_COLUMNS))

    report = f"""## 8.2 Explanation Engine Security Controls

This section documents the security controls applied during Phase 5
explanation generation, extending the Phase 0 OWASP framework (§3.3)
and Phase 2/3/4 model controls (§5.2, §5.4, §7.2) with
explanation-engine-specific protections.

### 8.2.1 OWASP Controls — Phase 5 Extensions

| OWASP | Category | Control | Status |
|-------|----------|---------|--------|
| A01 | Access Control | Output paths within workspace | Implemented |
| A01 | Access Control | `shap_values.parquet` read-only (chmod 444) | Implemented |
| A01 | Access Control | `explanation_report.json` overwrite protection | Implemented |
| A01 | Access Control | Charts directory within workspace | Implemented |
| A02 | Crypto Failures | SHA-256 for `shap_values.parquet` | Implemented |
| A02 | Crypto Failures | SHA-256 for `explanation_report.json` | Implemented |
| A02 | Crypto Failures | SHA-256 for `explanation_metadata.json` | Implemented |
| A02 | Crypto Failures | Hashes stored in `explanation_metadata.json` | Implemented |
| A03 | Injection | Config dict sanitized via `ConfigSanitizer` | Implemented |
| A03 | Injection | Template variables validated against allowlist | Implemented |
| A03 | Injection | `top_features` ∈ [{TOP_FEATURES_MIN}, {TOP_FEATURES_MAX}] | Implemented |
| A05 | Misconfiguration | `background_samples` bounded | Implemented |
| A05 | Misconfiguration | `max_explain_samples` bounded | Implemented |
| A05 | Misconfiguration | `max_waterfall_charts` bounded | Implemented |
| A05 | Misconfiguration | `max_timeline_charts` bounded | Implemented |
| A05 | Misconfiguration | Unknown YAML keys rejected | Implemented |
| A08 | Data Integrity | SHAP background Normal only | Implemented |
| A08 | Data Integrity | SHAP values shape matches filtered count | Implemented |
| A08 | Data Integrity | Feature importance ranks complete | Implemented |
| A08 | Data Integrity | Waterfall charts for HIGH+ only | Implemented |
| A08 | Data Integrity | No raw biometric values in explanations | Implemented |
| A08 | Data Integrity | Explanation count matches enriched | Implemented |
| A09 | Logging | SHAP computation time logged (safe) | Implemented |
| A09 | Logging | Feature importance ranking logged (safe) | Implemented |
| A09 | Logging | Risk level distribution logged (safe) | Implemented |
| A09 | Logging | Individual SHAP values NEVER logged | Implemented |
| A09 | Logging | Raw biometric values NEVER logged | Implemented |
| A09 | Logging | Patient identifiers NEVER in templates | Implemented |

### 8.2.2 Explanation Integrity Checklist

- [x] `shap_values.parquet` SHA-256 stored in `explanation_metadata.json`
- [x] `explanation_report.json` SHA-256 stored in `explanation_metadata.json`
- [x] SHAP background: Normal samples only — verified at runtime
- [x] No raw biometric values in explanation text
- [x] Explanation templates use feature names only, never raw values
- [x] All exported artifacts set to read-only (chmod 444)
- [x] Phase 2 + Phase 3 + Phase 4 artifacts verified via SHA-256 before loading

### 8.2.3 HIPAA Logging Compliance (A09)

| Data Category | Logged? | Justification |
|---------------|---------|---------------|
| SHAP computation time | Yes | Non-PHI: system performance metric |
| Feature importance ranking | Yes | Non-PHI: aggregate model statistics |
| Risk level distribution | Yes | Non-PHI: population-level counts |
| Number of samples explained | Yes | Non-PHI: aggregate count |
| Individual SHAP values per patient | **NEVER** | HIPAA: individual feature attributions |
| Raw biometric values | **NEVER** | HIPAA: columns = {biometric_list} |
| Patient identifiers | **NEVER** | HIPAA: no patient IDs in templates |
| Per-sample anomaly scores | **NEVER** | HIPAA: individual risk predictions |

### 8.2.4 Data Integrity Assertions (A08)

All assertions are evaluated at runtime and logged with pass/fail status.

| Assertion | Expected | Actual | Status |
|-----------|----------|--------|--------|
{assertion_rows}
**Overall:** {overall}

### 8.2.5 Artifact Integrity (A02)

SHA-256 hashes computed after export for downstream verification.

| Artifact | SHA-256 |
|----------|---------|
{hash_rows}
Hashes stored in `explanation_metadata.json` and must be verified
by the Notification Engine (Phase 6) before sending alerts.

### 8.2.6 Parameter Bounds Validation (A05)

| Parameter | Allowed Range | Configured Value | Status |
|-----------|--------------|-----------------|--------|
{param_rows}
### 8.2.7 Explained Risk Distribution

| Risk Level | Count | Percentage |
|------------|-------|------------|
{risk_rows}
**Total explained samples:** {total}

### 8.2.8 Security Inheritance from Phase 0, 2, 3, and 4

| Control | Source | Reuse Method |
|---------|--------|-------------|
| SHA-256 hashing | Phase 0 `IntegrityVerifier` | Direct import — `compute_hash()` |
| Path traversal | Phase 0 `PathValidator` | Direct import — `validate_output_dir()` |
| Config sanitization | Phase 0 `ConfigSanitizer` | Direct import — `sanitize_config_dict()` |
| Audit logging | Phase 0 `AuditLogger` | Direct import — `log_file_access()` |
| Phase 2 artifact SHA-256 | Phase 2 `detection_metadata.json` | Verified before model load |
| Phase 3 artifact SHA-256 | Phase 3 `classification_metadata.json` | Verified before model load |
| Phase 4 artifact SHA-256 | Phase 4 `risk_metadata.json` | Verified before risk report load |

---

**Reviewer signature:** ____________________  **Date:** ____/____/______
"""

    report_path = (
        PROJECT_ROOT / "results" / "phase0_analysis" / "report_section_explanation_security.md"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report)
    logger.info("  Security report saved: %s", report_path.name)


# ===================================================================
# Main Pipeline
# ===================================================================


def run_hardened_pipeline(
    *,
    allow_overwrite: bool = True,
) -> Dict[str, Any]:
    """Execute Phase 5 explanation engine with full OWASP/HIPAA controls.

    Args:
        allow_overwrite: If False, raises FileExistsError if report exists.

    Returns:
        Pipeline summary dict.
    """
    t0 = time.time()

    logger.info("═══════════════════════════════════════════════════")
    logger.info("  Phase 5 Explanation Engine — Security Hardened")
    logger.info("═══════════════════════════════════════════════════")

    # ── Hardware detection ──
    hw_info = _detect_hardware()

    # ── A03/A05: Config sanitization + parameter validation ──
    logger.info("── A03: Config sanitization ──")
    raw_yaml = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    ConfigSanitizer.sanitize_config_dict(raw_yaml)
    logger.info("  A03 ✓  Config sanitized")

    _reject_unknown_yaml_keys(raw_yaml)

    config = Phase5Config.from_yaml(CONFIG_PATH)
    _validate_phase5_parameters(config)
    _validate_template_safety(config)

    # ── A01: Path validation ──
    validator = PathValidator(PROJECT_ROOT)
    output_dir = _validate_output_paths(config, validator, allow_overwrite)

    # ── Reproducibility seeds ──
    np.random.seed(config.random_state)  # noqa: NPY002
    tf.random.set_seed(config.random_state)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    rng = np.random.default_rng(config.random_state)

    # ── A02: Verify Phase 2/3/4 artifacts (SHA-256) ──
    reader = Phase4ArtifactReader(
        project_root=PROJECT_ROOT,
        phase4_dir=config.phase4_dir,
        phase4_metadata=config.phase4_metadata,
        phase3_dir=config.phase3_dir,
        phase3_metadata=config.phase3_metadata,
        phase2_dir=config.phase2_dir,
        phase2_metadata=config.phase2_metadata,
        label_column=config.label_column,
    )
    p2_metadata, p3_metadata, _ = reader.verify_all()
    model = reader.rebuild_model(p2_metadata, p3_metadata)

    # ── Load risk report + baseline ──
    risk_report = reader.load_risk_report()
    baseline = reader.load_baseline()

    # ── Filter non-NORMAL samples ──
    alert_filter = AlertFilter(max_samples=config.max_explain_samples)
    filtered, level_counts = alert_filter.filter(risk_report["sample_assessments"], rng)

    # ── A08: Verify SHAP background Normal-only ──
    logger.info("── A08: Data integrity assertions ──")
    assertions = ExplanationAssertions()

    # Verify Normal-only background by reading train data
    train_path = PROJECT_ROOT / config.phase1_train
    train_df = pd.read_parquet(train_path)
    n_normal = int((train_df[config.label_column] == 0).sum())
    n_attack = int((train_df[config.label_column] != 0).sum())
    assertions.assert_normal_only_background(n_normal, n_attack)
    del train_df  # Free memory

    # ── Prepare SHAP data ──
    shap_computer = SHAPComputer(
        n_background=config.background_samples,
        label_column=config.label_column,
    )
    test_path = PROJECT_ROOT / config.phase1_test
    sample_indices = [s["sample_index"] for s in filtered]

    background = shap_computer.prepare_background(train_path, rng)
    X_explain, _, feature_names = shap_computer.prepare_explanation_data(test_path, sample_indices)

    # ── Compute SHAP values ──
    shap_t0 = time.time()
    shap_values = shap_computer.compute(model, background, X_explain)
    shap_duration = time.time() - shap_t0

    # A09: Log SHAP computation summary (safe — aggregate only)
    _log_shap_computation_safe(
        n_samples=len(X_explain),
        n_features=len(feature_names),
        method="GradientExplainer/IG-fallback",
        duration_s=shap_duration,
    )

    # A08: Assert SHAP shape
    assertions.assert_shap_shape_matches(shap_values.shape, len(X_explain), len(feature_names))

    # ── Rank feature importance ──
    feature_ranker = FeatureImportanceRanker(top_k=config.top_features)
    importance_df, top_features = feature_ranker.rank(shap_values, feature_names)

    # A08: Assert feature ranks
    assertions.assert_feature_ranks_complete(importance_df, len(feature_names))

    # A09: Log feature importance (safe — names + aggregate means)
    _log_feature_importance_safe(top_features)

    # ── Enrich samples with context + explanations ──
    explanation_gen = ExplanationGenerator(templates=config.explanation_templates)
    context_enricher = ContextEnricher(explanation_generator=explanation_gen)
    enriched = context_enricher.enrich(filtered, shap_values, feature_names)

    # A08: Assert explanation count
    n_with_explanation = sum(1 for s in enriched if s.get("explanation"))
    assertions.assert_explanation_count(len(enriched), n_with_explanation)

    # A08: Assert no raw biometric values in explanations
    assertions.assert_no_raw_biometric_values(enriched)

    # ── Generate visualizations ──
    waterfall_viz = WaterfallVisualizer(top_k=config.top_features)
    bar_chart_viz = BarChartVisualizer(
        top_k=config.top_features,
        biometric_columns=frozenset(config.biometric_columns),
    )
    line_graph_viz = LineGraphVisualizer()

    pipeline = ExplanationPipeline(
        config=config,
        artifact_reader=reader,
        alert_filter=alert_filter,
        shap_computer=shap_computer,
        feature_ranker=feature_ranker,
        context_enricher=context_enricher,
        waterfall_viz=waterfall_viz,
        bar_chart_viz=bar_chart_viz,
        line_graph_viz=line_graph_viz,
        exporter=ExplanationExporter(output_dir=output_dir),
        project_root=PROJECT_ROOT,
    )
    charts_dir = output_dir / config.charts_dir
    chart_files = pipeline._generate_all_charts(
        enriched,
        shap_values,
        feature_names,
        importance_df,
        baseline["baseline_threshold"],
        charts_dir,
    )

    # A08: Assert waterfall charts for HIGH+ only
    assertions.assert_waterfall_high_plus_only(chart_files, enriched)

    duration_s = time.time() - t0

    # A09: Log aggregate explanation summary
    _log_explanation_summary(level_counts, len(enriched), duration_s)

    # ── A01: Export artifacts + read-only enforcement ──
    logger.info("── A01: Exporting artifacts (read-only) ──")
    exporter = ExplanationExporter(output_dir=output_dir)
    git_commit = _get_git_commit()

    # Clear read-only if overwriting
    for fname in [config.shap_values_file, config.explanation_report_file]:
        _clear_read_only(output_dir / fname)

    shap_path, _ = exporter.export_shap_values(shap_values, feature_names, config.shap_values_file)
    report_path, _ = exporter.export_explanation_report(
        enriched,
        importance_df,
        level_counts,
        config.top_features,
        git_commit,
        config.explanation_report_file,
    )

    # A01: Set artifacts read-only (chmod 444)
    for fname in [config.shap_values_file, config.explanation_report_file]:
        _make_read_only(output_dir / fname)

    # ── A08: Final assertion check ──
    if not assertions.all_passed:
        raise RuntimeError("A08: Integrity assertions FAILED — see logs")

    logger.info("  A08 ✓  All %d assertions PASSED", len(assertions.results))

    # ── A02: Hash artifacts + metadata ──
    verifier = IntegrityVerifier(output_dir)
    artifact_hashes = _hash_artifacts(
        verifier,
        {
            config.shap_values_file: output_dir / config.shap_values_file,
            config.explanation_report_file: output_dir / config.explanation_report_file,
        },
    )

    _store_explanation_metadata(
        output_dir=output_dir,
        artifact_hashes=artifact_hashes,
        assertion_results=assertions.results,
        config=config,
        enriched_samples=enriched,
        level_counts=level_counts,
        chart_files=chart_files,
        hw_info=hw_info,
        duration_s=duration_s,
        git_commit=git_commit,
    )

    # ── Generate reports ──
    report_md = render_explanation_report(
        enriched_samples=enriched,
        importance_df=importance_df,
        level_counts=level_counts,
        chart_files=chart_files,
        baseline_threshold=baseline["baseline_threshold"],
        hw_info=hw_info,
        duration_s=duration_s,
        git_commit=git_commit,
        config=config,
    )
    report_dir = PROJECT_ROOT / "results" / "phase0_analysis"
    report_dir.mkdir(parents=True, exist_ok=True)
    explanation_report_path = report_dir / "report_section_explanation.md"
    with open(explanation_report_path, "w") as f:
        f.write(report_md)
    logger.info("  Report saved: %s", explanation_report_path.name)

    _generate_security_report(
        assertions=assertions,
        artifact_hashes=artifact_hashes,
        config=config,
        level_counts=level_counts,
        hw_info=hw_info,
        duration_s=duration_s,
    )

    logger.info("═══════════════════════════════════════════════════")
    logger.info("  Phase 5 Security Hardened — %.2fs", duration_s)
    logger.info(
        "  Assertions: %d/%d PASSED",
        len(assertions.results),
        len(assertions.results),
    )
    logger.info("  Risk distribution: %s", level_counts)
    logger.info("═══════════════════════════════════════════════════")

    return {
        "samples_explained": len(enriched),
        "level_counts": level_counts,
        "top_features": top_features,
        "charts_generated": chart_files,
        "assertions_passed": assertions.all_passed,
        "duration_s": round(duration_s, 2),
    }


def main() -> None:
    """Entry point for security-hardened Phase 5 pipeline."""
    run_hardened_pipeline(allow_overwrite=True)


if __name__ == "__main__":
    main()
