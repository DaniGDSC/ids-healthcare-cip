#!/usr/bin/env python3
"""Phase 4 — Risk-Adaptive Engine.

Loads Phase 3 classification model and Phase 2 attention output, then
computes risk scores with dynamic thresholding and concept drift detection.

Pipeline steps:
    1. Load and verify Phase 2/3 artifacts (SHA-256)
    2. Compute baseline (Median + MAD) from Normal-only attention scores
    3. Dynamic thresholding with rolling windows and time-of-day k(t)
    4. Concept drift detection (drift_ratio > 0.20 triggers fallback)
    5. Risk scoring (NORMAL / LOW / MEDIUM / HIGH / CRITICAL)
    6. Export artifacts (baseline_config.json, threshold_config.json,
       risk_report.json, drift_log.csv)
    7. Generate report_section_risk_adaptive.md

Usage::

    python -m src.phase4_risk_engine.phase4_risk_adaptive
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

# Phase 2 builders (for model architecture rebuild)
from src.phase2_detection_engine.phase2.assembler import DetectionModelAssembler
from src.phase2_detection_engine.phase2.attention_builder import (  # noqa: F401
    AttentionBuilder,
    BahdanauAttention,
)
from src.phase2_detection_engine.phase2.bilstm_builder import BiLSTMBuilder
from src.phase2_detection_engine.phase2.cnn_builder import CNNBuilder
from src.phase2_detection_engine.phase2.reshaper import DataReshaper

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
CONFIG_PATH: Path = PROJECT_ROOT / "config" / "phase4_config.yaml"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Risk level constants
RISK_NORMAL: str = "NORMAL"
RISK_LOW: str = "LOW"
RISK_MEDIUM: str = "MEDIUM"
RISK_HIGH: str = "HIGH"
RISK_CRITICAL: str = "CRITICAL"

# Named constants for clarity
N_FEATURES: int = 29
HASH_CHUNK_SIZE: int = 65_536


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file.

    Args:
        filepath: Path to the file.

    Returns:
        Hex digest string.
    """
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(HASH_CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def _get_git_commit() -> str:
    """Get current git commit hash.

    Returns:
        40-character hex string or 'unknown'.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(PROJECT_ROOT),
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


def _detect_hardware() -> Dict[str, str]:
    """Detect GPU/CPU and return hardware info dict.

    Returns:
        Dict with keys: device, cuda, tensorflow, python, platform.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        device_name = gpus[0].name
        logger.info("  GPU detected: %s", device_name)
        info = {"device": f"GPU: {device_name}", "cuda": "available"}
    else:
        cpu_info = platform.processor() or platform.machine()
        logger.info("  CPU fallback: %s", cpu_info)
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        info = {"device": f"CPU: {cpu_info}", "cuda": "N/A (CPU execution)"}

    info["tensorflow"] = tf.__version__
    info["python"] = platform.python_version()
    info["platform"] = platform.platform()
    return info


# ---------------------------------------------------------------------------
# Step 1: Load and verify artifacts
# ---------------------------------------------------------------------------


def _verify_phase3_artifacts(config: Dict[str, Any]) -> Dict[str, Any]:
    """Load and SHA-256 verify Phase 3 classification artifacts.

    Args:
        config: Parsed YAML config dict.

    Returns:
        Phase 3 classification_metadata dict.

    Raises:
        FileNotFoundError: If metadata or artifact files are missing.
        ValueError: If SHA-256 mismatch is detected.
    """
    logger.info("── Step 1a: Verify Phase 3 artifacts (SHA-256) ──")
    meta_path = PROJECT_ROOT / config["data"]["phase3_metadata"]
    if not meta_path.exists():
        raise FileNotFoundError(f"Phase 3 metadata not found: {meta_path}")

    metadata = json.loads(meta_path.read_text())
    phase3_dir = PROJECT_ROOT / config["data"]["phase3_dir"]

    for artifact_name, hash_info in metadata["artifact_hashes"].items():
        artifact_path = phase3_dir / artifact_name
        if not artifact_path.exists():
            raise FileNotFoundError(f"Phase 3 artifact missing: {artifact_path}")

        expected = hash_info["sha256"]
        actual = _compute_sha256(artifact_path)
        if actual != expected:
            raise ValueError(
                f"SHA-256 mismatch for {artifact_name}: "
                f"expected={expected[:16]}…, actual={actual[:16]}…"
            )
        logger.info("  SHA-256 ✓  %s", artifact_name)

    return metadata


def _verify_phase2_artifacts(config: Dict[str, Any]) -> Dict[str, Any]:
    """Load and SHA-256 verify Phase 2 detection artifacts.

    Args:
        config: Parsed YAML config dict.

    Returns:
        Phase 2 detection_metadata dict.

    Raises:
        FileNotFoundError: If metadata or artifact files are missing.
        ValueError: If SHA-256 mismatch is detected.
    """
    logger.info("── Step 1b: Verify Phase 2 artifacts (SHA-256) ──")
    meta_path = PROJECT_ROOT / config["data"]["phase2_metadata"]
    if not meta_path.exists():
        raise FileNotFoundError(f"Phase 2 metadata not found: {meta_path}")

    metadata = json.loads(meta_path.read_text())
    phase2_dir = PROJECT_ROOT / config["data"]["phase2_dir"]

    for artifact_name, hash_info in metadata["artifact_hashes"].items():
        artifact_path = phase2_dir / artifact_name
        if not artifact_path.exists():
            raise FileNotFoundError(f"Phase 2 artifact missing: {artifact_path}")

        expected = hash_info["sha256"]
        actual = _compute_sha256(artifact_path)
        if actual != expected:
            raise ValueError(
                f"SHA-256 mismatch for {artifact_name}: "
                f"expected={expected[:16]}…, actual={actual[:16]}…"
            )
        logger.info("  SHA-256 ✓  %s", artifact_name)

    return metadata


def _load_attention_output(config: Dict[str, Any]) -> pd.DataFrame:
    """Load Phase 2 attention output parquet.

    Args:
        config: Parsed YAML config dict.

    Returns:
        DataFrame with attn_0..attn_127, Label, split columns.
    """
    attn_path = PROJECT_ROOT / config["data"]["phase2_dir"] / "attention_output.parquet"
    attn_df = pd.read_parquet(attn_path)
    logger.info("  Loaded attention output: %s", attn_df.shape)
    return attn_df


def _load_phase1_data(
    config: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load Phase 1 preprocessed train/test parquets.

    Args:
        config: Parsed YAML config dict.

    Returns:
        Tuple of (X_train, y_train, X_test, y_test, feature_names).
    """
    label_col = config["data"]["label_column"]
    train_path = PROJECT_ROOT / config["data"]["phase1_train"]
    test_path = PROJECT_ROOT / config["data"]["phase1_test"]

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    feature_names = [c for c in train_df.columns if c != label_col]
    X_train = train_df[feature_names].values.astype(np.float32)
    y_train = train_df[label_col].values.astype(np.int32)
    X_test = test_df[feature_names].values.astype(np.float32)
    y_test = test_df[label_col].values.astype(np.int32)

    logger.info(
        "  Phase 1 data: train=%s, test=%s, features=%d",
        X_train.shape,
        X_test.shape,
        len(feature_names),
    )
    return X_train, y_train, X_test, y_test, feature_names


def _rebuild_classification_model(
    p2_metadata: Dict[str, Any],
    p3_metadata: Dict[str, Any],
    config: Dict[str, Any],
) -> tf.keras.Model:
    """Rebuild full classification model architecture and load weights.

    Args:
        p2_metadata: Phase 2 detection metadata with hyperparameters.
        p3_metadata: Phase 3 classification metadata with head config.
        config: Parsed YAML config dict.

    Returns:
        Fully loaded classification model with sigmoid output.
    """
    logger.info("── Rebuilding classification model ──")
    hp = p2_metadata["hyperparameters"]

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

    assembler = DetectionModelAssembler(
        timesteps=hp["timesteps"],
        n_features=N_FEATURES,
        builders=builders,
    )
    detection_model = assembler.assemble()

    # Attach classification head (same as Phase 3)
    p3_hp = p3_metadata["hyperparameters"]
    x = tf.keras.layers.Dense(
        p3_hp["dense_units"],
        activation=p3_hp["dense_activation"],
        name="dense_head",
    )(detection_model.output)
    x = tf.keras.layers.Dropout(p3_hp["head_dropout_rate"], name="drop_head")(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)
    full_model = tf.keras.Model(detection_model.input, output, name="classification_engine")

    # Load Phase 3 weights
    weights_path = PROJECT_ROOT / config["data"]["phase3_dir"] / "classification_model.weights.h5"
    full_model.load_weights(str(weights_path))
    logger.info(
        "  Model loaded: %d params, %d layers",
        full_model.count_params(),
        len(full_model.layers),
    )
    return full_model


# ---------------------------------------------------------------------------
# Step 2: Baseline computation
# ---------------------------------------------------------------------------


def _compute_baseline(attn_df: pd.DataFrame, mad_multiplier: float) -> Dict[str, Any]:
    """Compute baseline threshold from Normal-only attention scores.

    Uses Median + k*MAD where MAD = Median(|Xi - Median(X)|).

    Args:
        attn_df: Attention output DataFrame with Label and split columns.
        mad_multiplier: k multiplier for MAD (default 3.0).

    Returns:
        Baseline config dict with median, mad, threshold, n_normal.
    """
    logger.info("── Step 2: Baseline computation ──")

    # Filter Normal samples from training split only
    train_normal = attn_df[(attn_df["Label"] == 0) & (attn_df["split"] == "train")]
    attn_cols = [c for c in attn_df.columns if c.startswith("attn_")]

    logger.info("  Loaded %d Normal samples for baseline computation", len(train_normal))

    # Compute per-sample attention magnitude (L2 norm across 128 dims)
    attn_values = train_normal[attn_cols].values.astype(np.float64)
    sample_magnitudes = np.linalg.norm(attn_values, axis=1)

    median = float(np.median(sample_magnitudes))
    mad = float(np.median(np.abs(sample_magnitudes - median)))
    baseline_threshold = median + mad_multiplier * mad

    logger.info("  Baseline threshold: %.6f, MAD: %.6f", baseline_threshold, mad)
    logger.info("  Median: %.6f, k: %.1f", median, mad_multiplier)

    baseline = {
        "median": median,
        "mad": mad,
        "baseline_threshold": baseline_threshold,
        "mad_multiplier": mad_multiplier,
        "n_normal_samples": len(train_normal),
        "n_attention_dims": len(attn_cols),
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }
    return baseline


# ---------------------------------------------------------------------------
# Step 3: Dynamic thresholding
# ---------------------------------------------------------------------------


def _get_k_for_hour(hour: int, k_schedule: List[Dict[str, Any]]) -> float:
    """Get sensitivity multiplier k(t) for a given hour.

    Args:
        hour: Hour of day (0-23).
        k_schedule: List of {start_hour, end_hour, k} dicts.

    Returns:
        Appropriate k value for the hour.
    """
    for entry in k_schedule:
        if entry["start_hour"] <= hour < entry["end_hour"]:
            return entry["k"]
    return 3.0  # Default fallback


def _compute_dynamic_thresholds(
    anomaly_scores: np.ndarray,
    baseline: Dict[str, Any],
    window_size: int,
    k_schedule: List[Dict[str, Any]],
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Compute dynamic thresholds using rolling Median + k(t)*MAD windows.

    Args:
        anomaly_scores: Model sigmoid outputs, shape (N,).
        baseline: Baseline config with median, mad, baseline_threshold.
        window_size: Rolling window size.
        k_schedule: Time-of-day k(t) schedule.

    Returns:
        Tuple of (dynamic_thresholds array, window_log list).
    """
    logger.info("── Step 3: Dynamic thresholding ──")
    n_samples = len(anomaly_scores)
    dynamic_thresholds = np.full(n_samples, baseline["baseline_threshold"])
    window_log: List[Dict[str, Any]] = []

    # Simulate time progression (spread across 24 hours)
    hours = np.linspace(0, 24, n_samples, endpoint=False)

    for i in range(window_size, n_samples):
        window = anomaly_scores[i - window_size : i]
        w_median = float(np.median(window))
        w_mad = float(np.median(np.abs(window - w_median)))

        hour = int(hours[i]) % 24
        k_t = _get_k_for_hour(hour, k_schedule)
        dyn_threshold = w_median + k_t * max(w_mad, 1e-8)
        dynamic_thresholds[i] = dyn_threshold

        # Log every Nth window for monitoring
        if i % window_size == 0:
            window_log.append(
                {
                    "sample_index": int(i),
                    "hour": hour,
                    "k_t": k_t,
                    "window_median": round(w_median, 6),
                    "window_mad": round(w_mad, 6),
                    "dynamic_threshold": round(dyn_threshold, 6),
                }
            )

    logger.info(
        "  Dynamic thresholds computed: %d samples, window_size=%d",
        n_samples,
        window_size,
    )
    return dynamic_thresholds, window_log


# ---------------------------------------------------------------------------
# Step 4: Concept drift detection
# ---------------------------------------------------------------------------


def _detect_concept_drift(
    dynamic_thresholds: np.ndarray,
    baseline_threshold: float,
    drift_threshold: float,
    recovery_threshold: float,
    recovery_windows: int,
    window_size: int,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Detect concept drift and apply fallback locking.

    Args:
        dynamic_thresholds: Dynamic threshold array.
        baseline_threshold: Static baseline threshold.
        drift_threshold: Drift ratio trigger (e.g. 0.20).
        recovery_threshold: Recovery ratio (e.g. 0.10).
        recovery_windows: Consecutive windows below recovery to resume.
        window_size: Window size for drift detection.

    Returns:
        Tuple of (adjusted_thresholds, drift_events list).
    """
    logger.info("── Step 4: Concept drift detection ──")
    adjusted = dynamic_thresholds.copy()
    drift_events: List[Dict[str, Any]] = []
    locked = False
    consecutive_stable = 0

    for i in range(window_size, len(adjusted)):
        if i % window_size != 0:
            if locked:
                adjusted[i] = baseline_threshold
            continue

        drift_ratio = abs(adjusted[i] - baseline_threshold) / max(baseline_threshold, 1e-8)

        if drift_ratio > drift_threshold and not locked:
            locked = True
            consecutive_stable = 0
            logger.info(
                "  Concept drift detected: drift_ratio=%.4f at sample %d",
                drift_ratio,
                i,
            )
            drift_events.append(
                {
                    "sample_index": int(i),
                    "drift_ratio": round(drift_ratio, 4),
                    "action": "FALLBACK_LOCKED",
                    "dynamic_threshold": round(float(adjusted[i]), 6),
                    "baseline_threshold": round(baseline_threshold, 6),
                }
            )
            adjusted[i] = baseline_threshold

        elif locked:
            adjusted[i] = baseline_threshold
            if drift_ratio < recovery_threshold:
                consecutive_stable += 1
            else:
                consecutive_stable = 0

            if consecutive_stable >= recovery_windows:
                locked = False
                consecutive_stable = 0
                logger.info(
                    "  Drift recovered: %d consecutive stable windows at sample %d",
                    recovery_windows,
                    i,
                )
                drift_events.append(
                    {
                        "sample_index": int(i),
                        "drift_ratio": round(drift_ratio, 4),
                        "action": "RESUMED_DYNAMIC",
                        "dynamic_threshold": round(float(dynamic_thresholds[i]), 6),
                        "baseline_threshold": round(baseline_threshold, 6),
                    }
                )

    n_locked = int(np.sum(np.isclose(adjusted, baseline_threshold)))
    logger.info(
        "  Drift events: %d, locked samples: %d/%d",
        len(drift_events),
        n_locked,
        len(adjusted),
    )
    return adjusted, drift_events


# ---------------------------------------------------------------------------
# Step 5: Threshold fallback (integrated in drift detection above)
# ---------------------------------------------------------------------------
# Fallback logic is embedded in _detect_concept_drift:
# - Lock dynamic_threshold = baseline_threshold on drift
# - Resume dynamic after recovery_windows consecutive windows below threshold


# ---------------------------------------------------------------------------
# Step 6: Risk scoring
# ---------------------------------------------------------------------------


def _classify_risk(
    anomaly_scores: np.ndarray,
    thresholds: np.ndarray,
    mad: float,
    risk_config: Dict[str, float],
    raw_features: np.ndarray,
    biometric_cols: List[str],
    feature_names: List[str],
) -> List[Dict[str, Any]]:
    """Classify each sample into risk levels.

    Risk levels:
        NORMAL:   distance < 0
        LOW:      0 <= distance < 0.5*MAD
        MEDIUM:   0.5*MAD <= distance < 1.0*MAD
        HIGH:     1.0*MAD <= distance < 2.0*MAD
        CRITICAL: distance >= 2.0*MAD AND cross-modal fusion

    Args:
        anomaly_scores: Model sigmoid outputs, shape (N,).
        thresholds: Current thresholds (post-drift-adjustment), shape (N,).
        mad: Median Absolute Deviation from baseline.
        risk_config: Dict with low_upper, medium_upper, high_upper.
        raw_features: Raw feature values for cross-modal detection, shape (N, F).
        biometric_cols: List of biometric column names.
        feature_names: List of all feature names.

    Returns:
        List of per-sample risk assessment dicts.
    """
    logger.info("── Step 6: Risk scoring ──")

    low_bound = risk_config["low_upper"] * mad
    medium_bound = risk_config["medium_upper"] * mad
    high_bound = risk_config["high_upper"] * mad

    bio_indices = [feature_names.index(c) for c in biometric_cols if c in feature_names]
    net_indices = [i for i in range(len(feature_names)) if i not in bio_indices]

    risk_results: List[Dict[str, Any]] = []
    level_counts = {
        RISK_NORMAL: 0,
        RISK_LOW: 0,
        RISK_MEDIUM: 0,
        RISK_HIGH: 0,
        RISK_CRITICAL: 0,
    }

    for i in range(len(anomaly_scores)):
        score = float(anomaly_scores[i])
        threshold = float(thresholds[i])
        distance = score - threshold

        if distance < 0:
            level = RISK_NORMAL
        elif distance < low_bound:
            level = RISK_LOW
        elif distance < medium_bound:
            level = RISK_MEDIUM
        elif distance < high_bound:
            level = RISK_HIGH
        else:
            # Check cross-modal fusion for CRITICAL
            if i < len(raw_features):
                bio_vals = raw_features[i, bio_indices] if bio_indices else np.array([])
                net_vals = raw_features[i, net_indices] if net_indices else np.array([])

                # Cross-modal: both biometric AND network show anomalous values
                # (values beyond 2 std from 0 in both modalities)
                bio_anomaly = bool(np.any(np.abs(bio_vals) > 2.0)) if len(bio_vals) > 0 else False
                net_anomaly = bool(np.any(np.abs(net_vals) > 2.0)) if len(net_vals) > 0 else False

                if bio_anomaly and net_anomaly:
                    level = RISK_CRITICAL
                else:
                    level = RISK_HIGH
            else:
                level = RISK_HIGH

        level_counts[level] += 1
        risk_results.append(
            {
                "sample_index": i,
                "anomaly_score": round(score, 6),
                "threshold": round(threshold, 6),
                "distance": round(distance, 6),
                "risk_level": level,
            }
        )

    for lvl, count in level_counts.items():
        pct = count / len(anomaly_scores) * 100 if len(anomaly_scores) > 0 else 0
        logger.info("  %s: %d (%.1f%%)", lvl, count, pct)

    return risk_results


# ---------------------------------------------------------------------------
# Step 7: Artifact export
# ---------------------------------------------------------------------------


def _export_baseline(output_dir: Path, baseline: Dict[str, Any], filename: str) -> Path:
    """Export immutable baseline config.

    Args:
        output_dir: Output directory path.
        baseline: Baseline config dict.
        filename: Output filename.

    Returns:
        Path to exported file.
    """
    path = output_dir / filename
    with open(path, "w") as f:
        json.dump(baseline, f, indent=2)
    logger.info("  Exported baseline config: %s", filename)
    return path


def _export_threshold_config(
    output_dir: Path,
    baseline: Dict[str, Any],
    window_log: List[Dict[str, Any]],
    config: Dict[str, Any],
    filename: str,
) -> Path:
    """Export current threshold configuration.

    Args:
        output_dir: Output directory path.
        baseline: Baseline config dict.
        window_log: Dynamic threshold window log.
        config: Phase 4 YAML config.
        filename: Output filename.

    Returns:
        Path to exported file.
    """
    threshold_cfg = {
        "baseline_threshold": baseline["baseline_threshold"],
        "current_dynamic_threshold": (
            window_log[-1]["dynamic_threshold"] if window_log else baseline["baseline_threshold"]
        ),
        "k_schedule": config["dynamic_threshold"]["k_schedule"],
        "window_size": config["dynamic_threshold"]["window_size"],
        "window_log_entries": len(window_log),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    path = output_dir / filename
    with open(path, "w") as f:
        json.dump(threshold_cfg, f, indent=2)
    logger.info("  Exported threshold config: %s", filename)
    return path


def _export_risk_report(
    output_dir: Path,
    risk_results: List[Dict[str, Any]],
    baseline: Dict[str, Any],
    metrics: Dict[str, Any],
    hw_info: Dict[str, str],
    duration_s: float,
    git_commit: str,
    filename: str,
) -> Path:
    """Export risk report with per-sample assessments and summary.

    Args:
        output_dir: Output directory path.
        risk_results: Per-sample risk assessment list.
        baseline: Baseline config dict.
        metrics: Phase 3 evaluation metrics.
        hw_info: Hardware info dict.
        duration_s: Pipeline duration in seconds.
        git_commit: Current git commit hash.
        filename: Output filename.

    Returns:
        Path to exported file.
    """
    # Summary counts
    level_counts = {}
    for r in risk_results:
        lvl = r["risk_level"]
        level_counts[lvl] = level_counts.get(lvl, 0) + 1

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline": "phase4_risk_adaptive",
        "git_commit": git_commit,
        "hardware": hw_info,
        "duration_seconds": round(duration_s, 2),
        "baseline": {
            "median": baseline["median"],
            "mad": baseline["mad"],
            "threshold": baseline["baseline_threshold"],
            "n_normal_samples": baseline["n_normal_samples"],
        },
        "phase3_metrics": {
            "accuracy": metrics.get("accuracy", 0),
            "f1_score": metrics.get("f1_score", 0),
            "auc_roc": metrics.get("auc_roc", 0),
        },
        "risk_distribution": level_counts,
        "total_samples": len(risk_results),
        "sample_assessments": risk_results,
    }
    path = output_dir / filename
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("  Exported risk report: %s (%d samples)", filename, len(risk_results))
    return path


def _export_drift_log(
    output_dir: Path,
    drift_events: List[Dict[str, Any]],
    filename: str,
) -> Path:
    """Export drift events as CSV.

    Args:
        output_dir: Output directory path.
        drift_events: List of drift event dicts.
        filename: Output filename.

    Returns:
        Path to exported file.
    """
    if drift_events:
        df = pd.DataFrame(drift_events)
    else:
        df = pd.DataFrame(
            columns=[
                "sample_index",
                "drift_ratio",
                "action",
                "dynamic_threshold",
                "baseline_threshold",
            ]
        )
    path = output_dir / filename
    df.to_csv(path, index=False)
    logger.info("  Exported drift log: %s (%d events)", filename, len(drift_events))
    return path


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _generate_report(
    baseline: Dict[str, Any],
    risk_results: List[Dict[str, Any]],
    drift_events: List[Dict[str, Any]],
    window_log: List[Dict[str, Any]],
    config: Dict[str, Any],
    hw_info: Dict[str, str],
    duration_s: float,
    p3_metrics: Dict[str, Any],
    git_commit: str,
) -> str:
    """Generate report_section_risk_adaptive.md content.

    Args:
        baseline: Baseline config dict.
        risk_results: Per-sample risk assessments.
        drift_events: Concept drift events.
        window_log: Dynamic threshold window entries.
        config: Phase 4 YAML config.
        hw_info: Hardware info dict.
        duration_s: Pipeline duration.
        p3_metrics: Phase 3 evaluation metrics.
        git_commit: Current git commit hash.

    Returns:
        Markdown report string.
    """
    # Risk distribution
    level_counts: Dict[str, int] = {}
    for r in risk_results:
        lvl = r["risk_level"]
        level_counts[lvl] = level_counts.get(lvl, 0) + 1

    n_total = len(risk_results)
    risk_rows = ""
    for lvl in [RISK_NORMAL, RISK_LOW, RISK_MEDIUM, RISK_HIGH, RISK_CRITICAL]:
        count = level_counts.get(lvl, 0)
        pct = count / n_total * 100 if n_total > 0 else 0
        risk_rows += f"| {lvl} | {count} | {pct:.1f}% |\n"

    # Drift events summary
    drift_rows = ""
    for d in drift_events:
        drift_rows += (
            f"| {d['sample_index']} | {d['drift_ratio']:.4f}"
            f" | {d['action']} | {d['dynamic_threshold']:.6f} |\n"
        )
    if not drift_rows:
        drift_rows = "| — | — | No drift events detected | — |\n"

    # k(t) schedule
    k_rows = ""
    for entry in config["dynamic_threshold"]["k_schedule"]:
        k_rows += (
            f"| {entry['start_hour']:02d}:00–{entry['end_hour']:02d}:00" f" | {entry['k']} |\n"
        )

    # Window log sample (first 5)
    win_rows = ""
    for w in window_log[:5]:
        win_rows += (
            f"| {w['sample_index']} | {w['hour']:02d}:00"
            f" | {w['k_t']} | {w['window_median']:.6f}"
            f" | {w['dynamic_threshold']:.6f} |\n"
        )
    if len(window_log) > 5:
        win_rows += "| ... | ... | ... | ... | ... |\n"

    lo = config["risk_levels"]["low_upper"]
    med = config["risk_levels"]["medium_upper"]
    hi = config["risk_levels"]["high_upper"]

    report = f"""## 7.1 Risk-Adaptive Engine

This section documents the Phase 4 Risk-Adaptive Engine, which applies
dynamic thresholding, concept drift detection, and multi-level risk scoring
to the Phase 3 classification output for IoMT healthcare environments.

### 7.1.1 Baseline Computation

Baseline computed from Normal-only training samples using Median + k*MAD:

| Parameter | Value |
|-----------|-------|
| Normal samples (train) | {baseline['n_normal_samples']} |
| Attention dimensions | {baseline['n_attention_dims']} |
| Median | {baseline['median']:.6f} |
| MAD | {baseline['mad']:.6f} |
| k (multiplier) | {baseline['mad_multiplier']} |
| **Baseline threshold** | **{baseline['baseline_threshold']:.6f}** |

Formula: `baseline_threshold = Median + {baseline['mad_multiplier']} * MAD`

### 7.1.2 Dynamic Thresholding

Rolling window Median + k(t)*MAD with time-of-day sensitivity:

| Time Window | k(t) |
|-------------|------|
{k_rows}
**Window size:** {config['dynamic_threshold']['window_size']} samples

Sample window log:

| Sample | Hour | k(t) | Window Median | Dynamic Threshold |
|--------|------|------|---------------|-------------------|
{win_rows}
### 7.1.3 Concept Drift Detection

| Parameter | Value |
|-----------|-------|
| Drift threshold | {config['concept_drift']['drift_threshold']} (20%) |
| Recovery threshold | {config['concept_drift']['recovery_threshold']} (10%) |
| Recovery windows | {config['concept_drift']['recovery_windows']} consecutive |
| Drift events detected | {len(drift_events)} |

Drift events:

| Sample Index | Drift Ratio | Action | Dynamic Threshold |
|-------------|-------------|--------|-------------------|
{drift_rows}
### 7.1.4 Risk Level Classification

| Risk Level | Count | Percentage |
|------------|-------|------------|
{risk_rows}
**Total samples scored:** {n_total}

Risk thresholds (MAD-relative):

| Level | Condition |
|-------|-----------|
| NORMAL | distance < 0 |
| LOW | 0 <= distance < {lo}*MAD |
| MEDIUM | {lo}*MAD <= distance < {med}*MAD |
| HIGH | {med}*MAD <= distance < {hi}*MAD |
| CRITICAL | distance >= {hi}*MAD AND cross-modal |

### 7.1.5 CRITICAL Risk Protocol

When a CRITICAL risk level is assigned:

1. Immediate on-site alert dispatched
2. Suspicious device isolated from network
3. Escalation chain: IT admin + doctor on duty + manager
4. Medical device is **NOT** shut down (patient safety)
5. Full context logged for human review

### 7.1.6 Phase 3 Model Performance (Inherited)

| Metric | Value |
|--------|-------|
| Accuracy | {p3_metrics.get('accuracy', 0):.4f} |
| F1-score | {p3_metrics.get('f1_score', 0):.4f} |
| AUC-ROC | {p3_metrics.get('auc_roc', 0):.4f} |
| Test samples | {p3_metrics.get('test_samples', 0)} |

### 7.1.7 Execution Summary

| Metric | Value |
|--------|-------|
| Device | {hw_info.get('device', 'N/A')} |
| TensorFlow | {hw_info.get('tensorflow', 'N/A')} |
| Python | {hw_info.get('python', 'N/A')} |
| Platform | {hw_info.get('platform', 'N/A')} |
| Duration | {duration_s:.2f}s |
| Git commit | `{git_commit[:12]}` |

### 7.1.8 Artifacts Exported

| Artifact | Description |
|----------|-------------|
| `baseline_config.json` | Median, MAD, baseline threshold (IMMUTABLE) |
| `threshold_config.json` | Current dynamic threshold, k(t) schedule |
| `risk_report.json` | Per-sample risk levels with distances |
| `drift_log.csv` | Concept drift events and fallback triggers |

---

**Reviewer signature:** ____________________  **Date:** ____/____/______
"""
    return report


# ---------------------------------------------------------------------------
# Main pipeline orchestrator
# ---------------------------------------------------------------------------


def run_pipeline() -> Dict[str, Any]:
    """Execute Phase 4 Risk-Adaptive Engine pipeline.

    Returns:
        Risk report summary dict.
    """
    t0 = time.time()

    logger.info("═══════════════════════════════════════════════════")
    logger.info("  Phase 4 Risk-Adaptive Engine")
    logger.info("═══════════════════════════════════════════════════")

    # ── Load config ──
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    # ── Hardware detection ──
    hw_info = _detect_hardware()

    # ── Reproducibility seeds ──
    np.random.seed(config["random_state"])  # noqa: NPY002
    tf.random.set_seed(config["random_state"])
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    # ── Step 1: Verify and load artifacts ──
    p3_metadata = _verify_phase3_artifacts(config)
    p2_metadata = _verify_phase2_artifacts(config)
    attn_df = _load_attention_output(config)
    X_train, y_train, X_test, y_test, feature_names = _load_phase1_data(config)

    # Load Phase 3 metrics (DO NOT recompute)
    p3_metrics_path = PROJECT_ROOT / config["data"]["phase3_dir"] / "metrics_report.json"
    p3_metrics = json.loads(p3_metrics_path.read_text())["metrics"]

    # ── Rebuild and load classification model ──
    model = _rebuild_classification_model(p2_metadata, p3_metadata, config)

    # ── Reshape test data for model inference ──
    hp = p2_metadata["hyperparameters"]
    reshaper = DataReshaper(timesteps=hp["timesteps"], stride=hp["stride"])
    X_test_w, y_test_w = reshaper.reshape(X_test, y_test)
    logger.info("  Test data reshaped: %s", X_test_w.shape)

    # ── Get model predictions (anomaly scores) ──
    logger.info("── Computing anomaly scores ──")
    anomaly_scores = model.predict(X_test_w, verbose=0).flatten()
    logger.info(
        "  Anomaly scores: min=%.4f, max=%.4f, mean=%.4f",
        float(anomaly_scores.min()),
        float(anomaly_scores.max()),
        float(anomaly_scores.mean()),
    )

    # ── Step 2: Baseline computation ──
    baseline = _compute_baseline(attn_df, config["baseline"]["mad_multiplier"])

    # ── Step 3: Dynamic thresholding ──
    dynamic_thresholds, window_log = _compute_dynamic_thresholds(
        anomaly_scores=anomaly_scores,
        baseline=baseline,
        window_size=config["dynamic_threshold"]["window_size"],
        k_schedule=config["dynamic_threshold"]["k_schedule"],
    )

    # ── Step 4 + 5: Concept drift detection + fallback ──
    adjusted_thresholds, drift_events = _detect_concept_drift(
        dynamic_thresholds=dynamic_thresholds,
        baseline_threshold=baseline["baseline_threshold"],
        drift_threshold=config["concept_drift"]["drift_threshold"],
        recovery_threshold=config["concept_drift"]["recovery_threshold"],
        recovery_windows=config["concept_drift"]["recovery_windows"],
        window_size=config["dynamic_threshold"]["window_size"],
    )

    # ── Step 6: Risk scoring ──
    # Use raw test features for cross-modal fusion detection
    # Truncate to match windowed sample count
    raw_test_features = X_test[: len(anomaly_scores)]
    risk_results = _classify_risk(
        anomaly_scores=anomaly_scores,
        thresholds=adjusted_thresholds,
        mad=baseline["mad"],
        risk_config=config["risk_levels"],
        raw_features=raw_test_features,
        biometric_cols=config["biometric_columns"],
        feature_names=feature_names,
    )

    # ── Step 7: Export artifacts ──
    logger.info("── Step 7: Exporting artifacts ──")
    output_dir = PROJECT_ROOT / config["output"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    duration_s = time.time() - t0
    git_commit = _get_git_commit()

    _export_baseline(output_dir, baseline, config["output"]["baseline_file"])
    _export_threshold_config(
        output_dir, baseline, window_log, config, config["output"]["threshold_file"]
    )
    _export_risk_report(
        output_dir,
        risk_results,
        baseline,
        p3_metrics,
        hw_info,
        duration_s,
        git_commit,
        config["output"]["risk_report_file"],
    )
    _export_drift_log(output_dir, drift_events, config["output"]["drift_log_file"])

    # ── Generate report ──
    report_md = _generate_report(
        baseline=baseline,
        risk_results=risk_results,
        drift_events=drift_events,
        window_log=window_log,
        config=config,
        hw_info=hw_info,
        duration_s=duration_s,
        p3_metrics=p3_metrics,
        git_commit=git_commit,
    )
    report_dir = PROJECT_ROOT / "results" / "phase0_analysis"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "report_section_risk_adaptive.md"
    with open(report_path, "w") as f:
        f.write(report_md)
    logger.info("  Report saved: %s", report_path.name)

    # ── Summary ──
    level_counts: Dict[str, int] = {}
    for r in risk_results:
        lvl = r["risk_level"]
        level_counts[lvl] = level_counts.get(lvl, 0) + 1

    logger.info("═══════════════════════════════════════════════════")
    logger.info("  Phase 4 Risk-Adaptive Engine — %.2fs", duration_s)
    logger.info("  Baseline threshold: %.6f", baseline["baseline_threshold"])
    logger.info("  Drift events: %d", len(drift_events))
    logger.info("  Risk distribution: %s", level_counts)
    logger.info("═══════════════════════════════════════════════════")

    return {
        "baseline": baseline,
        "risk_distribution": level_counts,
        "drift_events": len(drift_events),
        "duration_s": duration_s,
    }


def main() -> None:
    """Entry point for Phase 4 Risk-Adaptive Engine."""
    run_pipeline()


if __name__ == "__main__":
    main()
