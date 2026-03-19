"""Phase 5 Explanation Engine — SHAP-based feature attribution for IoMT alerts.

Loads Phase 2/3/4 artifacts (model, risk report, attention output, baseline),
filters non-NORMAL samples, computes SHAP values, generates human-readable
explanations, waterfall/bar/timeline visualizations, and exports all artifacts.

STRICT RULES:
- DO NOT rebuild model — load weights into rebuilt architecture
- DO NOT recompute risk levels — load from risk_report.json
- DO NOT recompute attention weights — load from attention_output.parquet
- DO NOT recompute thresholds — load from baseline_config.json
- Run SHAP ONLY on samples with risk > NORMAL

Usage::

    python -m src.phase5_explanation_engine.phase5_explanation
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

import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Phase 2 SOLID components (reused, NOT duplicated)
from src.phase2_detection_engine.phase2.assembler import (
    DetectionModelAssembler,
)
from src.phase2_detection_engine.phase2.attention_builder import (  # noqa: F401
    AttentionBuilder,
    BahdanauAttention,
)
from src.phase2_detection_engine.phase2.bilstm_builder import BiLSTMBuilder
from src.phase2_detection_engine.phase2.cnn_builder import CNNBuilder
from src.phase2_detection_engine.phase2.reshaper import DataReshaper

logger = logging.getLogger(__name__)

# ── Named constants ──────────────────────────────────────────────────

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
CONFIG_PATH: Path = PROJECT_ROOT / "config" / "phase5_config.yaml"

N_FEATURES: int = 29
TIMESTEPS: int = 20
STRIDE: int = 1
HASH_CHUNK: int = 65_536

# SHAP defaults
BACKGROUND_SAMPLES: int = 100
MAX_EXPLAIN_SAMPLES: int = 200
TOP_K_FEATURES: int = 10
MAX_WATERFALL_CHARTS: int = 5
MAX_TIMELINE_CHARTS: int = 3

# Risk levels (matching Phase 4)
RISK_LEVELS_NON_NORMAL = ("LOW", "MEDIUM", "HIGH", "CRITICAL")

# Biometric columns (HIPAA — names only, never raw values in logs)
BIOMETRIC_COLUMNS = frozenset(
    {"Temp", "SpO2", "Pulse_Rate", "SYS", "DIA", "Heart_rate", "Resp_Rate", "ST"}
)

# Explanation templates
TEMPLATES: Dict[str, str] = {
    "CRITICAL": (
        "CRITICAL ALERT: Sample {idx} at T={time}. "
        "Top factors: {f1}={v1:.4f} ({p1:.1f}%), "
        "{f2}={v2:.4f} ({p2:.1f}%), "
        "{f3}={v3:.4f} ({p3:.1f}%). "
        "Immediate action required."
    ),
    "HIGH": (
        "HIGH ALERT: Suspicious activity detected at sample {idx}. "
        "Primary indicator: {f1} contributing {p1:.1f}%."
    ),
    "MEDIUM": ("MEDIUM: Anomaly detected at sample {idx}. " "Monitor closely. Key feature: {f1}."),
    "LOW": "LOW: Minor anomaly at sample {idx}. No immediate action needed.",
}

# Chart styling
CHART_DPI: int = 150
FIG_WIDTH: float = 10.0
FIG_HEIGHT: float = 6.0


# ── Utility functions ────────────────────────────────────────────────


def _compute_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(HASH_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _get_git_commit() -> str:
    """Get current git commit hash for artifact versioning."""
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


def _detect_hardware() -> Dict[str, str]:
    """Detect GPU/CPU availability and return hardware info dict."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        info = {"device": f"GPU: {gpus[0].name}", "cuda": "available"}
    else:
        cpu_info = platform.processor() or platform.machine()
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        info = {"device": f"CPU: {cpu_info}", "cuda": "N/A (CPU execution)"}
    info["tensorflow"] = tf.__version__
    info["python"] = platform.python_version()
    info["platform"] = platform.platform()
    return info


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML config file."""
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


# ── 1. Load & verify artifacts ───────────────────────────────────────


def _verify_artifacts(
    p2_meta_path: Path,
    p3_meta_path: Path,
    p4_meta_path: Path,
    p2_dir: Path,
    p3_dir: Path,
    p4_dir: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Verify all Phase 2/3/4 artifacts via SHA-256 hashes.

    Args:
        p2_meta_path: Path to Phase 2 detection_metadata.json.
        p3_meta_path: Path to Phase 3 classification_metadata.json.
        p4_meta_path: Path to Phase 4 risk_metadata.json.
        p2_dir: Phase 2 output directory.
        p3_dir: Phase 3 output directory.
        p4_dir: Phase 4 output directory.

    Returns:
        Tuple of (p2_metadata, p3_metadata, p4_metadata).

    Raises:
        ValueError: If any SHA-256 mismatch is detected.
    """
    logger.info("── Artifact verification (SHA-256) ──")
    verified = 0

    # Phase 2
    p2_meta = json.loads(p2_meta_path.read_text())
    for name, info in p2_meta["artifact_hashes"].items():
        actual = _compute_sha256(p2_dir / name)
        if actual != info["sha256"]:
            raise ValueError(f"SHA-256 mismatch: Phase 2 {name}")
        verified += 1
    logger.info("  Phase 2: %d artifacts verified", verified)

    # Phase 3
    v3 = 0
    p3_meta = json.loads(p3_meta_path.read_text())
    for name, info in p3_meta["artifact_hashes"].items():
        actual = _compute_sha256(p3_dir / name)
        if actual != info["sha256"]:
            raise ValueError(f"SHA-256 mismatch: Phase 3 {name}")
        v3 += 1
    logger.info("  Phase 3: %d artifacts verified", v3)

    # Phase 4
    v4 = 0
    p4_meta = json.loads(p4_meta_path.read_text())
    for name, info in p4_meta["artifact_hashes"].items():
        actual = _compute_sha256(p4_dir / name)
        if actual != info["sha256"]:
            raise ValueError(f"SHA-256 mismatch: Phase 4 {name}")
        v4 += 1
    logger.info("  Phase 4: %d artifacts verified", v4)
    logger.info("  Total: %d artifacts verified", verified + v3 + v4)

    return p2_meta, p3_meta, p4_meta


def _rebuild_model(
    p2_metadata: Dict[str, Any],
    p3_metadata: Dict[str, Any],
    p3_dir: Path,
) -> tf.keras.Model:
    """Rebuild full classification model and load Phase 3 weights.

    Args:
        p2_metadata: Phase 2 metadata with hyperparameters.
        p3_metadata: Phase 3 metadata with head config.
        p3_dir: Phase 3 directory containing weights file.

    Returns:
        Loaded Keras model with sigmoid output.
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

    # Attach classification head (same as Phase 3/4)
    p3_hp = p3_metadata["hyperparameters"]
    x = tf.keras.layers.Dense(
        p3_hp["dense_units"],
        activation=p3_hp["dense_activation"],
        name="dense_head",
    )(detection_model.output)
    x = tf.keras.layers.Dropout(p3_hp["head_dropout_rate"], name="drop_head")(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)
    full_model = tf.keras.Model(detection_model.input, output, name="classification_engine")

    weights_path = p3_dir / "classification_model.weights.h5"
    full_model.load_weights(str(weights_path))
    logger.info(
        "  Model loaded: %d params, %d layers",
        full_model.count_params(),
        len(full_model.layers),
    )
    return full_model


def _load_risk_report(p4_dir: Path) -> Dict[str, Any]:
    """Load risk_report.json from Phase 4 output.

    Args:
        p4_dir: Phase 4 output directory.

    Returns:
        Full risk report dict with sample_assessments.
    """
    path = p4_dir / "risk_report.json"
    report = json.loads(path.read_text())
    logger.info(
        "  Risk report loaded: %d samples, dist=%s",
        report["total_samples"],
        report["risk_distribution"],
    )
    return report


def _load_baseline(p4_dir: Path) -> Dict[str, Any]:
    """Load baseline_config.json from Phase 4 output.

    Args:
        p4_dir: Phase 4 output directory.

    Returns:
        Baseline config dict.
    """
    path = p4_dir / "baseline_config.json"
    baseline = json.loads(path.read_text())
    logger.info(
        "  Baseline loaded: threshold=%.6f, mad=%.6f",
        baseline["baseline_threshold"],
        baseline["mad"],
    )
    return baseline


# ── 2. Filter non-NORMAL samples ─────────────────────────────────────


def _filter_non_normal(
    sample_assessments: List[Dict[str, Any]],
    max_samples: int,
    rng: np.random.Generator,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Filter samples where risk_level != NORMAL.

    Args:
        sample_assessments: Per-sample risk dicts from risk_report.json.
        max_samples: Maximum samples to explain (stratified sampling).
        rng: Random number generator.

    Returns:
        Tuple of (filtered_samples, level_counts).
    """
    non_normal = [s for s in sample_assessments if s["risk_level"] != "NORMAL"]

    level_counts: Dict[str, int] = {}
    for level in RISK_LEVELS_NON_NORMAL:
        count = sum(1 for s in non_normal if s["risk_level"] == level)
        level_counts[level] = count

    # Stratified sampling if too many
    if len(non_normal) > max_samples:
        sampled: List[Dict[str, Any]] = []
        for level in RISK_LEVELS_NON_NORMAL:
            level_samples = [s for s in non_normal if s["risk_level"] == level]
            n_take = max(
                1,
                int(max_samples * len(level_samples) / len(non_normal)),
            )
            n_take = min(n_take, len(level_samples))
            indices = rng.choice(len(level_samples), n_take, replace=False)
            sampled.extend(level_samples[i] for i in sorted(indices))
        non_normal = sampled

    counts_str = ", ".join(f"{k}={v}" for k, v in level_counts.items())
    logger.info(
        "  Filtered %d samples for explanation (%s)",
        len(non_normal),
        counts_str,
    )
    return non_normal, level_counts


# ── 3. SHAP computation ──────────────────────────────────────────────


def _prepare_background_data(
    train_path: Path,
    label_column: str,
    n_background: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Prepare SHAP background data from Normal training samples.

    Args:
        train_path: Path to train_phase1.parquet.
        label_column: Name of label column.
        n_background: Number of background samples.
        rng: Random number generator.

    Returns:
        Background data array, shape (n_background, TIMESTEPS, N_FEATURES).
    """
    logger.info("── Preparing SHAP background data ──")
    train_df = pd.read_parquet(train_path)
    feature_names = [c for c in train_df.columns if c != label_column]

    normal_mask = train_df[label_column] == 0
    X_normal = train_df.loc[normal_mask, feature_names].values.astype(np.float32)

    reshaper = DataReshaper(timesteps=TIMESTEPS, stride=STRIDE)
    y_dummy = np.zeros(len(X_normal), dtype=np.int32)
    X_windows, _ = reshaper.reshape(X_normal, y_dummy)

    indices = rng.choice(len(X_windows), n_background, replace=False)
    background = X_windows[indices]
    logger.info("  Background data: %s from %d Normal windows", background.shape, len(X_windows))
    return background


def _prepare_explanation_data(
    test_path: Path,
    label_column: str,
    sample_indices: List[int],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Prepare windowed test data for SHAP explanation.

    Args:
        test_path: Path to test_phase1.parquet.
        label_column: Name of label column.
        sample_indices: Indices of samples to explain (from risk_report).

    Returns:
        Tuple of (X_explain windows, anomaly_scores placeholder, feature_names).
    """
    test_df = pd.read_parquet(test_path)
    feature_names = [c for c in test_df.columns if c != label_column]

    X_test = test_df[feature_names].values.astype(np.float32)
    y_test = test_df[label_column].values.astype(np.int32)

    reshaper = DataReshaper(timesteps=TIMESTEPS, stride=STRIDE)
    X_windows, _ = reshaper.reshape(X_test, y_test)

    # Clamp indices to valid range
    valid = [i for i in sample_indices if i < len(X_windows)]
    X_explain = X_windows[valid]

    logger.info("  Explanation data: %s (%d samples)", X_explain.shape, len(valid))
    return X_explain, X_windows, feature_names


def _compute_shap_values(
    model: tf.keras.Model,
    background: np.ndarray,
    X_explain: np.ndarray,
) -> np.ndarray:
    """Compute SHAP values using GradientExplainer.

    Falls back to gradient-based attribution if SHAP fails.

    Args:
        model: Loaded Keras classification model.
        background: Background data, shape (B, T, F).
        X_explain: Samples to explain, shape (N, T, F).

    Returns:
        SHAP values array, shape (N, T, F).
    """
    logger.info("── Computing SHAP values ──")
    logger.info("  Explaining %d samples against %d background", len(X_explain), len(background))

    try:
        import shap

        explainer = shap.GradientExplainer(model, background)
        shap_vals = explainer.shap_values(X_explain)

        # GradientExplainer may return list for multi-output
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]

        shap_vals = np.array(shap_vals)
        # Squeeze trailing output dim: (N, T, F, 1) → (N, T, F)
        if shap_vals.ndim == 4 and shap_vals.shape[-1] == 1:
            shap_vals = shap_vals.squeeze(-1)
        logger.info("  SHAP computed via GradientExplainer: %s", shap_vals.shape)
        return shap_vals

    except Exception as e:
        logger.warning("  GradientExplainer failed (%s), using gradient attribution", e)
        return _gradient_attribution(model, background, X_explain)


def _gradient_attribution(
    model: tf.keras.Model,
    background: np.ndarray,
    X_explain: np.ndarray,
    n_steps: int = 50,
) -> np.ndarray:
    """Integrated gradients fallback for feature attribution.

    Args:
        model: Loaded Keras model.
        background: Background data for baseline.
        X_explain: Samples to explain.
        n_steps: Integration steps.

    Returns:
        Attribution values, shape (N, T, F).
    """
    logger.info("  Computing integrated gradients (%d steps)", n_steps)
    baseline_ref = np.mean(background, axis=0, keepdims=True)
    attributions = np.zeros_like(X_explain)

    batch_size = 32
    for start in range(0, len(X_explain), batch_size):
        end = min(start + batch_size, len(X_explain))
        batch = X_explain[start:end]

        batch_attr = np.zeros_like(batch)
        for step in range(n_steps):
            alpha = step / n_steps
            interpolated = baseline_ref + alpha * (batch - baseline_ref)
            inp = tf.constant(interpolated, dtype=tf.float32)

            with tf.GradientTape() as tape:
                tape.watch(inp)
                pred = model(inp, training=False)
            grads = tape.gradient(pred, inp).numpy()
            batch_attr += grads / n_steps

        attributions[start:end] = batch_attr * (batch - baseline_ref)

    logger.info("  Integrated gradients computed: %s", attributions.shape)
    return attributions


# ── 4. Feature importance ranking ─────────────────────────────────────


def _compute_feature_importance(
    shap_values: np.ndarray,
    feature_names: List[str],
    top_k: int,
) -> Tuple[pd.DataFrame, List[Tuple[str, float]]]:
    """Compute global feature importance from SHAP values.

    Aggregates 3D SHAP values (N, T, F) over timesteps → (N, F),
    then takes mean(|values|) over samples → (F,).

    Args:
        shap_values: SHAP values, shape (N, T, F).
        feature_names: List of feature names.
        top_k: Number of top features to return.

    Returns:
        Tuple of (full importance DataFrame, top_k list of (name, score)).
    """
    # Aggregate over timesteps: mean absolute SHAP per feature per sample
    per_sample = np.mean(np.abs(shap_values), axis=1)  # (N, F)

    # Global: mean over samples
    global_importance = np.mean(per_sample, axis=0)  # (F,)

    df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": global_importance}).sort_values(
        "mean_abs_shap", ascending=False
    )
    df["rank"] = range(1, len(df) + 1)

    top_features = [(row["feature"], row["mean_abs_shap"]) for _, row in df.head(top_k).iterrows()]

    logger.info("  Top %d features:", top_k)
    for name, score in top_features:
        logger.info("    %s: %.6f", name, score)

    return df, top_features


# ── 5. Context enrichment ────────────────────────────────────────────


def _get_top3_features(
    sample_shap: np.ndarray,
    feature_names: List[str],
) -> List[Dict[str, Any]]:
    """Extract top 3 contributing features for a single sample.

    Args:
        sample_shap: SHAP values for one sample, shape (T, F).
        feature_names: List of feature names.

    Returns:
        List of dicts with feature, shap_value, contribution_pct.
    """
    per_feature = np.mean(np.abs(sample_shap), axis=0)
    total = per_feature.sum()
    if total < 1e-12:
        total = 1.0

    ranked_idx = np.argsort(per_feature)[::-1][:3]
    top3 = []
    for idx in ranked_idx:
        top3.append(
            {
                "feature": feature_names[idx],
                "shap_value": round(float(per_feature[idx]), 6),
                "contribution_pct": round(float(per_feature[idx] / total * 100), 2),
            }
        )
    return top3


def _enrich_samples(
    filtered_samples: List[Dict[str, Any]],
    shap_values: np.ndarray,
    feature_names: List[str],
) -> List[Dict[str, Any]]:
    """Attach SHAP context and explanation to each filtered sample.

    Args:
        filtered_samples: Non-NORMAL sample dicts from risk_report.
        shap_values: SHAP values, shape (N, T, F).
        feature_names: List of feature names.

    Returns:
        Enriched sample dicts with top_features and explanation.
    """
    logger.info("── Context enrichment ──")
    enriched = []

    for i, sample in enumerate(filtered_samples):
        if i >= len(shap_values):
            break

        top3 = _get_top3_features(shap_values[i], feature_names)
        explanation = _generate_explanation(
            sample["risk_level"],
            sample["sample_index"],
            top3,
        )

        enriched.append(
            {
                "sample_index": sample["sample_index"],
                "risk_level": sample["risk_level"],
                "anomaly_score": sample["anomaly_score"],
                "threshold": sample["threshold"],
                "timestamp": f"T={sample['sample_index']}",
                "top_features": top3,
                "explanation": explanation,
            }
        )

    logger.info("  Enriched %d samples with explanations", len(enriched))
    return enriched


# ── 6. Explanation generation ─────────────────────────────────────────


def _generate_explanation(
    risk_level: str,
    sample_idx: int,
    top3: List[Dict[str, Any]],
) -> str:
    """Generate human-readable explanation from template.

    Args:
        risk_level: One of LOW/MEDIUM/HIGH/CRITICAL.
        sample_idx: Sample index for identification.
        top3: Top 3 contributing features with SHAP values.

    Returns:
        Human-readable explanation string.
    """
    template = TEMPLATES.get(risk_level, TEMPLATES["LOW"])

    if risk_level == "CRITICAL" and len(top3) >= 3:
        return template.format(
            idx=sample_idx,
            time=sample_idx,
            f1=top3[0]["feature"],
            v1=top3[0]["shap_value"],
            p1=top3[0]["contribution_pct"],
            f2=top3[1]["feature"],
            v2=top3[1]["shap_value"],
            p2=top3[1]["contribution_pct"],
            f3=top3[2]["feature"],
            v3=top3[2]["shap_value"],
            p3=top3[2]["contribution_pct"],
        )
    elif risk_level == "HIGH" and len(top3) >= 1:
        return template.format(
            idx=sample_idx,
            f1=top3[0]["feature"],
            p1=top3[0]["contribution_pct"],
        )
    elif risk_level == "MEDIUM" and len(top3) >= 1:
        return template.format(
            idx=sample_idx,
            f1=top3[0]["feature"],
        )
    else:
        return template.format(idx=sample_idx)


# ── 7. Visualization ─────────────────────────────────────────────────


def _plot_waterfall(
    sample: Dict[str, Any],
    shap_vals: np.ndarray,
    feature_names: List[str],
    baseline_threshold: float,
    output_path: Path,
) -> Path:
    """Render waterfall chart showing cumulative feature contributions.

    Args:
        sample: Enriched sample dict.
        shap_vals: SHAP values for this sample, shape (T, F).
        feature_names: List of feature names.
        baseline_threshold: Baseline prediction value.
        output_path: Path to save PNG.

    Returns:
        Path to saved chart.
    """
    per_feature = np.mean(shap_vals, axis=0)
    ranked_idx = np.argsort(np.abs(per_feature))[::-1][:TOP_K_FEATURES]

    names = [feature_names[i] for i in ranked_idx]
    values = [per_feature[i] for i in ranked_idx]

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    cumulative = baseline_threshold
    starts = []
    for v in values:
        starts.append(cumulative)
        cumulative += v

    colors = ["#d62728" if v > 0 else "#1f77b4" for v in values]
    ax.barh(range(len(names)), values, left=starts, color=colors)

    ax.axvline(
        x=baseline_threshold,
        color="gray",
        linestyle="--",
        linewidth=1,
        label=f"Baseline ({baseline_threshold:.4f})",
    )
    ax.axvline(
        x=cumulative,
        color="black",
        linestyle="-",
        linewidth=1.5,
        label=f"Final ({cumulative:.4f})",
    )

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Model Output")
    ax.set_title(f"Waterfall — Sample {sample['sample_index']} " f"({sample['risk_level']})")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    fig.savefig(output_path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_feature_importance_bar(
    importance_df: pd.DataFrame,
    top_k: int,
    output_path: Path,
) -> Path:
    """Render horizontal bar chart of top features by mean |SHAP|.

    Args:
        importance_df: DataFrame with feature, mean_abs_shap columns.
        top_k: Number of top features to show.
        output_path: Path to save PNG.

    Returns:
        Path to saved chart.
    """
    top = importance_df.head(top_k).copy()
    top = top.sort_values("mean_abs_shap", ascending=True)

    # Color by biometric (blue) vs network (red)
    colors = ["#1f77b4" if f in BIOMETRIC_COLUMNS else "#d62728" for f in top["feature"]]

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax.barh(top["feature"], top["mean_abs_shap"], color=colors)
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title(f"Top {top_k} Feature Importance (Global)")

    # Legend
    from matplotlib.patches import Patch

    legend_items = [
        Patch(facecolor="#d62728", label="Network features"),
        Patch(facecolor="#1f77b4", label="Biometric features"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_anomaly_timeline(
    anomaly_scores: List[float],
    baseline_threshold: float,
    incident_id: int,
    output_path: Path,
) -> Path:
    """Render anomaly score timeline with threshold crossing.

    Args:
        anomaly_scores: Consecutive anomaly scores for the incident window.
        baseline_threshold: Threshold for marking crossing point.
        incident_id: Incident identifier for title.
        output_path: Path to save PNG.

    Returns:
        Path to saved chart.
    """
    t = list(range(1, len(anomaly_scores) + 1))

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT * 0.7))
    ax.plot(t, anomaly_scores, "b-o", markersize=4, label="Anomaly Score")
    ax.axhline(
        y=baseline_threshold,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Threshold ({baseline_threshold:.4f})",
    )

    # Mark first crossing point
    for i, score in enumerate(anomaly_scores):
        if score > baseline_threshold:
            ax.axvline(
                x=t[i],
                color="orange",
                linestyle=":",
                alpha=0.7,
                label=f"Crossing at T={t[i]}",
            )
            break

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Anomaly Score")
    ax.set_title(f"Anomaly Timeline — Incident {incident_id}")
    ax.legend(fontsize=8)
    ax.set_xlim(0.5, len(t) + 0.5)
    plt.tight_layout()
    fig.savefig(output_path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _generate_all_charts(
    enriched_samples: List[Dict[str, Any]],
    shap_values: np.ndarray,
    feature_names: List[str],
    importance_df: pd.DataFrame,
    baseline_threshold: float,
    charts_dir: Path,
    max_waterfall: int,
    max_timeline: int,
) -> List[str]:
    """Generate all visualization charts.

    Args:
        enriched_samples: Enriched sample dicts.
        shap_values: SHAP values array.
        feature_names: Feature name list.
        importance_df: Feature importance DataFrame.
        baseline_threshold: Baseline threshold from Phase 4.
        charts_dir: Output directory for charts.
        max_waterfall: Max waterfall charts to generate.
        max_timeline: Max timeline charts to generate.

    Returns:
        List of generated chart filenames.
    """
    logger.info("── Generating visualizations ──")
    charts_dir.mkdir(parents=True, exist_ok=True)
    generated: List[str] = []

    # 1. Feature importance bar chart
    bar_path = charts_dir / "feature_importance.png"
    _plot_feature_importance_bar(importance_df, TOP_K_FEATURES, bar_path)
    generated.append("feature_importance.png")
    logger.info("  Saved: feature_importance.png")

    # 2. Waterfall charts for CRITICAL + HIGH samples
    crit_high = [
        (i, s) for i, s in enumerate(enriched_samples) if s["risk_level"] in ("CRITICAL", "HIGH")
    ]
    for idx, (shap_idx, sample) in enumerate(crit_high[:max_waterfall]):
        if shap_idx >= len(shap_values):
            break
        fname = f"waterfall_{sample['sample_index']}.png"
        _plot_waterfall(
            sample,
            shap_values[shap_idx],
            feature_names,
            baseline_threshold,
            charts_dir / fname,
        )
        generated.append(fname)
    logger.info("  Saved: %d waterfall charts", min(len(crit_high), max_waterfall))

    # 3. Timeline charts for first N incidents
    timeline_samples = [s for s in enriched_samples if s["risk_level"] in ("CRITICAL", "HIGH")]
    for idx, sample in enumerate(timeline_samples[:max_timeline]):
        # Extract TIMESTEPS consecutive anomaly scores around incident
        center = sample["sample_index"]
        window_scores = []
        for s in enriched_samples:
            if abs(s["sample_index"] - center) < TIMESTEPS:
                window_scores.append((s["sample_index"], s["anomaly_score"]))
        window_scores.sort()

        if len(window_scores) < 3:
            # Pad with baseline for sparse incidents
            scores = [sample["anomaly_score"]] * TIMESTEPS
        else:
            scores = [s[1] for s in window_scores[:TIMESTEPS]]

        fname = f"timeline_{sample['sample_index']}.png"
        _plot_anomaly_timeline(
            scores, baseline_threshold, sample["sample_index"], charts_dir / fname
        )
        generated.append(fname)
    logger.info("  Saved: %d timeline charts", min(len(timeline_samples), max_timeline))

    return generated


# ── 8. Artifact export ────────────────────────────────────────────────


def _export_artifacts(
    shap_values: np.ndarray,
    feature_names: List[str],
    enriched_samples: List[Dict[str, Any]],
    importance_df: pd.DataFrame,
    chart_files: List[str],
    level_counts: Dict[str, int],
    hw_info: Dict[str, str],
    duration_s: float,
    git_commit: str,
    output_dir: Path,
    shap_file: str,
    report_file: str,
    metadata_file: str,
) -> Dict[str, str]:
    """Export all Phase 5 artifacts and compute SHA-256 hashes.

    Args:
        shap_values: SHAP values array.
        feature_names: Feature name list.
        enriched_samples: Enriched sample dicts.
        importance_df: Feature importance DataFrame.
        chart_files: List of chart filenames.
        level_counts: Risk level counts.
        hw_info: Hardware info dict.
        duration_s: Pipeline duration in seconds.
        git_commit: Git commit hash.
        output_dir: Output directory.
        shap_file: SHAP values parquet filename.
        report_file: Explanation report JSON filename.
        metadata_file: Metadata JSON filename.

    Returns:
        Dict mapping artifact names to SHA-256 hashes.
    """
    logger.info("── Exporting Phase 5 artifacts ──")
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_hashes: Dict[str, str] = {}

    # 1. shap_values.parquet — aggregated per-feature SHAP
    per_sample_shap = np.mean(np.abs(shap_values), axis=1)  # (N, F)
    shap_df = pd.DataFrame(per_sample_shap, columns=feature_names)
    shap_path = output_dir / shap_file
    shap_df.to_parquet(shap_path, index=False)
    artifact_hashes[shap_file] = _compute_sha256(shap_path)
    logger.info("  Saved: %s (%s)", shap_file, shap_df.shape)

    # 2. explanation_report.json — human-readable explanations
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline": "phase5_explanation",
        "git_commit": git_commit,
        "total_explained": len(enriched_samples),
        "risk_level_counts": level_counts,
        "feature_importance": importance_df.head(TOP_K_FEATURES)[
            ["feature", "mean_abs_shap", "rank"]
        ].to_dict(orient="records"),
        "explanations": enriched_samples,
    }
    report_path = output_dir / report_file
    report_path.write_text(json.dumps(report, indent=2))
    artifact_hashes[report_file] = _compute_sha256(report_path)
    logger.info("  Saved: %s (%d explanations)", report_file, len(enriched_samples))

    # 3. explanation_metadata.json — SHA-256 + summary
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline": "phase5_explanation",
        "git_commit": git_commit,
        "hardware": hw_info,
        "duration_seconds": round(duration_s, 2),
        "samples_explained": len(enriched_samples),
        "shap_method": "GradientExplainer (integrated gradients fallback)",
        "background_samples": BACKGROUND_SAMPLES,
        "risk_level_counts": level_counts,
        "charts_generated": chart_files,
        "artifact_hashes": {
            k: {"sha256": v, "algorithm": "SHA-256"} for k, v in artifact_hashes.items()
        },
    }
    meta_path = output_dir / metadata_file
    meta_path.write_text(json.dumps(metadata, indent=2))
    logger.info("  Saved: %s", metadata_file)

    return artifact_hashes


# ── 9. Report generation ─────────────────────────────────────────────


def _generate_report(
    enriched_samples: List[Dict[str, Any]],
    importance_df: pd.DataFrame,
    level_counts: Dict[str, int],
    chart_files: List[str],
    baseline_threshold: float,
    hw_info: Dict[str, str],
    duration_s: float,
    git_commit: str,
) -> str:
    """Generate report_section_explanation.md content.

    Args:
        enriched_samples: Enriched sample dicts.
        importance_df: Feature importance DataFrame.
        level_counts: Risk level counts.
        chart_files: Generated chart filenames.
        baseline_threshold: Baseline threshold value.
        hw_info: Hardware info dict.
        duration_s: Pipeline duration.
        git_commit: Git commit hash.

    Returns:
        Markdown report string.
    """
    top10 = importance_df.head(TOP_K_FEATURES)
    feat_rows = ""
    for _, row in top10.iterrows():
        feat_rows += f"| {int(row['rank'])} | {row['feature']} " f"| {row['mean_abs_shap']:.6f} |\n"

    dist_rows = ""
    for level in RISK_LEVELS_NON_NORMAL:
        count = level_counts.get(level, 0)
        dist_rows += f"| {level} | {count} |\n"

    # Example explanations (first of each level)
    example_rows = ""
    seen_levels: set = set()
    for s in enriched_samples:
        if s["risk_level"] not in seen_levels:
            seen_levels.add(s["risk_level"])
            expl = s["explanation"][:120]
            example_rows += f"| {s['risk_level']} | {s['sample_index']} | {expl} |\n"

    wf_list = "\n".join(f"- `{f}`" for f in chart_files if f.startswith("waterfall_"))
    tl_list = "\n".join(f"- `{f}`" for f in chart_files if f.startswith("timeline_"))

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    report = f"""\
## 8.1 Explanation Engine — SHAP-Based Feature Attribution

This section presents the Phase 5 Explanation Engine results,
providing interpretable feature attributions for all non-NORMAL
risk samples using SHAP (SHapley Additive exPlanations).

### 8.1.1 Samples Explained

| Risk Level | Count |
|------------|-------|
{dist_rows}\
| **Total** | **{len(enriched_samples)}** |

Baseline threshold: {baseline_threshold:.6f}

### 8.1.2 Global Feature Importance (Top {TOP_K_FEATURES})

| Rank | Feature | Mean |SHAP| |
|------|---------|--------------|
{feat_rows}
Features are ranked by mean absolute SHAP value across all
explained samples. SHAP values aggregated over {TIMESTEPS}
timesteps per sliding window.

### 8.1.3 Explanation Examples

| Level | Sample | Explanation |
|-------|--------|-------------|
{example_rows}
### 8.1.4 Visualizations

**Feature importance bar chart:**
- `charts/feature_importance.png`

**Waterfall charts (CRITICAL/HIGH samples):**
{wf_list}

**Anomaly timeline charts:**
{tl_list}

### 8.1.5 SHAP Methodology

| Parameter | Value |
|-----------|-------|
| Method | GradientExplainer (integrated gradients fallback) |
| Background samples | {BACKGROUND_SAMPLES} (Normal class, training set) |
| Explained samples | {len(enriched_samples)} |
| Input shape | ({TIMESTEPS}, {N_FEATURES}) per window |
| Aggregation | mean(abs(SHAP)) over timesteps |
| Feature count | {N_FEATURES} |

### 8.1.6 Execution Details

| Parameter | Value |
|-----------|-------|
| Hardware | {hw_info.get('device', 'N/A')} |
| TensorFlow | {hw_info.get('tensorflow', 'N/A')} |
| Duration | {duration_s:.2f}s |
| Git commit | `{git_commit[:12]}` |

---

**Generated:** {timestamp}
**Pipeline:** Phase 5 Explanation Engine
**Artifacts:** data/phase5/
"""
    return report


# ── Main pipeline orchestrator ────────────────────────────────────────


def run_explanation_pipeline(
    config_path: Path = CONFIG_PATH,
) -> Dict[str, Any]:
    """Execute the Phase 5 Explanation Engine pipeline.

    Pipeline steps:
    1. Load & verify Phase 2/3/4 artifacts (SHA-256)
    2. Rebuild classification model from weights
    3. Filter non-NORMAL samples from risk_report.json
    4. Prepare SHAP background data (100 Normal training samples)
    5. Compute SHAP values (GradientExplainer)
    6. Rank feature importance (mean |SHAP|)
    7. Enrich samples with context + human-readable explanations
    8. Generate waterfall, bar, and timeline charts
    9. Export artifacts (parquet, JSON, charts, metadata)
    10. Generate §8.1 markdown report

    Args:
        config_path: Path to phase5_config.yaml.

    Returns:
        Summary dict with sample counts and artifact paths.
    """
    t0 = time.time()

    logger.info("═══════════════════════════════════════════════════")
    logger.info("  Phase 5 Explanation Engine")
    logger.info("═══════════════════════════════════════════════════")

    # Load config
    cfg = _load_yaml(config_path)
    data_cfg = cfg["data"]
    shap_cfg = cfg["shap"]
    out_cfg = cfg["output"]
    seed = cfg.get("random_state", 42)

    # Reproducibility
    np.random.seed(seed)  # noqa: NPY002
    tf.random.set_seed(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    rng = np.random.default_rng(seed)

    hw_info = _detect_hardware()
    git_commit = _get_git_commit()

    # Resolve paths
    p2_dir = PROJECT_ROOT / data_cfg["phase2_dir"]
    p3_dir = PROJECT_ROOT / data_cfg["phase3_dir"]
    p4_dir = PROJECT_ROOT / data_cfg["phase4_dir"]
    p2_meta = PROJECT_ROOT / data_cfg["phase2_metadata"]
    p3_meta = PROJECT_ROOT / data_cfg["phase3_metadata"]
    p4_meta = PROJECT_ROOT / data_cfg["phase4_metadata"]
    train_path = PROJECT_ROOT / data_cfg["phase1_train"]
    test_path = PROJECT_ROOT / data_cfg["phase1_test"]
    label_col = data_cfg["label_column"]
    output_dir = PROJECT_ROOT / out_cfg["output_dir"]
    charts_dir = output_dir / out_cfg["charts_dir"]

    # 1. Verify artifacts
    p2_metadata, p3_metadata, _ = _verify_artifacts(
        p2_meta, p3_meta, p4_meta, p2_dir, p3_dir, p4_dir
    )

    # 2. Rebuild model
    model = _rebuild_model(p2_metadata, p3_metadata, p3_dir)

    # 3. Load risk report + baseline
    risk_report = _load_risk_report(p4_dir)
    baseline = _load_baseline(p4_dir)

    # 4. Filter non-NORMAL samples
    max_explain = shap_cfg.get("max_explain_samples", MAX_EXPLAIN_SAMPLES)
    filtered, level_counts = _filter_non_normal(risk_report["sample_assessments"], max_explain, rng)

    # 5. Prepare data
    sample_indices = [s["sample_index"] for s in filtered]
    background = _prepare_background_data(
        train_path,
        label_col,
        shap_cfg.get("background_samples", BACKGROUND_SAMPLES),
        rng,
    )
    X_explain, _, feature_names = _prepare_explanation_data(test_path, label_col, sample_indices)

    # 6. Compute SHAP values
    shap_values = _compute_shap_values(model, background, X_explain)

    # 7. Feature importance
    importance_df, top_features = _compute_feature_importance(
        shap_values,
        feature_names,
        shap_cfg.get("top_features", TOP_K_FEATURES),
    )

    # 8. Enrich samples
    enriched = _enrich_samples(filtered, shap_values, feature_names)

    # 9. Generate charts
    chart_files = _generate_all_charts(
        enriched,
        shap_values,
        feature_names,
        importance_df,
        baseline["baseline_threshold"],
        charts_dir,
        shap_cfg.get("max_waterfall_charts", MAX_WATERFALL_CHARTS),
        shap_cfg.get("max_timeline_charts", MAX_TIMELINE_CHARTS),
    )

    duration_s = time.time() - t0

    # 10. Export artifacts
    _export_artifacts(
        shap_values=shap_values,
        feature_names=feature_names,
        enriched_samples=enriched,
        importance_df=importance_df,
        chart_files=chart_files,
        level_counts=level_counts,
        hw_info=hw_info,
        duration_s=duration_s,
        git_commit=git_commit,
        output_dir=output_dir,
        shap_file=out_cfg["shap_values_file"],
        report_file=out_cfg["explanation_report_file"],
        metadata_file=out_cfg["metadata_file"],
    )

    # 11. Generate markdown report
    report_md = _generate_report(
        enriched_samples=enriched,
        importance_df=importance_df,
        level_counts=level_counts,
        chart_files=chart_files,
        baseline_threshold=baseline["baseline_threshold"],
        hw_info=hw_info,
        duration_s=duration_s,
        git_commit=git_commit,
    )
    report_dir = PROJECT_ROOT / "results" / "phase0_analysis"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "report_section_explanation.md"
    report_path.write_text(report_md)
    logger.info("  Report saved: %s", report_path.name)

    logger.info("═══════════════════════════════════════════════════")
    logger.info("  Phase 5 complete — %.2fs", duration_s)
    logger.info("  Explained: %d samples, %d charts", len(enriched), len(chart_files))
    logger.info("═══════════════════════════════════════════════════")

    return {
        "samples_explained": len(enriched),
        "level_counts": level_counts,
        "top_features": top_features,
        "charts_generated": chart_files,
        "duration_s": round(duration_s, 2),
    }


# ── Entry point ───────────────────────────────────────────────────────


def main() -> None:
    """Entry point for Phase 5 Explanation Engine."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    run_explanation_pipeline()


if __name__ == "__main__":
    main()
