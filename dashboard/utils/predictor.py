"""Inference pipeline for uploaded CSV data.

Handles: HIPAA PII check, scaling, reshaping, prediction, and risk scoring.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from dashboard.utils.loader import (
    PATHS,
    PROJECT_ROOT,
    load_baseline_config,
    load_feature_names,
)

logger = logging.getLogger(__name__)

PII_COLUMNS = {
    "srcaddr", "dstaddr", "sport", "dport", "srcmac", "dstmac",
    "dir", "flgs", "patient_id", "name", "ssn", "mrn", "dob",
    "address", "phone", "email",
}

TIMESTEPS = 20
N_FEATURES = 29


@st.cache_resource
def load_model() -> Optional[tf.keras.Model]:
    """Load the classification model (cached, load once).

    Rebuilds the CNN-BiLSTM-Attention architecture and loads v2 weights.
    """
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))

    try:
        from src.phase2_detection_engine.phase2.assembler import DetectionModelAssembler
        from src.phase2_detection_engine.phase2.attention_builder import (
            AttentionBuilder,
            BahdanauAttention,  # noqa: F401
        )
        from src.phase2_detection_engine.phase2.bilstm_builder import BiLSTMBuilder
        from src.phase2_detection_engine.phase2.cnn_builder import CNNBuilder

        builders = [
            CNNBuilder(filters_1=64, filters_2=128, kernel_size=3, activation="relu", pool_size=2),
            BiLSTMBuilder(units_1=128, units_2=64, dropout_rate=0.3),
            AttentionBuilder(units=128),
        ]
        assembler = DetectionModelAssembler(timesteps=TIMESTEPS, n_features=N_FEATURES, builders=builders)
        det = assembler.assemble()

        x = det.output
        x = tf.keras.layers.Dense(64, activation="relu", name="dense_head")(x)
        x = tf.keras.layers.Dropout(0.3, name="drop_head")(x)
        x = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)
        model = tf.keras.Model(det.input, x, name="classification_engine_v2")

        weights_path = PATHS["classification_weights"]
        if weights_path.exists():
            model.load_weights(str(weights_path))
            logger.info("Model loaded: %d params", model.count_params())
        else:
            st.warning(f"Model weights not found: {weights_path}")
            return None

        return model
    except Exception as e:
        logger.error("Model load failed: %s", e)
        return None


@st.cache_resource
def load_scaler() -> Optional[Any]:
    """Load the fitted RobustScaler."""
    path = PATHS["scaler"]
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.warning("Scaler load failed: %s", e)
        return None


def detect_pii_columns(df: pd.DataFrame) -> List[str]:
    """Detect potential PII columns in uploaded data."""
    found = []
    for col in df.columns:
        if col.lower().strip() in PII_COLUMNS:
            found.append(col)
    return found


def preprocess_upload(df: pd.DataFrame) -> Optional[np.ndarray]:
    """Preprocess uploaded CSV: select features, scale, reshape.

    Args:
        df: Raw uploaded DataFrame.

    Returns:
        Windowed array of shape (N_windows, 20, 29) or None on failure.
    """
    feature_names = load_feature_names()
    scaler = load_scaler()

    # Match feature columns (case-insensitive)
    col_map = {c.lower(): c for c in df.columns}
    matched = []
    missing = []
    for feat in feature_names:
        key = feat.lower()
        if key in col_map:
            matched.append(col_map[key])
        else:
            missing.append(feat)

    if missing:
        st.error(f"Missing features ({len(missing)}): {', '.join(missing[:10])}")
        return None

    X = df[matched].values.astype(np.float32)

    if scaler is not None:
        X = scaler.transform(X)

    if len(X) < TIMESTEPS:
        st.error(f"Need at least {TIMESTEPS} rows, got {len(X)}")
        return None

    # Sliding window reshape
    n_windows = len(X) - TIMESTEPS + 1
    windows = np.array([X[i : i + TIMESTEPS] for i in range(n_windows)])
    return windows


def predict(X_windows: np.ndarray) -> np.ndarray:
    """Run model inference on windowed data.

    Args:
        X_windows: Array of shape (N, 20, 29).

    Returns:
        Probability array of shape (N,).
    """
    model = load_model()
    if model is None:
        return np.zeros(len(X_windows))
    probs = model.predict(X_windows, batch_size=128, verbose=0)
    return probs.ravel()


def score_risk(
    anomaly_scores: np.ndarray,
    baseline: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Apply MAD-based risk scoring.

    Args:
        anomaly_scores: Prediction probabilities.
        baseline: MAD baseline config. Loaded from artifact if None.

    Returns:
        List of per-sample risk assessments.
    """
    if baseline is None:
        baseline = load_baseline_config()

    if baseline is None:
        median = 0.129
        mad = 0.025
        threshold = 0.204
    else:
        median = baseline["median"]
        mad = baseline["mad"]
        threshold = baseline["baseline_threshold"]

    mad_safe = max(mad, 1e-8)
    results = []
    for i, score in enumerate(anomaly_scores):
        distance = float(score) - threshold
        if distance < 0:
            level = "NORMAL"
        elif distance < 0.5 * mad_safe:
            level = "LOW"
        elif distance < 1.0 * mad_safe:
            level = "MEDIUM"
        elif distance < 2.0 * mad_safe:
            level = "HIGH"
        else:
            level = "CRITICAL"

        results.append({
            "sample_index": i,
            "anomaly_score": round(float(score), 6),
            "threshold": round(threshold, 6),
            "distance": round(distance, 6),
            "risk_level": level,
        })
    return results
