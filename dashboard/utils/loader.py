"""Artifact loader with Streamlit caching.

Loads JSON, Parquet, and model artifacts from the RA-X-IoMT pipeline
with TTL-based caching to avoid redundant disk I/O.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
MODELS_DIR: Path = PROJECT_ROOT / "models"

# Artifact paths
PATHS = {
    "risk_report": DATA_DIR / "phase4" / "risk_report.json",
    "baseline_config": DATA_DIR / "phase4" / "baseline_config.json",
    "explanation_report": DATA_DIR / "phase5" / "explanation_report.json",
    "shap_values": DATA_DIR / "phase5" / "shap_values.parquet",
    "monitoring_log": DATA_DIR / "phase7" / "monitoring_log.json",
    "metrics_v1": DATA_DIR / "phase3" / "metrics_report.json",
    "metrics_v2": DATA_DIR / "phase3" / "metrics_wustl_v2.json",
    "threshold_analysis": DATA_DIR / "phase3" / "threshold_analysis.json",
    "ablation_results": DATA_DIR / "phase3" / "ablation_results_v2.json",
    "best_hyperparams": DATA_DIR / "phase3" / "best_hyperparams_v2.json",
    "diagnosis": DATA_DIR / "phase3" / "diagnosis_after.json",
    "preprocessing": DATA_DIR / "processed" / "preprocessing_report.json",
    "detection_metadata": DATA_DIR / "phase2" / "detection_metadata.json",
    "train_parquet": DATA_DIR / "processed" / "train_phase1.parquet",
    "test_parquet": DATA_DIR / "processed" / "test_phase1.parquet",
    "scaler": MODELS_DIR / "scalers" / "robust_scaler.pkl",
    "detection_weights": DATA_DIR / "phase2" / "detection_model.weights.h5",
    "classification_weights": DATA_DIR / "phase3" / "classification_model_v2.weights.h5",
    "notification_log": DATA_DIR / "phase6" / "notification_log.json",
    "delivery_report": DATA_DIR / "phase6" / "delivery_report.json",
}


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load a JSON file, returning None on failure."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning("Failed to load %s: %s", path.name, e)
        return None


@st.cache_data(ttl=60)
def load_risk_report() -> Optional[Dict[str, Any]]:
    """Load Phase 4 risk report with sample assessments."""
    return _load_json(PATHS["risk_report"])


@st.cache_data(ttl=60)
def load_baseline_config() -> Optional[Dict[str, Any]]:
    """Load Phase 4 MAD baseline configuration."""
    return _load_json(PATHS["baseline_config"])


@st.cache_data(ttl=60)
def load_explanation_report() -> Optional[Dict[str, Any]]:
    """Load Phase 5 SHAP explanation report."""
    return _load_json(PATHS["explanation_report"])


@st.cache_data(ttl=60)
def load_shap_values() -> Optional[np.ndarray]:
    """Load SHAP values parquet as numpy array (N, 20, 29)."""
    path = PATHS["shap_values"]
    try:
        df = pd.read_parquet(path)
        n_samples = len(df) // 20
        return df.values.reshape(n_samples, 20, -1)
    except Exception as e:
        logger.warning("Failed to load SHAP values: %s", e)
        return None


@st.cache_data(ttl=60)
def load_monitoring_log() -> Optional[List[Dict[str, Any]]]:
    """Load Phase 7 monitoring state transitions."""
    return _load_json(PATHS["monitoring_log"])


@st.cache_data(ttl=300)
def load_metrics_v1() -> Optional[Dict[str, Any]]:
    """Load Phase 3 v1 baseline metrics."""
    return _load_json(PATHS["metrics_v1"])


@st.cache_data(ttl=300)
def load_metrics_v2() -> Optional[Dict[str, Any]]:
    """Load Phase 3 v2 final test metrics."""
    return _load_json(PATHS["metrics_v2"])


@st.cache_data(ttl=300)
def load_threshold_analysis() -> Optional[Dict[str, Any]]:
    """Load threshold sensitivity analysis."""
    return _load_json(PATHS["threshold_analysis"])


@st.cache_data(ttl=300)
def load_ablation_results() -> Optional[Dict[str, Any]]:
    """Load ablation study results."""
    return _load_json(PATHS["ablation_results"])


@st.cache_data(ttl=300)
def load_preprocessing_report() -> Optional[Dict[str, Any]]:
    """Load Phase 1 preprocessing report."""
    return _load_json(PATHS["preprocessing"])


@st.cache_data(ttl=300)
def load_diagnosis() -> Optional[Dict[str, Any]]:
    """Load training diagnosis report."""
    return _load_json(PATHS["diagnosis"])


@st.cache_data(ttl=300)
def load_detection_metadata() -> Optional[Dict[str, Any]]:
    """Load Phase 2 detection engine metadata."""
    return _load_json(PATHS["detection_metadata"])


@st.cache_data(ttl=300)
def load_best_hyperparams() -> Optional[Dict[str, Any]]:
    """Load best hyperparameters from Bayesian tuning."""
    return _load_json(PATHS["best_hyperparams"])


@st.cache_data(ttl=300)
def load_feature_names() -> List[str]:
    """Load the 29 feature names from preprocessing report."""
    report = load_preprocessing_report()
    if report and "output" in report:
        return report["output"]["feature_names"]
    return [
        "SrcBytes", "DstBytes", "SrcLoad", "DstLoad", "SrcGap", "DstGap",
        "SIntPkt", "DIntPkt", "SIntPktAct", "DIntPktAct", "sMaxPktSz",
        "dMaxPktSz", "sMinPktSz", "dMinPktSz", "Dur", "Trans", "TotBytes",
        "Load", "pSrcLoss", "pDstLoss", "Packet_num", "Temp", "SpO2",
        "Pulse_Rate", "SYS", "DIA", "Heart_rate", "Resp_Rate", "ST",
    ]


@st.cache_data(ttl=60)
def load_delivery_report() -> Optional[List[Dict[str, Any]]]:
    """Load Phase 6 delivery report."""
    data = _load_json(PATHS["delivery_report"])
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "deliveries" in data:
        return data["deliveries"]
    return data


def get_artifact_status() -> Dict[str, bool]:
    """Check existence of all pipeline artifacts."""
    return {name: path.exists() for name, path in PATHS.items()}


def get_last_modified(artifact: str) -> Optional[str]:
    """Get last modification time of an artifact file."""
    path = PATHS.get(artifact)
    if path and path.exists():
        mtime = os.path.getmtime(path)
        from datetime import datetime, timezone
        dt = datetime.fromtimestamp(mtime, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    return None
