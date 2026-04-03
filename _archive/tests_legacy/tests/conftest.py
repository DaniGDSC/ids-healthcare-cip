"""Shared test fixtures for cross-phase integration testing.

Provides lightweight mock data that mimics the artifact flow
between pipeline phases without requiring real model training.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pytest


# ── Constants ───────────────────────────────────────────────────────

N_FEATURES = 24
N_TRAIN = 200
N_TEST = 50
TIMESTEPS = 20
ATTACK_RATE = 0.125
FEATURE_NAMES = [
    "SrcBytes", "DstBytes", "SrcLoad", "DstLoad",
    "SIntPkt", "DIntPkt", "SIntPktAct",
    "sMaxPktSz", "dMaxPktSz", "sMinPktSz",
    "Dur", "TotBytes", "Load", "pSrcLoss", "pDstLoss", "Packet_num",
    "Temp", "SpO2", "Pulse_Rate", "SYS", "DIA", "Heart_rate",
    "Resp_Rate", "ST",
]

BIOMETRIC_COLUMNS = ["Temp", "SpO2", "Pulse_Rate", "SYS", "DIA", "Heart_rate", "Resp_Rate", "ST"]


# ── Phase 1 fixtures ───────────────────────────────────────────────

@pytest.fixture
def rng() -> np.random.RandomState:
    """Seeded random state for reproducible tests."""
    return np.random.RandomState(42)


@pytest.fixture
def mock_features(rng: np.random.RandomState) -> np.ndarray:
    """Mock feature matrix (N_TEST, N_FEATURES)."""
    return rng.standard_normal((N_TEST, N_FEATURES)).astype(np.float32)


@pytest.fixture
def mock_labels(rng: np.random.RandomState) -> np.ndarray:
    """Mock binary labels with ~12.5% attack rate."""
    n_attack = int(N_TEST * ATTACK_RATE)
    labels = np.zeros(N_TEST, dtype=np.int32)
    labels[:n_attack] = 1
    rng.shuffle(labels)
    return labels


@pytest.fixture
def mock_attack_categories(mock_labels: np.ndarray) -> np.ndarray:
    """Mock attack category strings aligned with labels."""
    cats = np.array(["normal"] * len(mock_labels), dtype=object)
    attack_idx = np.where(mock_labels == 1)[0]
    for i, idx in enumerate(attack_idx):
        cats[idx] = "Spoofing" if i % 2 == 0 else "Data Alteration"
    return cats


@pytest.fixture
def feature_names() -> List[str]:
    """24 feature names (post-variance filtering)."""
    return list(FEATURE_NAMES)


# ── Phase 2 fixtures ───────────────────────────────────────────────

@pytest.fixture
def mock_windows(rng: np.random.RandomState) -> np.ndarray:
    """Mock windowed data (N_windows, TIMESTEPS, N_FEATURES)."""
    n_windows = N_TEST - TIMESTEPS + 1
    return rng.standard_normal((n_windows, TIMESTEPS, N_FEATURES)).astype(np.float32)


@pytest.fixture
def mock_window_labels(mock_labels: np.ndarray) -> np.ndarray:
    """Mock windowed labels with any_attack strategy."""
    n_windows = N_TEST - TIMESTEPS + 1
    labels = np.zeros(n_windows, dtype=np.int32)
    for i in range(n_windows):
        if mock_labels[i:i + TIMESTEPS].max() > 0:
            labels[i] = 1
    return labels


@pytest.fixture
def mock_attention_vectors(rng: np.random.RandomState) -> np.ndarray:
    """Mock 128-D attention context vectors."""
    n_windows = N_TEST - TIMESTEPS + 1
    return rng.standard_normal((n_windows, 128)).astype(np.float32)


# ── Phase 4 fixtures ───────────────────────────────────────────────

@pytest.fixture
def mock_anomaly_scores(rng: np.random.RandomState) -> np.ndarray:
    """Mock sigmoid outputs (anomaly scores)."""
    n_windows = N_TEST - TIMESTEPS + 1
    return rng.uniform(0, 1, n_windows).astype(np.float32)


@pytest.fixture
def mock_thresholds(mock_anomaly_scores: np.ndarray) -> np.ndarray:
    """Mock dynamic thresholds."""
    return np.full_like(mock_anomaly_scores, 0.5)


@pytest.fixture
def mock_baseline() -> Dict[str, Any]:
    """Mock Phase 4 baseline config."""
    return {
        "median": 0.18,
        "mad": 0.025,
        "baseline_threshold": 0.255,
        "mad_multiplier": 3.0,
        "n_normal_samples": 175,
        "n_attention_dims": 128,
    }


@pytest.fixture
def mock_risk_result() -> Dict[str, Any]:
    """Single mock risk assessment result with all Phase 4 fields."""
    return {
        "sample_index": 0,
        "anomaly_score": 0.75,
        "threshold": 0.5,
        "distance": 0.25,
        "risk_level": "HIGH",
        "attention_flag": False,
        "clinical_severity": 4,
        "clinical_severity_name": "EMERGENT",
        "response_time_minutes": 15,
        "device_action": "isolate_network",
        "patient_safety_flag": True,
        "clinical_rationale": "Base: HIGH on generic_iomt_sensor | PATIENT SAFETY FLAG",
        "alert_emit": True,
        "alert_reason": "escalation",
        "alert_aggregated_count": 0,
        "scenario": "high_threat",
        "cia_scores": {"C": 0.24, "I": 0.72, "A": 0.24},
        "cia_max_dimension": "I",
        "cia_modifier": 0.72,
        "attack_category": "Spoofing",
        "explanation": {
            "level": "attention_and_shap",
            "timestep_importance": [0.04, 0.05, 0.03, 0.06, 0.82],
            "top_features": [
                {"feature": "DIntPkt", "importance": 0.082},
                {"feature": "TotBytes", "importance": 0.034},
            ],
        },
        "stakeholder_views": {
            "soc_analyst": {"alert_level": "HIGH", "threat_type": "Spoofing"},
            "clinician": {"message": "Active threat detected", "urgency": "Emergent"},
            "ciso": {"incident_severity": 4, "compliance_impacts": []},
            "biomed_engineer": {"device_type": "generic_iomt_sensor"},
        },
    }
