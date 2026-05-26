"""Module 3 — risk-scoring configuration constants.

Single source of truth for every constant the rest of Module 3 depends
on. Importing these from one place lets ``tests/test_rq1_*`` pin the
canonical values and ``tools/diagnostics/*`` reuse them without
duplicating literals.

Composite formula:
    R = w1·C_detect + w2·D_crit + w3·S_data + w4·D_clinical_tier
"""

from __future__ import annotations

from common.phi import BIOMETRIC_COLUMNS

# ── Risk formula weights ───────────────────────────────────────────────
WEIGHTS: dict[str, float] = {"w1": 0.40, "w2": 0.25, "w3": 0.15, "w4": 0.20}

# Risk level thresholds — 3 boundaries, 4 tiers
RISK_THRESHOLDS: list[tuple[float, str]] = [
    (0.80, "CRITICAL"),
    (0.60, "HIGH"),
    (0.40, "MEDIUM"),
]

# ── Biometric features ────────────────────────────────────────────────
# Stable list ordering for downstream callers; backed by the canonical
# PHI set in common/phi.py.
BIOMETRIC_FEATURES: list[str] = sorted(BIOMETRIC_COLUMNS)

# Clinical-acuity sigma threshold — fraction of biometric features
# beyond ±SIGMA_THRESHOLD sigma contributes to D_clinical_tier. The
# 1.5σ choice matches NEWS2 "moderate deviation" tier convention for
# vital-sign monitoring; documented to defend the magic number.
SIGMA_THRESHOLD: float = 1.5

# Non-zero threshold for feature-active detection (used by compute_s_data
# bio_active / net_active fractions — a feature is treated as "present"
# if |value| > FEATURE_ACTIVE_EPSILON).
FEATURE_ACTIVE_EPSILON: float = 0.01

# ── DAE binary-decision threshold for fusion quadrant analysis ────────
# DAE.predict_proba clips to [0, 1] with min-max scaling from benign
# training errors; 0.5 is the midpoint between "in-distribution" (~0)
# and "out-of-distribution" (~1).
DAE_BINARY_THRESHOLD: float = 0.5

# ── CIA threat profile per attack category ────────────────────────────
CIA_THREATS: dict[str, dict[str, float]] = {
    "Spoofing":        {"C": 0.6, "I": 0.9, "A": 0.3},
    "Data Alteration": {"C": 0.3, "I": 1.0, "A": 0.2},
}

# ── Device criticality tiers ──────────────────────────────────────────
DEVICE_TIERS: dict[str, float] = {
    "life_sustaining":  1.0,   # infusion pumps, ventilators
    "vital_monitoring": 0.8,   # ECG, pulse oximeter
    "diagnostic":       0.5,   # blood pressure, temperature
    "auxiliary":        0.3,   # environmental sensors
}
DEFAULT_DEVICE_TIER: str = "vital_monitoring"   # WUSTL-EHMS-2020 default

# Pre-computed per-category D_crit lookup (max(C, I, A) × base_tier).
_BASE_TIER: float = DEVICE_TIERS[DEFAULT_DEVICE_TIER]
CIA_SCORE: dict[str, float] = {
    cat: _BASE_TIER * max(t.values()) for cat, t in CIA_THREATS.items()
}
DEFAULT_CIA_SCORE: float = _BASE_TIER * 0.5   # fallback for unknown category

# ── Data sensitivity classification ───────────────────────────────────
DATA_SENSITIVITY: dict[str, float] = {
    "phi_realtime":     1.0,   # real-time vital signs (SpO2, HR, BP)
    "phi_stored":       0.7,   # stored patient records
    "device_telemetry": 0.4,   # network flow metadata
    "non_sensitive":    0.1,   # timestamps, flags
}

# ── Response mapping per risk level ───────────────────────────────────
RESPONSE_MAPPING: dict[str, dict] = {
    "CRITICAL": {
        "action": "Immediate network isolation + page physician + escalate to CISO",
        "max_response_min": 5,
        "auto_actions": ["isolate_device", "page_oncall", "snapshot_forensics"],
    },
    "HIGH": {
        "action": "Active investigation + isolate segment + notify biomedical engineering",
        "max_response_min": 15,
        "auto_actions": ["isolate_segment", "notify_biomed", "create_ticket"],
    },
    "MEDIUM": {
        "action": "Flag for review + enhanced monitoring + notify security team",
        "max_response_min": 60,
        "auto_actions": ["enhanced_logging", "notify_soc"],
    },
    "LOW": {
        "action": "Log for audit + review at next shift",
        "max_response_min": 480,
        "auto_actions": ["log_event"],
    },
    "NORMAL": {
        "action": "No action — routine logging",
        "max_response_min": 0,
        "auto_actions": [],
    },
}


__all__ = [
    "WEIGHTS",
    "RISK_THRESHOLDS",
    "BIOMETRIC_FEATURES",
    "SIGMA_THRESHOLD",
    "FEATURE_ACTIVE_EPSILON",
    "DAE_BINARY_THRESHOLD",
    "CIA_THREATS",
    "DEVICE_TIERS",
    "DEFAULT_DEVICE_TIER",
    "CIA_SCORE",
    "DEFAULT_CIA_SCORE",
    "DATA_SENSITIVITY",
    "RESPONSE_MAPPING",
]
