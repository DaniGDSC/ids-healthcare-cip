"""Module 3 config constants — invariants."""
from __future__ import annotations

from module3_risk_scoring.config import (
    BIOMETRIC_FEATURES,
    CIA_SCORE,
    CIA_THREATS,
    DAE_BINARY_THRESHOLD,
    DATA_SENSITIVITY,
    DEFAULT_DEVICE_TIER,
    DEVICE_TIERS,
    FEATURE_ACTIVE_EPSILON,
    RESPONSE_MAPPING,
    RISK_THRESHOLDS,
    SIGMA_THRESHOLD,
    WEIGHTS,
)


def test_weights_sum_to_one():
    total = sum(WEIGHTS.values())
    assert abs(total - 1.0) < 1e-9, f"weights must sum to 1, got {total}"


def test_weights_keys():
    assert set(WEIGHTS.keys()) == {"w1", "w2", "w3", "w4"}


def test_risk_thresholds_invariants():
    """Threshold invariants (Sprint 1.4 — invariant-pinning).

    Pin *properties* of the table, not the specific numeric values, so
    a legitimate threshold recalibration (Sprint 5 retrain, future
    formula version) doesn't require editing this test:

      - strictly monotonic descending (no equal-tier boundaries)
      - every threshold in [0, 1]
      - CRITICAL is the highest, lowest is non-NORMAL (NORMAL is the
        implicit ``default=`` tier, not in the table)
    """
    thresholds = [t for t, _ in RISK_THRESHOLDS]
    labels = [name for _, name in RISK_THRESHOLDS]

    # Strictly descending — no equal boundaries
    assert thresholds == sorted(thresholds, reverse=True)
    assert len(set(thresholds)) == len(thresholds)
    # All in [0, 1]
    assert all(0.0 <= t <= 1.0 for t in thresholds)
    # CRITICAL is the first (highest) entry; LOW is the last (lowest).
    # NORMAL is the implicit default — must NOT appear in the table.
    assert labels[0] == "CRITICAL"
    assert labels[-1] == "LOW"
    assert "NORMAL" not in labels


def test_risk_threshold_covers_canonical_severities():
    """Every canonical severity that ``assign_risk_levels`` can emit
    above the NORMAL default must have a threshold entry. NORMAL is
    the default, so it does NOT need a threshold."""
    labels = [name for _, name in RISK_THRESHOLDS]
    canonical = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
    assert set(labels) == canonical, (
        f"Threshold labels {set(labels)} != canonical surfaced "
        f"severities {canonical}"
    )


def test_device_tiers_in_unit_interval():
    for tier, score in DEVICE_TIERS.items():
        assert 0.0 <= score <= 1.0, f"tier {tier} score {score} out of [0, 1]"


def test_default_device_tier_known():
    assert DEFAULT_DEVICE_TIER in DEVICE_TIERS


def test_data_sensitivity_monotonic():
    """phi_realtime > phi_stored > device_telemetry > non_sensitive."""
    sens = DATA_SENSITIVITY
    assert sens["phi_realtime"] > sens["phi_stored"]
    assert sens["phi_stored"] > sens["device_telemetry"]
    assert sens["device_telemetry"] > sens["non_sensitive"]


def test_cia_threats_have_full_triad():
    for cat, profile in CIA_THREATS.items():
        assert set(profile.keys()) == {"C", "I", "A"}, (
            f"category {cat} missing CIA dim"
        )
        for dim, score in profile.items():
            assert 0.0 <= score <= 1.0


def test_response_mapping_complete():
    """Every risk level has a response entry."""
    expected = {"CRITICAL", "HIGH", "MEDIUM", "LOW", "NORMAL"}
    assert set(RESPONSE_MAPPING.keys()) == expected


def test_response_mapping_critical_fastest():
    """CRITICAL must have the shortest max_response_min."""
    crit = RESPONSE_MAPPING["CRITICAL"]["max_response_min"]
    for tier in ("HIGH", "MEDIUM", "LOW"):
        assert RESPONSE_MAPPING[tier]["max_response_min"] >= crit


def test_biometric_features_non_empty():
    assert len(BIOMETRIC_FEATURES) > 0
    assert all(isinstance(f, str) for f in BIOMETRIC_FEATURES)


def test_sigma_threshold_positive():
    assert SIGMA_THRESHOLD > 0


def test_dae_binary_threshold_midpoint():
    assert DAE_BINARY_THRESHOLD == 0.5


def test_feature_active_epsilon_small():
    assert 0 < FEATURE_ACTIVE_EPSILON < 1.0


def test_cia_score_lookup_pre_computed():
    """Pre-computed lookup must match the runtime max(C, I, A) × base_tier."""
    base = DEVICE_TIERS[DEFAULT_DEVICE_TIER]
    for cat, profile in CIA_THREATS.items():
        expected = base * max(profile.values())
        assert abs(CIA_SCORE[cat] - expected) < 1e-9
