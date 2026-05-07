"""ARCHITECTURE.md Step [8] — Context Enrichment contract tests.

Locks the four invariants of ``src/context_enrichment.py``:

* I1 ``patchable`` is **required** — no silent default. Reverting to
     ``alert.get("patchable", True)`` would re-introduce the safety-floor
     bug the doc says was fixed.
* I2 UNKNOWN-device fail-safe: when ``device_class`` is missing, the
     enriched alert defaults to ``UNKNOWN / HIGH / tier_2 / patchable=False``
     and emits ``DEVICE_NOT_IN_INVENTORY`` in ``warning_flags``.
* I3 The legacy field name ``device_patchable`` is accepted as an alias
     for ``patchable`` (back-compat with ``evaluation_alerts.json``).
* I4 ``configs/composite_risk_weights.yaml`` weights sum to 1.0 and
     ``configs/risk_adaptive_thresholds.yaml`` is loadable.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.context_enrichment import (
    CLINICAL_TIER_WEIGHTS,
    DEFAULT_DEVICE_CLASS_TO_TIER,
    MissingRequiredField,
    enrich_alert_context,
    score_alert_from_dict,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── I1: patchable is required ─────────────────────────────────────────


def test_patchable_missing_raises_missing_required_field():
    with pytest.raises(MissingRequiredField):
        enrich_alert_context({
            "risk_score": 0.5,
            "device_class": "ventilator",
            "device_criticality": "CRITICAL",
        })


def test_patchable_present_passes_through():
    e = enrich_alert_context({
        "patchable": False,
        "device_class": "ventilator",
        "device_criticality": "CRITICAL",
    })
    assert e["patchable"] is False


def test_legacy_device_patchable_alias_accepted():
    """I3: ``evaluation_alerts.json`` uses ``device_patchable``; the
    enrichment module must read either name."""
    e = enrich_alert_context({
        "device_patchable": True,
        "device_class": "infusion_pump",
        "device_criticality": "CRITICAL",
    })
    assert e["patchable"] is True


# ── I2: UNKNOWN-device fail-safe ──────────────────────────────────────


def test_unknown_device_fails_conservative():
    e = enrich_alert_context({"patchable": True, "risk_score": 0.5})
    assert e["device_class"] == "UNKNOWN"
    assert e["patchable"] is False, (
        "UNKNOWN-device fail-safe must override input patchable=True "
        "with patchable=False (conservative)"
    )
    assert e["device_criticality"] == "HIGH"
    assert e["clinical_tier"] == "tier_2_high_clinical"
    assert "DEVICE_NOT_IN_INVENTORY" in (e.get("warning_flags") or [])


def test_unknown_device_extends_existing_warning_flags():
    e = enrich_alert_context({
        "patchable": True,
        "warning_flags": ["EXISTING_FLAG"],
    })
    flags = e.get("warning_flags") or []
    assert "EXISTING_FLAG" in flags
    assert "DEVICE_NOT_IN_INVENTORY" in flags


# ── Clinical tier mapping ─────────────────────────────────────────────


def test_clinical_tier_lookup_for_known_device_classes():
    for device, expected_tier in DEFAULT_DEVICE_CLASS_TO_TIER.items():
        e = enrich_alert_context({
            "patchable": True,
            "device_class": device,
            "device_criticality": "MEDIUM",
        })
        assert e["clinical_tier"] == expected_tier
        assert e["clinical_tier_weight"] == CLINICAL_TIER_WEIGHTS[expected_tier]


def test_clinical_tier_weights_match_doc():
    """ARCHITECTURE.md Step [8] D_clinical_tier weights table."""
    assert CLINICAL_TIER_WEIGHTS["tier_1_life_critical"] == 1.0
    assert CLINICAL_TIER_WEIGHTS["tier_2_high_clinical"] == 0.8
    assert CLINICAL_TIER_WEIGHTS["tier_3_moderate"] == 0.5
    assert CLINICAL_TIER_WEIGHTS["tier_4_supportive"] == 0.3
    assert CLINICAL_TIER_WEIGHTS["tier_5_administrative"] == 0.1


# ── I4: configs/*.yaml are loadable + weights sum to 1.0 ──────────────


def test_composite_weights_yaml_is_loadable():
    from module3_risk_scoring.module3_risk_scores import load_composite_weights
    w = load_composite_weights()
    assert set(w.keys()) == {"w1", "w2", "w3", "w4"}
    assert abs(sum(w.values()) - 1.0) < 1e-6, (
        f"Composite weights must sum to 1.0; got {sum(w.values())}"
    )


def test_tier_boundaries_yaml_is_loadable_and_descending():
    from module3_risk_scoring.module3_risk_scores import load_tier_boundaries
    boundaries = load_tier_boundaries()
    mins = [m for m, _ in boundaries]
    assert mins == sorted(mins, reverse=True), (
        "Tier boundaries must be descending (CRITICAL ≥ HIGH ≥ MEDIUM)"
    )


def test_risk_adaptive_thresholds_yaml_loaded():
    """``src/risk_scorer.py`` constants must reflect the YAML when present."""
    yaml_path = PROJECT_ROOT / "configs" / "risk_adaptive_thresholds.yaml"
    assert yaml_path.exists(), "configs/risk_adaptive_thresholds.yaml missing"
    from src.risk_scorer import (
        DEFAULT_THRESHOLD,
        _THRESHOLD_MULT,
        _THRESHOLD_MULT_BY_DEVICE,
    )
    # CRITICAL+unpatchable must be ≤0.70 (≥30% reduction per spec)
    assert _THRESHOLD_MULT[("CRITICAL", False)] <= 0.70
    # ventilator+unpatchable must be ≤0.70
    assert _THRESHOLD_MULT_BY_DEVICE[("ventilator", False)] <= 0.70
    assert 0.0 < DEFAULT_THRESHOLD < 1.0


# ── score_alert_from_dict end-to-end ──────────────────────────────────


def test_score_alert_from_dict_runs_with_required_fields():
    """Using the new ``score_alert_from_dict`` should produce a valid
    ``ScoredAlert`` for an alert that has all required fields."""
    out = score_alert_from_dict({
        "patchable": False,
        "device_class": "ventilator",
        "device_criticality": "CRITICAL",
        "risk_score": 0.85,
    })
    # CRITICAL+unpatchable+high score must surface (safety floor).
    assert out.should_surface is True


def test_score_alert_from_dict_raises_on_missing_patchable():
    with pytest.raises(MissingRequiredField):
        score_alert_from_dict({
            "device_class": "ventilator",
            "device_criticality": "CRITICAL",
            "risk_score": 0.85,
        })
