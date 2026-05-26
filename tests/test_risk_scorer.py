"""Safety-floor regression tests for the C2 Risk-Adaptive Scoring Engine.

The non-negotiable invariants pinned here (from CLAUDE.md +
research_spec.yaml.component_2) are the most important correctness rules
in the system:

* CRITICAL + unpatchable devices ALWAYS surface — the IDS is the only
  compensating control.
* Maintenance + known-vendor IP reduces confidence (no longer a hard
  suppression after EA-02).
* Newly enrolled devices (baseline_days < 14) get a 30% threshold cut.
* similar_events_past_30d > 5 reduces risk_multiplier by 0.20, never
  below 0.50.

These tests guard against regression in those rules. Any change that
breaks one of them is a safety-impacting bug.
"""
from __future__ import annotations

import pytest

from src.data_models import ScoredAlert
from src.risk_scorer import (
    DEFAULT_THRESHOLD,
    get_threshold,
    score_alert,
    score_alert_static,
)

# ── Test fixtures ──────────────────────────────────────────────────────


def _dc(criticality: str = "LOW", patchable: bool = True, **extra) -> dict:
    """Build a minimal device_context dict."""
    return {"criticality": criticality, "patchable": patchable, **extra}


# ── Safety floor: CRITICAL + unpatchable ───────────────────────────────


def test_critical_unpatchable_always_surfaces_at_low_score():
    """ST-03 regression: CRITICAL + unpatchable surfaces even at 0.20."""
    out = score_alert(0.20, _dc("CRITICAL", patchable=False), None)
    assert out.should_surface is True


def test_critical_unpatchable_surfaces_at_zero_score():
    """Edge case: even a near-zero anomaly score on CRITICAL+unpatchable
    must surface — the safety floor is unconditional."""
    out = score_alert(0.01, _dc("CRITICAL", patchable=False), None)
    assert out.should_surface is True


def test_critical_unpatchable_risk_multiplier_at_least_1_5():
    """Spec: risk_multiplier >= 1.5 for CRITICAL + unpatchable."""
    out = score_alert(0.5, _dc("CRITICAL", patchable=False), None)
    assert out.risk_multiplier >= 1.5


def test_critical_unpatchable_threshold_reduced_at_least_30_percent():
    """Spec: threshold lowered >= 30% from DEFAULT_THRESHOLD."""
    out = score_alert(0.5, _dc("CRITICAL", patchable=False), None)
    assert out.threshold <= DEFAULT_THRESHOLD * 0.70 + 1e-9


def test_critical_patchable_lower_multiplier_than_unpatchable():
    """Patchable CRITICAL devices have lower risk multiplier than
    unpatchable (both still > 1.0)."""
    unpatch = score_alert(0.5, _dc("CRITICAL", patchable=False), None)
    patch = score_alert(0.5, _dc("CRITICAL", patchable=True), None)
    assert patch.risk_multiplier < unpatch.risk_multiplier
    assert patch.risk_multiplier > 1.0


# ── Safety floor: LOW + patchable default ──────────────────────────────


def test_low_patchable_default_multiplier_is_one():
    """Spec: LOW + patchable → risk_multiplier == 1.0."""
    out = score_alert(0.5, _dc("LOW", patchable=True), None)
    assert out.risk_multiplier == 1.0


def test_low_patchable_uses_default_threshold():
    """Spec: LOW + patchable → threshold == DEFAULT_THRESHOLD (0.50)."""
    out = score_alert(0.5, _dc("LOW", patchable=True), None)
    assert out.threshold == DEFAULT_THRESHOLD


def test_low_patchable_below_threshold_does_not_surface():
    """0.40 anomaly on LOW+patchable → 0.40 < 0.50 → should_surface=False."""
    out = score_alert(0.40, _dc("LOW", patchable=True), None)
    assert out.should_surface is False


def test_low_patchable_above_threshold_surfaces():
    out = score_alert(0.60, _dc("LOW", patchable=True), None)
    assert out.should_surface is True


# ── EA-02: maintenance window + known vendor IP ────────────────────────


def test_maintenance_window_known_vendor_reduced_confidence():
    """EA-02: both flags set → risk_multiplier=0.5, reason populated."""
    event = {"is_maintenance_window": True, "is_known_vendor_ip": True}
    out = score_alert(0.80, _dc("HIGH"), event)
    assert out.risk_multiplier == 0.5
    assert out.suppression_reason is not None
    assert "maintenance" in out.suppression_reason.lower()


def test_maintenance_window_known_vendor_surfaces_above_threshold():
    """0.80 × 0.5 = 0.40 < 0.50 default → does NOT surface."""
    event = {"is_maintenance_window": True, "is_known_vendor_ip": True}
    out = score_alert(0.80, _dc("HIGH"), event)
    assert out.adjusted_score == 0.40
    assert out.should_surface is False


def test_maintenance_window_alone_does_not_reduce():
    """is_maintenance_window WITHOUT is_known_vendor_ip → no EA-02 path."""
    event = {"is_maintenance_window": True, "is_known_vendor_ip": False}
    out = score_alert(0.80, _dc("HIGH", patchable=True), event)
    # Normal HIGH+patchable multiplier 1.10 applies; no suppression_reason.
    assert out.suppression_reason is None
    assert out.risk_multiplier > 1.0


def test_known_vendor_alone_does_not_reduce():
    """is_known_vendor_ip WITHOUT is_maintenance_window → no EA-02 path."""
    event = {"is_maintenance_window": False, "is_known_vendor_ip": True}
    out = score_alert(0.80, _dc("HIGH"), event)
    assert out.suppression_reason is None


# ── Adaptive rule: similar events past 30d ─────────────────────────────


def test_similar_events_above_threshold_reduces_multiplier():
    """similar_events_past_30d > 5 → reduce risk_multiplier by 0.20."""
    event = {"similar_events_past_30d": 10}
    out_with = score_alert(0.5, _dc("HIGH", patchable=True), event)
    out_without = score_alert(0.5, _dc("HIGH", patchable=True), None)
    assert out_with.risk_multiplier == pytest.approx(
        out_without.risk_multiplier - 0.20, abs=1e-4
    )


def test_similar_events_at_threshold_does_not_reduce():
    """similar_events_past_30d == 5 → does NOT reduce (>5 required)."""
    event = {"similar_events_past_30d": 5}
    out = score_alert(0.5, _dc("HIGH", patchable=True), event)
    baseline = score_alert(0.5, _dc("HIGH", patchable=True), None)
    assert out.risk_multiplier == baseline.risk_multiplier


def test_similar_events_floor_at_0_5():
    """Even with massive similar-event count, multiplier never < 0.5."""
    event = {"similar_events_past_30d": 1000}
    out = score_alert(0.5, _dc("LOW", patchable=True), event)
    assert out.risk_multiplier >= 0.5


# ── TM-02: baseline quarantine for new devices ─────────────────────────


def test_baseline_quarantine_new_device_lowers_threshold():
    """baseline_days < 14 → threshold reduced by 30%."""
    event = {"baseline_days": 7}
    out = score_alert(0.5, _dc("LOW", patchable=True), event)
    # Default threshold 0.50 × 0.70 = 0.35
    assert out.threshold == pytest.approx(DEFAULT_THRESHOLD * 0.70, abs=1e-4)


def test_baseline_quarantine_at_14_days_does_not_apply():
    """baseline_days == 14 → no quarantine (< 14 required)."""
    event = {"baseline_days": 14}
    out = score_alert(0.5, _dc("LOW", patchable=True), event)
    assert out.threshold == DEFAULT_THRESHOLD


def test_baseline_quarantine_default_90_days_no_change():
    """baseline_days defaults to 90 → no quarantine."""
    out = score_alert(0.5, _dc("LOW", patchable=True), {})
    assert out.threshold == DEFAULT_THRESHOLD


# ── score_alert_static (baseline path) ─────────────────────────────────


def test_score_alert_static_above_default_threshold():
    assert score_alert_static(0.51)["surfaced"] is True


def test_score_alert_static_at_default_threshold_not_surfaced():
    """Strict > comparison: 0.50 → False."""
    assert score_alert_static(0.50)["surfaced"] is False


def test_score_alert_static_below_threshold():
    assert score_alert_static(0.10)["surfaced"] is False


def test_score_alert_static_returns_dict():
    out = score_alert_static(0.5)
    assert isinstance(out, dict)
    assert "surfaced" in out


# ── get_threshold ──────────────────────────────────────────────────────


def test_get_threshold_critical_unpatchable_below_low_patchable():
    """Acceptance-test M7 invariant: CRITICAL+unpatchable threshold strictly
    lower (more sensitive) than LOW+patchable.
    """
    crit_unp = get_threshold("CRITICAL", False)
    low_patch = get_threshold("LOW", True)
    assert crit_unp < low_patch


def test_get_threshold_case_insensitive():
    """Criticality strings get upper-cased internally."""
    assert get_threshold("critical", False) == get_threshold("CRITICAL", False)
    assert get_threshold("Low", True) == get_threshold("LOW", True)


def test_get_threshold_unknown_criticality_defaults_to_1_0_mult():
    """Unknown tier → multiplier 1.0 → threshold == DEFAULT_THRESHOLD."""
    assert get_threshold("ARTIFICIAL_TIER", True) == DEFAULT_THRESHOLD


# ── score_alert API robustness ─────────────────────────────────────────


def test_score_alert_handles_none_event_context():
    """event_context=None must not crash."""
    out = score_alert(0.5, _dc("MEDIUM"), None)
    assert isinstance(out, ScoredAlert)


def test_score_alert_handles_missing_criticality_key():
    """Missing criticality defaults to LOW per source."""
    out = score_alert(0.5, {"patchable": True}, None)
    assert out.risk_multiplier == 1.0  # LOW+patchable default


def test_score_alert_handles_missing_patchable_key():
    """Missing patchable defaults to True per source."""
    out = score_alert(0.5, {"criticality": "LOW"}, None)
    assert out.risk_multiplier == 1.0


def test_score_alert_rounds_fields_to_four_decimals():
    out = score_alert(0.12345678, _dc("HIGH", patchable=True), None)
    # adjusted_score, threshold, risk_multiplier all rounded to 4dp.
    assert len(str(out.adjusted_score).split(".")[-1]) <= 4
    assert len(str(out.threshold).split(".")[-1]) <= 4


def test_score_alert_adjusted_score_capped_at_one():
    """Multiplier could push score > 1.0; gets capped at 1.0."""
    out = score_alert(0.95, _dc("CRITICAL", patchable=False), None)
    # 0.95 × 1.5 = 1.425 → capped at 1.0
    assert out.adjusted_score <= 1.0


# ── ScoredAlert shape ─────────────────────────────────────────────────


def test_score_alert_returns_scored_alert_dataclass():
    out = score_alert(0.5, _dc(), None)
    assert isinstance(out, ScoredAlert)
    assert hasattr(out, "adjusted_score")
    assert hasattr(out, "threshold")
    assert hasattr(out, "should_surface")
    assert hasattr(out, "risk_multiplier")
    assert hasattr(out, "suppression_reason")
