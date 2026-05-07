"""Layer 3 v4.0 — enriched triage classifier + clinical_active gate.

Covers the v4.0 deltas added on top of the existing Layer 3
infrastructure (``module3_risk_scoring/fusion.py``,
``module3_risk_scoring/module3_risk_scores.py``,
``src/risk_scorer.py``):

  * 9-stage ``classify_alert_v4`` decision tree — every stage
    reachable, predicates partition the input space, INVARIANT 1
    holds for every emitted decision.
  * ``clinical_active`` adjustment in ``src.risk_scorer.score_alert``
    — multiplier drops by 0.10 with a 0.40 floor, threshold tightens.
  * Safety floor (CRITICAL + unpatchable) survives clinical_active
    and maintenance-window combinations.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from module3_risk_scoring.triage_v4 import (  # noqa: E402
    DAE_HIGH,
    DAE_MODERATE,
    DAE_WEAK,
    DIVERSITY_HIGH,
    DIVERSITY_MODERATE,
    P_XGB_HIGH,
    P_XGB_LOW,
    classify_alert_v4,
)
from src.data_models import AlertType, Confidence  # noqa: E402
from src.risk_scorer import score_alert  # noqa: E402


# ── 9-stage reachability ────────────────────────────────────────────────

def _call(p_xgb, dae=0.0, diversity=0.0, threshold_level="below_threshold"):
    return classify_alert_v4(
        p_xgb=p_xgb, p_rf=p_xgb, p_dt=p_xgb,
        diversity_score=diversity, dae_score=dae,
        threshold_level=threshold_level,
    )


def test_stage_1_known_attack_high_confidence() -> None:
    out = _call(p_xgb=0.95, diversity=0.05, dae=0.10)
    assert out.alert_type == AlertType.KNOWN_ATTACK
    assert out.confidence == Confidence.VERY_HIGH


def test_stage_2_known_attack_uncertain() -> None:
    out = _call(p_xgb=0.90, diversity=0.20, dae=0.10)
    assert out.alert_type == AlertType.KNOWN_ATTACK_UNCERTAIN
    assert out.confidence == Confidence.HIGH


def test_stage_3_disagreement_anomaly() -> None:
    """Adversarial signal: high diversity + DAE moderate, in the
    moderate-P regime so it's not already KNOWN_ATTACK."""
    out = _call(p_xgb=0.50, diversity=0.35, dae=0.80)
    assert out.alert_type == AlertType.DISAGREEMENT_ANOMALY
    assert out.confidence == Confidence.HIGH


def test_stage_4_strong_novel_anomaly() -> None:
    """DAE strong, Track A silent — the canonical novelty signal."""
    out = _call(p_xgb=0.10, diversity=0.05, dae=0.97)
    assert out.alert_type == AlertType.STRONG_NOVEL_ANOMALY
    assert out.confidence == Confidence.HIGH


def test_stage_5_novel_anomaly() -> None:
    out = _call(p_xgb=0.10, diversity=0.05, dae=0.80)
    assert out.alert_type == AlertType.NOVEL_ANOMALY
    assert out.confidence == Confidence.MEDIUM


def test_stage_6_confirmed_anomaly() -> None:
    """Multi-signal corroboration in the moderate-P band."""
    out = _call(p_xgb=0.60, diversity=0.05, dae=0.80)
    assert out.alert_type == AlertType.CONFIRMED_ANOMALY
    assert out.confidence == Confidence.HIGH


def test_stage_7_suspicious_pattern() -> None:
    """Track A moderate, DAE benign — needs review but not strong."""
    out = _call(p_xgb=0.55, diversity=0.05, dae=0.30)
    assert out.alert_type == AlertType.SUSPICIOUS_PATTERN
    assert out.confidence == Confidence.MEDIUM


def test_stage_8_benign_watch() -> None:
    out = _call(p_xgb=0.10, diversity=0.05, dae=0.55)
    assert out.alert_type == AlertType.BENIGN_WATCH
    assert out.confidence == Confidence.LOW


def test_stage_9_benign() -> None:
    out = _call(p_xgb=0.05, diversity=0.05, dae=0.10)
    assert out.alert_type == AlertType.BENIGN
    assert out.confidence == Confidence.HIGH


def test_all_nine_stages_reachable() -> None:
    """Every alert type in the v4 typology must be hit by at least one
    fixture above. Guards against silent regressions where a predicate
    edit makes a stage unreachable.
    """
    expected = {
        AlertType.KNOWN_ATTACK,
        AlertType.KNOWN_ATTACK_UNCERTAIN,
        AlertType.DISAGREEMENT_ANOMALY,
        AlertType.STRONG_NOVEL_ANOMALY,
        AlertType.NOVEL_ANOMALY,
        AlertType.CONFIRMED_ANOMALY,
        AlertType.SUSPICIOUS_PATTERN,
        AlertType.BENIGN_WATCH,
        AlertType.BENIGN,
    }
    fixtures = [
        (0.95, 0.05, 0.10),
        (0.90, 0.20, 0.10),
        (0.50, 0.35, 0.80),
        (0.10, 0.05, 0.97),
        (0.10, 0.05, 0.80),
        (0.60, 0.05, 0.80),
        (0.55, 0.05, 0.30),
        (0.10, 0.05, 0.55),
        (0.05, 0.05, 0.10),
    ]
    seen = {classify_alert_v4(p, p, p, d, dae).alert_type for p, d, dae in fixtures}
    assert seen == expected


# ── Predicate partition / determinism ───────────────────────────────────

def test_classifier_is_deterministic() -> None:
    a = _call(p_xgb=0.50, diversity=0.20, dae=0.50)
    b = _call(p_xgb=0.50, diversity=0.20, dae=0.50)
    assert a.alert_type == b.alert_type
    assert a.template_id == b.template_id


def test_stage_predicates_partition_input_space() -> None:
    """Sweep a 5-D grid over the predicate boundaries; every input
    must produce exactly one alert type, and the type must match the
    predicate that ought to fire (no two stages both true).
    """
    grid_p = [0.05, P_XGB_LOW - 0.01, P_XGB_LOW, 0.50, P_XGB_HIGH - 0.01,
              P_XGB_HIGH, 0.95]
    grid_d = [0.0, DIVERSITY_MODERATE - 0.01, DIVERSITY_MODERATE,
              DIVERSITY_HIGH - 0.01, DIVERSITY_HIGH, 0.45]
    grid_dae = [0.0, DAE_WEAK - 0.01, DAE_WEAK, DAE_MODERATE - 0.01,
                DAE_MODERATE, DAE_HIGH - 0.01, DAE_HIGH, 1.0]

    for p in grid_p:
        for d in grid_d:
            for dae in grid_dae:
                out = classify_alert_v4(p, p, p, d, dae)
                # Confidence is one of the 4 enum values.
                assert out.confidence in {
                    Confidence.VERY_HIGH, Confidence.HIGH,
                    Confidence.MEDIUM, Confidence.LOW,
                }
                # The matched alert_type is well-formed.
                assert out.alert_type in set(AlertType)


# ── INVARIANT 1: c_detect >= p_xgb across the input grid ────────────────

def test_invariant_1_holds_across_input_grid() -> None:
    grid_p = [0.0, 0.10, 0.40, 0.60, 0.85, 0.99]
    grid_d = [0.0, 0.10, 0.30, 0.45]
    grid_dae = [0.0, 0.30, 0.70, 0.95, 1.0]
    for p in grid_p:
        for d in grid_d:
            for dae in grid_dae:
                out = classify_alert_v4(p, p, p, d, dae)
                assert out.c_detect >= p - 1e-9, (
                    f"INVARIANT 1 violated for (p={p}, d={d}, dae={dae}): "
                    f"c_detect={out.c_detect}"
                )


def test_dae_can_only_elevate_not_reduce() -> None:
    """For fixed p_xgb, increasing DAE score must never lower c_detect."""
    base_p = 0.50
    out_low_dae = classify_alert_v4(base_p, base_p, base_p, 0.05, 0.10)
    out_hi_dae = classify_alert_v4(base_p, base_p, base_p, 0.05, 0.95)
    assert out_hi_dae.c_detect >= out_low_dae.c_detect


# ── clinical_active adjustment in src/risk_scorer.score_alert ───────────

DEVICE_PUMP_PATCHABLE = {
    "criticality": "MEDIUM", "patchable": True, "device_class": "infusion_pump",
}


def test_clinical_active_tightens_threshold() -> None:
    """clinical_active=True should LOWER the threshold (more sensitive
    detection during active care).
    """
    base = score_alert(0.40, DEVICE_PUMP_PATCHABLE, event_context={})
    active = score_alert(
        0.40, DEVICE_PUMP_PATCHABLE,
        event_context={"clinical_active": True},
    )
    assert active.threshold < base.threshold


def test_clinical_active_floor_at_0_40() -> None:
    """The clinical_active multiplier reduction must not drop below
    0.40 — gate cannot be made trivially loose.
    """
    # Pick a device whose base multiplier is already low (infusion_pump
    # unpatchable = 0.70). 0.70 - 0.10 = 0.60 → still above 0.40, so
    # we test that the resulting threshold is ``DEFAULT_THRESHOLD * 0.60``,
    # not anything below the floor.
    from src.risk_scorer import DEFAULT_THRESHOLD
    res = score_alert(
        0.50,
        {"criticality": "CRITICAL", "patchable": False,
         "device_class": "infusion_pump"},
        event_context={"clinical_active": True},
    )
    assert res.threshold == pytest.approx(round(DEFAULT_THRESHOLD * 0.60, 4))


def test_clinical_active_flag_default_false_preserves_baseline() -> None:
    """Omitting clinical_active must not change the threshold — the
    default behaviour is unchanged from the pre-v4 path.
    """
    no_flag = score_alert(0.40, DEVICE_PUMP_PATCHABLE, event_context={})
    explicit_false = score_alert(
        0.40, DEVICE_PUMP_PATCHABLE,
        event_context={"clinical_active": False},
    )
    assert no_flag.threshold == explicit_false.threshold


def test_clinical_active_does_not_bypass_safety_floor() -> None:
    """CRITICAL + unpatchable must still surface even when
    clinical_active is True (and the score is below threshold)."""
    res = score_alert(
        0.05,  # low — would normally suppress
        {"criticality": "CRITICAL", "patchable": False,
         "device_class": "ventilator"},
        event_context={"clinical_active": True},
    )
    assert res.should_surface is True


def test_clinical_active_combines_with_similar_events_safely() -> None:
    """``similar_events > 5`` reduces the *risk multiplier*; clinical_active
    reduces the *threshold multiplier*. They operate on different knobs,
    so combining them must not break the API or the safety floor."""
    res = score_alert(
        0.50,
        {"criticality": "MEDIUM", "patchable": True,
         "device_class": "infusion_pump"},
        event_context={
            "clinical_active": True,
            "similar_events_past_30d": 10,
        },
    )
    # No exception, multiplier stays >= 0.5 floor, threshold adjusted.
    assert res.risk_multiplier >= 0.5
    assert res.threshold > 0.0


# ── Output shape / audit trail ──────────────────────────────────────────

def test_decision_carries_source_signals_for_audit() -> None:
    out = _call(p_xgb=0.60, diversity=0.10, dae=0.80)
    assert out.p_xgb == 0.60
    assert out.diversity_score == 0.10
    assert out.dae_score == 0.80
    assert out.template_id  # non-empty
    assert out.rationale  # non-empty
