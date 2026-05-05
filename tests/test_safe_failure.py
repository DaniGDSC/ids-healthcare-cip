from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from module3_risk_scoring.module3_risk_scores import (
    _sanitise_features,
    classify_fusion,
)
from src.data_models import DataQuality, FusionClass
from src.risk_scorer import score_alert


def test_missing_device_context_defaults_safe() -> None:
    """Failure mode 1: no criticality/patchable metadata should still surface a strong alert."""
    result = score_alert(anomaly_score=0.7, device_context={}, event_context=None)
    assert result.should_surface is True
    assert result.adjusted_score >= 0.7


def test_mve_timeout_does_not_suppress() -> None:
    """Failure mode 2: risk scoring is independent of explanation generation."""
    context = {"criticality": "CRITICAL", "patchable": False}
    result = score_alert(anomaly_score=0.6, device_context=context, event_context=None)
    assert result.should_surface is True
    assert result.risk_multiplier >= 1.5


def test_unknown_attack_type_high_score() -> None:
    """Failure mode 3: a strong signal on a HIGH-criticality device should surface."""
    context = {"criticality": "HIGH", "patchable": True}
    result = score_alert(anomaly_score=0.85, device_context=context, event_context=None)
    assert result.should_surface is True
    assert result.threshold < 0.7


def test_feature_distortion_robustness() -> None:
    """Failure mode 4: extreme anomaly scores should still surface under normal context."""
    context = {"criticality": "MEDIUM", "patchable": True}
    result = score_alert(anomaly_score=0.95, device_context=context, event_context=None)
    assert result.should_surface is True
    assert result.adjusted_score >= 0.8


def test_non_patchable_priority() -> None:
    """Failure mode 5: unpatchable devices should receive elevated priority."""
    context = {"criticality": "HIGH", "patchable": False}
    result = score_alert(anomaly_score=0.55, device_context=context, event_context=None)
    assert result.risk_multiplier > 1.0
    assert result.should_surface is True


def test_critical_unpatchable_surfaces_in_maintenance_window() -> None:
    """Failure mode 6: ST-03 safety floor must hold on the maintenance-window path.

    A CRITICAL+unpatchable device is the IDS's responsibility alone — there
    is no compensating control. Even when an alert lands inside an approved
    maintenance window with traffic from a known vendor IP, the safety floor
    must override the confidence-reduction path and keep should_surface=True.
    """
    context = {"criticality": "CRITICAL", "patchable": False}
    event = {"is_maintenance_window": True, "is_known_vendor_ip": True}
    result = score_alert(anomaly_score=0.8, device_context=context, event_context=event)
    assert result.should_surface is True


def test_low_patchable_suppressed_in_maintenance_window() -> None:
    """Maintenance-window suppression still applies to non-critical devices.

    Confirms the safety-floor fix did not over-correct: a LOW+patchable
    device with a weak score should still be suppressed during a maintenance
    window with a known vendor IP.
    """
    context = {"criticality": "LOW", "patchable": True}
    event = {"is_maintenance_window": True, "is_known_vendor_ip": True}
    result = score_alert(anomaly_score=0.8, device_context=context, event_context=event)
    assert result.should_surface is False


# ── M1 fix: two-stage fusion classifier ──────────────────────────────────

def test_fusion_class_known_attack() -> None:
    """P_xgb >= 0.85 => KNOWN_ATTACK regardless of DAE."""
    arr = classify_fusion(
        c_track_a=np.array([0.9, 0.86]),
        c_track_b=np.array([0.0, 0.99]),
        xgb_threshold=0.05,
        dae_threshold=0.5,
    )
    assert all(c == "KNOWN_ATTACK" for c in arr)


def test_fusion_class_novel_anomaly() -> None:
    """Track B alone => NOVEL_ANOMALY."""
    arr = classify_fusion(
        c_track_a=np.array([0.01]),
        c_track_b=np.array([0.9]),
        xgb_threshold=0.05,
        dae_threshold=0.5,
    )
    assert arr[0] == "NOVEL_ANOMALY"


def test_fusion_class_confirmed_anomaly() -> None:
    """Both flag below high-confidence => CONFIRMED_ANOMALY."""
    arr = classify_fusion(
        c_track_a=np.array([0.5]),
        c_track_b=np.array([0.9]),
        xgb_threshold=0.05,
        dae_threshold=0.5,
    )
    assert arr[0] == "CONFIRMED_ANOMALY"


def test_fusion_class_benign() -> None:
    """Neither flags => BENIGN."""
    arr = classify_fusion(
        c_track_a=np.array([0.01]),
        c_track_b=np.array([0.1]),
        xgb_threshold=0.05,
        dae_threshold=0.5,
    )
    assert arr[0] == "BENIGN"


def test_score_alert_propagates_fusion_class() -> None:
    """`score_alert(fusion_class=...)` puts the class on the ScoredAlert."""
    result = score_alert(
        anomaly_score=0.7,
        device_context={"criticality": "HIGH", "patchable": True},
        event_context=None,
        fusion_class="KNOWN_ATTACK",
    )
    assert result.fusion_class == FusionClass.KNOWN_ATTACK


def test_score_alert_default_fusion_class_is_benign() -> None:
    """Calls without `fusion_class` keep the BENIGN default for back-compat."""
    result = score_alert(
        anomaly_score=0.7,
        device_context={"criticality": "HIGH", "patchable": True},
        event_context=None,
    )
    assert result.fusion_class == FusionClass.BENIGN


# ── M2 fix: data quality flag from sanitization ──────────────────────────

def test_sanitise_features_clean_input() -> None:
    """Clean input → all rows OK, no imputation."""
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    X_clean, flags = _sanitise_features(X)
    assert (flags == "OK").all()
    np.testing.assert_array_equal(X_clean, X)


def test_sanitise_features_marks_imputed_rows() -> None:
    """Rows with NaN/Inf are flagged IMPUTED_NAN; clean rows stay OK."""
    X = np.array([
        [1.0, 2.0],          # clean
        [np.nan, 3.0],       # imputed
        [np.inf, -np.inf],   # imputed
        [4.0, 5.0],          # clean
    ])
    X_clean, flags = _sanitise_features(X)
    assert flags.tolist() == ["OK", "IMPUTED_NAN", "IMPUTED_NAN", "OK"]
    assert np.isfinite(X_clean).all()
    np.testing.assert_array_equal(X_clean[0], [1.0, 2.0])
    np.testing.assert_array_equal(X_clean[3], [4.0, 5.0])


def test_score_alert_propagates_data_quality() -> None:
    """`score_alert(data_quality=...)` puts the flag on the ScoredAlert."""
    result = score_alert(
        anomaly_score=0.7,
        device_context={"criticality": "HIGH", "patchable": True},
        event_context=None,
        data_quality="IMPUTED_NAN",
    )
    assert result.data_quality == DataQuality.IMPUTED_NAN


def test_score_alert_default_data_quality_is_ok() -> None:
    """Calls without `data_quality` keep the OK default for back-compat."""
    result = score_alert(
        anomaly_score=0.7,
        device_context={"criticality": "HIGH", "patchable": True},
        event_context=None,
    )
    assert result.data_quality == DataQuality.OK


# ── GAP-A6: deterministic ATT&CK technique-ID lookup ─────────────────────

def test_attck_lookup_covers_all_5_alert_types() -> None:
    """Every alert type the generator can produce has an ATT&CK mapping."""
    from src.mve_generator import attck_for_alert_type
    for alert_type, expected_id in [
        ("T1", "T1071"),
        ("T2", "T1078"),
        ("T3", "T1021"),
        ("T4", "T1041"),
        ("T5", "T1565"),
    ]:
        tid, label = attck_for_alert_type(alert_type)
        assert tid == expected_id, f"{alert_type} → {tid} (expected {expected_id})"
        assert label, f"{alert_type} has no human-readable ATT&CK label"


def test_attck_lookup_unknown_alert_type_returns_empty() -> None:
    """Unknown alert types must return ('', '') — caller falls back gracefully."""
    from src.mve_generator import attck_for_alert_type
    assert attck_for_alert_type("T99") == ("", "")
    assert attck_for_alert_type("") == ("", "")


# ── GAP-A1: per-device-class threshold override ──────────────────────────

def test_device_class_override_lowers_infusion_pump_threshold() -> None:
    """Unpatchable infusion pump uses the 0.70 multiplier (not 0.80 default)."""
    result = score_alert(
        anomaly_score=0.36,   # 0.50*0.70=0.35 ≤ 0.36 ≤ 0.50*0.80=0.40
        device_context={
            "criticality": "MEDIUM",     # criticality table would say 0.95
            "patchable": False,
            "device_class": "infusion_pump",
        },
        event_context=None,
    )
    # Override threshold = 0.50 * 0.70 = 0.35; 0.36 surfaces.
    assert result.threshold == 0.35
    assert result.should_surface is True


def test_device_class_override_for_ehr_workstation() -> None:
    """EHR workstation patchable uses the 0.95 multiplier."""
    result = score_alert(
        anomaly_score=0.40,
        device_context={
            "criticality": "MEDIUM",
            "patchable": True,
            "device_class": "ehr_workstation",
        },
        event_context=None,
    )
    assert result.threshold == round(0.50 * 0.95, 4)


def test_device_class_unknown_falls_back_to_conservative() -> None:
    """Unrecognised device_class uses the 0.80 fallback (not the criticality table)."""
    result = score_alert(
        anomaly_score=0.0,
        device_context={
            "criticality": "LOW",        # criticality table = 1.00
            "patchable": True,
            "device_class": "some_widget",   # unknown
        },
        event_context=None,
    )
    # Unknown-device fallback 0.80, NOT criticality 1.00.
    assert result.threshold == round(0.50 * 0.80, 4)


def test_no_device_class_uses_criticality_table() -> None:
    """Empty/missing device_class preserves the original (criticality, patchable) lookup."""
    result = score_alert(
        anomaly_score=0.0,
        device_context={"criticality": "LOW", "patchable": True},  # no device_class
        event_context=None,
    )
    # Criticality table: ("LOW", True) → 1.00 → threshold 0.50.
    assert result.threshold == 0.50


# ── GAP-A3: SHAP stability check ─────────────────────────────────────────

def test_shap_stability_identical_features_is_one() -> None:
    """When perturbations leave top-k unchanged, Jaccard = 1.0."""
    from module4_explanations.module4_online_explainer import compute_shap_stability

    class _StubExplainer:
        """Returns identical SHAP values for all rows → top-k stable."""
        def shap_values(self, X):
            n = X.shape[0]
            sv = np.zeros((n, 5))
            sv[:, 0] = 1.0   # feature 0 dominates everywhere
            sv[:, 1] = 0.5
            sv[:, 2] = 0.3
            return sv

    score = compute_shap_stability(
        _StubExplainer(), np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        n_samples=5,
    )
    assert score == 1.0


def test_shap_stability_noisy_features_is_low() -> None:
    """Random per-row SHAP → top-k churns → Jaccard well below 1.0."""
    from module4_explanations.module4_online_explainer import compute_shap_stability

    class _NoisyExplainer:
        def __init__(self):
            self._rng = np.random.default_rng(seed=7)
        def shap_values(self, X):
            return self._rng.normal(size=(X.shape[0], 25))

    score = compute_shap_stability(
        _NoisyExplainer(), np.zeros(25), n_samples=10,
    )
    # Top-3 of 25 random features rarely repeats; expected well below 0.5.
    assert score < 0.5


def test_shap_context_default_stability_is_one() -> None:
    """SHAPContext default stability_score = 1.0 (no fragility)."""
    from src.data_models import SHAPContext
    ctx = SHAPContext(
        top_category="biometric",
        top_features=["Heart_rate"],
        shap_direction="elevated",
        confidence_from_shap="HIGH",
    )
    assert ctx.stability_score == 1.0


# ── GAP-A5: OperatorDecision schema validation ───────────────────────────

def test_operator_decision_valid_record_passes() -> None:
    from src.data_models import OperatorDecision
    rec = OperatorDecision(
        alert_id="a-001",
        operator_role="IT_generalist",
        operator_action_taken="approved_isolation",
        decision_time_seconds=12.5,
        timestamp="2026-05-05T18:00:00Z",
        operator_confidence=4,
    )
    rec.validate()  # should not raise


def test_operator_decision_missing_alert_id_raises() -> None:
    from src.data_models import OperatorDecision
    rec = OperatorDecision(
        alert_id="",   # empty
        operator_role="biomed_engineer",
        operator_action_taken="dismissed",
        decision_time_seconds=8.0,
        timestamp="2026-05-05T18:00:00Z",
    )
    import pytest as _pt
    with _pt.raises(ValueError, match="alert_id"):
        rec.validate()


def test_operator_decision_negative_time_raises() -> None:
    from src.data_models import OperatorDecision
    rec = OperatorDecision(
        alert_id="a-002",
        operator_role="nurse_manager",
        operator_action_taken="escalated",
        decision_time_seconds=-1.0,
        timestamp="2026-05-05T18:00:00Z",
    )
    import pytest as _pt
    with _pt.raises(ValueError, match="decision_time_seconds"):
        rec.validate()


def test_operator_decision_confidence_out_of_range_raises() -> None:
    from src.data_models import OperatorDecision
    rec = OperatorDecision(
        alert_id="a-003",
        operator_role="IT_generalist",
        operator_action_taken="logged",
        decision_time_seconds=2.0,
        timestamp="2026-05-05T18:00:00Z",
        operator_confidence=7,   # invalid Likert
    )
    import pytest as _pt
    with _pt.raises(ValueError, match="operator_confidence"):
        rec.validate()


# ── GAP-A7: device_class plumbed through Module 1 → parquet → Module 6 ──

def test_device_class_helper_returns_one_of_five_classes() -> None:
    """common.device_class.derive_device_class_row maps to the five-class taxonomy."""
    from common.device_class import derive_device_class_row
    valid = {"ventilator", "patient_monitor", "infusion_pump",
             "ehr_workstation", "other"}
    import pandas as pd
    # Synthetic ventilator-shaped row (Resp_Rate + SpO2 + 4 bio active)
    row = pd.Series({
        "Temp": 1.0, "SpO2": 1.0, "Pulse_Rate": 1.0, "Heart_rate": 1.0,
        "Resp_Rate": 1.0, "ST": 0.0, "Sport": 0.0, "SrcBytes": 0.0,
    })
    assert derive_device_class_row(row) == "ventilator"
    # Synthetic ehr-workstation-shaped row (no bio activity, network present)
    row2 = pd.Series({
        "Temp": 0.0, "SpO2": 0.0, "Pulse_Rate": 0.0, "Heart_rate": 0.0,
        "Resp_Rate": 0.0, "ST": 0.0, "Sport": 0.5, "SrcBytes": 0.5,
    })
    assert derive_device_class_row(row2) == "ehr_workstation"
    # Truly empty row → "other"
    row3 = pd.Series({})
    assert derive_device_class_row(row3) in valid


# ── GAP-A2: per-stakeholder MVE views ────────────────────────────────────

def _stub_mve():
    """Build a minimal MVEOutput for view-derivation tests."""
    from src.data_models import MVEOutput
    return MVEOutput(
        layer_1={
            "baseline_behavior": "device behaves normally",
            "deviation_description": "outbound bytes 3x baseline",
            "confidence_indicator": "confidence: HIGH",
        },
        layer_2={
            "affected_system": "infusion pump",
            "patient_care_impact": "drug delivery may be affected",
            "phi_exposure": "no PHI",
            "severity_label": "CRITICAL",
            "severity_rationale": "life-sustaining device under attack",
        },
        layer_3={
            "immediate_action": "Block outbound traffic at switch port for 10.4.12.0/24",
            "clinical_constraint": "DO NOT power off device. Switch-port block is SAFE.",
            "escalation_path": "Biomed Engineering on-call",
            "timeframe": "Act within 15 minutes",
        },
    )


def test_role_view_it_generalist_is_passthrough() -> None:
    """IT generalist sees the default wording — no transformation."""
    from src.mve_generator import derive_role_view
    base = _stub_mve()
    view = derive_role_view(base, role="IT_generalist", alert_type="T1")
    assert view.layer_1 == base.layer_1
    assert view.layer_3 == base.layer_3


def test_role_view_layer_2_is_invariant_across_roles() -> None:
    """INVARIANT 6 cross-role consistency: severity unchanged by view choice."""
    from src.mve_generator import derive_role_view
    base = _stub_mve()
    for role in ("IT_generalist", "biomed_engineer", "nurse_manager"):
        v = derive_role_view(base, role=role, alert_type="T5")
        assert v.layer_2 == base.layer_2, f"layer_2 drifted for role={role}"
        assert v.layer_2["severity_label"] == "CRITICAL"


def test_role_view_biomed_uses_biomed_verbs() -> None:
    """Biomed view rewrites immediate_action to verify/document/coordinate."""
    from src.mve_generator import derive_role_view
    base = _stub_mve()
    view = derive_role_view(base, role="biomed_engineer", alert_type="T5")
    action = view.layer_3["immediate_action"].lower()
    assert "verify" in action or "document" in action
    # And it must NOT instruct biomed to push network policy.
    assert "block port at switch" not in action
    assert "firewall rule" not in action


def test_role_view_nurse_uses_clinical_verbs() -> None:
    """Nurse-manager view stays in clinical-impact framing."""
    from src.mve_generator import derive_role_view
    base = _stub_mve()
    view = derive_role_view(base, role="nurse_manager", alert_type="T5")
    action = view.layer_3["immediate_action"].lower()
    assert "monitor" in action or "verify" in action or "document" in action
    # Nurse role MUST NOT see infrastructure or device-power instructions.
    assert "block port at switch" not in action
    assert "firewall rule" not in action
    assert "power-cycle" not in action


def test_role_view_clinical_constraint_preserved_across_roles() -> None:
    """INVARIANT 7: DO NOT wording must survive role transformation."""
    from src.mve_generator import derive_role_view
    base = _stub_mve()
    for role in ("IT_generalist", "biomed_engineer", "nurse_manager"):
        v = derive_role_view(base, role=role, alert_type="T5")
        assert "DO NOT" in v.layer_3["clinical_constraint"], (
            f"DO NOT lost in {role} view"
        )


def test_role_view_unknown_role_falls_back_to_it_generalist() -> None:
    """Unknown role → safe default (IT generalist)."""
    from src.mve_generator import derive_role_view
    base = _stub_mve()
    view = derive_role_view(base, role="janitor", alert_type="T1")
    assert view.layer_3 == base.layer_3   # fell back to passthrough


def test_operator_role_enum_values() -> None:
    """The three values are stable strings consumers can match against."""
    from src.data_models import OperatorRole
    assert OperatorRole.IT_GENERALIST.value == "IT_generalist"
    assert OperatorRole.BIOMED_ENGINEER.value == "biomed_engineer"
    assert OperatorRole.NURSE_MANAGER.value == "nurse_manager"


def test_module5_render_views_returns_three_keyed_views() -> None:
    """Module 5's render_views_for_alert wraps derive_role_view for all 3 roles."""
    from module5_responses.module5_pipeline import render_views_for_alert
    base = _stub_mve()
    views = render_views_for_alert(base, alert_type="T5")
    assert set(views) == {"IT_generalist", "biomed_engineer", "nurse_manager"}
    # Each view shares layer_2 (cross-role consistency).
    sev = {v.layer_2["severity_label"] for v in views.values()}
    assert sev == {"CRITICAL"}
    # Each view preserves DO NOT.
    for role, v in views.items():
        assert "DO NOT" in v.layer_3["clinical_constraint"], role


def test_test_phase1_parquet_carries_device_class_after_a7_closure() -> None:
    """The on-disk test parquet must expose the GAP-A7 schema upgrade."""
    from pathlib import Path
    import pandas as pd
    p = Path(__file__).resolve().parent.parent / "data/processed/test_phase1.parquet"
    if not p.exists():
        import pytest as _pt
        _pt.skip(f"{p} not present in this environment")
    df = pd.read_parquet(p, columns=["row_id", "device_class"])
    assert "row_id" in df.columns
    assert "device_class" in df.columns
    assert len(df) > 0
    assert df["device_class"].isin(
        ["ventilator", "patient_monitor", "infusion_pump",
         "ehr_workstation", "other"]
    ).all(), "Unexpected device_class label in test parquet"
