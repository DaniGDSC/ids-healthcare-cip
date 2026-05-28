"""Tests for Phase-1 upgrades (observation baseline, MITRE gloss,
reversibility metadata, parametrized actions).

These tests exercise the *new* surfaces. End-to-end CLI regeneration of
artifacts is covered by re-running the Phase-0 baseline tool after a
Phase-1 PR and comparing against the recorded floors.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


# ── Phase 1.1 — observation_phrase ─────────────────────────────────


def test_observation_phrase_with_known_baseline():
    """observation_phrase formats a deviation clause when baseline exists."""
    from module4_explanations import feature_groups as fg

    baselines = {
        "SYS": {
            "median": 0.0, "iqr_low": -0.5, "iqr_high": 0.5,
            "unit": "mmHg", "decimal_places": 0, "is_biometric": True, "n_benign": 100,
        }
    }
    fg._load_feature_baselines.cache_clear()
    phrase = fg.observation_phrase("SYS", 2.5, baselines=baselines)
    assert "+2" in phrase
    assert "mmHg" in phrase
    assert "IQR-widths" in phrase
    assert "above" in phrase
    assert "well outside baseline" in phrase


def test_observation_phrase_below_baseline():
    from module4_explanations.feature_groups import observation_phrase
    baselines = {
        "SpO2": {"median": 0.0, "iqr_low": 0.0, "iqr_high": 0.5,
                 "unit": "%", "decimal_places": 0, "is_biometric": True, "n_benign": 100},
    }
    out = observation_phrase("SpO2", -2.5, baselines=baselines)
    assert "below" in out
    assert "extreme deviation from baseline" in out


def test_observation_phrase_unknown_feature_returns_empty():
    from module4_explanations.feature_groups import observation_phrase
    assert observation_phrase("totally_made_up", 5.0, baselines={}) == ""


def test_observation_phrase_none_value_returns_empty():
    from module4_explanations.feature_groups import observation_phrase
    assert observation_phrase("SYS", None) == ""


def test_observation_phrase_degenerate_distribution():
    """Features with near-zero IQR (e.g. binary Flgs in normalised space)
    must NOT emit a runaway ``~4e9 IQR-widths`` number — the degenerate
    fallback should kick in instead."""
    from module4_explanations.feature_groups import observation_phrase
    baselines = {
        "Flgs": {
            "median": 0.0, "iqr_low": 0.0, "iqr_high": 0.0,
            "unit": "", "decimal_places": 2, "is_biometric": False, "n_benign": 100,
        }
    }
    out = observation_phrase("Flgs", -4.0, baselines=baselines)
    assert "IQR-widths" not in out
    assert "benign values cluster tightly" in out
    assert "outside benign baseline" in out


def test_observation_phrase_degenerate_at_baseline():
    """Degenerate distribution + observed value equal to median → near-baseline."""
    from module4_explanations.feature_groups import observation_phrase
    baselines = {
        "Flgs": {
            "median": 0.0, "iqr_low": 0.0, "iqr_high": 0.0,
            "unit": "", "decimal_places": 2, "is_biometric": False, "n_benign": 100,
        }
    }
    out = observation_phrase("Flgs", 0.0, baselines=baselines)
    assert "near baseline" in out


def test_deviation_band_thresholds():
    from module4_explanations.feature_groups import _deviation_band
    assert _deviation_band(0.0) == "near baseline"
    assert _deviation_band(0.4) == "near baseline"
    assert _deviation_band(1.0) == "slightly outside baseline"
    assert _deviation_band(2.0) == "well outside baseline"
    assert _deviation_band(5.0) == "extreme deviation from baseline"
    assert _deviation_band(-5.0) == "extreme deviation from baseline"


def test_baseline_artifact_exists_and_well_formed():
    """The tools/build_feature_baselines.py output must be present and
    every entry must carry the keys observation_phrase relies on."""
    path = Path(__file__).resolve().parent.parent / "artifacts" / "feature_baselines.json"
    if not path.exists():
        pytest.skip("feature_baselines.json not built — run tools/build_feature_baselines.py")
    data = json.loads(path.read_text())
    assert data, "baseline file is empty"
    required = {"median", "iqr_low", "iqr_high", "unit", "decimal_places", "is_biometric"}
    for feat, stats in data.items():
        missing = required - stats.keys()
        assert not missing, f"feature {feat} missing keys: {missing}"


# ── Phase 1.4 — MITRE plain_gloss ──────────────────────────────────


def test_mitre_lookup_includes_plain_gloss():
    """``_lookup_mitre_reference`` exposes ``plain_gloss`` for known categories."""
    from src.mve_generator import _lookup_mitre_reference, _load_mitre_mapping
    _load_mitre_mapping.cache_clear()
    ref = _lookup_mitre_reference("Data Alteration")
    assert ref is not None
    assert ref["id"].startswith("T")
    assert "plain_gloss" in ref
    assert ref["plain_gloss"], "Data Alteration should have a non-empty gloss after Phase 1.4"


def test_mitre_lookup_unknown_category_returns_none():
    from src.mve_generator import _lookup_mitre_reference
    assert _lookup_mitre_reference("definitely_not_a_real_category") is None
    assert _lookup_mitre_reference("") is None


def test_mitre_gloss_present_for_all_in_corpus_categories():
    """Every non-excluded category with non-low confidence should have a
    plain_gloss — Phase 1.4 acceptance."""
    import yaml
    path = Path(__file__).resolve().parent.parent / "config" / "attack_to_mitre_mapping.yaml"
    data = yaml.safe_load(path.read_text())
    cats = (data.get("attack_categories") or {})
    for name, cat in cats.items():
        if cat.get("excluded_from_coverage_audit"):
            continue
        primary = cat.get("primary_technique") or {}
        if str(primary.get("confidence", "")).lower() == "low":
            continue
        assert primary.get("plain_gloss"), (
            f"attack category {name!r} is missing primary_technique.plain_gloss "
            "(Phase 1.4 acceptance — all non-excluded categories need one)"
        )


# ── Phase 1.3 — actions_metadata schema + population ──────────────


def test_action_catalogue_has_expected_disruption():
    from module5_responses.config import ACTION_CATALOGUE
    for action, spec in ACTION_CATALOGUE.items():
        assert "expected_disruption" in spec, (
            f"action {action!r} is missing expected_disruption (Phase 1.3)"
        )
        assert spec["expected_disruption"], (
            f"action {action!r} has empty expected_disruption — fill in or remove the key"
        )


def test_select_adaptive_response_emits_actions_metadata():
    from module5_responses.adaptive import select_adaptive_response
    out = select_adaptive_response(
        risk_level="HIGH", risk_score=0.75,
        attack_category="Spoofing", device_tier="vital_monitoring",
    )
    assert "actions_metadata" in out
    assert len(out["actions_metadata"]) == len(out["actions"])
    sample = out["actions_metadata"][0]
    assert set(sample.keys()) == {
        "name", "cost", "reversible", "requires_approval", "expected_disruption",
    }
    assert [m["name"] for m in out["actions_metadata"]] == out["actions"]


def test_response_schema_accepts_actions_metadata():
    from common.alert_response_schema import Response, EscalationChain
    payload = {
        "actions": ["log_event"],
        "action_descriptions": ["Log event to SIEM for audit trail"],
        "actions_metadata": [{
            "name": "log_event", "cost": 0.1, "reversible": True,
            "requires_approval": False,
            "expected_disruption": "No clinical impact.",
        }],
        "escalation_chain": EscalationChain(primary=None, secondary=None, tertiary=None),
        "escalation_rationale": "No attack detected",
        "max_response_min": 480, "priority": 4, "rationale": "Base",
        "device_tier": "vital_monitoring", "device_constraint_applied": False,
    }
    r = Response(**payload)
    assert r.actions_metadata[0].reversible is True


def test_response_schema_default_actions_metadata_empty_for_legacy():
    """Legacy artifacts without ``actions_metadata`` should still validate."""
    from common.alert_response_schema import Response, EscalationChain
    r = Response(
        actions=["log_event"], action_descriptions=["x"],
        escalation_chain=EscalationChain(primary=None, secondary=None, tertiary=None),
        escalation_rationale="", max_response_min=0, priority=4,
        rationale="", device_tier="vital_monitoring",
        device_constraint_applied=False,
    )
    assert r.actions_metadata == []


# ── Phase 1.2 — parametrized actions ───────────────────────────────


def test_annotate_role_known():
    from module5_responses.config import annotate_role
    out = annotate_role("Security lead")
    # Compact format ``[NNNN/Nm]`` adopted to keep MVE Layer 3 inside the
    # 150-word budget — see annotate_role docstring.
    assert "4401" in out
    assert "/" in out and "m]" in out


def test_annotate_role_case_insensitive():
    from module5_responses.config import annotate_role
    out = annotate_role("Charge nurse on duty")
    assert "4470" in out


def test_annotate_role_unknown_pass_through():
    from module5_responses.config import annotate_role
    out = annotate_role("Hospital Cafeteria Manager")
    assert out == "Hospital Cafeteria Manager"


def test_annotate_role_prefers_longest_match():
    """``ICU charge nurse`` must not be annotated as the shorter ``Charge Nurse``
    when both are valid prefixes — the longest key wins."""
    from module5_responses.config import annotate_role
    out = annotate_role("(3) ICU charge nurse")
    assert "4471" in out


def test_synthesize_raw_alert_includes_alert_id():
    from common.device_class import synthesize_raw_alert
    out = synthesize_raw_alert(42, "Spoofing", 0.65)
    assert out["alert_id"] == "ALERT-0042"


def test_generate_mve_prepends_alert_id_and_annotates_escalation():
    """End-to-end: generate_mve with synthesized raw_alert should produce
    a Layer 3 immediate_action prefixed by ALERT-XXXX and an
    escalation_path whose roles carry extensions."""
    from common.device_class import synthesize_raw_alert
    from src.mve_generator import generate_mve

    raw = synthesize_raw_alert(17, "Spoofing", 0.85)
    device_ctx = {
        "device_type": "patient_monitor",
        "criticality": "HIGH",
        "clinical_function": "Patient monitor — vital signs tracking",
        "patchable": True,
    }
    mve = generate_mve(
        raw_alert=raw,
        device_context=device_ctx,
        baseline={"baseline_days": 90},
        user_context=None,
        shap_context=None,
        event_context=None,
        force_rule_based=True,
        risk_level="HIGH",
    )
    action = mve.layer_3["immediate_action"]
    esc = mve.layer_3["escalation_path"]
    assert "ALERT-0017" in action
    assert "patient_monitor" in action
    # Compact role annotation ``[ext/SLA]`` — see annotate_role.
    assert "[" in esc and "/" in esc and "m]" in esc, (
        f"escalation_path missing compact ext/SLA annotation: {esc!r}"
    )
