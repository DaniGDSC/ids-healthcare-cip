"""Module 4 config — constants invariants + C3 single-source check."""
from __future__ import annotations

from module4_explanations.config import (
    BASELINE_TRACK_A_MODELS,
    CLINICIAN_TEMPLATES,
    FEATURE_CONCEPTS,
    NLG_TEMPLATES,
    SHAP_MODELS,
    SKIP_SHAP_MODELS,
    TOP_K_FEATURES,
    TOP_N_WATERFALL,
    TRACK_A_MODELS,
    format_clinician_template,
)


def test_clinician_templates_has_4_severities():
    assert set(CLINICIAN_TEMPLATES.keys()) == {"CRITICAL", "HIGH", "MEDIUM", "LOW"}


def test_clinician_templates_all_have_idx_placeholder():
    """Every template must accept the ``{idx}`` placeholder so callers
    can choose whether to mention sample index."""
    for severity, template in CLINICIAN_TEMPLATES.items():
        assert "{idx}" in template, f"{severity}: missing {{idx}} placeholder"


def test_format_clinician_template_with_sample_index():
    out = format_clinician_template(
        "CRITICAL", sample_index=42, narrative="test narrative",
        secondary_note="",
    )
    assert "(Sample 42)" in out
    assert "test narrative" in out


def test_format_clinician_template_without_sample_index():
    out = format_clinician_template(
        "HIGH", sample_index=None, narrative="x", secondary_note="",
    )
    assert "(Sample" not in out


def test_nlg_templates_complete():
    """6-step NLG needs all the keys generate_clinician_alert reads."""
    needed = {
        "severity_header", "detection_sentence",
        "feature_explanation_network", "feature_explanation_biometric",
        "risk_context", "acuity_note_normal", "acuity_note_abnormal",
        "action_recommendation",
    }
    assert needed.issubset(set(NLG_TEMPLATES.keys()))


def test_shap_models_subset_of_track_a():
    assert set(SHAP_MODELS).issubset(set(TRACK_A_MODELS.keys()))


def test_skip_shap_complements_shap_models():
    """SKIP_SHAP_MODELS = TRACK_A_MODELS - SHAP_MODELS."""
    assert SKIP_SHAP_MODELS | set(SHAP_MODELS) == set(TRACK_A_MODELS.keys())
    assert SKIP_SHAP_MODELS & set(SHAP_MODELS) == set()


def test_track_a_runtime_is_xgboost_only():
    """Runtime Track A registry must expose only XGBoost (post Phase 2)."""
    assert set(TRACK_A_MODELS.keys()) == {"xgboost"}


def test_baseline_models_disjoint_from_runtime():
    """RF/DT live in BASELINE_TRACK_A_MODELS only — never in TRACK_A_MODELS."""
    assert set(BASELINE_TRACK_A_MODELS.keys()) == {"random_forest", "decision_tree"}
    assert set(BASELINE_TRACK_A_MODELS.keys()) & set(TRACK_A_MODELS.keys()) == set()


def test_feature_concepts_carry_required_keys():
    for feat, concept in FEATURE_CONCEPTS.items():
        assert "label" in concept, f"{feat}: missing label"
        assert "category" in concept, f"{feat}: missing category"
        assert concept["category"] in {"network", "biometric"}, f"{feat}: bad category"
        assert "direction_high" in concept and "direction_low" in concept


def test_top_constants_positive():
    assert TOP_K_FEATURES > 0
    assert TOP_N_WATERFALL > 0


def test_canonical_biometric_features_in_concepts():
    """Every biometric column from common.phi has a FEATURE_CONCEPTS entry."""
    from common.phi import BIOMETRIC_COLUMNS
    for feat in BIOMETRIC_COLUMNS:
        assert feat in FEATURE_CONCEPTS, f"{feat} missing from FEATURE_CONCEPTS"
