"""Path B · commit 3 — Hard 150-word budget rejection in generate_mve.

The 3-layer MVE has a non-negotiable 150-word total cap (L1 ≤ 60,
L2 ≤ 50, L3 ≤ 60). The prompt instructs the LLM to respect it; this
test verifies that when an LLM ignores the instruction and overshoots,
``generate_mve`` rejects the LLM output and falls back to the
deterministic rule-based template (which is budget-safe by
construction).

Complements ``test_mve_word_budget.py`` which checks the
SHAP/MITRE-injection word budget on the rule-based path. This file is
specifically about the LLM-overshoot → rule-based-fallback contract.
"""
from __future__ import annotations

import src.mve_generator as mve_mod
from src.data_models import MVEOutput


def _over_budget_mve() -> MVEOutput:
    """An MVE with ~200 words across the 3 layers — over the 150 cap."""
    filler = " ".join(f"word{i:03d}" for i in range(25))
    return MVEOutput(
        layer_1={
            "baseline_behavior":      filler,
            "deviation_description":  filler,
            "confidence_indicator":   "Confidence: HIGH — strong signal",
        },
        layer_2={
            "affected_system":     filler,
            "patient_care_impact": filler,
            "phi_exposure":        "Real-time vitals visible",
            "severity_label":      "HIGH",
            "severity_rationale":  filler,
        },
        layer_3={
            "immediate_action":    f"isolate device {filler}",
            "clinical_constraint": "DO NOT disrupt active monitoring",
            "escalation_path":     "(1) IT Security, (2) Charge Nurse",
            "timeframe":           "Act within 1h",
        },
    )


def _minimal_inputs() -> dict:
    return dict(
        raw_alert={"alert_name": "anomalous outbound", "protocol": "https"},
        device_context={
            "device_type": "patient_monitor",
            "criticality": "HIGH",
            "patchable": True,
        },
        baseline={"normal_destinations": ["10.0.0.1"], "baseline_days": 90},
        user_context=None,
        risk_level="HIGH",
    )


def test_over_budget_llm_falls_back_to_rule_based(monkeypatch):
    """An over-budget MVE from the LLM path triggers rule-based fallback."""
    over_budget = _over_budget_mve()
    assert over_budget.total_word_count > 150, (
        "fixture sanity: over-budget MVE must actually exceed 150 words"
    )
    monkeypatch.setattr(
        mve_mod, "_generate_llm_openai",
        lambda *a, **kw: over_budget,
    )
    # Mock Anthropic too so a regression bypassing OpenAI doesn't silently pass.
    monkeypatch.setattr(
        mve_mod, "_generate_llm_anthropic",
        lambda *a, **kw: over_budget,
    )

    mve = mve_mod.generate_mve(**_minimal_inputs())

    assert mve.provider == "rule_based", (
        f"over-budget LLM output should trigger rule_based fallback; "
        f"got provider={mve.provider!r}, word_count={mve.total_word_count}"
    )
    assert mve.total_word_count <= 150, (
        f"final MVE still exceeds 150 words after fallback: "
        f"{mve.total_word_count}. Hard cap not honored."
    )


def test_in_budget_llm_kept(monkeypatch):
    """An in-budget LLM MVE is kept — guards against false-positive fallback."""
    in_budget = MVEOutput(
        layer_1={
            "baseline_behavior":      "Device usually talks to internal hosts.",
            "deviation_description":  "Outbound to external IP detected.",
            "confidence_indicator":   "Confidence: HIGH — strong signal",
        },
        layer_2={
            "affected_system":     "Patient monitor in ICU bed 12.",
            "patient_care_impact": "Vital-sign telemetry potentially at risk.",
            "phi_exposure":        "Real-time vitals visible",
            "severity_label":      "HIGH",
            "severity_rationale":  "Active monitoring of high-acuity patient.",
        },
        layer_3={
            "immediate_action":    "Isolate device from external network.",
            "clinical_constraint": "DO NOT disrupt active monitoring.",
            "escalation_path":     "(1) IT Security, (2) Charge Nurse",
            "timeframe":           "Act within 1 hour.",
        },
    )
    assert in_budget.total_word_count <= 150, "fixture sanity check"
    monkeypatch.setattr(
        mve_mod, "_generate_llm_openai",
        lambda *a, **kw: in_budget,
    )

    mve = mve_mod.generate_mve(**_minimal_inputs())
    assert mve.provider == "openai", (
        f"in-budget LLM output should be kept; got provider={mve.provider!r}"
    )
