"""Word-budget enforcement for SHAP + MITRE injections in mve_generator.

Layers 1+2+3 combined must stay ≤ 150 words per the MVE spec
(see `tests/acceptance_tests.py::test_layer1_length_constraint`).

The G1/G2 (top-3 SHAP injection) and G3 (MITRE technique injection)
fixes in `src.mve_generator.generate_mve` each add 6-12 words to
Layer 1's `deviation_description`. The fix comments claim this stays
"well under the 150-word cap" — these tests assert that claim across
every attack category that triggers MITRE injection, with realistic
SHAP context attached. A regression here means either the injections
got chattier or the base rule-based templates grew, pushing total
words over budget.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.mve_generator import generate_mve

MAX_TOTAL_WORDS = 150          # MVEOutput contract (data_models.py:71)
MAX_L1_WORDS = 60              # data_models.py:79

# Categories that produce a MITRE reference (excludes 'normal' baseline,
# which has primary_technique=NONE and is excluded from coverage audit).
MITRE_INJECTING_CATEGORIES = [
    "Spoofing",
    "Data Alteration",
    "Reconnaissance",
    "Initial Access",
    "Lateral Movement",
    "Exfiltration",
]

# Realistic per-sample top-3 SHAP features (matches what analyst_report.json
# carries for the test split — short tokenic feature names).
REALISTIC_SHAP_CONTEXT = {
    "top_features": ["DIntPkt", "SrcLoad", "Sport"],
    "top_category": "network_protocol",
    "shap_direction": "elevated",
}


def _make_alert(category: str) -> dict:
    return {
        "alert_id": "TEST-WB-0001",
        "severity": "HIGH",
        "alert_type": "anomalous_outbound_connection",
        "attack_category": category,
        "src_ip": "10.0.0.42",
        "dest_ip": "203.0.113.7",
        "dest_port": 443,
        "proto": "TCP",
    }


def _make_device() -> dict:
    return {
        "device_type": "patient_monitor",
        "clinical_function": "vitals_monitoring",
        "location": "ICU-3",
        "criticality": "HIGH",
        "patchable": True,
    }


def _make_baseline() -> dict:
    return {
        "normal_destinations": ["internal EHR hosts", "vendor update servers"],
        "normal_protocols": ["HTTPS", "DICOM"],
        "normal_hours": "business hours",
        "baseline_days": 90,
    }


@pytest.mark.parametrize("category", MITRE_INJECTING_CATEGORIES)
def test_total_word_count_within_cap_with_both_injections(category):
    """Layer 1+2+3 stays ≤ 150 words after both SHAP and MITRE injection."""
    mve = generate_mve(
        raw_alert=_make_alert(category),
        device_context=_make_device(),
        baseline=_make_baseline(),
        user_context=None,
        shap_context=REALISTIC_SHAP_CONTEXT,
        force_rule_based=True,
    )
    total = mve.total_word_count
    assert total <= MAX_TOTAL_WORDS, (
        f"category={category!r}: total_word_count={total} > {MAX_TOTAL_WORDS}. "
        f"The G1/G2/G3 injection comment claims word-budget impact is "
        f"~6-12 words and stays under cap — that claim is now violated. "
        f"Layer 1: {mve.layer_1}"
    )


@pytest.mark.parametrize("category", MITRE_INJECTING_CATEGORIES)
def test_layer1_word_count_within_cap_with_both_injections(category):
    """Layer 1 alone stays ≤ 60 words after both injections — the more
    stressed budget since both injections append to layer_1.
    """
    mve = generate_mve(
        raw_alert=_make_alert(category),
        device_context=_make_device(),
        baseline=_make_baseline(),
        user_context=None,
        shap_context=REALISTIC_SHAP_CONTEXT,
        force_rule_based=True,
    )
    l1_words = sum(
        len(mve.layer_1.get(f, "").split())
        for f in ("baseline_behavior", "deviation_description", "confidence_indicator")
    )
    assert l1_words <= MAX_L1_WORDS, (
        f"category={category!r}: layer_1 word count={l1_words} > {MAX_L1_WORDS}. "
        f"Both injections append to deviation_description and have "
        f"pushed Layer 1 past its 60-word ceiling. Layer 1: {mve.layer_1}"
    )


def test_mitre_injection_actually_fired():
    """Sanity guard for the parametrized tests above: confirm at least
    one of them actually exercises the MITRE-injection path. If MITRE
    never fires, the word-budget tests are vacuous wrt G3.
    """
    mve = generate_mve(
        raw_alert=_make_alert("Exfiltration"),
        device_context=_make_device(),
        baseline=_make_baseline(),
        user_context=None,
        shap_context=REALISTIC_SHAP_CONTEXT,
        force_rule_based=True,
    )
    dev = mve.layer_1.get("deviation_description", "")
    assert "MITRE T" in dev, (
        f"MITRE reference not found in Layer 1 deviation_description for "
        f"Exfiltration — G3 injection path may have regressed. "
        f"deviation_description: {dev!r}"
    )


def test_shap_injection_actually_fired():
    """Sanity guard: confirm SHAP top-3 features appear in Layer 1."""
    mve = generate_mve(
        raw_alert=_make_alert("Exfiltration"),
        device_context=_make_device(),
        baseline=_make_baseline(),
        user_context=None,
        shap_context=REALISTIC_SHAP_CONTEXT,
        force_rule_based=True,
    )
    dev = mve.layer_1.get("deviation_description", "")
    for feat in REALISTIC_SHAP_CONTEXT["top_features"]:
        assert feat in dev, (
            f"SHAP feature {feat!r} not found in Layer 1 — G1/G2 injection "
            f"path may have regressed. deviation_description: {dev!r}"
        )
