"""ARCHITECTURE.md Step [11] / [12] / [13] / [14] — M6 dashboard affordances.

Locks the new fields populated on every curated alert by
``module6_evaluation.curate_demo_alerts``:

* ``shared_anchor``        — Step [13] / INVARIANT 9 (5-field block).
* ``shap_stability_score`` — Step [11] (float in [0, 1]).
* ``shap_is_stable``       — Step [11] (bool, ``score >= 0.90``).
* ``shap_source``          — Step [11] / Step [12] gap flag.
* ``mve_mode``             — Step [12] (``A_llm`` / ``B_rule``).
* ``configs/tier_routing.yaml`` + ``configs/hospital_capabilities.yaml``
  exist and are loadable.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_PATH = PROJECT_ROOT / "results" / "reports" / "evaluation_alerts.json"


@pytest.fixture(scope="module")
def alerts() -> list[dict]:
    if not EVAL_PATH.exists():
        pytest.skip(
            f"{EVAL_PATH} missing — run "
            "`python -m module6_evaluation.curate_demo_alerts` first."
        )
    with EVAL_PATH.open(encoding="utf-8") as f:
        return json.load(f)


# ── INVARIANT 9: shared anchor block ──────────────────────────────────


def test_every_alert_has_shared_anchor(alerts):
    for a in alerts:
        anchor = a.get("shared_anchor")
        assert isinstance(anchor, dict), (
            f"Alert {a.get('alert_id')} missing shared_anchor block"
        )
        for k in ("alert_id", "risk_tier", "device_id", "one_line_summary", "timestamp"):
            assert k in anchor, (
                f"Alert {a.get('alert_id')} shared_anchor missing {k!r}"
            )
            assert anchor[k], (
                f"Alert {a.get('alert_id')} shared_anchor.{k} is empty"
            )


def test_shared_anchor_alert_id_matches_top_level(alerts):
    for a in alerts:
        assert a["shared_anchor"]["alert_id"] == a["alert_id"]


def test_shared_anchor_risk_tier_matches_risk_level(alerts):
    for a in alerts:
        assert a["shared_anchor"]["risk_tier"] == a["risk_level"]


def test_shared_anchor_device_id_is_string(alerts):
    """Catches the regression where ``active_device`` (a bool flag in
    legacy schemas) leaked into ``device_id`` as a True/False value."""
    for a in alerts:
        dev = a["shared_anchor"]["device_id"]
        assert isinstance(dev, str) and dev, (
            f"Alert {a['alert_id']} device_id is not a non-empty string: "
            f"{dev!r}"
        )


# ── Step [11]: SHAP stability indicator + shap_source gap flag ────────


def test_every_alert_has_shap_stability_fields(alerts):
    for a in alerts:
        assert "shap_stability_score" in a
        assert "shap_is_stable" in a
        assert "shap_source" in a


def test_shap_stability_score_in_unit_interval(alerts):
    for a in alerts:
        s = float(a["shap_stability_score"])
        assert 0.0 <= s <= 1.0


def test_shap_is_stable_consistent_with_score(alerts):
    """``is_stable`` MUST equal ``score >= 0.90`` per the doc cutoff."""
    for a in alerts:
        score = float(a["shap_stability_score"])
        assert bool(a["shap_is_stable"]) == bool(score >= 0.90)


def test_shap_source_in_allowed_values(alerts):
    valid = {"xgboost", "xgboost_low_confidence", "dae_recon"}
    for a in alerts:
        assert a["shap_source"] in valid, (
            f"Alert {a['alert_id']} has invalid shap_source: "
            f"{a['shap_source']!r}"
        )


def test_novel_alerts_get_low_confidence_shap_flag(alerts):
    """NOVEL_ANOMALY / STRONG_NOVEL_ANOMALY alerts MUST be flagged
    ``xgboost_low_confidence`` per Step [11] "Known gap"."""
    for a in alerts:
        fc = a.get("fusion_class", "")
        if fc in ("NOVEL_ANOMALY", "STRONG_NOVEL_ANOMALY"):
            assert a["shap_source"] == "xgboost_low_confidence", (
                f"Alert {a['alert_id']} ({fc}) should flag SHAP as "
                f"low-confidence; got {a['shap_source']!r}"
            )


# ── Step [12]: Mode A / Mode B audit visibility ───────────────────────


def test_every_alert_has_mve_mode(alerts):
    for a in alerts:
        if "mve_structured" in a:
            assert a.get("mve_mode") in ("A_llm", "B_rule"), (
                f"Alert {a['alert_id']} has mve_structured but no/invalid "
                f"mve_mode: {a.get('mve_mode')!r}"
            )


# ── Step [14]: tier routing + hospital capabilities YAMLs loadable ────


def test_tier_routing_yaml_exists():
    p = PROJECT_ROOT / "configs" / "tier_routing.yaml"
    assert p.exists(), f"{p} missing — Step [14] policy YAML required"


def test_hospital_capabilities_yaml_exists():
    p = PROJECT_ROOT / "configs" / "hospital_capabilities.yaml"
    assert p.exists(), f"{p} missing — Step [14] policy YAML required"


def test_tier_routing_loader_returns_rules():
    from module5_responses.tier_routing_v4 import load_tier_routing_yaml
    body = load_tier_routing_yaml()
    rules = body.get("routing_rules") or []
    assert len(rules) > 0, "tier_routing.yaml must declare routing_rules"


def test_hospital_capabilities_loader_returns_presets():
    from module5_responses.tier_routing_v4 import (
        get_available_tiers,
        load_hospital_capabilities,
    )
    cfg = load_hospital_capabilities()
    assert "deployment_size" in cfg
    avail = get_available_tiers()
    assert "L1" in avail, "Every deployment must staff at least L1"


# ── Doc-named files exist (M6 description rename) ─────────────────────


def test_compute_rq1_metrics_module_exists():
    p = PROJECT_ROOT / "module6_evaluation" / "compute_rq1_metrics.py"
    assert p.exists(), (
        "compute_rq1_metrics.py missing — ARCHITECTURE.md M6 description "
        "names this file (renamed from compute_rq2_metrics.py)."
    )


def test_curate_demo_alerts_module_exists():
    p = PROJECT_ROOT / "module6_evaluation" / "curate_demo_alerts.py"
    assert p.exists(), (
        "curate_demo_alerts.py missing — ARCHITECTURE.md M6 description "
        "lists this as the canonical curation entry point."
    )


def test_build_stratified_eval_set_archived():
    """The Phase-4 4-way split made this script's test-split source
    obsolete; it lives under docs/_archive/ now. Re-introducing it
    in module6_evaluation/ would revert the doc's separation-of-concerns
    invariant."""
    live = PROJECT_ROOT / "module6_evaluation" / "build_stratified_eval_set.py"
    assert not live.exists(), (
        f"{live} should not exist — archived to docs/_archive/."
    )
