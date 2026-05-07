"""ARCHITECTURE.md Step [11] — SHAP context contract tests.

Locks the four invariants the doc commits to for ``SHAPContext``:

* I1 ``stability_score`` is a float in [0, 1].
* I2 ``is_stable = (stability_score >= 0.90)`` per the doc cutoff.
* I3 ``shap_source ∈ {"xgboost", "xgboost_low_confidence", "dae_recon"}``;
     NOVEL_ANOMALY / STRONG_NOVEL_ANOMALY alerts MUST be flagged
     ``"xgboost_low_confidence"`` (XGBoost SHAP is not faithful when
     DAE drove the alert).
* I4 ``shap_background.pkl`` exists and contains 200 stratified
     training samples — the persisted background dataset cited by
     the doc's TreeSHAP description.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pytest

from module4_explanations.module4_online_explainer import (
    AlertExplainer,
    STABILITY_HIGH,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── I4: persisted background dataset ──────────────────────────────────


def test_shap_background_pkl_exists():
    path = PROJECT_ROOT / "results/models/shap_background.pkl"
    assert path.exists(), (
        f"{path} missing — TreeSHAP background sample is supposed to be "
        "persisted (ARCHITECTURE.md Step [11]). Run "
        "`python -m module4_explanations.build_shap_background`."
    )


def test_shap_background_has_expected_shape():
    body = joblib.load(PROJECT_ROOT / "results/models/shap_background.pkl")
    bg = body["background"]
    assert body["n_samples"] == 200, (
        f"Background sample size {body['n_samples']} != 200 (doc-mandated)"
    )
    assert bg.shape[0] == 200
    assert bg.shape[1] == 25, (
        f"Background features {bg.shape[1]} != 25 raw features"
    )


# ── I1 + I2: stability_score / is_stable on SHAPContext ───────────────


def test_build_shap_context_populates_stability_and_is_stable():
    """The SHAPContext dict produced by ``build_shap_context`` must
    carry both the score and the boolean cutoff at 0.90."""
    top = [
        {"feature": "Heart_rate", "shap_value":  0.42},
        {"feature": "DIntPkt",    "shap_value":  0.18},
        {"feature": "Sport",      "shap_value": -0.12},
    ]
    # Stable case
    ctx = AlertExplainer.build_shap_context(top, stability_score=0.97)
    assert ctx["stability_score"] == 0.97
    assert ctx["is_stable"] is True

    # Unstable case
    ctx = AlertExplainer.build_shap_context(top, stability_score=0.42)
    assert ctx["stability_score"] == 0.42
    assert ctx["is_stable"] is False


def test_stability_high_cutoff_is_doc_value():
    """Catch silent threshold drift away from the 0.90 the doc cites."""
    assert STABILITY_HIGH == 0.90


# ── I3: shap_source flag for NOVEL_ANOMALY / STRONG_NOVEL_ANOMALY ─────


def test_shap_source_default_is_xgboost():
    """Without a fusion_class hint, the default source is plain
    XGBoost (faithful for KNOWN_ATTACK / CONFIRMED_ANOMALY)."""
    top = [{"feature": "Heart_rate", "shap_value": 0.5}]
    ctx = AlertExplainer.build_shap_context(top)
    assert ctx["shap_source"] == "xgboost"


@pytest.mark.parametrize(
    "fusion_class",
    ["NOVEL_ANOMALY", "STRONG_NOVEL_ANOMALY"],
)
def test_shap_source_flags_low_confidence_for_novel_alerts(fusion_class: str):
    """XGBoost SHAP is unfaithful when the DAE drove the alert; the
    contract surfaces that as ``shap_source = "xgboost_low_confidence"``."""
    top = [{"feature": "Heart_rate", "shap_value": 0.5}]
    ctx = AlertExplainer.build_shap_context(top, fusion_class=fusion_class)
    assert ctx["shap_source"] == "xgboost_low_confidence", (
        f"NOVEL alert ({fusion_class}) must flag SHAP as low-confidence; "
        f"got {ctx['shap_source']!r}"
    )


@pytest.mark.parametrize(
    "fusion_class",
    ["KNOWN_ATTACK", "CONFIRMED_ANOMALY", "BENIGN"],
)
def test_shap_source_stays_xgboost_for_non_novel(fusion_class: str):
    top = [{"feature": "Heart_rate", "shap_value": 0.5}]
    ctx = AlertExplainer.build_shap_context(top, fusion_class=fusion_class)
    assert ctx["shap_source"] == "xgboost"


# ── SHAPContext data model has the doc-mandated fields ────────────────


def test_shap_context_dataclass_has_all_doc_fields():
    """If someone removes a field from SHAPContext, the doc's contract
    silently breaks. Lock the field set."""
    from src.data_models import SHAPContext
    ctx = SHAPContext(
        top_category="timing_pattern",
        top_features=["DIntPkt"],
        shap_direction="elevated",
        confidence_from_shap="MEDIUM",
    )
    # Defaults for the new fields
    assert ctx.stability_score == 1.0
    assert ctx.is_stable is True
    assert ctx.shap_source == "xgboost"

    ctx2 = SHAPContext(
        top_category="timing_pattern",
        top_features=["DIntPkt"],
        shap_direction="elevated",
        confidence_from_shap="MEDIUM",
        stability_score=0.7,
        is_stable=False,
        shap_source="xgboost_low_confidence",
    )
    assert ctx2.is_stable is False
    assert ctx2.shap_source == "xgboost_low_confidence"
