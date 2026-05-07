"""v5 architectural lock: Track A is XGB-only at runtime.

Phase A (this commit) made ``classify_alert_v4`` independent of
RandomForest / DecisionTree probabilities. These tests pin that
decision so a future regression that re-introduces a multi-model
dependency in the runtime classification path is caught at PR time.

Phase B (post-defense) will collapse the DAE cascade input from 28
to 26 dims and remove RF/DT from runtime entirely; that work lives
in ``module2_detection/retrain_dae_26dim.py`` and is not exercised
here.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from module3_risk_scoring.triage_v4 import classify_alert_v4
from src.data_models import AlertType

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRIAGE_PATH = PROJECT_ROOT / "module3_risk_scoring" / "triage_v4.py"


# ── Signature lock ────────────────────────────────────────────────────


def test_classify_alert_v4_required_args_are_xgb_and_dae_only():
    """``p_xgb`` and ``dae_score`` are the only positional/required
    parameters. ``p_rf``, ``p_dt``, ``diversity_score`` are optional
    kwargs (back-compat) and must not be required."""
    sig = inspect.signature(classify_alert_v4)
    required = {
        name for name, p in sig.parameters.items()
        if p.default is inspect.Parameter.empty
        and p.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    assert required == {"p_xgb", "dae_score"}, (
        f"Required args drifted: {required}. v5 contract is XGB + DAE only."
    )


def test_classify_alert_v4_optional_kwargs_present_for_back_compat():
    """Legacy callers can still pass p_rf/p_dt/diversity_score; the
    signature must accept them without raising."""
    sig = inspect.signature(classify_alert_v4)
    for legacy_kw in ("p_rf", "p_dt", "diversity_score"):
        assert legacy_kw in sig.parameters, (
            f"Back-compat kwarg {legacy_kw!r} missing — pre-v5 callers will break."
        )
        assert sig.parameters[legacy_kw].default is None


# ── Runtime independence from RF/DT/diversity ────────────────────────


def test_decision_invariant_under_p_rf_p_dt_changes():
    """For fixed (p_xgb, dae_score), wiggling p_rf/p_dt must not
    change the alert_type, confidence, c_detect, or template_id."""
    base = classify_alert_v4(p_xgb=0.50, dae_score=0.97)
    for p_rf, p_dt in [(0.0, 0.0), (0.5, 0.5), (0.99, 0.01)]:
        out = classify_alert_v4(p_xgb=0.50, dae_score=0.97, p_rf=p_rf, p_dt=p_dt)
        assert out.alert_type == base.alert_type
        assert out.confidence == base.confidence
        assert out.c_detect == base.c_detect
        assert out.template_id == base.template_id


def test_decision_invariant_under_diversity_changes():
    """diversity_score must be ignored by the predicates (echoed only
    onto the audit field of the returned decision)."""
    base = classify_alert_v4(p_xgb=0.95, dae_score=0.10)
    out = classify_alert_v4(p_xgb=0.95, dae_score=0.10, diversity_score=0.99)
    assert out.alert_type == base.alert_type
    assert out.confidence == base.confidence
    assert out.c_detect == base.c_detect
    assert out.diversity_score == 0.99  # echoed audit only
    assert base.diversity_score == 0.0


# ── DISAGREEMENT_ANOMALY now means Track-A-vs-Track-B disagreement ──


def test_disagreement_anomaly_fires_on_xgb_borderline_dae_strong():
    """v5 redefinition: XGB in 0.40–0.85 borderline AND DAE >= 0.95."""
    out = classify_alert_v4(p_xgb=0.50, dae_score=0.97)
    assert out.alert_type == AlertType.DISAGREEMENT_ANOMALY


def test_disagreement_anomaly_does_not_fire_when_xgb_low():
    """Low XGB + high DAE is STRONG_NOVEL_ANOMALY, not DISAGREEMENT."""
    out = classify_alert_v4(p_xgb=0.10, dae_score=0.97)
    assert out.alert_type == AlertType.STRONG_NOVEL_ANOMALY


def test_disagreement_anomaly_does_not_fire_when_xgb_high():
    """High XGB + high DAE is KNOWN_ATTACK_UNCERTAIN, not DISAGREEMENT."""
    out = classify_alert_v4(p_xgb=0.95, dae_score=0.97)
    assert out.alert_type == AlertType.KNOWN_ATTACK_UNCERTAIN


# ── Source-text guard against accidentally re-importing diversity ──


def test_triage_v4_does_not_call_normalised_diversity():
    """v5 retired ``_normalised_diversity``. If it sneaks back into the
    classifier the old c_detect formula is partially restored and the
    INVARIANT-1 reasoning in the file's docstring stops being true."""
    src = TRIAGE_PATH.read_text(encoding="utf-8")
    assert "_normalised_diversity(" not in src, (
        "_normalised_diversity is back in triage_v4 — v5 retired this term."
    )
    # And the c_detect formula should be the simple two-arg max.
    assert "c_detect = max(p_xgb, dae_score)" in src


# ── Retrain script exists for Phase B ─────────────────────────────────


def test_phase_b_retrain_script_present():
    """The DAE 26-dim retrain script is the documented Phase B path.
    If someone deletes it, the Phase B plan in
    ``docs/post_defense_track_a_simplification.md`` becomes stale."""
    script = PROJECT_ROOT / "module2_detection" / "retrain_dae_26dim.py"
    assert script.exists(), (
        "module2_detection/retrain_dae_26dim.py missing — "
        "post-defense DAE retrain plan is no longer reachable."
    )
    src = script.read_text(encoding="utf-8")
    assert "26-dim cascade" in src
    assert "[25 raw || P_xgb_val]" in src
