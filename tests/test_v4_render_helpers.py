"""Pure-function tests for the v4 visual helpers in ``module6_app.py``.

The Streamlit-side renderers (``render_alert_type_badge`` etc.) are
thin wrappers around ``st.markdown`` / ``st.success`` and need a
ScriptRunContext to exercise; they are verified manually. This file
covers only the deterministic, framework-free pieces:

  * :func:`derive_v4_fields` over real ``evaluation_alerts.json`` rows
    and over edge-case synthetic inputs.
  * :func:`badge_for_alert_type` totality across the 9-type taxonomy.
  * :func:`confidence_display` totality across the 4 levels.
  * :func:`anomalous_dims_markdown` empty-list silence.

The point of these tests is to lock the heuristic in place — anyone
replacing the heuristic with a native Layer 3 field read should see
exactly which rows change classification by re-running this file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from module6_evaluation.module6_app import derive_v4_fields
from module6_evaluation.presentation_v4 import (
    BADGE_FOR_ALERT_TYPE,
    CONFIDENCE_INDICATOR,
    MODE_A_LLM,
    MODE_B_RULE_BASED,
    anomalous_dims_markdown,
    badge_for_alert_type,
    confidence_display,
)
from src.data_models import AlertType, Confidence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_PATH = PROJECT_ROOT / "results" / "reports" / "evaluation_alerts.json"


# ── badge_for_alert_type totality ──────────────────────────────────────


def test_badge_covers_all_nine_alert_types():
    assert set(BADGE_FOR_ALERT_TYPE) == set(AlertType)


@pytest.mark.parametrize("at", list(AlertType))
def test_badge_returns_dict_with_required_fields(at):
    badge = badge_for_alert_type(at)
    assert {"color", "icon", "label", "urgency"} <= set(badge)
    assert badge["color"].startswith("#") and len(badge["color"]) == 7


def test_disagreement_anomaly_is_purple():
    """The thesis defense's headline visual upgrade."""
    assert badge_for_alert_type(AlertType.DISAGREEMENT_ANOMALY)["color"] == "#9333EA"


def test_badge_unknown_string_falls_back_to_benign():
    assert badge_for_alert_type("__not_a_type__") == BADGE_FOR_ALERT_TYPE[AlertType.BENIGN]


# ── confidence_display totality ────────────────────────────────────────


@pytest.mark.parametrize("level", list(Confidence))
def test_confidence_returns_dot_pattern(level):
    style = confidence_display(level)
    assert style["symbol"].startswith("●")
    assert style["color"] in {"green", "orange", "gray"}


def test_confidence_unknown_falls_back_to_low():
    assert confidence_display("__nope__") == CONFIDENCE_INDICATOR[Confidence.LOW]


# ── anomalous_dims_markdown ────────────────────────────────────────────


def test_anomalous_dims_empty_returns_empty_string():
    assert anomalous_dims_markdown([], ["a", "b", "c"]) == ""


def test_anomalous_dims_renders_feature_names_and_count():
    out = anomalous_dims_markdown([1, 2], ["x", "y", "z"])
    assert "y" in out
    assert "z" in out
    assert "**2**" in out


# ── derive_v4_fields heuristic ─────────────────────────────────────────


@pytest.mark.parametrize(
    "alert,expected_type",
    [
        # Known-category attacks
        (
            {"ground_truth": "attack", "attack_category": "Spoofing",
             "risk_level": "CRITICAL", "risk_score": 0.95},
            AlertType.KNOWN_ATTACK,
        ),
        (
            {"ground_truth": "attack", "attack_category": "Data Alteration",
             "risk_level": "CRITICAL", "risk_score": 0.70},
            AlertType.KNOWN_ATTACK_UNCERTAIN,
        ),
        (
            {"ground_truth": "attack", "attack_category": "Spoofing",
             "risk_level": "MEDIUM", "risk_score": 0.55},
            AlertType.CONFIRMED_ANOMALY,
        ),
        (
            {"ground_truth": "attack", "attack_category": "Spoofing",
             "risk_level": "LOW", "risk_score": 0.30},
            AlertType.SUSPICIOUS_PATTERN,
        ),
        # Novel (non-cataloged) attacks
        (
            {"ground_truth": "attack", "attack_category": "Reconnaissance",
             "risk_level": "CRITICAL", "risk_score": 0.92},
            AlertType.STRONG_NOVEL_ANOMALY,
        ),
        (
            {"ground_truth": "attack", "attack_category": "Reconnaissance",
             "risk_level": "HIGH", "risk_score": 0.78},
            AlertType.NOVEL_ANOMALY,
        ),
        # Benign rows
        (
            {"ground_truth": "benign", "attack_category": "normal",
             "risk_level": "LOW", "risk_score": 0.10},
            AlertType.BENIGN_WATCH,
        ),
        # Benign with high risk = false-positive look = adversarial cue
        (
            {"ground_truth": "benign", "attack_category": "normal",
             "risk_level": "CRITICAL", "risk_score": 0.91},
            AlertType.DISAGREEMENT_ANOMALY,
        ),
    ],
)
def test_derive_v4_fields_alert_type(alert, expected_type):
    at, _, _ = derive_v4_fields(alert)
    assert at == expected_type


@pytest.mark.parametrize(
    "score,expected",
    [
        (0.95, Confidence.VERY_HIGH),
        (0.85, Confidence.VERY_HIGH),
        (0.84, Confidence.HIGH),
        (0.70, Confidence.HIGH),
        (0.69, Confidence.MEDIUM),
        (0.50, Confidence.MEDIUM),
        (0.49, Confidence.LOW),
        (0.0, Confidence.LOW),
    ],
)
def test_derive_v4_fields_confidence_thresholds(score, expected):
    alert = {
        "ground_truth": "attack",
        "attack_category": "Spoofing",
        "risk_level": "CRITICAL",
        "risk_score": score,
    }
    _, conf, _ = derive_v4_fields(alert)
    assert conf == expected


def test_derive_v4_fields_mode_explicit_field_wins(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    alert = {
        "ground_truth": "attack",
        "attack_category": "Spoofing",
        "risk_level": "CRITICAL",
        "risk_score": 0.9,
        "generation_mode": MODE_A_LLM,
    }
    _, _, mode = derive_v4_fields(alert)
    assert mode == MODE_A_LLM


def test_derive_v4_fields_mode_falls_back_to_env(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    alert = {"ground_truth": "attack", "attack_category": "Spoofing",
             "risk_level": "CRITICAL", "risk_score": 0.9}
    _, _, mode = derive_v4_fields(alert)
    assert mode == MODE_B_RULE_BASED

    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    _, _, mode = derive_v4_fields(alert)
    assert mode == MODE_A_LLM


def test_derive_v4_fields_handles_missing_fields():
    """An empty dict should not crash; should classify as BENIGN_WATCH."""
    at, conf, _ = derive_v4_fields({})
    assert at == AlertType.BENIGN_WATCH
    assert conf == Confidence.LOW


def test_derive_v4_fields_garbage_score_is_low_confidence():
    at, conf, _ = derive_v4_fields(
        {"ground_truth": "attack", "attack_category": "Spoofing",
         "risk_level": "CRITICAL", "risk_score": "n/a"}
    )
    assert at == AlertType.KNOWN_ATTACK_UNCERTAIN  # CRITICAL+known but score=0
    assert conf == Confidence.LOW


# ── End-to-end: derivation over real evaluation_alerts.json ────────────


@pytest.mark.skipif(not EVAL_PATH.exists(), reason="evaluation_alerts.json not present")
def test_derive_v4_fields_over_real_alerts():
    """Every alert in the evaluation set classifies without crashing."""
    alerts = json.loads(EVAL_PATH.read_text())
    types_seen: set[AlertType] = set()
    for a in alerts:
        at, conf, mode = derive_v4_fields(a)
        assert isinstance(at, AlertType)
        assert isinstance(conf, Confidence)
        assert mode in (MODE_A_LLM, MODE_B_RULE_BASED)
        types_seen.add(at)
    # Sanity: at least 3 distinct categories appear in 20 mixed alerts.
    assert len(types_seen) >= 3, f"too uniform a distribution: {types_seen}"
