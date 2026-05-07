"""Day 2 — Dashboard polish: tests for the new helpers.

Covers:
  * Top-bar-role → :class:`OperatorRole` bridge totality.
  * MITRE per-role rendering across the 9-class taxonomy via
    ``format_mitre_for_alert_type`` (we don't re-test the formatter
    itself — that lives in ``module4_explanations`` — only that the
    Dashboard bridge plugs into it correctly).
  * ``_aggregate_study_decision_quality`` over real ``study_responses_*.json``
    files (and over a synthetic empty directory).
  * ``render_dae_top_features`` accepts the existing ``dae_top_features``
    shape and is silent on empty / wrong shapes.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from module6_evaluation.module6_app import (
    ROLES,
    _TOPBAR_TO_OPERATOR_ROLE,
    _aggregate_study_decision_quality,
    derive_v4_fields,
    get_current_operator_role,
    render_dae_top_features,
)
from module4_explanations.triage_v4_adapter import format_mitre_for_alert_type
from src.data_models import AlertType, OperatorRole

EVAL_DIR = Path(__file__).resolve().parent.parent / "results" / "reports"


# ── Role-string bridge ────────────────────────────────────────────────


def test_topbar_role_bridge_covers_every_role():
    assert set(_TOPBAR_TO_OPERATOR_ROLE) == set(ROLES)


@pytest.mark.parametrize(
    "topbar_role,enum_value",
    [
        ("IT Generalist", OperatorRole.IT_GENERALIST),
        ("Biomed",        OperatorRole.BIOMED_ENGINEER),
        ("Nurse",         OperatorRole.NURSE_MANAGER),
    ],
)
def test_topbar_role_maps_to_correct_enum(topbar_role, enum_value):
    assert _TOPBAR_TO_OPERATOR_ROLE[topbar_role] == enum_value


def test_get_current_operator_role_default(monkeypatch):
    fake_st = MagicMock()
    fake_st.session_state = {}
    monkeypatch.setattr("module6_evaluation.module6_app.st", fake_st)
    assert get_current_operator_role() == OperatorRole.IT_GENERALIST


def test_get_current_operator_role_round_trip(monkeypatch):
    fake_st = MagicMock()
    fake_st.session_state = {"role": "Nurse"}
    monkeypatch.setattr("module6_evaluation.module6_app.st", fake_st)
    assert get_current_operator_role() == OperatorRole.NURSE_MANAGER


def test_get_current_operator_role_unknown_falls_back(monkeypatch):
    fake_st = MagicMock()
    fake_st.session_state = {"role": "__not_a_role__"}
    monkeypatch.setattr("module6_evaluation.module6_app.st", fake_st)
    assert get_current_operator_role() == OperatorRole.IT_GENERALIST


# ── MITRE per role plugged into our heuristic ─────────────────────────


@pytest.mark.parametrize("at", list(AlertType))
@pytest.mark.parametrize("role", list(OperatorRole))
def test_format_mitre_for_alert_type_total(at, role):
    """The bridge composition must be total: every (alert_type, role)
    pair returns a non-empty string. The Layer 4 adapter's job is the
    actual content; the dashboard just trusts it not to raise or empty."""
    line = format_mitre_for_alert_type(at, role)
    assert isinstance(line, str)
    assert line.strip() != ""


def test_format_mitre_role_outputs_differ_for_known_attack():
    it_line     = format_mitre_for_alert_type(AlertType.KNOWN_ATTACK, OperatorRole.IT_GENERALIST)
    biomed_line = format_mitre_for_alert_type(AlertType.KNOWN_ATTACK, OperatorRole.BIOMED_ENGINEER)
    nurse_line  = format_mitre_for_alert_type(AlertType.KNOWN_ATTACK, OperatorRole.NURSE_MANAGER)
    assert "T1071" in it_line          # technique-id form for IT
    assert "T1071" not in biomed_line  # plain prose for biomed
    assert "T1071" not in nurse_line   # plain prose for nurse
    assert it_line != biomed_line != nurse_line


# ── Decision-quality aggregator ───────────────────────────────────────


def test_aggregate_study_decision_quality_handles_empty_dir(tmp_path, monkeypatch):
    """Empty directory ⇒ n=0 + None metrics; never raises."""
    monkeypatch.setattr("module6_evaluation.module6_app.EVAL_DIR", tmp_path)
    _aggregate_study_decision_quality.clear()  # st.cache_data: clear cached return
    out = _aggregate_study_decision_quality()
    assert out == {"n": 0, "avg_confidence": None, "followed_pct": None}


def test_aggregate_study_decision_quality_real_files():
    """Real study_responses_*.json should aggregate to non-empty stats."""
    _aggregate_study_decision_quality.clear()
    out = _aggregate_study_decision_quality()
    if out["n"] == 0:
        pytest.skip("No study_responses_*.json present in this checkout")
    # avg_confidence must be in [1, 5]
    assert out["avg_confidence"] is not None
    assert 1.0 <= out["avg_confidence"] <= 5.0
    # followed_pct must be in [0, 100]
    assert out["followed_pct"] is not None
    assert 0.0 <= out["followed_pct"] <= 100.0


def test_aggregate_study_decision_quality_skips_malformed(tmp_path, monkeypatch):
    """Malformed JSON / non-list payloads are ignored, not fatal."""
    (tmp_path / "study_responses_bad.json").write_text("{not json")
    (tmp_path / "study_responses_empty.json").write_text("[]")
    (tmp_path / "study_responses_dict.json").write_text('{"a": 1}')  # not a list
    (tmp_path / "study_responses_real.json").write_text(json.dumps([
        {"confidence": 4, "chosen_action": "isolate", "correct_action": "isolate"},
        {"confidence": 3, "chosen_action": "monitor", "correct_action": "isolate"},
    ]))
    monkeypatch.setattr("module6_evaluation.module6_app.EVAL_DIR", tmp_path)
    _aggregate_study_decision_quality.clear()
    out = _aggregate_study_decision_quality()
    assert out["n"] == 2
    assert out["avg_confidence"] == pytest.approx(3.5)
    assert out["followed_pct"] == pytest.approx(50.0)


# ── DAE top-features renderer (silent on bad input) ───────────────────


def test_render_dae_top_features_silent_on_empty(monkeypatch):
    fake_st = MagicMock()
    monkeypatch.setattr("module6_evaluation.module6_app.st", fake_st)
    render_dae_top_features([])
    render_dae_top_features(None)
    render_dae_top_features("not a list")
    fake_st.expander.assert_not_called()


def test_render_dae_top_features_renders_expander(monkeypatch):
    fake_st = MagicMock()
    fake_st.expander.return_value.__enter__ = lambda self: None
    fake_st.expander.return_value.__exit__ = lambda self, *a: None
    monkeypatch.setattr("module6_evaluation.module6_app.st", fake_st)
    items = [
        {"feature": "sMaxPktSz", "weighted_error": 2.47e-06, "pct_contribution": 34.3},
        {"feature": "SrcBytes",  "weighted_error": 1.13e-06, "pct_contribution": 15.7},
    ]
    render_dae_top_features(items)
    fake_st.expander.assert_called_once()
    title = fake_st.expander.call_args[0][0]
    assert "DAE Anomaly Details" in title
    assert "2 dim" in title


# ── End-to-end shim plumbing on real evaluation alerts ────────────────


def test_dashboard_layer1_mitre_renders_for_every_real_alert():
    """Sanity: derive_v4_fields → format_mitre_for_alert_type composition
    is total over the actual evaluation set, for all three roles."""
    alerts = json.loads((EVAL_DIR / "evaluation_alerts.json").read_text())
    for a in alerts:
        at, _, _ = derive_v4_fields(a)
        for role in OperatorRole:
            line = format_mitre_for_alert_type(at, role)
            assert line and isinstance(line, str)
