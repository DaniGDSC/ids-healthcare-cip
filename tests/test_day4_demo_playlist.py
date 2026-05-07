"""Day 4 — Browse Mode + Demo Playlist tests.

Covers the deterministic pieces:
  * Playlist YAML loads with 5 beats in the right narrative order.
  * Every playlist alert_id resolves to a real alert OR a synthetic.
  * The synthetic adversarial classifies as DISAGREEMENT_ANOMALY through
    the v4 heuristic (no special-casing).
  * ``_get_alerts_for_demo_mode`` returns the full set when the toggle
    is OFF and the playlist when it's ON.
  * ``_filter_responses_for_demo_mode`` keeps real responses, drops
    synthetics, and preserves narrative order.
  * Empty / missing config files degrade gracefully (caching aside).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from module6_evaluation.module6_app import (
    _filter_responses_for_demo_mode,
    _get_alerts_for_demo_mode,
    _playlist_alert_ids,
    derive_v4_fields,
    load_demo_playlist,
    load_synthetic_demo_alerts,
)
from src.data_models import AlertType


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_PATH = PROJECT_ROOT / "results" / "reports" / "evaluation_alerts.json"


@pytest.fixture
def fake_session(monkeypatch):
    state: dict = {}
    fake_st = MagicMock()
    fake_st.session_state = state
    monkeypatch.setattr("module6_evaluation.module6_app.st", fake_st)
    return state


# ── Playlist YAML round-trip ──────────────────────────────────────────


def test_playlist_loads_with_five_beats():
    load_demo_playlist.clear()
    playlist = load_demo_playlist()
    assert isinstance(playlist, dict)
    assert len(playlist["alerts"]) == 5


def test_playlist_narrative_positions_are_1_to_5_in_order():
    load_demo_playlist.clear()
    positions = [a["narrative_position"] for a in load_demo_playlist()["alerts"]]
    assert positions == [1, 2, 3, 4, 5], (
        "Playlist must enumerate beats 1..5 contiguously and in order."
    )


def test_playlist_each_beat_has_required_fields():
    load_demo_playlist.clear()
    REQUIRED = {
        "alert_id",
        "narrative_position",
        "narrative_beat",
        "narrative_label",
        "narrative_short_desc",
        "time_budget_seconds",
    }
    for entry in load_demo_playlist()["alerts"]:
        missing = REQUIRED - entry.keys()
        assert not missing, f"Beat {entry.get('narrative_position')!r} missing {missing}"


def test_total_time_budget_under_eleven_minutes():
    load_demo_playlist.clear()
    total = sum(
        e.get("time_budget_seconds", 0)
        for e in load_demo_playlist()["alerts"]
    )
    assert total <= 660, f"Beat budgets sum to {total}s — over 11 min"


# ── Playlist ↔ alert resolution ────────────────────────────────────────


def test_every_playlist_id_resolves_to_real_or_synthetic():
    load_demo_playlist.clear()
    load_synthetic_demo_alerts.clear()
    real = json.loads(EVAL_PATH.read_text())
    real_ids = {a["alert_id"] for a in real}
    syn_ids = {a["alert_id"] for a in load_synthetic_demo_alerts()}
    missing = []
    for aid in _playlist_alert_ids():
        if aid not in real_ids and aid not in syn_ids:
            missing.append(aid)
    assert not missing, f"Unresolved playlist ids: {missing}"


def test_synthetic_alert_classifies_as_disagreement_anomaly():
    """The synthetic adversarial entry must round-trip through the v4
    heuristic to DISAGREEMENT_ANOMALY — otherwise the demo's purple
    badge never fires."""
    load_synthetic_demo_alerts.clear()
    syns = load_synthetic_demo_alerts()
    assert len(syns) >= 1
    for syn in syns:
        if syn["alert_id"] == "SYNTHETIC_DEMO_001":
            at, _, _ = derive_v4_fields(syn)
            assert at == AlertType.DISAGREEMENT_ANOMALY
            return
    pytest.fail("SYNTHETIC_DEMO_001 not present in synthetic alerts")


def test_synthetic_alerts_carry_marker():
    load_synthetic_demo_alerts.clear()
    for syn in load_synthetic_demo_alerts():
        assert syn.get("is_synthetic_demo") is True


# ── Demo-mode filter helpers ───────────────────────────────────────────


def test_get_alerts_demo_off_returns_full_set(fake_session):
    fake_session["demo_mode"] = False
    out = _get_alerts_for_demo_mode()
    real = json.loads(EVAL_PATH.read_text())
    assert len(out) == len(real)


def test_get_alerts_demo_on_returns_playlist_in_order(fake_session):
    load_demo_playlist.clear()
    load_synthetic_demo_alerts.clear()
    fake_session["demo_mode"] = True
    out = _get_alerts_for_demo_mode()
    out_ids = [a["alert_id"] for a in out]
    assert out_ids == _playlist_alert_ids(), (
        f"Demo Mode must return playlist in narrative order, got {out_ids}"
    )


def test_filter_responses_demo_off_passthrough(fake_session):
    fake_session["demo_mode"] = False
    fake_responses = [{"alert_id": "EVAL-9999"}, {"alert_id": "EVAL-9998"}]
    assert _filter_responses_for_demo_mode(fake_responses) == fake_responses


def test_filter_responses_demo_on_keeps_real_drops_synthetic(fake_session):
    """Sim has no precomputed response artefact for synthetic alerts, so
    they must be silently skipped — Sim shows fewer than 5 alerts when
    Demo Mode is on. That's by design (synthetic adversarial belongs
    in Browse, not Sim)."""
    load_demo_playlist.clear()
    fake_session["demo_mode"] = True
    fake_responses = [
        {"alert_id": "EVAL-2294"},
        {"alert_id": "EVAL-3058"},
        {"alert_id": "EVAL-9999"},  # not in playlist — should be filtered out
    ]
    out = _filter_responses_for_demo_mode(fake_responses)
    out_ids = [r["alert_id"] for r in out]
    assert "EVAL-2294" in out_ids
    assert "EVAL-3058" in out_ids
    assert "EVAL-9999" not in out_ids
    assert "SYNTHETIC_DEMO_001" not in out_ids
    # Order must follow playlist
    pids = _playlist_alert_ids()
    assert [pids.index(i) for i in out_ids] == sorted(pids.index(i) for i in out_ids)


# ── Robustness ────────────────────────────────────────────────────────


def test_load_demo_playlist_missing_returns_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "module6_evaluation.module6_app.PLAYLIST_PATH",
        tmp_path / "no_such_file.yaml",
    )
    load_demo_playlist.clear()
    assert load_demo_playlist() == {"alerts": []}


def test_load_synthetic_alerts_missing_returns_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "module6_evaluation.module6_app.SYNTHETIC_ALERTS_PATH",
        tmp_path / "no_such_synthetic.yaml",
    )
    load_synthetic_demo_alerts.clear()
    assert load_synthetic_demo_alerts() == []
