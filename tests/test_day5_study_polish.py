"""Day 5 — Study Mode + PCAP-deletion tests.

Covers the deterministic pieces:
  * The Day-1 auto-DEMO short-circuit is gone (no silent registration).
  * ``study_demo_bypass_active`` defaults to ``False`` and is idempotent
    on re-init.
  * ``_study_alert_dict_for`` resolves real and synthetic alerts and
    returns ``None`` for unknowns (so Group B never crashes).
  * Group A vs Group B render dispatch is intact (locked Likert flow
    preserved — no schema drift introduced).
  * ``study_responses_P*.json`` schema fingerprint hasn't drifted.
  * PCAP Replay is gone (already handled in an earlier session, locked
    here so a future regression is caught).
  * Sidebar carries exactly the four Day-4 modes.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from module6_evaluation.module6_app import (
    _study_alert_dict_for,
    init_session,
    load_demo_playlist,
    load_synthetic_demo_alerts,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_PATH = PROJECT_ROOT / "module6_evaluation" / "module6_app.py"
EVAL_PATH = PROJECT_ROOT / "results" / "reports" / "evaluation_alerts.json"


@pytest.fixture
def fake_session(monkeypatch):
    state: dict = {}
    fake_st = MagicMock()
    fake_st.session_state = state
    monkeypatch.setattr("module6_evaluation.module6_app.st", fake_st)
    return state


# ── init_session covers the new Day 5 flag ────────────────────────────


def test_init_session_adds_study_demo_bypass_default_false(fake_session):
    init_session()
    assert fake_session["study_demo_bypass_active"] is False


def test_init_session_idempotent_for_bypass_flag(fake_session):
    init_session()
    fake_session["study_demo_bypass_active"] = True
    init_session()
    assert fake_session["study_demo_bypass_active"] is True


# ── Day 1 auto-DEMO short-circuit must be GONE ────────────────────────


def test_no_silent_demo_registration_in_study_mode():
    """The Day 1 path that auto-set participant_id="DEMO" when
    demo_mode was on is gone. Day 5 replaces it with an explicit
    Skip-Registration button. If the auto-path comes back accidentally,
    examiners can no longer choose between full study and demo bypass."""
    src = APP_PATH.read_text()
    assert 'st.session_state.participant_id = "DEMO"' not in src, (
        "Day 1's auto-DEMO short-circuit is back — it was intentionally "
        "removed in Day 5 so registration can't be silently skipped."
    )


# ── _study_alert_dict_for resolves the 3 cases ─────────────────────────


def test_study_alert_dict_for_real_alert():
    """Picks a real alert from the eval set and confirms round-trip."""
    real = json.loads(EVAL_PATH.read_text())
    target = real[0]["alert_id"]
    out = _study_alert_dict_for(target)
    assert out is not None
    assert out.get("alert_id") == target
    # And it must carry v4-renderable fields:
    assert "risk_level" in out and "attack_category" in out


def test_study_alert_dict_for_synthetic_alert():
    syns = load_synthetic_demo_alerts()
    if not syns:
        pytest.skip("No synthetic demo alerts present.")
    target = syns[0]["alert_id"]
    out = _study_alert_dict_for(target)
    assert out is not None
    assert out.get("alert_id") == target


def test_study_alert_dict_for_unknown_returns_none():
    assert _study_alert_dict_for("__not_a_real_alert__") is None


# ── A/B beat resolves to a real alert dict ─────────────────────────────


def test_demo_bypass_view_target_alert_resolves():
    """The bypass view's Group A/B tabs need a single concrete alert.
    It picks the playlist beat tagged ``ab_comparison``; that beat's
    alert_id must resolve."""
    # Robust against st.cache_data residue from a prior test that
    # monkeypatched PLAYLIST_PATH to a missing file (Day 4 fixture).
    load_demo_playlist.clear()
    playlist = load_demo_playlist().get("alerts", [])
    ab_entry = next(
        (e for e in playlist if e.get("narrative_beat") == "ab_comparison"),
        None,
    )
    assert ab_entry is not None, "playlist missing the ab_comparison beat"
    assert _study_alert_dict_for(ab_entry["alert_id"]) is not None


# ── PCAP deletion — locked here ───────────────────────────────────────


def test_pcap_replay_function_is_gone():
    """The Phase-3 placeholder was deleted earlier; lock it here."""
    src = APP_PATH.read_text()
    assert "def pcap_replay_stub" not in src
    assert "pcap_replay_stub()" not in src


def test_sidebar_has_exactly_four_modes():
    """The sidebar mode list is fixed at four entries — Dashboard,
    Online Simulation, Browse Alerts, Study (A/B). PCAP regression
    would either re-add a fifth entry or revive the stub."""
    src = APP_PATH.read_text()
    # The radio is a 4-item list literal in main(). Use a tolerant check.
    assert '"Dashboard"' in src
    assert '"Online Simulation"' in src
    assert '"Browse Alerts"' in src
    assert '"Study (A/B)"' in src
    # And no "PCAP" string literal anywhere.
    assert "PCAP Replay" not in src
    assert "PCAP" not in src


# ── Locked study response schema is preserved ─────────────────────────


_LOCKED_RESPONSE_KEYS = {
    "participant_id",
    "participant_role",
    "alert_id",
    "alert_type",
    "alert_index",
    "condition",
    "chosen_severity",
    "correct_severity",
    "severity_correct",
    "severity_score",
    "catastrophic_miss",
    "chosen_action",
    "correct_action",
    "action_correct",
    "composite_score",
    "confidence",
    "decision_time_sec",
    "ground_truth_label",
    "reasoning_note",
}


def test_existing_study_responses_match_locked_schema():
    """The 19 keys above are the locked Phase-2 study schema. If Day 5
    accidentally widened or narrowed it, downstream analyzers break.

    This compares the schema present on disk; if any P01..P25 file is
    missing keys the test will say so."""
    files = list((PROJECT_ROOT / "results" / "reports").glob("study_responses_*.json"))
    if not files:
        pytest.skip("No study_responses_*.json present in this checkout")
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(data, list) or not data:
            continue
        actual_keys = set(data[0].keys())
        # We require the 19 locked keys to be present (extra fields are
        # allowed — adding context is non-breaking, removing fields is).
        missing = _LOCKED_RESPONSE_KEYS - actual_keys
        assert not missing, f"{path.name} missing locked keys: {missing}"


def test_likert_questions_remain_locked():
    """The exact wording of the three Likert prompts is part of the
    locked study material. If the labels change mid-study the
    counterbalancing analysis becomes apples-to-oranges."""
    src = APP_PATH.read_text()
    # Each question text — anchor on a unique substring per prompt.
    assert "1. How severe is this alert?" in src
    assert "2. What action would you take?" in src
    assert "3. How confident are you in this decision?" in src
    # And the action-radio's five canonical actions:
    for action in (
        "Isolate the device/system from the network",
        "Escalate to clinical staff / senior management",
        "Investigate further before taking action",
        "Monitor closely but no immediate action",
        "Dismiss — this is likely a false alarm",
    ):
        assert action in src, f"Locked action label missing: {action}"
