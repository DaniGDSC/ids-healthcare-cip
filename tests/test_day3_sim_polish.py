"""Day 3 — Online Simulation polish: tests for the new helpers + state.

Covers the deterministic, framework-free pieces:
  * :func:`_is_safety_floor_alert` totality across the (risk_level,
    device_patchable) truth table.
  * Auto-pause once-per-index latching state machine — modelled in pure
    Python so we don't need ``streamlit.testing`` to drive the fragment.
  * ``init_session`` adds the Day 3 keys (researcher_mode,
    auto_paused_at_index, safety_floor_banner) with the right defaults.
  * Real evaluation_alerts.json sanity: at least one alert in the eval
    set is a safety-floor alert (otherwise the demo can't show it).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from module6_evaluation.module6_app import (
    _is_safety_floor_alert,
    init_session,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_PATH = PROJECT_ROOT / "results" / "reports" / "evaluation_alerts.json"


@pytest.fixture
def fake_session(monkeypatch):
    """Replace ``st.session_state`` with a plain dict for the test."""
    state: dict = {}
    fake_st = MagicMock()
    fake_st.session_state = state
    monkeypatch.setattr("module6_evaluation.module6_app.st", fake_st)
    return state


# ── _is_safety_floor_alert ────────────────────────────────────────────


@pytest.mark.parametrize(
    "alert,expected",
    [
        ({"risk_level": "CRITICAL", "device_patchable": False}, True),
        ({"risk_level": "CRITICAL", "device_patchable": True},  False),
        ({"risk_level": "HIGH",     "device_patchable": False}, False),
        ({"risk_level": "HIGH",     "device_patchable": True},  False),
        ({"risk_level": "MEDIUM",   "device_patchable": False}, False),
        ({"risk_level": "LOW",      "device_patchable": False}, False),
        ({"risk_level": "critical", "device_patchable": False}, True),  # case-insensitive
    ],
)
def test_safety_floor_truth_table(alert, expected):
    assert _is_safety_floor_alert(alert) is expected


def test_safety_floor_missing_patchable_defaults_safe():
    """No ``device_patchable`` field ⇒ assumed patchable ⇒ no auto-pause."""
    assert _is_safety_floor_alert({"risk_level": "CRITICAL"}) is False


def test_safety_floor_empty_alert_does_not_pause():
    assert _is_safety_floor_alert({}) is False


# ── init_session adds Day 3 keys with right defaults ──────────────────


def test_init_session_adds_day3_keys(fake_session):
    init_session()
    assert fake_session["researcher_mode"] is False
    assert fake_session["auto_paused_at_index"] is None
    assert fake_session["safety_floor_banner"] is False


def test_init_session_idempotent_for_day3_keys(fake_session):
    """User flips researcher_mode on; second init_session call must not
    silently flip it back."""
    init_session()
    fake_session["researcher_mode"] = True
    fake_session["auto_paused_at_index"] = 5
    fake_session["safety_floor_banner"] = True
    init_session()
    assert fake_session["researcher_mode"] is True
    assert fake_session["auto_paused_at_index"] == 5
    assert fake_session["safety_floor_banner"] is True


# ── Auto-pause latching state machine (pure-function model) ──────────
#
# The Sim's auto-pause is enforced by an inline check in
# ``simulation_mode``. To verify the semantics without booting Streamlit,
# we model the same state machine here and exercise the four scenarios
# the latching strategy is meant to handle.


def _step(state: dict, alert: dict) -> dict:
    """One tick of the auto-pause check, mirroring simulation_mode."""
    if (
        state["sim_running"]
        and state["auto_paused_at_index"] != state["sim_index"]
        and _is_safety_floor_alert(alert)
    ):
        state["sim_running"] = False
        state["auto_paused_at_index"] = state["sim_index"]
        state["safety_floor_banner"] = True
    return state


SF = {"risk_level": "CRITICAL", "device_patchable": False}   # safety-floor
OK = {"risk_level": "HIGH",     "device_patchable": True}    # ok


def test_auto_pause_fires_on_first_safety_floor():
    state = {"sim_running": True, "sim_index": 0,
             "auto_paused_at_index": None, "safety_floor_banner": False}
    _step(state, SF)
    assert state["sim_running"] is False
    assert state["safety_floor_banner"] is True
    assert state["auto_paused_at_index"] == 0


def test_auto_pause_does_not_re_fire_on_same_index():
    """After Resume on the same index, the next tick must NOT auto-pause
    again (latching). That's how we avoid pause-thrash with 12 of 20
    alerts being CRITICAL+unpatchable in the eval set."""
    state = {"sim_running": True, "sim_index": 0,
             "auto_paused_at_index": None, "safety_floor_banner": False}
    _step(state, SF)
    # User clicks Resume:
    state["sim_running"] = True
    state["safety_floor_banner"] = False
    # auto_paused_at_index INTENTIONALLY left at 0 — that's the latch.
    _step(state, SF)
    assert state["sim_running"] is True
    assert state["safety_floor_banner"] is False


def test_auto_pause_re_fires_on_next_safety_floor_index():
    """After advancing to a fresh index that is also safety-floor, the
    pause should fire again — operator gets one explicit ack per
    distinct alert."""
    state = {"sim_running": True, "sim_index": 0,
             "auto_paused_at_index": 0, "safety_floor_banner": False}
    # Advance to new index — sim's auto-advance does this between ticks
    state["sim_index"] = 1
    _step(state, SF)
    assert state["sim_running"] is False
    assert state["auto_paused_at_index"] == 1


def test_auto_pause_skips_non_safety_floor():
    state = {"sim_running": True, "sim_index": 3,
             "auto_paused_at_index": None, "safety_floor_banner": False}
    _step(state, OK)
    assert state["sim_running"] is True
    assert state["safety_floor_banner"] is False


def test_auto_pause_inert_when_already_paused():
    """Manual pause already in effect — auto-pause must not re-trigger
    a banner."""
    state = {"sim_running": False, "sim_index": 0,
             "auto_paused_at_index": None, "safety_floor_banner": False}
    _step(state, SF)
    assert state["sim_running"] is False
    assert state["safety_floor_banner"] is False  # banner only when AUTO-paused


# ── End-to-end check on real eval data ─────────────────────────────────


def test_eval_set_contains_at_least_one_safety_floor_alert():
    """If this fails, the defense demo can't show INVARIANT 2 in action.
    Day 4 demo playlist would need to synthesize one."""
    alerts = json.loads(EVAL_PATH.read_text())
    n_safety = sum(1 for a in alerts if _is_safety_floor_alert(a))
    assert n_safety >= 1, (
        "No CRITICAL+unpatchable alerts in the evaluation set; "
        "auto-pause won't ever trigger during demo."
    )
