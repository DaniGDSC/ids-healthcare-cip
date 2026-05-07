"""State-shape tests for the top-bar globals in ``module6_app.py``.

Streamlit's ``AppTest`` framework would let us drive page changes and
assert ``session_state`` persists across them, but it pulls in a heavy
runtime that the rest of this codebase doesn't use. This file verifies
the deterministic pieces — defaults, accessors, role↔renderer mapping —
without booting Streamlit. It's the closest you can get to "the role
selector persists" without an actual browser.

Persistence across page changes is a Streamlit *guarantee* (anything
keyed by ``key=...`` in ``session_state`` survives reruns); we verify
the keys and defaults are wired up correctly so that guarantee applies.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from module6_evaluation.module6_app import (
    DEMO_ALERT_LIMIT,
    ROLE_DISPLAY_LABEL,
    ROLES,
    _ROLE_TO_LEGACY_VIEW,
    get_current_role,
    get_demo_mode,
    init_session,
    set_current_mode,
)
from module6_evaluation.presentation_v4 import MODE_A_LLM, MODE_B_RULE_BASED


@pytest.fixture
def fake_session(monkeypatch):
    """Replace ``st.session_state`` with a plain dict for the duration of a test."""
    state: dict = {}
    fake_st = MagicMock()
    fake_st.session_state = state
    monkeypatch.setattr("module6_evaluation.module6_app.st", fake_st)
    return state


# ── Constants ─────────────────────────────────────────────────────────


def test_roles_tuple_has_three_distinct_entries():
    assert len(ROLES) == 3
    assert len(set(ROLES)) == 3


def test_role_display_labels_cover_every_role():
    assert set(ROLE_DISPLAY_LABEL) == set(ROLES)
    # Every label has an emoji prefix per the spec ("🖥️ ...", "⚕️ ...", "👩‍⚕️ ...").
    for role, label in ROLE_DISPLAY_LABEL.items():
        assert role in label


def test_role_to_legacy_view_covers_every_role():
    assert set(_ROLE_TO_LEGACY_VIEW) == set(ROLES)
    # Spec invariant (per CLAUDE.md scope rules):
    # IT Generalist → Security Analyst (network/IDS view)
    # Biomed       → Administrator    (device-fleet / biomed-engineering view)
    # Nurse        → Clinician        (patient-care view)
    assert _ROLE_TO_LEGACY_VIEW["IT Generalist"] == "Security Analyst"
    assert _ROLE_TO_LEGACY_VIEW["Biomed"]        == "Administrator"
    assert _ROLE_TO_LEGACY_VIEW["Nurse"]         == "Clinician"


def test_demo_alert_limit_is_five():
    assert DEMO_ALERT_LIMIT == 5


# ── init_session() defaults ───────────────────────────────────────────


def test_init_session_sets_top_bar_defaults(fake_session):
    init_session()
    assert fake_session["role"] == "IT Generalist"
    assert fake_session["demo_mode"] is False
    assert fake_session["latest_mve_mode"] is None


def test_init_session_is_idempotent(fake_session):
    """Second call must not overwrite an existing user selection."""
    init_session()
    fake_session["role"] = "Biomed"
    fake_session["demo_mode"] = True
    init_session()
    assert fake_session["role"] == "Biomed"
    assert fake_session["demo_mode"] is True


# ── Accessors ─────────────────────────────────────────────────────────


def test_get_current_role_returns_default_when_unset(fake_session):
    assert get_current_role() == "IT Generalist"


def test_get_current_role_round_trips_through_session(fake_session):
    fake_session["role"] = "Nurse"
    assert get_current_role() == "Nurse"


def test_get_demo_mode_default_false(fake_session):
    assert get_demo_mode() is False


def test_get_demo_mode_round_trips(fake_session):
    fake_session["demo_mode"] = True
    assert get_demo_mode() is True


def test_set_current_mode_writes_session(fake_session):
    set_current_mode(MODE_A_LLM)
    assert fake_session["latest_mve_mode"] == MODE_A_LLM
    set_current_mode(MODE_B_RULE_BASED)
    assert fake_session["latest_mve_mode"] == MODE_B_RULE_BASED


def test_set_current_mode_rejects_garbage(fake_session):
    """Defensive: only the two canonical strings update state."""
    set_current_mode(MODE_A_LLM)
    set_current_mode("__not_a_mode__")
    assert fake_session["latest_mve_mode"] == MODE_A_LLM  # unchanged
