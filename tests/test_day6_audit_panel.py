"""Day 6 — Last 5 Decisions audit-panel tests.

Covers the deterministic pieces:
  * Decision-event filter is correct (mechanical events filtered out).
  * Recent-decisions ordering is most-recent-first.
  * Chain integrity verifier honours session segments (each Streamlit
    session re-seeds the chain to ``0*64``; the file is a concatenation
    of segments, not one global chain).
  * Compact summary handles heterogeneous event schemas
    (response_submit / alert_response / online_interaction) and missing
    fields gracefully.
  * The audit panel call is wired into all 4 page entry points.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from module6_evaluation.module6_app import (
    DECISION_EVENT_TYPES,
    _CHAIN_SEED,
    _count_total_decisions,
    _decision_summary,
    _mark_decision_submitted,
    init_session,
    load_recent_decisions,
    verify_audit_chain_integrity,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_PATH = PROJECT_ROOT / "module6_evaluation" / "module6_app.py"
AUDIT_TRAIL_PATH = PROJECT_ROOT / "results" / "reports" / "audit_trail.jsonl"


@pytest.fixture
def fake_session(monkeypatch):
    state: dict = {}
    fake_st = MagicMock()
    fake_st.session_state = state
    monkeypatch.setattr("module6_evaluation.module6_app.st", fake_st)
    return state


@pytest.fixture
def temp_audit_trail(monkeypatch, tmp_path):
    """Point the audit-trail loader at a fresh file so tests don't share
    state with the real ``results/reports/audit_trail.jsonl`` or with
    each other (st.cache_data is cleared each call)."""
    p = tmp_path / "audit_trail.jsonl"
    monkeypatch.setattr("module6_evaluation.module6_app._AUDIT_TRAIL_PATH", p)
    load_recent_decisions.clear()
    _count_total_decisions.clear()
    verify_audit_chain_integrity.clear()
    return p


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _chain(records_no_hashes: list[dict]) -> list[dict]:
    """Helper — assigns prev_hash / integrity_hash so the chain links."""
    out = []
    prev = _CHAIN_SEED
    for i, r in enumerate(records_no_hashes):
        ihash = f"hash_{i:08d}"
        out.append({**r, "prev_hash": prev, "integrity_hash": ihash})
        prev = ihash
    return out


# ── Decision filter & ordering ────────────────────────────────────────


def test_decision_event_types_set_is_locked():
    assert DECISION_EVENT_TYPES == frozenset(
        {"response_submit", "alert_response", "online_interaction"}
    )


def test_load_recent_decisions_filters_mechanical_events(temp_audit_trail):
    records = _chain([
        {"event_type": "sim_pause",        "timestamp": "2026-05-01T10:00:00"},
        {"event_type": "response_submit",  "timestamp": "2026-05-01T10:01:00", "alert_id": "A"},
        {"event_type": "sim_jump",         "timestamp": "2026-05-01T10:02:00"},
        {"event_type": "alert_response",   "timestamp": "2026-05-01T10:03:00", "alert_id": "B"},
        {"event_type": "study_start",      "timestamp": "2026-05-01T10:04:00"},
        {"event_type": "online_interaction","timestamp": "2026-05-01T10:05:00", "alert_id": "C"},
    ])
    _write_jsonl(temp_audit_trail, records)
    out = load_recent_decisions(10)
    assert len(out) == 3
    assert {r["event_type"] for r in out} == {
        "response_submit", "alert_response", "online_interaction"
    }


def test_load_recent_decisions_most_recent_first(temp_audit_trail):
    records = _chain([
        {"event_type": "response_submit", "timestamp": "2026-05-01T10:00:00", "alert_id": "OLD"},
        {"event_type": "response_submit", "timestamp": "2026-05-01T10:30:00", "alert_id": "MID"},
        {"event_type": "response_submit", "timestamp": "2026-05-01T11:00:00", "alert_id": "NEW"},
    ])
    _write_jsonl(temp_audit_trail, records)
    out = load_recent_decisions(5)
    assert [r["alert_id"] for r in out] == ["NEW", "MID", "OLD"]


def test_load_recent_decisions_caps_at_n(temp_audit_trail):
    records = _chain(
        [{"event_type": "response_submit", "timestamp": f"2026-05-01T10:{i:02d}:00",
          "alert_id": f"A{i:02d}"} for i in range(10)]
    )
    _write_jsonl(temp_audit_trail, records)
    assert len(load_recent_decisions(5)) == 5
    load_recent_decisions.clear()
    assert len(load_recent_decisions(3)) == 3


def test_count_total_decisions_excludes_mechanical(temp_audit_trail):
    records = _chain([
        {"event_type": "sim_pause",       "timestamp": "2026-05-01T10:00:00"},
        {"event_type": "response_submit", "timestamp": "2026-05-01T10:01:00"},
        {"event_type": "alert_response",  "timestamp": "2026-05-01T10:02:00"},
    ])
    _write_jsonl(temp_audit_trail, records)
    assert _count_total_decisions() == 2


def test_load_recent_decisions_missing_file_returns_empty(temp_audit_trail):
    # File never written
    assert load_recent_decisions(5) == []
    load_recent_decisions.clear()
    _count_total_decisions.clear()
    assert _count_total_decisions() == 0


def test_load_recent_decisions_skips_malformed_lines(temp_audit_trail, tmp_path):
    p = temp_audit_trail
    p.write_text(
        json.dumps(_chain([{"event_type": "response_submit",
                            "timestamp": "2026-05-01T10:00:00",
                            "alert_id": "OK"}])[0]) + "\n"
        + "{not json\n"
        + "\n"
        + json.dumps({"event_type": "response_submit",
                      "timestamp": "2026-05-01T11:00:00",
                      "alert_id": "OK2",
                      "prev_hash": "hash_00000000",
                      "integrity_hash": "hash_00000001"}) + "\n",
        encoding="utf-8",
    )
    out = load_recent_decisions(5)
    assert {r["alert_id"] for r in out} == {"OK", "OK2"}


# ── Chain integrity ───────────────────────────────────────────────────


def test_chain_integrity_empty_log_returns_true(temp_audit_trail):
    assert verify_audit_chain_integrity() is True


def test_chain_integrity_valid_single_session(temp_audit_trail):
    records = _chain([
        {"event_type": "sim_pause",       "timestamp": "2026-05-01T10:00:00"},
        {"event_type": "response_submit", "timestamp": "2026-05-01T10:01:00"},
        {"event_type": "alert_response",  "timestamp": "2026-05-01T10:02:00"},
    ])
    _write_jsonl(temp_audit_trail, records)
    assert verify_audit_chain_integrity() is True


def test_chain_integrity_valid_across_session_restarts(temp_audit_trail):
    """Each Streamlit session restarts at prev_hash='0'*64.
    Multiple concatenated segments are still considered valid."""
    seg1 = _chain([
        {"event_type": "response_submit", "timestamp": "2026-05-01T10:00:00"},
        {"event_type": "response_submit", "timestamp": "2026-05-01T10:01:00"},
    ])
    seg2 = _chain([  # New session: chain restarts from seed.
        {"event_type": "alert_response",  "timestamp": "2026-05-01T11:00:00"},
        {"event_type": "alert_response",  "timestamp": "2026-05-01T11:01:00"},
    ])
    _write_jsonl(temp_audit_trail, seg1 + seg2)
    assert verify_audit_chain_integrity() is True


def test_chain_integrity_broken_link_returns_false(temp_audit_trail):
    records = _chain([
        {"event_type": "response_submit", "timestamp": "2026-05-01T10:00:00"},
        {"event_type": "response_submit", "timestamp": "2026-05-01T10:01:00"},
    ])
    # Tamper: rewrite the second record's prev_hash to something arbitrary
    records[1]["prev_hash"] = "TAMPERED"
    _write_jsonl(temp_audit_trail, records)
    assert verify_audit_chain_integrity() is False


def test_chain_integrity_missing_hash_fields_returns_false(temp_audit_trail):
    bad = {"event_type": "response_submit", "timestamp": "2026-05-01T10:00:00"}
    _write_jsonl(temp_audit_trail, [bad])
    assert verify_audit_chain_integrity() is False


def test_chain_integrity_on_real_audit_trail():
    """Sanity: the on-disk audit_trail.jsonl in this checkout chains
    cleanly across all session segments. If this fails, the writer or
    a third-party tool tampered with the file."""
    if not AUDIT_TRAIL_PATH.exists():
        pytest.skip("audit_trail.jsonl missing")
    verify_audit_chain_integrity.clear()
    assert verify_audit_chain_integrity() is True


# ── _decision_summary normalises heterogeneous event schemas ──────────


def test_summary_response_submit_uses_action_field():
    s = _decision_summary({
        "event_type": "response_submit",
        "timestamp": "2026-05-07T14:30:45",
        "alert_id": "EVAL-3544",
        "action": "isolate",
        "confidence": 4,
        "role": "IT Generalist",
    })
    assert s["Time"] == "14:30:45"
    assert s["Alert"] == "EVAL-3544"
    assert s["Role"] == "🖥️ IT"
    assert s["Event"] == "Submit"
    assert s["Action"] == "isolate"
    assert s["Conf"] == "4"


def test_summary_alert_response_uses_condition_field():
    s = _decision_summary({
        "event_type": "alert_response",
        "timestamp": "2026-05-07T14:30:45",
        "alert_id": "EVAL-3407",
        "condition": "with_mve",
        "role": "Nurse",
    })
    assert s["Event"] == "Study"
    assert s["Action"] == "with_mve"
    assert s["Role"] == "👩‍⚕️ RN"


def test_summary_online_interaction_uses_action_type():
    s = _decision_summary({
        "event_type": "online_interaction",
        "timestamp": "2026-05-07T14:30:45",
        "alert_id": 17,
        "action_type": "confirm",
        "role": "Biomed",
    })
    assert s["Event"] == "Sim"
    assert s["Action"] == "confirm"
    assert s["Role"] == "⚕️ Bio"


def test_summary_handles_missing_fields():
    s = _decision_summary({})
    assert s["Time"] == "—"
    assert s["Alert"] == "—"
    assert s["Role"] == "—"
    assert s["Action"] == "—"
    assert s["Conf"] == "—"


def test_summary_truncates_long_action():
    long_action = "Snapshot device for forensics and notify L2 specialist immediately"
    s = _decision_summary({
        "event_type": "response_submit",
        "timestamp": "2026-05-07T14:30:45",
        "alert_id": "X",
        "action": long_action,
    })
    assert len(s["Action"]) <= 28
    assert s["Action"].endswith("…")


# ── _mark_decision_submitted sets the auto-expand flag ───────────────


def test_mark_decision_submitted_flips_flag(fake_session):
    init_session()
    assert fake_session["audit_panel_just_submitted"] is False
    _mark_decision_submitted()
    assert fake_session["audit_panel_just_submitted"] is True


def test_init_session_adds_audit_panel_flag(fake_session):
    init_session()
    assert fake_session["audit_panel_just_submitted"] is False


# ── Panel call wired into all 4 page entry points ─────────────────────


def test_panel_call_wired_into_all_four_pages():
    """The panel must be reachable from every page so examiners see the
    same audit log no matter where they navigate. Locked here to catch
    accidental removals during a future page rewrite."""
    src = APP_PATH.read_text()
    # Each page function appends one call. Anchor on the function name
    # then check the panel call appears between it and the next def.
    import re
    for fn in ("dashboard_mode", "simulation_mode", "browse_mode", "study_mode"):
        m = re.search(rf"^def {fn}\(", src, re.MULTILINE)
        assert m, f"page function {fn} not found"
        nxt = re.search(r"^def [A-Za-z_]+\(", src[m.end():], re.MULTILINE)
        body_end = m.end() + (nxt.start() if nxt else len(src))
        body = src[m.start():body_end]
        assert "render_last_5_decisions_panel()" in body, (
            f"{fn} is missing the Day 6 audit-panel call"
        )
