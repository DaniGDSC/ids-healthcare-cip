"""Day 7 — Visual-polish invariants: CSS, color palette, page titles.

These tests freeze the visual decisions made on Day 7 so a future
Streamlit / palette change has to be intentional and tracked.

What's tested:
  * ``PROJECTOR_CSS`` exists and carries the 17px + DO-NOT-box rules.
  * ``main()`` injects the CSS exactly once, immediately after
    ``st.set_page_config``.
  * Each of the 4 page entry functions emits a title that begins with
    an emoji prefix (so the projector audience sees a consistent
    visual anchor across pages).
  * ``SUSPICIOUS_PATTERN`` color is the Day-7 amber (``#F59E0B``), not
    the original ``#FACC15`` yellow that washed out next to
    ``CONFIRMED_ANOMALY`` on a projector.
  * The locked-since-Day-1 ``DISAGREEMENT_ANOMALY`` purple is
    untouched — Day 7 polish must not collide with the headline visual.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from module6_evaluation.module6_app import PROJECTOR_CSS
from module6_evaluation.presentation_v4 import BADGE_FOR_ALERT_TYPE
from src.data_models import AlertType

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_PATH = PROJECT_ROOT / "module6_evaluation" / "module6_app.py"


# ── PROJECTOR_CSS contents ─────────────────────────────────────────────


def test_projector_css_carries_17px_base():
    assert "font-size: 17px" in PROJECTOR_CSS


def test_projector_css_has_do_not_box_rule():
    """The DO-NOT red box has to keep its 2px-border rule — that's the
    most visually load-bearing element when an examiner glances at a
    CRITICAL+clinical alert."""
    assert ".do-not-box" in PROJECTOR_CSS
    assert "border: 2px solid #DC2626" in PROJECTOR_CSS


def test_projector_css_uses_data_testid_selectors():
    """Selectors should target Streamlit's stable ``data-testid``
    hooks rather than auto-generated CSS classes that change on every
    Streamlit minor version."""
    assert 'data-testid="stMetric"' in PROJECTOR_CSS
    assert 'data-testid="stButton"' in PROJECTOR_CSS
    # No reliance on auto-generated class names (would start with .css-).
    assert ".css-" not in PROJECTOR_CSS


# ── CSS injected exactly once in main() ────────────────────────────────


def test_main_injects_projector_css_once():
    src = APP_PATH.read_text()
    assert src.count("st.markdown(PROJECTOR_CSS, unsafe_allow_html=True)") == 1, (
        "PROJECTOR_CSS must be injected exactly once — duplicates double "
        "the inline-style cost on every rerun."
    )


def test_main_injects_css_after_set_page_config():
    """Streamlit ignores st.markdown calls issued before
    ``st.set_page_config``; the injection must come after."""
    src = APP_PATH.read_text()
    cfg_pos = src.find("st.set_page_config(page_title=\"IoMT IDS Dashboard\"")
    css_pos = src.find("st.markdown(PROJECTOR_CSS")
    assert cfg_pos != -1 and css_pos != -1
    assert cfg_pos < css_pos, (
        "PROJECTOR_CSS injection must follow st.set_page_config in main()."
    )


# ── Page titles share a consistent emoji-prefix format ────────────────


_EMOJI_PREFIX_RE = re.compile(r'st\.title\("[\\\\-￿]')


def test_dashboard_title_has_emoji_prefix():
    src = APP_PATH.read_text()
    # \U0001f512 = 🔒
    assert 'st.title("\\U0001f512 IoMT IDS \\u2014 Real-Time Dashboard")' in src


def test_simulation_title_has_emoji_prefix():
    src = APP_PATH.read_text()
    # \U0001f4e1 = 📡
    assert 'st.title("\\U0001f4e1 IoMT IDS \\u2014 Online Simulation")' in src


def test_browse_title_has_emoji_prefix():
    src = APP_PATH.read_text()
    # \U0001f4c2 = 📂
    assert 'st.title("\\U0001f4c2 IoMT Alert Browser")' in src


def test_study_titles_have_emoji_prefix():
    """Study has multiple titles (registration, completion, demo bypass).
    All clipboard-prefixed for consistency."""
    src = APP_PATH.read_text()
    # \U0001f4cb = 📋 (clipboard)
    assert 'st.title("\\U0001f4cb Healthcare IDS Alert Evaluation Study")' in src
    assert 'st.title("\\U0001f4cb Study Complete")' in src
    # The demo-bypass title was already prefixed in Day 5 — written
    # with a literal em-dash rather than the \\u2014 escape, so check
    # for the literal form.
    assert 'st.title("\\U0001f4cb Study Mode — A/B Demo Bypass")' in src


# ── Day 7 palette decision — SUSPICIOUS amber, others unchanged ──────


def test_suspicious_pattern_is_amber_not_yellow():
    """Day 7: SUSPICIOUS moved from #FACC15 (yellow) to #F59E0B (amber)
    so it is distinguishable from CONFIRMED_ANOMALY's #EAB308 yellow at
    projector contrast. If this test fails the color was reverted —
    re-run the projector simulation before reverting."""
    assert BADGE_FOR_ALERT_TYPE[AlertType.SUSPICIOUS_PATTERN]["color"] == "#F59E0B"


def test_suspicious_and_confirmed_have_different_colors():
    a = BADGE_FOR_ALERT_TYPE[AlertType.SUSPICIOUS_PATTERN]["color"]
    b = BADGE_FOR_ALERT_TYPE[AlertType.CONFIRMED_ANOMALY]["color"]
    assert a != b


def test_disagreement_anomaly_purple_unchanged():
    """The Day-2 purple lock must survive Day-7 polish: it's the
    headline v4 visual upgrade and any palette change here would land
    on the defense slide deck the same morning."""
    assert BADGE_FOR_ALERT_TYPE[AlertType.DISAGREEMENT_ANOMALY]["color"] == "#9333EA"


def test_critical_threat_red_unchanged():
    assert BADGE_FOR_ALERT_TYPE[AlertType.KNOWN_ATTACK]["color"] == "#DC2626"
    assert BADGE_FOR_ALERT_TYPE[AlertType.KNOWN_ATTACK_UNCERTAIN]["color"] == "#DC2626"
