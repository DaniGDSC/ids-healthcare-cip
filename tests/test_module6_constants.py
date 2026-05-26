"""Module 6 constants — role names, tier colors, action priority maps."""
from __future__ import annotations

import pytest

from module6_evaluation.constants import (
    ACTIONS,
    PAGE_SPLIT,
    ROLE_DISPLAY_LIST,
    ROLE_DISPLAY_NAMES,
    ROLE_INTERNAL_KEY,
    ROLE_ORDER,
    ROLE_SHORT_LABELS,
    TIER_COLORS,
    TIER_STREAMLIT_COLORS,
    _ACTION_DISPLAY,
    _ACTION_PRIORITY,
    _CATEGORY_TO_DEVICE,
    _CRIT_COLOR_HEX,
    _SPLIT_FILES,
    resolve_suffix,
)


def test_role_display_names_three_roles():
    assert set(ROLE_DISPLAY_NAMES.keys()) == {"analyst", "administrator", "clinician"}
    assert ROLE_DISPLAY_NAMES["analyst"] == "IT Generalist"
    assert ROLE_DISPLAY_NAMES["administrator"] == "Biomed Engineer"
    assert ROLE_DISPLAY_NAMES["clinician"] == "Nurse Manager"


def test_role_internal_key_is_inverse():
    for k, v in ROLE_DISPLAY_NAMES.items():
        assert ROLE_INTERNAL_KEY[v] == k


def test_role_order_three_canonical():
    assert ROLE_ORDER == ("analyst", "administrator", "clinician")


def test_role_display_list_matches_order():
    assert ROLE_DISPLAY_LIST == [ROLE_DISPLAY_NAMES[k] for k in ROLE_ORDER]


def test_role_short_labels_present():
    assert ROLE_SHORT_LABELS["analyst"] == "IT"
    assert ROLE_SHORT_LABELS["administrator"] == "Biomed"
    assert ROLE_SHORT_LABELS["clinician"] == "Nurse"


def test_actions_canonical_order():
    assert ACTIONS == ["dismiss", "monitor", "investigate", "isolate", "escalate"]


def test_tier_colors_four_tiers():
    assert set(TIER_COLORS.keys()) == {"CRITICAL", "HIGH", "MEDIUM", "LOW"}


def test_tier_streamlit_colors_named():
    # Streamlit's inline-color syntax accepts only named colors.
    valid = {"red", "blue", "green", "orange", "violet", "yellow", "gray"}
    for tier, color in TIER_STREAMLIT_COLORS.items():
        assert color in valid


def test_action_priority_buckets():
    expected = {"isolate", "escalate", "investigate", "monitor"}
    actual = set(_ACTION_PRIORITY.values())
    assert actual == expected


def test_action_display_eight_actions():
    assert len(_ACTION_DISPLAY) == 8
    # First element of each tuple is rank — should be unique-ish for sorting.
    ranks = [v[0] for v in _ACTION_DISPLAY.values()]
    assert max(ranks) <= 8


def test_crit_color_hex_four_tiers():
    assert set(_CRIT_COLOR_HEX.keys()) == {"CRITICAL", "HIGH", "MEDIUM", "LOW"}


def test_category_to_device_known_categories():
    assert "Spoofing" in _CATEGORY_TO_DEVICE
    assert "Data Alteration" in _CATEGORY_TO_DEVICE
    assert _CATEGORY_TO_DEVICE["Spoofing"] == "iomt_device"


def test_page_split_known_pages():
    assert set(PAGE_SPLIT.keys()) == {
        "Dashboard", "Online Simulation", "Browse Alerts",
        "Study (A/B)", "PCAP Replay",
    }
    assert PAGE_SPLIT["Dashboard"] == "test"
    assert PAGE_SPLIT["Online Simulation"] == "demo"


def test_split_files_test_no_suffix():
    assert _SPLIT_FILES["test"] == ""
    assert _SPLIT_FILES["demo"] == "_demo"


# ── resolve_suffix ─────────────────────────────────────────────────────


def test_resolve_suffix_none_returns_empty():
    assert resolve_suffix(None) == ""


def test_resolve_suffix_test_empty():
    assert resolve_suffix("test") == ""


def test_resolve_suffix_demo_demo():
    assert resolve_suffix("demo") == "_demo"


def test_resolve_suffix_unknown_raises():
    # Y9 enforcement: strict validation via common.split_paths.suffix.
    with pytest.raises(ValueError):
        resolve_suffix("staging")


def test_resolve_suffix_case_sensitive():
    with pytest.raises(ValueError):
        resolve_suffix("TEST")
