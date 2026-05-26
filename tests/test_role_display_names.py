"""Role display-name consistency guard.

After the spec-triad rename, the canonical role display names live in
`module6_evaluation.module6_app.ROLE_DISPLAY_NAMES`:

    analyst       → IT Generalist
    administrator → Biomed Engineer
    clinician     → Nurse Manager

This test pins:
  1. The mapping exists and uses exactly those three internal keys.
  2. The mapping values match the spec triad strings.
  3. `config/role_action_authorization.yaml::roles.*.display_name`
     stays in sync with the in-code mapping.
  4. Inverse map round-trips correctly.
  5. The short labels (Dashboard pills) are well-formed.
  6. Legacy alias `ROLES` now contains the spec triad in canonical order.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EXPECTED_DISPLAY = {
    "analyst": "IT Generalist",
    "administrator": "Biomed Engineer",
    "clinician": "Nurse Manager",
}


@pytest.fixture(scope="module")
def m6():
    from module6_evaluation import module6_app
    return module6_app


@pytest.fixture(scope="module")
def role_yaml():
    path = PROJECT_ROOT / "config" / "role_action_authorization.yaml"
    if not path.exists():
        pytest.skip(f"{path} missing")
    with open(path) as f:
        return yaml.safe_load(f)


def test_role_display_names_const_present(m6):
    assert hasattr(m6, "ROLE_DISPLAY_NAMES"), (
        "module6_app.ROLE_DISPLAY_NAMES missing — rename was reverted?"
    )
    assert isinstance(m6.ROLE_DISPLAY_NAMES, dict)


def test_role_display_names_match_spec_triad(m6):
    assert m6.ROLE_DISPLAY_NAMES == EXPECTED_DISPLAY, (
        f"Display names drifted from spec triad: {m6.ROLE_DISPLAY_NAMES} "
        f"vs expected {EXPECTED_DISPLAY}"
    )


def test_role_internal_key_round_trip(m6):
    """ROLE_INTERNAL_KEY must be the inverse of ROLE_DISPLAY_NAMES."""
    for key, display in m6.ROLE_DISPLAY_NAMES.items():
        assert m6.ROLE_INTERNAL_KEY[display] == key, (
            f"round-trip failed for {key}: ROLE_INTERNAL_KEY[{display!r}] "
            f"= {m6.ROLE_INTERNAL_KEY[display]!r}"
        )


def test_role_order_uses_canonical_keys(m6):
    assert set(m6.ROLE_ORDER) == set(EXPECTED_DISPLAY.keys()), (
        f"ROLE_ORDER drift: {m6.ROLE_ORDER} vs expected keys "
        f"{sorted(EXPECTED_DISPLAY.keys())}"
    )
    assert len(m6.ROLE_ORDER) == 3


def test_role_display_list_matches_order(m6):
    """ROLE_DISPLAY_LIST must be display names in ROLE_ORDER order."""
    expected = [m6.ROLE_DISPLAY_NAMES[k] for k in m6.ROLE_ORDER]
    assert m6.ROLE_DISPLAY_LIST == expected


def test_role_short_labels_present(m6):
    """Dashboard pills need short labels for each canonical key."""
    assert hasattr(m6, "ROLE_SHORT_LABELS")
    assert set(m6.ROLE_SHORT_LABELS.keys()) == set(EXPECTED_DISPLAY.keys())
    # Each short label should be 2-10 chars (Dashboard pill width budget)
    for key, label in m6.ROLE_SHORT_LABELS.items():
        assert 2 <= len(label) <= 10, (
            f"short label for {key} = {label!r} is outside [2,10] char budget"
        )


def test_legacy_ROLES_uses_spec_triad(m6):
    """The legacy `ROLES` list must now point at the spec triad
    (preserved as an alias for external imports).
    """
    assert m6.ROLES == m6.ROLE_DISPLAY_LIST, (
        "ROLES alias drifted from ROLE_DISPLAY_LIST"
    )
    # And the values must be exactly the spec triad.
    assert set(m6.ROLES) == set(EXPECTED_DISPLAY.values())


def test_yaml_display_names_match_code(role_yaml, m6):
    """`config/role_action_authorization.yaml` must declare the same
    display_name strings as ROLE_DISPLAY_NAMES.
    """
    for key, info in role_yaml["roles"].items():
        assert key in m6.ROLE_DISPLAY_NAMES, (
            f"YAML has role key {key!r} that isn't in ROLE_DISPLAY_NAMES"
        )
        yaml_display = info.get("display_name")
        assert yaml_display == m6.ROLE_DISPLAY_NAMES[key], (
            f"YAML display_name for {key!r} = {yaml_display!r} "
            f"!= code mapping {m6.ROLE_DISPLAY_NAMES[key]!r}"
        )


def test_yaml_short_labels_match_code(role_yaml, m6):
    for key, info in role_yaml["roles"].items():
        yaml_short = info.get("short_label")
        assert yaml_short == m6.ROLE_SHORT_LABELS[key], (
            f"YAML short_label for {key!r} = {yaml_short!r} "
            f"!= code {m6.ROLE_SHORT_LABELS[key]!r}"
        )


def test_yaml_primary_view_uses_display_name(role_yaml, m6):
    """primary_view should start with the role's display name (e.g.
    'IT Generalist view') — catches drift where the YAML uses the old
    'Security Analyst View' / 'Administrator View' / 'Clinician View' labels.
    """
    for key, info in role_yaml["roles"].items():
        display = m6.ROLE_DISPLAY_NAMES[key]
        pv = info.get("primary_view", "")
        assert display in pv, (
            f"primary_view for {key!r} = {pv!r} doesn't contain "
            f"the display name {display!r}"
        )
