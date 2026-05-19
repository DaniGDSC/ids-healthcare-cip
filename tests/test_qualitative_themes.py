"""Smoke tests for the manually-coded qualitative themes manifest."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
YAML_PATH = REPO / "survey" / "qualitative_themes.yaml"

ROLES = ("biomed_engineer", "IT_generalist", "nurse_manager")


@pytest.fixture(scope="module")
def themes() -> dict:
    if not YAML_PATH.exists():
        pytest.skip("Run analysis/extract_qualitative_rationales.py, then "
                    "code themes manually in survey/qualitative_themes.yaml")
    return yaml.safe_load(YAML_PATH.read_text())


def test_all_three_roles_addressed(themes: dict) -> None:
    for role in ROLES:
        assert role in themes.get("themes_per_role", {}), \
            f"Missing role: {role}"


def test_methodology_discloses_llm_persona(themes: dict) -> None:
    """Manifest must call out that rationales are LLM-generated."""
    method = (themes.get("methodology") or "").lower()
    assert "llm" in method, (
        "qualitative_themes.yaml.methodology must disclose LLM-persona data "
        "source (defense-critical)."
    )


def test_coded_metadata_present_once_coded(themes: dict) -> None:
    """Skip until coded; once coded, require last_coded + coded_by."""
    if not themes.get("last_coded"):
        pytest.skip("Themes not yet coded")
    assert themes.get("coded_by"), \
        "coded_by missing — fill in after manual coding"


def test_each_role_has_at_least_one_theme_once_coded(themes: dict) -> None:
    if not themes.get("last_coded"):
        pytest.skip("Themes not yet coded")
    for role, entry in themes["themes_per_role"].items():
        total = (len(entry.get("positive_themes") or [])
                 + len(entry.get("confusion_patterns") or []))
        assert total > 0, f"Role {role} has no coded themes"
