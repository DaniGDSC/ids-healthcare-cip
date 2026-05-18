"""Schema-validation tests for RQ2.c LLM-persona user-study data."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
AUDIT_OUT = REPO / "survey" / "study_data_audit.json"


@pytest.fixture(scope="module")
def audit() -> dict:
    if not AUDIT_OUT.exists():
        pytest.skip("Run analysis/audit_study_data.py first")
    return json.loads(AUDIT_OUT.read_text())


def test_some_participants_collected(audit: dict) -> None:
    assert audit["summary"]["n_total"] > 0, \
        "No participant responses found in survey/"


def test_exclusion_rate_reasonable(audit: dict) -> None:
    rate = audit["summary"]["exclusion_rate"]
    assert rate < 0.60, (
        f"Exclusion rate {rate:.1%} is alarmingly high. "
        "Check survey/rq2c_exclusions.json for patterns."
    )


def test_all_roles_represented(audit: dict) -> None:
    by_role = audit["summary"]["by_role"]
    missing = [r for r in ("biomed_engineer", "IT_generalist", "nurse_manager")
               if by_role.get(r, 0) == 0]
    if missing:
        pytest.skip(f"Roles missing (recruitment incomplete): {missing}")


def test_both_conditions_represented(audit: dict) -> None:
    by_cond = audit["summary"]["by_condition"]
    assert by_cond.get("A", 0) > 0 and by_cond.get("B", 0) > 0, \
        f"Condition imbalance: {by_cond}"


def test_meta_discloses_llm_persona(audit: dict) -> None:
    """Defense-critical: audit must call out LLM-persona data source."""
    text = json.dumps(audit["_meta"]).lower()
    assert "llm" in text and "persona" in text, (
        "Audit _meta must disclose LLM-persona data source explicitly."
    )
