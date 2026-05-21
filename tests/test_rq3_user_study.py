"""Schema + methodology-disclosure tests for the RQ3 Track 5 outputs."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ESC = REPO / "analysis" / "outputs" / "rq3_escalation.json"
WRAP = REPO / "analysis" / "outputs" / "rq3_user_study.json"

ROLES = ("biomed_engineer", "IT_generalist", "nurse_manager")


@pytest.fixture(scope="module")
def escalation() -> dict:
    if not ESC.exists():
        pytest.skip("Run analysis/compute_rq3_escalation.py first")
    return json.loads(ESC.read_text())


@pytest.fixture(scope="module")
def wrapper() -> dict:
    if not WRAP.exists():
        pytest.skip("Run analysis/compute_rq3_per_role.py first")
    return json.loads(WRAP.read_text())


# ── Escalation JSON ──────────────────────────────────────────────────


def test_escalation_schema_complete(escalation: dict) -> None:
    for key in ("_meta", "methodology_notes", "limitations",
                "overall", "per_role", "cell_diagnostics"):
        assert key in escalation, f"Missing top-level key: {key}"


def test_escalation_methodology_discloses_llm_persona(escalation: dict) -> None:
    text = " ".join(escalation["methodology_notes"]).lower()
    assert "llm-persona" in text or "llm persona" in text, (
        "methodology_notes must disclose LLM-persona data source."
    )


def test_escalation_methodology_discloses_no_correction(
    escalation: dict,
) -> None:
    text = " ".join(escalation["methodology_notes"]).lower()
    assert ("no multiple-comparisons correction" in text
            or "no multiple comparisons correction" in text), (
        "methodology_notes must disclose absence of multiple-comparisons "
        "correction across the 3 role tests."
    )


def test_escalation_all_three_roles_present(escalation: dict) -> None:
    for role in ROLES:
        assert role in escalation["per_role"], f"Missing role: {role}"


def test_escalation_p_values_in_valid_range(escalation: dict) -> None:
    def check(cell: dict) -> None:
        p = cell.get("p_value")
        if p is not None:
            assert 0 <= p <= 1, f"Invalid p-value: {p}"

    check(escalation["overall"])
    for role in ROLES:
        check(escalation["per_role"].get(role, {}))


def test_escalation_cramers_v_in_valid_range(escalation: dict) -> None:
    """Cramer's V for 2x2 = |phi|, in [0, 1]."""
    def check(cell: dict) -> None:
        v = cell.get("cramers_v")
        if v is not None:
            assert 0 <= v <= 1, f"Invalid Cramer's V: {v}"

    check(escalation["overall"])
    for role in ROLES:
        check(escalation["per_role"].get(role, {}))


def test_escalation_taxonomy_provenance_disclosed(escalation: dict) -> None:
    """_meta must record the locked-on date + pre-data-collection status."""
    meta = escalation["_meta"]
    assert meta.get("taxonomy_locked_on"), \
        "_meta.taxonomy_locked_on missing"
    assert meta.get("escalation_actions"), \
        "_meta.escalation_actions missing"


def test_escalation_contingency_sums_match_n_cells(escalation: dict) -> None:
    """contingency cells must sum to n_A + n_B per cell-block."""
    for cell in [escalation["overall"], *escalation["per_role"].values()]:
        c = cell["contingency_2x2"]
        n_a = cell["n_A"]
        n_b = cell["n_B"]
        assert c["A_escalated"] + c["A_not"] == n_a, (
            f"Contingency A sum != n_A in scope={cell.get('_scope')}"
        )
        assert c["B_escalated"] + c["B_not"] == n_b, (
            f"Contingency B sum != n_B in scope={cell.get('_scope')}"
        )


# ── RQ3-lens wrapper ─────────────────────────────────────────────────


def test_wrapper_schema_complete(wrapper: dict) -> None:
    for key in ("_meta", "methodology_notes", "limitations",
                "per_role_accuracy_confidence",
                "overall_accuracy_confidence",
                "per_role_escalation", "overall_escalation"):
        assert key in wrapper, f"Missing top-level key: {key}"


def test_wrapper_discloses_rq3_lens(wrapper: dict) -> None:
    meta_text = json.dumps(wrapper["_meta"]).lower()
    assert "rq3" in meta_text and "distributed" in meta_text, (
        "_meta must describe RQ3 distributed-responsibility framing."
    )


def test_wrapper_methodology_discloses_llm_persona(wrapper: dict) -> None:
    text = " ".join(wrapper["methodology_notes"]).lower()
    assert "llm-persona" in text or "llm persona" in text, (
        "Wrapper methodology_notes must carry the LLM-persona disclosure."
    )
