"""Field-surfacing tests (Sprint 2.5).

For every field in ``Response`` / ``Explanation`` that the upgrade
phases (1.2/1.3/2/3.1/3.2/4.1) committed to writing, verify it
actually appears in the regenerated ``alert_responses.json`` artifact
at or above the documented floor.

Catches Category 4 "field exists but never written" bugs — Phase 4.1
added ``Response.auto_execute`` to the schema but the writer code had
to be updated separately; if the schema add lands without the writer
the test would have failed loud.

Floors are documented inline so a future phase can lift them with one
PR. The test runs against both splits (test + demo) so demo doesn't
silently fall behind.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


REPORTS = Path(__file__).resolve().parent.parent / "results" / "reports"


# Field → minimum coverage as a fraction of all records. Per-split
# overrides are taken when present; otherwise the default is used.
#
# Some fields are written for *every* record (default-True booleans,
# always-emitted dicts) — those have floor 1.0. Others are only
# written when the analyst entry carries the precursor (e.g.
# counterfactual + stability are only attached when SHAP exists, so
# the floor is set against XGBoost-flagged density which is ~50% on
# our corpus). The floors err on the *low* side so the test is
# robust to corpus shuffles between splits.

RESPONSE_FIELDS = {
    "actions":                    {"default": 1.0},
    "action_descriptions":        {"default": 1.0},
    "actions_metadata":           {"default": 1.0},  # Phase 1.3
    "escalation_chain":           {"default": 1.0},
    "escalation_rationale":       {"default": 1.0},
    "max_response_min":           {"default": 1.0},
    "priority":                   {"default": 1.0},
    "rationale":                  {"default": 1.0},
    "device_tier":                {"default": 1.0},
    "device_constraint_applied":  {"default": 1.0},
    # Phase 2.4 — only set when feasible counterfactual exists
    "try_first_action":           {"default": 0.40},
    # Phase 3.1 — every non-NORMAL alert has one (records exclude NORMAL)
    "playbook":                   {"default": 1.0},
    # Phase 3.2 — always written, mismatch=False for aligned alerts
    "routing_warning":            {"default": 1.0},
    # Phase 4.1 / Sprint 2.5 — always-emit policy: pipeline writes the
    # canonical tier-policy default then demotes to False when stability
    # is UNSTABLE.
    "auto_execute_recommended":   {"default": 1.0},  # Path B · commit 6 rename
}


# clinician_summary floor is 0.50 because the summary is only attached
# to XGBoost-flagged alerts (~60-70% of alert_responses). The rest are
# DAE-flagged or risk-elevated context records where there's no
# attribution to summarise. Phase 5 (extend SHAP to RF/DAE) would
# raise this floor.
EXPLANATION_FIELDS = {
    "clinician_summary":          {"default": 0.50},
    "analyst_available":          {"default": 1.0},
    "mve":                        {"default": 1.0},
    # Phase 2 — only when XGBoost-flagged (so floor is per-split density)
    "counterfactual":             {"default": 0.40, "demo": 0.50},
    # Phase 4.1 — same XGBoost-flagged density
    "stability":                  {"default": 0.40, "demo": 0.50},
}


def _coverage(records: list[dict], dotted_path: str) -> float:
    """% of records that have a non-empty value at ``dotted_path``."""
    n = 0
    for r in records:
        obj = r
        for part in dotted_path.split("."):
            if not isinstance(obj, dict):
                obj = None
                break
            obj = obj.get(part)
            if obj is None:
                break
        # Pydantic-default fields show up as their default value (e.g.
        # ``auto_execute: True``); treat ``False`` and 0 as "present".
        if obj is None:
            continue
        if obj == "" or obj == [] or obj == {}:
            continue
        n += 1
    return n / len(records) if records else 0.0


def _load_records(split: str) -> list[dict]:
    suffix = "_demo" if split == "demo" else ""
    path = REPORTS / f"alert_responses{suffix}.json"
    if not path.exists():
        pytest.skip(f"{path.name} missing — run phase1_regen_module5 first")
    envelope = json.loads(path.read_text())
    return envelope.get("records", envelope) if isinstance(envelope, dict) else envelope


def _floor_for(field_name: str, split: str, table: dict) -> float:
    spec = table[field_name]
    return float(spec.get(split, spec["default"]))


# ── Response field surfacing ──────────────────────────────────────


@pytest.mark.parametrize("split", ["test", "demo"])
@pytest.mark.parametrize("field", list(RESPONSE_FIELDS.keys()))
def test_response_field_surfaced(split, field):
    """Each Response field must appear in records at ≥ its floor."""
    records = _load_records(split)
    coverage = _coverage(records, f"response.{field}")
    floor = _floor_for(field, split, RESPONSE_FIELDS)
    assert coverage + 1e-9 >= floor, (
        f"response.{field!r} coverage on {split!r}: "
        f"{coverage:.2%} < floor {floor:.0%}. Either the writer "
        f"regressed, the floor is wrong for this split, or the field "
        f"is now dead code."
    )


# ── Explanation field surfacing ──────────────────────────────────


@pytest.mark.parametrize("split", ["test", "demo"])
@pytest.mark.parametrize("field", list(EXPLANATION_FIELDS.keys()))
def test_explanation_field_surfaced(split, field):
    records = _load_records(split)
    coverage = _coverage(records, f"explanation.{field}")
    floor = _floor_for(field, split, EXPLANATION_FIELDS)
    assert coverage + 1e-9 >= floor, (
        f"explanation.{field!r} coverage on {split!r}: "
        f"{coverage:.2%} < floor {floor:.0%}."
    )


# ── Schema completeness — every field in schema has a floor entry ──


def test_response_floor_table_covers_every_schema_field():
    """If a new field is added to Response, the floor table must be
    updated — otherwise the field silently goes untested."""
    from common.alert_response_schema import Response
    schema_fields = set(Response.model_fields.keys())
    documented = set(RESPONSE_FIELDS.keys())
    missing = schema_fields - documented
    assert not missing, (
        f"Response schema has {missing} fields with no floor entry in "
        f"RESPONSE_FIELDS — add a floor (1.0 if always-emitted, lower "
        f"if conditional) so the field surfacing test exercises them."
    )


def test_explanation_floor_table_covers_every_schema_field():
    from common.alert_response_schema import Explanation
    schema_fields = set(Explanation.model_fields.keys())
    documented = set(EXPLANATION_FIELDS.keys())
    missing = schema_fields - documented
    assert not missing, (
        f"Explanation schema has {missing} fields with no floor entry "
        f"in EXPLANATION_FIELDS — see RESPONSE equivalent for fix."
    )
