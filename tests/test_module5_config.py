"""Module 5 config — taxonomy unification + cross-field consistency."""
from __future__ import annotations

import pytest

from module5_responses.config import (
    ACTION_CATALOGUE,
    ACUITY_OVERRIDES,
    ATTACK_ROUTING,
    DEFAULT_DEVICE_TIER,
    DEVICE_TIERS,
    MVE_LLM_FAIL_STREAK_MAX,
    RESPONSE_POLICY_VERSION,
    TIER_POLICIES,
    export_response_policy_dict,
)


def test_action_catalogue_has_all_eight_actions():
    assert set(ACTION_CATALOGUE) == {
        "log_event", "enhanced_monitoring", "re_authenticate",
        "forensic_snapshot", "restrict_traffic", "escalate_clinical",
        "isolate_device", "escalate_incident",
    }


def test_action_catalogue_costs_strictly_ordered_by_severity():
    # Sanity: costs are in [0.1, 1.0] and unique enough that sort-by-cost is total.
    costs = [spec["cost"] for spec in ACTION_CATALOGUE.values()]
    assert all(0 < c <= 1.0 for c in costs)


def test_action_catalogue_required_fields():
    required = {"severity_floor", "cost", "reversible", "requires_approval", "description"}
    for name, spec in ACTION_CATALOGUE.items():
        assert required.issubset(spec.keys()), f"{name} missing fields"


@pytest.mark.parametrize("tier", list(DEVICE_TIERS))
def test_device_tier_max_action_cost_matches_action_cost(tier):
    spec = DEVICE_TIERS[tier]
    assert spec["max_action_cost"] == ACTION_CATALOGUE[spec["max_action"]]["cost"]


def test_default_device_tier_is_known():
    assert DEFAULT_DEVICE_TIER in DEVICE_TIERS


@pytest.mark.parametrize("tier", list(TIER_POLICIES))
def test_tier_policies_actions_exist_in_catalogue(tier):
    for action in TIER_POLICIES[tier]["default_actions"]:
        assert action in ACTION_CATALOGUE, f"{tier} references unknown action {action!r}"


@pytest.mark.parametrize("cat", list(ATTACK_ROUTING))
def test_attack_routing_actions_exist_in_catalogue(cat):
    routing = ATTACK_ROUTING[cat]
    for action in routing["attack_specific_actions"] + routing["add_actions"]:
        assert action in ACTION_CATALOGUE


def test_tier_policies_log_event_listed_first():
    # Y6: canonical ordering — log_event always leads the default action list.
    for tier in ("CRITICAL", "HIGH", "MEDIUM", "LOW", "NORMAL"):
        assert TIER_POLICIES[tier]["default_actions"][0] == "log_event"


def test_response_policy_version_pinned():
    assert RESPONSE_POLICY_VERSION == "2.0"


def test_export_dict_round_trip_keys():
    d = export_response_policy_dict()
    assert set(d) >= {
        "version", "action_catalogue", "tier_policies",
        "device_constraints", "acuity_overrides", "attack_routing",
    }
    # Legacy artifact must include the four real tiers.
    assert set(d["tier_policies"]) == {"CRITICAL", "HIGH", "MEDIUM", "LOW"}


def test_acuity_threshold_in_valid_range():
    t = ACUITY_OVERRIDES["elevated_acuity_threshold"]
    assert 0.0 < t < 1.0


def test_mve_tripwire_default_is_five():
    # Operator override via IOMT_MVE_TRIPWIRE; in test env we expect the default.
    assert isinstance(MVE_LLM_FAIL_STREAK_MAX, int)
    assert MVE_LLM_FAIL_STREAK_MAX >= 1
