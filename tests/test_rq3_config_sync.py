"""RQ3 Q8 — YAML config drift guard.

Verifies that `config/tier_routing.yaml` and
`config/role_action_authorization.yaml` stay in sync with the canonical
sources of truth in the codebase:

  • Tier names match `module3_risk_scoring.RISK_THRESHOLDS` (+ implicit LOW)
  • Action vocabulary covers `module6_evaluation.module6_app.ACTIONS`
  • Tier policies reference real tier names (no typos)
  • Framework version + last_synced metadata present
  • Every tier referenced in the role-authorization invariants block
    actually exists in tier_routing

Catches the manual-sync drift risk called out in RQ3 §9 gap analysis.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_DIR = PROJECT_ROOT / "config"


@pytest.fixture(scope="module")
def tier_routing_cfg():
    path = CONFIG_DIR / "tier_routing.yaml"
    if not path.exists():
        pytest.skip(f"{path} missing")
    with open(path) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def role_auth_cfg():
    path = CONFIG_DIR / "role_action_authorization.yaml"
    if not path.exists():
        pytest.skip(f"{path} missing")
    with open(path) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def canonical_tiers():
    """Module 3 names the threshold boundaries; LOW is the implicit
    fall-through tier below the lowest threshold.
    """
    from module3_risk_scoring.module3_risk_scores import RISK_THRESHOLDS
    return {t[1] for t in RISK_THRESHOLDS} | {"LOW"}


@pytest.fixture(scope="module")
def canonical_actions():
    """Module 6's user-facing action list (the union of acknowledge /
    monitor / investigate / isolate / escalate / dismiss the operator
    can pick from in the Online Simulation page).
    """
    from module6_evaluation import module6_app as m6
    return set(m6.ACTIONS)


# ── tier_routing.yaml drift checks ────────────────────────────────────


def test_tier_routing_has_all_canonical_tiers(tier_routing_cfg, canonical_tiers):
    yaml_tiers = set(tier_routing_cfg["tier_routes"].keys())
    missing = canonical_tiers - yaml_tiers
    extra = yaml_tiers - canonical_tiers
    assert not missing, (
        f"tier_routing.yaml missing tiers: {missing}. "
        f"Module 3 RISK_THRESHOLDS expects {sorted(canonical_tiers)}."
    )
    assert not extra, (
        f"tier_routing.yaml has unknown tiers: {extra}. "
        f"If a new tier was added to Module 3, update the YAML; otherwise drop."
    )


def test_tier_routing_invariant_overrides_reference_known_tiers(tier_routing_cfg, canonical_tiers):
    """Each invariant override's `when.computed_tier` must point at a
    real tier (no typos like 'CRTICAL'). `when: always` is the global-
    scope sentinel and skips the tier check.
    """
    for inv in tier_routing_cfg.get("invariant_overrides", []) or []:
        when = inv.get("when", {})
        if not isinstance(when, dict) or when == "always":
            # Global-scope sentinel like `when: always` (parsed as either
            # a non-dict scalar or — defensively — a string-keyed dict) —
            # no tier to check.
            continue
        ct = when.get("computed_tier")
        if ct is None:
            continue
        if isinstance(ct, str):
            assert ct in canonical_tiers, (
                f"invariant {inv.get('id')}: computed_tier={ct!r} "
                f"not in canonical tiers {sorted(canonical_tiers)}"
            )
        elif isinstance(ct, list):
            for t in ct:
                assert t in canonical_tiers, (
                    f"invariant {inv.get('id')}: computed_tier list has "
                    f"{t!r} not in canonical tiers"
                )


def test_tier_routing_version_metadata(tier_routing_cfg):
    """Drift sentinel: `last_synced` + `version` must be present."""
    assert tier_routing_cfg.get("version"), "tier_routing.yaml: missing version"
    assert tier_routing_cfg.get("last_synced"), "tier_routing.yaml: missing last_synced"


# ── role_action_authorization.yaml drift checks ──────────────────────


def test_role_auth_covers_canonical_actions(role_auth_cfg, canonical_actions):
    """Every action the operator can pick (`module6_app.ACTIONS`) must
    be authorized by SOMEONE in the YAML. Missing actions mean the UI
    surfaces an option with no authorization policy.
    """
    yaml_actions = set(role_auth_cfg["actions"].keys())
    missing = canonical_actions - yaml_actions
    assert not missing, (
        f"role_action_authorization.yaml missing canonical actions: {missing}. "
        f"module6_app.ACTIONS = {sorted(canonical_actions)}; YAML actions = "
        f"{sorted(yaml_actions)}."
    )


def test_role_auth_initiators_are_known_roles(role_auth_cfg):
    """Every action's `initiators` list must reference roles declared in
    the top-level `roles:` block.
    """
    declared_roles = set(role_auth_cfg["roles"].keys())
    for action_name, action_info in role_auth_cfg["actions"].items():
        initiators = action_info.get("initiators", []) or []
        bad = [r for r in initiators if r not in declared_roles]
        assert not bad, (
            f"action {action_name!r}: initiators reference unknown roles "
            f"{bad}. declared: {sorted(declared_roles)}"
        )


def test_role_auth_approval_chain_known_roles(role_auth_cfg):
    declared_roles = set(role_auth_cfg["roles"].keys())
    for action_name, action_info in role_auth_cfg["actions"].items():
        chain = action_info.get("approval_chain", []) or []
        bad = [r for r in chain if r not in declared_roles]
        assert not bad, (
            f"action {action_name!r}: approval_chain has unknown roles {bad}"
        )


def test_role_auth_visibility_keys_are_known_roles(role_auth_cfg):
    """Each action's `visibility` dict must key on declared roles only."""
    declared_roles = set(role_auth_cfg["roles"].keys())
    for action_name, action_info in role_auth_cfg["actions"].items():
        vis = action_info.get("visibility", {}) or {}
        bad = [r for r in vis.keys() if r not in declared_roles]
        assert not bad, (
            f"action {action_name!r}: visibility keys {bad} not in declared roles"
        )


def test_role_auth_version_metadata(role_auth_cfg):
    assert role_auth_cfg.get("version"), "role_action_authorization.yaml: missing version"
    assert role_auth_cfg.get("last_synced"), "role_action_authorization.yaml: missing last_synced"


# ── cross-yaml consistency ────────────────────────────────────────────


def test_isolate_blocked_on_life_sustaining_consistency(role_auth_cfg, tier_routing_cfg):
    """The `isolate` action's blocked_on_device_class must match
    tier_routing's device_overlays.life_sustaining.isolation_blocked = true.
    """
    isolate = role_auth_cfg["actions"].get("isolate", {})
    blocked = set(isolate.get("blocked_on_device_class", []) or [])
    life_sus = tier_routing_cfg.get("device_overlays", {}).get("life_sustaining", {})
    overlay_blocked = bool(life_sus.get("isolation_blocked"))
    if "life_sustaining" in blocked:
        assert overlay_blocked, (
            "role_auth.isolate blocks life_sustaining but "
            "tier_routing.device_overlays.life_sustaining.isolation_blocked is false"
        )
    if overlay_blocked:
        assert "life_sustaining" in blocked, (
            "tier_routing blocks isolation on life_sustaining but "
            "role_auth.isolate.blocked_on_device_class doesn't list it"
        )
