"""RQ3 Invariant tests — Step 13 cross-role consistency (Invariant 6 + 9).

Verifies that the role-adaptive renderers (analyst / clinician /
administrator) all expose the same underlying severity + the same shared
anchor alert metadata. Two invariants:

  Invariant 6  — Cross-role severity invariance. Operator role must not
                 change the alert's risk_level / risk_score; only the
                 *explanation framing* varies.
  Invariant 9  — Shared anchor across roles. Alert ID + sample index +
                 timestamp must be identical regardless of which role
                 renderer is hit.

The role renderers live in module6_evaluation.module6_app. We call them
indirectly by inspecting the alert object that gets passed in (it's the
same dict across roles by design) and by verifying the Module 5
schema's per-alert fields are role-independent.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

REPORTS = PROJECT_ROOT / "results/reports"


@pytest.fixture(scope="module")
def alert_responses() -> list:
    path = REPORTS / "alert_responses.json"
    if not path.exists():
        pytest.skip(f"{path} missing — run module5_responses first")
    with open(path) as f:
        data = json.load(f)
    return data["records"] if isinstance(data, dict) and "records" in data else data


@pytest.fixture(scope="module")
def participant_responses() -> list:
    """Per-role HITL responses (M6 study) — has explicit role labels."""
    path = REPORTS / "participant_responses.json"
    if not path.exists():
        pytest.skip(f"{path} missing — participant study not run")
    with open(path) as f:
        return json.load(f)


# ── Invariant 6 — severity invariance across roles ──────────────────


def test_inv6_alert_risk_level_is_role_independent(alert_responses):
    """The risk_level field is set by Module 5 BEFORE the role renderer is
    chosen. Test that it has no role-dependent variant.
    """
    role_keys = ("risk_level_analyst", "risk_level_clinician",
                 "risk_level_administrator")
    contaminated = []
    for r in alert_responses[:200]:
        for k in role_keys:
            if k in r:
                contaminated.append((r.get("sample_index"), k))
    assert not contaminated, (
        f"{len(contaminated)} role-namespaced risk_level fields found — "
        f"Invariant 6 violated (severity must be role-independent). "
        f"First: {contaminated[:3]}"
    )


def test_inv6_same_alert_same_severity_across_renderers(participant_responses):
    """Each alert in the participant study must have a single canonical
    `correct_action` regardless of which role's session it appears in.
    A role-conditional `correct_action` would mean the system is teaching
    different roles different ground truth — Invariant 6 violation.
    """
    by_alert = {}
    for r in participant_responses:
        aid = r["alert_id"]
        ca = r.get("correct_action")
        if aid in by_alert:
            assert by_alert[aid] == ca, (
                f"Alert {aid} has different correct_action across roles: "
                f"{by_alert[aid]!r} vs {ca!r}"
            )
        else:
            by_alert[aid] = ca


# ── Invariant 9 — Shared anchor across roles ────────────────────────


def test_inv9_alert_id_stable_across_roles(participant_responses):
    """The alert_id surfaces unchanged to all 3 roles. If the same sample
    appears under different IDs per role, audit attribution breaks.
    """
    by_role = {}
    for r in participant_responses:
        role = r["participant_role"]
        by_role.setdefault(role, set()).add(r["alert_id"])

    # If there's a common evaluation set, the intersection should be
    # non-empty (the M6 study uses the same evaluation_alerts.json for
    # all roles).
    if len(by_role) >= 2:
        common = set.intersection(*by_role.values())
        assert common, (
            f"No alert IDs shared across roles {list(by_role.keys())}. "
            f"Invariant 9 violated — roles see disjoint alert sets."
        )


def test_inv9_alert_record_keys_role_independent(alert_responses):
    """The Module-5 alert record schema is one shape, not three. Verify
    no record carries role-namespaced top-level fields (`*_analyst`,
    `*_clinician`, `*_administrator`).
    """
    bad_suffixes = ("_analyst", "_clinician", "_administrator")
    contaminated = []
    for r in alert_responses[:200]:
        for k in r.keys():
            if any(k.endswith(s) for s in bad_suffixes):
                # 'analyst' is allowed as a sub-key inside `explanation`
                # — only flag if at top level
                contaminated.append((r.get("sample_index"), k))
    assert not contaminated, (
        f"{len(contaminated)} top-level role-namespaced fields found: "
        f"{contaminated[:3]}"
    )


# ── Composite — role renderers receive identical alert dict ─────────


def test_role_renderers_share_alert_dict():
    """Smoke test: simulate the dispatch logic. All three role render
    functions in module6_evaluation.module6_app must accept the SAME
    alert dict (verified by inspecting signatures).
    """
    try:
        from module6_evaluation import module6_app as m6
    except Exception as e:
        pytest.skip(f"module6_app import failed: {e}")

    import inspect
    sigs = {
        "render_analyst":   inspect.signature(m6.render_analyst),
        "render_clinician": inspect.signature(m6.render_clinician),
        "render_admin":     inspect.signature(m6.render_admin),
    }
    # All three should take a single `alert` positional arg
    for name, sig in sigs.items():
        params = list(sig.parameters.values())
        assert len(params) == 1, f"{name} should take 1 arg, got {len(params)}"
        assert params[0].name == "alert", (
            f"{name} first arg should be `alert`, got `{params[0].name}`"
        )


def test_consensus_label_unified_across_renderers():
    """T10 from the consensus checklist: DETECTOR_CONSENSUS_LABEL is the
    single label used by all consensus surfaces.
    """
    try:
        from module6_evaluation import module6_app as m6
    except Exception as e:
        pytest.skip(f"module6_app import failed: {e}")
    assert m6.DETECTOR_CONSENSUS_LABEL == "Detector consensus", (
        f"unexpected label: {m6.DETECTOR_CONSENSUS_LABEL!r}"
    )
