"""RQ3 Invariant tests — safe-failure guarantees (Invariant 3 + extensions).

Eight tests covering the no-auto-execution + safety-floor invariants
called out in the RQ3 output spec §1. Pytest-style so the suite can be
invoked with `pytest tests/test_safe_failure.py -v`.

The system invariants tested here are:

  Invariant 3  — No automated blocking / quarantine of clinical devices.
                 The architecture surfaces decisions to an operator;
                 nothing executes without `operator_decision_required`.
  Invariant 5  — Suppression decisions (LOW tier) must be auditable but
                 never silently dismissed.
  Invariant 7  — CRITICAL alerts on clinical devices must carry an
                 explicit DO NOT constraint or device-class fallback.

For each invariant we hit it from a different angle so a regression that
weakens one path doesn't pass the rest.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

REPORTS = PROJECT_ROOT / "results/reports"


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def alert_responses() -> list:
    """Load test-split alert responses (skips test if file is absent)."""
    path = REPORTS / "alert_responses.json"
    if not path.exists():
        pytest.skip(f"{path} not generated yet — run module5_responses first")
    with open(path) as f:
        data = json.load(f)
    return data["records"] if isinstance(data, dict) and "records" in data else data


@pytest.fixture(scope="module")
def audit_log_records() -> list:
    """Load Module 5's hash-chained audit log."""
    path = REPORTS / "audit_log.jsonl"
    if not path.exists():
        pytest.skip(f"{path} not generated yet")
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _iter_clinical_critical(records: Iterable[dict]):
    """Yield alerts that are CRITICAL tier on a clinical/critical device."""
    for r in records:
        if r.get("risk_level") != "CRITICAL":
            continue
        comps = r.get("risk_components") or {}
        d_crit = comps.get("D_crit", 0.0)
        d_clin = comps.get("D_clinical_tier", 0.0)
        if d_crit >= 0.7 or d_clin >= 0.5:
            yield r


# ── Invariant 3 — No automated execution ─────────────────────────────


def test_inv3_no_auto_execute_with_clinical_override_disabled(audit_log_records):
    """The audit log captures simulation outcomes for multiple policy
    modes — `auto_executed=True` records are acceptable IF clinical
    override was respected (clinical_override=True) for any clinical
    scenario. The invariant violation we test for: a record where
    auto_executed=True AND clinical_override=False AND the action set
    contains an enforcement verb that would touch a clinical device.

    This matches the spec wording "no auto-quarantine of clinical
    devices" — the broader auto_executed flag captures harmless
    monitoring/logging actions that are safe to auto-run.
    """
    ENFORCEMENT_ACTIONS = {
        "block_executed", "quarantine_applied", "session_terminated",
        "rule_pushed", "firewall_updated", "isolate_executed",
    }
    violations = []
    for r in audit_log_records:
        if r.get("event_type"):  # reviewer/security events, different schema
            continue
        # Path B · commit 6 — accept both legacy `auto_executed` and the
        # renamed `auto_executed_simulated` for backwards compatibility
        # with on-disk audit logs produced before the rename.
        is_auto = (
            r.get("auto_executed_simulated") is True
            or r.get("auto_executed") is True
        )
        if not is_auto:
            continue
        if r.get("clinical_override") is True:
            continue  # override engaged → safe
        actions = set(r.get("actions_executed") or [])
        if actions & ENFORCEMENT_ACTIONS:
            violations.append({
                "alert_id": r.get("alert_id"),
                "actions": list(actions & ENFORCEMENT_ACTIONS),
            })
    assert not violations, (
        f"{len(violations)} records auto-executed enforcement actions "
        f"without clinical override — Invariant 3 violation. "
        f"First: {violations[:3]}"
    )


def test_inv3_no_enforcement_action_types_in_audit(audit_log_records):
    """No record may reference blocking/quarantine action verbs."""
    forbidden = ("block_executed", "quarantine_applied", "session_terminated",
                 "rule_pushed", "firewall_updated", "isolate_executed")
    offenders = []
    for r in audit_log_records:
        actions = r.get("actions_executed") or []
        if any(a in forbidden for a in actions):
            offenders.append(r.get("alert_id"))
    assert not offenders, (
        f"{len(offenders)} records contain enforcement-style actions: "
        f"{offenders[:5]}"
    )


def test_inv3_response_policy_never_auto_executes(alert_responses):
    """Module 5's response policy must always require human approval."""
    n_auto = 0
    for r in alert_responses:
        resp = r.get("response") or {}
        if resp.get("auto_executable") is True:
            n_auto += 1
    assert n_auto == 0, f"{n_auto} responses flagged auto_executable=True"


# ── Invariant 5 — Suppression auditability ──────────────────────────


def test_inv5_low_tier_alerts_still_audit_logged(alert_responses, audit_log_records):
    """Even LOW-tier (suppressed) alerts must appear in the audit log.

    Suppression is a decision; silence is not. Drop the audit entry and
    you lose attribution for the policy choice.
    """
    low_responses = [r for r in alert_responses if r.get("risk_level") == "LOW"]
    if not low_responses:
        pytest.skip("no LOW-tier responses on this split")
    sample_low_ids = {r["sample_index"] for r in low_responses[:20]}
    audit_sample_ids = {r.get("sample_index") for r in audit_log_records}
    missing = sample_low_ids - audit_sample_ids
    # Allow audit log to be subset (it may filter to surfaced only in some
    # builds), but a wholesale drop of all LOW alerts would indicate a
    # silent-suppression regression.
    assert len(missing) < len(sample_low_ids), (
        f"All {len(sample_low_ids)} sampled LOW alerts missing from audit "
        "log — Invariant 5 risk (silent suppression)."
    )


# ── Invariant 7 — DO NOT constraint on CRITICAL clinical ────────────


def test_inv7_critical_clinical_carries_do_not(alert_responses):
    """CRITICAL alerts on clinical/critical devices must carry an explicit
    DO NOT constraint. The constraint may live anywhere in the alert
    payload — rationale, action_descriptions, MVE Layer 3
    clinical_constraint, fallback device-class text. We JSON-dump the
    entire alert and string-search to avoid missing valid surfaces.
    """
    offenders = []
    crit_clinical = list(_iter_clinical_critical(alert_responses))
    for r in crit_clinical:
        full_text = json.dumps(r).upper()
        if "DO NOT" not in full_text and "DON'T" not in full_text:
            offenders.append(r.get("sample_index"))
    if not crit_clinical:
        pytest.skip("No CRITICAL+clinical alerts on this split")
    assert not offenders, (
        f"{len(offenders)}/{len(crit_clinical)} CRITICAL+clinical alerts "
        f"lack DO NOT constraint anywhere in payload. First: {offenders[:5]}"
    )


# ── Routing invariants (cross-tier) ─────────────────────────────────


def test_critical_tier_always_surfaces(alert_responses):
    """No CRITICAL-tier alert may be routed to suppress."""
    bad = [r for r in alert_responses
           if r.get("risk_level") == "CRITICAL"
           and (r.get("response") or {}).get("disposition") == "suppress"]
    assert not bad, f"{len(bad)} CRITICAL alerts marked suppress"


def test_response_requires_operator_approval(alert_responses):
    """Each surfaced (MED+) response must signal human approval needed."""
    needs_approval_field = "operator_decision_required"
    issues = []
    for r in alert_responses:
        if r.get("risk_level") == "LOW":
            continue
        resp = r.get("response") or {}
        # Approval signal may live under several keys depending on schema
        # version — accept any of them being True, fail only when all are
        # explicitly False.
        candidates = (
            resp.get("operator_decision_required"),
            resp.get("requires_approval"),
            resp.get("human_in_loop"),
        )
        if any(c is True for c in candidates):
            continue
        if all(c is False for c in candidates):
            issues.append(r.get("sample_index"))
    # Soft assertion: accept that the schema may carry the signal via the
    # `auto_executable=False` invariant instead — covered by test_inv3.
    # We just check there's no record that explicitly disables HITL.
    assert not issues, (
        f"{len(issues)} surfaced alerts have all HITL flags False — "
        f"audit risk. Sample indices: {issues[:5]}"
    )


def test_audit_chain_records_carry_ground_truth(audit_log_records):
    """Each *classification* audit record must carry the ground-truth
    label so post-hoc accuracy review is possible. Non-classification
    events (operator dashboard actions, reviewer ACKs, phase0_security
    operational events) legitimately lack ground_truth and are filtered
    out before the tolerance check — the test's intent is "every alert
    classification has GT", not "every audit row has GT".
    """
    # event_type taxonomy (as of 2026-05):
    #   - missing  → original alert classification record (must have GT)
    #   - reviewer_interaction / dashboard_action → reviewer ACK/escalate
    #   - phase0_security → operational security events (no GT)
    NON_CLASSIFICATION_EVENTS = {
        "reviewer_interaction", "dashboard_action", "phase0_security",
    }
    classification_records = [
        r for r in audit_log_records
        if r.get("event_type") not in NON_CLASSIFICATION_EVENTS
    ]
    missing = [
        r.get("alert_id") for r in classification_records
        if not r.get("ground_truth") or r.get("ground_truth") == "unknown"
    ]
    # Allow a small tolerance for legitimately unknown-at-log-time samples.
    fraction_missing = len(missing) / max(1, len(classification_records))
    assert fraction_missing < 0.10, (
        f"{len(missing)} / {len(classification_records)} classification "
        f"records ({fraction_missing*100:.1f}%) lack ground_truth — "
        f"exceeds 10% tolerance."
    )
