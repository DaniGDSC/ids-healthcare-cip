# RQ3 — Tier × Surfacing Appendix

**Purpose:** Paper-appendix wrapper that points RQ3 readers at the
shared tier × surfacing truth table without duplicating it. The full
table lives in [`docs/rq1_tier_surfacing_truth.md`](rq1_tier_surfacing_truth.md)
because the underlying data (tier boundaries, surfacing decision,
ground-truth crosstab) is the same artifact RQ1 audits.

This appendix adds **RQ3-specific lens**: how each tier maps to the
HITL workflow surfaces the user study exercised, what operator response
each tier triggers, and how the audit chain attributes the decision.

---

## 1. Tier → operator surface mapping

| Tier | Surfacing decision | Operator surface | Action vocabulary exposed | Audit channel |
|------|--------------------|-------------------|---------------------------|---------------|
| CRITICAL | escalate | Triage queue → T3 SOC + biomed | acknowledge / escalate / dismiss (with rationale) | `audit_log.jsonl` (alert audit + reviewer_interaction) |
| HIGH     | surface (actionable) | Triage queue → T2 SOC analyst | acknowledge / escalate / investigate / dismiss | same |
| MEDIUM   | surface (informational) | Triage queue → T1 SOC analyst | acknowledge / monitor / investigate / dismiss | same |
| LOW      | suppress | Audit log only — not on operator queue | (no operator-side action) | `audit_log.jsonl` (alert audit; no reviewer event by construction) |

> See [`config/tier_routing.yaml`](../config/tier_routing.yaml) for the
> machine-readable form of this mapping (used by Module 5's response
> builder).

## 2. Role × tier authorization

The user study exercised three operator roles. Each saw the same
underlying tier but a different action set (per
[`config/role_action_authorization.yaml`](../config/role_action_authorization.yaml)):

| Role | CRITICAL | HIGH | MEDIUM | LOW |
|------|----------|------|--------|-----|
| analyst (IT Generalist proxy) | ack / esc / dismiss + isolate (with admin approval) / investigate / block_destination | same minus mandatory escalate | ack / monitor / investigate / dismiss | (no surface) |
| clinician (Bedside Clinician proxy) | ack / esc / dismiss + verify_clinical / request_biomed | same | ack / dismiss + verify_clinical | (no surface) |
| administrator (Biomed / Service Line proxy) | ack / esc / dismiss + approval queue + verify_clinical / request_biomed | same | same | (no surface) |

**Invariant 6 (severity invariance):** The `risk_level` field is set by
Module 5 BEFORE the role renderer dispatches; switching roles changes
the *action vocabulary* and *explanation framing*, not the underlying
severity. Tests: `tests/test_step13_cross_role_consistency.py`.

**Invariant 9 (shared anchor):** `alert_id` / `sample_index` / `timestamp`
are identical across role views. Verified across the 300 M6 records.

## 3. Audit attribution per tier

For each (tier, role, action) combination, the audit chain records:

- **Alert-audit channel** (`event_type=null`, hash-chained): alert_id,
  sample_index, timestamp, actions_executed, auto_executed (always
  false — Invariant 3), clinical_override, ground_truth, integrity_hash,
  prev_hash, signature.
- **Reviewer channel** (`event_type=reviewer_interaction|dashboard_action`,
  same hash chain): alert_id, action (acknowledge/escalate/dismiss),
  reviewer_id, reviewer_role, decision_time_seconds, rationale (when
  dismiss), signature.

Spec §3's 10 required fields are split across these two channels by
design. See `results/rq3_audit_integrity.json` for the full coverage
audit.

## 4. LOW-tier suppression — defendability

The user study saw zero LOW-tier alerts on the operator queue (LOW =
suppress per tier_routing.yaml). This was deliberate:

- RQ3 measures HITL *decision quality on surfaced alerts*, not
  *vigilance fatigue on noise*.
- LOW-tier alerts are still audit-logged (Invariant 5 — no silent
  suppression). 19 LOW-tier false negatives on the RQ1 test split sit
  in the audit log; none are on critical devices (see RQ1 §4).

## 5. Cross-reference table

| Question | Where to look |
|----------|---------------|
| Full tier × ground-truth crosstab (n=2,448) | `docs/rq1_tier_surfacing_truth.md` §2 |
| Tier × device-criticality crosstab | `docs/rq1_tier_surfacing_truth.md` §3 |
| Tier routing YAML (machine-readable) | `config/tier_routing.yaml` |
| Role × action authorization YAML | `config/role_action_authorization.yaml` |
| RQ3 per-role accuracy by condition | `analysis/outputs/rq3_per_role.json` |
| Audit chain integrity | `results/rq3_audit_integrity.json` |
| Invariant test suite (19 tests) | `tests/test_safe_failure.py`, `tests/test_step13_cross_role_consistency.py`, `tests/test_step16_audit_integrity.py` |

---

## Reproducibility

This appendix is documentation only — no script regenerates it.
Updates required when:
- `config/tier_routing.yaml` or `config/role_action_authorization.yaml`
  changes (the YAML drift CI test under `tests/test_rq3_config_sync.py`
  catches code/config mismatch)
- Tier definitions change in `module3_risk_scoring.RISK_THRESHOLDS`
  (would require RQ1 tier-truth table update first)
