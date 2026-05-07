# Layer 6 v4.0 Implementation Notes

This file records what changed when applying the Layer 6 v4.0
implementation prompt to a codebase that already had a working Layer
6 (`module5_responses/{module5_pipeline,module5_responses}.py`,
`module6_evaluation/module6_app.py::AuditTrailWriter`).

## Audit summary

| v4.0 requirement | Status before this batch |
|---|---|
| Action-set selection by severity tier | already in `module5_responses/module5_pipeline.py::PolicyEngine` (LOW/MEDIUM/HIGH/CRITICAL) |
| Device-tier-aware constraint downgrade | already in `module5_responses/module5_responses.py::select_adaptive_response` + `DEVICE_TIERS` + `MITIGATION_ACTIONS` |
| Audit-record builder | already in `module5_responses/module5_responses.py::build_audit_record` |
| Append-only audit log | already in `module6_evaluation/module6_app.py::AuditTrailWriter` (with hash-chained JSONL) |
| INVARIANT 4 (audit complete + append-only) | already enforced; covered by `tests/test_audit_append_only.py` (3 tests) |
| Operator-decision Likert form | already in `module6_app.py::likert_form` |
| Hospital-realistic fallbacks | partial — `DEVICE_TIERS` has fallback_required logic for clinical devices |
| Effectiveness aggregation | already in `module5_responses/module5_responses.py::compute_effectiveness` |
| **9-class v4 ``AlertType`` → tier routing** | **missing** — the legacy `PolicyEngine` is keyed by severity tier (LOW/MEDIUM/HIGH/CRITICAL), not by the v4 9-type alert taxonomy, so DISAGREEMENT_ANOMALY couldn't route to a security specialist while CONFIRMED_ANOMALY went to senior IT |
| **Confidence-based tier adjustment** | **missing** — no helper consumed the v4 ``Confidence`` enum |
| **Hospital-realistic fallback for after-hours / clinical-active** | **missing** — the existing fallback logic was device-tier-only |
| **`operator_followed_recommendation` helper** | **missing** — comparison logic between operator action and recommended actions did not exist as a reusable helper |
| **INVARIANT 3 grep verification as pytest** | **missing** — INVARIANT 3 was satisfied by the codebase de facto (no execution primitives in `module5_responses/`) but no pytest enforced it as a regression check |

The remaining items below were the actual gaps and are what this batch
adds.

## What this batch added

### `module5_responses/tier_routing_v4.py`

A pure-function module — no I/O, no execution. Three exports:

  * `TierLevel` — 8-value enum (`L1_IMMEDIATE`, `L1_WITH_REVIEW`, `L1`,
    `L1_WITH_SENIOR`, `L2_SPECIALIST`, `L2_SECURITY_SPECIALIST`,
    `AUDIT_LOG`, `SUPPRESSED`).
  * `recommend_tier_v4(alert_type, confidence, *, is_after_hours,
    clinical_active)` — total over the 9-class :class:`AlertType` and
    4-level :class:`Confidence`. Maps each alert type to its
    prescribed tier, applies LOW-confidence demotion, and prepends/
    appends the hospital-realistic fallbacks. Unrecognised alert-type
    strings fall through to the `NOVEL_ANOMALY` policy (most cautious
    "investigate" route).
  * `operator_followed_recommendation(operator_action,
    recommended_actions)` — decision-quality helper for the audit
    record. Case-insensitive match by default; rejects placeholder
    selections like `"— Select action —"` and empty/whitespace inputs
    so the metric isn't inflated by no-ops.

#### Routing table (the prompt's prescribed mapping)

| AlertType | Tier | Adversarial | Immediate? |
|---|---|---|---|
| KNOWN_ATTACK | L1_IMMEDIATE | – | yes |
| KNOWN_ATTACK_UNCERTAIN | L1_WITH_REVIEW | – | yes |
| **DISAGREEMENT_ANOMALY** | **L2_SECURITY_SPECIALIST** | **yes** | – |
| STRONG_NOVEL_ANOMALY | L2_SPECIALIST | – | – |
| NOVEL_ANOMALY | L2_SPECIALIST | – | – |
| CONFIRMED_ANOMALY | L1_WITH_SENIOR | – | yes |
| SUSPICIOUS_PATTERN | L1 | – | – |
| BENIGN_WATCH | AUDIT_LOG | – | – |
| BENIGN | SUPPRESSED | – | – |

The `DISAGREEMENT_ANOMALY` row is the v4-only adversarial route — the
test suite pins that this is the only alert type that produces
`L2_SECURITY_SPECIALIST` and the only one with `adversarial_flag` and
`requires_security_specialist` both True.

#### Confidence adjustment

`LOW` confidence demotes urgent ladder-rungs:

  * `L1_IMMEDIATE` → `L1`
  * `L1` → `AUDIT_LOG`

Other tiers (in particular `L2_SPECIALIST` and
`L2_SECURITY_SPECIALIST`) are NOT on the demotion ladder — a
LOW-confidence DISAGREEMENT_ANOMALY still goes to security, because
the disagreement signal itself is what matters. `MEDIUM`, `HIGH`,
`VERY_HIGH` all preserve the base tier — the helper never silently
*promotes*, that would surprise operators.

#### Hospital-realistic fallback adjustments

  * `is_after_hours=True` appends `"On-call rotation activation"`.
  * `clinical_active=True` *prepends* `"Coordinate with clinical
    staff first (active care)"` — must be FIRST so the operator
    talks to the unit before any device action.

### Tests — `tests/test_layer6_v4_routing.py` (34 tests)

Seven sections:

  1. **Routing totality** — every `AlertType` produces a recommendation
    with a non-empty rationale; string round-trips match enum routes;
    unknown strings fall back to NOVEL_ANOMALY.
  2. **Spec-mandated routing** — parametrised over all 9 alert types,
    pinning the exact `TierLevel` each one produces.
  3. **Adversarial exclusivity** — only DISAGREEMENT_ANOMALY routes to
    L2_SECURITY_SPECIALIST and only it carries the adversarial flags.
  4. **Confidence-based adjustment** — LOW demotes the two urgent
    rungs; HIGH/VERY_HIGH/MEDIUM preserve; LOW does not demote L2
    routes.
  5. **Hospital-realistic fallbacks** — after-hours appends on-call;
    clinical_active prepends coordinate-first; no extras when both
    flags are off.
  6. **`operator_followed_recommendation`** — match/non-match,
    case-insensitive, placeholder selection rejected, empty inputs
    rejected, empty recommendation list returns False (defensive).
  7. **INVARIANT 3 grep verification** — parametrised over six
    execution primitives (`subprocess`, `os.system`, `iptables`,
    `firewall_rule_add`, `os.popen`, `Popen(`); the test walks
    `module5_responses/` and asserts each pattern produces zero
    hits. Includes a self-check that the walker actually visits
    `.py` files so the other tests are not vacuous.

Full suite: 324 tests passing (was 290; +34 from this batch).

## What was *not* added (and why)

The prompt prescribes a parallel `pipeline/module5_response/` and
`pipeline/module6_evaluation/` layout with separate `TierRecommender`,
`ResponseRecommender`, `AuditTrailManager`, and `Layer6Orchestrator`
modules. This is already covered by:

  * `module5_responses/module5_pipeline.py` (1366 lines) —
    `PolicyEngine` (severity-tier action selection), tier policies,
    response-policy export
  * `module5_responses/module5_responses.py` (837 lines) —
    `select_adaptive_response`, `build_audit_record`,
    `compute_effectiveness`, `DEVICE_TIERS`, `MITIGATION_ACTIONS`
  * `module6_evaluation/module6_app.py::AuditTrailWriter` — append-
    only JSONL audit trail with hash chaining, plus the operator
    Likert form, all already integrated with the dashboard
  * `tests/test_audit_append_only.py` — INVARIANT 4 verification
    (existing 3 tests covering the append-only contract)

Per CLAUDE.md "prefer editing existing files over creating new ones"
and "don't add abstractions beyond what the task requires", this
existing infrastructure was not duplicated. The actual deltas the
prompt asks for — the 9-type AlertType tier routing, the Confidence-
based adjustment, the hospital-realistic fallback flags, the
decision-quality helper, and the INVARIANT 3 grep test — were added
on top of it as a small pure-function module + a single test file.

The prompt also prescribes a separate `audit_trail.py` with
`OperatorDecision`, `AuditTrailManager`, `query_decisions`,
`get_audit_summary`, etc. The existing `AuditTrailWriter` already
provides hash-chained JSONL persistence; query/summary code lives
inline in the dashboard's `module6_app.py`. Re-implementing those
behind an alternate API would split the audit log into two formats
that downstream consumers would have to reconcile, so they were
left in place.
