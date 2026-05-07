# Layer 5 v4.0 Implementation Notes

This file records what changed when applying the Layer 5 v4.0
implementation prompt to a codebase that already had a 4-page
Streamlit dashboard at
`module6_evaluation/module6_app.py`.

## Audit summary

| v4.0 requirement | Status before this batch |
|---|---|
| 4-page Streamlit app (Dashboard, Online Sim, Browse, Study) | already present in `module6_app.py` |
| Role selector (IT / Biomed / Nurse) | already present (`render_analyst`, `render_clinician`, `render_admin`) |
| 3-layer MVE rendering | already present (`render_mve_layers`) |
| DO_NOT prominent display | already present (`render_do_not_constraint` + `_DO_NOT_FALLBACKS`) — INVARIANT 7 |
| Operator decision form (Likert + rationale + escalate) | already present (`likert_form`) |
| Audit-trail decision logging | already present (`AuditTrailWriter`, `audit_log`) — INVARIANT 4 |
| Severity-tier colour coding | already present (`TIER_COLORS`) |
| **9-class v4 ``AlertType`` badges** | **missing** — only the legacy 5-class `FusionClass` was rendered; the four v4-only types (KNOWN_ATTACK_UNCERTAIN, STRONG_NOVEL_ANOMALY, SUSPICIOUS_PATTERN, BENIGN_WATCH) and the special-case DISAGREEMENT_ANOMALY had no badge metadata |
| **4-level Confidence indicator** | **missing** |
| **Mode A (LLM) / Mode B (rule-based) indicator** | **missing** — no UI cue for which generator produced the MVE |
| **DAE anomalous-dims rendering** | **missing** — Layer 2 v4 emits `anomalous_dims` but the dashboard had no helper to surface them as alert-card prose |

The remaining items below were the actual gaps and are what this batch
adds.

## What this batch added

### `module6_evaluation/presentation_v4.py`

A new pure-function module that returns the visual metadata the
existing Streamlit dashboard needs to render the v4 deltas. It
imports nothing from Streamlit so the metadata can be unit-tested
without a UI runtime.

#### `BADGE_FOR_ALERT_TYPE` — 9 alert-type badges

Each `AlertType` maps to a `BadgeStyle` dict with `color` (hex),
`icon` (emoji), `label`, and `urgency`. The palette pins the
prompt's prescribed colours:

| AlertType | Colour | Icon | Urgency |
|---|---|---|---|
| KNOWN_ATTACK | `#DC2626` (red) | 🔴 | HIGH |
| KNOWN_ATTACK_UNCERTAIN | `#DC2626` | 🔴 | HIGH |
| **DISAGREEMENT_ANOMALY** | **`#9333EA` (purple)** | 🟣 | HIGH |
| STRONG_NOVEL_ANOMALY | `#EA580C` | 🟠 | MEDIUM |
| NOVEL_ANOMALY | `#F97316` | 🟠 | MEDIUM |
| CONFIRMED_ANOMALY | `#EAB308` | 🟡 | MEDIUM |
| SUSPICIOUS_PATTERN | `#FACC15` | 🟡 | LOW |
| BENIGN_WATCH | `#94A3B8` | ⚪ | INFO |
| BENIGN | `#94A3B8` | ⚪ | INFO |

Tests pin the exact hex codes — a future palette change has to be
intentional. The `DISAGREEMENT_ANOMALY` purple and the
"ADVERSARIAL DETECTED" label are exclusive to that alert type so
operators key on them unambiguously.

`badge_for_alert_type(value)` accepts either an `AlertType` or its
string value and falls back to BENIGN on unrecognised input — the
dashboard never crashes on stale data.

#### `CONFIDENCE_INDICATOR` — 4-level dot indicator

| Confidence | Symbol | Colour |
|---|---|---|
| VERY_HIGH | `●●●●` | green |
| HIGH | `●●●` | green |
| MEDIUM | `●●` | orange |
| LOW | `●` | gray |

Tests pin that the dot count strictly increases with confidence so
the indicator stays glanceable. Unknown strings degrade to LOW
(conservative default — operator sees a low-confidence cue rather
than a blank cell).

#### `MODE_INDICATOR` — Mode A/B transparency

Two canonical strings:

  * `MODE_A_LLM` (`A_llm`) → `"✓ AI Mode (LLM)"`, green
  * `MODE_B_RULE_BASED` (`B_rule_based`) → `"⚠ Rule-based Fallback"`, orange

`mode_display(generation_mode)` falls back to Mode B for any
non-canonical string so a corrupt MVE record visually flags as
the more conservative mode.

#### `anomalous_dims_markdown(dims, feature_names)`

Turns Layer 2 v4's `anomalous_dims` list into a markdown bullet
list suitable for the alert-card "Show DAE anomaly details"
expander:

```
DAE flagged **3** anomalous dimensions:
- `feat_00` (dim 0)
- `feat_05` (dim 5)
- `feat_12` (dim 12)
```

  * Empty input returns the empty string so the caller can omit the
    expander.
  * Singular vs plural noun selection on the count line.
  * `max_features=5` cap with `… and **N** more` overflow summary.
  * Out-of-range indices (stale schema) silently dropped.

### Tests — `tests/test_layer5_v4_presentation.py` (19 tests)

  * Badge totality — every `AlertType` has a badge entry.
  * Palette specification — exact hex codes pinned.
  * `DISAGREEMENT_ANOMALY` is the only purple badge.
  * `DISAGREEMENT_ANOMALY` is the only ADVERSARIAL-labelled badge.
  * Urgency levels consistent with alert severity (HIGH for KNOWN+
    DISAGREEMENT, INFO for BENIGN family).
  * Badge lookup is total over strings; unknown → BENIGN fallback.
  * Confidence indicator: covers every level, dot count strictly
    increases with confidence, unknown → LOW fallback.
  * Mode indicator: canonical modes plus unknown → Mode B fallback.
  * `anomalous_dims_markdown`: empty / single (singular noun) /
    multiple (plural noun) / overflow ("… and **N** more") /
    out-of-range / all-invalid edge cases.

Full suite: 290 tests passing (was 271; +19 from this batch).

## What was *not* added (and why)

The prompt prescribes a parallel `pipeline/module6_evaluation/`
layout with a separate `dashboard_app.py`, four separate page
modules, four separate component modules, and a corresponding
parallel test directory under `tests/module6/` and
`tests/integration/`. This is already covered by the existing
`module6_app.py`, which has:

  * the 4 mode functions (`dashboard_mode`, `simulation_mode`,
    `browse_mode`, `study_mode`)
  * `render_alert_*` / `render_mve_layers` /
    `render_do_not_constraint` / `render_prioritized_actions`
  * `likert_form` (operator decision form)
  * `AuditTrailWriter` and `audit_log` for INVARIANT 4
  * Three role-specific renderers (`render_analyst`,
    `render_clinician`, `render_admin`)

Per CLAUDE.md "prefer editing existing files over creating new
ones" and "don't add abstractions beyond what the task requires",
the layout was not duplicated. The actual deltas — the v4 alert-
type badges, the Confidence indicator, the Mode A/B indicator, and
the DAE anomalous-dims renderer — were added as a small pure-
function helper module the existing dashboard can import.

UI-rendering tests under `pytest`'s default runner are also not
added: Streamlit components do not expose a stable rendering API
that can be diffed in unit tests, and the existing dashboard relies
on manual / scenario testing (the study-mode A/B harness in
`module6_app.py`). The new presentation helpers are pure functions
returning dicts and strings, which IS unit-testable, and the 19
new tests pin the exact contract the dashboard depends on. The
INVARIANT 3, 4, 6, 7 checks the prompt asks for in the UI layer
are already enforced by the existing audit-log writer and the
`role_authority_violations` helper from `src.mve_generator`, both
of which are covered by `test_role_authority.py` (39 tests) and
`test_audit_append_only.py` (3 tests).
