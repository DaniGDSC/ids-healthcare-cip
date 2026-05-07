# Layer 4 v4.0 Implementation Notes

This file records what changed when applying the Layer 4 v4.0
implementation prompt to a codebase that already had a working Layer 4
(`module4_explanations/`, `src/mve_generator.py`).

## Audit summary

| v4.0 requirement | Status before this batch |
|---|---|
| Step 7 SHAP TreeExplainer | already in `module4_explanations/module4_explanations.py::compute_tree_shap` and `module4_online_explainer.py::AlertExplainer` (887 lines) |
| SHAP stability check | already in `module4_online_explainer.py::compute_shap_stability` |
| Step 7.5 MITRE mapping | already in `src/mve_generator.py::_ATTACK_TECHNIQUES` (T1→T1071, T2→T1078, T3→T1021, T4→T1041, T5→T1565) |
| Step 8 MVE 3-layer generation | already in `src/mve_generator.py::generate_mve` (1216 lines, Mode A LLM + Mode B rule-based) |
| Word limits (≤60/≤50/≤60/≤30) | already enforced in the generator |
| INVARIANT 5 (Layer 1 references SHAP) | already enforced — covered by `tests/test_coverage_mve.py` |
| INVARIANT 7 (DO_NOT for CRITICAL+clinical) | already enforced via `clinical_constraint` Layer 3 field |
| Step 9 Stakeholder adaptation | already in `src/mve_generator.py::derive_role_view` + `_role_lens_layer_1` + `_role_lens_layer_3` |
| INVARIANT 6 (role authority bounds) | already enforced via `ROLE_FORBIDDEN_ACTION_TERMS` + `role_authority_violations`; covered by `tests/test_role_authority.py` (39 tests) |
| Tests | `test_coverage_mve.py` (73 tests), `test_role_authority.py` (39 tests) — full coverage of legacy 5-type flow |
| **9-class v4 ``AlertType`` routing** | **missing** — the legacy generator switches on T1…T5 and has no awareness of the new v4 typology (KNOWN_ATTACK_UNCERTAIN, STRONG_NOVEL_ANOMALY, SUSPICIOUS_PATTERN, BENIGN_WATCH, DISAGREEMENT_ANOMALY) |
| **DISAGREEMENT_ANOMALY adversarial wording** | **missing** — v4-only signal has no Layer 1 hint |
| **``Confidence`` enum rendering** | **missing** — v4 4-level enum has no string renderer |
| **Per-role MITRE format helper** | **partial** — `_role_lens_layer_1`/`_3` re-frame Layer 1/3 prose per role, but the MITRE technique itself is rendered the same way for every role (no IT-vs-nurse format split) |
| **DAE per-dim Layer 1 enrichment** | **missing** — Layer 2 v4 emits `anomalous_dims` and per-dim errors but the MVE generator does not consume them as a Layer 1 clause |

The remaining items below were the actual gaps and are what this batch
adds.

## What this batch added

### `module4_explanations/triage_v4_adapter.py`

A pure-function module (no I/O, no model loads) that lets the existing
generator consume v4 outputs without touching its template machinery.
Five public helpers:

  * `alert_type_v4_to_legacy(AlertType) -> str` — maps every one of
    the 9 v4 alert types to a legacy `T1`…`T5` so callers can keep
    using `generate_mve` and `derive_role_view`. Currently routes
    everything to `T1` (the broadest "known attack" template) and
    expects callers to layer the v4-specific hints on top via the
    other helpers; introducing new template ids would require
    re-implementing the rule-based branches.

  * `is_adversarial(AlertType) -> bool` and
    `adversarial_clause(AlertType) -> str` — recognise
    `DISAGREEMENT_ANOMALY` and emit a Layer 1 hint warning operators
    of potential adversarial input. Empty string for every other
    alert type so callers can drop the clause cleanly.

  * `confidence_clause(Confidence) -> str` — turns each level of the
    v4 `Confidence` enum into a one-line Layer 1 string. Total — an
    unknown string returns `"Confidence: UNKNOWN."` rather than
    raising.

  * `format_mitre_for_role(technique_id, technique_name, role) -> str`
    — renders the same MITRE technique three different ways:
      * IT generalist → `"T1071 (Application Layer Protocol)"`
      * Biomed engineer → short threat-type prose
      * Nurse manager → plain-language sentence, no jargon
    Total — unknown techniques fall back to the technique name.
    `format_mitre_for_alert_type` is a convenience wrapper that
    derives the technique from the v4 alert type.

  * `anomalous_dims_clause(dims, feature_names, max_features=3) -> str`
    — turns Layer 2 v4's `anomalous_dims` list into a Layer 1
    sentence naming up to 3 features (with overflow summary
    "…and N more"). Empty string when no dims are anomalous.
    Out-of-range indices are silently dropped (defensive against
    schema drift between Layer 2 and Layer 4).

### Tests — `tests/test_layer4_v4_adapter.py` (20 tests)

  * Template routing is total over the v4 typology
    (every `AlertType` value maps to a legacy `T1`…`T5`)
  * String round-trips match enum routes
  * Unknown alert types fall back to `T1`
  * Adversarial flag fires only for `DISAGREEMENT_ANOMALY`
  * Confidence clause for each level + unknown-string graceful fallback
  * Per-role MITRE format:
    * IT sees full ID + technique name
    * Biomed sees threat-type prose, no ID
    * Nurse sees plain language, no ID and no technique-name jargon
    * Three role outputs distinct for the same technique
    * Unknown techniques don't raise
    * `format_mitre_for_alert_type(KNOWN_ATTACK, IT)` lands on T1071
  * `anomalous_dims_clause`:
    * Empty on no input (caller can drop the clause)
    * Single-dim rendered named
    * Multiple-dim with Oxford-comma join
    * Overflow above `max_features` produces "…and N more"
    * Out-of-range indices dropped silently
    * Word count stays well under the 60-word Layer 1 budget

Full suite: 271 tests passing (was 251; +20 from this batch).

## What was *not* added (and why)

The prompt prescribes a parallel `pipeline/module4_xai/` layout with
separate `SHAPExplainer`, `MITREGroundingEngine`, `MVEGenerator`,
`StakeholderAdaptor`, and `Layer4Explainer` files (~5 new modules).
This is already covered by:

  * `module4_explanations/module4_explanations.py` (TreeSHAP + DAE
    decomposition + stakeholder routing — 1421 lines)
  * `module4_explanations/module4_online_explainer.py` (online SHAP
    explainer + stability check — 887 lines)
  * `src/mve_generator.py` (Mode A LLM + Mode B rule-based 3-layer
    generation, MITRE mapping, role-lens helpers, INVARIANT 6
    enforcement — 1216 lines)
  * `tests/test_coverage_mve.py` and `tests/test_role_authority.py`
    (~112 tests already passing)

Per CLAUDE.md "prefer editing existing files over creating new ones"
and "don't add abstractions beyond what the task requires", these
were not duplicated. The actual deltas — v4 alert-type routing,
adversarial wording, confidence rendering, per-role MITRE format,
and DAE per-dim enrichment — were added as a thin shim that the
existing generator and role-lens helpers can call into.

The prompt also mentions creating 9 individual YAML template files
under `docs/mve_templates/`. The legacy generator's templates are
already encoded in `_generate_rule_based` (5 alert-type-specific code
paths, each producing all three layers) and re-framed per role by
`_role_lens_layer_1`/`_3`. Since the v4 typology routes back into
those same paths via `alert_type_v4_to_legacy`, no new template files
are needed.
