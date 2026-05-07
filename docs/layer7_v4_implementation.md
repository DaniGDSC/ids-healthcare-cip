# Layer 7 v4.0 Implementation Notes

This file records what changed when applying the Layer 7 v4.0
implementation prompt to a codebase that already had a comprehensive
multi-method evaluation suite under `module6_evaluation/` and
`analysis/`.

## Audit summary

| v4.0 method | Status before this batch |
|---|---|
| **Method 1** — Multi-Stakeholder LLM Persona Simulation | **complete** — `analysis/run_llm_persona_simulation.py` (317 lines) executed all 100 personas (50 IT × 30 Biomed × 20 Nurse) × 20 alerts = 2000 LLM calls, results in `survey/study_responses_LLM_*.json` (one file per persona). Analysis in `analysis/analyze_llm_simulation.py` (287 lines) → `survey/m5_multi_role_result.yaml`. |
| **Method 2** — Self-Consistency | **complete** — `analysis/run_self_consistency.py` + `analyze_self_consistency.py` (302 lines). Results in `survey/m5_self_consistency_{raw.json,result.yaml}`. |
| **Method 4** — Heuristic Evaluation | **complete** — `docs/heuristic_evaluation.md` covers Nielsen 10, DARPA XAI 4, NIST AI RMF, healthcare-specific (HFMEA, HL7, HIPAA) with PASS evidence. |
| **Method 5** — Comparative Case Study | **complete** — `analysis/analyze_rq3.py` (555 lines) + `docs/case_study_comparisons.md`. |
| **Method 6** — Formal Compliance | **complete** — `results/reports/req_trace_matrix.yaml` with 25 PASS / 38 REQ-MVE-XX entries documented. |
| **Method 7** — Information Gain | **complete** — referenced in `docs/heuristic_evaluation.md` (Method 7 info-gain 8/8 dims, P2 row). |
| MITRE grounding validation | **complete** — `attck_for_alert_type` in `src/mve_generator.py` covered by `tests/test_layer4_v4_adapter.py` (per-role MITRE distinctness) and the existing `test_coverage_mve.py`. |
| Acceptance tests M1-M8 | **complete** — `tests/acceptance_tests.py` (584 lines), `alignment_report.yaml` shows all metrics PASSING (M1=1.0, M2=1.0, M3=1.0, M4=1.0, M5=1.0, M8=0.931). |
| **9-alert-types end-to-end validation** | **missing** — the v4 enriched typology was added across Layers 3–6 in this batch; nothing exercised the four new alert types (KNOWN_ATTACK_UNCERTAIN, STRONG_NOVEL_ANOMALY, SUSPICIOUS_PATTERN, BENIGN_WATCH) plus DISAGREEMENT_ANOMALY through every v4 helper layer in a single artifact. |

The remaining item below was the actual gap and is what this batch
adds.

## What this batch added

### `module6_evaluation/validate_nine_alert_types.py`

A standalone validator that drives a representative synthetic input
through the full v4 helper stack for each of the nine
:class:`AlertType` values and writes a structured YAML report. The
inputs are tuned to the prompt's prescribed predicate boundaries so
each row triggers exactly one stage of the 9-stage decision tree:

```
KNOWN_ATTACK            p_xgb=0.95, diversity=0.05, dae=0.10
KNOWN_ATTACK_UNCERTAIN  p_xgb=0.95, diversity=0.20, dae=0.10
DISAGREEMENT_ANOMALY    p_xgb=0.50, diversity=0.35, dae=0.80
STRONG_NOVEL_ANOMALY    p_xgb=0.10, diversity=0.05, dae=0.97
NOVEL_ANOMALY           p_xgb=0.10, diversity=0.05, dae=0.80
CONFIRMED_ANOMALY       p_xgb=0.60, diversity=0.05, dae=0.80
SUSPICIOUS_PATTERN      p_xgb=0.55, diversity=0.05, dae=0.30
BENIGN_WATCH            p_xgb=0.10, diversity=0.05, dae=0.55
BENIGN                  p_xgb=0.05, diversity=0.05, dae=0.10
```

For every input the validator records the output of:

  * **Layer 3** — `classify_alert_v4` (`triage_v4.py`): triage
    classifier produces the expected alert type and INVARIANT 1
    (c_detect ≥ p_xgb) holds.
  * **Layer 4** — `triage_v4_adapter.py`: legacy template id ∈
    {T1..T5}, adversarial flag exclusive to DISAGREEMENT, per-role
    MITRE rendering non-empty for IT / Biomed / Nurse.
  * **Layer 5** — `presentation_v4.py`: badge metadata totality, the
    purple `#9333EA` colour exclusive to DISAGREEMENT_ANOMALY.
  * **Layer 6** — `tier_routing_v4.py`: tier recommendation matches
    the prescribed table; `requires_security_specialist` and
    `L2_SECURITY_SPECIALIST` exclusive to DISAGREEMENT_ANOMALY.

The validator writes `results/reports/nine_alert_types_validation.yaml`
with:

  * `format`/`format_version` for downstream consumers
  * `summary` (passed/failed/pass_rate/overall_status)
  * `invariants_verified` (the seven cross-layer invariants checked)
  * `per_type` — for each AlertType, the recorded triage/Layer 4/
    Layer 5/Layer 6 outputs plus a list of `failures` (empty on PASS).

Run via:

```bash
python -m module6_evaluation.validate_nine_alert_types
```

CI-friendly — exits 0 on PASS, 1 on any failure, with the failing
types logged at ERROR level.

### Tests — `tests/test_layer7_v4_nine_types_validation.py` (12 tests)

  * Report well-formedness (format, summary, invariants_verified
    list).
  * Summary covers all nine types and pass + failed = 9.
  * Overall status is PASS (otherwise the helper stack drifted out of
    contract — this is a regression alarm).
  * Every AlertType present exactly once.
  * Every entry passed (failures list empty).
  * `DISAGREEMENT_ANOMALY` is the only L2_SECURITY_SPECIALIST route.
  * `DISAGREEMENT_ANOMALY` is the only entry with `adversarial_flag`.
  * `DISAGREEMENT_ANOMALY` is the only entry with the purple
    `#9333EA` badge.
  * Layer 4 legacy template ∈ {T1..T5} for every type.
  * Per-role MITRE rendering non-empty for every type × role.
  * INVARIANT 1 (c_detect ≥ p_xgb) recorded on every type.
  * On-disk report parses and matches the in-process contract
    (skipped when the YAML hasn't been generated).

Full suite: 336 tests passing (was 324; +12 from this batch).

## What was *not* added (and why)

The prompt prescribes parallel `pipeline/module6_evaluation/methods/`
modules for Methods 1, 2, 4, 5, 6, 7 plus a separate
`MITREGroundingValidator`. These are already implemented and have
been run with real outputs:

  * Method 1: 2000 LLM calls executed, `survey/m5_multi_role_*` on
    disk
  * Method 2: stability rerun executed, `survey/m5_self_consistency_*`
  * Method 4: documented under `docs/heuristic_evaluation.md`
  * Method 5: comparative analysis in `docs/case_study_comparisons.md`
  * Method 6: `results/reports/req_trace_matrix.yaml`
  * Method 7: info-gain 8/8 documented in heuristic_evaluation.md
  * Acceptance: `tests/acceptance_tests.py` + `alignment_report.yaml`

Per CLAUDE.md "prefer editing existing files over creating new ones"
and "don't add abstractions beyond what the task requires", these
methods were not duplicated. The actual gap — that the four v4-only
alert types and the v4 cross-layer routing for DISAGREEMENT_ANOMALY
had no end-to-end validation artifact — was filled with a single
self-contained validator and its 12 contract tests.
