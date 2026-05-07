# Layer 3 v4.0 Implementation Notes

This file records what changed when applying the Layer 3 v4.0
implementation prompt to a codebase that already had a working Layer 3
(`module3_risk_scoring/`, `src/risk_scorer.py`).

## Audit summary

| v4.0 requirement | Status before this batch |
|---|---|
| `c_detect = max(c_track_a, c_track_b)`, INVARIANT 1 | already enforced — `module3_risk_scoring/module3_risk_scores.py::compute_c_detect` (batch path) and `Layer2Detector.score_alert` (per-alert, added in Layer 2 v4) |
| `R = 0.40·C_detect + 0.25·D_crit + 0.15·S_data + 0.20·D_clinical_tier` | already in `module3_risk_scoring/module3_risk_scores.py::compute_composite_risk` |
| `D_clinical_tier` (renamed from A_patient) | already named `d_clinical_tier` everywhere downstream |
| Severity tier mapping (LOW/MEDIUM/HIGH/CRITICAL) | already in `module3_risk_scoring/module3_risk_scores.py::assign_risk_levels` |
| Per-device-class threshold multipliers | already in `src/risk_scorer.py::_THRESHOLD_MULT_BY_DEVICE` |
| Safety floor (CRITICAL+unpatchable) | already in `src/risk_scorer.py::score_alert` (line 262) |
| Maintenance window does NOT bypass safety floor | already in `src/risk_scorer.py::score_alert` (line 207, ST-03 fix) |
| `similar_events_past_30d > 5` adjustment | already in `src/risk_scorer.py::score_alert` (line 244) |
| `DISAGREEMENT_ANOMALY` for adversarial detection | already in `FusionClass` (Enhancement 4) |
| `KNOWN_ATTACK`, `CONFIRMED_ANOMALY`, `NOVEL_ANOMALY`, `BENIGN` | already in `FusionClass` |
| `KNOWN_ATTACK_UNCERTAIN`, `STRONG_NOVEL_ANOMALY`, `SUSPICIOUS_PATTERN`, `BENIGN_WATCH` (4 new alert types) | **missing** — `FusionClass` covers 5 of the prompt's 9 types |
| `Confidence` indicator (VERY_HIGH/HIGH/MEDIUM/LOW) | **missing** — never modeled |
| 9-stage triage decision tree | **missing** — fusion logic exists but doesn't refine into the 9-class typology |
| `clinical_active` adjustment | **missing** — `score_alert` knows about `is_maintenance_window`, `similar_events_past_30d`, `baseline_days`, but not `clinical_active` |

The remaining items below were the actual gaps and are what this batch
adds.

## What this batch added

### `AlertType` and `Confidence` enums

[src/data_models.py](../src/data_models.py): added two new enums.

  * `AlertType` — the 9-class enriched triage typology
    (KNOWN_ATTACK, KNOWN_ATTACK_UNCERTAIN, DISAGREEMENT_ANOMALY,
    STRONG_NOVEL_ANOMALY, NOVEL_ANOMALY, CONFIRMED_ANOMALY,
    SUSPICIOUS_PATTERN, BENIGN_WATCH, BENIGN).
  * `Confidence` — VERY_HIGH / HIGH / MEDIUM / LOW.

The existing `FusionClass` (5-class cascade-fusion outcome) was **left
untouched** to preserve backward compatibility with
`module3_risk_scores`, `multiclass_fusion`, and ~40 callsites across
the codebase. `AlertType` is a v4 refinement layered on top — it
overlaps with `FusionClass` on the 5 shared types and adds the 4 new
ones.

### `classify_alert_v4` — 9-stage triage classifier

[module3_risk_scoring/triage_v4.py](../module3_risk_scoring/triage_v4.py):
new module exposing a single function `classify_alert_v4(p_xgb, p_rf,
p_dt, diversity_score, dae_score, threshold_level)` and a
`TriageDecisionV4` dataclass.

The 9 stages, in evaluation order:

```
Stage 1 KNOWN_ATTACK            p_xgb >= 0.85 AND diversity < 0.15
Stage 2 KNOWN_ATTACK_UNCERTAIN  p_xgb >= 0.85 AND diversity >= 0.15
Stage 3 DISAGREEMENT_ANOMALY    diversity >= 0.30 AND dae >= 0.70
Stage 4 STRONG_NOVEL_ANOMALY    p_xgb < 0.40   AND dae >= 0.95
Stage 5 NOVEL_ANOMALY           p_xgb < 0.40   AND 0.70 <= dae < 0.95
Stage 6 CONFIRMED_ANOMALY       0.40 <= p_xgb < 0.85 AND dae >= 0.70
Stage 7 SUSPICIOUS_PATTERN      0.40 <= p_xgb < 0.85 AND dae < 0.70
Stage 8 BENIGN_WATCH            p_xgb < 0.40   AND 0.50 <= dae < 0.70
Stage 9 BENIGN                  default
```

Each decision carries:

  * `c_detect = max(p_xgb, dae_score, normalised_diversity)`
  * the source signals (for audit reproducibility)
  * a `template_id` (consumed by Layer 5 MVE templates — the test
    suite asserts these are unique per `AlertType`)

INVARIANT 1 is enforced at the top of `classify_alert_v4`: an
`AssertionError` is raised if `c_detect < p_xgb`. This is a real
runtime check, not a debug-only Python `assert` that
`python -O` would strip.

### `clinical_active` gate adjustment

[src/risk_scorer.py](../src/risk_scorer.py): the `score_alert` event
context now recognises `clinical_active=True`. When set, the per-
device threshold multiplier drops by 0.10 (with a 0.40 floor),
producing a tighter gate during active patient care so we don't miss
soft signals during procedures.

The reduction does not bypass the existing safety floor — CRITICAL +
unpatchable still surfaces unconditionally regardless of the
`clinical_active` flag, and the multiplier floor (0.40) prevents the
gate from being made trivially loose.

When `clinical_active` is omitted or False, the threshold/multiplier
behaviour is byte-equivalent to the legacy path
(`test_clinical_active_flag_default_false_preserves_baseline`).

### Tests — `tests/test_layer3_v4_triage.py` (20 tests)

  * 9 stage-reachability tests (one per `AlertType`)
  * `test_all_nine_stages_reachable` (sweeps the 9 fixtures, asserts
    every type was hit)
  * `test_classifier_is_deterministic`
  * `test_stage_predicates_partition_input_space` — sweeps a 7×6×8
    grid over the predicate boundaries; every input produces exactly
    one well-formed `(AlertType, Confidence)` pair
  * `test_invariant_1_holds_across_input_grid` — sweeps a 6×4×5 grid,
    asserts `c_detect >= p_xgb` everywhere
  * `test_dae_can_only_elevate_not_reduce` — fixed `p_xgb`, varied DAE
  * 5 `clinical_active` tests:
    * tightens threshold when set
    * 0.40 floor holds
    * default-False preserves baseline
    * does not bypass CRITICAL+unpatchable safety floor
    * combines safely with `similar_events_past_30d`
  * `test_decision_carries_source_signals_for_audit`

Full suite: 251 tests passing (was 231; +20 from this batch).

## What was *not* added (and why)

The prompt prescribes a parallel `pipeline/module3_risk_scoring/`
layout with separate `TriageFusionEngine`, `ContextEnricher`,
`RiskScorer`, `RiskAdaptiveGate`, and `Layer3Orchestrator` files. This
is already covered by:

  * the existing `module3_risk_scoring/fusion.py` (binary +
    multiclass cascade fusion)
  * `module3_risk_scoring/module3_risk_scores.py` (composite risk +
    severity tier + per-device thresholds + batch `compute_c_detect`)
  * `src/risk_scorer.py` (per-alert gate + safety floor + maintenance-
    window logic)

Per CLAUDE.md "prefer editing existing files over creating new ones"
and "don't add abstractions beyond what the task requires", these
were not duplicated. The actual deltas the prompt asks for —
the 4 missing alert types, `Confidence` levels, the 9-stage decision
tree, and `clinical_active` — were added on top of the existing
modules.

The prompt also specifies a `ContextEnricher` with a
`device_inventory.yaml` file. The project already resolves device
class via `common.device_class.derive_device_class_array` (heuristic
from biometric features) and looks up criticality via the dict tables
in `src/risk_scorer.py`. A YAML-driven device inventory would be a
good future enhancement (the current heuristic has a documented
limitation — see `track_a_performance.yaml::PREREQUISITE_GAP_PB_1`),
but it is orthogonal to the v4.0 deltas this batch is delivering.
