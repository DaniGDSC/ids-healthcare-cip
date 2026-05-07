# Layer 2 v4.0 Implementation Notes

This file records what changed when applying the Layer 2 v4.0
implementation prompt to a codebase that already had a working
`module2_detection/layer2_detector.py` (`Layer2Detector` +
`Layer2Output`).

## Audit summary

| v4.0 requirement | Status before this batch |
|---|---|
| Step 1 sanitization (NaN → BENIGN_MEDIAN, OK/DEGRADED/FAILED flag) | already in `src.preprocessing.sanitize_features`, called from `Layer2Detector.score_alert` |
| Step 2a Track A (3 calibrated probas + diversity_score) | already in `Layer2Detector.score_alert` |
| Step 2b Track B (28-dim cascade DAE) | already in `Layer2Detector.score_alert` |
| R1 diversity_score | populated in `Layer2Output.diversity_score` |
| R2 multi-threshold (p80/p95/p99) buckets | computed in `Layer2Detector.__init__`, surfaced as `Layer2Output.threshold_level` |
| R3 per-dimension errors + `anomalous_dims` | populated in `Layer2Output` (full per_dim_errors array + per-dim p95 cutoffs) |
| R5 per-class threshold lookup | resolved via `src.risk_scorer.get_track_a_surfacing_threshold` |
| Existing tests | 16 in `test_layer2_detector.py`, 7 in `test_feature_sanitization.py`, plus `test_two_stage_fusion.py` and `test_unified_fusion.py` |
| INVARIANT 1 (c_detect = max(c_track_a, c_track_b), c_detect ≥ p_xgb) | **not enforced in Layer 2** — only in the batch path of Module 3 |
| R4 percentile-rank DAE score calibration | **not used** — the detector applied a piecewise-linear scaling around the single DAE threshold; the canonical Layer 1 v4 `dae_calibration.json` was ignored |
| Multi-threshold loaded from `dae_thresholds.json` | derived from `train_errors` at every detector construction; the canonical Layer 1 v4 JSON was ignored |
| EA-06 NaN-replacement-with-median verification | only the `data_quality_flag` was checked; whether the imputed value was the median (vs. 0.0 — exploitable) was not |
| Layer 2 latency budget verification | not asserted by any test |

## What this batch added

### INVARIANT 1 enforcement at Layer 2

[module2_detection/layer2_detector.py](../module2_detection/layer2_detector.py):

  * Added `c_detect: float` and `dae_score_calibration: str` fields to
    `Layer2Output`.
  * `score_alert` now computes `c_detect = max(c_track_a, c_track_b)`
    and raises `AssertionError` if `c_detect < p_xgb`. The assertion is
    a real safety check, not a debug-only `assert` statement that
    Python optimizers can strip.
  * `Layer2Output.as_dict()` exports both new fields so Layer 3 / audit
    consumers see them.

### R4 — percentile-rank DAE score calibration

[module2_detection/layer2_detector.py](../module2_detection/layer2_detector.py):

  * On construction, the detector reads
    `results/models/dae_calibration.json` (the canonical Layer 1 v4
    artifact) and caches the `percentile_lookup` array.
  * `score_alert` now computes
    `dae_score = searchsorted(percentile_lookup, raw_error) / n` —
    a [0, 1] rank against the benign training error distribution.
  * The legacy linear-threshold scaling is preserved as a fallback so
    deployments without the calibration JSON keep working unchanged.
  * `Layer2Output.dae_score_calibration` advertises which path was used
    (`"percentile_rank"` vs `"linear_threshold"`) for audit / debug.

### Multi-threshold from canonical JSON

`Layer2Detector.__init__` now prefers `results/models/dae_thresholds.json`
when present; falls back to recomputing the percentiles from the DAE's
persisted `train_errors` when not. Both paths land on the same
percentiles for the same model, but reading from the canonical JSON
means the detector, the Layer 1 v4 calibration, and any tooling that
quotes "the operational thresholds" are all looking at the same numbers.

### Tests — `tests/test_layer2_v4_invariants.py` (9 tests)

  * `test_c_detect_field_present_and_equals_max`
  * `test_invariant_1_c_detect_geq_p_xgb_across_batch` — sweeps 20
    real test rows
  * `test_invariant_1_holds_for_off_manifold_input` — extreme input
    that pushes the DAE high
  * `test_dae_score_uses_percentile_rank_when_calibration_present`
  * `test_dae_score_matches_searchsorted_against_canonical_lookup` —
    byte-equivalence check against the JSON lookup
  * `test_dae_score_in_unit_interval_for_extreme_inputs`
  * `test_multi_thresholds_match_canonical_json`
  * `test_ea06_nan_replaced_with_benign_median_not_zero` — picks a
    feature with non-zero median, injects NaN, asserts the output is
    the median (not 0.0)
  * `test_score_alert_p95_latency_under_budget` — 50 calls, P95 must
    be under the prompt's 500 ms total Layer 2 budget

Full suite: 231 tests passing (was 222; +9 from this batch).

## What was *not* added (and why)

The prompt prescribes a parallel `pipeline/module1_preprocessing/` /
`pipeline/module2_detection/` layout with separate
`FeatureSanitizer`, `TrackAInference`, `TrackBInference`, and
`Layer2Detector` files. This is already covered by the existing
`Layer2Detector` (single-class orchestration) plus
`src.preprocessing.sanitize_features`,
`module3_risk_scoring.module3_risk_scores.compute_c_detect` (the
batch path), and `src.risk_scorer.get_track_a_surfacing_threshold`.
Per CLAUDE.md "prefer editing existing files over creating new ones"
and "don't add abstractions beyond what the task requires", the layout
was not duplicated. INVARIANT 1, R4 calibration, and canonical
multi-threshold loading — the actual deltas the prompt asks for — were
added to the existing detector instead.

The prompt also describes `joblib.Parallel` execution of the three
Track A trees. The existing detector calls them sequentially because
each tree is fast on a 25-feature single sample (the parallelism
overhead of `joblib.Parallel(n_jobs=3, prefer="threads")` exceeds the
work). The latency test confirms the per-alert P95 is well inside the
500 ms budget without the parallel hop.
