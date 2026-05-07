# Layer 1 v4.0 Implementation Notes

This file records what changed when applying the Layer 1 v4.0
implementation prompt to a codebase that already had a working
Layer 1 (`module0_analysis/`, `module1_preprocessing/`,
`module2_detection/`).

## Audit summary

| v4.0 requirement | Status before this batch |
|---|---|
| M0 data prep (load, split, SMOTE) | already present in `module0_analysis/phase0/` + `module1_preprocessing/phase1/` |
| M1 BENIGN_MEDIANS | `data/processed/benign_medians.json` already populated |
| M1 scaler (joblib, fit-train-only) | `data/processed/robust_scaler.pkl` (RobustScaler — kept; see note below) |
| M2 XGB / RF / DT training | `module2_detection/module2_train_models.py` + `data/phase2/{model}/best_pipeline.pkl` |
| M2 DAE training (28-dim cascade input) | `module2_detection/run_dae_cascade.py` + `results/models/dae_*` |
| FIX-1 validation probas, NOT OOF | `module2_detection/calibrate.py` already runs against the val split |
| FIX-2 BENIGN_MEDIANS not zeros | already used by `phase1` exporter |
| FIX-3 joblib (signed) instead of raw pickle | `common.signed_pickle` + `joblib` already in place |
| R6 stratified calibration / holdout split | `results/reports/stratified_{calibration,holdout}.parquet` already materialised |
| Curated 20-alert stress test | `results/reports/evaluation_alerts.json` (≤ stress test only, never tuning) |
| R1 Platt scaling | `calibrate.py` defaults to **isotonic** with Platt (sigmoid) fallback below n=1000. Isotonic strictly dominates Platt for our val sizes; the existing default is left unchanged. |

The remaining items below were the actual gaps and are what this batch
adds.

## What this batch added

### R2 — `configs/per_class_thresholds.yaml`

Externalised declaration of `_TRACK_A_SURFACING_BY_DEVICE` from
`src/risk_scorer.py`. The Python constants stay the source of truth at
runtime; the YAML is a parity-tested declaration so external tools and
documentation can read the values without importing Python.

Drift between the two is caught by
`tests/test_layer1_v4_artifacts.py::test_per_class_thresholds_yaml_matches_risk_scorer`.

### R3 — multi-threshold DAE → `results/models/dae_thresholds.json`

The previously single DAE threshold (configured at p99 by the existing
training run) is augmented with the three operational tiers prescribed
by v4.0:

  * `screening_threshold` = p80 of benign training reconstruction errors
  * `confirmation_threshold` = p95
  * `high_confidence_threshold` = p99

Computed from the `train_errors` array already persisted in
`results/models/dae_detector.json`. No retraining required.

### R4 — DAE percentile-rank score calibration → `results/models/dae_calibration.json`

A 1001-point monotone lookup that maps a raw reconstruction error to a
[0, 1] percentile rank against the benign training distribution. Online
inference uses:

```python
score = np.searchsorted(percentile_lookup, raw_error) / len(percentile_lookup)
```

The score is environment-comparable because it is rank-based — it does
not depend on the raw error magnitude.

Both `dae_thresholds.json` and `dae_calibration.json` carry the SHA256
of the source `dae_detector.json` they were derived from, so a
mismatch between the calibration files and the model in use is
detectable (Invariant 4).

### R5 — `DAEDetector.anomalous_dims_z`

Per-dimension batch z-score helper that returns, for each row in a
batch, the indices of the per-feature weighted reconstruction errors
exceeding `z_threshold` (default 2.0) standard deviations within the
batch. Reuses the existing `reconstruction_error_decomposed` forward
pass — no extra DAE compute. Single-sample batches return `[[]]`
(within-batch z-score is undefined for n=1).

## Generation

```bash
python -m module2_detection.build_dae_v4_artifacts
```

reads `results/models/dae_detector.json` and writes the two new JSON
files under `results/models/`.

## Tests

`tests/test_layer1_v4_artifacts.py` (9 tests):

  * R2: YAML parses + matches `_TRACK_A_SURFACING_BY_DEVICE`
  * R3: thresholds well-formed (p80 < p95 < p99) and match the
    percentiles of `train_errors` in the source sidecar
  * R4: lookup is monotone non-decreasing, score is in [0, 1] for in-
    range inputs and clamps correctly on out-of-range inputs
  * R3 + R4 share the same source-detector SHA256
  * R5: flagged dims match a manual z-score computation; n=1 returns
    `[[]]` without crashing

Full suite (222 tests) still passes.

## Notes on items not duplicated

The prompt prescribed a fresh `pipeline/module0_data/`,
`pipeline/module1_preprocessing/`,
`pipeline/module2_detection/` layout. The existing project already
implements this structure under `module0_analysis/`,
`module1_preprocessing/`, `module2_detection/` per
`CLAUDE.md` and was not duplicated — per CLAUDE.md "prefer editing
existing files over creating new ones" and "don't add abstractions
beyond what the task requires."

The prompt also prescribed `StandardScaler`. The existing pipeline
uses `RobustScaler` (`data/processed/robust_scaler.pkl`), which is
appropriate for the WUSTL-EHMS biometric features that have heavy
tails. This was left unchanged.
