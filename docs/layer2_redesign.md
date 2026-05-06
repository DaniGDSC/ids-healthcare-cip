# Layer 2 Redesign — Per-alert Detector

**Date:** 2026-05-06
**Status:** implemented — `module2_detection/layer2_detector.py` is the canonical Layer 2 entry point.
**Spec:** `docs/system_architecture_final.md` § Layer 2 (locked diagram).

## What changed

The training pipeline (`module2_train_models.py`) and the multi-class refactor stayed batch-oriented. Layer 3 / Layer 4 / Layer 5 callers needed a *per-alert* entry point that runs the full Layer 2 stack on one flow record and returns the canonical output bundle the locked architecture diagram specifies. That entry point is now [`module2_detection/layer2_detector.py`](../module2_detection/layer2_detector.py).

## Public API

```python
from module2_detection.layer2_detector import Layer2Detector, Layer2Output

det = Layer2Detector(prefer_calibrated=True)            # loads once
out = det.score_alert(raw_features, device_class="infusion_pump")
out.p_xgb, out.p_rf, out.p_dt   # 3 calibrated probabilities
out.c_track_a                   # max(p_xgb, p_rf, p_dt)
out.diversity_score             # std(p_xgb, p_rf, p_dt)
out.dae_score                   # in [0, 1]
out.dae_score_raw_error         # raw reconstruction error
out.c_track_b                   # = dae_score (alias)
out.device_class_threshold      # per-device surfacing threshold
out.data_quality_flag           # OK / DEGRADED / FAILED
out.nan_rate                    # fraction of NaN/Inf input cells
out.calibration_used            # bool — at least one tree calibrated?
out.threshold_level             # "single" — Task 4 deferred
out.anomalous_dims              # [] — Task 5 deferred
out.per_dim_errors              # None — Task 5 deferred
out.as_dict()                   # JSON-friendly serialization
```

## Mapping to the architecture diagram

| Diagram field | Code field | Notes |
| --- | --- | --- |
| `P_xgb_calibrated` | `out.p_xgb` | Falls back to raw when no calibrator |
| `P_rf_calibrated` | `out.p_rf` | Same fallback |
| `P_dt_calibrated` | `out.p_dt` | Same fallback |
| `c_track_a = max(probas)` | `out.c_track_a` | Computed in detector |
| `diversity_score = std(P_xgb, P_rf, P_dt)` | `out.diversity_score` | Mirrors `module3_risk_scoring/multiclass_fusion.py::diversity_score` |
| `DAE_score [0, 1]` | `out.dae_score` | Saturating monotone map of raw error around `dae.threshold` |
| `c_track_b = DAE_score` | `out.c_track_b` | Alias for downstream binary fusion |
| `threshold_level (★)` | `out.threshold_level` | **`"single"` — Task 4 deferred** |
| `anomalous_dims (★)` | `out.anomalous_dims` | **`[]` — Task 5 deferred** |
| `reconstruction_quality (★)` | `out.dae_score_raw_error` | Raw error is the rawest available proxy |
| `data_quality_flag` | `out.data_quality_flag` | OK / DEGRADED / FAILED — EA-06 mitigation already in `src/preprocessing.py` |
| `device_class_threshold` | `out.device_class_threshold` | Per-device, from `src/risk_scorer.py::get_track_a_surfacing_threshold` |

## Step-by-step implementation

**Step 1 — Feature sanitization.** `src/preprocessing.py::sanitize_features` already handles NaN/Inf → `BENIGN_MEDIAN[feature_idx]` per the EA-06 fix. The detector calls it directly; nothing new in this layer. Note: scaling is **not** re-applied — the detector assumes input has already been transformed by the persisted scaler. Live alerts must run through `RobustScaler.transform` before reaching `score_alert`.

**Step 2a — Track A.** Loads three classifiers from `results/models/`. When a `*_calibrator.pkl` artefact is present (produced by `module2_detection/calibrate.py`), it's preferred over the raw signed pickle; otherwise transparent fallback. The `CalibratedClassifierCV(prefit)` wrapper exposes `predict_proba` so the call site is identical to the raw classifier. Diversity is `np.std([p_xgb, p_rf, p_dt])`.

**Step 2b — Track B.** Cascade input is `[25 raw_sanitized || P_xgb, P_rf, P_dt]` (28-dim). DAE is loaded once via `DAEDetector.from_artefacts` (no pickle). DAE_score is a saturating map:

```python
if recon_err <= dae.threshold:
    score = 0.5 * (recon_err / dae.threshold)              # [0, 0.5]
else:
    score = 0.5 + 0.5 * min((err - threshold)/threshold, 1.0)  # [0.5, 1]
```

Raw error is preserved separately in `dae_score_raw_error` so callers that want to apply their own threshold (or a future multi-threshold table) can do so without re-running the DAE.

## Deferred items — explicit stubs

| Diagram star (★) | Stub today | When implemented |
| --- | --- | --- |
| Multi-threshold (80/95/99) | `threshold_level = "single"` | Task 4 (4 hours est.) — emits `"strong" / "moderate" / "weak"` |
| Per-dimension errors | `anomalous_dims = []`, `per_dim_errors = None` | Task 5 (1 day est.) — populates from `dae.reconstruction_error_decomposed` |

These stubs are **part of the API**, not afterthoughts. Layer 3 / 4 / 5 code that depends on the Layer 2 output dataclass already has the field names; flipping the deferred items will be a behavioural change inside Layer 2 with no downstream API churn.

## Integration paths

- **Layer 3 fusion**: the unified `module3_risk_scoring/fusion.py::fuse(...)` accepts both `(p_xgb, p_rf, p_dt, dae_score)` (binary mode) and softmax tuples (multiclass). The Layer 2 output's three probas + DAE score feed binary fusion directly. For multiclass routing, callers also need the multi-class softmax matrices, which the per-alert detector does **not** yet emit (it loads the binary trees). A multi-class variant of the detector is a clean future extension — same shape, just swap which trees are loaded.
- **Layer 4 explanation**: `mve_generator` consumes `c_track_a`, `dae_score`, `data_quality_flag` from `Layer2Output.as_dict()`.
- **Layer 5 presentation**: `module5_responses` formats the alert badge from `c_track_a`, `c_track_b`, `device_class_threshold`, `data_quality_flag`.

## Test coverage

[tests/test_layer2_detector.py](../tests/test_layer2_detector.py) — 10 tests, all pass:

1. Output dataclass field shape
2. All probabilities in `[0, 1]`
3. Diversity non-negative and bounded
4. `c_track_a` invariant: equals `max(p_xgb, p_rf, p_dt)`
5. Per-device threshold resolves correctly (infusion_pump=0.03, ehr_workstation=0.10, default=0.05)
6. Clean input → `data_quality_flag = "OK"`
7. NaN-injection (5 cells nulled) → `DEGRADED` or `FAILED` (EA-06 contract)
8. Deferred fields match stub contract (Tasks 4+5 stable until those land)
9. `as_dict()` is JSON-serialisable
10. Calibration-status introspection reports truth

Tests skip cleanly if `results/models/` is missing artefacts (CI without the heavy pipeline pre-baked).

Total suite after Layer 2 redesign: **134/134 passing** (111 baseline + 4 calibration + 9 unified-fusion + 10 layer2-detector).

## Files changed in this leg

- **NEW** [module2_detection/layer2_detector.py](../module2_detection/layer2_detector.py) — `Layer2Detector` + `Layer2Output` dataclass
- **NEW** [tests/test_layer2_detector.py](../tests/test_layer2_detector.py) — 10 contract tests
- **NEW** [docs/layer2_redesign.md](layer2_redesign.md) — this document

No files modified. The new module composes existing helpers (`sanitize_features`, `get_track_a_surfacing_threshold`, `DAEDetector.from_artefacts`, signed-pickle loaders).

## Verification command

```bash
python -m pytest tests/test_layer2_detector.py -v
python -m pytest tests/ --ignore=tests/test_coverage_mve.py
```
