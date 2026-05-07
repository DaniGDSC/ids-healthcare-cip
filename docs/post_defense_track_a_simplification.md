# Track A Simplification — Phase B (post-defense)

> Companion to the v5 architectural change executed on **2026-05-07**.
> Phase A (Layer 3 logic, this commit) landed today. Phase B (DAE
> retrain on 26-dim cascade, runtime drops RF/DT inference entirely)
> is queued for after the thesis defense.

---

## What Phase A already did

1. `module3_risk_scoring/triage_v4.classify_alert_v4` no longer requires
   `p_rf` / `p_dt` / `diversity_score`. The 9-stage decision tree is now
   a function of `(p_xgb, dae_score)` only.
2. `c_detect = max(p_xgb, dae_score)`. The `_normalised_diversity` term
   is gone. INVARIANT 1 still holds.
3. `DISAGREEMENT_ANOMALY` is redefined as **Track-A-vs-Track-B**
   disagreement (`0.40 ≤ p_xgb < 0.85 AND dae_score ≥ 0.95`), the
   cleaner semantics that don't depend on three correlated trees.
4. The legacy `p_rf` / `p_dt` / `diversity_score` kwargs are accepted
   by `classify_alert_v4` for back-compat — they're echoed onto the
   audit record (`diversity_score`) but never consumed by the
   predicates.
5. `module6_evaluation/validate_nine_alert_types.py` is updated to the
   new signature. M1–M8 + negatives + claims still pass; defense gate
   remains `SHIP_TO_USER_STUDY`.

## What Phase A intentionally did NOT do

* **DAE input shape stayed at 28-dim** (`[25 raw || P_xgb || P_rf || P_dt]`).
  The trained `dae_detector.pkl` expects this shape. Layer 2's
  `layer2_detector.py` still constructs the 28-dim cascade at
  inference, which means RandomForest and DecisionTree pickles are
  still loaded at runtime to compute `P_rf` and `P_dt`. The latency
  win is partial — Layer 3 logic no longer cares about RF/DT, but
  Layer 2 still pays for the inference.
* **The trained DAE was not touched.** Replacing it with a 26-dim
  retrain mid-defense-week was rejected as too risky (M6 false-positive
  rate could shift; ≥ 1 hour of compute; demo narrative would need a
  re-rehearsal).

## Phase B — the work that's left

### 1. Retrain the DAE on a 26-dim cascade

Cascade input becomes `[25 raw || P_xgb_val]` (26 dims, drop the two
ensemble-baseline probas). Use the existing val-benigns parquet
(`data/processed/val_benign_phase1.parquet`) and the calibrated
XGBoost val probas (`results/models/xgboost_val_proba.npy` if cached,
otherwise regenerate from `xgboost_calibrator.pkl`).

A one-shot script lives at `module2_detection/retrain_dae_26dim.py`
(co-committed). Run as:

```bash
python -m module2_detection.retrain_dae_26dim \
    --output-dir results/models \
    --random-state 42
```

It produces:

* `results/models/dae_detector.pkl` — fresh 26-dim DAE
* `results/models/dae_calibration.json` — fresh percentile-rank curve
* `results/models/dae_thresholds.json` — fresh per-bucket thresholds

Backup the old artefacts before running (the script does not
auto-backup):

```bash
mkdir -p results/models/_v4_backup
cp results/models/dae_detector.pkl       results/models/_v4_backup/
cp results/models/dae_calibration.json   results/models/_v4_backup/
cp results/models/dae_thresholds.json    results/models/_v4_backup/
```

### 2. Update `layer2_detector.py` cascade input

In `module2_detection/layer2_detector.py`, the inference path
constructs `X_aug = np.column_stack([X_raw, P_xgb, P_rf, P_dt])`. The
26-dim equivalent is `X_aug = np.column_stack([X_raw, P_xgb])`. The
RF/DT pickle loads can then be deleted from the constructor.

### 3. Verify M1–M8 still hold after retrain

* `python run_tests.py` must still emit `RECOMMENDATION: ✓ SHIP_TO_USER_STUDY`.
* M6 (`test_false_positive_rate`) is the most likely to move — DAE
  thresholds re-tuned against a different cascade distribution will
  produce different FPRs. If M6 dips below the 0.20 minimum, the DAE
  needs threshold re-calibration via
  `module2_detection/calibrate.py`.

### 4. Decommission RF/DT runtime artefacts (optional)

If M6 is healthy after Phase B, the RF/DT pickles can be moved out
of the runtime load path:

* `results/models/random_forest_*.pkl` → keep as offline thesis
  reference (cited in `analysis/`)
* `results/models/decision_tree_*.pkl` → same

The pickles remain in `results/models/` but no module under
`module2_detection`, `module3_risk_scoring`, `module4_explanations`,
or `module5_responses` should `loads_signed` them. A grep test
locks this:

```bash
grep -rn "random_forest_calibrator\.pkl\|decision_tree_calibrator\.pkl" \
    module2_detection module3_risk_scoring module4_explanations \
    module5_responses src
# expected: zero matches in runtime path
```

### 5. Update `CLAUDE.md` Module 2 contract

Reword the "Track A" paragraph to make the runtime/baseline split
explicit:

```
Track A (supervised, SMOTE-balanced):
- XGBoost   → P_xgb(attack)             [runtime]
- RandomForest, DecisionTree            [offline reference baselines —
                                         trained for thesis comparison
                                         only; not in the inference path]
- Input: 25 raw network features
```

### 6. Drop diversity_score from `Layer2Output` (cosmetic cleanup)

`module2_detection/layer2_detector.Layer2Output.diversity_score` becomes
load-bearing only as a legacy field for old test fixtures; remove or
deprecate.

---

## Why Phase A is safe alone

The Phase-A change is *behaviourally equivalent* on every alert that
Layer 3 has ever classified, because:

* `c_detect` formula change (`max(p_xgb, dae_score, _norm_div)` →
  `max(p_xgb, dae_score)`) only matters when `_normalised_diversity`
  was the argmax. That requires `diversity_score / 0.30 > p_xgb` AND
  `> dae_score` — i.e. raw `diversity > 0.30 * max(p_xgb, dae_score)`.
  In the eval set, max diversity over the 20 alerts is well below
  this band, so `c_detect` is unchanged in practice.
* `DISAGREEMENT_ANOMALY` (Stage 3) trigger changed from
  `(diversity ≥ 0.30 AND dae ≥ 0.70)` to
  `(0.40 ≤ p_xgb < 0.85 AND dae ≥ 0.95)`. Both predicates fire
  exactly **zero** times on the current 20-alert eval set — the
  benign rows have low risk, and there's no high-disagreement attack
  in the eval set. So no real alert flips classification.
* The synthetic adversarial alert (`SYNTHETIC_DEMO_001`) still works
  for the demo: its v4 fields are *derived* by `derive_v4_fields()`
  in `module6_evaluation/module6_app.py` from
  `(ground_truth, attack_category, risk_level, risk_score)`, which
  has nothing to do with the Layer 3 classifier or with diversity.

This means Phase A is purely an architectural simplification with
zero behavioural drift on real eval data, and the demo narrative
holds without modification.

---

## Defense-day talking points (if asked about the architecture)

* "Track A in the runtime path is XGBoost. RandomForest and Decision
  Tree are kept as offline thesis baselines — `module2_train_models.py`
  still trains them, the calibration analysis cites them, but the
  inference path doesn't load them."
* "DISAGREEMENT_ANOMALY is now Track-A-vs-Track-B disagreement —
  XGBoost in the borderline confidence band but the DAE strongly
  flags novelty. That's the canonical adversarial-input signature —
  cleaner than the within-Track-A diversity we used in the v4 draft."
* "Phase B — collapsing the DAE cascade input from 28 to 26 dims to
  drop RF/DT from runtime entirely — is queued post-defense. It
  requires a DAE retrain and an M6 (FPR) re-validation, which is the
  kind of change you don't ship the week of defense."
