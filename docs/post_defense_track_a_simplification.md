# Track A Simplification — Phase A + Phase B (completion log)

> **Status (2026-05-07)**: Phase A and Phase B both shipped pre-defense.
> The runtime + training paths are now XGBoost-only and the DAE is
> raw 25-dim (no cascade). This file is kept as a completion log; the
> mid-Phase-B intermediate (26-dim cascade) was archived under
> `docs/_archive/retrain_dae_26dim.py`.

---

## Phase A (shipped 2026-05-07 morning)

1. `module3_risk_scoring/triage_v4.classify_alert_v4` no longer
   requires `p_rf` / `p_dt` / `diversity_score`. The 9-stage decision
   tree is now a function of `(p_xgb, dae_score)` only.
2. `c_detect = max(p_xgb, dae_score)`. The `_normalised_diversity` term
   was retired. INVARIANT 1 still holds.
3. `DISAGREEMENT_ANOMALY` is redefined as **Track-A-vs-Track-B**
   disagreement (`0.40 ≤ p_xgb < 0.85 AND dae_score ≥ 0.95`).
4. The legacy `p_rf` / `p_dt` / `diversity_score` kwargs are accepted
   by `classify_alert_v4` for back-compat — they're echoed onto the
   audit record (`diversity_score`) but never consumed by the
   predicates.
5. `module6_evaluation/validate_nine_alert_types.py` updated to the
   new signature. M1–M8 + negatives + claims still pass; defense
   gate is `SHIP_TO_USER_STUDY`.

## Phase B (shipped 2026-05-07 afternoon)

### What landed

1. **DAE retrained on raw 25 features** (no cascade).
   - `module2_detection/module2_train_models.py::train_track_b_dae`
     rewritten — drops `[X_benign \|\| probas]` augmentation; trains
     on the held-out benign val subset (`benign_only_val.parquet`,
     2141 rows).
   - `dae_final_report.json::architecture` = `"raw_25dim"`,
     `n_track_a_features` = `0`, `n_total_features` = `25`. Locked
     by `tests/test_track_a_xgb_only_v5.py::test_dae_artifact_reports_raw_25dim_architecture`.
   - `dae_thresholds.json` + `dae_calibration.json` regenerated via
     `module2_detection/build_dae_v4_artifacts.py` (now operates on
     25-dim error distribution).

2. **Production training defaults to XGBoost-only**.
   - `module2_train_models.py` adds `--include-baselines`
     (default off). Without the flag, only XGBoost trains; RF and
     DT pickles are not emitted. With the flag, all 3 are trained
     and the thesis Section 4 comparison reproduces.
   - `predict_demo()` skips models whose pipelines are absent.
   - `module2_detection/calibrate.py` already had skip-on-missing
     for individual models — no change needed.
   - `common/model_registry.get_track_a_classifiers` now returns
     whatever pipelines exist (XGB-only by default, all 3 with
     baselines).

3. **Layer 2 detector tolerates missing baselines**.
   - `module2_detection/layer2_detector.Layer2Detector.__init__`
     hard-requires only XGBoost; RF/DT load attempts log INFO and
     fall back to `p_rf = p_dt = p_xgb` for legacy
     `Layer2Output.diversity_score` field continuity.
   - `score_alert()` no longer constructs a 28-dim cascade — the
     DAE is called on raw 25-dim input directly.
   - `_compute_per_dim_thresholds()` returns 25 per-dim cutoffs
     (was 28).

4. **M3 cascade fusion → max(Track_A, Track_B-raw)**.
   - `compute_c_detect()` rewritten — drops the
     `_load_track_a_probas_for_dae` joblib parallel dispatch and
     the pre-allocated `(n, n_raw + 3)` cascade buffer. DAE runs
     on the sanitized raw matrix directly.
   - Helper `_load_track_a_probas_for_dae` and `_run_single_track_a`
     removed; `joblib` import removed.

5. **M4 explanations** on raw 25-dim DAE.
   - `module4_explanations/module4_explanations.py::compute_dae_feature_errors`
     called with raw `X_test` (no augmentation, no slice-back).
   - SHAP loops + analyst/clinician builders skip Track A models
     whose pickles are absent.

6. **Demo playlist** re-curated.
   - Beat 1 remapped: EVAL-0895 → EVAL-0570 (MEDIUM Spoofing on
     ventilator). Beats 2/3/5 + the synthetic Beat 4 unchanged.

### What got archived

- `module2_detection/retrain_dae_26dim.py` was the mid-Phase-B
  intermediate (26-dim cascade, drop RF+DT probas, keep P_xgb only).
  It was overtaken by the full 25-dim Phase B and lives at
  `docs/_archive/retrain_dae_26dim.py`. Re-introducing it would
  revert the documented architecture; locked by
  `tests/test_track_a_xgb_only_v5.py::test_phase_b_retrain_script_archived`.

### Verification

- `pytest tests/` — 516 passed, 1 skipped.
- `python run_tests.py` — `RECOMMENDATION: ✓ SHIP_TO_USER_STUDY`.
- M6 `test_false_positive_rate` = 77.4% (well above 40% target,
  no regression vs. pre-Phase-B 77.4%).
- XGBoost AUC unchanged at 0.9952 (re-fit identical).

### Known follow-ups (not blocking defense)

- `Layer2Output.diversity_score` is now `0.0` whenever RF/DT pickles
  are absent (computed as `std([p_xgb, p_xgb, p_xgb]) = 0`). This is
  a schema-only artefact — Module 3 ignores it post-Phase-A. A
  future cleanup could remove the field entirely from
  `Layer2Output`. Tracked, not urgent.
- `module2_detection/_features.py` and `module2_detection/calibrate.py`
  still iterate over `("xgboost", "random_forest", "decision_tree")`
  but skip-on-missing — cosmetically the loops could be tightened to
  the available set.
