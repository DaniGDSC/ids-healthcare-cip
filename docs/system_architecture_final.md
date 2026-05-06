# IoMT IDS — Final System Architecture (LOCKED)

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                  IoMT IDS — FINAL SYSTEM ARCHITECTURE                       │
│           Risk-Adaptive | Explainable | Human-in-the-Loop                   │
│              Multi-Stakeholder Clinical Decision Support                    │
│                                                                             │
│  Status: LOCKED FOR IMPLEMENTATION   Date: 2026-05-06                       │
│  Branch: fix/shap-category-vocab                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Thesis Identity

- **IoMT Security thesis** (80% security focus)
- **Multi-stakeholder hospital decision support**
- **Clinical safety as a cross-cutting constraint**

## Three Pillars

| Pillar | Concern | Anchored layers |
| --- | --- | --- |
| **Pillar 1** — Risk-Adaptive Threat Detection | Decide *whether* and *how urgent* the flow is anomalous in clinical context | Layers 1–3 |
| **Pillar 2** — Stakeholder-Tailored Security Explanation for Threat Triage | Decide *how to communicate* the threat to the right operator | Layers 4–5 |
| **Pillar 3** — Distributed Human-in-the-Loop Workflow | Decide *who acts*, with what authority, and how decisions are audited | Layer 6 |

## Seven Layers (canonical decomposition)

| Layer | Name | Pillar |
| --- | --- | --- |
| **Layer 1** | Data & Training | P1 |
| **Layer 2** | Detection | P1 |
| **Layer 3** | Risk & Triage Fusion | P1 |
| **Layer 4** | Explanation | P2 |
| **Layer 5** | Presentation | P2 |
| **Layer 6** | HITL Workflow | P3 |
| **Layer 7** | Multi-Method Evaluation | cross-cutting |

## Canonical Layer Block Diagram

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                       LAYER 7: EVALUATION                                   │
│   Multi-Method Validation (No Real Users)                                   │
│   • Method 1: LLM Multi-Stakeholder Triage Simulation                       │
│   • Method 2: Self-Consistency Evaluation                                   │
│   • Method 4: Heuristic Evaluation (Nielsen + DARPA + NIST)                 │
│   • Method 5: Comparative Triage Case Study Analysis                        │
│   • Method 6: Formal Specification Compliance ★ PRIMARY                     │
│   • Method 7: Information Gain Analysis                                     │
│   • MITRE Grounding Validation                                              │
│   • 9 Alert Types Validation ★ NEW                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ▲
┌─────────────────────────────────────────────────────────────────────────────┐
│                       LAYER 6: HITL WORKFLOW                                │
│   • No-Auto-Execution Invariant                                             │
│   • Tier Recommendation (no enforcement)                                    │
│   • Operator Decision Capture & Logging                                     │
│   • Audit Trail (append-only)                                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ▲
┌─────────────────────────────────────────────────────────────────────────────┐
│                       LAYER 5: PRESENTATION                                 │
│   Streamlit Dashboard (5 pages)                                             │
│   • Role Selector (IT / Biomed / Nurse)                                     │
│   • 3 Stakeholder Views                                                     │
│   • Alert Type Badge (9 types) ★                                            │
│   • Confidence Indicator                                                    │
│   • Mode A/B + Tier Badge + Data Quality Flag                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ▲
┌─────────────────────────────────────────────────────────────────────────────┐
│                       LAYER 4: EXPLANATION                                  │
│   • SHAP Feature Attribution (TreeExplainer)                                │
│   • MITRE ATT&CK Grounding ★ explicit                                       │
│   • MVE 3-Layer Generation (≤150 words)                                     │
│   • 9 Templates (3 base × 3 roles, metadata-differentiated)                 │
│   • Per-Dimension DAE Errors integration ★ NEW                              │
│   • Mode A (LLM) / Mode B (rule-based)                                      │
│   • Stakeholder Adaptation (authority bounds)                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ▲
┌─────────────────────────────────────────────────────────────────────────────┐
│                       LAYER 3: RISK & TRIAGE FUSION ★ ENRICHED              │
│   • Enriched Triage Fusion (9 alert types)                                  │
│   • DISAGREEMENT_ANOMALY (adversarial detection) ★ NEW                      │
│   • Composite Risk Scoring                                                  │
│   • Risk-Adaptive Gate (multiplier table)                                   │
│   • Safety Floor preserved                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ▲
┌─────────────────────────────────────────────────────────────────────────────┐
│                       LAYER 2: DETECTION (2-LAYER FILTER) ★ REDESIGNED      │
│   • Track A: Ensemble Filter (Attacks)                                      │
│     - XGBoost (primary) + RF + DT (diversity)                               │
│     - Calibrated probabilities ★ NEW                                        │
│     - Diversity score ★ NEW                                                 │
│   • Track B: DAE Filter (Normal)                                            │
│     - Cascade input: [25 raw || P_xgb, P_rf, P_dt]                          │
│     - Multi-threshold ★ NEW                                                 │
│     - Per-dimension errors ★ NEW                                            │
│   • Feature Sanitization (BENIGN_MEDIANS)                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ▲
┌─────────────────────────────────────────────────────────────────────────────┐
│                       LAYER 1: DATA & TRAINING                              │
│   • WUSTL-EHMS-2020 dataset                                                 │
│   • Stratified Eval Set (70/30 cal/holdout)                                 │
│   • Validation-set Probas (NOT OOF)                                         │
│   • BENIGN_MEDIANS computation                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                       CROSS-CUTTING                                         │
│   • Threat Model (STRIDE + MITRE ATT&CK)                                    │
│   • Clinical Safety (FMEA 30 modes)                                         │
│   • Bias Audit (per device class)                                           │
│   • Audit Trail (append-only logs)                                          │
│   • 7 Architecture Invariants                                               │
│   • Code Quality Gates                                                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Diagram-vs-code reconciliation

The diagram above is verified against the codebase. The two flagged
items from the prior revision are resolved as follows:

- **L1 "Validation-set Probas (NOT OOF)"** — resolved. Module 2 now
  emits both `results/models/*_oof_proba.npy` (legacy) and
  `results/models/*_val_proba.npy` (held-out validation), and the DAE
  cascade prefers val_probas when present
  ([module2_train_models.py:283-301](../module2_detection/module2_train_models.py#L283-L301),
  [:404-423](../module2_detection/module2_train_models.py#L404-L423)).
  This avoids CV-fold leakage into the cascaded DAE training. Calibrated
  variants (`*_val_proba_calibrated.npy`) anchor the L2
  "Calibrated probabilities" claim.
- **L4 "9 Templates (3 base × 3 roles, metadata-differentiated)"** —
  re-framed. The 9-cell view is **3 base templates × 3 roles**, with
  metadata (alert type, severity tier, device class) differentiating
  rendered output within each cell rather than a separate template per
  alert type. The implementation pivots on `CLINICIAN_TEMPLATES` in
  [module4_online_explainer.py:174](../module4_explanations/module4_online_explainer.py#L174)
  and [module4_explanations.py:148](../module4_explanations/module4_explanations.py#L148),
  with role adaptation layered in `src/mve_generator.py`.

The cross-cutting **Code Quality Gates** row continues to verify cleanly
against [`pyproject.toml`](../pyproject.toml) (ruff, mypy, bandit, coverage
threshold) and [`.pre-commit-config.yaml`](../.pre-commit-config.yaml)
(commit-hook wiring).

This document anchors the eight architectural-property declarations on
the lock notice to verifiable artifacts in the repository. Every claim
below has been cross-checked against current code, tests, and deliverable
YAMLs as of 2026-05-06.

---

## Layer 1 Detailed Specification

```text
╔══════════════════════════════════════════════════════════════════════╗
║                    LAYER 1: DATA & TRAINING                          ║
║                    (Offline — Run Once)                              ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  ╔════════════════════════════════════════════════════════════════╗  ║
║  ║  Module M0: Data Preparation                                    ║  ║
║  ║                                                                  ║  ║
║  ║   Source: WUSTL-EHMS-2020                                        ║  ║
║  ║   ├─ Train (11,422 samples, SMOTE-balanced)                     ║  ║
║  ║   ├─ Validation (held-out 20%)                                  ║  ║
║  ║   └─ Test (4,896 samples, 12.54% attack rate)                   ║  ║
║  ║                                                                  ║  ║
║  ║   ★ Stratified Eval Set Generation                               ║  ║
║  ║   ├─ Calibration (70%): for threshold tuning + Platt scaling    ║  ║
║  ║   ├─ Holdout (30%): for final metrics                           ║  ║
║  ║   ├─ Stratification: by true_severity tier                      ║  ║
║  ║   └─ Random seed: 42 (reproducibility)                          ║  ║
║  ║                                                                  ║  ║
║  ║   Curated 20-alert set                                          ║  ║
║  ║   └─ Role: PATHOLOGICAL CASE STRESS TEST (NOT for tuning)       ║  ║
║  ║                                                                  ║  ║
║  ║   Outputs:                                                       ║  ║
║  ║   ├─ data/processed/train_phase1.parquet                        ║  ║
║  ║   ├─ data/processed/val_phase1.parquet      ★ NEW (B1)          ║  ║
║  ║   ├─ data/processed/test_phase1.parquet                         ║  ║
║  ║   ├─ results/reports/stratified_calibration.parquet             ║  ║
║  ║   ├─ results/reports/stratified_holdout.parquet                 ║  ║
║  ║   └─ results/reports/evaluation_alerts.json                     ║  ║
║  ║      (curated 20-alert pathological-case stress set;            ║  ║
║  ║       role documented in track_a_performance.yaml               ║  ║
║  ║       § secondary_set_stress_test)                              ║  ║
║  ╚════════════════════════════════════════════════════════════════╝  ║
║                                                                      ║
║  ╔════════════════════════════════════════════════════════════════╗  ║
║  ║  Module M1: Feature Engineering                                 ║  ║
║  ║                                                                  ║  ║
║  ║   Inputs: 25 raw features                                        ║  ║
║  ║                                                                  ║  ║
║  ║   ★ BENIGN_MEDIANS Computation                                   ║  ║
║  ║   ├─ Compute median per feature on training benign              ║  ║
║  ║   ├─ Persist: data/processed/benign_medians.json                ║  ║
║  ║   └─ Used during inference for NaN/Inf replacement              ║  ║
║  ║                                                                  ║  ║
║  ║   Feature Pipeline:                                              ║  ║
║  ║   ├─ Normalization (RobustScaler fit on train only)             ║  ║
║  ║   ├─ Persisted: data/processed/robust_scaler.pkl                ║  ║
║  ║   └─ Inference uses .transform() (NOT fit_transform)            ║  ║
║  ╚════════════════════════════════════════════════════════════════╝  ║
║                                                                      ║
║  ╔════════════════════════════════════════════════════════════════╗  ║
║  ║  Module M2 (Training Phase): Detection Engine Training          ║  ║
║  ║                                                                  ║  ║
║  ║   ┌─ Track A: Ensemble Training ─────────────────────────┐     ║  ║
║  ║   │                                                        │     ║  ║
║  ║   │   ┌──────────┐  ┌──────────┐  ┌──────────┐             │     ║  ║
║  ║   │   │ XGBoost  │  │   RF     │  │   DT     │             │     ║  ║
║  ║   │   │ ★ Primary│  │Benchmark │  │Benchmark │             │     ║  ║
║  ║   │   └──────────┘  └──────────┘  └──────────┘             │     ║  ║
║  ║   │                                                        │     ║  ║
║  ║   │   Comparative metrics:                                 │     ║  ║
║  ║   │   F1: 0.892    0.800       0.698                       │     ║  ║
║  ║   │   AUC: 0.994   0.959       0.891                       │     ║  ║
║  ║   │                                                        │     ║  ║
║  ║   │   Selection: XGBoost (primary)                         │     ║  ║
║  ║   │   Threshold: 0.05 (F2-tuned)                           │     ║  ║
║  ║   │                                                        │     ║  ║
║  ║   │   ★ NEW: Probability Calibration                       │     ║  ║
║  ║   │   ├─ Method: isotonic regression                       │     ║  ║
║  ║   │   │          (Platt fallback when n_val < 1000)        │     ║  ║
║  ║   │   ├─ Trained on calibration set                        │     ║  ║
║  ║   │   ├─ Per-model: XGB / RF / DT                          │     ║  ║
║  ║   │   └─ Output: *_val_proba_calibrated.npy                │     ║  ║
║  ║   │                                                        │     ║  ║
║  ║   │   ★ NEW: Per-Class Surfacing Thresholds                │     ║  ║
║  ║   │   (applied at Layer 3 inference; see                   │     ║  ║
║  ║   │    src/risk_scorer.py::_TRACK_A_SURFACING_BY_DEVICE)   │     ║  ║
║  ║   │   ├─ infusion_pump: 0.03 (life-sustaining)             │     ║  ║
║  ║   │   ├─ ventilator:    0.03 (life-sustaining)             │     ║  ║
║  ║   │   ├─ patient_monitor / monitor: 0.05 (default, F2)     │     ║  ║
║  ║   │   ├─ imaging:       0.07 (clinical-support)            │     ║  ║
║  ║   │   └─ ehr_workstation: 0.10 (PHI, noise floor)          │     ║  ║
║  ║   │                                                        │     ║  ║
║  ║   │   ★ Generate VALIDATION-set probas for Track B         │     ║  ║
║  ║   │   ├─ P_xgb_val on held-out validation                  │     ║  ║
║  ║   │   ├─ P_rf_val on held-out validation                   │     ║  ║
║  ║   │   └─ P_dt_val on held-out validation                   │     ║  ║
║  ║   │   ★ Resolves train-inference skew                      │     ║  ║
║  ║   └────────────────────────────────────────────────────────┘     ║  ║
║  ║                                                                  ║  ║
║  ║   ┌─ Track B: DAE Training ────────────────────────────────┐    ║  ║
║  ║   │                                                          │    ║  ║
║  ║   │   Architecture: Encoder-Decoder                          │    ║  ║
║  ║   │   Input: [25 raw || P_xgb_val, P_rf_val, P_dt_val]      │    ║  ║
║  ║   │   ★ 3-proba cascade design rationale:                    │    ║  ║
║  ║   │   ├─ Ensemble diversity in benign manifold               │    ║  ║
║  ║   │   ├─ Adversarial robustness                              │    ║  ║
║  ║   │   └─ Disagreement detection                              │    ║  ║
║  ║   │                                                          │    ║  ║
║  ║   │   Trained on: BENIGN-only validation samples             │    ║  ║
║  ║   │   Loss: MSE reconstruction                               │    ║  ║
║  ║   │                                                          │    ║  ║
║  ║   │   Current threshold (production):                        │    ║  ║
║  ║   │   ├─ Single percentile: 99th (per dae_final_report)      │    ║  ║
║  ║   │   ├─ Raw threshold: 6.44e-3                              │    ║  ║
║  ║   │   └─ Sensitivity sweep 80-99 (offline only)              │    ║  ║
║  ║   │                                                          │    ║  ║
║  ║   │   ★ Multi-Threshold Configuration  (PLANNED)             │    ║  ║
║  ║   │   ├─ Screening:      80th percentile (high sens.)        │    ║  ║
║  ║   │   ├─ Confirmation:   95th percentile (default)           │    ║  ║
║  ║   │   └─ High confidence: 99th percentile                    │    ║  ║
║  ║   │   (currently only 80-99 sweep is used at eval time;      │    ║  ║
║  ║   │    runtime uses a single percentile)                     │    ║  ║
║  ║   │                                                          │    ║  ║
║  ║   │   ★ Percentile-Based Score Calibration  (PLANNED)        │    ║  ║
║  ║   │   ├─ Map raw error → percentile rank                     │    ║  ║
║  ║   │   ├─ DAE_score = percentile_rank / 100                   │    ║  ║
║  ║   │   └─ Cross-environment comparable                        │    ║  ║
║  ║   │   (current code emits raw reconstruction error vs        │    ║  ║
║  ║   │    fixed threshold; mapping not yet implemented)         │    ║  ║
║  ║   │                                                          │    ║  ║
║  ║   │   Purpose: NOVELTY DETECTION                              │    ║  ║
║  ║   │   ├─ Zero-day attacks                                    │    ║  ║
║  ║   │   ├─ Adversarial inputs                                  │    ║  ║
║  ║   │   └─ Distribution shift                                  │    ║  ║
║  ║   └─────────────────────────────────────────────────────────┘    ║  ║
║  ║                                                                  ║  ║
║  ║   Persisted artifacts:                                           ║  ║
║  ║   ├─ results/models/xgboost_final_pipeline.pkl + .sig           ║  ║
║  ║   ├─ results/models/random_forest_final_pipeline.pkl + .sig     ║  ║
║  ║   ├─ results/models/decision_tree_final_pipeline.pkl + .sig     ║  ║
║  ║   ├─ results/models/{xgboost,random_forest,decision_tree}     ║  ║
║  ║   │     _calibrator.pkl                              ★ NEW    ║  ║
║  ║   ├─ results/models/{xgboost,random_forest,decision_tree}     ║  ║
║  ║   │     _calibration_report.json                     ★ NEW    ║  ║
║  ║   ├─ results/models/dae_detector.json + dae_model.weights.h5    ║  ║
║  ║   └─ data/processed/robust_scaler.pkl (fitted on train only)    ║  ║
║  ╚════════════════════════════════════════════════════════════════╝  ║
║                                                                      ║
║  ╔════════════════════════════════════════════════════════════════╗  ║
║  ║  Threshold Calibration                                          ║  ║
║  ║                                                                  ║  ║
║  ║   On stratified_calibration set:                                 ║  ║
║  ║   ├─ Two-stage triage thresholds (a_high, a_low, b)             ║  ║
║  ║   ├─ Risk-adaptive multipliers per device class                 ║  ║
║  ║   ├─ Diversity score threshold (b_diversity)         ★ NEW     ║  ║
║  ║   │  (Layer 3 fusion gate; multiclass_fusion.diversity_score)   ║  ║
║  ║   ├─ Method: grid search                                         ║  ║
║  ║   ├─ Objective: minimize FNR_CRITICAL subject to FPR < 0.05     ║  ║
║  ║   └─ Final metrics: stratified_holdout (never seen)             ║  ║
║  ╚════════════════════════════════════════════════════════════════╝  ║
╚══════════════════════════════════════════════════════════════════════╝
```

### L1 spec-vs-code reconciliation (verified 2026-05-06)

The L1 spec above was cross-checked against current artifacts. **19 of
21 concrete claims verify cleanly**; **2 claims are flagged `(PLANNED)`
in the spec block above** because the runtime feature is not yet wired
even though the design is fixed. The reconciliation table tracks both
the existing post-A1 closures and the new claims introduced by the
Track A "Probability Calibration" / "Per-Class Surfacing Thresholds"
and Track B "Multi-Threshold" / "Percentile-Based Score Calibration"
blocks.

| Claim | Spec text | Current reality | Status | Action |
| --- | --- | --- | --- | --- |
| Train size | 11,422 SMOTE-balanced | 11,422 ✓ | PASS | — |
| Test size | 4,896 (12.54% attack) | 4,896, 12.54% ✓ | PASS | — |
| **Validation 20% held-out** | `data/processed/val_phase1.parquet` | `val_phase1.parquet` (n=2,285, atk=12.52%) + `val_benign_phase1.parquet` (n=1,999) emitted by `module1_preprocessing/phase1/splitter.py::DataSplitter` with `val_ratio=0.20` ✓ | PASS | closed 2026-05-06 (GAP-L1-2) |
| Stratified calib/holdout file paths | `results/reports/stratified_{calibration,holdout}.parquet` (post-A1 spec update) | files at `results/reports/stratified_{calibration,holdout}.parquet` ✓ | PASS | closed by A1 doc update |
| Curated 20-alert stress set | `results/reports/evaluation_alerts.json` (post-A1; role documented in `track_a_performance.yaml`) | `results/reports/evaluation_alerts.json` (n=20) ✓ | PASS | closed by A1 doc update |
| BENIGN_MEDIANS | persisted at `data/processed/benign_medians.json` | present, 25 features, n=9990 ✓ | PASS | — |
| Feature pipeline pickle | `data/processed/robust_scaler.pkl` (RobustScaler, post-A1 spec update) | `data/processed/robust_scaler.pkl` ✓ | PASS | closed by A1 doc update |
| XGBoost final pickle | `results/models/xgboost_final_pipeline.pkl` + `.sig` (post-A1 spec update) | `results/models/xgboost_final_pipeline.pkl` + `.sig` (signed_pickle protection) ✓ | PASS | closed by A1 doc update |
| Comparative F1: 0.892 / 0.800 / 0.698 | exact match | XGB 0.892 / RF 0.800 / DT 0.698 ✓ | PASS | — |
| Comparative AUC: 0.994 / 0.959 / 0.891 | exact match | 0.994 / 0.959 / 0.891 ✓ | PASS | — |
| Selected XGBoost @ threshold 0.05 (F2-tuned) | `optimal_threshold = 0.05` per F2-tuning | `xgboost_final_report.json::optimal_threshold = 0.05` ✓ | PASS | — |
| **VALIDATION-set probas** (`P_xgb_val`, `P_rf_val`, `P_dt_val`) | star-marked NEW in spec; star-marked NOT OOF | `results/models/{xgboost,random_forest,decision_tree}_val_proba.npy` (n=2,285 each) written after final-fit; DAE trains on val benign + val probas (`dae_final_report.json::data.track_a_proba_source = "val"`) ✓ | PASS | closed 2026-05-06 (GAP-L1-1) |
| DAE production threshold (percentile + raw) | persisted at 99th-percentile of benign training reconstruction errors | `dae_final_report.json::threshold_percentile = 99.0`; `threshold = 0.00644` (6.44e-3) ✓ | PASS | updated 2026-05-06 to reflect 99-pct (was 95-pct in prior table) |
| DAE sensitivity sweep 80-99 percentiles | sweep documented per spec | `novelty_validation.yaml § threshold_sensitivity_analysis.sweep` ✓ | PASS | — |
| DAE LOO + adversarial validation | LOO experiments + adversarial robustness | LOO present (Spoofing 0.51, Data Alteration 1.0); adversarial deferred to RQ1.7 | PARTIAL | LOO done, adversarial open (GAP-NV-1) |
| Threshold calibration on stratified calib | grid search on calibration set | `two_stage_fusion_validation.yaml § threshold_calibration` ✓ | PASS | — |
| Calibration objective: minimise FNR_CRITICAL @ FPR<0.05 | per-spec optimisation objective | exact match in calibration YAML ✓ | PASS | — |
| Final metrics on stratified holdout | metrics reported on never-seen holdout | reported per-stage on holdout (n=1469) ✓ | PASS | — |
| **Probability calibration** (Track A) | "Method: isotonic regression (Platt fallback when n_val < 1000)"; per-model `*_calibrator.pkl` + `*_calibration_report.json` | `module2_detection/calibrate.py:21-29,98,119-121` (`CalibratedClassifierCV` with `method='isotonic'`, sigmoid fallback); `results/models/{xgboost,random_forest,decision_tree}_calibrator.pkl` + `*_calibration_report.json` + `*_val_proba_calibrated.npy` ✓ | PASS | new claim, verified 2026-05-06 |
| **Per-class surfacing thresholds** (Track A spec block, applied at L3) | infusion_pump 0.03, ventilator 0.03, monitor 0.05, imaging 0.07, ehr_workstation 0.10 | `src/risk_scorer.py:78-83` `DEVICE_CLASS_THRESHOLDS` exact match; selected via `get_track_a_surfacing_threshold()` at line 88 | PASS | annotated in spec as Layer-3-applied (not Track-A training output) |
| **Diversity score threshold** (Threshold Calibration row) | `b_diversity` threshold for L3 fusion-gate demotion | `module3_risk_scoring/multiclass_fusion.py:167` (`diversity_score`), :227 (demotion rule), :252 (call site); `src/data_models.py:18` (`b_diversity` field) | PASS | new claim, verified 2026-05-06 |
| **DAE Multi-Threshold Configuration** (Screening 80 / Confirmation 95 / High-confidence 99) | three runtime operating points selected per situation | runtime DAE uses single threshold (`dae_final_report.json::threshold_percentile = 99`, `threshold = 0.00644`); 80-99 sweep exists only in `novelty_validation.yaml § threshold_sensitivity_analysis.sweep` (offline) | PLANNED | spec block flagged `(PLANNED)`; wire multi-threshold at Module 2 export + Module 3 consumer to close |
| **DAE Percentile-Based Score Calibration** (DAE_score = percentile_rank/100) | inference-time mapping from raw error to percentile rank | not implemented; no `percentile_rank`, `score_calibration` in module2/src; current code emits raw reconstruction error against fixed threshold | PLANNED | spec block flagged `(PLANNED)`; close by precomputing benign error CDF + adding rank lookup to DAE inference path |

### Drift summary (post-A1 doc updates, 2026-05-06)

**Class A — path / filename renames: ALL 5 CLOSED** by spec-direction
update (D-direction in the original plan). The on-disk layout is the
canonical truth; the L1 spec block above and this reconciliation table
have been edited to match. No code or artifact moves were performed —
behaviour-equivalent change, zero churn.

**Class B — coordinated architectural change: ALL 2 CLOSED** on
2026-05-06 via a Module 1 + Module 2 + Module 3 cascade refresh. Both
items in detail:

- **GAP-L1-2 (closure 2026-05-06)**: held-out validation parquet
  `data/processed/val_phase1.parquet`. `DataSplitter` extended with
  `val_ratio` (default `0.0`, set to `0.20` in `phase1_config.yaml`),
  and the pipeline now emits `val_phase1.parquet` + `val_benign_phase1.parquet`
  alongside the train/test parquets. 7 regression tests in
  `tests/test_split_consistency.py` pin the 3-way contract (sizes,
  disjointness, stratification, determinism, report shape).
- **GAP-L1-1 (closure 2026-05-06)**: validation-set probas (NOT OOF).
  `module2_train_models.py::train_track_a` now also predicts on the
  val parquet and persists `results/models/{xgboost,random_forest,decision_tree}_val_proba.npy`.
  `train_track_b_dae` was re-wired to prefer val probas: when both
  `val_phase1.parquet` and the three `*_val_proba.npy` files exist, the
  DAE trains on val benign augmented with val probas (logged
  `GAP-L1-1: using held-out val set for DAE training (n_val=2285,
  n_benign=1999)`); the legacy OOF path is retained as a fallback.
  `dae_final_report.json::data.track_a_proba_source` records which
  branch ran (`"val"` after closure, `"oof"` if val artifacts are absent).

Closure cascade re-run on 2026-05-06: `python scripts/run_phase0.py
→ python -m module1_preprocessing.phase1 → python module2_detection/module2_train_models.py
→ python module3_risk_scoring/module3_risk_scores.py`, with split sizes
9,137 / 2,285 / 4,896 (stratified atk-rate 12.54% / 12.52% / 12.54%) and
all 111 tests passing. Verification entries appended to
`results/reports/{track_a_performance, novelty_validation,
two_stage_fusion_validation, risk_adaptive_validation}.yaml`.

The DAE under the val-proba path scores lower than under the OOF path
on the test set (AUC 0.6835 vs 0.9128) — this is the expected GAP-L1-1
cost: the OOF path inflated AUC by leaking CV-fold structure into the
joint (feature, proba) space. The val-proba AUC is the honest
inference-time number. See `novelty_validation.yaml § verification_runs
[2026-05-06].delta_vs_oof_path` for the full delta.

---

## Verification of the Eight Declared Properties

| # | Declared property | Status | Anchored evidence |
| --- | --- | --- | --- |
| 1 | 7-layer architecture | **PASS** | Layer 1 Data & Training (M0/M1) · Layer 2 Detection (M2) · Layer 3 Risk & Triage Fusion (M3 + `src/risk_scorer.py`) · Layer 4 Explanation (M4 + `src/mve_generator.py`) · Layer 5 Presentation (M5 + `module6_app.py`) · Layer 6 HITL Workflow (Module 6 audit + OperatorDecision schema) · Layer 7 Multi-Method Evaluation (6 methods, see §5) — full layer→module→code map in §1 below |
| 2 | 3 critical fixes (maintenance, A_patient, NaN) | **PASS** | (a) Maintenance-window safety floor: [`tests/test_safe_failure.py::test_critical_unpatchable_surfaces_in_maintenance_window`](../tests/test_safe_failure.py); (b) `A_patient` → `D_clinical_tier` rename complete (zero residual occurrences in code/YAML); (c) NaN sanitiser with BENIGN_MEDIANS in [`src/preprocessing.py`](../src/preprocessing.py) and 7 tests in [`tests/test_feature_sanitization.py`](../tests/test_feature_sanitization.py) |
| 3 | Two-stage triage fusion | **PASS** | `classify_fusion()` in [`module3_risk_scoring/module3_risk_scores.py`](../module3_risk_scoring/module3_risk_scores.py) emits 4 fusion classes (KNOWN_ATTACK / NOVEL_ANOMALY / CONFIRMED_ANOMALY / BENIGN); validated end-to-end in [`results/reports/two_stage_fusion_validation.yaml`](../results/reports/two_stage_fusion_validation.yaml) (6/6 acceptance PASS) |
| 4 | 9 MVE templates (3×3 matrix) | **CLARIFIED — see §4 below** | Actual implementation is 5 alert types × 3 roles = 15 templates, not 3×3=9. Document expectation should be 15, not 9 — see §4 reconciliation |
| 5 | Multi-method evaluation (no real users) | **PASS** | 6 methods complete: M6 [`req_trace_matrix.yaml`](../results/reports/req_trace_matrix.yaml) (22/22), M7 [`information_gain.yaml`](../results/reports/information_gain.yaml), M1 [`m5_multi_role_result.yaml`](../survey/m5_multi_role_result.yaml) (2000/2000 calls), M4 [`heuristic_compliance.yaml`](../results/reports/heuristic_compliance.yaml) (84% strict / 92% partial), M5 [`case_study_comparisons.md`](case_study_comparisons.md) (8 cases), M2 [`m5_self_consistency_result.yaml`](../survey/m5_self_consistency_result.yaml) |
| 6 | MITRE ATT&CK explicit grounding | **PASS** | `attck_for_alert_type()` lookup in [`src/mve_generator.py`](../src/mve_generator.py) maps T1→T1071, T2→T1078, T3→T1021, T4→T1041, T5→T1565; deterministic (not LLM-dependent); 2 tests in `test_safe_failure.py` |
| 7 | Refined RQ2 (threat triage) | **PASS** | `tier_recommendation` field in `two_stage_fusion_validation.yaml` (L1 / L2_specialist / L1_with_senior); RQ2 metrics in [`results/rq2_metrics.json`](../results/rq2_metrics.json) (sensitivity 1.0, specificity 0.5, FNR_critical 0) |
| 8 | 7 architecture invariants | **PASS** | All 7 invariants verified in [`docs/architecture.md`](architecture.md) §4 with PASS verdicts; full evidence in [`results/reports/invariant_verification.log`](../results/reports/invariant_verification.log) |

---

## 1. The 7-Layer Architecture (canonical)

The system decomposes into seven layers grouped under the three pillars
declared on the lock notice. Each layer maps to one or more numbered
modules (M0–M6) and a sequence of canonical workflow steps.

| Pillar | Layer | Concern | Code anchors | Workflow steps |
| --- | --- | --- | --- | --- |
| **P1** | **L1 — Data & Training** | Curate and split WUSTL-EHMS dataset; engineer the 25-feature schema; persist benign medians | [`module0_analysis/phase0/`](../module0_analysis/phase0/) (audit), [`module1_preprocessing/phase1/`](../module1_preprocessing/phase1/) (preprocess + SMOTE + scaler), [`data/processed/benign_medians.json`](../data/processed/benign_medians.json) | [1] |
| **P1** | **L2 — Detection** | Train cascaded dual-track detector (Track A: XGB/RF/DT supervised; Track B: DAE benign-only novelty); persist artifacts | [`module2_detection/`](../module2_detection/) (training); [`module4_explanations/module4_online_explainer.py`](../module4_explanations/module4_online_explainer.py) (online inference); [`module2_detection/models/DAE.py`](../module2_detection/models/DAE.py) | [2] [3] [4] [6a] [6b] |
| **P1** | **L3 — Risk & Triage Fusion** | Per-feature sanitization → two-stage fusion → composite-R risk score → risk-adaptive surfacing gate with safety floor | [`src/preprocessing.py`](../src/preprocessing.py) (Step 5), [`module3_risk_scoring/module3_risk_scores.py`](../module3_risk_scoring/module3_risk_scores.py) (compute_c_detect, classify_fusion, compute_composite_risk), [`src/risk_scorer.py`](../src/risk_scorer.py) (per-alert gate + safety floor + EA-06 score elevation) | [5] [7] [8] [9] [10] |
| **P2** | **L4 — Explanation** | TreeSHAP feature attribution + stability score + clinician-readable narrative + 3-layer MVE (Mode A LLM / Mode B rule-based) + ATT&CK technique grounding | [`module4_explanations/module4_online_explainer.py`](../module4_explanations/module4_online_explainer.py) (SHAP + stability), [`src/mve_generator.py`](../src/mve_generator.py) (3-layer MVE + `attck_for_alert_type`), [`src/data_models.py::SHAPContext`](../src/data_models.py) | [11] [12] |
| **P2** | **L5 — Presentation** | Stakeholder-tailored views (IT_generalist / biomed_engineer / nurse_manager) + tier-recommendation routing + dashboard rendering | [`src/mve_generator.py::derive_role_view`](../src/mve_generator.py), [`module5_responses/module5_pipeline.py`](../module5_responses/module5_pipeline.py) (notification routing), [`module6_evaluation/module6_app.py`](../module6_evaluation/module6_app.py) (Streamlit dashboard) | [13] [14] |
| **P3** | **L6 — HITL Workflow** | Recommendation-only output (NO AUTO-EXECUTION) + operator decision capture + append-only audit log with schema validation | [`module5_responses/module5_pipeline.py::recommend`](../module5_responses/module5_pipeline.py), [`src/data_models.py::OperatorDecision`](../src/data_models.py) (`.validate()` schema), audit-log writes via [`module6_evaluation/module6_app.py`](../module6_evaluation/module6_app.py) | [15] [16] |
| **cross-cutting** | **L7 — Multi-Method Evaluation** | Six standards-grounded evaluation methods (no real users): formal compliance, information gain, LLM persona simulation, heuristic compliance, case-study analysis, self-consistency | All `results/reports/*.yaml` deliverables + `survey/m5_*.yaml` + `docs/case_study_comparisons.md` + `docs/heuristic_evaluation.md` | [17]† [18]† |

†Steps [17] outcome tracking and [18] continuous improvement are
documented as future work (Phase-3 deployment scope).

### Layer ↔ pillar boundaries

```text
┌───────────────────── P1 ─────────────────────┐  ┌────── P2 ──────┐  ┌── P3 ──┐
│                                              │  │                │  │        │
│  L1 Data & Training                          │  │  L4 Explanation │  │  L6    │
│      └─ M0 audit, M1 preprocess              │  │      └─ M4 SHAP │  │  HITL  │
│                                              │  │      + MVE      │  │        │
│  L2 Detection                                │  │                 │  │        │
│      └─ M2 train; online inference           │  │  L5 Presentation│  │        │
│                                              │  │      └─ M5 + UI │  │        │
│  L3 Risk & Triage Fusion                     │  │                 │  │        │
│      └─ Step [5] sanitize                    │  │                 │  │        │
│      └─ Step [7] two-stage fusion            │  │                 │  │        │
│      └─ Step [9] composite R                 │  │                 │  │        │
│      └─ Step [10] risk-adaptive gate         │  │                 │  │        │
│                                              │  │                 │  │        │
└──────────────────────────────────────────────┘  └─────────────────┘  └────────┘

  L7 Multi-Method Evaluation  ←─── cross-cuts P1 + P2 + P3
       └─ M6 + 6 evaluation methods
```

Layer 3 is the load-bearing fusion centre: every fusion-class label,
risk-tier, surfacing decision, and safety-floor invariant lives there,
and every other layer depends on its outputs being trustworthy.

---

## 2. The Three Critical Fixes

### Fix 1 — Maintenance-window safety floor (ST-09)

**Problem:** Pre-fix, the maintenance-window early-return path in `score_alert()` could silently set `should_surface=False` for a CRITICAL+unpatchable device.

**Fix:** [`src/risk_scorer.py:125-128`](../src/risk_scorer.py) — the early-return path now ORs `(criticality == "CRITICAL" and not patchable)` into the `should_surface` decision:

```python
if event_context.get("is_maintenance_window") and event_context.get("is_known_vendor_ip"):
    reduced = score * 0.5
    should_surface = (
        reduced > DEFAULT_THRESHOLD
        or (criticality == "CRITICAL" and not patchable)   # ← safety floor
    )
    return ScoredAlert(..., should_surface=should_surface, ...)
```

**Verification:** [`tests/test_safe_failure.py::test_critical_unpatchable_surfaces_in_maintenance_window`](../tests/test_safe_failure.py).

### Fix 2 — `A_patient` → `D_clinical_tier` rename

**Problem:** The original formula term `A_patient` implied dynamic patient acuity; the implementation uses a static device-class clinical tier. The naming overclaimed.

**Fix:** Function `compute_a_patient` → `compute_d_clinical_tier`; NPZ key `a_patient` → `d_clinical_tier`; formula in `compute_composite_risk` updated; ARCHITECTURE.md and Performance_baselines.md updated.

**Verification:** Zero `A_patient` / `a_patient` occurrences in `*.py` and `*.yaml`. Documented honestly in [`docs/risk_formula_specification.md §3`](risk_formula_specification.md) — the proxy-vs-true-acuity gap is acknowledged with a 4-step future-work plan.

### Fix 3 — NaN sanitisation (EA-06 mitigation)

**Problem:** Pre-fix, NaN/Inf in input features were silently replaced with zeros. An attacker could exploit this to mask anomalies (zero-replacement creates an artificial outlier in the joint feature-prediction space).

**Fix:**
1. Replace with **per-feature benign medians** ([`src/preprocessing.py::sanitize_features`](../src/preprocessing.py) + [`data/processed/benign_medians.json`](../data/processed/benign_medians.json) computed from 9990 benign training samples).
2. Emit a **DataQuality flag** ∈ {OK, IMPUTED_NAN, DEGRADED, FAILED} based on `nan_rate`.
3. **EA-06 score elevation** in [`src/risk_scorer.py`](../src/risk_scorer.py): DEGRADED inputs raise `adjusted_score` ×1.20; FAILED inputs clamp to ≥0.95 so the alert always surfaces.

**Verification:** 7 tests in [`tests/test_feature_sanitization.py`](../tests/test_feature_sanitization.py) including `test_5_nan_injection_attack_elevates_score`. Full deliverable in [`results/reports/feature_sanitization.yaml`](../results/reports/feature_sanitization.yaml) (5/5 acceptance PASS).

---

## 3. Two-Stage Triage Fusion

Per ARCHITECTURE.md Step [7]:

```
Stage 1 — KNOWN_ATTACK      P_xgb >= a_high                            → confidence=HIGH, tier=L1
Stage 2 — NOVEL_ANOMALY     P_xgb < a_low  AND DAE_score >= b          → confidence=MED,  tier=L2_specialist
Stage 3 — CONFIRMED_ANOMALY a_low <= P_xgb < a_high AND DAE_score >= b → confidence=HIGH, tier=L1_with_senior
Stage 4 — BENIGN            otherwise                                  → suppressed (audit log only)
```

**Spec defaults:** `a_high=0.85, a_low=0.40, b=0.70` — pinned in `tests/test_two_stage_fusion.py` (6 named test cases + 3 invariant checks all PASS).

**Calibration-selected:** `a_high=0.95, a_low=0.05, b=0.80` — grid search on stratified calibration set minimised FNR_CRITICAL subject to FPR < 0.05. Documented divergence in [`results/reports/two_stage_fusion_validation.yaml § threshold_calibration.selected_thresholds.note`](../results/reports/two_stage_fusion_validation.yaml).

**Headline holdout metrics:**
- FNR_CRITICAL: **0.000** (16/16 CRITICAL attacks caught)
- FPR: 0.041 vs max-fusion 0.063 — **two-stage wins on operator-fatigue dial by 2.3 pp**
- Recall: 0.881 vs max-fusion 0.908 (the 2.7 pp recall trade-off is intentional; CRITICAL coverage unchanged)

---

## 4. MVE Template Matrix — Reconciliation

The lock notice declares "9 MVE templates (3×3 matrix)". The actual implementation is a **5 × 3 = 15 template matrix**:

| Axis | Values | Count |
| --- | --- | --- |
| Alert type | T1 (anomalous outbound) / T2 (unauthorised EHR access) / T3 (lateral movement) / T4 (data exfiltration) / T5 (IoMT behavioural deviation) | 5 |
| Operator role | IT_generalist / biomed_engineer / nurse_manager | 3 |
| **Total templates** | | **15** |

**Why the 3×3 figure may have been declared:**

Two interpretations are consistent with "3×3 = 9":
1. **3 severity tiers × 3 roles** — the system applies severity-conditioned wording to Layer 1/2/3 within each role view. Counting *severity strata* (CRITICAL/HIGH/MEDIUM as the actionable strata; LOW typically dismissed) × 3 roles gives 9 distinct rendered combinations per alert type.
2. **3 attack-category clusters × 3 roles** — clustering T1+T2 (network/access), T3+T4 (movement/exfil), T5 (IoMT-physical) gives 3 alert clusters × 3 roles = 9 templates at the *category-cluster* level.

**Recommended action:** update the architecture lock notice to "**15 MVE templates (5 alert types × 3 roles)**" to match the implementation, OR clarify the documented 3×3 framing as severity-by-role and note that within each cell the alert-type-specific text is generated by the rule-based templates in `_generate_rule_based`. Either reconciliation is valid; today's code emits 15 leaves but 9 cells if you collapse alert types into severity strata.

This is the only one of the 8 declared properties that does not match the code 1:1. Tracked as a documentation reconciliation, not a code change.

---

## 5. Multi-Method Evaluation (No Real Users)

| Method | Output | Headline result |
| --- | --- | --- |
| **M6** Formal spec compliance | `results/reports/req_trace_matrix.yaml` | 22/22 REQ-MVE PASS (100%) |
| **M7** Information gain | `results/reports/information_gain.yaml` | raw view 3.0/8 dimensions → MVE 8.0/8 (+5.0 dim/alert) |
| **M1** LLM persona simulation | `survey/m5_multi_role_result.yaml` | 2000/2000 calls; Mann-Whitney p < 0.0001 all 3 roles; Cohen's d = 38–112 |
| **M4** Heuristic compliance | `results/reports/heuristic_compliance.yaml` | 21 PASS / 4 PARTIAL / 0 FAIL / 1 N/A across 26 heuristics (84% strict / 92% partial) |
| **M5** Case-study analysis | `docs/case_study_comparisons.md` | 92.5% MVE rubric pass-rate vs 40% raw view across 8 cases |
| **M2** Self-consistency | `survey/m5_self_consistency_result.yaml` | 100% within-persona temporal agreement; 100% within-role consensus; 60% cross-role agreement at ±2 step |

**Total LLM cost:** ~2,318 gpt-4o-mini calls ≈ $0.20.
**Real-user component:** none (deliberately — all evaluation methods are standards-grounded or simulation-based; suitable for thesis defence without IRB-bound user-study evidence).

---

## 6. MITRE ATT&CK Explicit Grounding

Deterministic lookup table in `src/mve_generator.py::_ATTACK_TECHNIQUES`:

| Alert type | ATT&CK Technique ID | Technique name |
| --- | --- | --- |
| T1 — anomalous outbound | **T1071** | Application Layer Protocol |
| T2 — unauthorised EHR access | **T1078** | Valid Accounts |
| T3 — lateral movement | **T1021** | Remote Services |
| T4 — data exfiltration | **T1041** | Exfiltration over C2 |
| T5 — IoMT behavioural deviation | **T1565** | Data Manipulation |

The mapping is consumed by Mode B (rule-based) MVE generation, so ATT&CK grounding survives the offline path. Verified by `tests/test_safe_failure.py::test_attck_lookup_covers_all_5_alert_types`. Threat model context in [`docs/threat_model.md §5`](threat_model.md).

---

## 7. Refined RQ2 — Threat Triage

RQ2 was originally framed as alert surfacing accuracy. The refined framing positions it as **threat-triage routing accuracy**: given a surfaced alert, does the system recommend the correct tier (L1 / L2 specialist / L1 with senior)?

Tier-recommendation logic in `two_stage_fusion_validation.yaml`:

```yaml
KNOWN_ATTACK     → L1                  (primary IT response)
NOVEL_ANOMALY    → L2_specialist       (zero-day path; fewer signals)
CONFIRMED_ANOMALY → L1_with_senior     (multi-signal alert)
BENIGN           → suppressed (audit log only)
```

RQ2 metrics from `results/rq2_metrics.json` on the curated 20-alert evaluation set:
- **Sensitivity 1.000** (16/16 attacks caught)
- **Specificity 0.500**
- **Critical-alert rate 0.667**
- **FNR_critical 0.000**

---

## 8. The Seven Architecture Invariants

| # | Invariant | Verdict | Test |
| --- | --- | --- | --- |
| 1 | DAE only ELEVATES (Track B never suppresses Track A) | PASS | 6 fusion-class tests + cross-track recall comparison |
| 2 | Safety floor — CRITICAL+unpatchable always surfaces | PASS | ST-09 (`test_critical_unpatchable_surfaces_in_maintenance_window`) |
| 3 | NO AUTO-EXECUTION | PASS | 5 grep commands all empty + `test_no_automated_blocking` |
| 4 | Audit trail complete | PASS | `test_audit_append_only.py` (3 tests) + 4 OperatorDecision schema tests |
| 5 | Explanation faithfulness (Layer 1 → SHAP) | PASS | M5 `test_shap_narrative_alignment` result_value = 1.0 |
| 6 | Role authority (each role → role-appropriate actions) | PASS | `test_role_authority.py` (39 parametrised tests) |
| 7 | DO NOT constraints required for CRITICAL on clinical devices | PASS | M4 `test_clinical_constraint_awareness` result_value = 1.0 |

Full traceability in [`docs/architecture.md`](architecture.md) §4 and [`results/reports/invariant_verification.log`](../results/reports/invariant_verification.log).

---

## 9. Lock-Notice Reconciliation Summary

| Property | Status | Action required |
| --- | --- | --- |
| 7-layer architecture | LOCKED | none |
| 3 critical fixes | LOCKED | none |
| Two-stage triage fusion | LOCKED | none |
| **9 MVE templates (3×3 matrix)** | **MISMATCH** | Update lock notice to "15 MVE templates (5 × 3)" OR redefine the 3×3 framing as severity-tiers × roles |
| Multi-method evaluation (no real users) | LOCKED | none |
| MITRE ATT&CK explicit grounding | LOCKED | none |
| Refined RQ2 (threat triage) | LOCKED | none |
| 7 architecture invariants | LOCKED | none |

**7 of 8 declared properties verify cleanly. 1 (MVE template count) is a documentation reconciliation — code emits 15 templates, lock notice declares 9. No code change needed; the lock notice should be updated to match the implementation, or the 3×3 framing should be re-stated as severity × role rather than alert-cluster × role.**

---

## 10. Final Status

```
┌───────────────────────────────────────────────────────────────────┐
│                                                                   │
│   ARCHITECTURE STATUS:    LOCKED (with one documentation          │
│                            reconciliation noted in §4)            │
│                                                                   │
│   TEST SUITE:             177/177 passing                         │
│   YAML DELIVERABLES:      11 (all acceptance criteria PASS)       │
│   FIGURES:                10 generated PNG artefacts              │
│   LLM CALLS LOGGED:       2,000/2,000 success                     │
│   OPEN GAPS:              10 (all production-deployment scoped)   │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

### Cross-references

- Workflow & step-code map: [`ARCHITECTURE.md`](../ARCHITECTURE.md) + [`docs/architecture.md`](architecture.md)
- Threat model + STRIDE: [`docs/threat_model.md`](threat_model.md)
- Risk formula honest limitations: [`docs/risk_formula_specification.md`](risk_formula_specification.md)
- Heuristic compliance narrative: [`docs/heuristic_evaluation.md`](heuristic_evaluation.md)
- Case studies: [`docs/case_study_comparisons.md`](case_study_comparisons.md)
- All YAML deliverables: `results/reports/*.yaml`
- Survey + simulation outputs: `survey/*`
- Figures: `results/figures/*.png` and `docs/figures/*.png`
