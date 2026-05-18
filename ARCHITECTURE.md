# XAI-IDS-Healthcare Architecture Overview

This repository implements an offline-first, explainable intrusion-detection workflow for healthcare and IoMT environments. The system separates batch data preparation and artifact generation from the online user interface. In practice, the pipeline produces scored alerts, explanations, and response guidance offline, and the Streamlit dashboard mainly reads those artifacts for browsing, study flows, and evaluation.

## Module Overview

The codebase is organized as a 7-stage flow:

1. **Module 0 - Dataset Audit** (`module0_analysis/phase0/`)
   Validates and profiles the WUSTL-EHMS-2020 source dataset, including integrity checks, quality reporting, and reproducibility artifacts.

2. **Module 1 - Preprocessing** (`module1_preprocessing/phase1/`)
   Sanitizes identifiers, encodes categorical fields, handles missing data, removes redundant features, scales features, and produces a **stratified 4-way split** (Strategy 1 — Frozen Test + Demo Pool):
   - `train_phase1.parquet` (60%) — model fitting (Track A XGBoost, Track B DAE)
   - `val_phase1.parquet` (15%) — threshold calibration, hyperparameter tuning
   - `test_phase1.parquet` (15%) — **frozen**, used only for paper metrics reporting
   - `demo_phase1.parquet` (10%) — **frozen**, used only for dashboard alerts and user study scenarios
   - `benign_only_train.parquet` — derived from train split, for DAE training
   - `split_metadata.yaml` — reproducibility provenance (random_state, sample counts, class distributions)

   **Critical invariant:** test and demo splits must NEVER be seen by any model during training. Module 2 enforces this via hard runtime assertions. Stratification preserves class proportions within ±2% across all splits. Split deterministic via `random_state=42`.

3. **Module 2 - Detection Training** (`module2_detection/`)
   Trains the dual-track detection stack:
   - **Track A: XGBoost-only production classifier**. `module2_train_models.py` defaults to fitting **only XGBoost**; Random Forest and Decision Tree are gated behind `--include-baselines` and reproduce the thesis Section 4 comparison on demand. Selection rationale: comparative evaluation showed XGBoost dominates on F2 and AUC (0.9952 on the 4-way split test set); maxing three correlated tree models inflated FPR without measurable FNR benefit.
   - **Track B: a denoising autoencoder (DAE) trained on benign-only behavior, raw features only.** The cascade design `[raw || P_xgb, P_rf, P_dt]` was evaluated via leave-one-class-out ablation on EHMS-2020 (N=2 folds, ΔAUC = +0.02 marginal) and MedSec-25 (N=4 folds, ΔAUC = −0.19 regression). Cascade fails systematically on higher-dimensional inputs due to capacity dilution; **production runs DAE-raw 25-dim** (`dae_final_report.json::architecture == "raw_25dim"`). Cascade is preserved in the thesis only as a negative ablation result.
   - Trained artifacts under `results/models/`. Module 2 enforces a leakage assertion at training-data load: the demo split (`demo_phase1.parquet`) is **NEVER** loaded by training functions; attempting to do so raises `RuntimeError`. The test split (`test_phase1.parquet`) is loaded only as a held-out evaluation set after fitting on train+val (no parameter sees it during fit).

4. **Module 3 - Composite Risk Scoring** (`module3_risk_scoring/`)
   Loads detection outputs and computes the composite risk score:
   `R = 0.40*C_detect + 0.25*D_crit + 0.15*S_data + 0.20*D_clinical_tier`
   It also maps `R` into the four alert tiers `CRITICAL`, `HIGH`, `MEDIUM`, and `LOW`, and exports batch scoring artifacts such as `results/reports/risk_scores.npz`.

5. **Module 4 - Explanations** (`module4_explanations/`, `src/mve_generator.py`)
   Produces analyst- and clinician-facing explanations using SHAP-derived feature context plus a rule-based or optional LLM-backed Minimum Viable Explanation (MVE) generator. The offline outputs include `analyst_report.json`, `clinician_summaries.json`, and example explanation artifacts.

6. **Module 5 - Response Guidance** (`module5_responses/`)
   Converts scored alerts and explanation context into response recommendations, policy outputs, audit records, and safety-aware mitigation guidance. This layer recommends actions but does not auto-execute enforcement.

7. **Module 6 - Evaluation and UI** (`module6_evaluation/`)
   Curates evaluation alerts, assembles dashboard-ready artifacts, runs evaluation metrics, and provides the Streamlit interface. The key output is `results/reports/evaluation_alerts.json`, which powers the dashboard's browse and study experiences. **Important separation of concerns:**
   - **Paper metrics** are computed from `test_phase1.parquet` (frozen test split) → `results/reports/risk_scores.npz` → `results/rq1_metrics.json`
   - **Dashboard alerts** for user study and browse mode are curated from `demo_phase1.parquet` (frozen demo pool) → `results/reports/demo_scores.npz` → `results/reports/evaluation_alerts.json`
   - These two paths are independent. Records in test split never appear in dashboard; records in demo pool never appear in paper metrics.

   Submodules:

   - `module6_evaluation.py` — builds evaluation artifacts; routes alerts through `_src_adapter`
   - `_src_adapter.py` — bridges `evaluation_alerts.json` records into `src.risk_scorer.score_alert()` with safe defaults (`patchable=True`, `event_context=None`)
   - `compute_rq1_metrics.py` — reads test split scores, outputs `results/rq1_metrics.json` (FNR_critical, sensitivity, specificity, confusion matrix). **Renamed from `compute_rq2_metrics.py`**: the script computes detection metrics, which belong to RQ1 in the current research-question framing.
   - `curate_demo_alerts.py` — reads `demo_scores.npz`, performs stratified sampling across (risk_tier, fusion_class, attack_class), outputs `evaluation_alerts.json` (~20 alerts for user study)
   - `study_loader.py` — loads 20 `AlertScenario` objects per participant; MD5-seeded deterministic shuffle; counterbalanced A/B assignment
   - `study_analysis.py` — reads `survey/study_responses_*.json`, computes M5 via Mann-Whitney U, outputs `survey/m5_result.yaml`
   - `module6_app.py` — Streamlit dashboard; browse mode, study mode, and response collection

## End-to-End Data Flow

The pipeline operates on four stratified splits with strict separation of concerns:

```
data/raw/WUSTL-EHMS/...csv
    └─ Module 1 preprocessing (4-way stratified split, random_state=42)
        ├─ data/processed/train_phase1.parquet      (60% — model fitting)
        ├─ data/processed/val_phase1.parquet        (15% — threshold calibration)
        ├─ data/processed/test_phase1.parquet       (15% — frozen, paper metrics)
        ├─ data/processed/demo_phase1.parquet       (10% — frozen, dashboard/study)
        ├─ data/processed/benign_only_train.parquet (DAE training subset)
        └─ data/processed/split_metadata.yaml       (provenance)

[Training path — train + val only]
    train_phase1.parquet + val_phase1.parquet
        └─ Module 2: train Track A (XGBoost) + Track B (DAE-raw)
            └─ results/models/*.pkl (frozen artifacts)

[Paper metrics path — test split, frozen]
    test_phase1.parquet
        └─ Module 3: score with frozen models
            └─ results/reports/risk_scores.npz
                └─ Module 6: compute_rq1_metrics.py
                    └─ results/rq1_metrics.json
                       (FNR_critical, sensitivity, specificity, confusion matrix)

[Dashboard / user study path — demo pool, frozen]
    demo_phase1.parquet
        └─ Module 3: score with frozen models
            └─ results/reports/demo_scores.npz
                └─ Module 4 explanations + Module 5 responses (per surfaced alert)
                    └─ Module 6: curate_demo_alerts.py
                        └─ results/reports/evaluation_alerts.json (~20 stratified alerts)
                            └─ Streamlit dashboard (browse + study modes)
```

`evaluation_alerts.json` is the primary offline handoff into the dashboard for Browse mode and Study mode. It contains alert metadata, risk tier, surfacing state, device context, fusion class, and both presentation variants:

- `group_a_display`: raw/baseline alert view
- `group_b_display`: explanation-enhanced alert view

**Leakage prevention:** Module 2 training functions raise `RuntimeError` if `test_phase1.parquet` or `demo_phase1.parquet` is loaded. Verified by `tests/test_data_split_integrity.py` (no row overlap between any pair of splits; stratification within ±2%; leakage assertions trigger correctly).

## RQ1 Detection Metrics Flow

Detection metrics are computed exclusively from the frozen test split. Demo pool is never used for paper metrics.

```text
data/processed/test_phase1.parquet
-> module3_risk_scoring/module3_risk_scores.py (--split test)
-> results/reports/risk_scores.npz
-> module6_evaluation/compute_rq1_metrics.py
-> results/rq1_metrics.json
   (critical_alert_rate, fnr_critical, TP/FN/FP/TN, sensitivity, specificity)
```

Note: `compute_rq1_metrics.py` was previously named `compute_rq2_metrics.py`. The script computes detection metrics, which align with RQ1 in the current research-question framing. Renamed for consistency.

## RQ3 / A/B User Study Flow

User study scenarios are sourced from the demo pool only. Test split records never appear in the dashboard or study materials.

```text
data/processed/demo_phase1.parquet
-> module3_risk_scoring/module3_risk_scores.py (--split demo)
-> results/reports/demo_scores.npz
-> module6_evaluation/curate_demo_alerts.py     (stratified sampling: risk_tier × fusion_class × attack_class)
-> results/reports/evaluation_alerts.json       (~20 alerts)
-> module6_evaluation/study_loader.py           (MD5-seeded per-participant shuffle + A/B assignment)
-> module6_evaluation/module6_app.py            (Streamlit; collects survey/study_responses_<PID>.json)
-> survey/study_responses_*.json
-> module6_evaluation/study_analysis.py         (M5 Mann-Whitney -> survey/m5_result.yaml)
-> analysis/analyze_rq3.py                      (final A/B analysis -> analysis/outputs/)
```

## Canonical System Workflow

```text
┌─────────────────────────────────────────────────────────────────┐
│                    CANONICAL SYSTEM WORKFLOW                    │
└─────────────────────────────────────────────────────────────────┘

┌──── OFFLINE (one-time training) ────┐
│                                     │
│  [1] Data Preparation               │
│       └─ WUSTL-EHMS-2020            │
│       └─ 4-way stratified split:    │
│          • train (60%)              │
│          • val (15%)                │
│          • test (15%, frozen)       │
│          • demo (10%, frozen)       │
│       └─ random_state=42            │
│                                     │
│  [2] Track A Training               │
│       └─ XGBoost (production)       │
│       └─ RF, DT (baselines only)    │
│       └─ Train + val splits only    │
│                                     │
│  [3] Track B Training               │
│       └─ DAE on benign-only train   │
│       └─ Raw 25 features (no cascade)│
│       └─ Cascade dropped per ablation│
│                                     │
│  [4] Threshold Calibration          │
│       └─ Per-track thresholds (val) │
│       └─ Risk-adaptive parameters   │
│                                     │
└─────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────── ONLINE INFERENCE (per alert) ────────────────────────┐
│                                                                              │
│  ╔══════════════════════════════════════════════════════════════════╗        │
│  ║                  Network Flow Record (raw 25 features)            ║        │
│  ╚════════════════════════════════╤══════════════════════════════════╝        │
│                                   │                                          │
│  ┌────────────────────────────────▼────────────────────────────────┐         │
│  │  [STEP 5] FEATURE SANITIZATION (UPDATED — security fix)          │         │
│  │                                                                   │         │
│  │   Input: 25 raw features (potentially with NaN/Inf)              │         │
│  │                                                                   │         │
│  │   IF feature is NaN or Inf:                                       │         │
│  │      replace with BENIGN_MEDIAN[feature_idx]   ★ NOT 0.0          │         │
│  │      data_quality_flag = "DEGRADED" if rate > 5%                  │         │
│  │      log warning if rate > 5%                                     │         │
│  │   ELSE: pass through                                              │         │
│  │                                                                   │         │
│  │   Output: sanitized features + data_quality_flag                  │         │
│  │                                                                   │         │
│  │   ★ MITIGATION: Defends against EA-06 NaN injection attack        │         │
│  └────────────────────────────────┬────────────────────────────────┘         │
│                                   │                                          │
│         ┌─────────────────────────┴─────────────────────────┐                │
│         │                                                    │                │
│  ┌──────▼──────────────────────┐                ┌──────────▼─────────────────┐│
│  │ [STEP 6a] TRACK A           │                │ [STEP 6b] TRACK B          ││
│  │ Production Detection         │                │ Novelty Detection          ││
│  │                              │                │                            ││
│  │ XGBoost (production)         │                │ Input: 25 raw features     ││
│  │ ┌────────────┐               │                │ (NO cascade — dropped per  ││
│  │ │  XGBoost   │ → P_xgb       │                │  ablation evidence)        ││
│  │ │ (Primary)  │               │                │                            ││
│  │ └────────────┘               │                │ DAE.predict():             ││
│  │                              │                │  → reconstruction_error    ││
│  │ c_track_a = P_xgb            │                │  → normalize [0, 1]        ││
│  │                              │                │  → DAE_score               ││
│  │ ★ XGBoost-only production    │                │                            ││
│  │   (RF, DT are baselines      │                │ Threshold from training    ││
│  │    in thesis comparison      │                │ (95th percentile, val set) ││
│  │    only — not in inference)  │                │                            ││
│  │                              │                │ c_track_b = DAE_score      ││
│  │ ★ Selection rationale:       │                │                            ││
│  │   • Best F1 / AUC 0.9941     │                │ ★ Track B value:           ││
│  │   • max(XGB,RF,DT) inflated  │                │   • Detects anomalous      ││
│  │     FPR without FNR benefit  │                │     network signatures     ││
│  │                              │                │   • Limited on attacks     ││
│  │                              │                │     mimicking benign       ││
│  │                              │                │     (e.g., Spoofing)       ││
│  └──────────┬───────────────────┘                └──────────┬─────────────────┘│
│             │                                                │                  │
│             │                                                │                  │
│  ┌──────────▼────────────────────────────────────────────────▼─────────────┐  │
│  │  [STEP 7] TWO-STAGE FUSION (NEW — replaces simple max)                  │  │
│  │                                                                          │  │
│  │   Stage 1: KNOWN ATTACK                                                  │  │
│  │   ────────────────────                                                   │  │
│  │   IF P_xgb >= a_high (0.85, P_XGB_HIGH_CONF):                           │  │
│  │      alert_type = "KNOWN_ATTACK"                                         │  │
│  │      confidence = HIGH                                                   │  │
│  │      mve_template = known_pattern                                        │  │
│  │      tier_recommendation = L1                                            │  │
│  │                                                                          │  │
│  │   Stage 2: NOVEL ANOMALY                                                 │  │
│  │   ──────────────────────                                                 │  │
│  │   ELIF P_xgb < a_low (0.05, F2-tuned) AND DAE_score >= b (0.50):        │  │
│  │      alert_type = "NOVEL_ANOMALY"                                        │  │
│  │      confidence = MEDIUM                                                 │  │
│  │      mve_template = novel_pattern                                        │  │
│  │      tier_recommendation = L2_specialist                                 │  │
│  │      ★ Critical for IoMT zero-day                                        │  │
│  │                                                                          │  │
│  │   Stage 3: CONFIRMED ANOMALY                                             │  │
│  │   ──────────────────────────                                             │  │
│  │   ELIF a_low <= P_xgb < a_high AND DAE_score >= b:                      │  │
│  │      alert_type = "CONFIRMED_ANOMALY"                                    │  │
│  │      confidence = HIGH                                                   │  │
│  │      mve_template = multi_signal                                         │  │
│  │      tier_recommendation = L1_with_senior                                │  │
│  │                                                                          │  │
│  │   Stage 4: BENIGN                                                        │  │
│  │   ────────────                                                           │  │
│  │   ELSE:                                                                  │  │
│  │      alert_type = "BENIGN"                                               │  │
│  │      should_surface = False                                              │  │
│  │      → Suppressed (audit log only)                                      │  │
│  │                                                                          │  │
│  │   c_detect = max(P_xgb, DAE_score)  [for risk formula]                  │  │
│  │                                                                          │  │
│  │   ★ INVARIANT 1: DAE only elevates, never suppresses                    │  │
│  │      assert c_detect >= P_xgb                                            │  │
│  └──────────────────────────────┬────────────────────────────────────────┘  │
│                                  │                                             │
│  ┌───────────────────────────────▼────────────────────────────────────────┐  │
│  │  [STEP 8] CONTEXT ENRICHMENT (UPDATED — moved to src/, explicit defaults)│  │
│  │                                                                          │  │
│  │   Purpose: Transform numeric detection signal into clinical context     │  │
│  │   for downstream risk scoring (Step 9), MVE generation (Step 12), and   │  │
│  │   stakeholder views (Step 13).                                          │  │
│  │                                                                          │  │
│  │   Implementation: src/context_enrichment.py (shared runtime)            │  │
│  │   ★ Refactored from module6_evaluation/_src_adapter.py                  │  │
│  │     to align module boundaries (was: M6; now: src/)                     │  │
│  │                                                                          │  │
│  │   Inputs:                                                                │  │
│  │      ├─ Alert with src/dst IPs (from Step 7)                            │  │
│  │      ├─ device_inventory.yaml (asset database)                          │  │
│  │      ├─ device_clinical_tier_mapping.yaml (policy)                      │  │
│  │      └─ attack_to_mitre_mapping.yaml (threat intel, static)             │  │
│  │                                                                          │  │
│  │   Lookup logic:                                                          │  │
│  │      1. Match alert IP/MAC to device_inventory entry                    │  │
│  │      2. Extract: device_class, patchable, device_criticality,           │  │
│  │         data_sensitivity, manufacturer                                  │  │
│  │      3. Lookup clinical_tier from device_class via                      │  │
│  │         device_clinical_tier_mapping.yaml                               │  │
│  │      4. Map attack_category → MITRE ATT&CK techniques                   │  │
│  │         with confidence levels {HIGH, MEDIUM, LOW}                      │  │
│  │                                                                          │  │
│  │   D_clinical_tier weights (policy parameters, see mapping YAML):        │  │
│  │      ├─ tier_1_life_critical: 1.0   (e.g., infusion pump, ventilator)  │  │
│  │      ├─ tier_2_high_clinical: 0.8   (e.g., diagnostic monitor)         │  │
│  │      ├─ tier_3_moderate:      0.5   (e.g., EHR workstation)            │  │
│  │      ├─ tier_4_supportive:    0.3   (e.g., bedside terminal)           │  │
│  │      └─ tier_5_administrative: 0.1  (e.g., admin PC)                   │  │
│  │                                                                          │  │
│  │   ★ Required field 'patchable' must be present in alert; system        │  │
│  │     fails loudly (RuntimeError) if absent. No silent default.          │  │
│  │     Rationale: patchable=True default disabled safety floor             │  │
│  │     (CRITICAL+unpatchable always surfaces). Bug fixed.                  │  │
│  │                                                                          │  │
│  │   ┌─ UNKNOWN device handling (conservative-fail-safe) ──────────────┐ │  │
│  │   │  IF device not found in inventory:                              │ │  │
│  │   │     • device_class = "UNKNOWN"                                  │ │  │
│  │   │     • patchable = False        (conservative)                   │ │  │
│  │   │     • device_criticality = HIGH (conservative)                  │ │  │
│  │   │     • clinical_tier = tier_2_high_clinical (weight 0.8)         │ │  │
│  │   │     • data_sensitivity = UNKNOWN                                │ │  │
│  │   │     • warning_flag = "DEVICE_NOT_IN_INVENTORY"                  │ │  │
│  │   │     • Log event + emit secondary "rogue device" alert           │ │  │
│  │   │                                                                  │ │  │
│  │   │  Rationale: unknown device IS a security signal                 │ │  │
│  │   │  (rogue device, BYOD violation, asset management gap).          │ │  │
│  │   │  Should not be silently treated as low-risk.                    │ │  │
│  │   └────────────────────────────────────────────────────────────────┘ │  │
│  │                                                                          │  │
│  │   Outputs:                                                               │  │
│  │      Enriched alert with all 5 device fields populated +                │  │
│  │      mitre_techniques: [{id, name, confidence}, ...] +                  │  │
│  │      warning_flags: [...]                                               │  │
│  │                                                                          │  │
│  │   Tested by: tests/test_context_enrichment.py                           │  │
│  │   (UNKNOWN handling, missing-field assertions, IP matching,             │  │
│  │    MITRE coverage, tier consistency)                                    │  │
│  │                                                                          │  │
│  │   ★ KEY LIMITATION (Section 11): D_clinical_tier reflects               │  │
│  │     device class, NOT real-time patient acuity. The same infusion      │  │
│  │     pump on a stable post-op patient and a coding ICU patient gets     │  │
│  │     the same tier_1=1.0. Production deployment would integrate         │  │
│  │     EHR acuity scores (NEWS2/MEWS) — documented as Phase-3 work.       │  │
│  │                                                                          │  │
│  │   ★ KEY LIMITATION: MITRE mapping is rule-based and static, validated  │  │
│  │     against framework version X.Y. Production would benefit from       │  │
│  │     automated framework synchronization.                                │  │
│  └───────────────────────────────┬────────────────────────────────────────┘  │
│                                  │                                             │
│  ┌───────────────────────────────▼────────────────────────────────────────┐  │
│  │  [STEP 9] COMPOSITE RISK SCORING (UPDATED — sensitivity analysis, calibration)│
│  │                                                                          │  │
│  │   Configuration: configs/composite_risk_weights.yaml                     │  │
│  │   ★ Weights externalized; treated as policy parameters, not learned    │  │
│  │                                                                          │  │
│  │   Formula (linear weighted sum):                                         │  │
│  │                                                                          │  │
│  │      R = w_C × C_detect          (detection confidence, threat probability)│
│  │        + w_dcrit × D_crit         (device security criticality)         │  │
│  │        + w_sdata × S_data         (data sensitivity exposure)           │  │
│  │        + w_dclin × D_clinical_tier (clinical/patient impact proxy)      │  │
│  │                                                                          │  │
│  │   Default weights (from config, sum to 1.0):                            │  │
│  │      w_C     = 0.40   (detection dominates)                             │  │
│  │      w_dcrit = 0.25                                                     │  │
│  │      w_sdata = 0.15                                                     │  │
│  │      w_dclin = 0.20   ★ Renamed from A_patient → D_clinical_tier       │  │
│  │                                                                          │  │
│  │   Each factor ∈ [0, 1]; weights sum to 1.0; R ∈ [0, 1].                │  │
│  │                                                                          │  │
│  │   ★ Weights are POLICY PARAMETERS, not learned from data:              │  │
│  │      - Set by hospital security/clinical leadership                     │  │
│  │      - Reviewed annually                                                │  │
│  │      - Sensitivity analysis (Section 11) reports tier stability         │  │
│  │        under ±20% perturbation                                          │  │
│  │                                                                          │  │
│  │   Tier mapping (data-anchored, calibrated on test split):              │  │
│  │      ├─ R >= 0.80: CRITICAL  (top ~5%; immediate action)               │  │
│  │      ├─ 0.60 <= R < 0.80: HIGH  (next ~20%; investigate within 1 hour)│  │
│  │      ├─ 0.40 <= R < 0.60: MEDIUM (next ~40%; review within shift)     │  │
│  │      └─ R < 0.40: LOW  (bottom ~35%; audit log review only)           │  │
│  │      ★ Boundaries verified to fall between R distribution clusters     │  │
│  │        on test split, not through them (Section 11 figure X)           │  │
│  │                                                                          │  │
│  │   ┌─ KEY LIMITATIONS (Section 11) ─────────────────────────────────┐  │  │
│  │   │                                                                  │  │  │
│  │   │  L1. Linear weighted sum ≠ true multiplicative risk semantics  │  │  │
│  │   │      Standard security risk ≈ P(threat) × Impact, but linear    │  │  │
│  │   │      sum allows compensatory effects (e.g., high D_crit alone   │  │  │
│  │   │      pushes alert into HIGH tier even when C_detect = 0).      │  │  │
│  │   │      Production deployment would benefit from R = C_detect ×   │  │  │
│  │   │      V_asset structure. Linear retained for thesis: simpler,    │  │  │
│  │   │      bounded [0,1], easier to certify.                          │  │  │
│  │   │                                                                  │  │  │
│  │   │  L2. D_clinical_tier is device-class proxy for patient acuity  │  │  │
│  │   │      Same infusion pump on stable post-op vs coding ICU         │  │  │
│  │   │      patient gets the same tier_1 = 1.0. Production deployment │  │  │
│  │   │      with EHR integration (NEWS2/MEWS) would correct this      │  │  │
│  │   │      asymmetry. Reported FNR_critical averages across acuity    │  │  │
│  │   │      states; under-detects on unstable patients.                │  │  │
│  │   │                                                                  │  │  │
│  │   │  L3. D_crit and D_clinical_tier are correlated (r ≈ X reported │  │  │
│  │   │      in paper). Combined weight 0.45 effectively double-counts │  │  │
│  │   │      "device importance," exceeding C_detect's weight 0.40.    │  │  │
│  │   │      Acknowledged design choice (patient-safety bias); not bug.│  │  │
│  │   │                                                                  │  │  │
│  │   │  L4. Tier boundaries calibrated on test split distribution.    │  │  │
│  │   │      Different deployments (different device mix, different    │  │  │
│  │   │      attack distributions) may need recalibration.             │  │  │
│  │   └────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                          │  │
│  │   ★ INVARIANT: Risk tier ≠ should_surface                               │  │
│  │      Step 9 assigns tier (severity IF surfaced).                       │  │
│  │      Step 10 decides surfacing (whether to show operator).             │  │
│  │      See "Tier × Surfacing Truth Table" in Appendix B.                 │  │
│  │                                                                          │  │
│  │   ★ Audit logging: R component values logged on every alert            │  │
│  │      (c_detect, d_crit, s_data, d_clinical, computed_R, assigned_tier) │  │
│  │      Forensic value: "why was this CRITICAL?" answerable post-hoc.    │  │
│  │                                                                          │  │
│  │   Tested by tests/test_step9_composite_risk.py (NEW)                    │  │
│  └───────────────────────────────┬────────────────────────────────────────┘  │
│                                  │                                             │
│  ┌───────────────────────────────▼────────────────────────────────────────┐  │
│  │  [STEP 10] RISK-ADAPTIVE GATE (UPDATED — single decision tree, YAML config)│
│  │                                                                          │  │
│  │   Configuration: configs/risk_adaptive_thresholds.yaml                   │  │
│  │   ★ Hardcoded multiplier_table moved to YAML config                     │  │
│  │                                                                          │  │
│  │   Single decision tree (replaces dual-path early-return):               │  │
│  │                                                                          │  │
│  │   def should_surface(R, criticality, patchable, maintenance_active):    │  │
│  │       # Safety floor first (highest priority)                           │  │
│  │       if criticality == "CRITICAL" and not patchable:                   │  │
│  │           return True, "surfaced_safety_floor"                          │  │
│  │                                                                          │  │
│  │       # Maintenance window suppresses display only                      │  │
│  │       if maintenance_active:                                            │  │
│  │           return False, "suppressed_maintenance"                        │  │
│  │                                                                          │  │
│  │       # Normal threshold check                                          │  │
│  │       threshold = compute_threshold(criticality, patchable)             │  │
│  │       if R > threshold:                                                 │  │
│  │           return True, "surfaced_normal"                                │  │
│  │       return False, "suppressed_below_threshold"                        │  │
│  │                                                                          │  │
│  │   Reason captured on ScoredAlert.surfacing_reason ∈ {                   │  │
│  │      "surfaced_safety_floor",                                           │  │
│  │      "surfaced_normal",                                                 │  │
│  │      "suppressed_maintenance",                                          │  │
│  │      "suppressed_below_threshold"                                       │  │
│  │   }                                                                      │  │
│  │   ★ Forensic value: every suppression has explicit reason              │  │
│  │                                                                          │  │
│  │   Multiplier examples (full table in YAML):                             │  │
│  │      infusion_pump unpatchable: 0.70  (most sensitive)                 │  │
│  │      infusion_pump patchable:   0.85                                    │  │
│  │      patient_monitor unpatchable: 0.75                                  │  │
│  │      ehr_workstation patchable: 0.95                                    │  │
│  │      unknown fallback:          0.70  (conservative)                   │  │
│  │                                                                          │  │
│  │   Similar-events adjustment (campaign detection):                       │  │
│  │      IF similar_events_in_60min > 5:                                    │  │
│  │         multiplier = max(0.50, multiplier - 0.20)                       │  │
│  │      ★ Time window: 60 minutes (configured)                            │  │
│  │      ★ Similarity: same device + same attack_category                  │  │
│  │      ★ Lowers threshold (more sensitive) for suspected campaigns       │  │
│  │                                                                          │  │
│  │   ★ INVARIANT 2: Safety floor unconditional                             │  │
│  │      CRITICAL+unpatchable always surfaces, regardless of:              │  │
│  │      - maintenance window                                               │  │
│  │      - similar events                                                   │  │
│  │      - threshold value                                                  │  │
│  │                                                                          │  │
│  │   ★ Tier × surfacing truth table (Appendix B in paper):                │  │
│  │      Documents interaction between Step 9 (tier) and Step 10           │  │
│  │      (surfacing) for all combinations of risk_tier × patchable ×       │  │
│  │      maintenance to eliminate ambiguity in audit review.               │  │
│  └───────────────────────────────┬────────────────────────────────────────┘  │
│                                  │                                             │
│         ┌────────────────────────┴────────────────────────┐                   │
│         │ should_surface = False                           │                   │
│         ▼                                                  │                   │
│   ┌─────────────────────────────┐                         │                   │
│   │  SUPPRESSED                  │                         │                   │
│   │  → Tamper-evident audit log │  ★ Hash-chained        │                   │
│   │  → surfacing_reason logged  │  ★ Per-alert reason    │                   │
│   │  → Periodic review process  │                         │                   │
│   └─────────────────────────────┘                         │                   │
│                                                            │                   │
│         should_surface = True                              │                   │
│  ┌─────────────────────────────────────────────────────────▼───────────────┐ │
│  │                                                                          │ │
│  │  [STEP 11] SHAP EXPLANATION (UPDATED — stability measured, DAE gap noted)│ │
│  │  ───────────────────────────                                             │ │
│  │   For KNOWN_ATTACK / CONFIRMED_ANOMALY:                                  │ │
│  │      ├─ TreeSHAP on XGBoost (Track A)                                    │ │
│  │      ├─ Background dataset: 200-sample stratified train subset           │ │
│  │      │   persisted at results/models/shap_background.pkl                 │ │
│  │      ├─ Top-3 features extracted                                         │ │
│  │      ├─ Feature names mapped to clinician-readable labels                │ │
│  │      │   (configs/feature_categories.yaml)                                │ │
│  │      └─ Stability score (NEW):                                           │ │
│  │          • Generate SHAP for alert + 10 perturbations (±1% on continuous)│ │
│  │          • stability = mean overlap of top-3 across perturbations        │ │
│  │          • is_stable = (stability >= 0.90)                               │ │
│  │          • Persisted on SHAPContext.stability_score                      │ │
│  │                                                                          │ │
│  │   For NOVEL_ANOMALY:                                                     │ │
│  │      ★ KNOWN GAP: XGBoost SHAP is not faithful when DAE drives alert    │ │
│  │      Current behavior: XGBoost SHAP still computed but flagged          │ │
│  │         shap_source = "xgboost_low_confidence"                          │ │
│  │      ★ Future work: per-feature reconstruction-error attribution        │ │
│  │         from DAE for true novelty explanation                           │ │
│  │                                                                          │ │
│  │   Output: SHAPContext {                                                  │ │
│  │      top_features: List[str],                                           │ │
│  │      shap_values: array,                                                │ │
│  │      stability_score: float,                                            │ │
│  │      is_stable: bool,                                                   │ │
│  │      shap_source: Literal["xgboost", "xgboost_low_confidence", "dae_recon"]│
│  │   }                                                                      │ │
│  │                                                                          │ │
│  │  [STEP 12] MVE 3-LAYER GENERATION (UPDATED — invariants tested)          │ │
│  │  ────────────────────────────────                                        │ │
│  │   Layer 1: WHY anomalous (≤60 words)                                     │ │
│  │      ├─ References SHAP top features (Invariant 5)                       │ │
│  │      ├─ References MITRE ATT&CK with confidence (HIGH/MEDIUM/LOW)       │ │
│  │      └─ Confidence score from detection                                  │ │
│  │                                                                          │ │
│  │   Layer 2: CLINICAL IMPACT (≤50 words)                                   │ │
│  │      ├─ References specific clinical_tier (Invariant 8 NEW)              │ │
│  │      ├─ Concrete patient-care implication                                │ │
│  │      ├─ Severity tier (consistent with R)                                │ │
│  │      └─ Invariant across stakeholder roles (Invariant 6)                 │ │
│  │                                                                          │ │
│  │   Layer 3: RECOMMENDED ACTION (≤60 words action + ≤30 words DO_NOT)     │ │
│  │      ├─ Action priority order (isolate → restrict → ... → log)           │ │
│  │      ├─ Action verbs role-specific (configs/role_action_authorization.yaml)│
│  │      ├─ DO NOT constraint (clinical safety)                              │ │
│  │      └─ Required for CRITICAL on clinical devices (Invariant 7)          │ │
│  │                                                                          │ │
│  │   Mode A (LLM-based): primary                                            │ │
│  │      ★ Reproducibility: full prompt + response + model_version logged   │ │
│  │      ★ Word budget enforced post-generation (truncate at sentence)      │ │
│  │      ★ PHI flow: alert metadata only; no patient identifiers            │ │
│  │         (validated by tests/test_phi_not_in_llm_prompt.py)              │ │
│  │      ★ Provider/model: persisted per call for audit reproducibility     │ │
│  │                                                                          │ │
│  │   Mode B (rule-based): fallback when LLM fails                           │ │
│  │      ★ UI badge: "Rule-based fallback" displayed when Mode B active     │ │
│  │      ★ Templates per (attack_class × device_class) combination          │ │
│  │      ★ Always within word budget by construction                        │ │
│  │                                                                          │ │
│  │   Validation (post-generation, fail-closed):                             │ │
│  │      ├─ Invariant 5: SHAP top-3 features (or human-readable forms)      │ │
│  │      │   appear as substrings in Layer 1; if Mode A fails → Mode B      │ │
│  │      ├─ Invariant 6: Layer 2 severity invariant across role views       │ │
│  │      ├─ Invariant 7: DO_NOT present for CRITICAL+clinical devices       │ │
│  │      └─ Invariant 8: Layer 2 references clinical_tier name              │ │
│  │                                                                          │ │
│  │   Tested by:                                                             │ │
│  │      tests/test_step12_mve_faithfulness.py (Invariants 5, 7, 8)         │ │
│  │      tests/test_phi_not_in_llm_prompt.py   (PHI leak prevention)        │ │
│  │      tests/test_coverage_mve.py             (branch coverage, all types)│ │
│  └───────────────────────────────┬────────────────────────────────────────┘  │
│                                  │                                             │
│  ┌───────────────────────────────▼────────────────────────────────────────┐  │
│  │  [STEP 13] STAKEHOLDER ADAPTATION (UPDATED — shared anchor + auth matrix)│  │
│  │                                                                          │  │
│  │   Role from session/user: IT Generalist | Biomed Engineer | Nurse Mgr  │  │
│  │                                                                          │  │
│  │   ★ Shared Anchor (NEW — Invariant 9):                                  │  │
│  │      Every role view contains identical header:                         │  │
│  │      ┌──────────────────────────────────────────────────┐              │  │
│  │      │  Alert ID: ALERT-2024-1142                       │              │  │
│  │      │  Risk Tier: CRITICAL  |  Device: Bed 4-2 pump   │              │  │
│  │      │  One-line summary: "Suspected C2 communication" │              │  │
│  │      │  Timestamp: 2024-...                             │              │  │
│  │      └──────────────────────────────────────────────────┘              │  │
│  │      Layer 1 specialization happens BELOW the anchor.                  │  │
│  │      Prevents miscommunication during phone-based incident handling.   │  │
│  │                                                                          │  │
│  │   ★ Action Authorization (configs/role_action_authorization.yaml):       │  │
│  │      Each role has explicit authorized + forbidden action sets.        │  │
│  │      MVE Layer 3 generation queries this YAML to ensure suggested      │  │
│  │      actions are role-authorized.                                      │  │
│  │                                                                          │  │
│  │   IT Generalist View                                                    │  │
│  │   ───────────────────                                                   │  │
│  │   Layer 1: Technical detail, network features                            │  │
│  │      "Pattern matches T1071 (C2 communication). XGBoost: 92%.            │  │
│  │       Top features: outbound_bytes, beacon_period, dest_entropy."        │  │
│  │   Layer 2: Standard clinical impact                                      │  │
│  │   Layer 3: Network actions (isolate, restrict, snapshot)                │  │
│  │      DO NOT: clinical disruption warning                                 │  │
│  │                                                                          │  │
│  │   Biomed Engineer View                                                   │  │
│  │   ─────────────────────                                                  │  │
│  │   Layer 1: Device behavior interpretation                                │  │
│  │      "Infusion pump (Bed 4-2) showing unusual network behavior:          │  │
│  │       transmitting outside hospital network. Not normal telemetry."      │  │
│  │   Layer 2: Device function impact                                        │  │
│  │   Layer 3: Device actions (verify, document, coordinate)                │  │
│  │      DO NOT: power-cycle without backup ready                            │  │
│  │                                                                          │  │
│  │   Nurse Manager View                                                     │  │
│  │   ───────────────────                                                    │  │
│  │   Layer 1: Clinical impact framing                                       │  │
│  │      "Equipment in Bed 4-2 may be compromised. Patient safety priority." │  │
│  │   Layer 2: Patient monitoring implications                               │  │
│  │   Layer 3: Clinical actions (verify backup, monitor, document)          │  │
│  │      DO NOT: switch equipment without clinical reason                    │  │
│  │                                                                          │  │
│  │   ★ INVARIANT 6: Each role only authorizes role-appropriate actions      │  │
│  │      Verified by: tests/test_step13_cross_role_consistency.py            │  │
│  │   ★ INVARIANT 9 (NEW): Shared anchor identical across all role views     │  │
│  │      (alert_id, risk_tier, device_id, one_line_summary, timestamp)       │  │
│  │   ★ Cross-role severity invariance: deterministically tested             │  │
│  │      (in addition to post-hoc M5 study validation)                       │  │
│  └───────────────────────────────┬────────────────────────────────────────┘  │
│                                  │                                             │
│  ┌───────────────────────────────▼────────────────────────────────────────┐  │
│  │  [STEP 14] DISPLAY + TIER RECOMMENDATION (UPDATED — config-driven routing)│  │
│  │                                                                          │  │
│  │   Dashboard rendering:                                                   │  │
│  │      ├─ Alert card with role-appropriate view (Step 13)                 │  │
│  │      ├─ Shared anchor at top of every view (Invariant 9)                │  │
│  │      ├─ Tier badge: "🔵 Recommended for L2 review"                      │  │
│  │      ├─ Tooltip: tier rationale (cite top R contributor)                │  │
│  │      ├─ "Escalate" button (manual, no auto-route)                       │  │
│  │      ├─ Mode B badge if rule-based fallback active                      │  │
│  │      └─ SHAP stability indicator if stability_score < 0.90              │  │
│  │                                                                          │  │
│  │   Tier routing (configs/tier_routing.yaml — RECOMMENDATION ONLY):        │  │
│  │      Routing considers BOTH fusion_class AND risk_tier:                 │  │
│  │      ├─ KNOWN_ATTACK + CRITICAL → L1 + senior_engineer                  │  │
│  │      ├─ KNOWN_ATTACK + HIGH/MEDIUM → L1                                 │  │
│  │      ├─ NOVEL_ANOMALY + CRITICAL/HIGH → L2_specialist + IR              │  │
│  │      ├─ NOVEL_ANOMALY + MEDIUM/LOW → L1 (document for review)           │  │
│  │      └─ CONFIRMED_ANOMALY → L1 + senior_engineer                        │  │
│  │                                                                          │  │
│  │   ★ Routing accounts for clinical impact (not just detection class)     │  │
│  │   ★ Tooltip shows routing rationale when operator hovers                │  │
│  │                                                                          │  │
│  │   Hospital deployment sizing (configs/hospital_capabilities.yaml):       │  │
│  │      Small hospital (no L2_specialist available):                       │  │
│  │         L2_specialist → "document_for_external_consultant_review"      │  │
│  │         Notification → primary_contact, timeline = next business day   │  │
│  │      Medium/Large hospital: full tiered routing                         │  │
│  │      ★ NO queue separation, NO permission system in any size           │  │
│  │                                                                          │  │
│  │   Audit trail:                                                           │  │
│  │      ├─ Tier recommendation logged with rationale                       │  │
│  │      ├─ Routing decision logged (rule_id matched)                       │  │
│  │      └─ Actual handling logged (post-decision via Step 16)              │  │
│  └───────────────────────────────┬────────────────────────────────────────┘  │
│                                  │                                             │
│                          OPERATOR DECISION                                     │
│                                  │                                             │
│  ┌───────────────────────────────▼────────────────────────────────────────┐  │
│  │  [STEP 15] RESPONSE RECOMMENDATION (NO AUTO-EXECUTION)                   │  │
│  │                                                                          │  │
│  │   ★ INVARIANT 3: NO AUTO-EXECUTION (expanded grep)                      │  │
│  │      grep -rnE "subprocess|os\.system|iptables|netcat|nc\s|             │  │
│  │                  curl|wget|ssh|sudo|eval|exec\("                        │  │
│  │             pipeline/module5_response/                                   │  │
│  │      AND check imports:                                                  │  │
│  │      grep -rn "^import subprocess|^from subprocess"                     │  │
│  │             pipeline/module5_response/                                   │  │
│  │      Expected: empty                                                     │  │
│  │                                                                          │  │
│  │   Output schema (ResponseRecommendation):                                │  │
│  │      ├─ primary_action: str (human-readable)                            │  │
│  │      ├─ primary_action_code: str (machine-readable enum)                │  │
│  │      ├─ rationale: str                                                  │  │
│  │      ├─ estimated_clinical_impact: enum {minimal, moderate, high}       │  │
│  │      ├─ operator_decision_required: bool (always True)                  │  │
│  │      ├─ suggested_priority: int (1-5)                                   │  │
│  │      └─ do_not_actions: List[str] (explicit forbidden actions)          │  │
│  │                                                                          │  │
│  │   ★ Single source of truth for action recommendation:                   │  │
│  │      Step 13 (role-specific Layer 3) MUST align with Step 15's         │  │
│  │      primary_action_code. Different verbs allowed, same intent.        │  │
│  │      Verified by tests/test_step15_role_consistency.py.                │  │
│  │                                                                          │  │
│  │   All actions surface as RECOMMENDATIONS only:                          │  │
│  │      ├─ No command execution                                            │  │
│  │      ├─ No network operations                                           │  │
│  │      ├─ No firewall changes                                             │  │
│  │      └─ Operator approves/rejects manually                              │  │
│  │                                                                          │  │
│  │   Rationale:                                                             │  │
│  │      ├─ Prevents attacker abuse of automated containment                │  │
│  │      ├─ Prevents accidental clinical disruption                         │  │
│  │      └─ Preserves operator authority (HITL invariant)                   │  │
│  └───────────────────────────────┬────────────────────────────────────────┘  │
│                                  │                                             │
│  ┌───────────────────────────────▼────────────────────────────────────────┐  │
│  │  [STEP 16] OPERATOR DECISION LOGGING (UPDATED — hash-chain tamper evident)│  │
│  │                                                                          │  │
│  │   Schema (extended for forensic completeness):                           │  │
│  │   ┌─ Alert context ──────────────────────────────────────────────┐    │  │
│  │   │   alert_id, fusion_class, risk_tier                            │    │  │
│  │   │   recommended_action, primary_action_code                      │    │  │
│  │   ├─ Operator context ────────────────────────────────────────────┤    │  │
│  │   │   operator_role, view_role_rendered, view_role_match          │    │  │
│  │   │   participant_id (anonymous, for RQ3 study)                   │    │  │
│  │   │   group (A or B from counterbalanced study)                   │    │  │
│  │   ├─ Decision capture ──────────────────────────────────────────────┤    │  │
│  │   │   operator_action_taken, decision_time_seconds                │    │  │
│  │   │   operator_confidence (1-5, optional)                          │    │  │
│  │   │   operator_rationale (free text, optional)                     │    │  │
│  │   ├─ Explanation context (for RQ2 reproducibility) ──────────────┤    │  │
│  │   │   mve_mode_used (A_llm | B_rule)                              │    │  │
│  │   │   mve_text_shown (full text rendered to operator)             │    │  │
│  │   │   shap_features_shown (top-3)                                 │    │  │
│  │   │   shap_stability_score                                        │    │  │
│  │   │   For Mode A: llm_provider, llm_model_version,               │    │  │
│  │   │                full_prompt, full_response                     │    │  │
│  │   ├─ Tamper evidence (NEW) ──────────────────────────────────────┤    │  │
│  │   │   previous_hash: SHA256                                      │    │  │
│  │   │   entry_hash: SHA256(prev_hash + entry_serialized)            │    │  │
│  │   ├─ Forward compatibility (Step 17 placeholders) ───────────────┤    │  │
│  │   │   ground_truth_label: Optional[str] (filled later)            │    │  │
│  │   │   decision_quality: Optional[str] (filled later)              │    │  │
│  │   │   feedback_loop_consumed: bool = False                        │    │  │
│  │   └─ Timestamp ─────────────────────────────────────────────────────┘    │  │
│  │       timestamp_iso8601                                                  │  │
│  │                                                                          │  │
│  │   ★ decision_time_seconds = (operator_decision_time -                   │  │
│  │                              alert_displayed_to_operator_time)          │  │
│  │      Excludes pipeline latency. Documented unambiguously.               │  │
│  │                                                                          │  │
│  │   Storage: hash-chained JSON log (~50 LOC implementation)                │  │
│  │      ├─ Each entry includes SHA256 of previous entry                    │  │
│  │      ├─ Tampering with any entry breaks chain                           │  │
│  │      ├─ verify_audit_log_integrity() function provided                  │  │
│  │      └─ Tested by tests/test_step16_audit_integrity.py                  │  │
│  │                                                                          │  │
│  │   Production deployment: would route same schema to tamper-evident      │  │
│  │   audit store (SIEM, immutable database, or WORM storage).              │  │
│  │                                                                          │  │
│  │   ★ INVARIANT 4: Audit trail complete + tamper-evident                  │  │
│  └───────────────────────────────┬────────────────────────────────────────┘  │
│                                  │                                             │
│  ┌───────────────────────────────▼────────────────────────────────────────┐  │
│  │  [STEP 17] OUTCOME TRACKING (FUTURE WORK — acknowledged)                 │  │
│  │                                                                          │  │
│  │   Documented but NOT IMPLEMENTED (requires real deployment):             │  │
│  │      ├─ Outcome assessment (was it true positive?)                      │  │
│  │      ├─ Clinical follow-up tracking                                     │  │
│  │      └─ Operator decision quality assessment                            │  │
│  │                                                                          │  │
│  │  [STEP 18] CONTINUOUS IMPROVEMENT (FUTURE WORK — acknowledged)           │  │
│  │                                                                          │  │
│  │   Documented but NOT IMPLEMENTED:                                        │  │
│  │      ├─ Feedback into model retraining                                  │  │
│  │      ├─ Active learning architecture                                    │  │
│  │      └─ Threshold auto-tuning                                           │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└────────────────────────────────────────────────────────────────────────────────┘
```

### Workflow step → code map

The diagram above is the canonical design spec. Mapping each step to its current implementation, with deltas flagged where the code differs:

| Step | Code location | Implementation status |
| --- | --- | --- |
| [1] Data Preparation | `module1_preprocessing/phase1/pipeline.py` | **4-way stratified split** (Strategy 1 — Frozen Test + Demo Pool): train (60%), val (15%), test (15% frozen), demo (10% frozen). Class proportions preserved within ±2% (verified). Provenance in `data/processed/split_metadata.yaml`. Random state 42. **Hardening (security review)**: Phase 0 integrity verification (CSV bytes hashed against signed baseline before parsing); `PathValidator` rejects input/output paths that escape the workspace; strict-mode YAML loader rejects unknown top-level config sections (no silent fallback to defaults); `random_state` allowlist `{0, 7, 42}` warns on non-canonical seeds; `MissingValueHandler` refuses ffill without a `session_column` (cross-patient leakage protection); `RedundancyRemover` protects label columns from being dropped via tampered correlations. |
| [2] Track A Training | `module2_detection/module2_train_models.py` (`train_track_a`) | **XGBoost-only production both at training and runtime**. Default invocation fits only XGBoost; RF/DT are gated behind `--include-baselines` and emit thesis-Section-4 comparison artefacts on demand. Module 3 `triage_v4.py` consumes only `c_track_a = P_xgb`. Selection rationale: XGBoost dominates F1/AUC (0.9952); `max(P_xgb, P_rf, P_dt)` inflated FPR without measurable FNR benefit. SMOTE applied to train only, **inside the CV pipeline** (config exported by Module 1, applied here in Module 2). Leakage guard: `demo_phase1.parquet` is never loaded by training functions (raises `RuntimeError`); `test_phase1.parquet` is loaded only as a held-out evaluation set after fitting. |
| [3] Track B Training | `module2_detection/module2_train_models.py::train_track_b_dae` | **DAE on raw 25 features only** (no cascade). The artefact `results/models/dae_final_report.json` records `architecture: "raw_25dim"` and `n_track_a_features: 0` — `tests/test_track_a_xgb_only_v5.py` locks both. The 28-dim cascade `[raw \|\| P_xgb, P_rf, P_dt]` was evaluated via leave-one-class-out on EHMS-2020 (N=2, ΔAUC=+0.02 marginal) and MedSec-25 (N=4, ΔAUC=−0.19 regression) and rejected. Cascade is preserved only as a negative ablation result for the thesis. Trained on the held-out benign val subset (`benign_only_val.parquet`). |
| [4] Threshold Calibration | `module2_detection/models/_threshold.py::find_optimal_threshold` + `module2_detection/calibrate.py` | F2-optimal per-model thresholds (vectorized via `precision_recall_curve` — O(N log N) instead of the legacy O(T×N) Python loop). **Post-hoc probability calibration** (`module2_detection/calibrate.py`): Track A tree models produce uncalibrated probabilities by default (gradient-boosting shrinks toward 0.5; RandomForest gives hard tree-vote fractions; DecisionTree leaves are nearly one-hot). Each fitted Track A pipeline is wrapped with isotonic regression (≥1000 val samples) or Platt scaling (sigmoid, fewer samples), and the calibrated `*_val_proba_calibrated.npy` / `*_test_proba_calibrated.npy` arrays are persisted alongside the raw probas. The F2-tuned operating point is unchanged by calibration (calibration corrects probabilities, not the threshold). Risk-adaptive surfacing parameters live in `src/risk_scorer.py`. |
| [5] Feature Sanitization | `src/preprocessing.py::sanitize_features` (per-alert) + `module3_risk_scoring/module3_risk_scores.py::_sanitise_features` (batch) | Per-feature **BENIGN_MEDIAN** replacement (not 0.0) using `data/processed/benign_medians.json` (computed from 9990 benign training samples). `data_quality` ∈ {`OK`, `IMPUTED_NAN`, `DEGRADED`, `FAILED`} on `ScoredAlert.data_quality`. EA-06 mitigation in `src/risk_scorer.py`: `DEGRADED` ×1.20 score elevation; `FAILED` clamps `anomaly_score >= 0.95` so the alert always surfaces. Covered by `tests/test_feature_sanitization.py` (7 tests) and `results/reports/feature_sanitization.yaml`. |
| [6a] Track A inference | `module3_risk_scoring/module3_risk_scores.py::predict_track_a` | **XGBoost-only**: `c_track_a = P_xgb`. RF/DT not invoked at inference time. |
| [6b] Track B inference | `module3_risk_scoring/module3_risk_scores.py::predict_track_b` | DAE inference on raw features only. **Refactored from M4 to M3** to align module boundaries (detection in M3, explanations in M4). Returns `c_track_b = DAE_score` normalized to [0,1]. |
| [7] Two-Stage Fusion | `module3_risk_scoring/module3_risk_scores.py::compute_c_detect` + `::classify_fusion` | Fusion produces both `c_detect = max(c_track_a, c_track_b) = max(P_xgb, DAE_score)` and a per-alert `fusion_class` ∈ {`KNOWN_ATTACK`, `CONFIRMED_ANOMALY`, `NOVEL_ANOMALY`, `BENIGN`}. `KNOWN_ATTACK` boundary at `P_xgb ≥ 0.85` (`P_XGB_HIGH_CONF` in `src/data_models.py`). Persisted on `ScoredAlert.fusion_class`, in `risk_scores.npz["fusion_class"]`, and in `evaluation_alerts.json`. |
| [8] Context Enrichment | `src/context_enrichment.py` (refactored from `module6_evaluation/_src_adapter.py`) | Loads `device_inventory.yaml`, `device_clinical_tier_mapping.yaml`, `attack_to_mitre_mapping.yaml`. Matches alert IP/MAC to inventory entry. Required field `patchable` must be present (no default). UNKNOWN device handling: conservative fallback (`patchable=False`, `device_criticality=HIGH`, `clinical_tier=tier_2`, warning flag, secondary "rogue device" alert). MITRE mapping with confidence levels (HIGH/MEDIUM/LOW). M3 (`compute_composite_risk`) and M6 (dashboard rendering) both import from this module — single source of truth. Tested by `tests/test_context_enrichment.py`. |
| [9] Composite Risk Scoring | `module3_risk_scoring/module3_risk_scores.py::compute_composite_risk` | Linear weighted sum: `R = w_C·C_detect + w_dcrit·D_crit + w_sdata·S_data + w_dclin·D_clinical_tier`. Default weights (0.40 / 0.25 / 0.15 / 0.20) externalized to `configs/composite_risk_weights.yaml`. Tier mapping (CRITICAL/HIGH/MEDIUM/LOW) anchored to test-split percentiles, verified to fall between R distribution clusters. **Sensitivity analysis** in paper Section 11: tier assignment stable under ±20% weight perturbation for X% of alerts. **Acknowledged limitations**: linear sum vs multiplicative semantics (L1), D_clinical_tier as device-class proxy for patient acuity (L2), D_crit/D_clinical_tier correlation = double-counting "device importance" (L3), tier boundaries calibrated to test split (L4). **Audit**: R components logged per alert (c_detect, d_crit, s_data, d_clinical, R, tier) for forensic review. Tested by `tests/test_step9_composite_risk.py` (NEW): formula correctness, weight sum validation, tier boundary edge cases, sensitivity analysis fixture, R component audit logging. |
| [10] Risk-Adaptive Gate | `src/risk_scorer.py::score_alert` | Refactored to **single decision tree** (replaces dual-path early-return for cleaner reasoning). Multiplier table externalized to `configs/risk_adaptive_thresholds.yaml`. Surfacing decision returns `(should_surface, surfacing_reason)` where reason ∈ {`surfaced_safety_floor`, `surfaced_normal`, `suppressed_maintenance`, `suppressed_below_threshold`}. Safety floor (CRITICAL+unpatchable always surfaces) is highest-priority check. Maintenance window suppresses display, not detection. Covered by `tests/test_safe_failure.py::test_critical_unpatchable_surfaces_in_maintenance_window` and `tests/test_step10_surfacing_logic.py` (NEW). |
| [11] SHAP Explanation | `module4_explanations/module4_online_explainer.py::build_shap_context` | TreeSHAP on XGBoost with persisted background sample (`results/models/shap_background.pkl`, 200 stratified samples from train). **Stability score IMPLEMENTED**: 10 perturbations at ±1% on continuous features; `stability_score = mean overlap of top-3`; `is_stable = (stability >= 0.90)`. Persisted on `SHAPContext.stability_score`. Feature names mapped via `configs/feature_categories.yaml`. **Known gap**: for NOVEL_ANOMALY (DAE-driven), XGBoost SHAP flagged with `shap_source = "xgboost_low_confidence"`; per-feature DAE reconstruction-error attribution is future work. Covered by `tests/test_step11_shap_stability.py` (NEW). |
| [12] MVE 3-Layer Generation | `src/mve_generator.py::generate_mve` | Mode A (LLM via OpenAI API) and Mode B (rule-based) both present. **Invariants explicitly tested**: Invariant 5 (Layer 1 references SHAP top-3 features as substrings) — fails closed to Mode B; Invariant 7 (DO_NOT for CRITICAL+clinical); Invariant 8 NEW (Layer 2 references clinical_tier name). Word budgets enforced post-generation with sentence-boundary truncation. Mode A reproducibility: full prompt+response+model_version persisted per call. PHI flow documented in `configs/llm_data_flow.yaml`; verified by `tests/test_phi_not_in_llm_prompt.py` (NEW). Mode B templates per (attack_class × device_class). Coverage: `tests/test_step12_mve_faithfulness.py` (NEW), `tests/test_coverage_mve.py`. |
| [13] Stakeholder Adaptation | `src/mve_generator.py::derive_role_view` + `module5_responses/module5_pipeline.py::render_views_for_alert` | Per-role views: `OperatorRole` enum (IT_generalist / biomed_engineer / nurse_manager). **Invariant 9 NEW (shared anchor)**: alert_id, risk_tier, device_id, one_line_summary, timestamp identical across all role views; placed at top of every view to prevent miscommunication during incidents. **Action authorization** externalized to `configs/role_action_authorization.yaml`; MVE Layer 3 generation queries this YAML to ensure actions are role-authorized. Layer 2 invariant across roles (Invariant 6). Layer 3 `clinical_constraint` (DO NOT) preserved across roles (Invariant 7). **Cross-role severity invariance deterministically tested** by `tests/test_step13_cross_role_consistency.py` (NEW), supplementing post-hoc M5 Mann-Whitney study. |
| [14] Display & Tier Recommendation | `module6_evaluation/module6_app.py` + `results/reports/alert_responses.json` | Streamlit dashboard for browse/study modes. Dashboard shows shared anchor (Invariant 9), tier badge, Mode B badge, SHAP stability indicator if low. **Tier routing externalized** to `configs/tier_routing.yaml`: routing rules consider both `fusion_class` AND `risk_tier` (e.g., NOVEL+CRITICAL routes differently from NOVEL+LOW). **Hospital sizing** via `configs/hospital_capabilities.yaml`: small-hospital fallback documents NOVEL for external consultant review. Audit trail captures tier recommendation rationale + actual handling (post-decision via Step 16). |
| [15] Response Recommendation | `module5_responses/module5_pipeline.py::recommend` | Returns structured `ResponseRecommendation` dataclass: `primary_action` (string), `primary_action_code` (machine-readable enum), `rationale`, `estimated_clinical_impact`, `operator_decision_required` (always True), `suggested_priority`, `do_not_actions`. Single source of truth for action: Step 13 role-specific Layer 3 must align with `primary_action_code`. **NO AUTO-EXECUTION invariant verified** by expanded grep checking subprocess/os.system/iptables/netcat/curl/wget/ssh/sudo/eval/exec + import statements (`tests/negative_tests.py::test_no_automated_blocking`). Cross-role consistency verified by `tests/test_step15_role_consistency.py` (NEW). |
| [16] Operator Decision Logging | `module6_evaluation/module6_app.py` (study mode → `survey/study_responses_*.json`); `results/reports/alert_responses.json` for browse mode | **Hash-chained tamper-evident audit log** (~50 LOC implementation): each entry includes SHA256 of previous entry; `verify_audit_log_integrity()` validates entire chain. Schema extended for forensic completeness: alert context (alert_id, fusion_class, risk_tier), operator context (role, view_role_rendered, participant_id, group), decision capture (action_taken, decision_time_seconds, confidence, rationale), explanation context (mve_mode, mve_text_shown, shap_features, shap_stability; for Mode A: full prompt+response+model_version), tamper evidence (previous_hash, entry_hash), forward compatibility for Step 17 (ground_truth_label, decision_quality, feedback_loop_consumed — all Optional). `decision_time_seconds` measured from alert-displayed-to-operator time, excludes pipeline latency. Tested by `tests/test_step16_audit_integrity.py` (NEW). Production deployment would route same schema to SIEM/WORM storage. |
| [17] Outcome Tracking | — | **FUTURE WORK** — requires real deployment with ground-truth feedback channel (true-positive verification, clinical follow-up). Not implemented; documented as a Phase-3 capability. |
| [18] Continuous Improvement | — | **FUTURE WORK** — feedback-into-retraining, active learning, and threshold auto-tuning are not implemented. `feedback_loop_demo.py` is a single-pass simulation, not a closed loop. |

## Design Invariants

- **Split integrity (NEW)**: Test and demo splits are frozen. Module 2 training functions raise `RuntimeError` if `test_phase1.parquet` or `demo_phase1.parquet` is loaded. No row appears in more than one split. Verified by `tests/test_data_split_integrity.py`.
- **Track A is XGBoost-only in production**: RF and DT are comparative baselines reported in thesis Section 4 but excluded from inference. Selection backed by ablation (XGBoost dominates F1/AUC; max-fusion inflated FPR without FNR benefit).
- **Track B uses raw features only (no cascade)**: cascade design `[raw || probas]` evaluated and rejected. EHMS-2020 LOO showed marginal improvement (ΔAUC=+0.02); MedSec-25 LOO showed regression (ΔAUC=−0.19). Production design is DAE-raw.
- **Track B only elevates detection confidence**: fusion uses `max(Track_A, Track_B)`, so the DAE cannot suppress a stronger Track A signal. Experimentally validated on Spoofing fold (EHMS-2020): when Track B fails (AUC≈0.52), max() preserves Track A signal.
- **Risk tier and surfacing are separate concerns**:
  - Module 3 assigns `risk_level` from the batch composite score `R`
  - `src/risk_scorer.py` decides `should_surface` using adaptive thresholds, patchability, and event context
- **Composite risk weights are policy parameters** (NEW): `R = w_C·C_detect + w_dcrit·D_crit + w_sdata·S_data + w_dclin·D_clinical_tier`. Default weights (0.40/0.25/0.15/0.20) externalized to `configs/composite_risk_weights.yaml`. Not learned from data; set by hospital leadership. Sensitivity analysis on test split shows tier stability under ±20% perturbation. Tested by `tests/test_step9_composite_risk.py`.
- **Composite risk components are audit-logged** (NEW): every `ScoredAlert` records `c_detect`, `d_crit`, `s_data`, `d_clinical_tier`, `composite_R`, and `risk_tier` for forensic reproducibility. The question "why was this alert CRITICAL?" must be answerable post-hoc by inspecting components.
- **Tier boundaries are data-anchored** (NEW): boundaries (0.80, 0.60, 0.40) calibrated to test-split R distribution to avoid cutting through clusters. Documented in Section 11 with histogram. Different deployments may need recalibration.
- **Tier × surfacing truth table is documented** (NEW): paper Appendix B documents all combinations of (risk_tier × patchable × maintenance_active) → should_surface to eliminate ambiguity in audit review.
- **Offline-first explanations**: the MVE generator works without API keys through deterministic rule-based fallback logic.
- **Recommendation only, no enforcement**: Module 5 produces response guidance and audit outputs, not live containment actions.
- **Paper metrics never touch demo pool**: `compute_rq1_metrics.py` reads only `risk_scores.npz` (test split). Dashboard alerts in `evaluation_alerts.json` come only from `demo_scores.npz` (demo pool). The two paths are independent.
- **Context enrichment owns its module** (NEW): Step 8 lives in `src/context_enrichment.py`, imported by both M3 (composite risk scoring) and M6 (dashboard rendering). Single source of truth for device-to-clinical-context mapping. M3 no longer depends on M6 logic.
- **Patchable field is explicit, not defaulted** (UPDATED): `enrich_alert_context()` raises `RuntimeError` if `patchable` is absent from alert record. Previous behavior (`patchable=True` default) silently disabled the safety floor and is now considered a fixed bug. Evaluation artifacts must emit `patchable` for all alerts.
- **UNKNOWN device handling is conservative-fail-safe** (UPDATED): missing inventory entries trigger `patchable=False`, `device_criticality=HIGH`, `clinical_tier=tier_2_high_clinical`, warning flag, and a secondary "rogue device" alert. Unknown is treated as a security signal, not as low-risk.
- **_src_adapter is now a thin wrapper**: `module6_evaluation/_src_adapter.py` retained for backward compatibility with `evaluation_alerts.json` schema, but delegates all enrichment logic to `src.context_enrichment`. Safe defaults removed; failures are loud.
- **study_loader determinism**: shuffle seed = `int(hashlib.md5(participant_id.encode()).hexdigest(), 16)`; A/B assignment counterbalanced by `seed % 2`.
- **Safety floor holds on all paths** (`src/risk_scorer.py`): a CRITICAL+unpatchable device always surfaces. Step 10 refactored to single decision tree (replaces dual-path early-return) — safety floor is the highest-priority check. Covered by `tests/test_safe_failure.py::test_critical_unpatchable_surfaces_in_maintenance_window` and `::test_low_patchable_suppressed_in_maintenance_window`.
- **Surfacing reason is captured** (NEW): every `should_surface` decision records `surfacing_reason ∈ {surfaced_safety_floor, surfaced_normal, suppressed_maintenance, suppressed_below_threshold}` for forensic review. Audit cannot be ambiguous about why an alert did or didn't surface.
- **SHAP stability is measured, not assumed** (NEW): `SHAPContext.stability_score` captures top-3 feature consistency across 10 perturbations (±1% on continuous features). `is_stable = (stability >= 0.90)`. UI shows stability indicator when score is low. Covered by `tests/test_step11_shap_stability.py`.
- **MVE Layer 1 references actual SHAP features** (Invariant 5, NOW TESTED): Layer 1 text must contain SHAP top-3 feature names (or human-readable mappings) as substrings. Mode A failures fall back to Mode B (templated). Covered by `tests/test_step12_mve_faithfulness.py`.
- **MVE Layer 2 references clinical_tier** (Invariant 8, NEW): Layer 2 text must reference the specific tier name and at least one concrete patient-care implication. Strengthens RQ2.b.
- **MVE Layer 3 has DO_NOT for CRITICAL+clinical** (Invariant 7): preserved from prior version, now tested in `tests/test_step12_mve_faithfulness.py`.
- **Stakeholder views share an anchor** (Invariant 9, NEW): every role view contains identical header (alert_id, risk_tier, device_id, one_line_summary, timestamp). Layer 1 specialization happens below the anchor. Prevents miscommunication during phone-based incident handling. Covered by `tests/test_step13_cross_role_consistency.py`.
- **Role action authorization is config-driven** (NEW): `configs/role_action_authorization.yaml` defines authorized + forbidden actions per role. MVE Layer 3 generation queries this YAML. Closes Invariant 6 with explicit data, not implicit logic.
- **Cross-role severity invariance is deterministically tested** (NEW): in addition to post-hoc M5 Mann-Whitney study, every alert in test fixtures verifies severity is invariant across IT/Biomed/Nurse role views.
- **Audit log is hash-chained tamper-evident** (UPDATED): every entry contains SHA256(previous_hash + entry_serialized). Tampering breaks chain. `verify_audit_log_integrity()` validates entire log. Replaces the prior "append-only logical schema" claim. Production deployment would route to SIEM/WORM storage.
- **Mode A LLM calls are reproducibility-logged** (NEW): full prompt, full response, model version, provider name persisted per call. Required for HIPAA-grade audit reproducibility. PHI flow validated by `tests/test_phi_not_in_llm_prompt.py`.
- **Step 15 is single source of truth for actions** (NEW): `ResponseRecommendation.primary_action_code` is the canonical action; Step 13 role-specific Layer 3 verbs must align with it. Prevents inconsistencies between role-specific MVE and overall response recommendation.

## Ablation Evidence Summary

Two design decisions in the detection layer were validated empirically through ablation studies. Both are reported in the thesis as evidence-backed choices.

### Track A: XGBoost-only vs max-fusion

Comparison of three supervised classifiers on EHMS-2020 test split (per-model F1, AUC, FPR/FNR at F2-optimal threshold):

- XGBoost dominates on F1 and AUC (0.9952 on the 4-way test split, F2 0.9462)
- RF and DT show high pairwise correlation with XGBoost (shared training data, tree-based bias)
- `max(P_xgb, P_rf, P_dt)` inflated FPR without measurable FNR benefit on minority attack classes
- **Decision: production uses XGBoost only.** RF and DT retained as comparative baselines for thesis Section 4.

### Track B: DAE-raw vs DAE-cascade

Leave-one-class-out evaluation across two IoMT datasets:

| Dataset | N folds | DAE-raw mean AUC | DAE-cascade mean AUC | Δ vs raw | Verdict |
|---|---|---|---|---|---|
| EHMS-2020 | 2 | 0.758 | 0.778 | +0.021 | Marginal |
| MedSec-25 | 4 | 0.879 | 0.688 | −0.192 | Regression |

Cascade input `[raw || P_xgb]` failed to provide consistent improvement. On MedSec-25, cascade caused systematic regression across all 4 folds (worst case Reconnaissance: AUC 0.927 → 0.493). Mechanism: in higher-dimensional inputs, adding a low-signal proba dimension forces the autoencoder to allocate bottleneck capacity away from informative raw features (capacity dilution).

**Decision: production uses DAE on raw 25 features only.** Cascade is reported as a negative ablation result. Ablation artifacts: `dae_ablation_loo.yaml` (EHMS-2020), `dae_ablation_loo_medsec25.yaml` (MedSec-25).

### Track B failure mode (Spoofing)

EHMS-2020 Spoofing fold revealed Track B's fundamental limit: AUC=0.519 (near random) when attack signatures mimic benign network behavior. This is not a bug — it is intrinsic to benign-only anomaly detection. Mitigation:
- Track A (supervised, has seen Spoofing in train) catches this attack class
- Fusion `max(Track_A, Track_B)` ensures Track B's failure does not suppress Track A's signal (validated experimentally)
- Thesis threat model section explicitly notes that Track B targets attacks with anomalous signatures, not universal novelty


## Pipeline Configuration Files

The pipeline reads multiple YAML configuration files for policy parameters, asset state, and threat intelligence. Externalizing policy from code allows deployment-time review by clinical engineering, IT security, and patient safety leadership without code changes. The config files are grouped below by pipeline step.

### Step 8 — Context Enrichment configs

Step 8 is parameterized by three YAML configuration files that encode hospital policy, asset state, and threat intelligence. These are explicitly **policy parameters** (not learned), reviewed periodically by clinical engineering, IT security, and patient safety leadership.

### `configs/device_inventory.yaml`

Asset database mapping network identifiers to device context. Schema:

```yaml
devices:
  - device_id: string                 # unique (hostname or asset tag)
    ip_addresses: [string]             # IPv4/IPv6
    mac_addresses: [string]            # optional but recommended
    device_class: enum                 # see clinical_tier_mapping
    patchable: boolean                 # REQUIRED, no default
    device_criticality: enum           # CRITICAL | HIGH | MEDIUM | LOW
    data_sensitivity: enum             # PHI | biometric | telemetry | none
    manufacturer: string               # optional metadata
    model: string                      # optional metadata
    firmware_version: string           # optional, supports drift detection
    last_inventory_check: ISO8601      # optional, supports staleness check
    responsible_team: enum             # biomed | IT | clinical | unknown
```

Lookup priority for matching alert flow to inventory entry: (1) IP exact match, (2) MAC exact match if available, (3) hostname pattern match, (4) UNKNOWN fallback (conservative).

### `configs/device_clinical_tier_mapping.yaml`

Maps device class to clinical tier with rationale per device. Tier weights are policy parameters acknowledged in Section 11 limitations.

```yaml
mappings:
  infusion_pump:
    tier: tier_1_life_critical
    weight: 1.0
    rationale: "Direct medication delivery; failure = patient harm within minutes"
  ventilator:
    tier: tier_1_life_critical
    weight: 1.0
    rationale: "Life support; immediate failure = death"
  patient_monitor:
    tier: tier_1_life_critical
    weight: 1.0
    rationale: "Vital sign monitoring; failure = missed deterioration"
  ekg_machine:
    tier: tier_2_high_clinical
    weight: 0.8
    rationale: "Diagnostic; brief unavailability acceptable"
  ehr_workstation:
    tier: tier_3_moderate
    weight: 0.5
    rationale: "Clinical workflow but not real-time; PHI exposure concern"
  bedside_terminal:
    tier: tier_4_supportive
    weight: 0.3
    rationale: "Charting + low-criticality use"
  admin_workstation:
    tier: tier_5_administrative
    weight: 0.1
    rationale: "Non-clinical administrative use"

review:
  reviewers: ["CISO", "Clinical Engineering Director", "Patient Safety Officer"]
  review_period: "12 months"
```

### `configs/attack_to_mitre_mapping.yaml`

Static mapping from detection's `attack_category` to MITRE ATT&CK techniques with confidence levels. Validated against a pinned MITRE framework version.

```yaml
mappings:
  - attack_category: "Data Alteration"
    mitre_techniques:
      - id: "T1565"
        name: "Data Manipulation"
        confidence: HIGH
      - id: "T1565.001"
        name: "Stored Data Manipulation"
        confidence: MEDIUM
    last_validated: "[date]"

  - attack_category: "Spoofing"
    mitre_techniques:
      - id: "T1556"
        name: "Modify Authentication Process"
        confidence: MEDIUM
    last_validated: "[date]"

mitre_framework_version: "v14.1"  # pinned for reproducibility
```

### Sensitivity analyses (Section 11 acknowledgments)

Two sensitivity analyses are reported in the paper to defend policy-parameter choices:

**Analysis 1: D_clinical_tier values (Step 8)**
The composite risk formula (Step 9) uses `D_clinical_tier` at weight 0.20. Because tier values (1.0/0.8/0.5/0.3/0.1) are policy parameters, the paper reports: perturbing tier weights by ±20% across all devices shifts CRITICAL/HIGH/MEDIUM/LOW assignment for X% of alerts in the demo pool. Stable tier assignment under perturbation supports the framework's robustness even when specific weights are debatable.

**Analysis 2: Composite formula weights (Step 9)**
The four formula weights (w_C=0.40, w_dcrit=0.25, w_sdata=0.15, w_dclin=0.20) are perturbed across a grid of ±0.05 and ±0.10 increments (renormalized to sum to 1.0). Each perturbed weight set produces a tier assignment for the test split; agreement with baseline tier assignment is computed:

```
Mean tier agreement under ±0.05 perturbation: Y%
Mean tier agreement under ±0.10 perturbation: Z%
```

**Comparison with simple alternatives:**

| Formula variant | FNR_critical | FPR | Notes |
|---|---|---|---|
| Default weighted (current) | ? | ? | baseline |
| Equal weights (0.25 each) | ? | ? | does weighting help? |
| Detection-only (C_detect alone) | ? | ? | does context help? |
| Multiplicative R = C × V_asset | ? | ? | alternative formulation |

If default weighted does not outperform alternatives by meaningful margin on prioritization metrics, weights are not doing useful work — explicitly acknowledged.

**D_crit vs D_clinical_tier correlation analysis:**
Computed correlation r between D_crit and D_clinical_tier across device inventory; reported in paper. High correlation acknowledges double-counting of "device importance" (combined weight 0.45 > C_detect's 0.40); low correlation supports separation as semantically distinct.

### Other pipeline configuration files

In addition to the three Step 8 enrichment configs above, the pipeline reads five downstream configuration files. All policy parameters are externalized to YAML for deployment-time review.

#### `configs/composite_risk_weights.yaml` (Step 9)

```yaml
# Linear weighted sum: R = sum(w_i × component_i)
# Weights MUST sum to 1.0 (validated at load time)
# Reviewed annually by hospital security/clinical leadership

weights:
  detection_confidence:    0.40   # w_C — threat probability
  device_criticality:      0.25   # w_dcrit — security criticality
  data_sensitivity:        0.15   # w_sdata — data exposure impact
  clinical_tier:           0.20   # w_dclin — patient impact proxy

tier_boundaries:
  critical_min:  0.80
  high_min:      0.60
  medium_min:    0.40
  # below 0.40 → LOW

# Calibration metadata
calibration:
  anchored_to: "EHMS-2020 test split"
  date: "[date]"
  validated_against_distribution: true   # boundaries don't cut clusters
  expected_distribution:
    CRITICAL: "~5%"
    HIGH:     "~20%"
    MEDIUM:   "~40%"
    LOW:      "~35%"

review:
  reviewers: ["CISO", "Patient Safety Officer", "Clinical Engineering Director"]
  review_period: "12 months"

# Acknowledged limitations (Section 11):
# L1: Linear sum allows compensatory effects vs true multiplicative risk
# L2: clinical_tier is device-class proxy, not real-time patient acuity
# L3: device_criticality and clinical_tier are correlated (double-counting)
# L4: tier boundaries calibrated to test split; redeployment may need recalibration
```

#### `configs/risk_adaptive_thresholds.yaml` (Step 10)

```yaml
base_threshold: 0.50

device_multipliers:
  infusion_pump:
    unpatchable: 0.70
    patchable:   0.85
    rationale: "Lower threshold for unpatchable life-critical devices"
  patient_monitor:
    unpatchable: 0.75
    patchable:   0.90
  ehr_workstation:
    unpatchable: 0.80
    patchable:   0.95
  unknown:
    unpatchable: 0.70  # most conservative
    patchable:   0.80

similar_events_adjustment:
  threshold_count: 5
  reduction: 0.20
  floor: 0.50
  time_window_minutes: 60
  similarity_metric: "same_device + same_attack_category"
```

#### `configs/role_action_authorization.yaml` (Step 13)

```yaml
roles:
  IT_generalist:
    authorized_actions:
      - isolate_device_network
      - restrict_outbound_traffic
      - snapshot_traffic
      - alert_security_team
    forbidden_actions:
      - power_cycle_device
      - switch_clinical_equipment
      - modify_patient_care_workflow

  biomed_engineer:
    authorized_actions:
      - verify_device_function
      - document_anomaly
      - coordinate_with_manufacturer
      - schedule_firmware_update
    forbidden_actions:
      - isolate_device_network    # IT scope
      - modify_patient_orders     # clinical scope

  nurse_manager:
    authorized_actions:
      - verify_backup_equipment_ready
      - monitor_patient_directly
      - document_clinical_observation
      - escalate_to_physician
    forbidden_actions:
      - modify_device_settings
      - isolate_device_network
```

#### `configs/tier_routing.yaml` (Step 14)

```yaml
routing_rules:
  - id: "rule_known_critical"
    condition: "fusion_class == KNOWN_ATTACK and risk_tier == CRITICAL"
    primary: L1
    secondary: senior_engineer

  - id: "rule_novel_critical"
    condition: "fusion_class == NOVEL_ANOMALY and risk_tier in [CRITICAL, HIGH]"
    primary: L2_specialist
    secondary: incident_response

  - id: "rule_novel_low"
    condition: "fusion_class == NOVEL_ANOMALY and risk_tier in [MEDIUM, LOW]"
    primary: L1
    secondary: document_for_review

  - id: "rule_confirmed"
    condition: "fusion_class == CONFIRMED_ANOMALY"
    primary: L1
    secondary: senior_engineer
```

#### `configs/hospital_capabilities.yaml` (Step 14)

```yaml
deployment_size: medium  # small | medium | large

available_tiers:
  - L1
  - L2_specialist
  - senior_engineer

fallback_routing:
  L2_specialist_unavailable:
    action: "document_for_external_consultant_review"
    notification: "primary_contact"
    timeline: "next_business_day"
```

#### `configs/llm_data_flow.yaml` (Step 12)

Documents what data crosses the LLM API boundary. Validated by `tests/test_phi_not_in_llm_prompt.py`.

```yaml
mode_a_llm_inputs:
  allowed:
    - alert_id              # synthetic, not patient-derived
    - attack_category       # categorical
    - device_class          # categorical
    - shap_top3_features    # NetFlow feature names
    - risk_tier             # categorical
    - mitre_techniques      # public threat intel

  forbidden:
    - patient_id
    - patient_name
    - medical_record_number
    - room_number_with_patient_context
    - any_clinical_data_from_ehr

validation:
  sanitize_before_send: true
  log_full_prompt: true   # for audit reproducibility
```

#### `configs/feature_categories.yaml` (Step 11)

Maps the 25 raw feature names to clinician-readable labels and clinical-context categories. Used by SHAP feature display.

#### Configuration review process

All policy YAML files are reviewed periodically:
- `device_inventory.yaml`: monthly (asset team)
- `device_clinical_tier_mapping.yaml`: annually (clinical engineering + CISO + Patient Safety)
- `attack_to_mitre_mapping.yaml`: quarterly (security team, MITRE framework alignment)
- `composite_risk_weights.yaml`: annually (CISO + Patient Safety Officer + Clinical Engineering)
- `risk_adaptive_thresholds.yaml`: annually (CISO)
- `role_action_authorization.yaml`: annually (CISO + clinical engineering)
- `tier_routing.yaml`, `hospital_capabilities.yaml`: per-deployment
- `llm_data_flow.yaml`: locked; changes require privacy/security review
- `feature_categories.yaml`: when feature schema changes

## Directory Guide

- `src/`: shared runtime components such as `risk_scorer.py`, `mve_generator.py`, and data models
- `module0_analysis/` through `module6_evaluation/`: numbered pipeline stages
- `common/`: shared utilities such as model registry, signing, and PHI feature definitions
- `tests/`: acceptance, negative, and safety tests
- `results/`: generated models, reports, JSON artifacts, and charts
- `analysis/`: post-collection RQ3 analysis scripts
- `utils/`: one-off data conversion helpers

## Test Suite

| File | Purpose |
| --- | --- |
| `tests/acceptance_tests.py` | M1–M8 acceptance metrics on 50-alert fixture set |
| `tests/negative_tests.py` | 6 negative constraint tests (no discovery, no blocking, no CVSS, etc.) |
| `tests/test_safe_failure.py` | 5 failure-mode tests: missing context, timeout, unknown attack, extreme scores, unpatchable priority |
| `tests/test_coverage_mve.py` | MVE generator branch coverage: all 5 alert types, LLM path (mock), SHAP enrichment |
| `tests/test_feature_sanitization.py` | 7 tests covering BENIGN_MEDIAN imputation, data quality flags, EA-06 mitigation |
| `tests/test_data_split_integrity.py` | **NEW**: verifies (a) no row overlap between any pair of splits via row hash, (b) stratification preserved within ±2% across all splits, (c) leakage assertions trigger correctly when test/demo files are passed to training functions |
| `tests/test_context_enrichment.py` | **NEW**: 6+ tests covering Step 8 — IP-to-device matching, UNKNOWN device conservative fallback, `patchable` missing-field assertion (fails loudly), MITRE mapping coverage (no orphan attack_categories), D_clinical_tier consistency across calls, mixed-sensitivity device encoding |
| `tests/test_step9_composite_risk.py` | **NEW**: Step 9 composite risk tests — formula correctness against known fixtures, weights-sum-to-1.0 validation at config load, tier boundary edge cases (R=0.40, 0.60, 0.80), R component audit logging completeness, sensitivity analysis fixture (verify ±20% perturbation produces stable tier assignment for ≥X% of alerts), R distribution histogram check (boundaries don't cut clusters) |
| `tests/test_step10_surfacing_logic.py` | **NEW**: Step 10 surfacing decision tests — single decision tree paths, surfacing_reason capture for all 4 reasons, multiplier consistency from YAML config, similar-events adjustment within time window |
| `tests/test_step11_shap_stability.py` | **NEW**: Step 11 SHAP stability tests — perturbation-based stability score computation, is_stable threshold, top-3 overlap calculation, `shap_source` flagging for NOVEL_ANOMALY (XGBoost low-confidence flag) |
| `tests/test_step12_mve_faithfulness.py` | **NEW**: Step 12 MVE invariants — Invariant 5 (Layer 1 contains SHAP top-3 features as substrings, Mode A→B fallback on failure), Invariant 7 (DO_NOT for CRITICAL+clinical), Invariant 8 (Layer 2 references clinical_tier name), word budget enforcement |
| `tests/test_phi_not_in_llm_prompt.py` | **NEW**: validates that Mode A LLM prompts contain only allowlisted fields per `configs/llm_data_flow.yaml`; no patient identifiers, no MRN, no clinical EHR data |
| `tests/test_step13_cross_role_consistency.py` | **NEW**: Step 13 cross-role tests — Invariant 9 (shared anchor identical across roles), Invariant 6 (each role only authorizes role-appropriate verbs per `role_action_authorization.yaml`), Layer 2 severity invariance across IT/Biomed/Nurse |
| `tests/test_step15_role_consistency.py` | **NEW**: Step 15 cross-role consistency — every role's Layer 3 references same `primary_action_code` from `ResponseRecommendation`; expanded NO_AUTO_EXECUTION grep (subprocess/os.system/iptables/netcat/curl/wget/ssh/sudo/eval/exec + import statements) |
| `tests/test_step16_audit_integrity.py` | **NEW**: Step 16 hash-chain tests — append produces correct chain, tampering with any entry breaks `verify_audit_log_integrity()`, schema includes all forensic fields, decision_time_seconds semantics correct |

## Operational Model

The typical workflow is batch-first with strict separation between paper-metrics and dashboard-demo paths:

1. **Preprocess + split**: Module 1 produces 4 stratified parquet files + benign-only DAE training subset + provenance metadata.
2. **Train**: Module 2 trains XGBoost (Track A) and DAE (Track B) on train + val splits only. Hard-asserts that test/demo are not loaded.
3. **Score (paper path)**: Module 3 runs frozen models over test split → `risk_scores.npz` → Module 6 `compute_rq1_metrics.py` → `rq1_metrics.json`.
4. **Score (demo path)**: Module 3 runs frozen models over demo pool → `demo_scores.npz` → Module 4 explanations + Module 5 responses (per surfaced alert) → Module 6 `curate_demo_alerts.py` (stratified sampling) → `evaluation_alerts.json`.
5. **Study + Browse**: Streamlit dashboard reads `evaluation_alerts.json` for both browse mode and study mode. Per-participant deterministic shuffle and A/B assignment via `study_loader.py`.
6. **Analysis**: post-collection RQ3 analysis (`analyze_rq3.py`) reads `survey/study_responses_*.json`.

The Streamlit app is a presentation and study layer on top of the offline-computed demo artifacts, never the primary computation engine. Test split records do not appear in the dashboard at any point.
