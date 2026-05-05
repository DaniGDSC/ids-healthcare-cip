# XAI-IDS-Healthcare Architecture Overview

This repository implements an offline-first, explainable intrusion-detection workflow for healthcare and IoMT environments. The system separates batch data preparation and artifact generation from the online user interface. In practice, the pipeline produces scored alerts, explanations, and response guidance offline, and the Streamlit dashboard mainly reads those artifacts for browsing, study flows, and evaluation.

## Module Overview

The codebase is organized as a 7-stage flow:

1. **Module 0 - Dataset Audit** (`module0_analysis/phase0/`)
   Validates and profiles the WUSTL-EHMS-2020 source dataset, including integrity checks, quality reporting, and reproducibility artifacts.

2. **Module 1 - Preprocessing** (`module1_preprocessing/phase1/`)
   Sanitizes identifiers, encodes categorical fields, handles missing data, removes redundant features, splits train/test data, scales features, and exports `train_phase1.parquet`, `test_phase1.parquet`, and benign-only training data for Track B.

3. **Module 2 - Detection Training** (`module2_detection/`)
   Trains the dual-track detection stack:
   - Track A: supervised classifiers (`xgboost`, `random_forest`, `decision_tree`)
   - Track B: a denoising autoencoder (DAE) trained on benign behavior
   The trained artifacts are saved under `results/models/`.

4. **Module 3 - Composite Risk Scoring** (`module3_risk_scoring/`)
   Loads detection outputs and computes the composite risk score:
   `R = 0.40*C_detect + 0.25*D_crit + 0.15*S_data + 0.20*D_clinical_tier`
   It also maps `R` into the four alert tiers `CRITICAL`, `HIGH`, `MEDIUM`, and `LOW`, and exports batch scoring artifacts such as `results/reports/risk_scores.npz`.

5. **Module 4 - Explanations** (`module4_explanations/`, `src/mve_generator.py`)
   Produces analyst- and clinician-facing explanations using SHAP-derived feature context plus a rule-based or optional LLM-backed Minimum Viable Explanation (MVE) generator. The offline outputs include `analyst_report.json`, `clinician_summaries.json`, and example explanation artifacts.

6. **Module 5 - Response Guidance** (`module5_responses/`)
   Converts scored alerts and explanation context into response recommendations, policy outputs, audit records, and safety-aware mitigation guidance. This layer recommends actions but does not auto-execute enforcement.

7. **Module 6 - Evaluation and UI** (`module6_evaluation/`)
   Curates evaluation alerts, assembles dashboard-ready artifacts, runs evaluation metrics, and provides the Streamlit interface. The key output is `results/reports/evaluation_alerts.json`, which powers the dashboard's browse and study experiences.

   Submodules:

   - `module6_evaluation.py` — builds evaluation artifacts; routes alerts through `_src_adapter`
   - `_src_adapter.py` — bridges `evaluation_alerts.json` records into `src.risk_scorer.score_alert()` with safe defaults (`patchable=True`, `event_context=None`)
   - `compute_rq2_metrics.py` — reads `evaluation_alerts.json`, outputs `results/rq2_metrics.json` (FNR_critical, sensitivity, specificity, confusion matrix)
   - `study_loader.py` — loads 20 `AlertScenario` objects per participant; MD5-seeded deterministic shuffle; counterbalanced A/B assignment
   - `study_analysis.py` — reads `survey/study_responses_*.json`, computes M5 via Mann-Whitney U, outputs `survey/m5_result.yaml`
   - `module6_app.py` — Streamlit dashboard; browse mode, study mode, and response collection

## End-to-End Data Flow

The main data flow is:

`data/raw/WUSTL-EHMS/...csv`
-> Module 1 preprocessing
-> `data/processed/test_phase1.parquet`
-> Module 2 trained models in `results/models/`
-> Module 3 risk scores in `results/reports/risk_scores.npz`
-> Module 4 explanation artifacts
-> Module 5 response artifacts
-> Module 6 `evaluation_alerts.json`
-> Streamlit dashboard

`evaluation_alerts.json` is the primary offline handoff into the dashboard for Browse mode and Study mode. It contains alert metadata, risk tier, surfacing state, device context, and both presentation variants:

- `group_a_display`: raw/baseline alert view
- `group_b_display`: explanation-enhanced alert view

## RQ2 Analysis Flow

```text
results/reports/evaluation_alerts.json
-> module6_evaluation/compute_rq2_metrics.py
-> results/rq2_metrics.json
   (critical_alert_rate, fnr_critical, TP/FN/FP/TN, sensitivity, specificity)
```

## RQ3 / A/B User Study Flow

```text
results/reports/evaluation_alerts.json
-> module6_evaluation/study_loader.py    (MD5-seeded per-participant shuffle + A/B assignment)
-> module6_evaluation/module6_app.py     (Streamlit; collects survey/study_responses_<PID>.json)
-> survey/study_responses_*.json
-> module6_evaluation/study_analysis.py  (M5 Mann-Whitney -> survey/m5_result.yaml)
-> analysis/analyze_rq3.py              (final A/B analysis -> analysis/outputs/)
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
│       └─ Train/Val/Test split       │
│       └─ Stratified eval set        │
│                                     │
│  [2] Track A Training               │
│       └─ XGBoost / RF / DT          │
│       └─ Comparative evaluation     │
│       └─ Selected: XGBoost          │
│                                     │
│  [3] Track B Training               │
│       └─ DAE on benign-only         │
│       └─ Cascade input              │
│                                     │
│  [4] Threshold Calibration          │
│       └─ Per-track thresholds       │
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
│  │ Models in parallel:          │                │ Input augmentation:        ││
│  │ ┌────────────┐               │                │ [25 raw                    ││
│  │ │  XGBoost   │ → P_xgb       │                │  || P_xgb, P_rf, P_dt]    ││
│  │ │ (Primary)  │               │   ◄────────────│                            ││
│  │ ├────────────┤               │   probas       │ DAE.predict():             ││
│  │ │ RF         │ → P_rf        │   feed         │  → reconstruction_error    ││
│  │ │ (Reference)│               │                │  → normalize [0, 1]        ││
│  │ ├────────────┤               │                │  → DAE_score               ││
│  │ │ DT         │ → P_dt        │                │                            ││
│  │ │ (Reference)│               │                │ Threshold from             ││
│  │ └────────────┘               │                │ training (95th percentile) ││
│  │                              │                │                            ││
│  │ c_track_a = max(P_xgb,       │                │ c_track_b = DAE_score      ││
│  │                P_rf, P_dt)   │                │                            ││
│  │ ★ For thesis: P_xgb dominates│                │ ★ Novelty value:           ││
│  │   per comparative evaluation │                │   • LOO experiments        ││
│  │                              │                │   • Adversarial robustness ││
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
│  │  [STEP 8] CONTEXT ENRICHMENT                                             │  │
│  │                                                                          │  │
│  │   Lookup from device_inventory.yaml:                                     │  │
│  │      ├─ device_class                                                    │  │
│  │      ├─ patchable (boolean)                                             │  │
│  │      ├─ device_criticality (CRITICAL/HIGH/MEDIUM/LOW)                  │  │
│  │      └─ data_sensitivity (PHI/biometric/telemetry)                     │  │
│  │                                                                          │  │
│  │   D_clinical_tier (renamed from A_patient — honest)                     │  │
│  │      ├─ tier_1_life_critical: 1.0                                       │  │
│  │      ├─ tier_2_high_clinical: 0.8                                       │  │
│  │      ├─ tier_3_moderate: 0.5                                            │  │
│  │      ├─ tier_4_supportive: 0.3                                          │  │
│  │      └─ tier_5_administrative: 0.1                                      │  │
│  │                                                                          │  │
│  │   IF device_class == "UNKNOWN":                                         │  │
│  │      Infer from attack_category (with ⚠ flag)                           │  │
│  │      Conservative fallback                                              │  │
│  │                                                                          │  │
│  │   Threat intel mapping:                                                 │  │
│  │      └─ MITRE ATT&CK technique IDs from attack_category                 │  │
│  └───────────────────────────────┬────────────────────────────────────────┘  │
│                                  │                                             │
│  ┌───────────────────────────────▼────────────────────────────────────────┐  │
│  │  [STEP 9] COMPOSITE RISK SCORING (UPDATED — variable rename)             │  │
│  │                                                                          │  │
│  │   R = 0.40 × C_detect                                                    │  │
│  │     + 0.25 × D_crit                                                      │  │
│  │     + 0.15 × S_data                                                      │  │
│  │     + 0.20 × D_clinical_tier   ★ Renamed from A_patient                 │  │
│  │                                                                          │  │
│  │   Tier mapping:                                                          │  │
│  │      ├─ R >= 0.80: CRITICAL                                              │  │
│  │      ├─ 0.60 <= R < 0.80: HIGH                                           │  │
│  │      ├─ 0.40 <= R < 0.60: MEDIUM                                         │  │
│  │      └─ R < 0.40: LOW                                                    │  │
│  │                                                                          │  │
│  │   ★ Limitation acknowledged in paper Section 11:                         │  │
│  │      D_clinical_tier is proxy for true patient acuity (EHR future work)  │  │
│  └───────────────────────────────┬────────────────────────────────────────┘  │
│                                  │                                             │
│  ┌───────────────────────────────▼────────────────────────────────────────┐  │
│  │  [STEP 10] RISK-ADAPTIVE GATE (UPDATED — global+device multipliers)      │  │
│  │                                                                          │  │
│  │   base_threshold = 0.50                                                  │  │
│  │                                                                          │  │
│  │   multiplier_table = {                                                   │  │
│  │      ('infusion_pump', False): 0.70,    # critical, unpatchable         │  │
│  │      ('infusion_pump', True):  0.85,                                     │  │
│  │      ('monitor', False):       0.75,                                     │  │
│  │      ('monitor', True):        0.90,                                     │  │
│  │      ('ehr_workstation', False): 0.80,                                   │  │
│  │      ('ehr_workstation', True):  0.95,                                   │  │
│  │      'unknown_fallback':       0.80,    ★ Conservative                   │  │
│  │   }                                                                      │  │
│  │                                                                          │  │
│  │   IF similar_events > 5:                                                 │  │
│  │      multiplier = max(0.50, multiplier - 0.20)                           │  │
│  │                                                                          │  │
│  │   threshold = base_threshold × multiplier                                │  │
│  │   adjusted_score = min(1.0, R × risk_multiplier)                         │  │
│  │   should_surface = (adjusted_score > threshold)                          │  │
│  │                                                                          │  │
│  │   ★ INVARIANT 2: Safety floor                                            │  │
│  │   IF device_criticality == "CRITICAL" AND patchable == False:            │  │
│  │      should_surface = True   (always)                                    │  │
│  │      ★ Maintenance window SUPPRESSES_DISPLAY_NOT_DETECTION               │  │
│  │      ★ NO bypass — bug fixed                                             │  │
│  └───────────────────────────────┬────────────────────────────────────────┘  │
│                                  │                                             │
│         ┌────────────────────────┴────────────────────────┐                   │
│         │ should_surface = False                           │                   │
│         ▼                                                  │                   │
│   ┌─────────────────────────────┐                         │                   │
│   │  SUPPRESSED                  │                         │                   │
│   │  → Append-only audit log    │  ★ For forensic review  │                   │
│   │  → Periodic suppression     │                         │                   │
│   │     review process           │                         │                   │
│   └─────────────────────────────┘                         │                   │
│                                                            │                   │
│         should_surface = True                              │                   │
│  ┌─────────────────────────────────────────────────────────▼───────────────┐ │
│  │                                                                          │ │
│  │  [STEP 11] SHAP EXPLANATION (faithfulness + stability)                   │ │
│  │  ───────────────────────────                                             │ │
│  │   ├─ SHAP values from Track A (XGBoost)                                  │ │
│  │   ├─ Top-3 features extracted                                            │ │
│  │   ├─ Feature names mapped (clinician-readable)                           │ │
│  │   ├─ Stability check: ≥90% top-3 consistent across runs                  │ │
│  │   └─ Output: SHAPContext object                                          │ │
│  │                                                                          │ │
│  │  [STEP 12] MVE 3-LAYER GENERATION (≤150 words total)                     │ │
│  │  ────────────────────────────────                                        │ │
│  │   Layer 1: WHY anomalous (≤60 words)                                     │ │
│  │      ├─ References SHAP top features (faithfulness)                      │ │
│  │      ├─ References MITRE ATT&CK (threat intel grounding)                 │ │
│  │      └─ Confidence score                                                 │ │
│  │                                                                          │ │
│  │   Layer 2: CLINICAL IMPACT (≤50 words)                                   │ │
│  │      ├─ Patient care implications                                        │ │
│  │      ├─ Clinical disruption risk                                         │ │
│  │      └─ Severity tier (consistent with R)                                │ │
│  │                                                                          │ │
│  │   Layer 3: RECOMMENDED ACTION (≤60 words action + ≤30 words DO_NOT)     │ │
│  │      ├─ Action priority order (isolate → restrict → ... → log)           │ │
│  │      ├─ DO NOT constraint (clinical safety)                              │ │
│  │      └─ Required for CRITICAL on clinical devices                        │ │
│  │                                                                          │ │
│  │   Mode A (LLM-based): primary                                            │ │
│  │   Mode B (rule-based): fallback when LLM fails                           │ │
│  │      ★ UI badge: "Rule-based fallback" when Mode B active                │ │
│  │                                                                          │ │
│  │   ★ INVARIANT 5: Layer 1 references actual SHAP top features             │ │
│  │   ★ INVARIANT 7: DO_NOT present for CRITICAL on clinical devices         │ │
│  └───────────────────────────────┬────────────────────────────────────────┘  │
│                                  │                                             │
│  ┌───────────────────────────────▼────────────────────────────────────────┐  │
│  │  [STEP 13] STAKEHOLDER ADAPTATION (NEW — multi-stakeholder)              │  │
│  │                                                                          │  │
│  │   Role from session/user: IT Generalist | Biomed Engineer | Nurse       │  │
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
│  │   ★ Cross-role consistency on severity (validated via M5 study)          │  │
│  └───────────────────────────────┬────────────────────────────────────────┘  │
│                                  │                                             │
│  ┌───────────────────────────────▼────────────────────────────────────────┐  │
│  │  [STEP 14] DISPLAY + TIER RECOMMENDATION                                 │  │
│  │                                                                          │  │
│  │   Dashboard rendering:                                                   │  │
│  │      ├─ Alert card with role-appropriate view                           │  │
│  │      ├─ Tier badge: "🔵 Recommended for L2 review"                      │  │
│  │      ├─ Tooltip: tier rationale                                         │  │
│  │      ├─ "Escalate" button (manual, no auto-route)                       │  │
│  │      └─ Mode B badge if rule-based fallback active                      │  │
│  │                                                                          │  │
│  │   Tier recommendation logic (RECOMMENDATION ONLY):                       │  │
│  │      ├─ KNOWN_ATTACK → L1                                               │  │
│  │      ├─ NOVEL_ANOMALY → L2_specialist                                   │  │
│  │      └─ CONFIRMED_ANOMALY → L1_with_senior                              │  │
│  │                                                                          │  │
│  │   Hospital realities:                                                    │  │
│  │      ├─ Small hospital fallback: L1 documents NOVEL for review          │  │
│  │      ├─ External consultant path documented                             │  │
│  │      └─ NO queue separation, NO permission system                       │  │
│  │                                                                          │  │
│  │   Audit trail:                                                           │  │
│  │      ├─ Tier recommendation logged                                      │  │
│  │      └─ Actual handling logged (post-decision)                          │  │
│  └───────────────────────────────┬────────────────────────────────────────┘  │
│                                  │                                             │
│                          OPERATOR DECISION                                     │
│                                  │                                             │
│  ┌───────────────────────────────▼────────────────────────────────────────┐  │
│  │  [STEP 15] RESPONSE RECOMMENDATION (NO AUTO-EXECUTION)                   │  │
│  │                                                                          │  │
│  │   ★ INVARIANT 3: NO AUTO-EXECUTION                                       │  │
│  │      grep -rn "subprocess|os.system|iptables" pipeline/module5_response/ │  │
│  │      Expected: empty                                                     │  │
│  │                                                                          │  │
│  │   All actions surface as RECOMMENDATIONS (string output)                 │  │
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
│  │  [STEP 16] OPERATOR DECISION LOGGING (NEW — closed-loop awareness)       │  │
│  │                                                                          │  │
│  │   After alert displayed, capture:                                        │  │
│  │      ├─ alert_id, alert_type                                            │  │
│  │      ├─ recommended_action                                              │  │
│  │      ├─ operator_role                                                   │  │
│  │      ├─ operator_action_taken                                           │  │
│  │      ├─ decision_time_seconds                                           │  │
│  │      ├─ operator_confidence (1-5, optional)                             │  │
│  │      ├─ operator_rationale (free text, optional)                        │  │
│  │      └─ timestamp                                                        │  │
│  │                                                                          │  │
│  │   Storage: append-only audit log                                         │  │
│  │   Used for: forensic review, periodic analysis                           │  │
│  │                                                                          │  │
│  │   ★ INVARIANT 4: Audit trail complete                                   │  │
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
| [1] Data Preparation | `module1_preprocessing/phase1/pipeline.py` | Train/test split + SMOTE present. **Stratified eval set** is partial — see Performance_baselines.md GAP-PB-4. |
| [2] Track A Training | `module2_detection/module2_train_models.py` (`train_track_a`) | XGB/RF/DT all trained; XGBoost selected as primary (best F1, AUC 0.9941). |
| [3] Track B Training | `module2_detection/module2_train_models.py::train_track_b_dae` | Cascaded DAE input `[raw \|\| P_xgb, P_rf, P_dt]`. |
| [4] Threshold Calibration | `module2_detection/models/_threshold.py::find_optimal_threshold` | F2-optimal per-model thresholds; risk-adaptive parameters in `src/risk_scorer.py`. |
| [5] Feature Sanitization | `src/preprocessing.py::sanitize_features` (per-alert) + `module3_risk_scoring/module3_risk_scores.py::_sanitise_features` (batch) | Per-feature **BENIGN_MEDIAN** replacement (not 0.0) using `data/processed/benign_medians.json` (computed from 9990 benign training samples). `data_quality` ∈ {`OK`, `IMPUTED_NAN`, `DEGRADED`, `FAILED`} on `ScoredAlert.data_quality`. EA-06 mitigation in `src/risk_scorer.py`: `DEGRADED` ×1.20 score elevation; `FAILED` clamps `anomaly_score >= 0.95` so the alert always surfaces. Covered by `tests/test_feature_sanitization.py` (7 tests) and `results/reports/feature_sanitization.yaml`. |
| [6a] Track A inference | `module3_risk_scoring/module3_risk_scores.py::_load_track_a_probas_for_dae` | Parallel via joblib threading. |
| [6b] Track B inference | `module4_explanations/module4_online_explainer.py` (DAE branch) | Cascaded input. |
| [7] Two-Stage Fusion | `module3_risk_scoring/module3_risk_scores.py::compute_c_detect` + `::classify_fusion` | Fusion produces both `c_detect = np.maximum(c_track_a, c_track_b)` and a per-alert `fusion_class` ∈ {`KNOWN_ATTACK`, `CONFIRMED_ANOMALY`, `NOVEL_ANOMALY`, `BENIGN`}. `KNOWN_ATTACK` boundary at `P_xgb ≥ 0.85` (`P_XGB_HIGH_CONF` in `src/data_models.py`). Persisted on `ScoredAlert.fusion_class`, in `risk_scores.npz["fusion_class"]`, and in `evaluation_alerts.json`. |
| [8] Context Enrichment | `module6_evaluation/_src_adapter.py` + device fixtures | Device class + clinical tier present; threat-intel mapping is rule-based only (no live feed). |
| [9] Composite Risk Scoring | `module3_risk_scoring/module3_risk_scores.py::compute_composite_risk` | `R = 0.40·C_detect + 0.25·D_crit + 0.15·S_data + 0.20·D_clinical_tier`. |
| [10] Risk-Adaptive Gate | `src/risk_scorer.py::score_alert` | Multiplier, threshold, safety floor all enforced. **Maintenance window suppresses display, NOT detection** — CRITICAL+unpatchable always surfaces because the safety-floor check ORs into the early-return path (`tests/test_safe_failure.py::test_critical_unpatchable_surfaces_in_maintenance_window`). |
| [11] SHAP Explanation | `module4_explanations/module4_online_explainer.py::build_shap_context` | TreeSHAP + category mapping. **Stability check is not yet implemented** — `SHAPContext` does not carry a stability score. Tracked as future work. |
| [12] MVE 3-Layer Generation | `src/mve_generator.py::generate_mve` | Mode A (LLM via Anthropic API) and Mode B (rule-based) both present. |
| [13] Stakeholder Adaptation | `module5_responses/module5_pipeline.py` (notify-routing) | **Partial**: notification routing per stakeholder exists (primary IT Security, secondary Biomedical Engineering, nurse-manager paths in policy). Per-stakeholder *view rendering* is currently not differentiated — MVE is generated for the IT Generalist primary audience per CLAUDE.md, with biomed/nurse copy reached through Module 5 routing rather than dedicated views. |
| [14] Display & Audit | `module6_evaluation/module6_app.py` + `results/reports/alert_responses.json` | Streamlit dashboard for browse/study modes; audit trail captured per alert in response artifacts. |
| [15] Response Recommendation | `module5_responses/module5_pipeline.py::recommend` | NO AUTO-EXECUTION invariant verified by `tests/negative_tests.py::test_no_automated_blocking`. |
| [16] Operator Decision Logging | `module6_evaluation/module6_app.py` (study mode → `survey/study_responses_*.json`); `results/reports/alert_responses.json` for browse mode | Append-only capture of alert_id, recommended_action, operator_action_taken, decision_time, confidence, rationale, timestamp. Used today for Phase-2 study analysis (M5 Mann-Whitney). Production deployment would route the same schema to a SIEM-style audit store. |
| [17] Outcome Tracking | — | **FUTURE WORK** — requires real deployment with ground-truth feedback channel (true-positive verification, clinical follow-up). Not implemented; documented as a Phase-3 capability. |
| [18] Continuous Improvement | — | **FUTURE WORK** — feedback-into-retraining, active learning, and threshold auto-tuning are not implemented. `feedback_loop_demo.py` is a single-pass simulation, not a closed loop. |

## Design Invariants

- **Track B only elevates detection confidence**: fusion uses `max(Track_A, Track_B)`, so the DAE cannot suppress a stronger Track A signal.
- **Risk tier and surfacing are separate concerns**:
  - Module 3 assigns `risk_level` from the batch composite score `R`
  - `src/risk_scorer.py` decides `should_surface` using adaptive thresholds, patchability, and event context
- **Offline-first explanations**: the MVE generator works without API keys through deterministic rule-based fallback logic.
- **Recommendation only, no enforcement**: Module 5 produces response guidance and audit outputs, not live containment actions.
- **_src_adapter safe defaults**: `scored_from_eval_alert()` uses `patchable=True` and `event_context=None` when fields are absent in evaluation artifacts. Unknown devices are treated as low-risk for threshold purposes only.
- **study_loader determinism**: shuffle seed = `int(hashlib.md5(participant_id.encode()).hexdigest(), 16)`; A/B assignment counterbalanced by `seed % 2`.
- **Safety floor holds on all paths** (`src/risk_scorer.py`): a CRITICAL+unpatchable device always surfaces, including on the maintenance-window early-return path. The `should_surface` assignment in that branch ORs `(criticality == "CRITICAL" and not patchable)` into the threshold check. Covered by `tests/test_safe_failure.py::test_critical_unpatchable_surfaces_in_maintenance_window` and `::test_low_patchable_suppressed_in_maintenance_window`.

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

## Operational Model

The typical workflow is batch-first: preprocess data, train models, score alerts, generate explanations and response artifacts, then build evaluation outputs. The Streamlit app is best understood as a presentation and study layer on top of those generated artifacts rather than the primary computation engine.
