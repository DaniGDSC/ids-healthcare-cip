# IoMT IDS — System Workflow

Healthcare Intrusion Detection System for IoMT (Internet of Medical Things)
networks, built on the **WUSTL-EHMS-2020** dataset (network flow features +
patient biometrics). The current implementation is a **batch research
pipeline** of seven sequential modules; there is no streaming inference
service or real-time dashboard in the active codebase.

The detection layer is **dual-track**:

- **Track A (supervised):** XGBoost / Random Forest / Decision Tree on
  SMOTE-balanced training data — produces calibrated attack probabilities.
- **Track B (novelty):** Denoising Autoencoder trained on benign-only data —
  produces a per-sample reconstruction error that flags out-of-distribution
  behavior, including attacks the supervised models miss.

The two tracks are fused into a composite risk score that incorporates
device criticality, data sensitivity, and patient acuity, then explained,
routed to a response policy, and finally evaluated.

---

## Pipeline Workflow

```
                       data/raw/WUSTL-EHMS/
                wustl-ehms-2020_with_attacks_categories.csv
                                |
                                v
              +------------------------------+
              |   MODULE 0 — EDA             |
              |   pipeline/module0_analysis  |
              |                              |
              |  - Descriptive stats         |
              |  - Class distribution        |
              |  - Correlation matrix        |
              |  - Quality + repro reports   |
              +------------------------------+
                                |
                  results/phase0_analysis/
                                |
                                v
              +------------------------------+
              |   MODULE 1 — PREPROCESSING   |
              |   pipeline/module1_*         |
              |                              |
              |  1. HIPAA identifier removal |
              |  2. Encode Dir/Flgs, parse   |
              |     Sport                    |
              |  3. Missing-value handling   |
              |     (ffill biometrics,       |
              |      fill-zero network)      |
              |  4. Variance + correlation   |
              |     redundancy removal       |
              |  --- LEAKAGE BARRIER ---     |
              |  5. Stratified 70/30 split   |
              |  6. RobustScaler             |
              |     (fit train → transform)  |
              |  7. Track A: SMOTE config    |
              |  8. Track B: benign-only     |
              |     subset                   |
              +------------------------------+
                                |
                       data/processed/
                  train_phase1.parquet
                  test_phase1.parquet
                  train_benign_phase1.parquet
                  robust_scaler.pkl
                                |
                                v
+================================================================+
|           MODULE 2 — DUAL-TRACK DETECTION                       |
|           pipeline/module2_detection                            |
|================================================================|
|                                                                  |
|   +------------------------+    +--------------------------+   |
|   |  Track A (supervised)  |    |  Track B (novelty)       |   |
|   |                        |    |                          |   |
|   |  XGBoost               |    |  Denoising Autoencoder   |   |
|   |  Random Forest         |    |  trained on              |   |
|   |  Decision Tree         |    |  train_benign_phase1     |   |
|   |                        |    |                          |   |
|   |  trained on            |    |  Output:                 |   |
|   |  SMOTE-balanced        |    |    reconstruction_error  |   |
|   |  train_phase1          |    |    per sample            |   |
|   |                        |    |                          |   |
|   |  Output:               |    |  Threshold = function    |   |
|   |    p_attack ∈ [0,1]    |    |    of benign RE dist     |   |
|   +------------------------+    +--------------------------+   |
|                                                                  |
+================================================================+
                                |
                                v
              +------------------------------+
              |   MODULE 3 — COMPOSITE RISK  |
              |   module3_risk_scores.py     |
              |                              |
              |  C_detect = max(             |
              |    Track_A_proba,            |
              |    Track_B_normalized_RE)    |
              |                              |
              |  R = w1·C_detect             |
              |    + w2·D_crit               |
              |    + w3·S_data               |
              |    + w4·A_patient            |
              |                              |
              |  Tier mapping:               |
              |    LOW / MEDIUM /            |
              |    HIGH / CRITICAL           |
              |                              |
              |  Demonstrates dual-track     |
              |  fusion value (Track A ∪ B)  |
              +------------------------------+
                                |
                  results/reports/risk_report.json
                                |
                                v
              +------------------------------+
              |   MODULE 4 — EXPLANATIONS    |
              |   module4_explanations.py    |
              |                              |
              |  Track A:                    |
              |    TreeSHAP global +         |
              |    local attributions        |
              |    (XGB / RF / DT)           |
              |                              |
              |  Track B:                    |
              |    per-feature weighted      |
              |    reconstruction-error      |
              |    decomposition             |
              |                              |
              |  Stakeholder views:          |
              |    - analyst (forensic)      |
              |    - clinician (NLG)         |
              |    - administrator (agg.)    |
              +------------------------------+
                                |
                  results/reports/{global_importance_*.json,
                                   analyst_report.json,
                                   clinician_summaries.json,
                                   admin_dashboard.json,
                                   dae_feature_errors.npz}
                                |
                                v
              +------------------------------+
              |   MODULE 5 — RESPONSES       |
              |   module5_responses.py +     |
              |   module5_pipeline.py        |
              |                              |
              |  PolicyEngine                |
              |    + adaptive mitigation     |
              |      (magnitude / device /   |
              |       attack-aware)          |
              |    + clinical safety         |
              |      override                |
              |    + simulated ActionExec    |
              |    + NotificationService     |
              |    + immutable audit log     |
              |    + feedback-loop stub      |
              |                              |
              |  Worked examples:            |
              |    CRITICAL / HIGH / LOW     |
              +------------------------------+
                                |
                  results/reports/{response_policy.json,
                                   all_responses.json,
                                   audit_log.jsonl,
                                   effectiveness_analysis.json}
                                |
                                v
              +------------------------------+
              |   MODULE 6 — EVALUATION      |
              |   module6_evaluation.py +    |
              |   module6_app.py (Streamlit) |
              |                              |
              |  Batch script:               |
              |    - curate alert set        |
              |    - participant metrics     |
              |    - thesis figures          |
              |                              |
              |  Eval app (3 modes):         |
              |    a) offline browse +       |
              |       Likert questionnaire   |
              |    b) online simulation      |
              |       (stream test rows)     |
              |    c) dashboard (gauge,      |
              |       feed, SHAP, NLG,       |
              |       response panel)        |
              +------------------------------+
                                |
                  results/reports/{evaluation_alerts.json,
                                   evaluation_results.json,
                                   participant_responses.json}
                  results/charts/*.png
```

---

## Standalone Analyses (Phase B / C)

These three scripts run independently against the trained models +
test split. They are not part of `run_all_modules.py`.

```
data/processed/test_phase1.parquet  +  trained DAE / Module 3 outputs
                              |
              +---------------+----------------+
              |               |                |
              v               v                v
   +------------------+  +------------+  +---------------+
   | drift_detection  |  | dynamic_   |  | feedback_     |
   | (PSI + KS over   |  | threshold_ |  | loop_demo     |
   |  DAE RE stream)  |  | sim        |  | (closed-loop  |
   |                  |  | (sliding W |  |  iteration)   |
   | Recalibration    |  |  median +  |  |               |
   | trigger          |  |  k·MAD     |  | C.3 single    |
   |                  |  |  adaptive) |  | C.4 multi-    |
   |                  |  |            |  |     cycle     |
   |                  |  | Static vs  |  | C.5 AUROC     |
   |                  |  | adaptive   |  |     reweight  |
   |                  |  | grid W×k   |  | C.6 adjusted  |
   |                  |  |            |  |     config    |
   +------------------+  +------------+  +---------------+
              |               |                |
              v               v                v
   drift_detection_     dynamic_threshold_  feedback_loop_
   results.json         results.json        results.json
                                            adjusted_risk_
                                            configuration.json
```

---

## Data Flow Summary

```
+---------------------+      +---------------------+      +---------------------+
| data/raw/WUSTL-EHMS | ---> | Module 0  EDA       | ---> | results/phase0_*    |
+---------------------+      +---------------------+      +---------------------+
                                                                    |
+---------------------+      +---------------------+                |
| data/processed/     | <--- | Module 1  Preproc   | <--------------+
|   train_phase1      |      +---------------------+
|   test_phase1       |
|   train_benign      |
|   robust_scaler.pkl |
+---------------------+
           |
           +-------------------------+
           |                         |
           v                         v
+---------------------+      +---------------------+
| Module 2  Track A   |      | Module 2  Track B   |
| XGB / RF / DT       |      | DAE (benign-only)   |
| → p_attack          |      | → reconstr. error   |
+---------------------+      +---------------------+
           |                         |
           +------------+------------+
                        v
              +---------------------+
              | Module 3  Composite |
              | Risk: R = Σ wi·xi   |
              +---------------------+
                        |
                        v
              +---------------------+
              | Module 4  Explain   |
              | TreeSHAP + DAE      |
              | error decomposition |
              +---------------------+
                        |
                        v
              +---------------------+
              | Module 5  Respond   |
              | PolicyEngine +      |
              | audit + actions     |
              +---------------------+
                        |
                        v
              +---------------------+
              | Module 6  Evaluate  |
              | + Streamlit app     |
              +---------------------+
```

---

## Feature Schema

After Module 1 sanitization, the WUSTL-EHMS-2020 schema is **35 network +
8 biometric features** (some are dropped further by variance / correlation
filtering — the exact final feature list is recorded in
`data/processed/selected_features.json`).

| Group | Features |
|---|---|
| **Network (35)** | `Dir`, `Flgs`, `Sport`, `Dport`, `SrcBytes`, `DstBytes`, `SrcLoad`, `DstLoad`, `SrcGap`, `DstGap`, `SIntPkt`, `DIntPkt`, `SIntPktAct`, `DIntPktAct`, `SrcJitter`, `DstJitter`, `sMaxPktSz`, `dMaxPktSz`, `sMinPktSz`, `dMinPktSz`, `Dur`, `Trans`, `TotPkts`, `TotBytes`, `Load`, `Loss`, `pLoss`, `pSrcLoss`, `pDstLoss`, `Rate`, … |
| **Biometric (8)** | `Temp`, `SpO2`, `Pulse_Rate`, `SYS`, `DIA`, `Heart_rate`, `Resp_Rate`, `ST` |
| **Labels** | `Label` (binary), `Attack Category` (multi-class) |

Identifier columns dropped by HIPAA step: `SrcAddr`, `DstAddr`, `SrcMac`,
`DstMac`, `Packet_num` (also implicated in label leakage).

---

## Eval App Roles

The Streamlit interface in `pipeline/module6_evaluation/module6_app.py`
defines **three** stakeholder roles:

| Role | View Focus |
|---|---|
| Security Analyst | Forensic detail, SHAP waterfall, response panel |
| Clinician | Plain-language NLG, biometric context, patient impact |
| Administrator | Aggregate alert tier distribution, role accuracy charts |

Available actions in the eval workflow: `dismiss`, `monitor`,
`investigate`, `isolate`, `escalate`.

This is a research evaluation interface, **not** a production RBAC dashboard.
There is no LDAP, no SSO, no per-role authorization layer in the active code.

---

## Execution

```bash
# Module 0 — EDA
python -m pipeline.module0_analysis.module0_analysis

# Module 1 — Preprocessing
python -m pipeline.module1_preprocessing.phase1

# Modules 2..6 (orchestrated)
python run_all_modules.py
python run_all_modules.py --from 3
python run_all_modules.py --only 4

# Standalone analyses
python -m pipeline.drift_detection
python -m pipeline.dynamic_threshold_sim
python -m pipeline.feedback_loop_demo

# Evaluation app
streamlit run pipeline/module6_evaluation/module6_app.py
```

---

## What is *not* in the current code

The earlier project iteration described a streaming inference service with
a CNN-BiLSTM-Attention model, FastAPI, mTLS, LDAP, Splunk/HL7 integration,
Docker Compose, and a 6-panel RBAC dashboard. **None of that lives in the
active codebase.** The relevant files are preserved under `_archive/` for
reference but are not imported, tested, or executed by the current
pipeline. If you need any of those capabilities you must port them
forward — they are out of scope for the current modules.
