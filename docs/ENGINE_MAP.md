# Module-to-Source Mapping

Maps each pipeline module to its source code location. The current codebase
is organized as **seven sequential batch modules** under `pipeline/`, plus a
small set of standalone analysis scripts. There is no `src/phaseN_*` layout
and no production inference service — those belong to an earlier architecture
preserved under `_archive/` for reference only.

## Top-Level Layout

| Module | Name | Source |
|---|---|---|
| 0 | Exploratory Data Analysis | `pipeline/module0_analysis/` |
| 1 | Preprocessing & Sanitization | `pipeline/module1_preprocessing/` |
| 2 | Dual-Track Detection Training | `pipeline/module2_detection/` |
| 3 | Composite Risk Scoring | `pipeline/module3_risk_scoring/` |
| 4 | Stakeholder Explanations | `pipeline/module4_explanations/` |
| 5 | Closed-Loop Response Engine | `pipeline/module5_responses/` |
| 6 | Evaluation Artifacts + Eval App | `pipeline/module6_evaluation/` |
| — | Standalone analyses (drift / threshold / feedback) | `pipeline/*.py` |
| — | Orchestrator | `run_all_modules.py` |

## Module 0 — Exploratory Data Analysis

| File | Purpose |
|---|---|
| `pipeline/module0_analysis/module0_analysis.py` | Top-level EDA driver |
| `pipeline/module0_analysis/phase0/loader.py` | Load WUSTL-EHMS-2020 CSV |
| `pipeline/module0_analysis/phase0/analyzer.py` | Descriptive stats, distributions, correlations |
| `pipeline/module0_analysis/phase0/quality_report.py` | Data-quality report (Markdown) |
| `pipeline/module0_analysis/phase0/reproducibility_report.py` | SHA-256, environment capture |
| `pipeline/module0_analysis/phase0/exporter.py` | Write artifacts to `results/phase0_analysis/` |
| `pipeline/module0_analysis/phase0/security.py` | Path validation / safety checks |
| `pipeline/module0_analysis/phase0/config.py` + `config.yaml` | Module 0 configuration |

**Outputs:** `results/phase0_analysis/{stats_report.json, high_correlations.csv, correlation_matrix.parquet, report_section_dataset.md, report_section_quality.md}`

## Module 1 — Preprocessing & Sanitization

| File | Purpose |
|---|---|
| `pipeline/module1_preprocessing/phase1/__main__.py` | Entry point (`python -m pipeline.module1_preprocessing.phase1`) |
| `pipeline/module1_preprocessing/phase1/pipeline.py` | Orchestrates the preprocessing steps below |
| `pipeline/module1_preprocessing/phase1/hipaa.py` | Drop identifier columns (SrcMac, DstMac, SrcAddr, DstAddr, Packet_num) |
| `pipeline/module1_preprocessing/phase1/encoder.py` | Label-encode `Dir`, `Flgs`; parse `Sport` |
| `pipeline/module1_preprocessing/phase1/missing.py` | Forward-fill biometrics, fill-zero network |
| `pipeline/module1_preprocessing/phase1/variance.py` | Drop unary / zero-variance features |
| `pipeline/module1_preprocessing/phase1/redundancy.py` | Drop |corr| > 0.95 features |
| `pipeline/module1_preprocessing/phase1/splitter.py` | Stratified 70/30 split on `Attack Category` |
| `pipeline/module1_preprocessing/phase1/scaler.py` | RobustScaler fit on train, transform test |
| `pipeline/module1_preprocessing/phase1/smote.py` | Track A SMOTE (used inside CV in Module 2) |
| `pipeline/module1_preprocessing/phase1/exporter.py` | Write parquet + scaler + report |
| `pipeline/module1_preprocessing/phase1/report.py` | Phase-1 JSON report |
| `pipeline/module1_preprocessing/phase1_config.yaml` | Module 1 configuration |

**Outputs:** `data/processed/{train_phase1.parquet, test_phase1.parquet, train_benign_phase1.parquet, robust_scaler.pkl, selected_features.json, phase1_report.json}`

## Module 2 — Dual-Track Detection Training

The detection layer uses **two complementary tracks**:

- **Track A (supervised):** XGBoost, Random Forest, Decision Tree on the SMOTE-balanced training set.
- **Track B (novelty):** Denoising Autoencoder (DAE) trained on benign-only data; flags anomalies via reconstruction error.

| File | Purpose |
|---|---|
| `pipeline/module2_detection/module2_train_models.py` | Retrain final models with best HPs |
| `pipeline/module2_detection/models/XGBoost.py` | Track A model wrapper |
| `pipeline/module2_detection/models/RandomForest.py` | Track A model wrapper |
| `pipeline/module2_detection/models/DecisionTree.py` | Track A model wrapper |
| `pipeline/module2_detection/models/DAE.py` | Track B denoising autoencoder |
| `pipeline/module2_detection/tuning/run_xgboost.py` | CV hyperparameter tuning (Track A) |
| `pipeline/module2_detection/tuning/run_random_forest.py` | CV tuning |
| `pipeline/module2_detection/tuning/run_decision_tree.py` | CV tuning |
| `pipeline/module2_detection/tuning/run_dae.py` | DAE hyperparameter search (Track B) |
| `pipeline/module2_detection/phase2_5_config.yaml` | Best-HP and training configuration |

**Outputs:** trained models + per-model `final_report.json` artifacts under `results/models/` and `data/phase2/`.

> ⚠️ **No CNN-BiLSTM-Attention model exists in the current code.** That architecture lived in the previous research version (now under `_archive/`). The active detection layer is classical ML + a denoising autoencoder.

## Module 3 — Composite Risk Scoring

Single entry point: `pipeline/module3_risk_scoring/module3_risk_scores.py`.

Computes:

```
R = w1 · C_detect + w2 · D_crit + w3 · S_data + w4 · A_patient
```

where `C_detect = max(Track_A_proba, Track_B_normalized_RE)` fuses the two
detection tracks. Maps the fused score to alert tiers (LOW / MEDIUM / HIGH / CRITICAL)
and demonstrates dual-track value (cases where one track catches what the other misses).

**Outputs:** `results/reports/{risk_report.json, device_criticality.json, data_sensitivity.json}`

## Module 4 — Stakeholder Explanations

| File | Purpose |
|---|---|
| `pipeline/module4_explanations/module4_explanations.py` | Batch explanations for every test prediction |
| `pipeline/module4_explanations/module4_online_explainer.py` | Online (per-alert) explainer used by Module 6 simulation |

- **Track A** (XGBoost/RF/DT): TreeSHAP global + local feature attributions.
- **Track B** (DAE): per-feature weighted reconstruction-error decomposition.
- **Stakeholder views**: security analyst (forensic), clinician (plain-language NLG), administrator (aggregate dashboard data).

**Outputs:** `results/reports/{global_importance_*.json, per_category_importance_*.json, dae_feature_errors.npz, analyst_report.json, clinician_summaries.json, admin_dashboard.json, example_explanations.json, nlg_templates.json, feature_concepts.json}`

## Module 5 — Closed-Loop Response Engine

| File | Purpose |
|---|---|
| `pipeline/module5_responses/module5_responses.py` | Adaptive mitigation + audit trail + effectiveness analysis |
| `pipeline/module5_responses/module5_pipeline.py` | PolicyEngine class, clinical safety override, ActionExecutor, NotificationService, immutable audit, feedback-loop stub |

**Outputs:** `results/reports/{response_policy.json, all_responses.json, audit_log.jsonl, audit_trail.json, effectiveness_analysis.json, alert_responses.json, alert_responses_detail.csv}`

## Module 6 — Evaluation Artifacts + Eval App

| File | Purpose |
|---|---|
| `pipeline/module6_evaluation/module6_evaluation.py` | Curate alert set, compute participant metrics, generate thesis figures |
| `pipeline/module6_evaluation/module6_app.py` | Streamlit evaluation interface (offline study / online simulation / dashboard) |

**Roles in the eval app:** Security Analyst, Clinician, Administrator (3 roles).
**Modes:** offline browse with Likert questionnaires; online simulation streaming test rows through the trained models; dashboard with risk gauge, alert feed, SHAP waterfall, NLG, response panel.

**Outputs:** `results/reports/{evaluation_alerts.json, evaluation_results.json, participant_responses.json, evaluation_protocol.md}` and figures in `results/charts/`.

## Standalone Analyses (Phase B / C deliverables)

| File | Purpose |
|---|---|
| `pipeline/drift_detection.py` | PSI + Kolmogorov–Smirnov drift detection on the DAE RE stream; recalibration trigger; figures |
| `pipeline/dynamic_threshold_sim.py` | Sliding-window adaptive vs static threshold comparison for DAE RE and Module-3 risk tiers; W×k sensitivity grid |
| `pipeline/feedback_loop_demo.py` | Closed-loop feedback iteration: single + multi-cycle convergence, AUROC-based weight adjustment, adjusted config export |

**Outputs:** `results/reports/{drift_detection_results.json, dynamic_threshold_results.json, feedback_loop_results.json, feedback_recommendations.json, feedback_analysis.json, adjusted_risk_configuration.json}`

## Orchestrator

`run_all_modules.py` runs Modules 2 → 6 in sequence. Modules 0 and 1 are
expected to have been executed beforehand (their artifacts under
`data/processed/` and `results/phase0_analysis/` are inputs to Module 2).

```bash
python run_all_modules.py            # run modules 2..6
python run_all_modules.py --from 3   # resume from module 3
python run_all_modules.py --only 4   # run only module 4
```
