# IoMT IDS — Source Code Verification Checklist

> Cross-reference of the active prototype against the framework design.
> Version 3.0 — rewritten to match the current `pipeline/moduleN_*` layout.
> See [ENGINE_MAP.md](ENGINE_MAP.md) for the module-to-source map and
> [SYSTEM_WORKFLOW.md](SYSTEM_WORKFLOW.md) for the end-to-end workflow.

## Status Legend

| Symbol | Meaning |
|---|---|
| Implemented | Code exists, runs, produces the documented artifact |
| Partial | Code exists but is a stub, dry-run, or limited form |
| Out of scope | Removed in the current iteration; preserved under `_archive/` |

---

## Module 0 — Exploratory Data Analysis

| # | Requirement | File | Status |
|---|---|---|---|
| 0.1 | Load WUSTL-EHMS-2020 CSV | `pipeline/module0_analysis/phase0/loader.py` | Implemented |
| 0.2 | Verify shape and label columns | `pipeline/module0_analysis/phase0/analyzer.py` | Implemented |
| 0.3 | Class distribution + descriptive statistics | `pipeline/module0_analysis/phase0/analyzer.py` → `results/phase0_analysis/stats_report.json` | Implemented |
| 0.4 | High-correlation feature pairs (\|r\| > 0.95) | `phase0/analyzer.py` → `results/phase0_analysis/high_correlations.csv` | Implemented |
| 0.5 | Data-quality report (Markdown) | `phase0/quality_report.py` → `results/phase0_analysis/report_section_quality.md` | Implemented |
| 0.6 | Reproducibility report (SHA-256, env) | `phase0/reproducibility_report.py` | Implemented |
| 0.7 | Configurable via YAML | `pipeline/module0_analysis/phase0/config.yaml` | Implemented |

---

## Module 1 — Preprocessing & Sanitization

### 1.1 Identifier / HIPAA Sanitization

| # | Requirement | File | Status |
|---|---|---|---|
| 1.1.1 | Drop identifier columns implicated in leakage (`SrcMac`, `DstMac`, `SrcAddr`, `DstAddr`, `Packet_num`) | `pipeline/module1_preprocessing/phase1/hipaa.py` + `phase1_config.yaml:identifier_removal` | Implemented |
| 1.1.2 | Log preprocessing steps for audit trail | `phase1/report.py` → `data/processed/phase1_report.json` | Implemented |

### 1.2 Encoding & Cleaning

| # | Requirement | File | Status |
|---|---|---|---|
| 1.2.1 | Label-encode `Dir`, `Flgs`; parse `Sport` numerically | `phase1/encoder.py` | Implemented |
| 1.2.2 | Forward-fill biometric NaNs | `phase1/missing.py` (`biometric_strategy: ffill`) | Implemented |
| 1.2.3 | Fill-zero network NaNs | `phase1/missing.py` (`network_strategy: fill_zero`) | Implemented |
| 1.2.4 | Drop unary / zero-variance features | `phase1/variance.py` | Implemented |
| 1.2.5 | Drop redundant features (\|r\| > 0.95) | `phase1/redundancy.py` | Implemented |

### 1.3 Split & Scale (post-leakage barrier)

| # | Requirement | File | Status |
|---|---|---|---|
| 1.3.1 | Stratified 70/30 train/test split on `Attack Category` | `phase1/splitter.py` | Implemented |
| 1.3.2 | RobustScaler fit on train, transform test (no leakage) | `phase1/scaler.py` → `data/processed/robust_scaler.pkl` | Implemented |
| 1.3.3 | Track A: SMOTE config (applied inside CV in Module 2) | `phase1/smote.py` + `phase1_config.yaml:track_a.smote` | Implemented |
| 1.3.4 | Track B: benign-only training subset | `phase1/exporter.py` → `data/processed/train_benign_phase1.parquet` | Implemented |
| 1.3.5 | Selected feature list saved | `data/processed/selected_features.json` | Implemented |

---

## Module 2 — Dual-Track Detection

### 2.1 Track A — Supervised Classifiers (XGBoost / RF / DT)

| # | Requirement | File | Status |
|---|---|---|---|
| 2.1.1 | XGBoost model with best HPs | `pipeline/module2_detection/models/XGBoost.py` + `tuning/run_xgboost.py` | Implemented |
| 2.1.2 | Random Forest with best HPs | `models/RandomForest.py` + `tuning/run_random_forest.py` | Implemented |
| 2.1.3 | Decision Tree with best HPs | `models/DecisionTree.py` + `tuning/run_decision_tree.py` | Implemented |
| 2.1.4 | Trained on SMOTE-balanced full training set | `module2_train_models.py` | Implemented |
| 2.1.5 | Per-model `final_report.json` + saved artifact | `module2_train_models.py` → `results/models/` | Implemented |

### 2.2 Track B — Denoising Autoencoder (Novelty Detection)

| # | Requirement | File | Status |
|---|---|---|---|
| 2.2.1 | DAE architecture | `models/DAE.py` | Implemented |
| 2.2.2 | Trained on benign-only data (no leakage from attacks) | `module2_train_models.py` (Track B path) | Implemented |
| 2.2.3 | Hyperparameter search | `tuning/run_dae.py` | Implemented |
| 2.2.4 | Per-sample reconstruction error available downstream | `module2_train_models.py` → `data/phase2/dae/` | Implemented |

### 2.3 Reproducibility

| # | Requirement | File | Status |
|---|---|---|---|
| 2.3.1 | Fixed seed (`random_state: 42`) | `phase2_5_config.yaml` + `module2_train_models.py` | Implemented |
| 2.3.2 | Best hyperparameters frozen in config | `pipeline/module2_detection/phase2_5_config.yaml` | Implemented |

> **Out of scope:** the previous CNN-BiLSTM-Attention deep model is not part
> of the current pipeline. Earlier reports of `AUC=0.904 / F1=0.968` came
> from that architecture; the active code reports tree-based + DAE metrics
> in `results/models/*/final_report.json`.

---

## Module 3 — Composite Risk Scoring

| # | Requirement | File | Status |
|---|---|---|---|
| 3.1 | Fuse Track A + Track B into a single detection score | `pipeline/module3_risk_scoring/module3_risk_scores.py` (`C_detect = max(p_attack, normalized_RE)`) | Implemented |
| 3.2 | Weighted composite: `R = w1·C_detect + w2·D_crit + w3·S_data + w4·A_patient` | `module3_risk_scores.py` | Implemented |
| 3.3 | Device criticality table | `results/reports/device_criticality.json` | Implemented |
| 3.4 | Data sensitivity table | `results/reports/data_sensitivity.json` | Implemented |
| 3.5 | Tier mapping (LOW / MEDIUM / HIGH / CRITICAL) | `module3_risk_scores.py` | Implemented |
| 3.6 | Demonstrate dual-track fusion value (cases caught by ∪ but not by either alone) | `module3_risk_scores.py` → `results/reports/risk_report.json` | Implemented |

---

## Module 4 — Explanations

### 4.1 Track A — TreeSHAP

| # | Requirement | File | Status |
|---|---|---|---|
| 4.1.1 | Global feature importance (XGBoost / RF / DT) | `pipeline/module4_explanations/module4_explanations.py` → `global_importance_*.json` | Implemented |
| 4.1.2 | Per-attack-category importance | `module4_explanations.py` → `per_category_importance_*.json` | Implemented |
| 4.1.3 | Local SHAP attributions per test prediction | `module4_explanations.py` | Implemented |
| 4.1.4 | Beeswarm plots saved | `results/charts/beeswarm_{xgboost,random_forest,decision_tree}.png` | Implemented |

### 4.2 Track B — DAE Reconstruction Error Decomposition

| # | Requirement | File | Status |
|---|---|---|---|
| 4.2.1 | Per-feature weighted RE per sample | `module4_explanations.py` → `results/reports/dae_feature_errors.npz` | Implemented |
| 4.2.2 | Component breakdown chart | `results/charts/component_breakdown.png` | Implemented |

### 4.3 Stakeholder Outputs

| # | Requirement | File | Status |
|---|---|---|---|
| 4.3.1 | Security analyst forensic report | `analyst_report.json` | Implemented |
| 4.3.2 | Clinician plain-language summaries (NLG) | `clinician_summaries.json` + `nlg_templates.json` | Implemented |
| 4.3.3 | Administrator aggregate dashboard data | `admin_dashboard.json` | Implemented |
| 4.3.4 | Worked example explanations | `example_explanations.json` | Implemented |
| 4.3.5 | Online (per-alert) explainer for Module 6 simulation | `module4_online_explainer.py` → `online_sample_explanations.json`, `online_latency_profile.json` | Implemented |

---

## Module 5 — Closed-Loop Response Engine

### 5.1 Adaptive Response Engine

| # | Requirement | File | Status |
|---|---|---|---|
| 5.1.1 | Magnitude-aware mitigation selection | `pipeline/module5_responses/module5_responses.py` | Implemented |
| 5.1.2 | Device-constrained responses (safety-critical protection) | `module5_responses.py` | Implemented |
| 5.1.3 | Attack-category-aware escalation routing | `module5_responses.py` | Implemented |
| 5.1.4 | FDA-style audit trail with simulated outcomes | `module5_responses.py` → `audit_trail.json` | Implemented |
| 5.1.5 | Closed-loop effectiveness analysis | `module5_responses.py` → `effectiveness_analysis.json` | Implemented |

### 5.2 Response Pipeline (Tasks 5.1–5.8)

| # | Requirement | File | Status |
|---|---|---|---|
| 5.2.1 | Standalone `response_policy.json` | `module5_pipeline.py` | Implemented |
| 5.2.2 | `PolicyEngine` class | `module5_pipeline.py` | Implemented |
| 5.2.3 | Clinical safety override + confirmation request | `module5_pipeline.py` | Implemented |
| 5.2.4 | Simulated `ActionExecutor` with audit trail | `module5_pipeline.py` | Implemented |
| 5.2.5 | `NotificationService` per stakeholder | `module5_pipeline.py` | Implemented |
| 5.2.6 | Immutable JSONL audit logger | `module5_pipeline.py` → `results/reports/audit_log.jsonl` | Implemented |
| 5.2.7 | End-to-end CRITICAL / HIGH / LOW worked examples | `module5_pipeline.py` | Implemented |
| 5.2.8 | Feedback-loop stub | `module5_pipeline.py` (full demo lives in `pipeline/feedback_loop_demo.py`) | Implemented |

---

## Module 6 — Evaluation Artifacts + Eval App

### 6.1 Batch Evaluation Builder

| # | Requirement | File | Status |
|---|---|---|---|
| 6.1.1 | Curate evaluation alert set | `pipeline/module6_evaluation/module6_evaluation.py` → `evaluation_alerts.json` | Implemented |
| 6.1.2 | Compute metrics from participant responses (or simulated) | `module6_evaluation.py` → `evaluation_results.json` | Implemented |
| 6.1.3 | Thesis-ready figures | `results/charts/*.png` (accuracy by role / tier, comparison, cumulative F1, …) | Implemented |
| 6.1.4 | Evaluation protocol document | `evaluation_protocol.md` | Implemented |

### 6.2 Streamlit Eval App

| # | Requirement | File | Status |
|---|---|---|---|
| 6.2.1 | Three roles: Security Analyst, Clinician, Administrator | `module6_app.py:ROLES` | Implemented |
| 6.2.2 | Mode 6.3a — offline browse + Likert questionnaire | `module6_app.py` | Implemented |
| 6.2.3 | Mode 6.3b — online simulation streaming test rows | `module6_app.py` (auto-refresh) | Implemented |
| 6.2.4 | Mode 6.3c — dashboard (gauge, alert feed, SHAP waterfall, NLG, response panel, admin heatmap, tier distribution) | `module6_app.py` | Implemented |
| 6.2.5 | Action set: dismiss / monitor / investigate / isolate / escalate | `module6_app.py:ACTIONS` | Implemented |

> **Not implemented in the current code:** LDAP / SSO authentication, mTLS,
> 5-role enterprise RBAC, FastAPI inference endpoints, Splunk HEC, HL7v2
> bridge, Docker Compose deployment. These are out of scope for the
> research pipeline. See `_archive/` for the previous attempt at any of
> the above.

---

## Standalone Analyses (Phase B / C)

### B1 / B3 — Dynamic Thresholds

| # | Requirement | File | Status |
|---|---|---|---|
| B1.1 | Sort test set by time proxy (row index) | `pipeline/dynamic_threshold_sim.py` | Implemented |
| B1.2 | Sliding-window median / MAD on benign RE | `dynamic_threshold_sim.py` | Implemented |
| B1.3 | Adaptive `threshold_t = median + k·MAD` | `dynamic_threshold_sim.py` | Implemented |
| B1.4 | Static vs adaptive comparison over the stream | `dynamic_threshold_sim.py` → `dynamic_threshold_results.json` | Implemented |
| B1.5 | W × k sensitivity grid | `dynamic_threshold_sim.py` | Implemented |
| B1.6 | Comparison figures | `results/charts/adaptive_tier_thresholds.png`, etc. | Implemented |
| B3.1 | Adaptive risk-tier thresholds via rolling percentiles | `dynamic_threshold_sim.py` | Implemented |
| B3.2 | Static vs adaptive tier comparison | `dynamic_threshold_sim.py` | Implemented |

### B2 — Drift Detection

| # | Requirement | File | Status |
|---|---|---|---|
| B2.1 | PSI calculator | `pipeline/drift_detection.py` | Implemented |
| B2.2 | Kolmogorov–Smirnov detector | `drift_detection.py` | Implemented |
| B2.3 | Run drift detection over the test stream | `drift_detection.py` → `drift_detection_results.json` | Implemented |
| B2.4 | Simulated recalibration trigger | `drift_detection.py` | Implemented |
| B2.5 | Drift figures | `results/charts/` (PSI / KS plots) | Implemented |

### C — Feedback Loop

| # | Requirement | File | Status |
|---|---|---|---|
| C.3 | Single feedback iteration (before / after) | `pipeline/feedback_loop_demo.py` | Implemented |
| C.4 | Multi-iteration convergence (5 cycles, FPR/FNR plots) | `feedback_loop_demo.py` | Implemented |
| C.5 | AUROC-based component reweighting | `feedback_loop_demo.py` | Implemented |
| C.6 | Adjusted-config export + thesis tables | `feedback_loop_demo.py` → `adjusted_risk_configuration.json`, `feedback_recommendations.json` | Implemented |

---

## Reproducibility & Environment

| # | Requirement | File | Status |
|---|---|---|---|
| R.1 | `requirements.txt` / `pyproject.toml` | repo root | Implemented |
| R.2 | Python 3.10+ | `pyproject.toml` | Implemented |
| R.3 | Fixed random seeds across modules | `phase1_config.yaml`, `phase2_5_config.yaml`, `phase0/config.yaml` | Implemented |
| R.4 | SHA-256 of raw dataset | `results/phase0_analysis/` reproducibility report | Implemented |
| R.5 | Orchestrator script | `run_all_modules.py` | Implemented |

---

## Outputs Index

| Directory | Contents |
|---|---|
| `data/processed/` | Module 1 parquet splits, scaler, feature list, phase-1 report |
| `results/phase0_analysis/` | Module 0 stats, correlations, quality + repro reports |
| `results/models/` | Trained models + per-model `final_report.json` |
| `results/reports/` | Module 3 / 4 / 5 / 6 JSON artifacts, audit logs, evaluation results, drift / threshold / feedback outputs |
| `results/charts/` | All thesis figures (PNG) |

---

## Out of Scope (was in earlier docs)

The following items were claimed by previous versions of this checklist
but are **not present in the active codebase**:

- CNN-BiLSTM-Attention model + 477K-parameter architecture
- Streaming inference service (`inference_service.py`, `WUSTLFlowSimulator`, window buffer, state machine)
- 6-panel Streamlit RBAC dashboard with 5 enterprise roles
- FastAPI `/health`, `/health/detailed`, `/metrics` endpoints
- mTLS, LDAP / Active Directory authentication, Splunk HEC, QRadar syslog
- HL7v2 ORU^R01 biometric bridge
- Docker Compose deployment (`deploy/docker-compose.yml`)
- Hourly SQLite backup service
- Prometheus / Grafana metrics

These were part of an earlier production-oriented iteration. Their source
files are preserved under `_archive/` but are not imported or executed by
the current modules. Treat them as design history, not as current
verification scope.
