# Deployment & Execution Guide

This document describes how to **run** the current IoMT IDS research
pipeline. The pipeline is a **batch research workflow**, not a production
service: there is no API, no Docker image, no SIEM bridge, no live network
TAP. It produces JSON / parquet artifacts and figures from the
WUSTL-EHMS-2020 dataset, plus an interactive Streamlit evaluation app for
human-subject studies.

If you are looking for the previous "production" deployment guide
(FastAPI + Streamlit RBAC dashboard + Docker Compose + mTLS + LDAP +
Splunk + HL7), see the *Out of Scope* section at the end. Those
components are not part of the active codebase.

---

## 1. Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10+ (`pyproject.toml: requires-python = ">=3.10"`) |
| Disk | ~2 GB free for `data/processed/` + `results/` artifacts |
| RAM | 8 GB recommended (XGBoost + DAE training) |
| OS | Linux / macOS (developed on Linux 6.x) |
| GPU | Optional — DAE training is small enough to run on CPU |

No external services are required: no databases, no message brokers,
no SIEM, no certificate infrastructure.

---

## 2. Installation

```bash
git clone <repo-url> ids-healthcare-cip
cd ids-healthcare-cip

python3.10 -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
```

---

## 3. Dataset

The pipeline expects the WUSTL-EHMS-2020 raw CSV at:

```
data/raw/WUSTL-EHMS/wustl-ehms-2020_with_attacks_categories.csv
```

`data/raw/**` is in `.gitignore` and is **not** distributed with the
repository. Download the dataset from the original source (Washington
University's WUSTL-EHMS-2020 publication) and place the CSV at the path
above. The expected schema is 35 network features + 8 biometric features
+ `Label` + `Attack Category`.

The path is configurable in:

- `pipeline/module0_analysis/phase0/config.yaml` → `dataset.data_path`
- `pipeline/module1_preprocessing/phase1_config.yaml` → `data.input_dir`

After Module 0 finishes, the SHA-256 of the raw CSV is recorded in
`results/phase0_analysis/` for reproducibility.

---

## 4. Execution Order

Modules 0 and 1 must run first (they produce the inputs that Modules
2–6 consume). Modules 2–6 can then be run together via `run_all_modules.py`.

### 4.1 Module 0 — Exploratory Data Analysis

```bash
python -m pipeline.module0_analysis.module0_analysis
```

Produces: `results/phase0_analysis/{stats_report.json, high_correlations.csv, correlation_matrix.parquet, report_section_dataset.md, report_section_quality.md}`

### 4.2 Module 1 — Preprocessing

```bash
python -m pipeline.module1_preprocessing.phase1
```

Produces: `data/processed/{train_phase1.parquet, test_phase1.parquet, train_benign_phase1.parquet, robust_scaler.pkl, selected_features.json, phase1_report.json}`

### 4.3 Modules 2–6 — Orchestrated

```bash
# Full run
python run_all_modules.py

# Resume from a specific module
python run_all_modules.py --from 3

# Run only one module
python run_all_modules.py --only 4
```

The orchestrator runs each module as a subprocess and stops on first
non-zero exit. Default starting module is 2.

| Module | Purpose | Script |
|---|---|---|
| 2 | Train XGBoost / RF / DT (Track A) + DAE (Track B) | `pipeline/module2_detection/module2_train_models.py` |
| 3 | Composite risk score `R = Σ wi·xi` | `pipeline/module3_risk_scoring/module3_risk_scores.py` |
| 4 | TreeSHAP + DAE error decomposition + stakeholder outputs | `pipeline/module4_explanations/module4_explanations.py` |
| 5 | Adaptive response engine + audit trail | `pipeline/module5_responses/module5_responses.py` |
| 5b | PolicyEngine + clinical safety + feedback-loop stub | `pipeline/module5_responses/module5_pipeline.py` |
| 6 | Curate eval alerts + thesis figures | `pipeline/module6_evaluation/module6_evaluation.py` |

### 4.4 Standalone Analyses (Phase B / C)

These are independent of `run_all_modules.py` and assume Modules 2–3
have already produced their outputs.

```bash
python -m pipeline.drift_detection         # PSI + KS over DAE RE stream
python -m pipeline.dynamic_threshold_sim   # static vs adaptive threshold sweep
python -m pipeline.feedback_loop_demo      # closed-loop feedback iteration
```

### 4.5 Evaluation App (Streamlit)

```bash
streamlit run pipeline/module6_evaluation/module6_app.py
```

Three modes:

1. **Offline browse + Likert** — page through pre-computed alerts and
   capture human-subject responses.
2. **Online simulation** — stream test rows through the trained models
   and explainer in near-real-time.
3. **Dashboard** — risk gauge, alert feed, SHAP waterfall, NLG panel,
   response panel, admin heatmap, tier distribution.

Three roles: Security Analyst, Clinician, Administrator.
Five available actions: `dismiss`, `monitor`, `investigate`, `isolate`,
`escalate`. There is no LDAP, no SSO, no per-user authorization layer —
the role selector is a self-declared dropdown for the study participant.

---

## 5. Output Layout

```
data/
  raw/                          # ungitted raw CSV (you provide)
  processed/                    # Module 1 outputs (parquet, scaler, …)
  phase2/                       # Module 2 model artifacts
results/
  phase0_analysis/              # Module 0 stats + reports
  models/                       # Trained final models + final_report.json
  reports/                      # Modules 3 / 4 / 5 / 6 + Phase B/C JSON
  charts/                       # Thesis figures (PNG)
```

Notable artifacts:

| Path | Producer |
|---|---|
| `results/reports/risk_report.json` | Module 3 |
| `results/reports/global_importance_{xgboost,random_forest,decision_tree}.json` | Module 4 |
| `results/reports/dae_feature_errors.npz` | Module 4 |
| `results/reports/{analyst_report,clinician_summaries,admin_dashboard}.json` | Module 4 |
| `results/reports/{response_policy,all_responses,audit_log.jsonl}` | Module 5 |
| `results/reports/{evaluation_alerts,evaluation_results,participant_responses}.json` | Module 6 |
| `results/reports/{drift_detection,dynamic_threshold,feedback_loop}_results.json` | Phase B / C |
| `results/reports/adjusted_risk_configuration.json` | Phase C feedback loop |

---

## 6. Configuration Surface

Each module owns its own YAML config; there is no monolithic
`production.yaml`.

| File | Module | Notable keys |
|---|---|---|
| `pipeline/module0_analysis/phase0/config.yaml` | 0 | `dataset.data_path`, `analysis.correlation_threshold`, `analysis.random_state` |
| `pipeline/module1_preprocessing/phase1_config.yaml` | 1 | `identifier_removal.remove_columns`, `cleaning.biometric_strategy`, `splitting.train_ratio`, `track_a.smote.*` |
| `pipeline/module2_detection/phase2_5_config.yaml` | 2 | Best HPs per model, training settings, seed |

Module 3, 4, 5, 6 are currently configured via constants inside their
entry-point scripts. Tuning Module 3 weights (`w1..w4`) means editing
`pipeline/module3_risk_scoring/module3_risk_scores.py` directly.

---

## 7. Reproducibility

- Random seed `42` is set in every module config (`phase0/config.yaml`,
  `phase1_config.yaml`, `phase2_5_config.yaml`) and propagated through
  splitter, SMOTE, model training.
- Module 0 records the SHA-256 of the raw CSV.
- Module 1 records column lists and split ratios in `phase1_report.json`.
- Module 2 freezes best hyperparameters in `phase2_5_config.yaml`.
- A clean run from raw CSV to all artifacts should be byte-stable
  module-by-module given identical input + identical environment.

---

## 8. Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| `FileNotFoundError: data/raw/WUSTL-EHMS/...csv` | Raw dataset missing | Download CSV (see §3) |
| Module 2 fails: parquet missing | Module 1 not run | `python -m pipeline.module1_preprocessing.phase1` |
| Module 3+ fails: model artifacts missing | Module 2 not run / failed | `python run_all_modules.py --only 2` |
| Streamlit app: alerts list empty | Module 6 evaluation builder not run | `python run_all_modules.py --only 6` |
| Drift / threshold / feedback script fails | Phase 2/3 outputs missing | Run Modules 2 + 3 first |

For module-internal failures, each module logs to stdout with
`%(levelname)s` formatting; rerun with `PYTHONUNBUFFERED=1` for
realtime logs.

---

## 9. Out of Scope

The following capabilities were described in earlier deployment guides
but are **not present in the active codebase**. Source files (where they
existed) are preserved under `_archive/` for design history.

- **Production inference service** — FastAPI app, `/health` endpoints,
  Uvicorn workers, circuit breakers, request validation
- **Streaming pipeline** — `WUSTLFlowSimulator`, `KafkaFlowConsumer`,
  window buffer, state machine (INIT → CALIBRATING → OPERATIONAL → ALERT)
- **CNN-BiLSTM-Attention model** — 477K-parameter deep model with
  GradientExplainer and progressive unfreezing fine-tuning
- **6-panel enterprise dashboard with 5-role RBAC** — replaced by the
  3-role research evaluation app in Module 6
- **mTLS** — certificate generation, server / client cert provisioning
- **LDAP / Active Directory** authentication
- **Splunk HEC / QRadar syslog** alert forwarding
- **HL7v2 ORU^R01** biometric bridge
- **Docker Compose** deployment with `api`, `dashboard`, `backup` services
- **SQLite persistence + hourly automated backup**
- **Prometheus `/metrics` + Grafana dashboards**
- **HIPAA / FDA 21 CFR Part 11** audit logger with HMAC chain (the
  current `audit_log.jsonl` from Module 5 is a simulated audit trail
  for thesis evaluation, not a production-grade compliance log)

If you need any of these for an actual hospital deployment, you must
build them on top of the current pipeline — they are not provided.

---

## 10. Project Status

This codebase is a **research / thesis prototype**. It is suitable for:

- Reproducing the published metrics on WUSTL-EHMS-2020
- Running human-subject evaluation studies via the Streamlit app
- Extending the dual-track detection design (Track A supervised +
  Track B novelty)
- Drift / adaptive-threshold / feedback-loop ablation experiments

It is **not** suitable for:

- Direct deployment on a live hospital network
- Real-time detection on production traffic
- Multi-tenant or multi-user environments
- Compliance certification (HIPAA, FDA, GDPR) without significant
  additional engineering

Treat all metrics, alerts, and audit artifacts as research outputs.
