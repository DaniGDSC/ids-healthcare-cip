# RA-X-IoMT Security Monitoring Dashboard

Real-time IoMT intrusion detection monitoring dashboard for the
RA-X-IoMT research framework. Role-based views with MedSec-25
streaming simulation, SHAP explainability, and stakeholder-specific
cognitive translation.

## Quick Start

```bash
# From project root
cd /home/un1/project/ids-healthcare-cip

# Generate ground truth (required before first run)
python extract_project_reality.py

# Launch dashboard
streamlit run dashboard/app.py
```

The dashboard opens at <http://localhost:8501>.

## Roles

Access is role-based. Each role sees only the pages relevant to their
decision context:

| Role | Pages |
| ---- | ----- |
| IT Security Analyst | All pages (full access) |
| Clinical IT Administrator | Operational, Stakeholder, Risk, Evaluation, Model, Compliance |
| Attending Physician | Operational, Stakeholder, Risk, Devices |
| Hospital Manager | Operational, Stakeholder, Risk, Compliance |
| Regulatory Auditor | Operational, Stakeholder, Compliance |

Select your role from the sidebar dropdown.

## Pages (6 panels)

| Page | Description |
| ---- | ----------- |
| Live Monitor | System state, threat gauge, anomaly timeline, network traffic heatmap, risk distribution |
| Alert Feed | Real-time alerts with suggestions, explanations, ground truth validation |
| Explanations | SHAP/gradient feature attribution, temporal attention timeline |
| Performance | Latency histogram, throughput, rolling accuracy, model confidence |
| Stakeholder Intelligence | Role-specific views — SOC, clinician, CISO, biomed |
| System & Compliance | Model architecture, evaluation metrics, HIPAA/FDA audit |

## Stakeholder Intelligence Views

The Stakeholder Intelligence page adapts its content to the active role:

**IT Security Analyst** — Threat intelligence overview: active threat
count, novel/zero-day indicators, CIA impact breakdown, top anomaly
features, SOC playbook actions.

**Attending Physician** — Patient safety in plain language: green/red
status banners, device status ("operating normally" vs "network access
restricted"), safety guidance ("DO NOT power off — maintain patient
care"), expected resolution time.

**Hospital Manager / CISO** — Compliance and risk posture: incident
severity distribution, regulatory impact assessment (HIPAA, FDA 21 CFR
Part 11, Joint Commission), required reporting actions with timelines,
alert fatigue suppression metrics.

**Biomedical Engineer** — Device diagnostics: affected device list with
expandable details, anomaly indicator features, temporal attention
weight patterns, remediation checklists with interactive checkboxes.

## Simulation

The dashboard includes a MedSec-25 streaming simulator for live
demonstration. Controls are in the sidebar:

- **Modes:** Accelerated, Realtime, Stress
- **Scenarios:**
  - A: Benign Only (baseline)
  - B: Gradual Attack (ramp-up)
  - C: Abrupt Attack (sudden onset)
  - D: Mixed Cycle (alternating)

## Required Artifacts

The dashboard reads pre-computed artifacts from the RA-X-IoMT pipeline:

```text
project_ground_truth.json              -- Consolidated system state

data/processed/train_phase1.parquet    -- Phase 1 processed data (24 features)
data/processed/test_phase1.parquet
data/processed/preprocessing_report.json

data/phase2/detection_metadata.json    -- Phase 2 architecture metadata
data/phase2/attention_output.parquet   -- 128-D attention context vectors

data/phase2_5/finetuned_results.json   -- Phase 2.5 tuning results + metrics
data/phase2_5/finetuned_model.weights.h5
data/phase2_5/best_config.json
data/phase2_5/ablation_results.json

data/phase3/metrics_report.json        -- Phase 3 evaluation metrics

data/phase4/risk_report.json           -- Risk assessments with:
                                       --   clinical_severity, stakeholder_views,
                                       --   alert_emit, explanation, cia_scores
data/phase4/baseline_config.json       -- MAD baseline threshold
data/phase4/drift_log.csv              -- Concept drift events

data/phase5/explanation_report.json    -- SHAP feature importance
data/phase5/shap_values.parquet        -- SHAP value matrix

models/scalers/robust_scaler.pkl       -- RobustScaler (fitted on train)
```

Run `python extract_project_reality.py` to regenerate `project_ground_truth.json`
from current artifacts.

## Architecture

```text
dashboard/
  app.py                    -- Main Streamlit app (role-based routing)
  assets/
    style.css               -- Custom CSS (HIPAA banner, risk colors)
  components/
    panel_operational.py    -- Engine health
    panel_stakeholder.py    -- Cognitive translation (role-specific views)
    panel_alerts.py         -- Live alert feed
    panel_shap.py           -- SHAP explanations
    panel_evaluation.py     -- Model evaluation metrics
    panel_model.py          -- Architecture + training details
    panel_risk.py           -- Risk distribution analytics
    panel_devices.py        -- Device inventory
    panel_crossval.py       -- Cross-dataset + simulation
    panel_compliance.py     -- Compliance & audit trail
  simulation/
    medsec25_simulator.py   -- MedSec-25 stream generator
    scenarios.py            -- Attack scenario definitions
    statistics.py           -- k-schedule + drift display helpers
  streaming/
    window_buffer.py        -- Sliding window buffer for live data
    feature_aligner.py      -- Feature alignment for inference
    file_watcher.py         -- File system monitoring (watchdog)
  utils/
    loader.py               -- Cached data loaders
    metrics.py              -- Metric formatting + risk colors
```

## Tech Stack

- Streamlit >= 1.32
- TensorFlow >= 2.20.0
- Plotly >= 5.18 (interactive charts)
- Pandas >= 2.0, NumPy >= 1.24
- PyArrow >= 14.0 (Parquet I/O)
- Watchdog >= 3.0 (file monitoring)
- Joblib >= 1.3 (scaler serialization)

## HIPAA Compliance

The dashboard displays a research disclaimer banner. No Protected Health
Information (PHI) is stored or transmitted. Device IDs are SHA-256 hashed
to 8-character prefixes. Per-patient attention weights are not logged.
