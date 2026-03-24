# RA-X-IoMT Security Monitoring Dashboard

Real-time IoMT intrusion detection monitoring dashboard for the
RA-X-IoMT research framework. Built with Streamlit for thesis defense
live demonstration.

## Quick Start

```bash
# From project root
cd /home/un1/project/ids-healthcare-cip

# Install dependencies
pip install -r dashboard/requirements.txt

# Run dashboard
streamlit run dashboard/app.py
```

The dashboard opens at http://localhost:8501.

## Pages

| Page | Description |
|------|-------------|
| Live Monitoring | Engine health cards, risk donut chart, alert feed |
| Model Performance | AUC/F1 metrics, ROC curve, confusion matrix, threshold slider |
| SHAP Explanations | Global feature importance, per-alert waterfall, temporal timeline |
| Upload & Predict | CSV upload with HIPAA check, inference, risk scoring, CSV export |
| Dataset Overview | Class distribution, ablation study, preprocessing pipeline |

## Required Artifacts

The dashboard reads pre-computed artifacts from the RA-X-IoMT pipeline:

```
data/phase3/metrics_wustl_v2.json      — Final test metrics
data/phase3/metrics_report.json        — V1 baseline metrics
data/phase3/threshold_analysis.json    — Threshold sensitivity
data/phase3/ablation_results_v2.json   — Ablation study
data/phase3/diagnosis_after.json       — Training diagnosis
data/phase3/classification_model_v2.weights.h5
data/phase4/risk_report.json           — Risk assessments
data/phase4/baseline_config.json       — MAD baseline
data/phase5/explanation_report.json    — SHAP explanations
data/phase5/shap_values.parquet        — SHAP value matrix
data/phase7/monitoring_log.json        — Engine health log
data/processed/preprocessing_report.json
models/scalers/robust_scaler.pkl
```

## Tech Stack

- Streamlit >= 1.32
- TensorFlow 2.20.0
- Plotly (interactive charts)
- SHAP (explainability)
- Watchdog (file monitoring)
