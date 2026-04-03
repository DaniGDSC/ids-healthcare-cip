# Engine-to-Source Mapping

Maps each verification checklist engine to its source code location.

| Engine | Name | Source Folders |
|--------|------|----------------|
| 1 | Data Preprocessing | `src/phase0_dataset_analysis/`, `src/phase1_preprocessing/` |
| 2 | Detection Architecture | `src/phase2_detection_engine/` |
| 3 | Training & Fine-tuning | `src/phase2_5_fine_tuning/`, `src/phase3_classification_engine/` |
| 4 | Risk-Adaptive Scoring | `src/phase4_risk_engine/`, `src/production/score_calibrator.py` |
| 5 | Explainability | `src/phase5_explanation_engine/`, `src/production/clinical_explainer.py` |
| 6 | Alert & Dashboard | `dashboard/`, `src/production/alert_router.py` |
| 7 | Monitoring & Ops | `src/phase7_monitoring_engine/`, `src/production/security_monitor.py` |

## Engine 1 — Data Preprocessing (Checklist 1.x)

| Checklist | File | Purpose |
|-----------|------|---------|
| 1.1 Dataset | `src/phase0_dataset_analysis/phase0/loader.py` | Load WUSTL-EHMS-2020 CSV |
| 1.2 EDA | `src/phase0_dataset_analysis/phase0/eda.py` | Exploratory data analysis |
| 1.3 Cleaning | `src/phase1_preprocessing/phase1/cleaner.py` | NaN removal, deduplication |
| 1.4 Normalization | `src/phase1_preprocessing/phase1/scaler.py` | RobustScaler fitting |
| 1.5 Split | `src/phase1_preprocessing/phase1/splitter.py` | 70/15/15 stratified split |

## Engine 2 — Detection Architecture (Checklist 2.x)

| Checklist | File | Purpose |
|-----------|------|---------|
| 2.1 CNN | `src/phase2_detection_engine/phase2/cnn_builder.py` | Conv1D(64,128) spatial features |
| 2.2 BiLSTM | `src/phase2_detection_engine/phase2/bilstm_builder.py` | BiLSTM(32,16) temporal patterns |
| 2.3 Attention | `src/phase2_detection_engine/phase2/attention_builder.py` | Attention(32) weighting |
| 2.4 Feature Weight | `src/phase2_detection_engine/phase2/feature_weight.py` | 24 learnable per-feature weights |
| 2.5 Assembler | `src/phase2_detection_engine/phase2/assembler.py` | Model assembly pipeline |
| 2.6 Reshaper | `src/phase2_detection_engine/phase2/reshaper.py` | Window(20, stride=5) reshaping |

## Engine 3 — Training & Fine-tuning (Checklist 3.x)

| Checklist | File | Purpose |
|-----------|------|---------|
| 3.1 Training | `src/phase2_5_fine_tuning/run_finetuned_training.py` | 2-stage progressive unfreezing |
| 3.2 Hyperparameter | `src/phase2_5_fine_tuning/phase2_5/tuner.py` | Bayesian hyperparameter search |
| 3.3 Evaluation | `src/phase2_5_fine_tuning/phase2_5/evaluator.py` | Metrics computation |
| 3.4 Ablation | `src/phase2_5_fine_tuning/phase2_5/ablation.py` | Component ablation study |
| 3.5 Visualization | `src/phase2_5_fine_tuning/plot_training.py` | Training curve plots |

## Engine 4 — Risk-Adaptive Scoring (Checklist 4.x)

| Checklist | File | Purpose |
|-----------|------|---------|
| 4.1 Risk Scoring | `src/phase4_risk_engine/phase4/risk_scorer.py` | 5-level risk classification |
| 4.2 Attention Anomaly | `src/phase2_detection_engine/phase2/attention_anomaly.py` | MAD-based anomaly detection |
| 4.3 Calibration | `src/production/score_calibrator.py` | Platt scaling + percentile mapping |
| 4.4 CIA Modifier | `src/phase4_risk_engine/phase4/cia_modifier.py` | CIA-triad priority adjustment |
| 4.5 Clinical Impact | `src/phase4_risk_engine/phase4/clinical_impact.py` | Biometric status assessment |
| 4.6 Device Registry | `src/phase4_risk_engine/phase4/device_registry.py` | Device-specific detection profiles |
| 4.7 Alert Fatigue | `src/phase4_risk_engine/phase4/alert_fatigue.py` | Suppression + aggregation |
| 4.8 Ensemble | `src/phase4_risk_engine/phase4/ensemble_detector.py` | IsolationForest + Autoencoder |
| 4.9 Autoencoder | `src/phase4_risk_engine/phase4/autoencoder_detector.py` | Reconstruction-error detector |

## Engine 5 — Explainability (Checklist 5.x)

| Checklist | File | Purpose |
|-----------|------|---------|
| 5.1 SHAP | `src/phase5_explanation_engine/phase5/shap_explainer.py` | GradientExplainer SHAP values |
| 5.2 Benchmark | `src/phase5_explanation_engine/phase5/shap_benchmark.py` | GradientExplainer vs KernelSHAP |
| 5.3 Clinical | `src/production/clinical_explainer.py` | 7 explanation methods (NLG) |
| 5.4 Dashboard | `dashboard/components/panel_shap.py` | Role-specific SHAP rendering |

## Engine 6 — Alert & Dashboard (Checklist 6.x)

| Checklist | File | Purpose |
|-----------|------|---------|
| 6.1 Alert Routing | `src/production/alert_router.py` | Severity-based routing (5 levels) |
| 6.2 Dashboard App | `dashboard/app.py` | 6-panel Streamlit + RBAC |
| 6.3 Live Monitor | `dashboard/components/panel_live_monitor.py` | Real-time streaming display |
| 6.4 Alert Feed | `dashboard/components/panel_alerts.py` | Alert management + HITL actions |
| 6.5 Stakeholder | `dashboard/components/panel_stakeholder.py` | Role-specific intelligence views |
| 6.6 System Panel | `dashboard/components/panel_system.py` | Architecture + compliance + history |
| 6.7 Streaming | `dashboard/streaming/window_buffer.py` | Window buffer + state machine |
| 6.8 Simulation | `dashboard/simulation/wustl_simulator.py` | WUSTL-EHMS streaming simulator |

## Engine 7 — Monitoring & Operations (Checklist 7.x)

| Checklist | File | Purpose |
|-----------|------|---------|
| 7.1 Security Monitor | `src/production/security_monitor.py` | Config integrity, canary, heartbeat |
| 7.2 Feedback Loop | `src/production/feedback_loop.py` | HITL feedback + auto-recalibration |
| 7.3 Database | `src/production/database.py` | SQLite persistence (WAL mode) |
| 7.4 Auth | `src/production/auth.py` | Open / local / LDAP authentication |
| 7.5 API | `src/production/api.py` | FastAPI inference endpoints |
| 7.6 Backup | `src/production/backup.py` | Hourly automated backup (48 copies) |

## Cross-Cutting

| File | Purpose |
|------|---------|
| `config/production.yaml` | All tunable parameters |
| `config/production_loader.py` | Dot-notation config access |
| `src/production/inference_service.py` | Core inference pipeline (Tier 1/2) |
| `dashboard/streaming/feature_aligner.py` | Feature name alignment |
