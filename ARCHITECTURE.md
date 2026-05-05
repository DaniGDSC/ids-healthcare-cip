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
   `R = 0.40*C_detect + 0.25*D_crit + 0.15*S_data + 0.20*A_patient`
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

## Design Invariants

- **Track B only elevates detection confidence**: fusion uses `max(Track_A, Track_B)`, so the DAE cannot suppress a stronger Track A signal.
- **Risk tier and surfacing are separate concerns**:
  - Module 3 assigns `risk_level` from the batch composite score `R`
  - `src/risk_scorer.py` decides `should_surface` using adaptive thresholds, patchability, and event context
- **Offline-first explanations**: the MVE generator works without API keys through deterministic rule-based fallback logic.
- **Recommendation only, no enforcement**: Module 5 produces response guidance and audit outputs, not live containment actions.
- **_src_adapter safe defaults**: `scored_from_eval_alert()` uses `patchable=True` and `event_context=None` when fields are absent in evaluation artifacts. Unknown devices are treated as low-risk for threshold purposes only.
- **study_loader determinism**: shuffle seed = `int(hashlib.md5(participant_id.encode()).hexdigest(), 16)`; A/B assignment counterbalanced by `seed % 2`.
- **KNOWN ISSUE — Safety floor bypass** (`src/risk_scorer.py` lines 117–127): The maintenance-window early-return can produce `should_surface=False` for a CRITICAL+unpatchable device when `is_maintenance_window=True`, `is_known_vendor_ip=True`, and `anomaly_score ≤ 1.0`. The safety floor at line 155 is unreachable from that path. Pending fix: add `or (criticality == "CRITICAL" and not patchable)` to the `should_surface` assignment at line 124, and add a covering test.

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
