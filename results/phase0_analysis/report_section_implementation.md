# V. IMPLEMENTATION

This section documents the end-to-end implementation pipeline of
RA-X-IoMT, including data flow across all eight processing phases,
training fixes applied during iterative development, cross-dataset
validation findings, and the artifact integrity chain that ensures
reproducibility and tamper detection.

---

## A. Pipeline Overview

RA-X-IoMT is implemented as a sequential pipeline of eight phases
(0 through 7), each producing versioned artifacts with SHA-256
integrity verification. Phases communicate exclusively through
file-based artifacts — no shared in-memory state exists between
engines — enabling independent deployment, testing, and auditing
of each component.

**Table I. Implementation Pipeline**

| Phase | Name | Input | Output | Key Artifact |
|:-----:|------|-------|--------|--------------|
| 0 | Data Analysis | Raw WUSTL-EHMS CSV (16,318 × 45) | stats\_report.json, analysis\_report.json | dataset\_integrity.json (SHA-256 of source) |
| 1 | Preprocessing | Raw CSV | train\_phase1.parquet (19,980 × 30), test\_phase1.parquet (4,896 × 30) | robust\_scaler.pkl |
| 2 | Detection | train\_phase1.parquet | detection\_model.weights.h5 (474,496 params) | attention\_output.parquet (19,961 × 128) |
| 2.5 | Bayesian Tuning | train\_phase1.parquet | best\_hyperparams\_v2.json | ablation\_results\_v2.json |
| 3 | Classification | detection\_model.weights.h5 | classification\_model\_v2.weights.h5 (482,817 params) | metrics\_wustl\_v2.json |
| 4 | Risk-Adaptive | attention\_output.parquet, full model | baseline\_config.json, risk\_report.json (4,877 assessments) | risk\_metadata.json |
| 5 | Explanation | risk\_report.json, full model, test data | shap\_values.parquet (198 × 20 × 29), 9 charts | explanation\_report.json |
| 6 | Notification | explanation\_report.json, risk\_report.json | notification\_log.json (3,337 entries) | delivery\_report.json, escalation\_log.json |
| 7 | Monitoring | All engine artifacts | monitoring\_log.json (65 state transitions) | security\_audit\_log.json (428 verification events) |

### A.1. Phase 0 — Data Analysis

Exploratory analysis of the raw WUSTL-EHMS-2020 CSV (16,318 rows,
45 columns) produces descriptive statistics, distribution profiles,
and correlation analysis. The source file SHA-256 hash
(`8359da96...6cf2c25`) is recorded in `dataset_integrity.json` and
verified at every subsequent pipeline invocation.

### A.2. Phase 1 — Preprocessing

The 7-step preprocessing pipeline applies HIPAA Safe Harbor
de-identification (−8 quasi-identifier columns), redundancy removal
(−6 features with |r| > 0.95), stratified 70/30 train-test split,
SMOTE oversampling (k = 5, 11,422 → 19,980 training samples), and
RobustScaler normalization (fit on training data only). The pipeline
retains 29 features: 14 network flow, 7 derived network, and
8 biometric.

### A.3. Phase 2 — Detection Engine

The CNN-BiLSTM-Attention backbone (Conv1D(64) → MaxPool(2) →
Conv1D(128) → MaxPool(2) → BiLSTM(128) → BiLSTM(64) →
BahdanauAttention(128)) is trained as an unsupervised feature
extractor, outputting 128-dimensional context vectors. The detection
model produces no classification head — the final layer is
BahdanauAttention, verified by integrity assertion
(`has_head=False`). Output shapes are verified: train context
(19,961 × 128), test context (4,877 × 128), with attention weights
confirmed to sum to 1.0 for all samples.

### A.4. Phase 2.5 — Bayesian Hyperparameter Tuning

Bayesian optimization via keras-tuner explores 20 trials (2 executions
each) over a 324-configuration search space, optimizing val\_AUC.
The search converged on the Phase 2 default configuration
(val\_AUC = 0.8645), confirming architectural optimality. An ablation
study (CNN / CNN+BiLSTM / Full) quantified each component's
contribution.

### A.5. Phase 3 — Classification Engine

The detection backbone is extended with a classification head
(Dense(64, relu) → Dropout(0.3) → Dense(1, sigmoid)) and trained
via progressive unfreezing (3 phases, 45 total epochs) with
class\_weight = {0: 0.572, 1: 3.988}. Decision threshold is
calibrated to t\* = 0.608 via Youden's J statistic.

### A.6. Phase 4 — Risk-Adaptive Engine

The MAD-based dynamic threshold engine computes anomaly baselines
from 9,972 Normal-class attention vectors: median = 0.129,
MAD = 0.025, baseline threshold = 0.204 (k = 3.0). Test-set
assessment (4,877 samples) produces the following risk distribution:

| Risk Level | Count | Percentage |
|:----------:|------:|-----------:|
| NORMAL | 3,261 | 66.9% |
| LOW | 93 | 1.9% |
| MEDIUM | 85 | 1.7% |
| HIGH | 1,376 | 28.2% |
| CRITICAL | 62 | 1.3% |

One concept drift event was detected during the evaluation window.

### A.7. Phase 5 — Explanation Engine

SHAP GradientExplainer computes feature attributions for 198
non-Normal samples (of 200 max) using 100 Normal-class training
windows as background. The resulting SHAP matrix
(198 × 20 × 29) enables per-timestep, per-feature importance
ranking. Nine visualization artifacts are generated: 1 global
feature importance bar chart, 5 waterfall charts (CRITICAL/HIGH
samples), and 3 temporal anomaly timeline charts. Computation
completed in 21.68 seconds.

### A.8. Phase 6 — Notification Engine

Alert routing processes 1,616 non-Normal predictions through
severity-proportional channels. The delivery report tracks
per-alert, per-channel success status. CRITICAL alerts trigger
the full escalation chain: on-site alarm → IT admin dashboard →
IT admin email → doctor dashboard → manager email, with UUID-based
confirmation tokens and 5-minute timeout per escalation level.
All 12 upstream artifact hashes are verified (Phase 2–5) before
alert generation.

### A.9. Phase 7 — Monitoring Engine

Continuous health monitoring tracks 5 engines (Phase 2–6) via
heartbeat-based state machine. All engines transition
UNKNOWN → STARTING → UP within the startup sequence. The security
audit log records 428 SHA-256 verification events across all
pipeline artifacts, with all verifications reporting OK status.
Latency measurements range from 22.8 ms to 146.7 ms (p95 threshold:
100 ms).

---

## B. Training Fixes Applied

The classification engine underwent iterative diagnosis and repair.
Table II documents each problem identified, the fix applied, and
the measured impact on held-out test-set performance.

**Table II. Training Fix Progression**

| # | Problem Identified | Root Cause | Fix Applied | Impact (Test Set) |
|:-:|-------------------|------------|-------------|-------------------|
| 1 | Underfitting — loss did not converge | Insufficient training epochs (5 per phase), aggressive early stopping (patience = 3) | Increased epochs: 5 → 20/15/15 per phase; patience: 3 → 5 | Loss converged; 45 total epochs trained (early stopping triggered in Phase B at epoch 10) |
| 2 | Attack class collapse — 12.1% attack recall | No class weighting; model learns to predict majority class (Normal = 87.5%) | class\_weight computed from pre-SMOTE distribution: w\_normal = 0.572, w\_attack = 3.988 (7.0× penalty) | Attack recall: 12.1% → 57.7% (+377%); AUC: 0.6114 → 0.7243 (+18.5%) |
| 3 | Threshold bias — default t = 0.5 suboptimal for 12.5% attack prevalence | Decision threshold assumes balanced prior; actual test prior is 6.97:1 Normal-to-Attack | Youden's J statistic optimization: t\* = argmax(TPR − FPR) = 0.608 | Maximized J = 0.412 (TPR = 0.577, FPR = 0.165); accuracy improved from 68.5% (at t = 0.5) to 80.2% |
| 4 | Hyperparameter uncertainty — are Phase 2 defaults optimal? | No systematic exploration of the 324-configuration architectural space | Bayesian optimization: 20 trials × 2 executions, objective = val\_AUC | Confirmed defaults optimal (val\_AUC = 0.8645); no architectural change needed — performance gains came from training strategy |

**Cumulative impact:**

| Metric | v1 (Baseline) | v2 (All Fixes) | Delta |
|--------|:-------------:|:--------------:|:-----:|
| AUC-ROC | 0.6114 | 0.7243 | +18.5% |
| Attack Recall | 12.1% | 57.7% | +377% |
| Accuracy | 83.4% | 80.2% | −3.8% |
| F1 (weighted) | 0.8135 | 0.8233 | +1.2% |
| Threshold | 0.5 (default) | 0.608 (optimized) | — |

**Note on accuracy.** The accuracy decrease (83.4% → 80.2%) is a
designed tradeoff: the v1 model achieved high accuracy by predicting
nearly all samples as Normal (naive baseline = 87.5% accuracy). The
v2 model sacrifices 3.8% overall accuracy to detect 4.8× more attacks
— a favorable exchange in IoMT environments where missed intrusions
carry life-critical consequences.

---

## C. Cross-Dataset Validation Summary

Cross-dataset validation was conducted on the CICIoMT2024 dataset
(1,612,117 samples) to assess model generalizability beyond the
WUSTL-EHMS-2020 training distribution. This evaluation was
discontinued as a primary metric due to fundamental incompatibilities
between the two datasets.

### C.1. Feature Alignment

Of the 29 features required by the WUSTL-trained model, only
**5 features** (17.2%) could be semantically mapped from CICIoMT2024:

| WUSTL Feature | CICIoMT2024 Mapping | Variance |
|:-------------:|:-------------------:|---------:|
| dur | duration | 21.58 |
| load | rate | 1.77 × 10⁹ |
| srcload | srate | 1.77 × 10⁹ |
| totbytes | tot\_sum | 1.01 × 10⁶ |
| packet\_num | number | 0.67 |

The remaining **24 features** (82.8%) — including all 8 biometric
features (Temp, SpO2, Pulse\_Rate, SYS, DIA, Heart\_rate, Resp\_Rate,
ST), 11 network flow features, and 5 derived features — were imputed
using WUSTL Normal-class medians. One additional mapping (dstload →
drate) was excluded due to zero variance across all CICIoMT2024
samples, reflecting IoMT unidirectional traffic patterns.

### C.2. Class Distribution Mismatch

| | WUSTL-EHMS-2020 | CICIoMT2024 |
|--|:---------------:|:-----------:|
| Normal | 87.5% | 2.3% |
| Attack | 12.5% | 97.7% |
| Total samples | 16,318 | 1,612,117 |

The CICIoMT2024 dataset is 97.7% attack traffic — the inverse of
WUSTL's 87.5% Normal distribution. This extreme mismatch renders
cross-dataset accuracy comparisons meaningless, as the prior
probability assumptions encoded in the model's class\_weight and
decision threshold are violated.

### C.3. Cross-Dataset Results

| Metric | WUSTL Test | CICIoMT2024 |
|--------|:----------:|:-----------:|
| AUC-ROC | 0.7243 | 0.489 |
| Accuracy | 80.2% | 5.8% |
| Attack Recall | 57.7% | 3.8% |

The CICIoMT2024 AUC of 0.489 (below the random baseline of 0.5)
confirms that the WUSTL-trained model does not generalize to this
dataset. The primary failure modes are:

1. **Feature nullification.** With 24/29 features imputed as
   constants, the model effectively operates on a 5-feature
   subspace — insufficient for learned pattern recognition.
2. **Distribution inversion.** The model's decision boundary,
   calibrated for 12.5% attack prevalence, cannot accommodate a
   97.7% attack distribution without threshold re-calibration.
3. **Attack taxonomy divergence.** WUSTL-EHMS-2020 captures
   spoofing, injection, and probing attacks on vital-sign monitors;
   CICIoMT2024 targets a broader IoMT device taxonomy with
   different protocol-level attack signatures.

**Conclusion.** Results on WUSTL-EHMS-2020 remain the primary and
sole valid evaluation. Cross-dataset generalization in IoMT intrusion
detection requires domain-specific transfer learning, feature-space
alignment, and attack taxonomy harmonization — identified as future
work directions.

### C.4. Alignment Validation

All 8 alignment checks passed, confirming that the imputation
procedure did not introduce structural errors:

| Validation Check | Status |
|:----------------:|:------:|
| Exactly 29 features | PASS |
| Column names match WUSTL | PASS |
| Column order matches WUSTL | PASS |
| No NaN values | PASS |
| Label values binary | PASS |
| Mapped features have variance | PASS |
| Imputed features zero variance | PASS |
| dstload uses median | PASS |

---

## D. Reproducibility Statement

All experiments are fully reproducible under the following conditions:

**Random state.** All stochastic operations — data splitting,
SMOTE oversampling, weight initialization, dropout masking, Bayesian
search — use `random_state = 42` (NumPy, TensorFlow, scikit-learn).

**Software environment.**
- Python 3.12.3
- TensorFlow 2.20.0 (with CUDA 12 via pip: nvidia-cudnn-cu12-9.20.0)
- scikit-learn (StratifiedShuffleSplit, SMOTE, RobustScaler)
- keras-tuner 1.4.8 (BayesianOptimization)
- SHAP (GradientExplainer)
- Platform: Linux 6.17.0-19-generic, x86\_64

**Hardware.**
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU (6,153 MB, compute
  capability 8.9)
- Mixed precision: float16 (enabled for Bayesian tuning;
  classification training used float32 on CPU)

**Dataset integrity.**

| Artifact | SHA-256 |
|----------|---------|
| Raw WUSTL-EHMS-2020 CSV | `8359da96154fa60247df9d75e52d232077acb9886c09c36de2a0aaaee6cf2c25` |
| detection\_model.weights.h5 | `5aa425ca8129f3bca59e9c479d23d455664073267aec7e184455ade8373c1b91` |
| attention\_output.parquet | `33aeabba393332acb287b8ff8e37db10e81f3fd23451ee1cc44ecb647515b0dd` |
| classification\_model.weights.h5 (v1) | `0add48352ce5c231f0951c85ba0b56029df304723816ba10c9e5fbeb81cd99cf` |
| baseline\_config.json | `14f4d0acaca313b6b7e75cf60c1be89090e5f4aff1fd208d18764b2e7cf845df` |
| risk\_report.json | `16a3e744c14c180b5a923ae10ba943a3a9897d854d20d642a51b9f095f145f5f` |
| shap\_values.parquet | `2399f61f46fab98b5d7b77741418562f7a53e753e6a784e1f190ff2d05954eec` |
| explanation\_report.json | `f70bb0af5da48645b2bc5d53c74fb2cedb01638ace01b78e73322b19cb21949d` |

**Git provenance.**

| Phase | Git Commit |
|:-----:|:----------:|
| 2 (Detection) | `d19d8923fd0c4459` |
| 3 (Classification) | `e3f560caa780da02` |
| 4 (Risk-Adaptive) | `3259f3eee118ce19` |
| 5 (Explanation) | `901983542dbc6112` |

---

## E. Artifact Integrity Chain

RA-X-IoMT implements a cryptographic chain-of-custody across all
pipeline phases. Each phase computes SHA-256 hashes of its output
artifacts and records them in a metadata JSON file. Downstream
phases verify upstream hashes before processing, ensuring that any
artifact tampering — accidental or adversarial — is detected before
it propagates through the pipeline.

### E.1. Integrity Chain Diagram

```
Raw WUSTL-EHMS CSV
  │  SHA-256: 8359da96...6cf2c25
  │
  ▼
[Phase 0] Data Analysis
  │  → dataset_integrity.json (hash verified)
  │  → stats_report.json, analysis_report.json
  │
  ▼
[Phase 1] Preprocessing
  │  → train_phase1.parquet (19,980 × 30)
  │  → test_phase1.parquet (4,896 × 30)
  │  → robust_scaler.pkl
  │
  ▼
[Phase 2] Detection Engine
  │  → detection_model.weights.h5
  │     SHA-256: 5aa425ca...73c1b91
  │  → attention_output.parquet
  │     SHA-256: 33aeabba...15b0dd
  │
  ▼
[Phase 2.5] Bayesian Tuning
  │  → best_hyperparams_v2.json (val_AUC = 0.8645)
  │  → ablation_results_v2.json
  │
  ▼
[Phase 3] Classification Engine
  │  ← Loads detection_model.weights.h5 (SHA-256 verified)
  │  → classification_model_v2.weights.h5
  │  → metrics_wustl_v2.json (AUC = 0.7243)
  │
  ▼
[Phase 4] Risk-Adaptive Engine
  │  ← Verifies Phase 2-3 artifact hashes
  │  → baseline_config.json
  │     SHA-256: 14f4d0ac...845df
  │  → risk_report.json (4,877 assessments)
  │     SHA-256: 16a3e744...145f5f
  │
  ▼
[Phase 5] Explanation Engine
  │  ← Verifies Phase 2-4 artifact hashes
  │  → shap_values.parquet (198 × 20 × 29)
  │     SHA-256: 2399f61f...954eec
  │  → explanation_report.json
  │     SHA-256: f70bb0af...21949d
  │
  ▼
[Phase 6] Notification Engine
  │  ← Verifies ALL 12 upstream artifact hashes (Phase 2-5)
  │  → notification_log.json (3,337 entries)
  │  → delivery_report.json
  │  → escalation_log.json
  │
  ▼
[Phase 7] Monitoring Engine
     ← Continuous SHA-256 verification (60s cycle)
     → security_audit_log.json (428 events, all OK)
     → monitoring_log.json (5 engines tracked)
```

### E.2. Verification Coverage

| Verification Point | Artifacts Checked | Frequency |
|:------------------:|:-----------------:|:---------:|
| Phase 3 model load | detection\_model.weights.h5 | Once (at startup) |
| Phase 6 pre-alert | 12 artifacts (Phase 2–5) | Once (before alert generation) |
| Phase 7 continuous | All pipeline artifacts | Every 60 seconds |
| Phase 7 baseline | baseline\_config.json | Every 30 seconds |

### E.3. Tamper Detection Guarantees

The integrity chain provides the following guarantees:

1. **Source data provenance.** The raw CSV SHA-256 hash is recorded at
   Phase 0 and verified at Phase 1. Any modification to the source
   dataset — even a single byte — will produce a hash mismatch and
   halt the pipeline.

2. **Model weight integrity.** Detection and classification model
   weights are hash-verified before inference. A compromised model
   file (e.g., adversarial weight injection) will be detected before
   any predictions are generated.

3. **Continuous monitoring.** Phase 7 re-verifies all artifact hashes
   at 60-second intervals during runtime. Hash mismatches generate
   CRITICAL health alerts routed through the notification engine
   (Phase 6), ensuring that post-deployment tampering triggers
   immediate incident response.

4. **Audit trail completeness.** The security audit log
   (428 verification events) provides a timestamped, immutable record
   of every hash check performed during pipeline execution —
   satisfying regulatory audit requirements for clinical software
   (FDA 21 CFR Part 11, HIPAA Security Rule §164.312).

---

*Implementation validated on Ubuntu 24.04 LTS, TensorFlow 2.20.0,
Python 3.12.3. All SHA-256 hashes verified against stored baselines
with zero mismatches across 428 audit events.*
