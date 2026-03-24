# III. METHODOLOGY

This section presents the architecture, training methodology, and
operational mechanisms of the RA-X-IoMT framework. The system is designed
as a modular, 7-engine pipeline that processes raw IoMT traffic through
detection, classification, risk assessment, and explainable alerting —
operating under HIPAA compliance constraints at every stage.

---

## A. System Architecture Overview

RA-X-IoMT comprises seven independent engines operating in a pipeline
architecture. Each engine adheres to the Single Responsibility Principle
(SRP) and communicates via versioned file-based artifacts with SHA-256
integrity verification at every handoff boundary.

**Table I. RA-X-IoMT Engine Pipeline**

| # | Engine | Responsibility | Input | Output |
|:-:|--------|----------------|-------|--------|
| 1 | Data Processing | HIPAA de-identification, SMOTE balancing, RobustScaler normalization | Raw WUSTL-EHMS CSV | train\_phase1.parquet, test\_phase1.parquet |
| 2 | Detection | CNN-BiLSTM-Attention feature extraction (unsupervised backbone) | train\_phase1.parquet | detection\_model.weights.h5, 128-dim context vectors |
| 3 | Classification | Sigmoid binary classification + progressive unfreezing + threshold calibration | detection\_model.weights.h5 | classification\_model\_v2.weights.h5, predictions |
| 4 | Risk-Adaptive | MAD-based dynamic thresholding with time-of-day sensitivity | Anomaly scores, attention output | risk\_report.json, risk-scored predictions |
| 5 | Explanation | SHAP GradientExplainer feature attribution for non-Normal samples | classification\_model, test data | SHAP values, waterfall charts, feature importance |
| 6 | Notification | HIPAA-compliant multi-channel alert routing | risk\_report, SHAP explanations | Encrypted email, dashboard push, on-site alarms |
| 7 | Monitoring | Heartbeat-based health monitoring with 5-state FSM | All engine artifacts | Engine health status, integrity alerts |

The pipeline ensures strict data isolation: the test set is never accessed
during training (Engines 1-2), hyperparameter tuning (Engine 3), or risk
baseline computation (Engine 4). All random operations use seed = 42 for
full reproducibility.

---

## B. Data Preprocessing Pipeline (Engine 1)

Engine 1 transforms raw WUSTL-EHMS-2020 CSV data into model-ready Parquet
files through a 7-step pipeline. Design decisions are motivated by the
dual requirements of clinical data protection (HIPAA) and class imbalance
mitigation.

**Table II. Data Preprocessing Steps**

| Step | Operation | Details |
|:----:|-----------|---------|
| 1 | Integrity check | SHA-256 verification of source CSV |
| 2 | HIPAA Safe Harbor | Drop 8 quasi-identifier columns: SrcAddr, DstAddr, Sport, Dport, SrcMac, DstMac, Dir, Flgs |
| 3 | Missing value treatment | Forward-fill for biometric features; drop rows with missing network features (0 rows dropped) |
| 4 | Redundancy removal | Drop 6 features with Pearson \|r\| > 0.95: SrcJitter, pLoss, Rate, DstJitter, Loss, TotPkts |
| 5 | Stratified split | 70/30 train/test split (StratifiedShuffleSplit, seed = 42), preserving 12.5% attack prevalence |
| 6 | SMOTE oversampling | k\_neighbors = 5, strategy = auto; train: 11,422 → 19,980 samples (8,558 synthetic attack samples added) |
| 7 | RobustScaler normalization | Median/IQR scaling; fit on training partition only to prevent data leakage |

**Key design rationale:**

- **SMOTE before scaling.** SMOTE is applied in the original feature space
  to preserve inter-sample distance relationships. Applying SMOTE after
  normalization would generate synthetic samples in a distorted metric
  space, potentially degrading interpolation quality.

- **RobustScaler over StandardScaler.** The WUSTL-EHMS-2020 dataset
  contains heavy-tailed network flow features (e.g., Load, SrcLoad) with
  extreme outliers. RobustScaler (median + IQR) is resistant to outliers,
  whereas StandardScaler (mean + std) would compress the informative
  range of these features.

- **Feature retention.** After HIPAA de-identification (−8) and
  redundancy removal (−6), the pipeline retains N = 29 features:
  14 network flow features (SrcBytes, DstBytes, SrcLoad, DstLoad,
  SrcGap, DstGap, SIntPkt, DIntPkt, SIntPktAct, DIntPktAct, sMaxPktSz,
  dMaxPktSz, sMinPktSz, dMinPktSz) + 7 derived network features
  (Dur, Trans, TotBytes, Load, pSrcLoss, pDstLoss, Packet\_num) +
  8 biometric features (Temp, SpO2, Pulse\_Rate, SYS, DIA, Heart\_rate,
  Resp\_Rate, ST).

---

## C. CNN-BiLSTM-Attention Architecture (Engine 2)

The detection backbone employs a hybrid architecture that combines
convolutional feature extraction (CNN), bidirectional temporal encoding
(BiLSTM), and Bahdanau attention-based context aggregation. The
architecture is assembled from independent builder components following
the Builder design pattern, enabling ablation-study isolation of each
block.

**Table III. Model Architecture — Layer-by-Layer Specification**

| Layer | Type | Output Shape | Parameters |
|-------|------|:------------:|-----------:|
| Input | InputLayer | (None, 20, 29) | 0 |
| conv1 | Conv1D(64, 3, relu, same) | (None, 20, 64) | 5,632 |
| pool1 | MaxPooling1D(2) | (None, 10, 64) | 0 |
| conv2 | Conv1D(128, 3, relu, same) | (None, 10, 128) | 24,704 |
| pool2 | MaxPooling1D(2) | (None, 5, 128) | 0 |
| bilstm1 | Bidirectional(LSTM(128)) | (None, 5, 256) | 263,168 |
| drop1 | Dropout(0.3) | (None, 5, 256) | 0 |
| bilstm2 | Bidirectional(LSTM(64)) | (None, 5, 128) | 164,352 |
| drop2 | Dropout(0.3) | (None, 5, 128) | 0 |
| attention | BahdanauAttention(128) | (None, 128) | 16,640 |
| dense\_head | Dense(64, relu) | (None, 64) | 8,256 |
| drop\_head | Dropout(0.3) | (None, 64) | 0 |
| output | Dense(1, sigmoid) | (None, 1) | 65 |
| **Total** | — | — | **482,817** |

**Temporal windowing.** Raw feature vectors (N, 29) are reshaped into
sliding windows of shape (N\_windows, 20, 29) using a DataReshaper with
timesteps = 20 and stride = 1. Each window label is derived from the
last sample in the window (standard IDS convention), ensuring that the
model must detect attacks from preceding context rather than future
lookahead.

**CNN block.** Two Conv1D layers with doubling filter progression
(64 → 128) extract local spatial patterns from network flow features.
Kernel size k = 3 with padding = "same" preserves temporal resolution
before MaxPooling1D(2) downsamples by factor 2 at each stage, reducing
the sequence from 20 → 10 → 5 timesteps.

**BiLSTM block.** Two Bidirectional LSTM layers with halving unit
progression (128 → 64) encode forward and backward temporal dependencies.
The bidirectional design doubles the effective representation (256 → 128
output dimensions) while the halving progression acts as an information
bottleneck, forcing the network to compress temporal patterns into
increasingly abstract representations. Dropout (p = 0.3) between layers
prevents co-adaptation.

**Bahdanau attention.** A custom attention mechanism (registered as a
Keras serializable layer) computes per-timestep importance scores:

```
score_t = V^T · tanh(W · h_t)              (1)
alpha_t = softmax(score_t)                  (2)
context = sum_t(alpha_t · h_t)              (3)
```

where h\_t is the BiLSTM output at timestep t, W is a Dense(128, tanh)
score layer, V is a Dense(1) weight layer, and the softmax is applied
over the timestep axis. The resulting 128-dimensional context vector
serves as both the classification input and the interpretability anchor
for SHAP analysis (Engine 5).

---

## D. Risk-Adaptive Mechanism (Engine 4)

Engine 4 implements a dynamic, time-aware anomaly thresholding system
based on Median Absolute Deviation (MAD), providing non-parametric
robustness to distributional assumptions. Unlike fixed-threshold systems,
RA-X-IoMT adjusts its detection sensitivity based on time-of-day clinical
context.

### D.1. Dynamic Threshold Computation

The threshold at time t is computed as:

```
Threshold(t) = Median(W_t) + k(t) × MAD(W_t)            (4)
```

where W\_t is a rolling window of the 100 most recent anomaly scores,
and MAD(W\_t) = median(|W\_t − median(W\_t)|). A floor of
max(MAD, 1e-8) prevents degenerate thresholds when anomaly scores
are constant.

### D.2. Time-of-Day Sensitivity Schedule

The multiplier k(t) varies by clinical time context, reflecting the
observation that IoMT traffic patterns differ significantly across
hospital operational periods:

**Table IV. Time-of-Day k-Schedule**

| Time Window | k Value | Clinical Rationale |
|:-----------:|:-------:|:-------------------|
| 00:00 – 06:00 | 2.5 | Low traffic — reduced baseline; higher sensitivity needed to detect attacks that exploit quiet periods |
| 06:00 – 22:00 | 3.0 | Normal clinical operations — standard sensitivity balances detection and false alarm rates |
| 22:00 – 24:00 | 3.5 | Evening shift transition — elevated traffic variability from shift handoff; relaxed threshold reduces nuisance alerts |

### D.3. Risk Level Classification

Each sample is classified into one of five risk levels based on its
distance from the dynamic threshold:

**Table V. Risk Level Definitions**

| Level | Condition | Triggered Action |
|-------|-----------|------------------|
| NORMAL | distance < 0 | No action |
| LOW | 0 ≤ d < 0.5 × MAD | Audit log only |
| MEDIUM | 0.5 × MAD ≤ d < 1.0 × MAD | Dashboard notification |
| HIGH | 1.0 × MAD ≤ d < 2.0 × MAD | Encrypted email + Dashboard |
| CRITICAL | d ≥ 2.0 × MAD AND cross-modal confirmation | Immediate alarm + Escalation chain |

**Cross-modal fusion for CRITICAL.** The CRITICAL level requires both
network anomaly AND biometric anomaly to exceed 2.0σ simultaneously.
This cross-modal confirmation eliminates false CRITICAL alerts from
single-modality noise (e.g., network congestion without physiological
impact) and ensures that the highest-severity response — including
on-site alarm activation and IT admin escalation — is reserved for
genuine multi-domain intrusion signatures.

### D.4. Concept Drift Detection

Engine 4 monitors for concept drift by comparing rolling baseline
statistics against historical values. A drift event is triggered when
the baseline shift exceeds 20% (drift\_threshold = 0.20) over a 24-hour
window. Recovery requires 3 consecutive windows within the 10% recovery
threshold, preventing premature re-stabilization during sustained
distributional shifts.

---

## E. SHAP Explanation Methodology (Engine 5)

Engine 5 provides post-hoc feature attribution for every non-Normal
prediction, enabling clinicians and security analysts to understand
*why* a specific sample was flagged — a requirement for regulatory
transparency in clinical deployments.

### E.1. Explainer Configuration

- **Method:** SHAP GradientExplainer (primary), with integrated gradients
  fallback (50 interpolation steps) when GradientExplainer encounters
  numerical instability.
- **Background dataset:** 100 randomly sampled Normal-class (Label = 0)
  windows from the training set, reshaped to (100, 20, 29). This
  establishes the baseline expectation against which attributions are
  computed.
- **Explained samples:** Up to 200 non-Normal predictions from the test
  set (risk level != NORMAL as determined by Engine 4).

### E.2. Attribution Computation

For each explained sample x with background set B:

```
phi_j(x) = E_{b ~ B}[gradient_j(f(x)) × (x_j - b_j)]       (5)
```

where phi\_j is the SHAP value for feature j, and gradients are computed
via TensorFlow's GradientTape in eager mode. The resulting SHAP matrix
(N, 20, 29) provides per-timestep, per-feature importance scores.

### E.3. Feature Importance Ranking

Global feature importance is computed as the mean absolute SHAP value
across all explained samples:

```
Importance_j = (1/N) × sum_i |phi_j(x_i)|                    (6)
```

The top 10 features by this ranking are reported in each alert
explanation, enabling rapid triage by identifying which network flow
or biometric features drove the detection.

### E.4. Visualization Outputs

- **Waterfall charts:** Up to 5 charts for CRITICAL/HIGH samples,
  showing per-feature SHAP contributions to the prediction.
- **Bar charts:** Global feature importance ranking across all explained
  samples.
- **Timeline charts:** Up to 3 temporal anomaly progression visualizations
  showing how feature importances evolve across the 20-timestep window.

---

## F. Notification Engine (Engine 6)

Engine 6 routes alerts through HIPAA-compliant channels with
severity-proportional escalation.

**Table VI. Alert Routing by Risk Level**

| Risk Level | Dashboard | Email (TLS 1.3) | On-Site Alarm | Escalation Chain |
|:----------:|:---------:|:---------------:|:-------------:|:----------------:|
| LOW | — | — | — | — |
| MEDIUM | Yes | — | — | — |
| HIGH | Yes | Yes | — | — |
| CRITICAL | Yes | Yes | Yes | IT Admin → Doctor → Manager |

**HIPAA compliance controls:**

- Email bodies contain **no PHI** (no device IDs, patient data, raw
  scores, or biometric readings). Alerts reference a secure dashboard
  URL for clinical detail retrieval.
- Recipient addresses are logged as SHA-256 hashes (first 16 hex
  characters) to prevent identity exposure in audit trails.
- All email transmission requires TLS 1.3 minimum.
- CRITICAL escalation enforces a 300-second (5-minute) confirmation
  timeout per level, with 3 retries and exponential backoff before
  escalating to the next authority in the chain.
- Optional waterfall chart attachment (PNG) for HIGH/CRITICAL alerts,
  providing visual feature attribution without raw data exposure.

---

## G. Monitoring Engine (Engine 7)

Engine 7 provides continuous health monitoring across all pipeline
engines via a heartbeat-based finite state machine (FSM).

### G.1. Five-State Health Model

```
UNKNOWN ──heartbeat_received──→ STARTING
STARTING ──3 consecutive OK──→ UP
UP ←──latency_recovered──→ DEGRADED (latency_exceeded)
UP/DEGRADED ──5 missed heartbeats──→ DOWN
DOWN ──heartbeat_received──→ STARTING (recovery)
```

**Heartbeat parameters:** interval = 5 seconds, missed threshold = 5
consecutive misses (25-second grace period), latency threshold = 100 ms
(p95), consecutive OK for recovery = 3.

### G.2. Artifact Integrity Verification

Engine 7 performs periodic SHA-256 verification of all pipeline artifacts
(models, metadata, baseline configurations) at 60-second intervals. Hash
mismatches — indicating potential artifact tampering or corruption —
generate CRITICAL health alerts routed through Engine 6.

### G.3. Performance Collection

A circular buffer (capacity = 1,000 samples) collects per-engine
performance metrics at 30-second intervals: latency (p50/p95/p99),
throughput (samples/second), memory utilization (MB), and CPU percentage.
Warning thresholds are set at 80% memory and 90% CPU utilization.

---

## H. Training Methodology (Engine 3)

### H.1. Progressive Unfreezing Strategy

The classification engine employs transfer learning from the Phase 2
detection backbone with a 3-phase progressive unfreezing schedule. This
strategy prevents catastrophic forgetting of learned representations
while allowing the classification head to specialize for binary
detection.

**Table VII. Progressive Unfreezing Schedule**

| Phase | Trainable Layers | Frozen Layers | Epochs (max) | Learning Rate | Purpose |
|:-----:|:----------------:|:-------------:|:------------:|:-------------:|---------|
| A | dense\_head, drop\_head, output | CNN, BiLSTM-1, BiLSTM-2, Attention | 20 | 1 × 10⁻³ | Warm up classification head on frozen backbone features |
| B | + Attention | CNN, BiLSTM-1, BiLSTM-2 | 15 | 1 × 10⁻⁴ | Fine-tune attention weights for classification objective |
| C | + BiLSTM-2 | CNN, BiLSTM-1 | 15 | 1 × 10⁻⁵ | Fine-tune upper temporal encoder; CNN frozen to preserve low-level features |

**Training controls per phase:**
- Early stopping: patience = 5 epochs, monitor = val\_loss, restore
  best weights
- Learning rate reduction: ReduceLROnPlateau with factor = 0.5,
  patience = 3 epochs
- Batch size: 64
- Validation split: 20% of training data (stratified)
- Optimizer: Adam with per-phase learning rate

The progressive unfreezing yielded 45 total epochs across all three
phases (20 + 10 + 15), with early stopping triggered in Phase B —
indicating convergence of the attention fine-tuning after 10 of 15
scheduled epochs.

### H.2. Class Weight Computation

Class weights are derived from the **pre-SMOTE** original distribution
to compensate for residual class imbalance in the loss function:

```
w_class = N_total / (2 × N_class)                            (7)
```

| Class | N\_class | Weight | Interpretation |
|:-----:|:--------:|:------:|:--------------:|
| Normal (0) | 9,990 | 0.572 | Down-weighted — majority class |
| Attack (1) | 1,432 | 3.988 | 7.0× penalty — missed attacks are life-critical |

**Rationale for pre-SMOTE weighting.** Although SMOTE balances the
training set to 1:1, the class weights reflect the true prior
probability of attack events in clinical deployment. This dual
mechanism — SMOTE for representation + class\_weight for loss scaling —
prevents the model from exploiting the artificial balance of the
SMOTE-augmented set while maintaining sufficient minority-class gradient
signal.

### H.3. Decision Threshold Optimization

The default classification threshold (t = 0.5) is suboptimal for
imbalanced datasets where the prior probability of the positive class
differs from 50%. RA-X-IoMT calibrates the operating threshold using
**Youden's J statistic** on the ROC curve:

```
J = TPR - FPR                                                (8)
t* = argmax_t(J(t))                                          (9)
```

The optimal threshold t\* = 0.608 maximizes the separation between the
true positive rate and false positive rate, yielding
J\* = 0.577 − 0.165 = 0.412. This threshold is stored as a tunable
parameter in the classification artifact, enabling the risk-adaptive
engine (Engine 4) to further adjust the effective operating point based
on time-of-day context.

### H.4. Bayesian Hyperparameter Optimization

To validate the architectural configuration, Bayesian optimization
(keras-tuner BayesianOptimization) was conducted over the following
search space:

**Table VIII. Hyperparameter Search Space**

| Hyperparameter | Search Values | Combinations |
|----------------|:-------------:|:------------:|
| timesteps | {10, 20, 30} | 3 |
| cnn\_filters | {64, 128} | 2 |
| bilstm\_units | {64, 128, 256} | 3 |
| dropout\_rate | {0.2, 0.3, 0.5} | 3 |
| learning\_rate | {0.001, 0.0005} | 2 |
| dense\_units | {32, 64, 128} | 3 |
| **Total search space** | | **324** |

**Search configuration:**
- Objective: val\_AUC (maximize) — superior to accuracy for imbalanced
  data as it evaluates rank ordering across all thresholds
- Trials: 20 (6.2% of search space), 2 executions per trial (averaged)
- Hardware: NVIDIA RTX 4060 Laptop GPU, mixed\_float16 precision
- Training per trial: 20 epochs with early stopping (patience = 3)
- class\_weight applied in every trial
- Total search time: 3,276 seconds (~55 minutes, GPU mode)

**Result.** The Bayesian search converged on the Phase 2 default
configuration (timesteps = 20, cnn\_filters = 64, bilstm\_units = 128,
dropout\_rate = 0.3, learning\_rate = 0.001, dense\_units = 64) as optimal,
achieving val\_AUC = 0.8645. This convergence across 20 trials over a
324-configuration space provides empirical evidence that the original
architecture was well-specified, and that performance improvements
must come from training strategy (class weighting, threshold
calibration, progressive unfreezing) rather than architectural
modifications.

---

*All training experiments conducted with TensorFlow 2.20.0, Python 3.12.3,
NVIDIA RTX 4060 Laptop GPU (6,153 MB, compute capability 8.9), mixed
float16 precision, seed = 42.*
