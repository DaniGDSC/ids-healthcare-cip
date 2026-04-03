# IoMT IDS System Algorithm & Data Workflow

## System Overview

Healthcare Intrusion Detection System for IoMT (Internet of Medical Things) networks.
CNN-BiLSTM-Attention model (477K params) processes sliding windows of 20 network flows
(24 features each) to detect anomalies, classify risk, and generate clinical alerts.

---

## Algorithm Workflow

```
                        WUSTL Flow Data
                    (flows.npy, labels.npy)
                             |
                             v
              +------------------------------+
              |   1. STREAMING SIMULATOR     |
              |   wustl_simulator.py         |
              |                              |
              |  - Memory-mapped I/O (0.001ms)|
              |  - Sequential file ordering  |
              |  - Random jitter (10-50ms)   |
              |  - Per-flow ground truth     |
              +------------------------------+
                             |
                     vec[24], gt_label
                             |
                             v
              +------------------------------+
              |   2. WINDOW BUFFER           |
              |   window_buffer.py           |
              |                              |
              |  deque[maxlen=5000]           |
              |  Sliding window: last 20     |
              |  Output: (1, 20, 24)         |
              |                              |
              |  State machine:              |
              |  INIT -> CALIBRATE -> OPS    |
              |   <20    20-199     >=200    |
              +------------------------------+
                             |
                    window (1, 20, 24)
                             |
                             v
+================================================================+
|                    INFERENCE SERVICE                             |
|                    inference_service.py                          |
|================================================================|
|                                                                  |
|   +--------------------------------------------------------+   |
|   |  3. CNN-BiLSTM-ATTENTION MODEL                         |   |
|   |  @tf.function (compiled, single graph execution)       |   |
|   |                                                         |   |
|   |  Input (1, 20, 24)                                     |   |
|   |    |                                                    |   |
|   |    v                                                    |   |
|   |  Conv1D(64) -> MaxPool -> Conv1D(128) -> MaxPool       |   |
|   |    |                                                    |   |
|   |    v                                                    |   |
|   |  BiLSTM(128) -> Dropout -> BiLSTM(64)                  |   |
|   |    |                                                    |   |
|   |    v                                                    |   |
|   |  BahdanauAttention(128) -> context vector              |   |
|   |    |                                                    |   |
|   |    v                                                    |   |
|   |  Dense(32, relu) -> Dropout -> Dense(1, sigmoid)       |   |
|   |    |                                                    |   |
|   |    +---> score (raw sigmoid)                           |   |
|   |    +---> backbone (attention context vector)           |   |
|   |    +---> gradients (d_score/d_input)                   |   |
|   +--------------------------------------------------------+   |
|                          |                                       |
|           score, attn_magnitude, gradients                       |
|                          |                                       |
|                          v                                       |
|   +--------------------------------------------------------+   |
|   |  4. ATTENTION ANOMALY DETECTION                        |   |
|   |                                                         |   |
|   |  attn_magnitude = ||backbone||  (L2 norm)              |   |
|   |  threshold = median + 3 * MAD                          |   |
|   |                                                         |   |
|   |  attn_flag = (attn_magnitude > threshold)              |   |
|   |  If True: potential novel/zero-day threat              |   |
|   +--------------------------------------------------------+   |
|                          |                                       |
|                          v                                       |
|   +--------------------------------------------------------+   |
|   |  5. SCORE CALIBRATION                                  |   |
|   |  score_calibrator.py                                   |   |
|   |                                                         |   |
|   |  Auto-fits after 200 flows (calibration phase):        |   |
|   |                                                         |   |
|   |  IF n_attack >= 5 AND Youden's J > 0.3:               |   |
|   |    Two-Class Mode:                                     |   |
|   |    1. ROC-optimal threshold (Youden's J)               |   |
|   |    2. Platt scaling (LogReg, C=1e4)                    |   |
|   |    3. Target FPR ~10% (P90 of benign probs)            |   |
|   |    score -> Platt prob -> risk_level                   |   |
|   |                                                         |   |
|   |  ELSE (fallback):                                      |   |
|   |    Benign-Only Mode:                                   |   |
|   |    score -> percentile of benign dist -> risk_level    |   |
|   |    <P75: NORMAL | <P85: LOW | <P93: MEDIUM             |   |
|   |    <P97: HIGH   | >=P97: CRITICAL                      |   |
|   +--------------------------------------------------------+   |
|                          |                                       |
|                    risk_level                                    |
|                          |                                       |
|                          v                                       |
|   +--------------------------------------------------------+   |
|   |  6. ATTENTION ESCALATION                               |   |
|   |                                                         |   |
|   |  IF attn_flag AND risk in {NORMAL, LOW}:               |   |
|   |    -> Escalate to MEDIUM                               |   |
|   |    (Novel attack: model uncertain but attention         |   |
|   |     diverges from baseline)                             |   |
|   +--------------------------------------------------------+   |
|                          |                                       |
+================================================================+
                           |
                     risk_level
                           |
                           v
              +------------------------------+
              |  7. CIA RISK MODIFICATION    |
              |  cia_risk_modifier.py        |
              |                              |
              |  Scenario detection:         |
              |  - HIGH_THREAT:              |
              |    attn_flag OR HIGH/CRIT    |
              |    Weights: I>A>C            |
              |  - CLINICAL_EMERGENCY:       |
              |    risk == MEDIUM            |
              |    Weights: A>I>C            |
              |  - NORMAL_MONITORING:        |
              |    otherwise                 |
              |    Weights: I>=C>A           |
              |                              |
              |  CIA score per dimension:    |
              |  score[d] = threat[d]        |
              |           x device[d]        |
              |           x scenario[d]      |
              |                              |
              |  IF max(CIA) >= 0.7:         |
              |    -> Escalate risk by 1     |
              |  (CIA can only escalate)     |
              +------------------------------+
                           |
                   adjusted_risk_level
                           |
                           v
              +------------------------------+
              |  8. CLINICAL IMPACT          |
              |  clinical_impact.py          |
              |                              |
              |  Risk -> Severity:           |
              |  NORMAL  -> ROUTINE (1)      |
              |  LOW     -> ADVISORY (2)     |
              |  MEDIUM  -> URGENT (3)       |
              |  HIGH    -> EMERGENT (4)     |
              |  CRITICAL-> CRITICAL (5)     |
              |                              |
              |  Biometric check:            |
              |  For each vital sign:        |
              |    IF |value| > 1.5 sigma    |
              |    -> flag abnormal          |
              |                              |
              |  Escalation:                 |
              |  IF safety_device AND        |
              |     abnormal_vitals AND      |
              |     risk != NORMAL:          |
              |    -> severity += 1          |
              |                              |
              |  Patient safety flag:        |
              |  IF safety_device AND        |
              |     (HIGH/CRIT OR attn_flag) |
              |    -> flag = True            |
              |                              |
              |  Response protocol:          |
              |  ROUTINE:  log, 0 min        |
              |  ADVISORY: review, 8 hrs     |
              |  URGENT:   restrict, 60 min  |
              |  EMERGENT: isolate, 15 min   |
              |  CRITICAL: isolate, 5 min    |
              +------------------------------+
                           |
                  clinical_severity
                           |
                           v
              +------------------------------+
              |  9. ALERT FATIGUE MGMT      |
              |  alert_fatigue.py            |
              |                              |
              |  Decision rules (ordered):   |
              |                              |
              |  1. ROUTINE -> SUPPRESS      |
              |     (never emit)             |
              |                              |
              |  2. ESCALATION -> EMIT       |
              |     (severity increased)     |
              |                              |
              |  3. DE-ESCALATION -> EMIT    |
              |     (signal recovery)        |
              |                              |
              |  4. RATE LIMIT -> SUPPRESS   |
              |     (>5 alerts per 100 flows)|
              |                              |
              |  5. AGGREGATION:             |
              |     1st -> EMIT              |
              |     Every 10th -> EMIT       |
              |     2-9th -> SUPPRESS        |
              +------------------------------+
                           |
                   alert_emit (T/F)
                           |
                           v
              +------------------------------+
              |  10. CONDITIONAL EXPLANATION |
              |  conditional_explainer.py    |
              |                              |
              |  Policy by severity:         |
              |                              |
              |  ROUTINE/ADVISORY:           |
              |    -> NONE (0ms)             |
              |                              |
              |  URGENT:                     |
              |    -> LIGHTWEIGHT (~1ms)     |
              |    Attention weights only    |
              |    timestep_importance[20]   |
              |                              |
              |  EMERGENT/CRITICAL:          |
              |    -> FULL (~5-10ms)         |
              |    Attention weights +       |
              |    Gradient attribution:     |
              |    feat_imp = mean(|grads|)  |
              |    top_5 features ranked     |
              +------------------------------+
                           |
                    explanation{}
                           |
                           v
              +------------------------------+
              |  11. RESULT ASSEMBLY         |
              |                              |
              |  result = {                  |
              |    anomaly_score,            |
              |    risk_level,               |
              |    attention_flag,           |
              |    clinical_severity,        |
              |    patient_safety_flag,      |
              |    response_time_minutes,    |
              |    device_action,            |
              |    cia_scores {C, I, A},     |
              |    explanation {},            |
              |    alert_emit,               |
              |    ground_truth,             |
              |    latency_ms,               |
              |  }                           |
              +------------------------------+
                           |
                           v
              +------------------------------+
              |  12. BUFFER RECORDING        |
              |  window_buffer.py            |
              |                              |
              |  - Store in predictions[500] |
              |  - Update risk_counts        |
              |  - Update detection_counts   |
              |    (TP/FN/FP/TN cumulative)  |
              |  - If HIGH/CRITICAL:         |
              |    -> Add to alerts[200]     |
              |  - If CRITICAL:              |
              |    -> State = ALERT          |
              +------------------------------+
                           |
                           v
              +------------------------------+
              |  13. DASHBOARD DISPLAY       |
              |  @st.fragment(run_every=2)   |
              |                              |
              |  6 panels, 5 roles (RBAC):   |
              |                              |
              |  Live Monitor:               |
              |    Threat gauge, timeline,   |
              |    risk distribution,        |
              |    network heatmap           |
              |                              |
              |  Alert Feed:                 |
              |    Recent HIGH/CRITICAL,     |
              |    drill-down detail         |
              |                              |
              |  Explanations:               |
              |    Role-specific views       |
              |    (5 different renderers)   |
              |                              |
              |  Performance:                |
              |    Accuracy, Recall, F1,     |
              |    confusion matrix          |
              |                              |
              |  Stakeholder Intelligence:   |
              |    SOC / Clinician / CISO /  |
              |    BioMed role-specific      |
              |                              |
              |  System & Compliance:        |
              |    Model info, audit trail,  |
              |    FDA/HIPAA compliance      |
              +------------------------------+
```

---

## Data Flow Diagram

```
+------------------+     +------------------+     +------------------+
| WUSTL Dataset    |     | Phase 2 Model    |     | Phase 4 Baseline |
| flows.npy (7296) | --> | Weights (.h5)    | --> | Config (.json)   |
| labels.npy       |     | 477K params      |     | median, MAD      |
| filenames.json   |     | CNN-BiLSTM-Attn  |     | threshold        |
+------------------+     +------------------+     +------------------+
         |                        |                        |
         v                        v                        v
+========================================================================+
|                     STREAMING PIPELINE                                  |
|                                                                         |
|  Simulator         Buffer          Model           Calibrator          |
|  (per-flow)   -->  (window)   -->  (inference) --> (risk mapping)      |
|  vec[24]           (1,20,24)       score           risk_level          |
|  0.001ms           <1ms            ~15ms           <1ms                |
|                                                                         |
|  Flow 0-19: INITIALIZING (no inference, filling window)                |
|  Flow 20-199: CALIBRATING (inference active, alerts suppressed)        |
|  Flow 200+: OPERATIONAL (full pipeline, alerts enabled)                |
+========================================================================+
         |                                                    |
         v                                                    v
+------------------+                              +------------------+
| Risk Pipeline    |                              | Dashboard        |
|                  |                              | (Streamlit)      |
| Risk Scorer      |                              |                  |
|   -> CIA Modifier |                             | @st.fragment     |
|   -> Clinical    |                              | (run_every=2s)   |
|   -> Fatigue Mgr |                              |                  |
|   -> Explainer   |                              | 6 panels         |
|                  |                              | 5 roles (RBAC)   |
| ~175ms total     |                              | Live updating    |
+------------------+                              +------------------+
```

---

## Feature Schema (24 dimensions)

```
+------------------------------------------------------------------+
|                    MODEL INPUT FEATURES                            |
+------------------------------------------------------------------+
| NETWORK FEATURES (16)              | BIOMETRIC FEATURES (8)       |
|------------------------------------+------------------------------|
| SrcBytes    - Source bytes         | Temp       - Temperature     |
| DstBytes    - Destination bytes    | SpO2       - Blood oxygen    |
| SrcLoad     - Source load          | Pulse_Rate - Pulse rate      |
| DstLoad     - Destination load     | SYS        - Systolic BP     |
| SIntPkt     - Src inter-packet     | DIA        - Diastolic BP    |
| DIntPkt     - Dst inter-packet     | Heart_rate - Heart rate      |
| SIntPktAct  - Src inter-pkt active | Resp_Rate  - Respiratory     |
| sMaxPktSz   - Src max packet size  | ST         - ST segment      |
| dMaxPktSz   - Dst max packet size  |                              |
| sMinPktSz   - Src min packet size  |                              |
| Dur         - Flow duration        |                              |
| TotBytes    - Total bytes          |                              |
| Load        - Total load           |                              |
| pSrcLoss    - Source packet loss   |                              |
| pDstLoss    - Dest packet loss     |                              |
| Packet_num  - Packet count         |                              |
+------------------------------------------------------------------+
| 5 dropped (zero-variance in WUSTL):                               |
| SrcGap, DstGap, DIntPktAct, dMinPktSz, Trans                     |
+------------------------------------------------------------------+
```

---

## Key Thresholds & Decision Points

| Component | Threshold | Value | Purpose |
|-----------|-----------|-------|---------|
| Window Buffer | window_size | 20 | Model input timesteps |
| Window Buffer | calibration_threshold | 200 | Flows before alerts enabled |
| Baseline | median | 0.129 | Benign score center |
| Baseline | MAD | 0.025 | Score spread measure |
| Baseline | baseline_threshold | 0.255 | median + 3*MAD |
| Risk Scorer | LOW/MEDIUM | 0.5 * MAD | Risk boundary |
| Risk Scorer | MEDIUM/HIGH | 1.0 * MAD | Risk boundary |
| Risk Scorer | HIGH/CRITICAL | 2.0 * MAD | Risk boundary |
| CIA Modifier | escalation | >= 0.7 | CIA modifier escalation trigger |
| Clinical | biometric warn | 1.5 sigma | Vital sign abnormality |
| Calibrator | Youden's J | > 0.3 | Two-class mode activation |
| Calibrator | target FPR | 10% | Detection boundary tuning |
| Calibrator (benign) | P75/P85/P93/P97 | percentile | Risk level boundaries |
| Alert Fatigue | rate limit | 5 per 100 | Max alerts per device |
| Alert Fatigue | aggregation | every 10th | Same-severity suppression |
| Explanation | URGENT (sev>=3) | severity | Enable explanations |

---

## Latency Budget (per flow)

```
Component                    Time        Cumulative
-------------------------------------------------
Flow read (mmap)             0.001ms     0.001ms
Buffer append                <0.01ms     ~0.01ms
Window extraction            <0.1ms      ~0.1ms
Model inference (@tf.func)   ~15ms       ~15ms
  (score + backbone + grads in single compiled graph)
Score calibration            <0.1ms      ~15ms
Risk scoring                 <0.1ms      ~15ms
CIA modification             <0.1ms      ~15ms
Clinical assessment          <0.1ms      ~15ms
Alert fatigue                <0.1ms      ~15ms
Explanation (conditional):
  NONE                       0ms         ~15ms
  LIGHTWEIGHT                ~1ms        ~16ms
  FULL                       ~5-10ms     ~20-25ms
Buffer recording             <0.1ms      ~15-25ms
-------------------------------------------------
Total per flow:              ~15-25ms
Clinical SLA:                5 minutes (300,000ms)
Safety margin:               12,000-20,000x
```

---

## RBAC Access Matrix

```
                     Live    Alert  Explain  Perform  Stake   System
                     Monitor  Feed  ations   ance     holder  Comply
IT Security Analyst    X       X       X       X        X       X
Clinical IT Admin      X       X       X                X       X
Attending Physician    X       X       X                X
Hospital Manager       X               X       X        X       X
Regulatory Auditor     X               X                X       X
```

---

## System State Machine

```
                 flow_count < 20
INITIALIZING  ----------------------->  (no inference)
     |
     | flow_count == 20
     v
CALIBRATING   ----------------------->  (inference active,
     |          20 <= flow_count < 200     alerts suppressed,
     |                                     collecting calibration data)
     | flow_count == 200
     | (calibrator auto-fits)
     v
OPERATIONAL   ----------------------->  (full pipeline,
     |                                     alerts enabled)
     |
     | risk_level == CRITICAL
     v
ALERT         ----------------------->  (active threat,
     |                                     all alerts emitted)
     |
     | dataset exhausted
     v
EXHAUSTED     ----------------------->  (streaming complete,
                                          summary metrics displayed)
```
