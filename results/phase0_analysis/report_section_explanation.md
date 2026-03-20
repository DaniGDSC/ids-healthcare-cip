## 8.1 Explanation Engine — SHAP-Based Feature Attribution

This section presents the Phase 5 Explanation Engine results,
providing interpretable feature attributions for all non-NORMAL
risk samples using SHAP (SHapley Additive exPlanations).

### 8.1.1 Samples Explained

| Risk Level | Count |
|------------|-------|
| LOW | 93 |
| MEDIUM | 85 |
| HIGH | 1376 |
| CRITICAL | 62 |
| **Total** | **198** |

Baseline threshold: 0.203575

### 8.1.2 Global Feature Importance (Top 10)

| Rank | Feature | Mean |SHAP| |
|------|---------|--------------|
| 1 | DIntPkt | 0.016464 |
| 2 | TotBytes | 0.005599 |
| 3 | DstBytes | 0.002624 |
| 4 | SrcBytes | 0.002437 |
| 5 | SIntPkt | 0.001639 |
| 6 | Dur | 0.001618 |
| 7 | Resp_Rate | 0.001209 |
| 8 | ST | 0.001203 |
| 9 | SpO2 | 0.001155 |
| 10 | Heart_rate | 0.001099 |

Features are ranked by mean absolute SHAP value across all
explained samples. SHAP values aggregated over 20
timesteps per sliding window.

**Feature Interpretation:**
Among the top 10 features, 6 are network traffic indicators (DIntPkt, TotBytes, DstBytes...) and 4 are biometric signals (Resp_Rate, ST, SpO2...). Network features dominate, suggesting that traffic anomalies are the primary indicators of intrusion attempts.

### 8.1.3 Explanation Examples

| Level | Sample | Explanation |
|-------|--------|-------------|
| LOW | 189 | LOW: Minor anomaly at sample 189. No immediate action needed. |
| MEDIUM | 809 | MEDIUM: Anomaly detected at sample 809. Monitor closely. Key feature: DIntPkt. |
| HIGH | 23 | HIGH ALERT: Suspicious activity detected at sample 23. Primary indicator: DIntPkt anomaly contributing 44.0%. |
| CRITICAL | 93 | CRITICAL ALERT: Sample 93 at T=93. Top factors: DIntPkt anomaly (56.1%), SpO2 anomaly (6.8%), Dur anomaly (4.4%). Immedi |

### 8.1.4 Visualizations

**Feature importance bar chart:**
- `charts/feature_importance.png`

**Waterfall charts (CRITICAL/HIGH samples):**
- `waterfall_23.png`
- `waterfall_89.png`
- `waterfall_161.png`
- `waterfall_166.png`
- `waterfall_172.png`

**Anomaly timeline charts:**
- `timeline_23.png`
- `timeline_89.png`
- `timeline_161.png`

### 8.1.5 SHAP Methodology

| Parameter | Value |
|-----------|-------|
| Method | GradientExplainer (integrated gradients fallback) |
| Background samples | 100 (Normal class, training set) |
| Explained samples | 198 |
| Input shape | (20, 29) per window |
| Aggregation | mean(abs(SHAP)) over timesteps |
| Feature count | 29 |

**Method Justification:**
GradientExplainer is selected as the primary attribution method because
it is neural-network-native and efficiently handles deep learning models
with 3D temporal input (batch, timesteps, features). Unlike KernelExplainer,
which treats the model as a black box and requires exponential perturbations
for high-dimensional input, GradientExplainer leverages backpropagation
gradients to compute attributions in a single forward-backward pass per
background sample. The integrated gradients fallback provides theoretical
guarantees (completeness and sensitivity axioms) when the GradientExplainer
encounters compatibility issues with custom layers such as BahdanauAttention.
SHAP values are aggregated via mean(|SHAP|) over the temporal dimension to
produce per-feature importance scores that are interpretable by clinicians
and security analysts.

### 8.1.6 Execution Details

| Parameter | Value |
|-----------|-------|
| Hardware | CPU: x86_64 |
| TensorFlow | 2.20.0 |
| Duration | 21.68s |
| Git commit | `901983542dbc` |

### 8.1.7 Human-Centric Justification

Explainability is a critical requirement in healthcare IoMT intrusion
detection systems. SHAP-based feature attributions serve three purposes:

1. **Clinical Decision Support:** When an alert is raised, clinicians need
   to understand *why* the system flagged a particular sample. Waterfall
   charts trace each feature's contribution to the final prediction,
   enabling clinicians to assess whether the alert reflects genuine
   physiological distress or a network-based attack.

2. **Audit Trail Compliance:** Healthcare regulations (HIPAA, FDA
   cybersecurity guidance) require that automated decision systems provide
   transparent reasoning. The per-sample explanations and feature
   importance rankings form an auditable trail from raw sensor data to
   risk classification.

3. **Trust Calibration:** By showing the relative importance of biometric
   vs. network features, the explanation engine helps security analysts
   calibrate their trust in the system. When network features dominate
   (e.g., DIntPkt, TotBytes), the alert likely reflects a network
   intrusion; when biometric features dominate (e.g., SpO2, Heart_rate),
   the alert may indicate device tampering or sensor manipulation.

---

**Generated:** 2026-03-19 14:01:07 UTC
**Pipeline:** Phase 5 Explanation Engine (SOLID)
**Artifacts:** data/phase5/
