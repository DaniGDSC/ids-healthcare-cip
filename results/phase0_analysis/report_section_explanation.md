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

### 8.1.3 Explanation Examples

| Level | Sample | Explanation |
|-------|--------|-------------|
| LOW | 189 | LOW: Minor anomaly at sample 189. No immediate action needed. |
| MEDIUM | 809 | MEDIUM: Anomaly detected at sample 809. Monitor closely. Key feature: DIntPkt. |
| HIGH | 23 | HIGH ALERT: Suspicious activity detected at sample 23. Primary indicator: DIntPkt contributing 44.0%. |
| CRITICAL | 93 | CRITICAL ALERT: Sample 93 at T=93. Top factors: DIntPkt=0.0267 (56.1%), SpO2=0.0033 (6.8%), Dur=0.0021 (4.4%). Immediate |

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

### 8.1.6 Execution Details

| Parameter | Value |
|-----------|-------|
| Hardware | CPU: x86_64 |
| TensorFlow | 2.20.0 |
| Duration | 25.93s |
| Git commit | `718383d14552` |

---

**Generated:** 2026-03-19 12:59:31 UTC
**Pipeline:** Phase 5 Explanation Engine
**Artifacts:** data/phase5/
