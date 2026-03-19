## 7.1 Risk-Adaptive Engine

This section documents the Phase 4 Risk-Adaptive Engine, which applies
dynamic thresholding, concept drift detection, and multi-level risk scoring
to the Phase 3 classification output for IoMT healthcare environments.

### 7.1.1 Baseline Computation

Baseline computed from Normal-only training samples using Median + k*MAD:

| Parameter | Value |
|-----------|-------|
| Normal samples (train) | 9972 |
| Attention dimensions | 128 |
| Median | 0.128979 |
| MAD | 0.024865 |
| k (multiplier) | 3.0 |
| **Baseline threshold** | **0.203575** |

Formula: `baseline_threshold = Median + 3.0 * MAD`

### 7.1.2 Dynamic Thresholding

Rolling window Median + k(t)*MAD with time-of-day sensitivity:

| Time Window | k(t) |
|-------------|------|
| 00:00–06:00 | 2.5 |
| 06:00–22:00 | 3.0 |
| 22:00–24:00 | 3.5 |

**Window size:** 100 samples

Sample window log:

| Sample | Hour | k(t) | Window Median | Dynamic Threshold |
|--------|------|------|---------------|-------------------|
| 100 | 00:00 | 2.5 | 0.173613 | 0.395494 |
| 200 | 00:00 | 2.5 | 0.105064 | 0.178003 |
| 300 | 01:00 | 2.5 | 0.141695 | 0.271551 |
| 400 | 01:00 | 2.5 | 0.112829 | 0.205151 |
| 500 | 02:00 | 2.5 | 0.119683 | 0.220789 |
| ... | ... | ... | ... | ... |

### 7.1.3 Concept Drift Detection

| Parameter | Value |
|-----------|-------|
| Drift threshold | 0.2 (20%) |
| Recovery threshold | 0.1 (10%) |
| Recovery windows | 3 consecutive |
| Drift events detected | 1 |

Drift events:

| Sample Index | Drift Ratio | Action | Dynamic Threshold |
|-------------|-------------|--------|-------------------|
| 100 | 0.9427 | FALLBACK_LOCKED | 0.395494 |

### 7.1.4 Risk Level Classification

| Risk Level | Count | Percentage |
|------------|-------|------------|
| NORMAL | 3261 | 66.9% |
| LOW | 93 | 1.9% |
| MEDIUM | 85 | 1.7% |
| HIGH | 1376 | 28.2% |
| CRITICAL | 62 | 1.3% |

**Total samples scored:** 4877

Risk thresholds (MAD-relative):

| Level | Condition |
|-------|-----------|
| NORMAL | distance < 0 |
| LOW | 0 <= distance < 0.5*MAD |
| MEDIUM | 0.5*MAD <= distance < 1.0*MAD |
| HIGH | 1.0*MAD <= distance < 2.0*MAD |
| CRITICAL | distance >= 2.0*MAD AND cross-modal |

### 7.1.5 CRITICAL Risk Protocol

When a CRITICAL risk level is assigned:

1. Immediate on-site alert dispatched
2. Suspicious device isolated from network
3. Escalation chain: IT admin + doctor on duty + manager
4. Medical device is **NOT** shut down (patient safety)
5. Full context logged for human review

### 7.1.6 Phase 3 Model Performance (Inherited)

| Metric | Value |
|--------|-------|
| Accuracy | 0.8321 |
| F1-score | 0.8128 |
| AUC-ROC | 0.6119 |
| Test samples | 4877 |

### 7.1.7 Execution Summary

| Metric | Value |
|--------|-------|
| Device | CPU: x86_64 |
| TensorFlow | 2.20.0 |
| Python | 3.12.3 |
| Platform | Linux-6.17.0-14-generic-x86_64-with-glibc2.39 |
| Duration | 4.09s |
| Git commit | `3259f3eee118` |

### 7.1.8 Artifacts Exported

| Artifact | Description |
|----------|-------------|
| `baseline_config.json` | Median, MAD, baseline threshold (IMMUTABLE) |
| `threshold_config.json` | Current dynamic threshold, k(t) schedule |
| `risk_report.json` | Per-sample risk levels with distances |
| `drift_log.csv` | Concept drift events and fallback triggers |

---

**Reviewer signature:** ____________________  **Date:** ____/____/______
