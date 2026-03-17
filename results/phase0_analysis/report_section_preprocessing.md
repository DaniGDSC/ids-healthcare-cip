## 4.1 Data Preprocessing Pipeline

This section documents the seven-step preprocessing pipeline applied to the WUSTL-EHMS-2020 dataset prior to model training. Each step is justified with reference to the data quality assessment in §3.2 and the security controls documented in §3.3.

### Pipeline Steps Overview

| Step | Input Shape | Output Shape | Notes |
|------|-------------|--------------|-------|
| 1. Ingestion | — | 16,318 × 45 | Raw WUSTL-EHMS CSV |
| 2. HIPAA | 16,318 × 45 | 16,318 × 37 | 8 identifier cols dropped |
| 3. Missing | 16,318 × 37 | 16,318 × 37 | ffill bio, dropna net |
| 4. Redundancy | 16,318 × 37 | 16,318 × 31 | 6 correlated features dropped |
| 5. Split | 16,318 × 31 | train 11,422 / test 4,896 | Stratified 70/30 |
| 6. SMOTE | 11,422 × 29 | 19,980 × 29 | Train only |
| 7. Scale | 19,980 × 29 | 19,980 × 29 | RobustScaler (train fit) |

### Feature Reduction Summary

| Reason | Features Dropped | Remaining |
|--------|----------------:|----------:|
| HIPAA identifiers | 8 | 37 |
| Redundancy (|*r*| ≥ 0.95) | 6 | 31 |
| Non-numeric / label | 2 | 29 |
| **Total reduction** | **16** | **29** |

### 4.1.1 HIPAA Safe Harbor De-identification

**8 columns dropped:** [`SrcAddr`, `DstAddr`, `Sport`, `Dport`, `SrcMac`, `DstMac`, `Dir`, `Flgs`]

These columns encode network identifiers (IP addresses, MAC addresses, port numbers) and flow metadata that constitute environment-specific artefacts. Their removal satisfies HIPAA Safe Harbor §164.514(b)(2) and prevents the model from memorising topology-specific patterns that do not generalise to unseen network environments.

### 4.1.2 Context-Aware Missing Value Handling

| Stream | Strategy | Justification |
|--------|----------|---------------|
| Biometric (8 features) | Forward-fill (ffill) | Sensor dropout produces temporal gaps; the most recent valid reading is the best available estimate |
| Network (remaining features) | Row-wise dropna | Corrupted packets produce incomplete flow records that cannot be reliably imputed |

- Biometric cells filled: **0**
- Rows dropped (network NaN): **0**
- Rows remaining: **16,318**

### 4.1.3 Redundancy Elimination

High-correlation pairs (|*r*| ≥ 0.95) were identified in Phase 0 (§3.2.3) and read from `high_correlations.csv` — the correlation matrix was **not** recomputed. For each pair, the secondary feature was dropped, reducing the feature space by **6** columns:

| Dropped Feature | Reason |
|-----------------|--------|
| `SrcJitter` | |*r*| ≥ 0.95 with a retained feature |
| `pLoss` | |*r*| ≥ 0.95 with a retained feature |
| `Rate` | |*r*| ≥ 0.95 with a retained feature |
| `DstJitter` | |*r*| ≥ 0.95 with a retained feature |
| `Loss` | |*r*| ≥ 0.95 with a retained feature |
| `TotPkts` | |*r*| ≥ 0.95 with a retained feature |

### 4.1.4 Stratified Train/Test Split

| Partition | Samples | Ratio |
|-----------|--------:|------:|
| Train | 11,422 | 70% |
| Test | 4,896 | 30% |

Stratification via `StratifiedShuffleSplit` with `random_state=42` preserves the original class prior in both partitions, preventing evaluation bias from sampling variance.

### 4.1.5 SMOTE Oversampling (Train Only)

| Class | Before SMOTE | After SMOTE |
|-------|------------:|------------:|
| Normal (0) | 9,990 | 9,990 |
| Attack (1) | 1,432 | 9,990 |
| **Total** | **11,422** | **19,980** |

SMOTE (Synthetic Minority Oversampling Technique) with *k* = 5 is applied **exclusively to the training partition** to prevent synthetic data from contaminating the test evaluation. The oversampling is performed **before** scaling so that synthetic samples are generated in the original feature space, not in a normalised space where inter-feature distances are distorted.

### 4.1.6 Robust Scaling

RobustScaler (median / IQR normalisation) is chosen over StandardScaler (mean / std) or MinMaxScaler because the outlier analysis in §3.2.1 identified heavy-tailed distributions in network-traffic features. RobustScaler is insensitive to extreme values, preserving the morphology of attack signatures for downstream XAI (SHAP) interpretation.

Scaler fitted exclusively on training set (n=19,980). Test set transformed without refitting — preventing information leakage from test distribution.

### 4.1.7 Pipeline Output Summary

| Artifact | Format | Description |
|----------|--------|-------------|
| `train_phase1.parquet` | Apache Parquet | 19,980 rows × 29 features |
| `test_phase1.parquet` | Apache Parquet | 4,896 rows × 29 features |
| `robust_scaler.pkl` | joblib pickle | Fitted RobustScaler for inference |
| `preprocessing_report.json` | JSON | Per-step audit trail |

Total pipeline elapsed time: **0.08 s**
