## 4.1 Data Preprocessing Pipeline

This section documents the seven-step preprocessing pipeline applied to the WUSTL-EHMS-2020 dataset prior to model training. Each step is justified with reference to the data quality assessment in §3.2 and the security controls documented in §3.3.

### Pipeline Steps Overview

| Step | Input Shape | Output Shape | Notes |
|------|-------------|--------------|-------|
| 1. Ingestion | — | 16,318 × 45 | Raw WUSTL-EHMS CSV (signed integrity verified) |
| 2. HIPAA | 16,318 × 45 | 16,318 × 40 | 5 identifier cols dropped |
| 3. Missing | 16,318 × 40 | 16,318 × 40 | ffill bio, fill_zero net |
| 4. Redundancy | 16,318 × 40 | 16,318 × 34 | 6 correlated features dropped |
| 5. Split | 16,318 × 34 | train 9,790 / test 2,448 | Stratified 70/30 |
| 6. Scale | train 9,790 × 25 | train 9,790 × 25 | RobustScaler (train fit) |
| 7. SMOTE | (deferred) | (deferred) | enabled, applied inside Phase 2 CV |

### Feature Reduction Summary

| Reason | Features Dropped | Remaining |
|--------|----------------:|----------:|
| HIPAA identifiers | 5 | 40 |
| Redundancy (|*r*| ≥ 0.95) | 6 | 34 |
| Non-numeric / label | 2 | 32 |
| **Total reduction** | **13** | **32** |

### 4.1.1 HIPAA Safe Harbor De-identification

**5 columns dropped:** [`SrcAddr`, `DstAddr`, `SrcMac`, `DstMac`, `Packet_num`]

These columns encode network identifiers (IP addresses, MAC addresses, port numbers) and flow metadata that constitute environment-specific artefacts. Their removal satisfies HIPAA Safe Harbor §164.514(b)(2) and prevents the model from memorising topology-specific patterns that do not generalise to unseen network environments.

### 4.1.2 Context-Aware Missing Value Handling

| Stream | Strategy | Justification |
|--------|----------|---------------|
| Biometric (8 features) | Forward-fill (ffill) | Sensor dropout produces temporal gaps; the most recent valid reading is the best available estimate |
| Network (remaining features) | Row-wise dropna | Corrupted packets produce incomplete flow records that cannot be reliably imputed |

- Biometric cells filled: **0**
- Rows dropped (network NaN): **0**
- Rows remaining: **16,318**

### 4.1.x Residual Leakage Disclosure

The leakage barrier in this pipeline sits between Step 4 and Step 5. Steps 3–4 compute their decisions on the full dataset:

- **Cleaning**: median imputation is fit on the full dataset. Per-feature medians are population-level statistics, so the leak is bounded by the difference between the train median and the full-dataset median (typically <1% on this corpus).
- **Variance filter**: a feature is dropped if its full-dataset unique-value count falls below the threshold. The decision is binary, so the leak is upper-bounded by the count of features whose train-only `nunique` would have changed the verdict.
- **Redundancy filter**: feature pairs are read from Phase 0's `high_correlations.csv`, which was computed on the full dataset. The leak is upper-bounded by features whose train-only correlation falls below the threshold.

None of these are patient-data leaks (the cleaning step is now session-safe). They are *test-distribution* leaks that may modestly inflate test-set metrics. A future revision will compute Steps 3–4 over the train partition only.

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
| Train | 9,790 | 70% |
| Test | 2,448 | 30% |

Stratification via `StratifiedShuffleSplit` with `random_state=42` preserves the original class prior in both partitions, preventing evaluation bias from sampling variance.

### 4.1.5 SMOTE Configuration (applied in Phase 2 CV)

| Parameter | Value |
|-----------|-------|
| Enabled | yes |
| Sampling strategy | `auto` |
| `k_neighbors` | 5 |
| Applied at | Phase 2 cross-validation, train fold only |

SMOTE is configured here but executed inside the Phase 2 stratified cross-validation loop, where each training fold is resampled independently before the model is fit. Performing the resampling inside CV (rather than as a standalone Phase 1 step) prevents synthetic samples from any single fold from leaking into the validation fold, which would inflate every reported metric. The resampling is also performed in the **unscaled** feature space so synthetic interpolations are generated in the same geometry as the real data.

### 4.1.6 Robust Scaling

RobustScaler (median / IQR normalisation) is chosen over StandardScaler (mean / std) or MinMaxScaler because the outlier analysis in §3.2.1 identified heavy-tailed distributions in network-traffic features. RobustScaler is insensitive to extreme values, preserving the morphology of attack signatures for downstream explainability analysis.

Scaler fitted exclusively on training set (n=9,790). Test set transformed without refitting — preventing information leakage from test distribution. The fitted parameters are persisted as a JSON sidecar (`robust_scaler.json`), not a pickle, so loading the artefact never executes Python.

### 4.1.7 Pipeline Output Summary

| Artifact | Format | Description |
|----------|--------|-------------|
| `train_phase1.parquet` | Apache Parquet | 9,790 rows × 25 features |
| `test_phase1.parquet` | Apache Parquet | 2,448 rows × 25 features |
| `robust_scaler.json` | JSON sidecar | Fitted RobustScaler params (`center_`, `scale_`) — pickle-free |
| `preprocessing_report.json` | JSON | Per-step audit trail |

Total pipeline elapsed time: **0.13 s**
