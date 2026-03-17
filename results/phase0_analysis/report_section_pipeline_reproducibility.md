## 4.3 Pipeline Reproducibility and CI/CD Integration

This section documents the end-to-end reproducibility of the
Phase 0 → Phase 1 preprocessing pipeline, the CI/CD infrastructure
that enforces it, and the integration test results that verify
artifact correctness.

### 4.3.1 End-to-End Pipeline Summary

The complete preprocessing pipeline executes Phase 0 (exploratory
data analysis) followed by Phase 1 (data preprocessing) in a single
deterministic run.

| Metric | Value |
|--------|-------|
| Total execution time | **0.15 s** |
| Hardware | Intel Core i7-14700HX, Linux 6.17.0-14-generic |
| Python | 3.12.3 |
| scikit-learn | 1.8.0 |
| imbalanced-learn | 0.14.1 |
| pandas | 3.0.0 |
| numpy | 2.4.1 |
| pydantic | 2.12.5 |

### 4.3.2 Artifact Provenance Chain

```
raw CSV (16,318 × 45)
  │
  ├─── [Phase 0: Exploratory Data Analysis]
  │      ├── stats_report.json          — descriptive statistics, class distribution
  │      ├── high_correlations.csv      — 7 feature pairs (|r| ≥ 0.95)
  │      ├── dataset_integrity.json     — SHA-256 baseline
  │      └── report_section_quality.md  — §3.2 thesis report
  │
  └─── [Phase 1: Data Preprocessing]
         ├── train_phase1.parquet       — 19,980 × 30 (29 features + Label)
         ├── test_phase1.parquet        — 4,896 × 30  (29 features + Label)
         ├── robust_scaler.pkl          — fitted RobustScaler (29 features)
         ├── phase1_report.json         — per-step audit trail
         └── preprocessing_metadata.json — SHA-256 hashes for all artifacts
```

**Feature reduction:** 45 → HIPAA (−8) → Missing (−0) → Redundancy (−6) → Non-numeric (−2) → **29 features**

### 4.3.3 Integration Test Results

| Test | Status | Details |
|------|--------|---------|
| `train_phase1.parquet` exists | PASS | Shape: (19,980, 30) |
| `test_phase1.parquet` exists | PASS | Shape: (4,896, 30) |
| `scaler.pkl` loadable | PASS | RobustScaler, 29 features |
| No train/test overlap | PASS | 0 shared rows |
| Stratified split preserved | PASS | Train attack=12.54%, test attack=12.54% |
| SMOTE augmented minority | PASS | 11,422 → 19,980 (+8,558 synthetic) |
| SMOTE class balance | PASS | 50.0% attack rate post-SMOTE |
| Report JSON complete | PASS | 10 sections present |

### 4.3.4 Artifact Integrity Verification

SHA-256 hashes computed after export and stored in `preprocessing_metadata.json`:

| Artifact | SHA-256 |
|----------|---------|
| `train_phase1.parquet` | `a6775690db39f32509efb5add883454cd7c401c31406484cd2341f6052263803` |
| `test_phase1.parquet` | `81981d49023c3588758deb1486ee0e13665af91c8bfe2143504d92ba3318a9d0` |
| `robust_scaler.pkl` | `4e5c96032175543c2381b261a96a4f60a96567b15c8573db3f9125ce37ebcf02` |

Dataset integrity (Phase 0 baseline):
```
SHA-256: 8359da96154fa60247df9d75e52d232077acb9886c09c36de2a0aaaee6cf2c25
```

### 4.3.5 CI/CD Pipeline Architecture

```
Phase 0 Workflow                      Phase 1 Workflow
┌──────────┐                          ┌──────────────┐
│   lint   │                          │ lint-phase1  │
└────┬─────┘                          └──────┬───────┘
     │                                       │
┌────▼──────────┐                    ┌───────▼────────┐  ┌──────────────────┐
│ security-scan │                    │  test-phase1   │  │ security-scan-p1 │
└────┬──────────┘                    └───────┬────────┘  └────────┬─────────┘
     │                                       │                    │
┌────▼─────┐                         ┌───────▼────────────────────▼──┐
│   test   │                         │      integration-test         │
└────┬─────┘                         │  (Phase 0 → Phase 1 on 100   │
     │                               │   row subset + assertions)   │
┌────▼─────┐                         └──────────────┬────────────────┘
│  build   │                                        │
└──────────┘                         ┌──────────────▼─────────────┐
                                     │     build (Docker v2.0)    │
                                     └────────────────────────────┘
```

**Triggers:** `push` to `main`, `pull_request` to `main`, Phase 0 workflow completion.

**Integration test strategy:** A 100-row balanced subset (80 Normal + 20 Attack)
is generated from the full dataset. Phase 0 artifacts are computed for the subset,
then Phase 1 runs the complete 7-step pipeline. Eight assertions verify artifact
existence, shapes, overlap, stratification, and SMOTE correctness.

### 4.3.6 Docker Reproducibility

The combined Phase 0 + Phase 1 image (`analyst/phase0-phase1:2.0`) provides
deterministic execution:

```bash
# Run all Phase 0 + Phase 1 tests
docker run --rm analyst/phase0-phase1:2.0

# Run Phase 1 pipeline (mount data volume)
docker run --rm \
  -v $(pwd)/data:/home/analyst/app/data \
  -v $(pwd)/results:/home/analyst/app/results \
  analyst/phase0-phase1:2.0 \
  src.phase1_preprocessing.phase1.pipeline
```

### 4.3.7 Complete Reproducibility Statement

Full preprocessing pipeline reproducible via:

```
docker run analyst/phase0-phase1:2.0 src.phase1_preprocessing.phase1.pipeline
```

| Parameter | Value |
|-----------|-------|
| `random_state` | 42 |
| SMOTE `k_neighbors` | 5 |
| Train/test split | 70/30 (stratified) |
| Correlation threshold | 0.95 |
| Scaling method | RobustScaler (median / IQR) |
| Expected train shape | (19,980, 29) |
| Expected test shape | (4,896, 29) |
| HIPAA columns removed | 8 (SrcAddr, DstAddr, Sport, Dport, SrcMac, DstMac, Dir, Flgs) |
| Redundant features removed | 6 (SrcJitter, pLoss, Rate, DstJitter, Loss, TotPkts) |

All stochastic operations use `random_state=42`. Configuration is externalised
in `config/phase1_config.yaml` (version-controlled). The pipeline reads Phase 0
artifacts — it never recomputes correlation matrices, missing value statistics,
or dataset hashes.
