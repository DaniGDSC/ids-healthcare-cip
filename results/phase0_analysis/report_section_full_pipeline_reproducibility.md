## 6.2 Full Pipeline Reproducibility and CI/CD Integration

This section documents the complete four-phase pipeline artifact chain,
end-to-end timing, final model performance with regression baselines,
and the CI/CD infrastructure that enforces reproducibility across all phases.

### 6.2.1 Pipeline Artifact Provenance Chain

```
raw CSV (16,318 x 45)
  |
  +--- [Phase 0: Exploratory Data Analysis]
  |      +-- stats_report.json            -- descriptive statistics, class distribution
  |      +-- high_correlations.csv        -- 7 feature pairs (|r| >= 0.95)
  |      +-- dataset_integrity.json       -- SHA-256 baseline
  |      +-- report_section_quality.md    -- SS 3.2 thesis report
  |      +-- report_section_security.md   -- SS 3.3 OWASP controls
  |
  +--- [Phase 1: Data Preprocessing]
  |      +-- train_phase1.parquet         -- 19,980 x 30 (29 features + Label)
  |      +-- test_phase1.parquet          -- 4,896 x 30  (29 features + Label)
  |      +-- robust_scaler.pkl            -- fitted RobustScaler (29 features)
  |      +-- preprocessing_metadata.json  -- SHA-256 hashes for all artifacts
  |      +-- report_section_preprocessing.md -- SS 4.1 thesis report
  |
  +--- [Phase 2: Detection Engine]
  |      +-- detection_model.weights.h5   -- CNN-BiLSTM-Attention (474,496 params)
  |      +-- attention_output.parquet     -- context vectors (128-dim)
  |      +-- detection_metadata.json      -- SHA-256 hashes + hyperparameters
  |      +-- report_section_detection.md  -- SS 5.1 thesis report
  |
  +--- [Phase 3: Classification Engine]
         +-- classification_model.weights.h5 -- full model (482,817 params)
         +-- metrics_report.json             -- evaluation metrics + model summary
         +-- confusion_matrix.csv            -- 2x2 (Normal vs Attack)
         +-- training_history.json           -- 3-phase progressive unfreezing log
         +-- classification_metadata.json    -- SHA-256 hashes + integrity assertions
         +-- report_section_classification.md         -- SS 6.1 thesis report
         +-- report_section_classification_security.md -- SS 5.4 security controls
```

**Feature reduction:** 45 -> HIPAA (-8) -> Missing (-0) -> Redundancy (-6) -> Non-numeric (-2) -> **29 features**

**Model architecture:** Input(20,29) -> CNN(64,128) -> BiLSTM(128,64) -> BahdanauAttention(128) -> Dense(64,relu) -> Dropout(0.3) -> Dense(1,sigmoid)

### 6.2.2 End-to-End Pipeline Timing

| Phase | Duration | Hardware | Key Output |
|-------|----------|----------|------------|
| Phase 0 | 0.07 s | CPU | stats_report.json, high_correlations.csv |
| Phase 1 | 0.08 s | CPU | train.parquet (19,980 rows), test.parquet (4,896 rows) |
| Phase 2 | 2.99 s | CPU: x86_64 | detection_model.weights.h5 (474,496 params) |
| Phase 3 | 43.87 s | CPU: x86_64 | classification_model.weights.h5 (482,817 params) |
| **Total** | **47.01 s** | **CPU** | **Full pipeline on Intel Core i7-14700HX** |

### 6.2.3 Final Model Performance

| Metric | Value | Baseline | Delta |
|--------|-------|----------|-------|
| Accuracy | 0.8321 | 0.8321 | +0.00% |
| F1-score | 0.8128 | 0.8128 | +0.00% |
| Precision | 0.7977 | 0.7977 | +0.00% |
| Recall | 0.8321 | 0.8321 | +0.00% |
| AUC-ROC | 0.6119 | 0.6119 | +0.00% |

Regression thresholds: accuracy drop > 2% or F1 drop > 2% triggers CI failure.

### 6.2.4 Model Versioning

| Property | Value |
|----------|-------|
| `classification_model.weights.h5` git commit | `3259f3eee118ce1913b4bfa0fa3d0602f357b588` |
| Timestamp | 2026-03-18T18:31:45+00:00 |
| Phase 2 model SHA-256 | `5aa425ca8129f3bca59e9c479d23d455664073267aec7e184455ade8373c1b91` |
| Phase 3 model SHA-256 | `114fc8cc1c3d93e6fc5384828e7771d865f2070924fb398ca1985a7c6ba235e1` |
| n_classes | 2 (binary: Normal vs Attack) |
| Freeze strategy | Progressive unfreezing (A: head, B: +attention, C: +bilstm2) |
| Final accuracy | 0.8321 |
| Final F1 | 0.8128 |
| Final AUC-ROC | 0.6119 |

Model versioning stored in `classification_metadata.json` with full hyperparameters,
artifact SHA-256 hashes, and integrity assertion results.

### 6.2.5 Artifact Integrity Chain

All artifacts are SHA-256 verified at each phase boundary:

| Phase | Artifact | SHA-256 |
|-------|----------|---------|
| Phase 1 | `train_phase1.parquet` | `a6775690db39f325...` |
| Phase 1 | `test_phase1.parquet` | `81981d49023c3588...` |
| Phase 1 | `robust_scaler.pkl` | `4e5c96032175543c...` |
| Phase 2 | `detection_model.weights.h5` | `5aa425ca8129f3bc...` |
| Phase 2 | `attention_output.parquet` | `33aeabba39333acb...` |
| Phase 3 | `classification_model.weights.h5` | `114fc8cc1c3d93e6...` |
| Phase 3 | `metrics_report.json` | `3a97f4fa97967a66...` |
| Phase 3 | `confusion_matrix.csv` | `3ed35ddeb7533299...` |
| Phase 3 | `training_history.json` | `9443e99db75ea1bd...` |

### 6.2.6 CI/CD Pipeline Architecture

```
Phase 0 Workflow          Phase 1 Workflow          Phase 2 Workflow          Phase 3 Workflow
+----------+              +--------------+          +--------------+          +--------------+
|   lint   |              | lint-phase1  |          | lint-phase2  |          | lint-phase3  |
+----+-----+              +------+-------+          +------+-------+          +------+-------+
     |                           |                         |                         |
+----v----------+         +------v--------+  +------+  +---v----------+  +------+  +---v----------+  +------+
| security-scan |         | test-phase1   |  | sec  |  | test-phase2  |  | sec  |  | test-phase3  |  | sec  |
+----+----------+         +------+--------+  +--+---+  +------+-------+  +--+---+  +------+-------+  +--+---+
     |                           |              |             |              |             |              |
+----v-----+              +------v--------------v--+   +------v--------------v--+   +------v--------------v--+
|   test   |              | integration-test       |   | integration-test       |   | integration-test       |
+----+-----+              | (Phase 0 -> 1 on 100   |   | (Phase 0 -> 1 -> 2 on |   | (Phase 0 -> 1 -> 2 -> |
     |                    |  row subset)            |   |  100 row subset)       |   |  3 on 100 row subset) |
+----v-----+              +------------+-----------+   +------------+-----------+   +------------+-----------+
|  build   |                           |                            |                            |
+----------+              +------------v-----------+   +------------v-----------+   +------------v-----------+
                          | build (Docker v2.0)    |   | build (Docker v3.0)    |   | build (Docker v4.0)    |
                          +------------------------+   +------------------------+   +------------------------+
```

**Trigger chain:** Each phase workflow triggers on its predecessor's completion.
Phase 3 runs after Phase 2 succeeds via `workflow_run` event.

**Concurrency:** Each workflow group cancels in-progress runs on new pushes.

### 6.2.7 Integration Test Summary

| Phase | Tests | Status | Key Assertions |
|-------|-------|--------|---------------|
| Phase 0 | 64 | ALL PASS | Dataset integrity, statistics, correlations |
| Phase 1 | 50 | ALL PASS | Parquet shapes, stratification, SMOTE, no overlap |
| Phase 2 | 56 | ALL PASS | No classification head, attention weights sum=1.0, SHA-256 |
| Phase 3 | 38 | ALL PASS | Classification head present, metrics keys, CM shape, regression |
| **Total** | **208** | **ALL PASS** | **85% coverage (Phase 3 SOLID package)** |

### 6.2.8 Docker Reproducibility

The combined Phase 0-3 image (`analyst/phase0-phase3:4.0`) provides
deterministic execution:

```bash
# Run all Phase 0-3 unit tests
docker run --rm analyst/phase0-phase3:4.0

# Run full pipeline (mount data volume)
docker run --rm \
  -v $(pwd)/data:/home/analyst/app/data \
  -v $(pwd)/results:/home/analyst/app/results \
  analyst/phase0-phase3:4.0 \
  src.phase3_classification_engine.security_hardened_phase3
```

### 6.2.9 Reproducibility Statement

Full pipeline reproducible via:

```
docker run analyst/phase0-phase3:4.0 src.phase3_classification_engine.security_hardened_phase3
```

Expected `classification_model.weights.h5` metrics:
- accuracy = 0.8321, F1 = 0.8128, AUC-ROC = 0.6119

| Parameter | Value |
|-----------|-------|
| `random_state` | 42 |
| `tf.random.set_seed()` | 42 |
| `numpy.random.seed()` | 42 |
| `TF_DETERMINISTIC_OPS` | 1 |
| Train shape (windowed) | (19,961, 20, 29) |
| Test shape (windowed) | (4,877, 20, 29) |
| Detection parameters | 474,496 |
| Classification parameters | 482,817 (detection + 8,321 head) |
| Timesteps | 20 |
| Stride | 1 |
| Progressive unfreezing | 3 phases (A: lr=1e-3, B: lr=1e-4, C: lr=1e-5) |

All stochastic operations use `random_state=42`, `tf.random.set_seed(42)`,
and `numpy.random.seed(42)`. Configuration is externalised in
`config/phase3_config.yaml` (version-controlled). The pipeline reads
Phase 1 and Phase 2 artifacts -- it never recomputes preprocessing or
feature extraction steps.

### 6.2.10 Peer Review Readiness Checklist

- [x] All 4 phases tested end-to-end (208 unit tests)
- [x] All `report_section_*.md` generated (12 report sections)
- [x] All SHA-256 hashes verified (9 artifacts across 3 phases)
- [x] Regression tests passing (accuracy and F1 within 2% of baseline)
- [x] Docker image reproducible (`analyst/phase0-phase3:4.0`)
- [x] SBOM generated -- 0 critical CVEs (`sbom-phase3.json`)
- [x] OWASP Top 10 controls applied (A01, A02, A04, A05, A08, A09)
- [x] HIPAA compliance verified (no PHI in logs or outputs)
- [x] Model versioning with git commit hash in `classification_metadata.json`

---

**Reviewer signature:** ____________________  **Date:** ____/____/______
