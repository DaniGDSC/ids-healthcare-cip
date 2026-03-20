## 5.4 Classification Engine Security Controls

This section documents the security controls applied during Phase 3
classification, extending the Phase 0 OWASP framework (§3.3) and
Phase 2 model controls (§5.2) with classification-specific protections.

### 5.4.1 OWASP Controls — Phase 3 Extensions

| OWASP | Category | Control | Status |
|-------|----------|---------|--------|
| A01 | Access Control | Output paths within workspace | Implemented |
| A01 | Access Control | Read-only (chmod 444) after export | Implemented |
| A01 | Access Control | Overwrite protection | Implemented |
| A02 | Crypto Failures | SHA-256 for classification model | Implemented |
| A02 | Crypto Failures | SHA-256 for metrics report | Implemented |
| A02 | Crypto Failures | Hashes in `classification_metadata.json` | Implemented |
| A04 | Insecure Design | Unfreezing order validated | Implemented |
| A04 | Insecure Design | Learning rate decrease validated | Implemented |
| A05 | Misconfiguration | `dense_units` ∈ [1, 512] | Implemented |
| A05 | Misconfiguration | `dropout` ∈ [0.0, 0.8] | Implemented |
| A05 | Misconfiguration | `threshold` ∈ [0.01, 0.99] | Implemented |
| A05 | Misconfiguration | Unknown YAML keys rejected | Implemented |
| A08 | Data Integrity | Evaluation on test set only | Implemented |
| A08 | Data Integrity | No train/test data overlap | Implemented |
| A08 | Data Integrity | Confusion matrix sum verified | Implemented |
| A08 | Data Integrity | F1/precision/recall consistency | Implemented |
| A08 | Data Integrity | Classification head present | Implemented |
| A09 | Logging | Aggregate metrics logged (safe) | Implemented |
| A09 | Logging | Per-patient predictions NEVER logged | Implemented |
| A09 | Logging | Raw feature values NEVER logged | Implemented |

### 5.4.2 Model Integrity Checklist

- [x] `classification_model.weights.h5` SHA-256 stored in `classification_metadata.json`
- [x] `metrics_report.json` SHA-256 stored in `classification_metadata.json`
- [x] Evaluation performed on test set only — verified
- [x] No patient-level predictions logged — aggregate metrics only
- [x] Classification head present in exported model
- [x] Confusion matrix row sums match test set size
- [x] Phase 2 artifacts verified via SHA-256 before loading

### 5.4.3 Data Integrity Assertions (A08)

All assertions are evaluated at runtime and logged with pass/fail status.

| Assertion | Expected | Actual | Status |
|-----------|----------|--------|--------|
| Evaluation on test set only | 4877 test samples | 4877 eval samples | PASS |
| No train/test data overlap | overlap ≤ 0 (train=19961, test=4877) | overlap=0 | PASS |
| Confusion matrix sum = test samples | 4877 | 4877 | PASS |
| Metrics consistency (F1, precision, recall ∈ [0,1]) | all ∈ [0.0, 1.0] | F1=0.8133, prec=0.7968, rec=0.8347 | PASS |
| Prediction validity (accuracy, AUC ∈ [0,1]) | accuracy ∈ [0,1], AUC ∈ [0,1] | accuracy=0.8347, AUC=0.6124 | PASS |
| Classification head present | True (Dense output layer) | last_layer=Dense, has_head=True | PASS |

**Overall:** ALL PASSED

### 5.4.4 Progressive Unfreezing Validation (A04)

| Phase | Epochs | Learning Rate | Frozen Groups | Status |
|-------|--------|---------------|---------------|--------|
| Phase A — Head only | 5 | 0.001 | 4 frozen | PASS |
| Phase B — Attention + Head | 5 | 0.0001 | 3 frozen | PASS |
| Phase C — BiLSTM-2 + Attention + Head | 5 | 1e-05 | 2 frozen | PASS |

Validation: frozen count decreases across phases, learning rate decreases.

**Justification:** Progressive unfreezing chosen to prevent catastrophic
forgetting of Phase 2 feature extraction weights while adapting to
classification task.

### 5.4.5 HIPAA Logging Compliance (A09)

| Data Category | Logged? | Justification |
|---------------|---------|---------------|
| Model architecture | Yes | Non-PHI: structural metadata |
| Aggregate metrics (accuracy, F1) | Yes | Non-PHI: population-level stats |
| Confusion matrix | Yes | Non-PHI: aggregate counts |
| Training loss/accuracy | Yes | Non-PHI: model performance |
| Per-patient predictions | **NEVER** | HIPAA: individual classifications |
| Raw feature values | **NEVER** | HIPAA: columns = `DIA`, `Heart_rate`, `Pulse_Rate`, `Resp_Rate`, `ST`, `SYS`, `SpO2`, `Temp` |
| Individual attention weights | **NEVER** | HIPAA: patient-level representations |

### 5.4.6 Artifact Integrity (A02)

SHA-256 hashes computed after export for downstream verification.

| Artifact | SHA-256 |
|----------|---------|
| `classification_model.weights.h5` | `0add48352ce5c231f0951c85ba0b56029df304723816ba10c9e5fbeb81cd99cf` |
| `metrics_report.json` | `c7708b6ce40e1e23739731607906ccd355ed6a33db5925dd4a0724fbe98929b7` |
| `confusion_matrix.csv` | `a622d800a4f22bec2bc852384205947f835a88c96e1ce0cc6a0a39c1204b9bba` |
| `training_history.json` | `f9c383bb09af12986e18e5975d5037c3cbf4997fa06e0e6c397a36b3322b225b` |

Hashes stored in `classification_metadata.json` and must be
verified before loading artifacts in subsequent pipeline phases.

### 5.4.7 Parameter Bounds Validation (A05)

| Parameter | Allowed Range | Configured Value | Status |
|-----------|--------------|-----------------|--------|
| `dense_units` | [1, 512] | 64 | PASS |
| `head_dropout_rate` | [0.0, 0.8] | 0.3 | PASS |
| `threshold` | [0.01, 0.99] | 0.5 | PASS |
| `batch_size` | [8, 2048] | 256 | PASS |
| `random_state` | int | 42 | PASS |
| Unknown YAML keys | none allowed | 0 found | PASS |

### 5.4.8 Security Inheritance from Phase 0 and Phase 2

| Control | Source | Reuse Method |
|---------|--------|-------------|
| SHA-256 hashing | Phase 0 `IntegrityVerifier` | Direct import — `compute_hash()` |
| Path traversal | Phase 0 `PathValidator` | Direct import — `validate_output_dir()` |
| Config sanitization | Phase 0 `ConfigSanitizer` | Direct import — `sanitize_config_dict()` |
| Audit logging | Phase 0 `AuditLogger` | Direct import — `log_file_access()` |
| Phase 2 artifact SHA-256 | Phase 2 `detection_metadata.json` | Verified before model load |

---

**Reviewer signature:** ____________________  **Date:** ____/____/______
