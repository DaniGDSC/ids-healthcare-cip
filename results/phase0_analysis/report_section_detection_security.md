## 5.2 Detection Engine Security Controls

This section documents the security controls applied during Phase 2
detection, extending the Phase 0 OWASP framework (§3.3) with
model-specific protections.

### 5.2.1 OWASP Controls — Phase 2 Extensions

| OWASP | Category | Control | Status |
|-------|----------|---------|--------|
| A01 | Access Control | Output paths within workspace | Implemented |
| A01 | Access Control | Read-only (chmod 444) after export | Implemented |
| A01 | Access Control | Overwrite protection | Implemented |
| A02 | Crypto Failures | SHA-256 for model weights | Implemented |
| A02 | Crypto Failures | SHA-256 for attention parquet | Implemented |
| A02 | Crypto Failures | Hashes in `detection_metadata.json` | Implemented |
| A05 | Misconfiguration | `timesteps` ∈ [5, 100] | Implemented |
| A05 | Misconfiguration | `dropout_rate` ∈ [0.0, 0.8] | Implemented |
| A05 | Misconfiguration | CNN filters = powers of 2 | Implemented |
| A05 | Misconfiguration | Unknown YAML keys rejected | Implemented |
| A08 | Data Integrity | Attention weights sum = 1.0 | Implemented |
| A08 | Data Integrity | Output shape verified | Implemented |
| A08 | Data Integrity | No NaN/Inf in context vectors | Implemented |
| A08 | Data Integrity | No classification head | Implemented |
| A09 | Logging | Architecture logged (safe metadata) | Implemented |
| A09 | Logging | Layer shapes logged (safe metadata) | Implemented |
| A09 | Logging | Per-patient weights NEVER logged | Implemented |
| A09 | Logging | Aggregate stats only: mean, std | Implemented |

### 5.2.2 Model Integrity Checklist

- [x] `detection_model.weights.h5` SHA-256 stored in `detection_metadata.json`
- [x] `attention_output.parquet` SHA-256 stored in `detection_metadata.json`
- [x] No classification head in `detection_model.weights.h5` (context vector output only)
- [x] Attention weights verified: sum = 1.0 per sample (softmax normalisation)
- [x] No NaN or Inf in train context vectors
- [x] No NaN or Inf in test context vectors
- [x] Output shape matches expected dimensions (128-dim context)

### 5.2.3 Data Integrity Assertions (A08)

All assertions are evaluated at runtime and logged with pass/fail status.

| Assertion | Expected | Actual | Status |
|-----------|----------|--------|--------|
| No classification head | True (context vector output only) | last_layer=BahdanauAttention, has_head=False | PASS |
| Output shape (train_context) | (19961, 128) | (19961, 128) | PASS |
| Output shape (test_context) | (4877, 128) | (4877, 128) | PASS |
| No NaN/Inf in train_context | 0 NaN, 0 Inf | 0 NaN, 0 Inf | PASS |
| No NaN/Inf in test_context | 0 NaN, 0 Inf | 0 NaN, 0 Inf | PASS |
| Attention weights sum to 1.0 per sample | 1.0 (all samples) | mean=1.000000, min=1.000000, max=1.000000 | PASS |

**Overall:** ALL PASSED

### 5.2.4 HIPAA Logging Compliance (A09)

| Data Category | Logged? | Justification |
|---------------|---------|---------------|
| Model architecture | Yes | Non-PHI: structural metadata |
| Layer output shapes | Yes | Non-PHI: dimensional metadata |
| Parameter counts | Yes | Non-PHI: integer counts |
| Aggregate attention stats | Yes | Non-PHI: population-level stats |
| Per-patient attention weights | **NEVER** | HIPAA: may reveal treatment timelines |
| Raw biometric values | **NEVER** | HIPAA: columns = `DIA`, `Heart_rate`, `Pulse_Rate`, `Resp_Rate`, `ST`, `SYS`, `SpO2`, `Temp` |
| Individual context vectors | **NEVER** | HIPAA: patient-level representations |

### 5.2.5 Artifact Integrity (A02)

SHA-256 hashes computed after export for downstream verification
by the Classification Engine (Phase 3).

| Artifact | SHA-256 |
|----------|---------|
| `detection_model.weights.h5` | `5aa425ca8129f3bca59e9c479d23d455664073267aec7e184455ade8373c1b91` |
| `attention_output.parquet` | `33aeabba393332acb287b8ff8e37db10e81f3fd23451ee1cc44ecb647515b0dd` |

Hashes are stored in `detection_metadata.json` and must be
verified before loading artifacts in subsequent pipeline phases.

### 5.2.6 Parameter Bounds Validation (A05)

| Parameter | Allowed Range | Configured Value | Status |
|-----------|--------------|-----------------|--------|
| `timesteps` | [5, 100] | 20 | PASS |
| `dropout_rate` | [0.0, 0.8] | 0.3 | PASS |
| `cnn_filters_1` | power of 2 | 64 | PASS |
| `cnn_filters_2` | power of 2 | 128 | PASS |
| `random_state` | int | 42 | PASS |
| Unknown YAML keys | none allowed | 0 found | PASS |

### 5.2.7 Security Inheritance from Phase 0

The following Phase 0 controls (§3.3) are reused without duplication:

| Control | Phase 0 Module | Reuse Method |
|---------|---------------|-------------|
| SHA-256 hashing | `IntegrityVerifier` | Direct import — `compute_hash()` |
| Path traversal protection | `PathValidator` | Direct import — `validate_output_dir()` |
| Config sanitization | `ConfigSanitizer` | Direct import — `sanitize_config_dict()` |
| Audit logging | `AuditLogger` | Direct import — `log_file_access()`, `log_security_event()` |

---

**Reviewer signature:** ____________________  **Date:** ____/____/______
