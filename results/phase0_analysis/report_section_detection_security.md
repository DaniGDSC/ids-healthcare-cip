## 5.2 Detection Engine Security Controls

This section documents the security controls applied during Phase 2
detection, extending the Phase 0 OWASP framework (§3.3) with
model-specific protections.

### 5.2.1 OWASP Controls — Phase 2 Extensions

| OWASP ID | Risk Category | Control | Status |
|----------|---------------|---------|--------|
| A01 | Broken Access Control | Output paths validated within workspace boundary | Implemented |
| A01 | Broken Access Control | Artifacts set to read-only (chmod 444) after export | Implemented |
| A01 | Broken Access Control | Overwrite protection — existing artifacts not silently replaced | Implemented |
| A02 | Cryptographic Failures | SHA-256 hash computed for `detection_model.weights.h5` | Implemented |
| A02 | Cryptographic Failures | SHA-256 hash computed for `attention_output.parquet` | Implemented |
| A02 | Cryptographic Failures | Hashes stored in `detection_metadata.json` for Phase 3 verification | Implemented |
| A05 | Security Misconfiguration | `timesteps` validated ∈ [5, 100] | Implemented |
| A05 | Security Misconfiguration | `dropout_rate` validated ∈ [0.0, 0.8] | Implemented |
| A05 | Security Misconfiguration | CNN filters validated as powers of 2 | Implemented |
| A05 | Security Misconfiguration | Unknown YAML keys rejected | Implemented |
| A08 | Data Integrity | Attention weights sum to 1.0 per sample verified | Implemented |
| A08 | Data Integrity | Output shape matches expected dimensions verified | Implemented |
| A08 | Data Integrity | No NaN/Inf in context vectors verified | Implemented |
| A08 | Data Integrity | No classification head in model verified | Implemented |
| A09 | Security Logging | Model architecture summary logged (safe — structural metadata only) | Implemented |
| A09 | Security Logging | Layer output shapes logged (safe — dimensional metadata only) | Implemented |
| A09 | Security Logging | Per-patient attention weights NEVER logged (HIPAA risk) | Implemented |
| A09 | Security Logging | Aggregate attention stats only: mean, std | Implemented |

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
| Model architecture (layer names, types) | Yes | Non-PHI: structural metadata only |
| Layer output shapes | Yes | Non-PHI: dimensional metadata only |
| Parameter counts | Yes | Non-PHI: integer counts |
| Aggregate attention stats (mean, std) | Yes | Non-PHI: population-level statistics |
| Per-patient attention weights | **NEVER** | HIPAA risk: temporal focus patterns may reveal treatment timelines |
| Raw biometric values | **NEVER** | HIPAA: biometric columns = `DIA`, `Heart_rate`, `Pulse_Rate`, `Resp_Rate`, `ST`, `SYS`, `SpO2`, `Temp` |
| Individual context vectors | **NEVER** | HIPAA risk: patient-level representations |

### 5.2.5 Artifact Integrity (A02)

SHA-256 hashes computed after export for downstream verification
by the Classification Engine (Phase 3).

| Artifact | SHA-256 |
|----------|---------|
| `detection_model.weights.h5` | `32dde631b6153628eadfc16e88cf8f830265eaac2c8285f4873273b2df22bd3e` |
| `attention_output.parquet` | `a2d9a4d82cca9f3d2e87f794fb329361b12fde61a4a70effc10e3048e5dab81b` |

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
