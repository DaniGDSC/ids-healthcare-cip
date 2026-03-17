## 4.2 Preprocessing Security Controls

This section documents the security controls applied during Phase 1
data preprocessing, extending the Phase 0 OWASP framework (§3.3) with
data-pipeline-specific protections.

### 4.2.1 OWASP Controls — Phase 1 Extensions

| OWASP ID | Risk Category | Control | Status |
|----------|---------------|---------|--------|
| A01 | Broken Access Control | Output paths validated within workspace boundary | Implemented |
| A01 | Broken Access Control | Artifacts set to read-only (chmod 444) after export | Implemented |
| A01 | Broken Access Control | Overwrite protection — existing artifacts not silently replaced | Implemented |
| A02 | Cryptographic Failures | SHA-256 hash computed for train.parquet, test.parquet, scaler.pkl | Implemented |
| A02 | Cryptographic Failures | Hashes stored in `preprocessing_metadata.json` | Implemented |
| A03 | Injection | SMOTE `k_neighbors` validated ∈ [1, 10] | Implemented |
| A03 | Injection | `random_state` type-checked as `int` | Implemented |
| A03 | Injection | `train_ratio` validated ∈ [0.5, 0.9] | Implemented |
| A08 | Data Integrity | Train + test = original sample count verified | Implemented |
| A08 | Data Integrity | Zero train/test index overlap verified | Implemented |
| A08 | Data Integrity | SMOTE applied exclusively to training partition | Implemented |
| A09 | Security Logging | HIPAA column drops logged (names only, never values) | Implemented |
| A09 | Security Logging | SMOTE augmentation count logged (no sample values) | Implemented |
| A09 | Security Logging | Scaler parameters logged (center, scale — non-PHI) | Implemented |

### 4.2.2 HIPAA Preprocessing Compliance Checklist

- [x] PII fields dropped before any transformation: [`SrcAddr`, `DstAddr`, `Sport`, `Dport`, `SrcMac`, `DstMac`, `Dir`, `Flgs`]
- [x] No PHI values in any log file — column names only
- [x] Output artifacts set to read-only after export
- [x] Train/test overlap verified — 0 shared indices
- [x] Biometric values never logged: `DIA`, `Heart_rate`, `Pulse_Rate`, `Resp_Rate`, `ST`, `SYS`, `SpO2`, `Temp`
- [x] SMOTE sample values not logged — only aggregate counts

### 4.2.3 Data Integrity Assertions (A08)

All assertions are evaluated at runtime and logged with pass/fail status.

| Assertion | Expected | Actual | Status |
|-----------|----------|--------|--------|
| Train + test = original | 16318 | 16318 | PASS |
| No train/test overlap | 0 | 0 | PASS |
| SMOTE on train only | True | True | PASS |

**Overall:** ALL PASSED

### 4.2.4 Artifact Integrity (A02)

SHA-256 hashes computed after export for downstream verification
by the Detection Engine (Phase 2).

| Artifact | SHA-256 |
|----------|---------|
| `train_phase1.parquet` | `a6775690db39f32509efb5add883454cd7c401c31406484cd2341f6052263803` |
| `test_phase1.parquet` | `81981d49023c3588758deb1486ee0e13665af91c8bfe2143504d92ba3318a9d0` |
| `robust_scaler.pkl` | `4e5c96032175543c2381b261a96a4f60a96567b15c8573db3f9125ce37ebcf02` |

Hashes are stored in `preprocessing_metadata.json` and must be
verified before loading artifacts in subsequent pipeline phases.

### 4.2.5 Parameter Bounds Validation (A03)

| Parameter | Allowed Range | Configured Value | Status |
|-----------|--------------|-----------------|--------|
| `smote_k_neighbors` | [1, 10] | 5 | PASS |
| `random_state` | int | 42 | PASS |
| `train_ratio` | [0.5, 0.9] | 0.7 | PASS |
| `correlation_threshold` | (0, 1] | 0.95 | PASS |

### 4.2.6 Scaling Parameters (A09 — Safe to Log)

- **Features scaled:** 29
- **Center (median) range:** [0.0000, 293752.0000]
- **Scale (IQR) range:** [0.0388, 256815.1964]

RobustScaler center and scale values are derived from median and IQR
of *network traffic features* — they do not constitute PHI and are
safe to include in logs and reports.

### 4.2.7 Security Inheritance from Phase 0

The following Phase 0 controls (§3.3) are reused without duplication:

| Control | Phase 0 Module | Reuse Method |
|---------|---------------|-------------|
| SHA-256 hashing | `IntegrityVerifier` | Direct import — `compute_hash()` |
| Path traversal protection | `PathValidator` | Direct import — `validate_output_dir()` |
| Config sanitization | `ConfigSanitizer` | Direct import — `sanitize_config_dict()` |
| Audit logging | `AuditLogger` | Direct import — `log_file_access()`, `log_security_event()` |

---

**Reviewer signature:** ____________________  **Date:** ____/____/______
