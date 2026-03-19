## 7.2 Risk-Adaptive Engine Security Controls

This section documents the security controls applied during Phase 4
risk-adaptive threshold computation, extending the Phase 0 OWASP
framework (§3.3) and Phase 2/3 model controls (§5.2, §5.4) with
risk-engine-specific protections.

### 7.2.1 OWASP Controls — Phase 4 Extensions

| OWASP | Category | Control | Status |
|-------|----------|---------|--------|
| A01 | Access Control | Output paths within workspace | Implemented |
| A01 | Access Control | `baseline_config.json` write-once (chmod 444) | Implemented |
| A01 | Access Control | Overwrite protection for baseline | Implemented |
| A02 | Crypto Failures | SHA-256 for `baseline_config.json` | Implemented |
| A02 | Crypto Failures | SHA-256 for `risk_report.json` | Implemented |
| A02 | Crypto Failures | SHA-256 for `threshold_config.json` | Implemented |
| A02 | Crypto Failures | SHA-256 for `drift_log.csv` | Implemented |
| A02 | Crypto Failures | Hashes stored in `risk_metadata.json` | Implemented |
| A02 | Crypto Failures | Baseline SHA-256 verify-on-load | Implemented |
| A04 | Insecure Design | ZeroDivisionError guard on drift_ratio | Implemented |
| A04 | Insecure Design | k(t) ∈ [1.0, 5.0] | Implemented |
| A04 | Insecure Design | window_size ∈ [10, 1000] | Implemented |
| A05 | Misconfiguration | `drift_threshold` bounded | Implemented |
| A05 | Misconfiguration | `recovery_threshold` < `drift_threshold` | Implemented |
| A05 | Misconfiguration | `recovery_windows` bounded | Implemented |
| A05 | Misconfiguration | Unknown YAML keys rejected | Implemented |
| A05 | Misconfiguration | Risk level ordering validated | Implemented |
| A08 | Data Integrity | Normal-only baseline assertion | Implemented |
| A08 | Data Integrity | Baseline threshold ∈ [0, 1] | Implemented |
| A08 | Data Integrity | Dynamic thresholds >= 0 | Implemented |
| A08 | Data Integrity | Risk level consistency with distance | Implemented |
| A08 | Data Integrity | Risk distribution sum = total samples | Implemented |
| A08 | Data Integrity | `drift_log.csv` append-only verified | Implemented |
| A08 | Data Integrity | Baseline immutable keys verified | Implemented |
| A09 | Logging | Risk distribution logged (safe) | Implemented |
| A09 | Logging | Drift events logged (safe) | Implemented |
| A09 | Logging | Baseline summary logged (safe) | Implemented |
| A09 | Logging | Per-patient anomaly scores NEVER logged | Implemented |
| A09 | Logging | Device identifiers NEVER logged | Implemented |
| A09 | Logging | CRITICAL actions logged with token audit | Implemented |

### 7.2.2 Baseline Integrity Checklist

- [x] `baseline_config.json` computed from Normal-only training data
- [x] `baseline_config.json` SHA-256 stored in `risk_metadata.json`
- [x] `baseline_config.json` set to read-only (chmod 444) after export
- [x] SHA-256 verified on every load before threshold computation
- [x] Baseline threshold ∈ [0, 1] — validated at runtime
- [x] Baseline has all 7 required immutable keys
- [x] Phase 2 + Phase 3 artifacts verified via SHA-256 before loading

### 7.2.3 Runtime Security Checklist

- [x] ZeroDivisionError guard on `drift_ratio` computation
- [x] k(t) values bounded to [1.0, 5.0]
- [x] Window size bounded to [10, 1000]
- [x] Dynamic thresholds validated ∈ [0, 1] after computation
- [x] Risk levels verified consistent with distance values
- [x] `drift_log.csv` event count verified (append-only)
- [x] CRITICAL actions require IT admin token
- [x] No isolation executed without human confirmation

### 7.2.4 Data Integrity Assertions (A08)

All assertions are evaluated at runtime and logged with pass/fail status.

| Assertion | Expected | Actual | Status |
|-----------|----------|--------|--------|
| Baseline from Normal-only training samples | n_normal > 0, n_attack = 0 | n_normal=9972, n_attack=0 | PASS |
| Baseline config has all required keys | ['baseline_threshold', 'computed_at', 'mad', 'mad_multiplier', 'median', 'n_attention_dims', 'n_normal_samples'] | ['baseline_threshold', 'computed_at', 'mad', 'mad_multiplier', 'median', 'n_attention_dims', 'n_normal_samples'] | PASS |
| baseline_threshold ∈ [0, 1] | [0.0, 1.0] | 0.203575 | PASS |
| Dynamic thresholds >= 0 | all values >= 0.0 | min=0.095990, max=1.298908 | PASS |
| Risk level consistency with distance | 0 violations | 0 violations out of 4877 samples | PASS |
| Risk distribution sums to total samples | 4877 | 4877 | PASS |
| drift_log.csv append-only | 1 events | 1 events | PASS |

**Overall:** ALL PASSED

### 7.2.5 k(t) Schedule Validation (A04)

| Time Window | k(t) | Allowed Range | Status |
|-------------|------|---------------|--------|
| 00:00–06:00 | 2.5 | [1.0, 5.0] | PASS |
| 06:00–22:00 | 3.0 | [1.0, 5.0] | PASS |
| 22:00–24:00 | 3.5 | [1.0, 5.0] | PASS |

All k(t) values must be within [1.0, 5.0] to prevent
excessively loose or tight thresholds.

### 7.2.6 CRITICAL Action Security

CRITICAL risk requires BOTH biometric AND network modalities anomalous
(>2σ simultaneously). Before any device isolation action:

1. IT admin token must be provided and validated
2. Human confirmation is mandatory — no automated isolation
3. All CRITICAL actions are logged with timestamp, risk level, and authorization

| Sample | Action | Status | Timestamp |
|--------|--------|--------|-----------|
| 93 | ISOLATION_BLOCKED | BLOCKED | 2026-03-19T05:03:57.336813+00:00 |
| 317 | ISOLATION_BLOCKED | BLOCKED | 2026-03-19T05:03:57.336840+00:00 |
| 322 | ISOLATION_BLOCKED | BLOCKED | 2026-03-19T05:03:57.336865+00:00 |
| 442 | ISOLATION_BLOCKED | BLOCKED | 2026-03-19T05:03:57.336888+00:00 |
| 476 | ISOLATION_BLOCKED | BLOCKED | 2026-03-19T05:03:57.336910+00:00 |
| 519 | ISOLATION_BLOCKED | BLOCKED | 2026-03-19T05:03:57.336932+00:00 |
| 733 | ISOLATION_BLOCKED | BLOCKED | 2026-03-19T05:03:57.336954+00:00 |
| 735 | ISOLATION_BLOCKED | BLOCKED | 2026-03-19T05:03:57.336974+00:00 |
| 763 | ISOLATION_BLOCKED | BLOCKED | 2026-03-19T05:03:57.336994+00:00 |
| 765 | ISOLATION_BLOCKED | BLOCKED | 2026-03-19T05:03:57.337013+00:00 |
| 780 | ISOLATION_BLOCKED | BLOCKED | 2026-03-19T05:03:57.337034+00:00 |
| 839 | ISOLATION_BLOCKED | BLOCKED | 2026-03-19T05:03:57.337053+00:00 |
| 859 | ISOLATION_BLOCKED | BLOCKED | 2026-03-19T05:03:57.337073+00:00 |
| 896 | ISOLATION_BLOCKED | BLOCKED | 2026-03-19T05:03:57.337092+00:00 |
| 904 | ISOLATION_BLOCKED | BLOCKED | 2026-03-19T05:03:57.337111+00:00 |
| 1097 | ISOLATION_BLOCKED | BLOCKED | 2026-03-19T05:03:57.337130+00:00 |
| 1109 | ISOLATION_BLOCKED | BLOCKED | 2026-03-19T05:03:57.337150+00:00 |
| 1115 | ISOLATION_BLOCKED | BLOCKED | 2026-03-19T05:03:57.337170+00:00 |
| 1226 | ISOLATION_BLOCKED | BLOCKED | 2026-03-19T05:03:57.337190+00:00 |
| 1227 | ISOLATION_BLOCKED | BLOCKED | 2026-03-19T05:03:57.337209+00:00 |
| 1239 | ISOLATION_BLOCKED | BLOCKED | 2026-03-19T05:03:57.337233+00:00 |
| 1364 | ISOLATION_BLOCKED | BLOCKED | 2026-03-19T05:03:57.337252+00:00 |
| 1374 | ISOLATION_BLOCKED | BLOCKED | 2026-03-19T05:03:57.337272+00:00 |
| 1406 | ISOLATION_BLOCKED | BLOCKED | 2026-03-19T05:03:57.337292+00:00 |
| 1415 | ISOLATION_BLOCKED | BLOCKED | 2026-03-19T05:03:57.337315+00:00 |
| 1459 | ISOLATION_BLOCKED | BLOCKED | 2026-03-19T05:03:57.337336+00:00 |
| 1461 | ISOLATION_BLOCKED | BLOCKED | 2026-03-19T05:03:57.337355+00:00 |
| 1493 | ISOLATION_BLOCKED | BLOCKED | 2026-03-19T05:03:57.337375+00:00 |
| 1496 | ISOLATION_BLOCKED | BLOCKED | 2026-03-19T05:03:57.337397+00:00 |
| 1723 | ISOLATION_BLOCKED | BLOCKED | 2026-03-19T05:03:57.337417+00:00 |
| 1815 | ISOLATION_BLOCKED | BLOCKED | 2026-03-19T05:03:57.337438+00:00 |
| 1946 | ISOLATION_BLOCKED | BLOCKED | 2026-03-19T05:03:57.337458+00:00 |
| 1962 | ISOLATION_BLOCKED | BLOCKED | 2026-03-19T05:03:57.337477+00:00 |
| 1968 | ISOLATION_BLOCKED | BLOCKED | 2026-03-19T05:03:57.337496+00:00 |
| 1971 | ISOLATION_BLOCKED | BLOCKED | 2026-03-19T05:03:57.337515+00:00 |
| 2283 | ISOLATION_BLOCKED | BLOCKED | 2026-03-19T05:03:57.337534+00:00 |
| 2487 | ISOLATION_BLOCKED | BLOCKED | 2026-03-19T05:03:57.337553+00:00 |
| 2705 | ISOLATION_BLOCKED | BLOCKED | 2026-03-19T05:03:57.337572+00:00 |
| 2706 | ISOLATION_BLOCKED | BLOCKED | 2026-03-19T05:03:57.337591+00:00 |
| 2824 | ISOLATION_BLOCKED | BLOCKED | 2026-03-19T05:03:57.337610+00:00 |
| 2842 | ISOLATION_BLOCKED | BLOCKED | 2026-03-19T05:03:57.337629+00:00 |
| 2851 | ISOLATION_BLOCKED | BLOCKED | 2026-03-19T05:03:57.337648+00:00 |
| 2982 | ISOLATION_BLOCKED | BLOCKED | 2026-03-19T05:03:57.337667+00:00 |
| 3000 | ISOLATION_BLOCKED | BLOCKED | 2026-03-19T05:03:57.337686+00:00 |
| 3010 | ISOLATION_BLOCKED | BLOCKED | 2026-03-19T05:03:57.337706+00:00 |
| 3065 | ISOLATION_BLOCKED | BLOCKED | 2026-03-19T05:03:57.337725+00:00 |
| 3160 | ISOLATION_BLOCKED | BLOCKED | 2026-03-19T05:03:57.337744+00:00 |
| 3246 | ISOLATION_BLOCKED | BLOCKED | 2026-03-19T05:03:57.337762+00:00 |
| 3260 | ISOLATION_BLOCKED | BLOCKED | 2026-03-19T05:03:57.337781+00:00 |
| 3291 | ISOLATION_BLOCKED | BLOCKED | 2026-03-19T05:03:57.337800+00:00 |
| 3296 | ISOLATION_BLOCKED | BLOCKED | 2026-03-19T05:03:57.337818+00:00 |
| 3503 | ISOLATION_BLOCKED | BLOCKED | 2026-03-19T05:03:57.337837+00:00 |
| 3623 | ISOLATION_BLOCKED | BLOCKED | 2026-03-19T05:03:57.337856+00:00 |
| 3710 | ISOLATION_BLOCKED | BLOCKED | 2026-03-19T05:03:57.337875+00:00 |
| 3781 | ISOLATION_BLOCKED | BLOCKED | 2026-03-19T05:03:57.337894+00:00 |
| 3866 | ISOLATION_BLOCKED | BLOCKED | 2026-03-19T05:03:57.337913+00:00 |
| 3878 | ISOLATION_BLOCKED | BLOCKED | 2026-03-19T05:03:57.337932+00:00 |
| 3943 | ISOLATION_BLOCKED | BLOCKED | 2026-03-19T05:03:57.337951+00:00 |
| 4255 | ISOLATION_BLOCKED | BLOCKED | 2026-03-19T05:03:57.337971+00:00 |
| 4368 | ISOLATION_BLOCKED | BLOCKED | 2026-03-19T05:03:57.337989+00:00 |
| 4768 | ISOLATION_BLOCKED | BLOCKED | 2026-03-19T05:03:57.338009+00:00 |
| 4795 | ISOLATION_BLOCKED | BLOCKED | 2026-03-19T05:03:57.338028+00:00 |

### 7.2.7 HIPAA Logging Compliance (A09)

| Data Category | Logged? | Justification |
|---------------|---------|---------------|
| Risk distribution (aggregate) | Yes | Non-PHI: population-level counts |
| Drift events (sample_index, ratio) | Yes | Non-PHI: system health metrics |
| Baseline summary (threshold, MAD) | Yes | Non-PHI: statistical parameters |
| Dynamic threshold changes | Yes | Non-PHI: system adaptation metrics |
| Per-patient anomaly scores | **NEVER** | HIPAA: individual risk predictions |
| Device identifiers | **NEVER** | HIPAA: patient-linked device IDs |
| Raw biometric values | **NEVER** | HIPAA: columns = `DIA`, `Heart_rate`, `Pulse_Rate`, `Resp_Rate`, `ST`, `SYS`, `SpO2`, `Temp` |
| Individual feature vectors | **NEVER** | HIPAA: patient-level representations |

### 7.2.8 Artifact Integrity (A02)

SHA-256 hashes computed after export for downstream verification.

| Artifact | SHA-256 |
|----------|---------|
| `baseline_config.json` | `14f4d0acaca313b6b7e75cf60c1be89090e5f4aff1fd208d18764b2e7cf845df` |
| `threshold_config.json` | `1e88daf56513ac9be9094d4a8873ac5563fac61718310cdd2c5e0936ac018fe7` |
| `risk_report.json` | `16a3e744c14c180b5a923ae10ba943a3a9897d854d20d642a51b9f095f145f5f` |
| `drift_log.csv` | `fc8f2bb06e82799dee5a8e823c5dc0bd6c4d168bf05f8c43de14436fe664590d` |

Hashes stored in `risk_metadata.json` and must be verified before
loading artifacts in subsequent pipeline phases or audit reviews.

### 7.2.9 Parameter Bounds Validation (A05)

| Parameter | Allowed Range | Configured Value | Status |
|-----------|--------------|-----------------|--------|
| `mad_multiplier` | [0.5, 10.0] | 3.0 | PASS |
| `window_size` | [10, 1000] | 100 | PASS |
| `drift_threshold` | [0.05, 0.5] | 0.2 | PASS |
| `recovery_threshold` | [0.01, 0.49] | 0.1 | PASS |
| `recovery_windows` | [1, 10] | 3 | PASS |
| `k(t) schedule` | [1.0, 5.0] | all within range | PASS |
| `random_state` | int | 42 | PASS |
| Unknown YAML keys | none allowed | 0 found | PASS |

### 7.2.10 Risk Distribution Summary

| Risk Level | Count | Percentage |
|------------|-------|------------|
| NORMAL | 3261 | 66.9% |
| LOW | 93 | 1.9% |
| MEDIUM | 85 | 1.7% |
| HIGH | 1376 | 28.2% |
| CRITICAL | 62 | 1.3% |

**Total samples:** 4877

### 7.2.11 Security Inheritance from Phase 0, 2, and 3

| Control | Source | Reuse Method |
|---------|--------|-------------|
| SHA-256 hashing | Phase 0 `IntegrityVerifier` | Direct import — `compute_hash()` |
| Path traversal | Phase 0 `PathValidator` | Direct import — `validate_output_dir()` |
| Config sanitization | Phase 0 `ConfigSanitizer` | Direct import — `sanitize_config_dict()` |
| Audit logging | Phase 0 `AuditLogger` | Direct import — `log_file_access()` |
| Phase 2 artifact SHA-256 | Phase 2 `detection_metadata.json` | Verified before model load |
| Phase 3 artifact SHA-256 | Phase 3 `classification_metadata.json` | Verified before model load |

---

**Reviewer signature:** ____________________  **Date:** ____/____/______
