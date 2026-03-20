## 8.2 Explanation Engine Security Controls

This section documents the security controls applied during Phase 5
explanation generation, extending the Phase 0 OWASP framework (§3.3)
and Phase 2/3/4 model controls (§5.2, §5.4, §7.2) with
explanation-engine-specific protections.

### 8.2.1 OWASP Controls — Phase 5 Extensions

| OWASP | Category | Control | Status |
|-------|----------|---------|--------|
| A01 | Access Control | Output paths within workspace | Implemented |
| A01 | Access Control | `shap_values.parquet` read-only (chmod 444) | Implemented |
| A01 | Access Control | `explanation_report.json` overwrite protection | Implemented |
| A01 | Access Control | Charts directory within workspace | Implemented |
| A02 | Crypto Failures | SHA-256 for `shap_values.parquet` | Implemented |
| A02 | Crypto Failures | SHA-256 for `explanation_report.json` | Implemented |
| A02 | Crypto Failures | SHA-256 for `explanation_metadata.json` | Implemented |
| A02 | Crypto Failures | Hashes stored in `explanation_metadata.json` | Implemented |
| A03 | Injection | Config dict sanitized via `ConfigSanitizer` | Implemented |
| A03 | Injection | Template variables validated against allowlist | Implemented |
| A03 | Injection | `top_features` ∈ [1, 29] | Implemented |
| A05 | Misconfiguration | `background_samples` bounded | Implemented |
| A05 | Misconfiguration | `max_explain_samples` bounded | Implemented |
| A05 | Misconfiguration | `max_waterfall_charts` bounded | Implemented |
| A05 | Misconfiguration | `max_timeline_charts` bounded | Implemented |
| A05 | Misconfiguration | Unknown YAML keys rejected | Implemented |
| A08 | Data Integrity | SHAP background Normal only | Implemented |
| A08 | Data Integrity | SHAP values shape matches filtered count | Implemented |
| A08 | Data Integrity | Feature importance ranks complete | Implemented |
| A08 | Data Integrity | Waterfall charts for HIGH+ only | Implemented |
| A08 | Data Integrity | No raw biometric values in explanations | Implemented |
| A08 | Data Integrity | Explanation count matches enriched | Implemented |
| A09 | Logging | SHAP computation time logged (safe) | Implemented |
| A09 | Logging | Feature importance ranking logged (safe) | Implemented |
| A09 | Logging | Risk level distribution logged (safe) | Implemented |
| A09 | Logging | Individual SHAP values NEVER logged | Implemented |
| A09 | Logging | Raw biometric values NEVER logged | Implemented |
| A09 | Logging | Patient identifiers NEVER in templates | Implemented |

### 8.2.2 Explanation Integrity Checklist

- [x] `shap_values.parquet` SHA-256 stored in `explanation_metadata.json`
- [x] `explanation_report.json` SHA-256 stored in `explanation_metadata.json`
- [x] SHAP background: Normal samples only — verified at runtime
- [x] No raw biometric values in explanation text
- [x] Explanation templates use feature names only, never raw values
- [x] All exported artifacts set to read-only (chmod 444)
- [x] Phase 2 + Phase 3 + Phase 4 artifacts verified via SHA-256 before loading

### 8.2.3 HIPAA Logging Compliance (A09)

| Data Category | Logged? | Justification |
|---------------|---------|---------------|
| SHAP computation time | Yes | Non-PHI: system performance metric |
| Feature importance ranking | Yes | Non-PHI: aggregate model statistics |
| Risk level distribution | Yes | Non-PHI: population-level counts |
| Number of samples explained | Yes | Non-PHI: aggregate count |
| Individual SHAP values per patient | **NEVER** | HIPAA: individual feature attributions |
| Raw biometric values | **NEVER** | HIPAA: columns = `DIA`, `Heart_rate`, `Pulse_Rate`, `Resp_Rate`, `ST`, `SYS`, `SpO2`, `Temp` |
| Patient identifiers | **NEVER** | HIPAA: no patient IDs in templates |
| Per-sample anomaly scores | **NEVER** | HIPAA: individual risk predictions |

### 8.2.4 Data Integrity Assertions (A08)

All assertions are evaluated at runtime and logged with pass/fail status.

| Assertion | Expected | Actual | Status |
|-----------|----------|--------|--------|
| SHAP background Normal only | n_normal > 0, 0 < n_background <= n_normal | n_normal=9990, n_background=100 | PASS |
| shap_values shape correct | (198, *, 29) | (198, 20, 29) | PASS |
| Feature importance ranks complete | <= 29 features ranked | 29 features ranked | PASS |
| Explanation count matches enriched | 198 | 198 | PASS |
| No raw biometric values in explanations | 0 violations | 0 violations in 198 samples | PASS |
| Waterfall for HIGH+ only | All waterfalls for CRITICAL/HIGH samples | 5 waterfalls, all valid | PASS |

**Overall:** ALL PASSED

### 8.2.5 Artifact Integrity (A02)

SHA-256 hashes computed after export for downstream verification.

| Artifact | SHA-256 |
|----------|---------|
| `shap_values.parquet` | `2399f61f46fab98b5d7b77741418562f7a53e753e6a784e1f190ff2d05954eec` |
| `explanation_report.json` | `f70bb0af5da48645b2bc5d53c74fb2cedb01638ace01b78e73322b19cb21949d` |

Hashes stored in `explanation_metadata.json` and must be verified
by the Notification Engine (Phase 6) before sending alerts.

### 8.2.6 Parameter Bounds Validation (A05)

| Parameter | Allowed Range | Configured Value | Status |
|-----------|--------------|-----------------|--------|
| `background_samples` | [10, 1000] | 100 | PASS |
| `max_explain_samples` | [1, 1000] | 200 | PASS |
| `top_features` | [1, 29] | 10 | PASS |
| `max_waterfall_charts` | [0, 50] | 5 | PASS |
| `max_timeline_charts` | [0, 50] | 3 | PASS |
| `random_state` | int | 42 | PASS |
| Unknown YAML keys | none allowed | 0 found | PASS |

### 8.2.7 Explained Risk Distribution

| Risk Level | Count | Percentage |
|------------|-------|------------|
| LOW | 93 | 5.8% |
| MEDIUM | 85 | 5.3% |
| HIGH | 1376 | 85.1% |
| CRITICAL | 62 | 3.8% |

**Total explained samples:** 1616

### 8.2.8 Security Inheritance from Phase 0, 2, 3, and 4

| Control | Source | Reuse Method |
|---------|--------|-------------|
| SHA-256 hashing | Phase 0 `IntegrityVerifier` | Direct import — `compute_hash()` |
| Path traversal | Phase 0 `PathValidator` | Direct import — `validate_output_dir()` |
| Config sanitization | Phase 0 `ConfigSanitizer` | Direct import — `sanitize_config_dict()` |
| Audit logging | Phase 0 `AuditLogger` | Direct import — `log_file_access()` |
| Phase 2 artifact SHA-256 | Phase 2 `detection_metadata.json` | Verified before model load |
| Phase 3 artifact SHA-256 | Phase 3 `classification_metadata.json` | Verified before model load |
| Phase 4 artifact SHA-256 | Phase 4 `risk_metadata.json` | Verified before risk report load |

---

**Reviewer signature:** ____________________  **Date:** ____/____/______
