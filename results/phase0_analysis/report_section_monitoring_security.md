## 10.2 Monitoring Engine Security Controls (Phase 7)

This section documents the security controls applied during Phase 7
monitoring engine execution, extending the Phase 0 OWASP framework
(section 3.3) with monitoring-engine-specific protections.
The monitoring engine is the **highest-privilege process** and
requires extra hardening.

### 10.2.1 OWASP Controls — Phase 7 Extensions

| OWASP | Category | Control | Status |
|-------|----------|---------|--------|
| A01 | Access Control | `monitoring_log.json` read-only (chmod 444) | Implemented |
| A01 | Access Control | `security_audit_log.json` append-only | Implemented |
| A01 | Access Control | Engine ID allowlist (reject unknown) | Implemented |
| A01 | Access Control | Output paths via PathValidator | Implemented |
| A02 | Crypto Failures | HMAC-SHA256 heartbeat signing | Implemented |
| A02 | Crypto Failures | HMAC key rotation every 24h | Implemented |
| A02 | Crypto Failures | HMAC grace window for in-flight HB | Implemented |
| A02 | Crypto Failures | SHA-256 for all 4 exported artifacts | Implemented |
| A02 | Crypto Failures | Post-export hash via IntegrityVerifier | Implemented |
| A03 | Injection | `engine_id` validated against allowlist | Implemented |
| A03 | Injection | `timestamp` validated as ISO 8601 | Implemented |
| A03 | Injection | `latency_ms` bounded [0.0, 10000.0] | Implemented |
| A03 | Injection | Config sanitized via ConfigSanitizer | Implemented |
| A05 | Misconfiguration | `heartbeat_interval_seconds` [1, 60] | Implemented |
| A05 | Misconfiguration | `missed_heartbeat_threshold` [1, 20] | Implemented |
| A05 | Misconfiguration | `circular_buffer_size` [100, 10000] | Implemented |
| A05 | Misconfiguration | `artifact_integrity_check_interval` [10, 3600] | Implemented |
| A05 | Misconfiguration | Unknown YAML keys rejected | Implemented |
| A05 | Misconfiguration | Duplicate engine IDs rejected | Implemented |
| A08 | Data Integrity | FSM transitions follow defined rules only | Implemented |
| A08 | Data Integrity | Circular buffer bounded per engine | Implemented |
| A08 | Data Integrity | Artifact hashes verified post-export | Implemented |
| A08 | Data Integrity | `security_audit_log.json` append-only | Implemented |
| A08 | Data Integrity | Engine IDs in allowlist assertion | Implemented |
| A08 | Data Integrity | Heartbeat count within expected bounds | Implemented |
| A09 | Logging | State changes logged (engine_id + transition) | Implemented |
| A09 | Logging | Aggregate summary logged (counts only) | Implemented |
| A09 | Logging | Security violations logged (type + engine) | Implemented |
| A09 | Logging | Raw heartbeat payload NEVER logged | Implemented |
| A09 | Logging | Patient metrics NEVER logged | Implemented |
| A09 | Logging | HMAC keys NEVER logged | Implemented |


### 10.2.2 Monitoring Security Checklist

- [x] HMAC-SHA256 heartbeat signing with 24h key rotation
- [x] Heartbeat field validation (engine_id, timestamp, latency_ms)
- [x] Engine ID allowlist from config (reject unknown engines)
- [x] Config parameter bounds enforced (6 parameters)
- [x] Unknown YAML keys rejected
- [x] `monitoring_log.json` read-only after write (chmod 444)
- [x] `security_audit_log.json` append-only enforcement
- [x] FSM transitions validated against defined rules
- [x] Circular buffer bounds verified per engine
- [x] Artifact SHA-256 hashes verified post-export
- [x] HIPAA-compliant logging (aggregates only)
- [x] Phase 0 security controls reused (never duplicated)

### 10.2.3 HIPAA Logging Compliance (A09)

| Data Category | Logged? | Justification |
|---------------|---------|---------------|
| State transitions (engine_id + old/new state) | Yes | Non-PHI: system state only |
| Aggregate monitoring counts | Yes | Non-PHI: population-level counts |
| Security violation type + engine_id | Yes | Non-PHI: system security event |
| Number of engines monitored | Yes | Non-PHI: infrastructure count |
| Alert severity/category counts | Yes | Non-PHI: aggregate statistics |
| Raw heartbeat payload | **NEVER** | May contain timing side-channels |
| Per-engine latency values | **NEVER** | HIPAA: correlatable to patient load |
| HMAC keys or signatures | **NEVER** | Cryptographic material |
| Patient metrics or identifiers | **NEVER** | HIPAA: protected health information |
| Per-sample anomaly scores | **NEVER** | HIPAA: individual risk predictions |

### 10.2.4 Data Integrity Assertions (A08)

All assertions are evaluated at runtime and logged with pass/fail status.

| Assertion | Expected | Actual | Status |
|-----------|----------|--------|--------|
| FSM transitions follow defined rules | 0 invalid transitions | 0 invalid in 9 transitions | PASS |
| Circular buffer bounded | <= 1000 per engine | all 5 engines within bounds | PASS |
| Engine IDs in allowlist | all IDs in 5-engine allowlist | all 5 IDs valid | PASS |
| Heartbeat count within bounds | <= 50 | 9 | PASS |
| Artifact hashes verified | 0 violations | 4 verified, 0 violations | PASS |
| Audit log append-only | monotonically increasing size | PASS — no truncation | PASS |

**Overall:** ALL PASSED

### 10.2.5 Artifact Integrity (A02)

SHA-256 hashes computed after export and verified via Phase 0 IntegrityVerifier.

| Artifact | SHA-256 |
|----------|---------|
| `monitoring_log.json` | `9cb88d4235041cd53f29e6aa29cb28e3586197411bd727c2ffa898cfee15c99c` |
| `health_report.json` | `d5a9288fe3c8b98b9dc1b8e8e6772a973af38732323d81f0badf9376f7e2b40f` |
| `performance_report.json` | `7c8a7995d7791c5ea058e659b1f7b3c999fa211b2dd425dadb3ea39aa3655315` |
| `security_audit_log.json` | `cfd05db5a0c18b08b24285c501bfae0b15a510f4ccb2f8d649b4a252068893d8` |

### 10.2.6 Parameter Bounds Validation (A05)

| Parameter | Allowed Range | Configured Value | Status |
|-----------|--------------|-----------------|--------|
| `heartbeat_interval_seconds` | [1, 60] | 5 | PASS |
| `missed_heartbeat_threshold` | [1, 20] | 5 | PASS |
| `circular_buffer_size` | [100, 10000] | 1000 | PASS |
| `artifact_integrity_check_interval` | [10, 3600] | 60 | PASS |
| `n_cycles` | [1, 100] | 5 | PASS |
| `latency_p95_threshold_ms` | [10.0, 1000.0] | 100.0 | PASS |
| Unknown YAML keys | none allowed | 0 found | PASS |

### 10.2.7 HMAC-SHA256 Heartbeat Authentication (A02)

| Property | Value |
|----------|-------|
| Algorithm | HMAC-SHA256 |
| Key size | 256-bit (32 bytes) |
| Key generation | `os.urandom(32)` |
| Rotation interval | 24 hours |
| Grace window | Previous key retained after rotation |
| Message format | `engine_id|timestamp|latency_ms` |
| Comparison | `hmac.compare_digest()` (timing-safe) |

### 10.2.8 Threat Model

| Threat | Mitigation | OWASP |
|--------|------------|-------|
| Spoofed heartbeat | HMAC-SHA256 + engine ID allowlist | A02, A03 |
| Replay attack | ISO 8601 timestamp validation | A03 |
| Latency injection | Bounded [0.0, 10000.0] ms | A03 |
| Config poisoning | Frozenset allowlist rejection | A05 |
| Buffer overflow | Circular buffer deque(maxlen) | A08 |
| Audit log tampering | Append-only size monitoring | A01, A08 |
| Artifact modification | Read-only chmod 444 + SHA-256 | A01, A02 |
| Key compromise | 24h rotation + grace window | A02 |
| Privilege escalation | Input validation on all data | A01, A03 |
| Data leak via logs | HIPAA logging (aggregates only) | A09 |

### 10.2.9 Security Inheritance from Phase 0

| Control | Source | Reuse Method |
|---------|--------|-------------|
| SHA-256 hashing | Phase 0 `IntegrityVerifier` | Direct import — `compute_hash()` |
| Path traversal | Phase 0 `PathValidator` | Direct import — `validate_output_dir()` |
| Config sanitization | Phase 0 `ConfigSanitizer` | Direct import — `sanitize_config_dict()` |
| Audit logging | Phase 0 `AuditLogger` | Direct import — `log_security_event()` |

---

**Reviewer signature:** ____________________  **Date:** ____/____/______
