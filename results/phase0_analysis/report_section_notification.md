## 9.1 Notification Engine — Alert Routing & Delivery

This section documents the Phase 6 Notification Engine,
which routes Phase 5 SHAP-explained alerts to appropriate
notification channels based on risk level.

### 9.1.1 Alert Distribution

| Risk Level | Count | Routing |
|------------|-------|---------|
| CRITICAL | 7 | Dashboard + Email + Alarm + Escalation |
| HIGH | 170 | Dashboard + Email (waterfall attached) |
| MEDIUM | 10 | Dashboard |
| LOW | 11 | Log only |
| **Total** | **198** | |

### 9.1.2 Delivery Summary

| Metric | Value |
|--------|-------|
| Total delivery attempts | 399 |
| Successful | 399 |
| Failed | 0 |
| Success rate | 100.0% |
| Escalations triggered | 7 |
| Retry policy | 3 attempts, exponential backoff |

### 9.1.3 Channel Performance

| Channel | Success | Failed | Total |
|---------|---------|--------|-------|
| dashboard | 187 | 0 | 187 |
| dashboard_doctor | 7 | 0 | 7 |
| dashboard_it_admin | 7 | 0 | 7 |
| email | 177 | 0 | 177 |
| email_it_admin | 7 | 0 | 7 |
| email_manager | 7 | 0 | 7 |
| onsite_alarm | 7 | 0 | 7 |

### 9.1.4 HIPAA Compliance

| Field | In Email | In Dashboard | In Logs |
|-------|----------|-------------|---------|
| Risk level | Yes | Yes | Yes |
| Timestamp | Yes | Yes | Yes |
| Top 3 features (names only) | No | Yes | No |
| Dashboard URL | Yes | N/A | No |
| Device ID | **NEVER** | **NEVER** | **NEVER** |
| Patient data | **NEVER** | **NEVER** | **NEVER** |
| Raw anomaly scores | **NEVER** | **NEVER** | **NEVER** |
| SHAP values | **NEVER** | **NEVER** | **NEVER** |
| Biometric readings | **NEVER** | **NEVER** | **NEVER** |
| Recipient address | **NEVER** | N/A | Hashed (SHA-256) |

### 9.1.5 Email Protocol

| Parameter | Value |
|-----------|-------|
| Encryption | TLS TLSv1.3 minimum |
| Certificate verification | Required before send |
| Subject format | `RA-{idx}-IoMT Alert — Level: {RISK}` |
| Attachment | Waterfall chart (HIGH/CRITICAL only) |
| PHI in body | None |

### 9.1.6 Escalation Protocol (CRITICAL)

| Step | Target | Channel |
|------|--------|---------|
| 1 | On-site alarm | Hospital network API |
| 2 | IT admin | Email + Dashboard |
| 3 | Doctor on duty | Dashboard |
| 4 | Manager | Email |

Confirmation required within 300s (5 minutes).
If no confirmation: re-escalate to next level.

### 9.1.7 Artifact Integrity

| Artifact | SHA-256 |
|----------|---------|
| `notification_log.json` | `ecb42b4eb4ecb5fa…` |
| `delivery_report.json` | `0fffb5a0eb6a95e1…` |
| `escalation_log.json` | `7921971e84075105…` |

### 9.1.8 Execution Details

- Duration: 0.02s
- Git commit: `901983542dbc`
- Pipeline: phase6_notification

---

**Generated:** 2026-03-19 15:01:11 UTC
**Pipeline version:** 6.0
