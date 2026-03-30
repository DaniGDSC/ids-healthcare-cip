# Deployment Guide — RA-X-IoMT IDS

Production deployment guide for WUSTL-compatible hospital networks.

## Prerequisites

- Docker 24+ with Compose v2
- NVIDIA GPU (optional, CPU inference ~190ms)
- Hospital network TAP on IoMT VLAN (read-only mirror port)
- HL7v2 interface engine (Mirth Connect, Rhapsody) for biometrics
- SIEM (Splunk or QRadar) for alert forwarding
- SMTP relay for encrypted email notifications
- LDAP/Active Directory for user authentication

## 1. Certificate Setup (mTLS)

Generate certificates for inter-service communication:

```bash
mkdir -p /etc/iomt-ids/certs
cd /etc/iomt-ids/certs

# CA
openssl req -x509 -newkey rsa:4096 -days 365 -nodes \
  -keyout ca-key.pem -out ca.pem \
  -subj "/CN=iomt-ids-ca"

# Server cert (for FastAPI, WebSocket)
openssl req -newkey rsa:4096 -nodes -keyout server-key.pem \
  -out server.csr -subj "/CN=iomt-ids-server"
openssl x509 -req -in server.csr -CA ca.pem -CAkey ca-key.pem \
  -CAcreateserial -out server.pem -days 365

# Client cert (for Kafka, SIEM, SMTP)
openssl req -newkey rsa:4096 -nodes -keyout client-key.pem \
  -out client.csr -subj "/CN=iomt-ids-client"
openssl x509 -req -in client.csr -CA ca.pem -CAkey ca-key.pem \
  -CAcreateserial -out client.pem -days 365
```

## 2. Environment Configuration

```bash
# .env file (DO NOT commit to git)
IOMT_CERT_DIR=/etc/iomt-ids/certs
IOMT_SIEM_HOST=splunk.hospital.internal
IOMT_SIEM_PORT=8088
IOMT_SIEM_TOKEN=<splunk-hec-token>
IOMT_SMTP_HOST=smtp.hospital.internal
IOMT_SMTP_PORT=587
IOMT_LDAP_HOST=ldap.hospital.internal
IOMT_LDAP_BASE_DN=dc=hospital,dc=internal
IOMT_DEVICE_ID=generic_iomt_sensor
TF_DETERMINISTIC_OPS=1
CUDA_VISIBLE_DEVICES=0
```

## 3. Docker Deployment

```bash
docker compose up -d

# Verify health
docker compose exec pipeline curl http://localhost:8000/health
docker compose exec monitoring curl http://localhost:7000/health
```

## 4. SIEM Integration

### Splunk (HTTP Event Collector)

1. Create HEC token in Splunk: Settings → Data Inputs → HTTP Event Collector
2. Set sourcetype to `iomt:alert`
3. Configure in `config/phase4_config.yaml`:

```yaml
cia:
  enabled: true
  default_device_id: "generic_iomt_sensor"
```

### QRadar (syslog)

1. Add log source: Admin → Log Sources → Add
2. Protocol: UDP syslog, port 514
3. Log Source Type: Universal CEF

## 5. Biometric Integration (HL7v2)

Configure the hospital's interface engine to forward ORU^R01 messages:

| LOINC Code | Feature | Description |
| --- | --- | --- |
| 8310-5 | Temp | Body temperature |
| 59408-5 | SpO2 | Oxygen saturation |
| 8889-8 | Pulse_Rate | Pulse rate |
| 8480-6 | SYS | Systolic blood pressure |
| 8462-4 | DIA | Diastolic blood pressure |
| 8867-4 | Heart_rate | Heart rate |
| 9279-1 | Resp_Rate | Respiratory rate |

The BiometricBridge listens on the configured HL7 port and extracts these values.
Stale cache threshold: 60 seconds (configurable).

## 6. Authentication

### Mode: LDAP (production)

```yaml
# In API configuration
auth:
  mode: ldap
  ldap:
    host: ldap.hospital.internal
    port: 636
    use_ssl: true
    base_dn: "dc=hospital,dc=internal"
    user_dn_template: "cn={},ou=users,dc=hospital,dc=internal"
    role_mapping:
      "iomt-security": "IT Security Analyst"
      "iomt-clinical-it": "Clinical IT Administrator"
      "iomt-physicians": "Attending Physician"
      "iomt-managers": "Hospital Manager"
      "iomt-auditors": "Regulatory Auditor"
```

### Mode: Local (pilot)

```python
from src.production.auth import AuthProvider
auth = AuthProvider(mode="local", users={
    "admin": {"password_hash": AuthProvider.hash_password("changeme"), "role": "IT Security Analyst"},
})
```

## 7. HIPAA Compliance Checklist

- [ ] PHI columns removed in Phase 1 (SrcAddr, DstAddr, Sport, Dport, SrcMac, DstMac)
- [ ] Device IDs pseudonymized (SHA-256, 12-char prefix)
- [ ] Email bodies contain risk level + timestamp only (no patient data)
- [ ] SIEM payloads contain no PHI (CEF format, pseudonymized)
- [ ] Audit log enabled at `data/audit/fda_audit.jsonl`
- [ ] Audit log on append-only volume mount
- [ ] Dashboard access restricted by role
- [ ] TLS 1.2+ enforced on all connections
- [ ] Biometric values never transmitted in alerts

## 8. FDA 21 CFR Part 11 Audit

Verify audit log integrity:

```python
from src.production.audit_logger import FDAAuditLogger
logger = FDAAuditLogger("data/audit/fda_audit.jsonl")
result = logger.verify_chain()
print(f"Valid: {result['is_valid']}, Entries: {result['entries_checked']}")
```

## 9. Monitoring (Grafana)

Prometheus metrics available at `GET /metrics`:

- `iomt_inference_total` — total inferences
- `iomt_inference_latency_ms` — latency histogram
- `iomt_alerts_emitted_total` — alerts by severity
- `iomt_alerts_suppressed_total` — fatigue-suppressed
- `iomt_drift_events_total` — concept drift detections
- `iomt_buffer_flows` — current buffer size

Import the Grafana dashboard template from `config/grafana_dashboard.json` (if provided).

## 10. Rollout Phases

| Phase | Duration | Alert Recipients | Device Action |
| --- | --- | --- | --- |
| Shadow | Weeks 1-4 | IT Security only | None (observe) |
| Passive | Weeks 5-8 | IT Security + Clinical IT | Log + restrict |
| Active | Weeks 9-12 | All 5 roles | Full protocol |
| Expansion | Months 4-6 | Additional wards | Same protocol |

## 11. Disaster Recovery

### Model rollback

```bash
# List available model versions
ls -la data/phase2_5/finetuned_model.weights.h5*

# Restore previous version
cp data/phase2_5/finetuned_model.weights.h5.bak data/phase2_5/finetuned_model.weights.h5

# Restart inference service
docker compose restart pipeline
```

### Threshold reset

If drift detection locks thresholds and doesn't recover:

```bash
# Reset to baseline
python3 -c "
from src.phase4_risk_engine.phase4.fallback_manager import ThresholdFallbackManager
# Manual reset documented in Phase 4 config
"
```

## 12. Performance Tuning

| Parameter | Default | Tuning guidance |
| --- | --- | --- |
| Window size | 20 | Increase for longer attack patterns, decrease for faster detection |
| Calibration threshold | 100 | Lower for faster startup, higher for more stable baseline |
| Alert aggregation window | 10 | Lower = more alerts, higher = more suppression |
| Rate limit (per 100 samples) | 5 | Adjust based on SOC capacity |
| MAD multiplier | 3.0 | Lower = more sensitive, higher = fewer alerts |
| CIA escalation threshold | 0.7 | Lower = more escalations, higher = fewer |
| Stale biometric threshold | 60s | Based on monitor polling frequency |
