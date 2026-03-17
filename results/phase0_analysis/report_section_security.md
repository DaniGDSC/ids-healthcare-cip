## 3.3 Security Architecture and HIPAA Compliance

This section documents the security controls implemented in the Phase 0
analysis pipeline, mapped to the OWASP Top 10 (2021) risk framework and
HIPAA Safe Harbor de-identification requirements [WUSTL-EHMS-2020].

### 3.3.1 OWASP Controls Implemented

| OWASP ID | Risk Category             | Control Implemented                                         | Status      |
|----------|---------------------------|-------------------------------------------------------------|-------------|
| A01      | Broken Access Control     | Path traversal rejection (`..`, `~`, `$` patterns blocked)  | Implemented |
| A01      | Broken Access Control     | Workspace containment — all paths resolved within project   | Implemented |
| A01      | Broken Access Control     | Read-only advisory check on raw dataset (chmod 444)         | Implemented |
| A02      | Cryptographic Failures    | SHA-256 integrity hash computed on every dataset load        | Implemented |
| A02      | Cryptographic Failures    | Hash stored in `dataset_integrity.json` and verified        | Implemented |
| A03      | Injection                 | Config string sanitization (regex allowlist)                 | Implemented |
| A03      | Injection                 | Column name validation against DataFrame allowlist           | Implemented |
| A03      | Injection                 | No `eval()` or `exec()` used anywhere in pipeline            | Verified    |
| A05      | Security Misconfiguration | Config schema validation via typed dataclass                 | Implemented |
| A05      | Security Misconfiguration | Bound enforcement: `correlation_threshold` ∈ (0, 1)         | Implemented |
| A05      | Security Misconfiguration | Bound enforcement: `outlier_iqr_multiplier` > 0             | Implemented |
| A09      | Security Logging          | All file access events logged with ISO-8601 timestamps       | Implemented |
| A09      | Security Logging          | Biometric values NEVER logged — column names only            | Implemented |
| A09      | Security Logging          | Dedicated `phase0.security.audit` logger for security events | Implemented |

### 3.3.2 Dataset Integrity Verification

The raw dataset is verified using SHA-256 on every load:

```
Algorithm  : SHA-256
Digest     : 8359da96154fa60247df9d75e52d232077acb9886c09c36de2a0aaaee6cf2c25
Stored in  : results/phase0_analysis/dataset_integrity.json
```

On first execution, the hash is computed and stored as a baseline.
On subsequent executions, the recomputed hash is compared against the
stored value. Any mismatch raises an `IntegrityError` and halts the
pipeline, preventing analysis of tampered or corrupted data.

This confirms that all experimental results are derived from an
unmodified copy of the original WUSTL-EHMS-2020 dataset.

### 3.3.3 Data Anonymisation Statement

Network identifiers removed in Phase 1 preprocessing (HIPAA Safe Harbor):
[`SrcAddr`, `DstAddr`, `Sport`, `Dport`, `SrcMac`, `DstMac`, `Dir`, `Flgs`]

These 8 columns encode environment-specific network topology
artefacts (IP addresses, MAC addresses, port numbers, direction/flag fields).
Their removal serves two purposes:

1. **HIPAA compliance** — network identifiers constitute Protected Health
   Information (PHI) when associated with patient medical devices in a
   hospital IoMT deployment.
2. **Generalisation** — models trained on source/destination pairs memorise
   capture topology rather than learning transferable intrusion signatures.

Biometric columns (`DIA`, `Heart_rate`, `Pulse_Rate`, `Resp_Rate`, `ST`, `SYS`, `SpO2`, `Temp`) are processed in-memory only.
Their values are **never written to log files**, audit trails, or
intermediate artifacts. Only column names appear in analysis reports.

### 3.3.4 Security Assumptions

This pipeline assumes deployment in a HIPAA-compliant environment with
the following baseline security posture:

1. **Physical security** — the host machine resides in a physically
   secured facility with access controls (badge, biometric entry).
2. **OS-level access control** — the dataset directory is readable only
   by the pipeline service account; no shared or world-readable permissions.
3. **Encrypted storage** — the host filesystem uses full-disk encryption
   (e.g., LUKS, BitLocker, FileVault) for data-at-rest protection.
4. **Network isolation** — the analysis pipeline executes on an air-gapped
   or network-segmented machine with no outbound Internet access during
   processing.
5. **Audit retention** — security audit logs are retained for a minimum
   of 6 years per HIPAA §164.530(j).
6. **Reproducibility** — all experiments use `random_state=42`
   and stratified split 70/30
   with version-controlled code and externalised configuration.

### 3.3.5 Threat Model Boundaries

The following threats are **out of scope** for this Phase 0 pipeline:

- Side-channel attacks on the host CPU (e.g., Spectre, Meltdown)
- Supply-chain attacks on Python dependencies
- Adversarial ML attacks on downstream models (addressed in Phase 2+)
- Insider threats with root/admin access to the host machine
