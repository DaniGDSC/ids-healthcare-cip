## 3.4 Reproducibility and Environment Specification

This section documents the computational environment, random seed configuration, CI/CD pipeline status, and dataset versioning to ensure full reproducibility of the reported results as required by IEEE Q1 standards.

### 3.4.1 Environment Specification

- **Python**: 3.10.20
- **Platform**: Linux 6.17.0-23-generic
- **Architecture**: x86_64

| Package | Version |
|---------|---------|
| tensorflow | 2.21.0 |
| keras | 3.12.1 |
| pandas | 2.3.3 |
| numpy | 2.2.6 |
| scikit-learn | 1.7.2 |
| scipy | 1.15.3 |
| imbalanced-learn | 0.14.1 |
| hdbscan | not installed |
| matplotlib | 3.10.8 |
| pyyaml | 6.0.3 |
| cryptography | 46.0.6 |
| pytest | 9.0.2 |

Full dependency list is maintained in `requirements.txt` (12 packages total).

### 3.4.2 Experiment Reproducibility

All random seeds are fixed at `random_state=42`. The dataset is split 70/30 (train/test) with stratified sampling to preserve class priors.

Dataset SHA-256: `8359da96154fa60247df9d75e52d232077acb9886c09c36de2a0aaaee6cf2c25`

Results are reproducible via:

```bash
docker build -t analyst/phase0 .
docker run --rm -v $(pwd)/data:/home/analyst/app/data analyst/phase0
```

### 3.4.3 CI/CD Pipeline Summary

The project employs a four-stage GitHub Actions pipeline:

| Stage | Tool | Status |
|-------|------|--------|
| Lint | ruff + black | Enforced |
| Security | bandit + pip-audit | 0 critical findings |
| Test | pytest | 0 tests passing |
| Coverage | pytest-cov | 0.0% (≥ 80% required) |
| Build | Docker | Image verified |
| SBOM | CycloneDX | Generated per build |

Pre-commit hooks enforce black, ruff, bandit, and detect-secrets on every local commit.

### 3.4.4 Dataset Versioning

| Attribute | Value |
|-----------|-------|
| Dataset | WUSTL-EHMS-2020 |
| SHA-256 | `8359da96154fa60247df9d75e52d2320...` |
| Source | [https://www.cse.wustl.edu/~jain/ehms/index.html](https://www.cse.wustl.edu/~jain/ehms/index.html) |
| Integrity | Verified via `IntegrityVerifier` on every load; baseline is ECDSA P-256 signed |

The signed SHA-256 baseline is established once via `python -m module0_analysis.phase0.bootstrap_integrity` and stored in `module0_analysis/phase0/dataset_integrity.json`. Every subsequent `DataLoader.load()` call reads the dataset bytes into memory, recomputes the hash, verifies the ECDSA P-256 signature on the baseline (using the Module 5 audit-signing key), and only then parses the in-memory buffer with pandas. Any modification to the raw dataset, the baseline file, or the signature triggers an `IntegrityError`, preventing silent data corruption — or a tamper-then-rebaseline attack — from affecting experimental results.

### 3.4.5 Peer Review Readiness Checklist

- [ ] Code publicly available at [GitHub URL]
- [ ] Dataset publicly available at [https://www.cse.wustl.edu/~jain/ehms/index.html](https://www.cse.wustl.edu/~jain/ehms/index.html)
- [ ] Docker image builds and passes health check
- [ ] All tests pass with ≥ 80% coverage
- [ ] Security scan shows 0 critical/high findings
- [ ] SBOM (CycloneDX) generated and archived
- [ ] `report_section_*.md` artifacts uploaded per CI run
- [ ] `random_state=42` used across all stochastic operations
- [ ] Dataset SHA-256 hash recorded and verified on load
