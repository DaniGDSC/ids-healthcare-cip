## 7.3 Risk-Adaptive Engine — Reproducibility & CI/CD Validation

This section documents the Phase 4 Risk-Adaptive Engine reproducibility
validation, including latency benchmarks, baseline integrity verification,
and concept drift simulation results.

### 7.3.1 Latency Benchmarks

All benchmarks run on 1000 samples, 50 iterations.

| Component | p50 | p95 | p99 | SLA | Result |
|-----------|-----|-----|-----|-----|--------|
| RiskScorer | 4.42ms | 4.59ms | 4.65ms | <100ms | PASS |
| ThresholdUpdater | 14.07ms | 14.43ms | 15.29ms | <50ms | PASS |
| DriftDetector | 0.0002ms | 0.0002ms | 0.0002ms | <1ms | PASS |

**Benchmark parameters:**

| Parameter | Value |
|-----------|-------|
| N samples | 1000 |
| N features | 20 |
| Scorer iterations | 50 |
| Updater iterations | 50 |
| Detector iterations | 1000 |
| Window size | 100 |

### 7.3.2 Baseline Integrity Results

| Test | Status |
|------|--------|
| Tamper detection (SHA-256) | PASS |
| Hash round-trip consistency | PASS |
| Write-once enforcement (chmod 444) | PASS |
| Baseline immutable keys present | PASS |

**Integrity mechanism:** SHA-256 hash verification + chmod 444 write-once enforcement.

### 7.3.3 Drift Simulation Results

| Scenario | Status | Detail |
|----------|--------|--------|
| 25% shift → fallback triggered | PASS | 1 FALLBACK_LOCKED events |
| 15% shift → no fallback | PASS | 0 FALLBACK_LOCKED events |
| Recovery after stable windows | PASS | 1 RESUMED_DYNAMIC events |

**Drift parameters:**

| Parameter | Value |
|-----------|-------|
| Drift threshold | 20% (trigger fallback) |
| Recovery threshold | 10% (resume dynamic) |
| Recovery windows | 3 consecutive |
| Window size | 50 samples |
| Baseline threshold | 0.225 |

### 7.3.4 Full Pipeline Artifact Chain

| Phase | Input | Output | Hash |
|-------|-------|--------|------|
| 0 | Raw CSV | stats, integrity JSON | SHA-256 |
| 1 | Phase 0 stats | train/test parquet, metadata | SHA-256 |
| 2 | Phase 1 parquet | model weights, attention parquet | SHA-256 |
| 3 | Phase 2 model | classification weights, metrics | SHA-256 |
| 4 | Phase 3 + Phase 2 | baseline, threshold, risk, drift | SHA-256 |

Each phase verifies predecessor artifacts via SHA-256 hash comparison
against stored metadata before processing.

### 7.3.5 CI/CD Pipeline Architecture

```
push/PR → lint-phase4 → test-phase4 ──────────┐
                       → security-scan-phase4 ─┤
                                               ├→ benchmark-test → integration-test → build
```

| Job | Gate | Tool |
|-----|------|------|
| lint-phase4 | ruff + black | Static analysis |
| test-phase4 | 80% coverage | pytest-cov |
| security-scan-phase4 | bandit + pip-audit | SAST + CVE |
| benchmark-test | p95 SLA assertions | pytest-benchmark |
| integration-test | 5 artifact assertions | Phase 0→1→2→3→4 |
| build | Docker image | phase0-phase4:5.0 |

### 7.3.6 SBOM (Software Bill of Materials)

CycloneDX SBOM generated during CI/CD security scan phase:
- Format: CycloneDX JSON
- Scope: All Phase 0–4 dependencies from requirements.txt
- CVE policy: Fail build if any dependency has CVSS > 7.0
- Artifact: `sbom-phase4.json` (uploaded as CI artifact)

### 7.3.7 Reproducibility Statement

This Phase 4 Risk-Adaptive Engine produces **deterministic results** given:

1. **Fixed random seed** (`random_state=42`, `TF_DETERMINISTIC_OPS=1`)
2. **Immutable baseline** (Median + k*MAD, write-once, SHA-256 verified)
3. **Versioned model weights** (SHA-256 hashes in classification_metadata.json)
4. **Locked dependencies** (requirements.txt + CycloneDX SBOM)
5. **Git commit tracking** (embedded in all output artifacts)

The full Phase 0→1→2→3→4 pipeline can be reproduced by:

```bash
# 1. Clone at specific commit
git clone <repo> && git checkout <commit>

# 2. Install locked dependencies
pip install -r requirements.txt

# 3. Run full pipeline
python -m src.phase0_dataset_analysis.phase0
python -m src.phase1_preprocessing.phase1
python -m src.phase2_detection_engine.phase2
python -m src.phase3_classification_engine.phase3
python -m src.phase4_risk_engine.phase4.pipeline

# 4. Verify artifacts
python -c "from src.phase4_risk_engine.phase4 import *; print('All imports OK')"
```

Docker reproducibility:

```bash
docker build -t analyst/phase0-phase4:5.0 .
docker run --rm analyst/phase0-phase4:5.0
```

---

**Generated:** 2026-03-28 14:20:12 UTC
**Test framework:** pytest + pytest-benchmark
**Pipeline version:** 5.0
