## 8.3 Explanation Engine — Reproducibility & CI/CD Validation

This section documents the Phase 5 Explanation Engine reproducibility
validation, including SHAP computation performance, visualization
artifact verification, and full pipeline provenance.

### 8.3.1 SHAP Computation Performance

SHAP values computed using GradientExplainer with Integrated Gradients
fallback. Background samples drawn exclusively from Normal (Label=0)
training data. All timings measured end-to-end including model
inference, gradient computation, and attribution aggregation.

| Environment | Samples | Background | Top Features | Time | Timeout |
|-------------|---------|------------|--------------|------|---------|
| CI | 20 | 10 | 10 | <60s | 60s (hard fail) |
| Production | 198 | 100 | 10 | 21.68s | None |

**CI optimization parameters:**

| Parameter | CI Value | Production Value | Rationale |
|-----------|----------|------------------|-----------|
| `background_samples` | 10 | 100 | Reduced background for faster IG |
| `max_explain_samples` | 20 | 200 | Subset for CI speed |
| `random_state` | 42 | 42 | Deterministic in both environments |
| `TF_DETERMINISTIC_OPS` | 1 | 1 | Bit-exact TF operations |

Config selection: `PHASE5_ENV=ci` loads `config/phase5_ci_config.yaml`,
`PHASE5_ENV=production` (default) loads `config/phase5_config.yaml`.

### 8.3.2 Visualization Artifacts

| Chart Type | Trigger Condition | Count (Production) | Format | Validation |
|------------|-------------------|-------------------|--------|------------|
| Waterfall | HIGH or CRITICAL alert | 5 | PNG | SHAP contribution bars per feature |
| Bar chart | All alerts (aggregate) | 1 | PNG | Top 10 features by mean |SHAP| |
| Line graph | Post-incident timeline | 3 | PNG | x-axis = timesteps (0 to 19) |

**Chart validation in CI:**
- Waterfall charts generated only for HIGH/CRITICAL samples (A08 assertion)
- Bar chart contains exactly `top_features` (10) ranked features
- Line graph x-axis spans model timesteps (0 to 19)
- Visual regression: chart dimensions compared against baselines in
  `tests/baseline_charts/` (skipped if baselines not yet generated)
- All charts validated as non-empty PNG files (minimum 100x100 pixels)

**Headless rendering:** `MPLBACKEND=Agg` set in Dockerfile and CI
environment — no display server required.

### 8.3.3 Full Pipeline Artifact Chain

```
raw CSV (WUSTL-EHMS)
  → [Phase 0] → stats_report.json, dataset_integrity.json
  → [Phase 1] → train_phase1.parquet, test_phase1.parquet, robust_scaler.pkl
  → [Phase 2] → detection_model.weights.h5, attention_output.parquet
  → [Phase 3] → classification_model.weights.h5, metrics_report.json
  → [Phase 4] → baseline_config.json, risk_report.json, drift_log.csv
  → [Phase 5] → shap_values.parquet, explanation_report.json, charts/
```

Each phase verifies predecessor artifacts via SHA-256 hash comparison
against stored metadata before processing. Phase 5 verifies 10 upstream
artifacts (2 from Phase 2 + 4 from Phase 3 + 4 from Phase 4) before
computing SHAP explanations.

| Phase | Input Artifacts | Output Artifacts | Integrity |
|-------|-----------------|------------------|-----------|
| 0 | Raw CSV | stats, integrity JSON | SHA-256 |
| 1 | Phase 0 stats | train/test parquet, scaler | SHA-256 |
| 2 | Phase 1 parquet | model weights, attention parquet | SHA-256 |
| 3 | Phase 2 model | classification weights, metrics | SHA-256 |
| 4 | Phase 3 + Phase 2 | baseline, threshold, risk, drift | SHA-256 |
| 5 | Phase 4 + Phase 2 | SHAP values, explanations, charts | SHA-256 |

### 8.3.4 CI/CD Pipeline Architecture

```
push/PR → lint-phase5 → test-phase5 ──────────┐
                       → security-scan-phase5 ─┤
                                               ├→ integration-test → build
```

| Job | Gate | Tool |
|-----|------|------|
| lint-phase5 | ruff + black | Static analysis |
| test-phase5 | 80% coverage | pytest-cov (94 tests) |
| security-scan-phase5 | bandit + pip-audit | SAST + CVE |
| integration-test | 9 artifact assertions | Phase 0→1→2→3→4→5 |
| build | Docker image | analyst/phase0-phase5:6.0 |

**Integration test assertions (9 total):**

| # | Assertion | Check |
|---|-----------|-------|
| 1 | `shap_values.parquet` non-empty | `len(df) > 0` |
| 2 | `explanation_report.json` has risk levels | All non-NORMAL levels present |
| 3 | `charts/` contains waterfall, bar, line | File name pattern matching |
| 4 | NORMAL excluded from SHAP | No `risk_level == "NORMAL"` in explanations |
| 5 | Waterfall for HIGH/CRITICAL only | A08 metadata assertion status |
| 6 | Feature importance ranking verified | A08 metadata assertion status |
| 7 | All 6 integrity assertions passed | Metadata check |
| 8 | SHAP duration < 60s in CI | `duration_seconds < 60` |
| 9 | Artifacts read-only (chmod 444) | `stat.S_IWUSR` bit cleared |

### 8.3.5 SBOM (Software Bill of Materials)

CycloneDX SBOM generated during CI/CD security scan phase:
- Format: CycloneDX JSON
- Scope: All Phase 0–5 dependencies from `requirements.txt`
- Phase 5 additions: `shap>=0.42.0` (matplotlib, plotly already present)
- CVE policy: Fail build if any dependency has CVSS > 7.0
- Artifact: `sbom-phase5.json` (uploaded as CI artifact)
- SBOM verification: `shap` presence asserted in component list

### 8.3.6 Reproducibility Statement

This Phase 5 Explanation Engine produces **deterministic results** given:

1. **Fixed random seed** (`random_state=42`, `TF_DETERMINISTIC_OPS=1`)
2. **Immutable SHAP background** (Normal-only, SHA-256 verified source)
3. **Versioned model weights** (SHA-256 chain from Phase 2 → 3 → 4)
4. **Locked dependencies** (`requirements.txt` + CycloneDX SBOM)
5. **Git commit tracking** (embedded in `explanation_metadata.json`)

The full Phase 0→1→2→3→4→5 pipeline can be reproduced by:

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
python -m src.phase5_explanation_engine.security_hardened_phase5

# 4. Verify artifacts
python -c "import json; m=json.load(open('data/phase5/explanation_metadata.json')); print(f'OK: {m[\"samples_explained\"]} samples, {len(m[\"artifact_hashes\"])} hashes')"
```

Docker reproducibility:

```bash
docker build -t analyst/phase0-phase5:6.0 .
docker run --rm analyst/phase0-phase5:6.0
```

SHAP values deterministic: background samples fixed (`random_state=42`),
Normal-only filtering verified at runtime (A08 assertion), Integrated
Gradients with `n_integration_steps=50` for consistent attribution.

### 8.3.7 Peer Review Readiness Checklist

- [x] All 6 phases tested end-to-end (Phase 0 → 1 → 2 → 3 → 4 → 5)
- [x] 94 Phase 5 unit tests passing (80%+ coverage)
- [x] Charts generated and validated (waterfall, bar, timeline)
- [x] No HIPAA violations in explanations (A08 biometric assertion)
- [x] SHAP background verified: Normal samples only (A08 assertion)
- [x] Docker image reproducible (`analyst/phase0-phase5:6.0`)
- [x] SBOM generated with CVE policy (CVSS > 7.0 fails build)
- [x] All explanation templates use feature names only (no raw values)
- [x] Artifacts set to read-only (chmod 444) after export
- [x] SHA-256 hashes stored in `explanation_metadata.json`

---

**Generated:** 2026-03-19 21:01:07 UTC
**Test framework:** pytest + pytest-cov
**Pipeline version:** 6.0
