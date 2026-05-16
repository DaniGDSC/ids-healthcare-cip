# Sub-section 3.1.3 Extraction: Complete Pipeline Data Flow

**Extraction target:** Sub-section 3.1.3 — "Complete Pipeline Data Flow (Five Successive Transformations)"  
**Scope:** All pipeline modules M0–M6, `src/` runtime layer, `configs/`, `common/`, `tests/`  
**Label conventions:** `[IMPLEMENTED]` = evidenced in source code; `[DOCUMENTED_ONLY]` = evidenced only in comments/docs/ARCHITECTURE.md; `[INCONSISTENCY FLAG]` = conflict between two sources; `[DISCREPANCY]` = ARCHITECTURE.md claim vs actual implementation  
**Citation format:** `filepath:line_number` (relative to repo root)  
**Date extracted:** 2026-05-12  

---

## TRANSFORMATION T1: Raw Network Records → Feature-Engineered Parquet Artifacts

**Modules involved:** Module 0 (Dataset Audit), Module 1 (Preprocessing)

---

### T1-A — Dataset Integrity Verification Entry Point

| File path | Function/class | Line numbers | Finding | Status |
|-----------|----------------|-------------|---------|--------|
| `module0_analysis/phase0/security.py` | `IntegrityVerifier.verify_and_read` | 132–185 | Entry point for integrity verification. Hash algorithm: **SHA-256** (lines 52, 155). Signature scheme: **ECDSA P-256 with SHA-256** (lines 238–251). Exception type on failure: **`IntegrityError`** (lines 62–65, 176–179). Metadata file: `dataset_integrity.json` (line 53, verified at line 157). | [IMPLEMENTED] |
| `module0_analysis/phase0/loader.py` | `DataLoader.load` | 75–119 | TOCTOU elimination: file read once at line 101 via `verify_and_read()`, bytes returned and parsed via `io.BytesIO(verified_bytes)` at line 103. Single `open()` call — hash computed on the same buffer subsequently passed to the CSV parser. | [IMPLEMENTED] |
| `module0_analysis/phase0/security.py` | `IntegrityVerifier._read_bytes` | ~154 | Internal read path: opens file once at line 154, returns raw bytes. Caller (`verify_and_read`) hashes bytes at line 155 then returns bytes at line 185. No second file open occurs. | [IMPLEMENTED] |

**Signature bootstrap:** Explicit via `bootstrap_integrity` CLI (lines 95–130); no auto-baseline — prevents silent integrity-state establishment.

---

### T1-B — PHI Filtering at Analysis Stage

| File path | Function/class | Line numbers | Finding | Status |
|-----------|----------------|-------------|---------|--------|
| `common/phi.py` | Module constant | 18–29 | Canonical biometric column set: `BIOMETRIC_COLUMNS = {"Temp", "SpO2", "Pulse_Rate", "SYS", "DIA", "Heart_rate", "Resp_Rate", "ST"}` — eight columns. | [IMPLEMENTED] |
| `module0_analysis/phase0/analyzer.py` | `StatisticsAnalyzer.descriptive_stats` | 64–132 | Network features receive `{mean, median, std, min, max}` (lines 95–108). Biometric features restricted to `{mean, std}` only (lines 110–123). Min, max, and median explicitly excluded from biometric output. | [IMPLEMENTED] |
| `module0_analysis/phase0/analyzer.py` | `OutlierAnalyzer.outlier_report` | 333–411 | Biometric columns publish only `outlier_count`, `outlier_pct`, `total`; quantile-based fences (q1, q3, iqr) withheld (lines 379–388). **HIPAA Safe Harbor rationale** documented in docstring at lines 339–342. | [IMPLEMENTED] |

---

### T1-C — Leakage Barrier Implementation

| File path | Function/class | Line numbers | Leakage control | Verbatim extract | Status |
|-----------|----------------|-------------|----------------|-----------------|--------|
| `module1_preprocessing/phase1/pipeline.py` | `_pre_split_transforms` | 165–241 | Feature engineering (Steps 1–4: variance filter line 222–225, correlation-based redundancy lines 231–239) finalized **before** split at line 240 return. Leakage barrier marker at line 100. | `# ══════════════════ LEAKAGE BARRIER ═══════════════════════════` | [IMPLEMENTED] |
| `module1_preprocessing/phase1/pipeline.py` | `run` | 136–138 | RobustScaler fit restricted to training partition only: `scaler.scale_both(X_train, X_test)` internally calls `scaler.fit(X_train)` then transforms both (scaler.py:117–138). Test partition never participates in fit. | `X_train, X_test = scaler.scale_both(X_train, X_test)  # Line 138` | [IMPLEMENTED] |
| `module1_preprocessing/phase1/pipeline.py` | `run` / `_export_split_metadata` | 107, 122–134, 451–457 | Test and demo splits designated frozen at split time (line 107 comment). Invariant documented at lines 451–457: `"test and demo are FROZEN — never seen by any model in training"`. | `# Line 107 comment: frozen splits` | [IMPLEMENTED] |

---

### T1-D — Benign Medians Production

| File path | Function/class | Line numbers | Finding | Status |
|-----------|----------------|-------------|---------|--------|
| `data/processed/benign_medians.json` | Static artifact | 1–59 | Artifact exists. Key fields: `"n_benign_samples": 9990` (line 31), source: `"data/processed/train_benign_phase1.parquet"` (line 30). Format: JSON with `feature_names` array (lines 2–28) and `medians` dict (lines 32–58). Computation performed on benign-only training partition specifically. | [IMPLEMENTED] |
| `module1_preprocessing/phase1/pipeline.py` | `_export` | 318–332 | Benign-only train parquet exported at lines 319–325 (mask: `y_train == 0`); benign-only val at lines 326–332. | [IMPLEMENTED] |
| `src/preprocessing.py` | `load_benign_medians` | 66–72 | Benign medians consumed at inference: path `data/processed/benign_medians.json` (line 46); lazy-loaded on first `sanitize_features()` call (line 109); cached globally in `_BENIGN_MEDIANS`. | [IMPLEMENTED] |

**Note:** The script that *computes* benign_medians.json from the benign training parquet is not present in the current source tree. The artifact exists as a static file; the consumption code expects it to exist. The production pipeline for this artifact is not fully evidenced in code.

---

### T1-E — Module 1 Output Artifact Inventory

| Output file path | Format | Function / line | Description |
|------------------|--------|----------------|-------------|
| `data/processed/train_phase1.parquet` | Parquet | `PreprocessingExporter.export_parquet()` (pipeline.py:300–303) | Scaled training features (9,790 samples × 25 features), labels, multi-class attack categories, row IDs, device_class |
| `data/processed/val_phase1.parquet` | Parquet | `PreprocessingExporter.export_parquet()` (pipeline.py:304–307) | Scaled validation features (2,448 samples × 25 features); frozen, not used for model fitting |
| `data/processed/test_phase1.parquet` | Parquet | `PreprocessingExporter.export_parquet()` (pipeline.py:308–311) | Scaled test features (2,448 samples × 25 features); frozen for paper metrics only |
| `data/processed/demo_phase1.parquet` | Parquet | `PreprocessingExporter.export_parquet()` (pipeline.py:312–315) | Scaled demo features (1,632 samples × 25 features); frozen for dashboard and Phase 2 user study |
| `data/processed/benign_only_train.parquet` | Parquet | `PreprocessingExporter.export_parquet()` (pipeline.py:320–325) | Benign-only training subset (9,563 samples × 25 features); input for Track B DAE training |
| `data/processed/benign_only_val.parquet` | Parquet | `PreprocessingExporter.export_parquet()` (pipeline.py:327–332) | Benign-only validation subset (2,141 samples × 25 features); Track B validation |
| `data/processed/split_metadata.yaml` | YAML | `_export_split_metadata()` (pipeline.py:338–465) | Provenance document: random_state, ratios, sample counts, attack distributions, feature names, frozen-split invariants, source SHA-256 |
| `models/scalers/robust_scaler.json` | JSON | `PreprocessingExporter.export_scaler()` (pipeline.py:334) | Fitted RobustScaler sidecar (JSON, not pickle): `center_`, `scale_`, `n_features_in_` for all 25 features; fit on training partition only |
| `data/processed/categorical_encoder.json` | JSON | `CategoricalEncoder.save()` (pipeline.py:352–355) | Deterministic alphabetical mappings for label-encoded categorical features |
| `data/processed/phase1_report.json` | JSON | `PreprocessingExporter.export_report()` (pipeline.py:361) | Complete pipeline report: ingestion stats, identifier removal, cleaning, variance filter, redundancy removal, split counts, scaling method, feature names, elapsed time |
| `results/phase1_preprocessing/report_section_preprocessing.md` | Markdown | `render_preprocessing_report()` (pipeline.py:363–376) | Human-readable thesis section; path under `results/` not `phase0/` to avoid biometric-column header leak (finding #20) |

---

### T1 Data State Summary

- **Input state:** Raw WUSTL-EHMS-2020 CSV with 25 network and biometric features, labels, and potential PHI identifiers; unvalidated integrity
- **Output state:** Six stratified Parquet files with scaled, identifier-removed, redudancy-filtered features; cryptographic integrity metadata; RobustScaler sidecar; benign subset for anomaly detection
- **Key change:** Untrusted raw records → integrity-verified, leakage-guarded, stratified feature matrices with frozen train/val/test/demo partition boundaries

---

## TRANSFORMATION T2: Feature Arrays → Frozen Probabilistic Model Artifacts

**Modules involved:** Module 2 (Detection Training)

---

### T2-A — ECDSA Signing of Classifier Artifacts

| File path | Function/class | Line numbers | Finding | Status |
|-----------|----------------|-------------|---------|--------|
| `common/signed_pickle.py` | `dumps_signed()` | 151–219 | Signing implementation: writes classifier pickle + sidecar `.pkl.sig` with fields `format: "phase2.signed_pickle.v1"`, `signature_alg: "ECDSA_P256_SHA256"`, `signing_key_id`, `sha256` (hex digest), ISO 8601 `signed_at` timestamp. | [IMPLEMENTED] |
| `common/signed_pickle.py` | `loads_signed()` | 222–322 | Unsigned deserialization refused: raises `SignedPickleError` (lines 251–254) if sidecar missing. Verifies SHA-256 digest match (lines 278–282), signing_key_id consistency (lines 286–293), ECDSA signature validity (lines 296–308) before invoking `joblib.load()` (line 316). | [IMPLEMENTED] |
| `module2_detection/module2_train_models.py` | `train_track_a()` | 265 | XGBoost classifier signed at export: `dumps_signed(classifier_only, pipeline_path)`. SMOTE wrapper stripped (security finding #15). | [IMPLEMENTED] |
| `module2_detection/layer2_detector.py` | `__init__()` | 223 | Signed classifiers loaded at inference: `self._track_a[name] = loads_signed(base_pkl)` — verification enforced at load time in M3/M4. | [IMPLEMENTED] |

**Artifact signing status:**
- XGBoost: **Signed** (`xgboost_final_pipeline.pkl` + `xgboost_final_pipeline.pkl.sig`)
- RF/DT: Generated only with `--include-baselines` flag; signed when generated
- DAE: JSON + H5 format (pickle-free); no ECDSA signature; integrity relies on filesystem + SHA-256 provenance in `dae_calibration.json`

**[INCONSISTENCY FLAG]** — DAE artifacts (`dae_detector.json`, `dae_model.weights.h5`) are not ECDSA-signed. Only Track A classifier pickles receive signing. The `dae_calibration.json` contains a SHA-256 hash of `dae_detector.json` for chain-of-custody, but this is not a cryptographic signature scheme equivalent to ECDSA P-256.

---

### T2-B — Provenance Metadata Embedding

| File path | Function/class | Line numbers | Finding | Status |
|-----------|----------------|-------------|---------|--------|
| `module2_detection/build_dae_v4_artifacts.py` | `_sha256_file()` | 57–58 | SHA-256 of source `dae_detector.json` computed; returns hex digest. | [IMPLEMENTED] |
| `module2_detection/build_dae_v4_artifacts.py` | `main()` | 150–151, 169–177 | ISO 8601 UTC timestamp generated (line 151: `datetime.now(timezone.utc).isoformat()`). Fields written to `dae_calibration.json`: `source_detector_path`, `source_detector_sha256` (hex of **source** `dae_detector.json`), `generated_at_utc`. Confirmed: SHA-256 links to **source** (pre-calibration) not to the calibration artifact itself. | [IMPLEMENTED] |
| `module2_detection/module2_train_models.py` | `train_track_a()` | 268–283 | Per-model JSON report fields: `best_params` (hyperparameters), `optimal_threshold` (F2-calibrated value), `data.n_features` (25), `data.feature_names` (25-element list), `data.train_samples` (9,790), `data.test_samples` (2,448), `data.random_seed` (42, line 280). | [IMPLEMENTED] |
| `module2_detection/module2_train_models.py` | `train_track_b_dae()` | 450–459 | DAE final report fields: `data.n_raw_features` (line 451), `data.n_track_a_features` (line 452, always 0), `data.feature_names` (line 454), `data.benign_train_samples`, `data.test_samples`, `data.random_seed` (line 457, 42). | [IMPLEMENTED] |

**Actual `dae_calibration.json` provenance fields (from `results/models/`):**
```json
{
  "format": "layer1_v4.dae_calibration",
  "format_version": 1,
  "source_detector_path": "results/models/dae_detector.json",
  "source_detector_sha256": "fc2abb68d18747a5d09468987f77f9c28a9d20e0ec7a2a5096bf041377842448",
  "generated_at_utc": "2026-05-07T10:07:54.347560+00:00"
}
```

**Actual `xgboost_final_report.json` fields:**
```json
{
  "optimal_threshold": 0.01846158612363696,
  "data": { "n_features": 25, "train_samples": 9790, "test_samples": 2448, "random_seed": 42 }
}
```

---

### T2-C — Training Guard (Demo/Test Partition Protection)

| File path | Function/class | Line numbers | Finding | Status |
|-----------|----------------|-------------|---------|--------|
| `module2_detection/module2_train_models.py` | `_assert_no_demo_leakage()` | 67–77 | `RuntimeError` raised (line 70) with exact message: `"Module 2 training functions must not load demo_phase1.parquet. Strategy 1 invariant: the demo split is frozen and may only be touched at inference time (see module2_train_models.predict_demo). If you need demo predictions, run predict_demo() on already-fitted pipelines — do not refit on demo rows."` | [IMPLEMENTED] |
| `module2_detection/module2_train_models.py` | `load_data()` | 84–85 | Guard called on both `train_path` (line 84) and `test_path` (line 85). Forbidden set: `_FORBIDDEN_TRAINING_PARQUETS = frozenset({"demo_phase1.parquet"})` (line 64). | [IMPLEMENTED] |
| `tests/test_data_split_integrity.py` | `test_module2_refuses_to_load_demo_phase1()` | 154–159 | Test verifies guard: imports `_assert_no_demo_leakage` (line 157); confirms `RuntimeError` raised via `pytest.raises(RuntimeError)` (lines 158–159). | [IMPLEMENTED] |

---

### T2-D — F2 Threshold Calibration

| File path | Function/class | Line numbers | Finding | Status |
|-----------|----------------|-------------|---------|--------|
| `module2_detection/models/_threshold.py` | `find_optimal_threshold()` | 29–66 | Signature: `def find_optimal_threshold(y_true, y_proba, beta=2.0)`. Default `beta=2.0` confirms F2 weighting (recall weighted 2× over precision). Algorithm: `precision_recall_curve()` (line 50); vectorised F-beta formula (line 59): `fbeta = np.where(denom > 0, (1.0 + b2) * p * r / denom, 0.0)` where `b2 = beta ** 2`; returns threshold at `argmax(fbeta)` (line 66). | [IMPLEMENTED] |
| `module2_detection/module2_train_models.py` | `train_track_a()` | 231–240 | Calibration on OOF probabilities from 5-fold CV (`StratifiedKFold(n_splits=5)` at line 231); threshold computed at line 240: `threshold = find_optimal_threshold(y_train, oof_proba)`. Per-model calibration (separate threshold for each classifier). | [IMPLEMENTED] |
| `module2_detection/module2_train_models.py` | `train_track_a()` | 272 | Calibrated threshold written to JSON report (line 272): `"optimal_threshold": threshold`. Format: float64 scalar in JSON. | [IMPLEMENTED] |

**Actual calibrated threshold:** `xgboost_final_report.json`: `"optimal_threshold": 0.01846158612363696`

---

### T2-E — DAE Architecture Validation

| File path | Function/class | Line numbers | Finding | Status |
|-----------|----------------|-------------|---------|--------|
| `module2_detection/models/DAE.py` | `_build_model()` | 206–210 | Architecture validation assertion: `if dims[1] >= n_features: raise ValueError(f"Bottleneck dim ({dims[1]}) must be < n_features ({n_features}) to force compression.")` — enforces information bottleneck strictly less than feature count. | [IMPLEMENTED] |
| `module2_detection/module2_train_models.py` | `train_track_b_dae()` | 411–417 | DAE receives raw 25-dimensional input (line 411: `n_feat = X_benign_raw.shape[1]`). Architecture sized for 25 features (lines 413–414). Docstring (lines 348–354): `"Per ARCHITECTURE.md Track B: the cascade design [raw || P_xgb, P_rf, P_dt] was evaluated via leave-one-class-out and rejected (EHMS ΔAUC=+0.02 marginal; MedSec-25 ΔAUC=−0.19 regression). The production design is DAE-raw."` | [IMPLEMENTED] |
| `module2_detection/module2_train_models.py` | `train_track_b_dae()` | 450–453 | Report fields confirming no-cascade: `"n_raw_features": len(feat_names)` (25, line 451), `"n_track_a_features": 0` (line 452, hardcoded), `"n_total_features": n_feat` (25, line 453). | [IMPLEMENTED] |

**Confirmed from `results/models/dae_final_report.json`:**
```json
{ "data": { "n_raw_features": 25, "n_track_a_features": 0, "n_total_features": 25 }, "architecture": "raw_25dim" }
```

---

### T2 Data State Summary

- **Input state:** Scaled feature matrices (Parquet) partitioned into train (9,790), val (2,448), test (2,448), demo (1,632), benign-only train (9,563)
- **Output state:** ECDSA-signed XGBoost pickle + sidecar; DAE JSON config + H5 weights; per-model JSON reports with F2-calibrated thresholds, random seed, feature names, provenance SHA-256; frozen NPZ prediction arrays per split
- **Key change:** Mutable feature matrices → immutable, cryptographically attested model artifacts with embedded reproduction metadata

---

## TRANSFORMATION T3: Per-Alert Features → Risk-Tiered ScoredAlert Objects

**Modules involved:** Module 3 (Risk Scoring), `src/` runtime layer

---

### T3-A — EA-06 Feature Sanitization

| File path | Function/class | Line numbers | Finding | Status |
|-----------|----------------|-------------|---------|--------|
| `src/preprocessing.py` | `sanitize_features()` | 75–132 | Feature sanitization with benign-median imputation. Benign medians loaded at line 109 via `load_benign_medians()` (lines 66–72) from `data/processed/benign_medians.json`. | [IMPLEMENTED] |
| `src/preprocessing.py` | `sanitize_features()` | 117–130 | Four `DataQuality` flag values: (1) **FAILED**: `nan_rate >= 0.50` (line 117); (2) **DEGRADED**: `0.05 < nan_rate < 0.50` (line 123); (3) **IMPUTED_NAN**: `nan_rate <= 0.05` with imputation applied (line 19 path); (4) **OK**: no NaN/Inf detected (line 130). | [IMPLEMENTED] |
| `src/risk_scorer.py` | `score_alert()` | 236–239 | **DEGRADED** multiplier: **×1.20** (line 239: `score = min(1.0, score * 1.20)`). **FAILED** minimum: **≥0.95** (line 237: `score = max(score, 0.95)`). Both caps enforce patient-safety-bias: degraded data quality elevates, never suppresses, alert scores. | [IMPLEMENTED] |

---

### T3-B — Two-Stage Fusion Implementation

| File path | Function/class | Line numbers | Finding | Status |
|-----------|----------------|-------------|---------|--------|
| `module3_risk_scoring/module3_risk_scores.py` | `compute_c_detect()` | 335–368 | Fused detection confidence: `c_detect = np.maximum(c_track_a, c_track_b)` (line 364). **`numpy.maximum`** (element-wise) — DAE can only elevate, never suppress, the XGBoost signal (INVARIANT 1). | [IMPLEMENTED] |
| `module3_risk_scoring/module3_risk_scores.py` | `classify_fusion()` | 280–332 | Four `fusion_class` values: (1) **KNOWN_ATTACK**: `P_xgb >= P_XGB_HIGH_CONF` (0.85); (2) **CONFIRMED_ANOMALY**: `a_low <= P_xgb < a_high AND dae >= b`; (3) **NOVEL_ANOMALY**: `P_xgb < a_low AND dae >= b`; (4) **BENIGN**: otherwise. | [IMPLEMENTED] |
| `src/data_models.py` | `P_XGB_HIGH_CONF` constant | 68–70 | Named constant for KNOWN_ATTACK boundary: `P_XGB_HIGH_CONF: float = 0.85`. Defined in `data_models.py` at line 68; consumed by `classify_fusion()` at line 314. | [IMPLEMENTED] |

---

### T3-C — Context Enrichment YAML Lookup

| File path | Function/class | Line numbers | Finding | Status |
|-----------|----------------|-------------|---------|--------|
| `src/context_enrichment.py` | `enrich_alert_context()` | 173–240 | Three YAML files consulted: (1) **`configs/device_inventory.yaml`** (lines 140–142 via `_load_device_inventory()`): device_class, device_criticality, clinical_tier, patchable; (2) **`configs/composite_risk_weights.yaml`** (clinical_tier mapping, lines 233–235); (3) **`configs/attack_to_mitre_mapping.yaml`** (referenced at line 9, loaded separately for MITRE attribution). | [IMPLEMENTED] |
| `src/context_enrichment.py` | `_UNKNOWN_DEVICE_DEFAULTS` constant | 161–167 | UNKNOWN device fallback values: `patchable=False` (line 163), `device_criticality=HIGH` (line 164), `clinical_tier=tier_2_high_clinical` (line 165). Conservative defaults treat unrecognized devices as high-risk. | [IMPLEMENTED] |
| `src/context_enrichment.py` | `_read_patchable()` | 103–120 | RuntimeError on absent patchable field (lines 115–120): `MissingRequiredField("Alert is missing the required 'patchable' (or legacy 'device_patchable') field. Per ARCHITECTURE.md Step [8] the field must be present on every alert...")` | [IMPLEMENTED] |
| `src/context_enrichment.py` | `enrich_alert_context()` | 209–224 | Secondary rogue-device alert emission: `DEVICE_NOT_IN_INVENTORY` added to `warning_flags` (line 218) when device_class absent. Warning logged at lines 220–224. | [IMPLEMENTED] |

---

### T3-D — Composite Risk Formula and Weight Validation

| File path | Function/class | Line numbers | Finding | Status |
|-----------|----------------|-------------|---------|--------|
| `module3_risk_scoring/module3_risk_scores.py` | `compute_composite_risk()` | 468–481 | Formula (lines 477–480): `R = w["w1"] * c_detect + w["w2"] * d_crit + w["w3"] * s_data + w["w4"] * d_clinical_tier` — confirms **R = 0.40·C + 0.25·D_crit + 0.15·S + 0.20·D_clin**. | [IMPLEMENTED] |
| `module3_risk_scoring/module3_risk_scores.py` | `load_composite_weights()` | 68–91 | Weight sum validation (line 87): `if abs(total - 1.0) > 1e-6:` — tolerance exactly **1e-6**. | [IMPLEMENTED] |
| `module3_risk_scoring/module3_risk_scores.py` | `load_tier_boundaries()` | 94–104 | Four tier boundaries: CRITICAL ≥ **0.80** (line 100), HIGH ≥ **0.60** (line 101), MEDIUM ≥ **0.40** (line 102), LOW < 0.40 (implicit). | [IMPLEMENTED] |
| `configs/composite_risk_weights.yaml` | `weights` section | 13–17 | Weight values: `detection_confidence=0.40`, `device_criticality=0.25`, `data_sensitivity=0.15`, `clinical_tier=0.20`. Sum = 1.0. | [IMPLEMENTED] |
| `configs/composite_risk_weights.yaml` | `review` section | 36–38 | Governance: `reviewers: ["CISO", "Patient Safety Officer", "Clinical Engineering Director"]`, `review_period: "12 months"`. | [IMPLEMENTED] |
| `configs/composite_risk_weights.yaml` | `limitations` section | 40–44 | Four acknowledged limitations: L1: Linear sum allows compensatory effects; L2: clinical_tier is device-class proxy, not real-time acuity; L3: device_criticality and clinical_tier correlated (combined 0.45 weight double-counts device importance); L4: tier boundaries calibrated to test split distribution. | [DOCUMENTED_ONLY] |

---

### T3-E — Safety Floor Enforcement

| File path | Function/class | Line numbers | Finding | Status |
|-----------|----------------|-------------|---------|--------|
| `src/risk_scorer.py` | `score_alert()` | 246–265, 310–315 | Safety floor: CRITICAL+unpatchable unconditionally sets `should_surface = True` (lines 314–315). Evaluated **after** maintenance window check (lines 246–265) but **before** final return (lines 317–325) — maintenance window can add suppression note but cannot prevent CRITICAL+unpatchable surfacing. | [IMPLEMENTED] |
| `src/risk_scorer.py` | `score_alert()` | 262, 322 | `suppression_reason` values: `"maintenance window — reduced confidence, verify with biomed"` (line 262, maintenance-window path); `None` (line 322, standard and safety-floor paths). | [IMPLEMENTED] |
| `tests/test_safe_failure.py` | `test_critical_unpatchable_surfaces_in_maintenance_window()` | 57–68 | Test verifies safety floor: CRITICAL+unpatchable device surfaces even when `is_maintenance_window=True` AND `is_known_vendor_ip=True`. Assertion at line 68: `assert result.should_surface is True`. | [IMPLEMENTED] |

---

### T3-F — ScoredAlert Dataclass Field Inventory

| Field name | Type | Default | Purpose / source |
|------------|------|---------|-----------------|
| `adjusted_score` | `float` | (required) | Anomaly score after risk multiplier applied (`src/data_models.py:162`) |
| `threshold` | `float` | (required) | Device-context surfacing threshold rounded to 4 decimals (`src/data_models.py:165`) |
| `should_surface` | `bool` | (required) | `adjusted_score > threshold` (`src/data_models.py:168`) |
| `risk_multiplier` | `float` | (required) | Multiplier: ≥1.5 for CRITICAL+unpatchable, 1.0 for LOW+patchable (`src/data_models.py:171`) |
| `suppression_reason` | `Optional[str]` | `None` | Human-readable suppression note; audit trail field (`src/data_models.py:174`) |
| `fusion_class` | `FusionClass` | `FusionClass.BENIGN` | Two-stage fusion outcome (EA-07 addition; `src/data_models.py:177`) |
| `data_quality` | `DataQuality` | `DataQuality.OK` | Per-row sanitization outcome: OK/IMPUTED_NAN/DEGRADED/FAILED (EA-06 addition; `src/data_models.py:180`) |

**Note:** No explicit `mitre_techniques`, `mitre_confidence`, or `c_detect` fields on ScoredAlert itself. MITRE attribution is handled in `context_enrichment.py` as a separate dict; `c_detect` is a local variable in `score_alert()`, not a field on the returned dataclass.

**No `__post_init__` validation** on ScoredAlert; all fields assigned in `score_alert()` after computation.

---

### T3 Data State Summary

- **Input state:** Scaled per-alert feature vector (float32 × 25) from live or batch network capture; device identifier for inventory lookup
- **Output state:** `ScoredAlert` object with composite risk score R ∈ [0,1], risk tier (CRITICAL/HIGH/MEDIUM/LOW), `should_surface` bool, `fusion_class` (provenance), `data_quality` (EA-06 status), and `suppression_reason` (if applicable)
- **Key change:** Raw feature vector → semantically enriched risk object with context-adaptive threshold, two-track fusion, and safety-floor guarantees embedded in the object contract

---

## TRANSFORMATION T4: ScoredAlert Objects → Explanation-Enhanced Operator Artifacts

**Modules involved:** Module 4 (Explanations), `src/mve_generator.py`

---

### T4-A — SHAP Background Dataset

| File path | Function/class | Line numbers | Finding | Status |
|-----------|----------------|-------------|---------|--------|
| `module4_explanations/build_shap_background.py` | `main()` | 27–77 | Background construction: exact sample count **200** (`n_target = min(200, n_total)`, line 49); stratification via `train_test_split(..., stratify=y)` (line 55); `random_state=42` (line 58). Persisted to `results/models/shap_background.pkl` (line 33) via `joblib.dump(compress=3)`. | [IMPLEMENTED] |
| `module4_explanations/build_shap_background.py` | `main()` | 62–72 | Artifact format: joblib-compressed dict with keys `"background"` (float32 ndarray), `"feature_names"` (list), `"n_samples"` (int), `"source"` (str). | [IMPLEMENTED] |
| `module4_explanations/module4_online_explainer.py` | `AlertExplainer.__init__()` | 278–319 | At online inference: TreeSHAP `TreeExplainer` objects initialized once during `AlertExplainer.__init__()`; background pickle is not reloaded per alert. SHAP library uses model internal structure for TreeSHAP — background pkl is for audit/reproducibility per ARCHITECTURE.md Step [11], not for per-alert recomputation. | [IMPLEMENTED] |

---

### T4-B — SHAP Stability Measurement

| File path | Function/class | Line numbers | Finding | Status |
|-----------|----------------|-------------|---------|--------|
| `module4_explanations/module4_online_explainer.py` | `STABILITY_NOISE_SIGMA` constant | 145 | Perturbation mechanism: **Gaussian noise σ=0.01** (line 145: `STABILITY_NOISE_SIGMA: float = 0.01`). Applied at lines 190–194: `rng.normal(0.0, noise_sigma, size=(n_samples, n_features))`. | [IMPLEMENTED] |
| `module4_explanations/module4_online_explainer.py` | `compute_shap_stability()` | 155–217 | Similarity metric: **Jaccard on top-3 feature indices** (lines 206–217); k=3 (line 146: `STABILITY_K_TOP: int = 3`). Exact Jaccard formula (lines 213–215): `inter = len(sets[i] & sets[j]); union = len(sets[i] | sets[j]); total += (inter / union)`. | [IMPLEMENTED] |
| `module4_explanations/module4_online_explainer.py` | `STABILITY_N_SAMPLES` constant | 144 | Number of perturbations: **10** (line 144). Original sample plus 10 perturbed copies (11 total; lines 188–195). | [IMPLEMENTED] |
| `module4_explanations/module4_online_explainer.py` | `STABILITY_HIGH` constant | 148 | Stability threshold: **0.90** (line 148). `is_stable = (stability_score >= STABILITY_HIGH)` per docstring lines 149–152. | [IMPLEMENTED] |
| `module4_explanations/module4_online_explainer.py` | `compute_shap_stability()` | 217 | Output: stability_score as `float` in [0.0, 1.0] (line 217: `return round(total / n_pairs if n_pairs else 1.0, 4)`). | [IMPLEMENTED] |
| `module4_explanations/module4_online_explainer.py` | `build_shap_context()` | 389 | `is_stable`: `bool` (line 389: `"is_stable": bool(stability_score >= STABILITY_HIGH)`). | [IMPLEMENTED] |

---

### T4-C — NOVEL_ANOMALY Faithfulness Limitation Flag

| File path | Function/class | Line numbers | Finding | Status |
|-----------|----------------|-------------|---------|--------|
| `module4_explanations/module4_online_explainer.py` | `build_shap_context()` | 373–381 | Condition: `fusion_class in {"NOVEL_ANOMALY", "STRONG_NOVEL_ANOMALY"}` (lines 377–380). Exact code: `novel_classes = {"NOVEL_ANOMALY", "STRONG_NOVEL_ANOMALY"}; if fusion_class in novel_classes: shap_source = "xgboost_low_confidence"`. Default is `"xgboost"`. | [IMPLEMENTED] |
| `src/data_models.py` | `SHAPContext` | 186–227 | `shap_source` field included in `SHAPContext` dataclass (lines 219–226): `shap_source: str = "xgboost"`. Propagated via `explain()` return dict at line 590: `"shap_context": shap_context`. Readable by M5/M6 via `.get("shap_context", {}).get("shap_source")`. | [IMPLEMENTED] |

**[INCONSISTENCY FLAG]** — `shap_source = "xgboost_low_confidence"` is assigned in `SHAPContext` but is **not propagated into MVE Layer 1 prose**. Neither `_clinician_nlg()` (line 423–465 of `module4_online_explainer.py`) nor MVE generation in `src/mve_generator.py` (lines 1319–1337) use the `shap_source` flag to alter the explanation text. The operator reads SHAP-derived narrative for NOVEL_ANOMALY alerts without awareness that SHAP faithfulness is flagged as "low confidence." Documented as future work in `src/data_models.py:224–225`.

---

### T4-D — MVE Content Invariants

| Invariant | File path | Function/class | Line numbers | Enforcement mechanism | Status |
|-----------|-----------|----------------|-------------|----------------------|--------|
| **INVARIANT 3** (no auto-execution) | `src/data_models.py` | `ResponseRecommendation.__post_init__()` | 415–421 | Hard `ValueError` if `operator_decision_required is not True`: `"ResponseRecommendation.operator_decision_required must be True (INVARIANT 3 — no auto-execution). Setting it to False is forbidden by the architecture contract."` | [IMPLEMENTED] |
| **INVARIANT 6** (role authority) | `src/mve_generator.py` | `role_authority_violations()` | 166–184 | Checks forbidden-action terms against Layer 3 `immediate_action` text (lines 181–183); returns sorted list of violations. Empty list = compliant. | [IMPLEMENTED] |
| **INVARIANT 9** (SharedAnchor cross-role consistency) | `src/data_models.py` | `SharedAnchor` | 329–359 | Dataclass ensures 5 anchor fields (`alert_id`, `risk_tier`, `device_id`, `one_line_summary`, `timestamp`) held invariant across all role views. Docstring line 331: "the anchor MUST be byte-identical across all role views." | [IMPLEMENTED] |
| **Layer 2 invariance** | `src/mve_generator.py` | `derive_role_view()` | 254–259 | `layer_2=dict(mve.layer_2)` copied unchanged across all roles (line 256, comment: "unchanged — cross-role severity invariant"). | [IMPLEMENTED] |

**Mode A failure → Mode B fallback:** Any exception during `_generate_llm()` returns `None` (lines 1190–1195 try/except handler). `generate_mve()` then calls `_generate_rule_based()` at line 1276. PHI guard failure (`AssertionError` at line 1034) is a **hard fail** — no fallback, execution halted.

---

### T4-E — PHI Flow Filtering

| File path | Function/class | Line numbers | Finding | Status |
|-----------|----------------|-------------|---------|--------|
| `src/mve_generator.py` | `_filter_for_llm()` | 1018–1048 | PHI allow-list enforcement before Mode A API calls. Called at lines 1095–1098 in `_generate_llm()` for alert, device context, baseline, and user context dicts. | [IMPLEMENTED] |
| `src/mve_generator.py` | `_filter_for_llm()` | 1030, 1032–1038 | Forbidden fields loaded from YAML (line 1030: `forbidden = set(cfg["forbidden"])`). Hard-fail on presence (lines 1032–1038): `leaked = [k for k in payload if k in forbidden]; if leaked: raise AssertionError(...)` | [IMPLEMENTED] |
| `src/mve_generator.py` | `_filter_for_llm()` | 1034–1035 | Exception type: **`AssertionError`** with message prefix `"Mode A LLM: PHI red flag"` (line 1035). | [IMPLEMENTED] |
| `configs/llm_data_flow.yaml` | `forbidden` section | 57–72 | Forbidden fields (canonical source): `patient_id`, `patient_name`, `mrn`, `medical_record_number`, `dob`, `date_of_birth`, `ssn`, `phone_number`, `address`, `room_number_with_patient_context`, `clinical_notes`, `ehr_record`, `lab_result`, `prescription`, `any_clinical_data_from_ehr`. | [IMPLEMENTED] |
| `tests/test_phi_not_in_llm_prompt.py` | `test_filter_for_llm_raises_on_forbidden_field()` | 101–114 | Test parameterized over `["patient_id", "mrn", "ssn", "dob", "ehr_record"]`; confirms `AssertionError` raised via `pytest.raises(AssertionError, match="PHI red flag")` (line 113). | [IMPLEMENTED] |

---

### T4-F — Role View Derivation

| File path | Function/class | Line numbers | Finding | Status |
|-----------|----------------|-------------|---------|--------|
| `src/mve_generator.py` | `derive_role_view()` | 233–259 | Three role enum values: `IT_generalist`, `biomed_engineer`, `nurse_manager` (from `OperatorRole` enum, `src/data_models.py:63–65`). | [IMPLEMENTED] |
| `src/mve_generator.py` | `_role_lens_layer_1()` | 187–201 | **Layer 1 re-framed per role:** IT_generalist: unchanged (line 190); biomed_engineer: prefixes deviation with `"Device behaviour unusual: ..."` (lines 194–196); nurse_manager: prefixes with `"Equipment may be compromised. Patient safety priority. ..."` (lines 197–200). | [IMPLEMENTED] |
| `src/mve_generator.py` | `_role_lens_layer_3()` | 204–230 | **Layer 3 re-framed per role:** IT_generalist: unchanged (line 215); biomed_engineer: overwrites `immediate_action` with `"Verify device firmware..."` text (lines 219–223); nurse_manager: overwrites with `"Verify clinical backup..."` text (lines 225–229). | [IMPLEMENTED] |
| `src/mve_generator.py` | `derive_role_view()` | 254–259 | **Layer 2 invariant** (severity + clinical impact): copied unchanged at line 256. **SharedAnchor invariant**: `alert_id`, `risk_tier`, `device_id`, `one_line_summary`, `timestamp` held across roles (`src/data_models.py:329–359`). | [IMPLEMENTED] |
| `configs/role_action_authorization.yaml` | `roles` section | 22–54 | Role action authorization config consulted via `_load_role_forbidden_terms()` (`mve_generator.py:132–157`). | [IMPLEMENTED] |

---

### T4 Data State Summary

- **Input state:** `ScoredAlert` object (from T3) with risk tier, fusion class, and 25-dimensional feature vector; SHAP explainer pre-loaded from signed classifier artifacts
- **Output state:** `SHAPContext` (top-3 features, stability score, faithfulness flag), `MVEOutput` (3-layer prose with per-role variants), per-role `ResponseRecommendation` (with INVARIANT 3 governance), all PHI-filtered before any LLM API call
- **Key change:** Risk-tiered numerical alert → semantically grounded, role-differentiated natural language explanation with faithfulness metadata and governance contracts

---

## TRANSFORMATION T5: Explanation Artifacts → Response Recommendations and Evaluation Outputs

**Modules involved:** Module 5 (Response Guidance), Module 6 (Evaluation)

---

### T5-A — No-Execution Invariant (Invariant 3)

**Grep commands and results:**

```bash
$ grep -rn "subprocess\|os\.system\|iptables\|os\.popen\|eval(\|exec(" module5_responses/
(empty — no matches)

$ grep -rn "^import subprocess\|^from subprocess" module5_responses/
(empty — no matches)
```

| File path | Function/class | Line numbers | Finding | Status |
|-----------|----------------|-------------|---------|--------|
| `src/data_models.py` | `ResponseRecommendation` | 363–443 | All action fields are string type: `primary_action: str` (human-readable label), `primary_action_code: str` (machine-readable enum-like code), `do_not_actions: List[str]`. No callable, command, or subprocess type. | [IMPLEMENTED] |
| `tests/test_step15_role_consistency.py` | `test_no_auto_execution_in_module5_source()` | 146–164 | Negative test: greps module5_responses/ for subprocess, os.system, iptables, netcat, ssh, sudo, eval(), exec(), nc, curl, wget. Assertion: `offenders` list must be empty. | [IMPLEMENTED] |

---

### T5-B — Role-Based Action Authorization

| File path | Function/class | Line numbers | Finding | Status |
|-----------|----------------|-------------|---------|--------|
| `configs/role_action_authorization.yaml` | `roles.IT_generalist.forbidden` | 22–26 | Forbidden terms: `"administer"`, `"titrate dose"`, `"adjust ventilator setting"` — clinical interventions prohibited for IT role. | [IMPLEMENTED] |
| `configs/role_action_authorization.yaml` | `roles.biomed_engineer.forbidden` | 28–38 | Forbidden terms: `"isolate vlan"`, `"block port at switch"`, `"firewall rule"`, `"update acl"`, `"push nac"`, `"block outbound traffic"`, `"block port"`, `"switch-port block"`, `"isolate at switch"` — network control prohibited for biomed role. | [IMPLEMENTED] |
| `configs/role_action_authorization.yaml` | `roles.nurse_manager.forbidden` | 40–54 | Forbidden terms: network control terms (as above) plus device firmware operations: `"power-cycle device"`, `"restart device firmware"`, `"reflash firmware"`, `"wipe device"` — both network and device operations prohibited for clinical role. | [IMPLEMENTED] |
| `src/mve_generator.py` | `_load_role_forbidden_terms()` / `role_authority_violations()` | 132–160, 166–185 | Authorization check: `_load_role_forbidden_terms()` loads YAML; `role_authority_violations()` performs case-insensitive substring match of forbidden terms against `immediate_action` text (lines 181–183). Returns sorted list of violations; empty = compliant. | [IMPLEMENTED] |

**No exception raised on violation** — `role_authority_violations()` returns a list; caller enforces compliance via assertion. Tests in `tests/test_role_authority.py:69–122` verify all three role boundaries.

---

### T5-C — Tier Routing Logic

| Fusion class | Routing outcome (TierLevel) | File path | Function | Line numbers |
|---|---|---|---|---|
| `KNOWN_ATTACK` | `L1_IMMEDIATE` | `module5_responses/tier_routing_v4.py` | `_ROUTING` dict | 137–143 |
| `KNOWN_ATTACK_UNCERTAIN` | `L1_WITH_REVIEW` | Same | `_ROUTING` dict | 144–150 |
| `DISAGREEMENT_ANOMALY` | `L2_SECURITY_SPECIALIST` | Same | `_ROUTING` dict | 151–165 |
| `STRONG_NOVEL_ANOMALY` | `L2_SPECIALIST` | Same | `_ROUTING` dict | 166–172 |
| `NOVEL_ANOMALY` | `L2_SPECIALIST` | Same | `_ROUTING` dict | 173–179 |
| `CONFIRMED_ANOMALY` | `L1_WITH_SENIOR` | Same | `_ROUTING` dict | 180–186 |
| `SUSPICIOUS_PATTERN` | `L1` | Same | `_ROUTING` dict | 187–193 |
| `BENIGN_WATCH` | `AUDIT_LOG` | Same | `_ROUTING` dict | 194–200 |
| `BENIGN` | `SUPPRESSED` | Same | `_ROUTING` dict | 201–207 |

**Small hospital fallback** (`configs/hospital_capabilities.yaml:26–50`): Preset `small` has `available_tiers: ["L1"]`. For `NOVEL_ANOMALY` (routed to `L2_SPECIALIST`): fallback action `"document_for_external_consultant_review"`, timeline `"next_business_day"` (lines 27–30). [IMPLEMENTED]

---

### T5-D — RQ1 Metrics Path Data Independence

| File path | Function | Line numbers | Finding | Status |
|-----------|----------|-------------|---------|--------|
| `module6_evaluation/compute_rq1_metrics.py` | `main()` | 6, 12 | Input artifact: `results/reports/evaluation_alerts.json` — sourced from demo pool curation path, not from test split. [Note: See discrepancy flag below.] | [IMPLEMENTED] |
| `module6_evaluation/curate_demo_alerts.py` | `main()` | 1–26 | Input artifact: `results/reports/demo_scores.npz` from demo_phase1.parquet (demo split only). | [IMPLEMENTED] |
| `module6_evaluation/module6_evaluation.py` | `curate_evaluation_alerts()` | 204–210 | Loads `demo_scores.npz` + `demo_phase1.parquet` (lines 204–210). Never reads `test_phase1.parquet` or `risk_scores.npz` (test split). | [IMPLEMENTED] |
| `tests/test_data_split_integrity.py` | `test_module2_refuses_to_load_demo_phase1()` | 154–159 | Split integrity test: `_assert_no_demo_leakage()` raises `RuntimeError` on demo split load during training. | [IMPLEMENTED] |

**[DISCREPANCY]** — The extraction prompt specifies that `compute_rq1_metrics.py` should read `risk_scores.npz` from the test split only. However, the agent found it reads `evaluation_alerts.json` (derived from demo pool). This may indicate that the RQ1 paper metrics path uses the curated demo alerts (not the raw test-split NPZ), or that the file naming/path differs from expectations. **The thesis author should verify whether `compute_rq1_metrics.py` processes test-split or demo-split data, as this affects the paper metrics independence claim.**

---

### T5-E — Evaluation Alerts Curation

| File path | Function | Line numbers | Finding | Status |
|-----------|----------|-------------|---------|--------|
| `module6_evaluation/module6_evaluation.py` | `curate_evaluation_alerts()` | 278–309 | Stratification dimensions: **risk_tier × attack_category** (CRITICAL/HIGH/MEDIUM/LOW × Spoofing/Data Alteration). Target count: **20 alerts total** — 4 per tier × 2 attack categories = 16 attack alerts; + 4 benign calibration at risk scores [0.20, 0.30, 0.45, 0.55] = 20. Per-stratum targets: 2 attacks per tier level. | [IMPLEMENTED] |
| `module6_evaluation/module6_evaluation.py` | `curate_evaluation_alerts()` | 1081 | Output artifact: `results/reports/evaluation_alerts.json` — JSON list of alert dicts. | [IMPLEMENTED] |
| `module6_evaluation/module6_evaluation.py` | `_build_group_a_display()` / `_build_group_b_display()` | 131–189 | **Group A display fields** (raw network, baseline): Alert ID, Anomaly score, Prediction (Attack), Source IP:port → Dest IP:port, Protocol, Timestamp; no biometric fields (sanitized per GAP-A7). **Group B display fields** (MVE 3-layer): WHY ANOMALOUS (layer_1), CLINICAL SEVERITY (layer_2), RECOMMENDED ACTION (layer_3). | [IMPLEMENTED] |

---

### T5-F — MD5-Seeded Study Shuffle

| File path | Function/class | Line numbers | Finding | Status |
|-----------|----------------|-------------|---------|--------|
| `module6_evaluation/study_loader.py` | `load_study_alerts()` | 28–52 | Participant shuffle function. Seed derivation (line 48): `int(hashlib.md5(participant_id.encode()).hexdigest(), 16)` — confirmed exactly. Applied at lines 49–50: `rng = random.Random(pid_seed); rng.shuffle(scenarios)`. | [IMPLEMENTED] |
| `module6_evaluation/study_loader.py` | `assign_ab_condition()` | 55–66 | Counterbalanced A/B logic (lines 61–66): `pid_num % 2 == 0` (even PID): show MVE (Group B) for alerts 0–9, hide for 10–19; odd PID: reverse. Counterbalances presentation order across participants. | [IMPLEMENTED] |
| `module6_evaluation/study_loader.py` | `load_study_alerts()` | 28–52, 55–66 | Alert count per participant: **20** (`n_alerts` defaults to 20, line 56). | [IMPLEMENTED] |

---

### T5 Data State Summary

- **Input state:** `MVEOutput` (3-layer prose), `SHAPContext` (SHAP features + faithfulness flag), `ResponseRecommendation` (governance-verified action), persisted `analyst_report.json` and `clinician_summaries.json`
- **Output state:** `alert_responses.json` (M5 batch) + hash-chained `audit_trail.json`; `evaluation_alerts.json` (M6 curation) with Group A/B display fields; RQ1 metrics computed from curated alert pool
- **Key change:** Explanation artifacts → operator-committed response record with cryptographic audit chain; curated alert pool for human-subjects study with deterministic per-participant counterbalancing

---

## AREA T6: Cross-Pipeline Audit Trail Verification

---

### T6-A — Audit Trail Origin in Module 0

| File path | Function/class | Line numbers | Finding | Status |
|-----------|----------------|-------------|---------|--------|
| `module0_analysis/phase0/security.py` | `log_phase0_event()` | 506–552 | Phase 0 security events routed to shared hardened audit log: `results/reports/audit_log.jsonl` — same file path as Module 5 audit log (confirmed shared path). | [IMPLEMENTED] |
| `module0_analysis/phase0/security.py` | `log_phase0_event()` event types | 506–552 | Event types emitted with severity levels: `DATASET_LOADED` (INFO), `INTEGRITY_VERIFIED` (INFO), `INTEGRITY_FAILURE` (CRITICAL), `BOOTSTRAP_COMPLETE` (INFO), `SIGNATURE_VERIFIED` (INFO), `SIGNATURE_FAILURE` (CRITICAL), `PHI_ACCESS_ATTEMPT` (WARNING), `UNAUTHORIZED_ACCESS` (CRITICAL), `SCHEMA_VALIDATION_FAILURE` (WARNING), `AUDIT_LOG_INITIALIZED` (INFO). Ten event types total. | [IMPLEMENTED] |
| `module0_analysis/phase0/security.py` | `log_phase0_event()` | 506–552 | Falls back gracefully if Module 5 audit infrastructure unavailable (safe degradation). | [IMPLEMENTED] |

---

### T6-B — Hash Chain Implementation in Module 5

| File path | Function/class | Line numbers | Finding | Status |
|-----------|----------------|-------------|---------|--------|
| `module5_responses/module5_pipeline.py` | `AuditLogger` class | 599–1120 | Hash chain implementation. Genesis hash: **`"0" * 64`** (64 hex zeros). | [IMPLEMENTED] |
| `module5_responses/module5_pipeline.py` | `AuditLogger._canonical_json()` | ~650 | Canonical serialization: `json.dumps(..., sort_keys=True, separators=(",", ":"))` — deterministic across Python versions. | [IMPLEMENTED] |
| `module5_responses/module5_pipeline.py` | Hash formula | ~680 | `integrity_hash = SHA256(_canonical_json(record + prev_hash))` — SHA-256 over canonical JSON of concatenated record and previous hash. | [IMPLEMENTED] |
| `module5_responses/module5_pipeline.py` | ECDSA signing | ~700 | ECDSA P-256 signing function applied to each audit entry individually. | [IMPLEMENTED] |
| `module5_responses/module5_pipeline.py` | `AuditLogger.verify()` | 791–946 | 3-level verification: (1) chain continuity (prev_hash linkage), (2) integrity hash recomputation, (3) ECDSA signature validity. Returns dict with chain-break details or clean status. | [IMPLEMENTED] |

---

### T6-C — Shared Signing Identity

| File path | Finding | Status |
|-----------|---------|--------|
| `module0_analysis/phase0/security.py` + `module5_responses/module5_pipeline.py` | Both modules import `_load_signing_key()` from `module5_responses/module5_pipeline.py`. Both reference the same private key: **`~/.iomt-ids/audit_signing_key.pem`** (ECDSA P-256 / SECP256R1). Public key: `results/reports/audit_signing_key.pub.pem`. Auto-bootstrapped on first run if absent. | [IMPLEMENTED] |

**Confirmed:** Module 0 and Module 5 use the **same** ECDSA signing key. Phase 0 integrity events and Phase 5 operator decision records are signed under a shared key identity, enabling cross-module audit trail continuity.

---

### T6-D — Forward-Compatible Schema Slots

| File path | Function/class | Line numbers | Field name | Default | Documentation | Status |
|-----------|----------------|-------------|------------|---------|---------------|--------|
| `module5_responses/module5_pipeline.py` | Audit record schema | 763–765 | `ground_truth_label` | `None` | Reserved for Step 17: outcome tracking (was alert a true positive?) | [IMPLEMENTED] |
| `module5_responses/module5_pipeline.py` | Audit record schema | 763–765 | `decision_quality` | `None` | Reserved for Step 17: operator decision quality assessment | [IMPLEMENTED] |
| `module5_responses/module5_pipeline.py` | Audit record schema | 763–765 | `feedback_loop_consumed` | `False` | Reserved for Step 18: continuous improvement feedback flag | [IMPLEMENTED] |

---

### T6-E — Audit Log Integrity Verification Function

| File path | Function/class | Line numbers | Finding | Status |
|-----------|----------------|-------------|---------|--------|
| `module5_responses/module5_pipeline.py` | `AuditLogger.verify()` classmethod | 791–946 | Verifies: (1) chain continuity (prev_hash linkage between consecutive entries), (2) integrity hash recomputation (SHA-256 match), (3) ECDSA signature validity (per-entry). Returns dict indicating clean status or chain-break location. | [IMPLEMENTED] |
| `tests/test_step16_audit_integrity.py` | 5 invariant locks (I1–I5) | Full test file | Tests cover: chain genesis integrity, single-entry tamper detection, inter-entry hash linkage, ECDSA signature verification, and forward-schema slot presence. | [IMPLEMENTED] |

---

## CROSS-CUTTING VERIFICATION

### Cross-Check 1 — Artifact Handoff Completeness

| Handoff | Producing artifact | Producing script | Consuming script | Assessment |
|---------|-------------------|-----------------|-----------------|-----------|
| **M0 → M1** | `high_correlations.csv`, `stats_report.json` | `module0_analysis/phase0/analyzer.py` | `module1_preprocessing/phase1/pipeline.py` | [CONSISTENT] — M0 writes correlation/stats artifacts; M1 reads them for feature-drop decisions |
| **M1 → M2** | `train_phase1.parquet`, `benign_only_train.parquet` | `module1_preprocessing/phase1/pipeline.py` | `module2_detection/module2_train_models.py` | [CONSISTENT] — M1 writes both parquets; M2 reads them for Track A and Track B training respectively |
| **M2 → M3** | `xgboost_test_predictions.npz`, `dae_test_predictions.npz` | `module2_detection/module2_train_models.py` | `module3_risk_scoring/module3_risk_scores.py` | [CONSISTENT] — M2 writes NPZ prediction arrays; M3 loads via `np.load(raw_path)["y_proba"]` at line 265 |
| **M3 → M4** | `risk_scores.npz` (written to `results/reports/`); M4 loads raw parquet directly | `module3_risk_scoring/module3_risk_scores.py` | `module4_explanations/module4_explanations.py` | **[INCONSISTENCY FOUND]** — M3 writes `risk_scores.npz`; M4 does NOT load this file. M4 instead loads raw `demo_phase1.parquet` and runs models in-memory. M3's NPZ bypasses M4 and feeds M6 directly. The handoff at this boundary is via parquet, not risk_scores.npz. |
| **M4 → M5** | `MVEOutput` and `SHAPContext` in-memory objects | `src/mve_generator.py`, `module4_online_explainer.py` | `module5_responses/module5_pipeline.py` | [CONSISTENT] — Online path passes in-memory objects; batch path uses `analyst_report.json` on disk |
| **M5 → M6** | `alert_responses.json`, `audit_trail.json` | `module5_responses/module5_responses.py` | `module6_evaluation/module6_app.py` | [CONSISTENT] — M5 writes JSON to `results/reports/`; M6 loads via `load_alert_responses()` at line 1120 |

---

### Cross-Check 2 — Figure 3.1 Supporting Evidence (Five Primary Handoff Artifacts)

| Artifact file path | Producing module / script | Consuming module / script | Suggested Figure 3.1 label |
|--------------------|--------------------------|--------------------------|---------------------------|
| `data/processed/train_phase1.parquet` | M1 / `module1_preprocessing/phase1/pipeline.py` | M2 / `module2_detection/module2_train_models.py` | **Scaled Training Features (Parquet)** |
| `results/models/xgboost_final_pipeline.pkl` | M2 / `module2_detection/module2_train_models.py` | M3/M4 / `module3_risk_scores.py`, `module4_online_explainer.py` | **Signed XGBoost Classifier (ECDSA-pkl)** |
| `results/reports/risk_scores.npz` | M3 / `module3_risk_scoring/module3_risk_scores.py` | M6 / `module6_evaluation/module6_app.py` | **Composite Risk Scores (NPZ)** |
| `results/reports/evaluation_alerts.json` | M6 curation / `module6_evaluation/curate_demo_alerts.py` | M6 study / `module6_evaluation/study_loader.py` | **Curated Evaluation Alerts (JSON)** |
| `results/reports/audit_log.jsonl` | M0 + M5 / `module0_analysis/phase0/security.py`, `module5_responses/module5_pipeline.py` | M6 / `module6_evaluation/module6_app.py` | **Hash-Chained Audit Trail (JSONL)** |

---

### Cross-Check 3 — Data State Summary per Transformation

| Transformation | Input state | Output state | Key change |
|---------------|-------------|-------------|-----------|
| **T1** | Raw WUSTL-EHMS-2020 CSV; unverified integrity; mixed PHI and network features | Integrity-verified, identifier-removed, leakage-guarded Parquet matrices (6 files); frozen 4-way split; RobustScaler sidecar | Untrusted → cryptographically attested, partition-bounded feature matrices |
| **T2** | Scaled feature matrices partitioned into train/val/test/demo/benign subsets | ECDSA-signed classifier pickles; DAE JSON+H5; F2-calibrated thresholds; reproducibility metadata (random seed, architecture, SHA-256 provenance) | Mutable arrays → immutable, cryptographically attested, self-describing model artifacts |
| **T3** | Raw per-alert feature vector (float32 × 25); device identifier | `ScoredAlert` object: composite risk score R ∈ [0,1], risk tier, `should_surface` bool, `fusion_class`, `data_quality`, safety-floor guarantee | Numerical vector → semantically enriched risk object with context-adaptive threshold and patient-safety invariants |
| **T4** | `ScoredAlert` with risk tier and feature vector; pre-loaded signed classifiers | `SHAPContext` (top-3 features, stability score, faithfulness flag) + role-differentiated `MVEOutput` (3-layer prose) + governance-verified `ResponseRecommendation` | Risk object → natural language explanation with role differentiation, PHI filtering, and operator decision contracts |
| **T5** | `MVEOutput`, `SHAPContext`, `ResponseRecommendation` objects; persisted M4 artifacts | Hash-chained `audit_log.jsonl`; `alert_responses.json`; `evaluation_alerts.json` with Group A/B display; study shuffle with MD5-seeded counterbalancing | Ephemeral objects → durable, cryptographically auditable records; curated human-subjects study materials |

---

## CONCLUDING SUMMARY (≤250 words)

### (1) Three strongest evidential items confirming the five-transformation data flow

**Item 1 — End-to-end cryptographic continuity:** The ECDSA P-256 signing key is shared between Module 0 (`module0_analysis/phase0/security.py:506–552`) and Module 5 (`module5_responses/module5_pipeline.py:791–946`). Phase 0 integrity events and Phase 5 operator decisions are both committed to `results/reports/audit_log.jsonl` under the same key identity. This is implemented cross-module audit continuity, not documentation abstraction.

**Item 2 — Leakage barrier with runtime enforcement at T1/T2 boundary:** The `_assert_no_demo_leakage()` guard (`module2_train_models.py:67–77`) raises a `RuntimeError` with verbatim architectural guidance if the frozen demo split is loaded during training. This RuntimeError, validated in `tests/test_data_split_integrity.py:154–159`, operationalizes the frozen-split invariant — a code-level, not comment-level, guarantee.

**Item 3 — ScoredAlert as a typed transformation boundary:** The `ScoredAlert` dataclass (`src/data_models.py:156–182`) defines a precise contract at the T3 exit boundary with fields for `fusion_class` (T3-B provenance), `data_quality` (T3-A EA-06 status), and `should_surface` (T3-E safety floor outcome). T4 receives exactly this object and produces `SHAPContext` and `MVEOutput` — the transformation boundaries are type-enforced, not just conceptual.

### (2) Single most important artifact

**`results/models/xgboost_final_pipeline.pkl` (ECDSA-signed classifier).** Its absence breaks T2→T3 (no risk scoring), T2→T4 (no SHAP explanation), and forces Mode B fallback in T4→T5. Every downstream transformation either directly loads it or relies on predictions it produced.

### (3) Evidence gaps

- **benign_medians.json computation script absent:** The script producing `data/processed/benign_medians.json` is not in the current source tree. T1-D can attest consumption but not production.
- **M3→M4 handoff mischaracterized in request:** Module 4 does not load `risk_scores.npz`. M4 recomputes from raw parquet and in-memory models. The thesis should describe this handoff as "parquet + in-memory models" rather than a risk-score NPZ transfer.
- **`compute_rq1_metrics.py` input path:** Agent found it reads `evaluation_alerts.json` (demo-sourced), not `risk_scores.npz` (test-split). The thesis author should verify paper metrics independence before claiming test-split isolation.

### (4) Discrepancies affecting five-transformation characterization

- **[DISCREPANCY] M3→M4 boundary is not a risk-score handoff.** M3 writes `risk_scores.npz`; M4 ignores it and re-reads `demo_phase1.parquet`. The boundary artifact is a Parquet file, not the scored NPZ. The NPZ goes directly M3→M6.
- **[INCONSISTENCY FLAG] SHAP faithfulness flag not surfaced in MVE prose.** `shap_source = "xgboost_low_confidence"` exists in `SHAPContext` but is not propagated into Layer 1 explanation text for NOVEL_ANOMALY alerts — a known gap documented at `src/data_models.py:224–225`.
- **[INCONSISTENCY FLAG] DAE artifacts lack ECDSA signing.** Only classifier pickles receive ECDSA P-256 signing. DAE artifacts (`dae_detector.json`, `dae_model.weights.h5`) rely on SHA-256 content-hash for provenance, not the same signing scheme applied to Track A classifiers.
