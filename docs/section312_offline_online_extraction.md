# Sub-section 3.1.2 Extraction: Offline Batch and Online Inference Operational Model

**Extraction target:** Sub-section 3.1.2 — "Offline Batch and Online Inference Operational Model"  
**Scope:** `module0_analysis/`, `module1_preprocessing/`, `module2_detection/`, `module3_risk_scoring/`,
`module4_explanations/`, `module5_responses/`, `module6_evaluation/`, `src/`, `scripts/`, `configs/`,
`ARCHITECTURE.md`  
**Label conventions:** `[IMPLEMENTED]` = evidenced in source code; `[DOCUMENTED_ONLY]` = appears in
comments/docs/ARCHITECTURE.md but no corresponding implementation found;
`[INCONSISTENCY FLAG]` = conflict between two sources  
**Citation format:** `filepath:line_number` (relative to repo root)  
**Date extracted:** 2026-05-11  

---

## AREA 1: Offline Artifact Production Evidence

### 1.A — Artifact Persistence Patterns

**[IMPLEMENTED] — Disk-based artifact production:**

| File | Function | Lines | Artifact path/name | What is persisted | Status |
|------|----------|-------|-------------------|-------------------|--------|
| `module2_detection/module2_train_models.py` | `train_track_a` | 265 | `f"{name}_final_pipeline.pkl"` | ECDSA-signed fitted classifier (XGBoost/RF/DT) | [IMPLEMENTED] |
| `module2_detection/module2_train_models.py` | `train_track_a` | 285 | `f"{name}_final_report.json"` | Training metrics, best hyperparameters, optimal threshold, random seed | [IMPLEMENTED] |
| `module2_detection/module2_train_models.py` | `train_track_a` | 300 | `f"{name}_test_predictions.npz"` | y_true, y_pred, y_proba, row_id from test split | [IMPLEMENTED] |
| `module2_detection/module2_train_models.py` | `train_track_a` | 305 | `f"{name}_oof_proba.npy"` | Out-of-fold probabilities from cross-validation | [IMPLEMENTED] |
| `module2_detection/module2_train_models.py` | `train_track_a` | 325 | `f"{name}_val_proba.npy"` | Held-out validation-set probabilities (GAP-L1-1 fix) | [IMPLEMENTED] |
| `module2_detection/module2_train_models.py` | `train_track_b_dae` | 439-442 | `"dae_detector.json"` and `"dae_model.weights.h5"` | DAE architecture config + trained Keras weights | [IMPLEMENTED] |
| `module2_detection/module2_train_models.py` | `train_track_b_dae` | 463 | `"dae_final_report.json"` | DAE metrics, architecture, data source, random seed | [IMPLEMENTED] |
| `module2_detection/module2_train_models.py` | `train_track_b_dae` | 468-471 | `"dae_test_predictions.npz"` | y_true, y_pred, reconstruction_error from test split | [IMPLEMENTED] |
| `module2_detection/module2_train_models.py` | `predict_demo` | 540-543 | `f"{name}_demo_predictions.npz"` | y_true, y_pred, y_proba, row_id from demo split | [IMPLEMENTED] |
| `module2_detection/module2_train_models.py` | `predict_demo` | 553-557 | `"dae_demo_predictions.npz"` | y_true, y_pred_dae, reconstruction_error, row_id from demo split | [IMPLEMENTED] |
| `module3_risk_scoring/module3_risk_scores.py` | `save_outputs` | 1222 | `"risk_scores.npz"` | R, c_detect, d_crit, s_data, d_clinical_tier, c_track_a, c_track_b, risk_levels, y_true, fusion_class, data_quality | [IMPLEMENTED] |
| `module3_risk_scoring/module3_risk_scores.py` | `save_outputs` | 1237 | `"risk_scores_detail.csv"` | Per-row detailed breakdown of all risk components | [IMPLEMENTED] |
| `module3_risk_scoring/module3_risk_scores.py` | `save_outputs` | 1293 | `"risk_report.json"` | Risk formula, weights, thresholds, level distribution, dual-track fusion analysis, limitations | [IMPLEMENTED] |
| `module3_risk_scoring/module3_risk_scores.py` | `export_config_jsons` | 946-947 | `"device_criticality.json"` | Device tiers, CIA threat profiles, static configuration | [IMPLEMENTED] |
| `module3_risk_scoring/module3_risk_scores.py` | `export_config_jsons` | 960-961 | `"data_sensitivity.json"` | Data sensitivity classification tiers, static configuration | [IMPLEMENTED] |
| `module3_risk_scoring/module3_risk_scores.py` | `export_config_jsons` | 978-979 | `"risk_config.json"` | Risk formula, weights, thresholds, biometric features, device tiers, response mapping | [IMPLEMENTED] |
| `module3_risk_scoring/module3_demo_scores.py` | `main` | 99-118 | `"demo_scores.npz"` | row_id, y_true, c_track_a, c_track_b, c_detect, d_crit, s_data, d_clinical_tier, R, risk_levels, fusion_class, data_quality, attack_category | [IMPLEMENTED] |
| `module4_explanations/build_shap_background.py` | `main` | 63-72 | `"shap_background.pkl"` | 200-sample stratified training subset, feature names, metadata | [IMPLEMENTED] |
| `module4_explanations/module4_explanations.py` | `save_shap_values` | 260-261 | `f"shap_values_{model_name}.npz"` | TreeSHAP values, expected value, feature names | [IMPLEMENTED] |
| `module4_explanations/module4_explanations.py` | `save_global_importance` | 278-280 | `f"global_importance_{model_name}.json"` | Ranked feature importance by mean absolute SHAP | [IMPLEMENTED] |
| `module4_explanations/module4_explanations.py` | `save_dae_errors` | 320-321 | `"dae_feature_errors.npz"` | Per-feature reconstruction error, weighted error, feature weights | [IMPLEMENTED] |
| `module4_explanations/module4_explanations.py` | `build_analyst_report` | 668 | `"analyst_report.json"` | Per-alert analysis, model consensus, top features, severity | [IMPLEMENTED] |
| `module4_explanations/module4_explanations.py` | `build_clinician_summaries` | 732 | `"clinician_summaries.json"` | Plain-language alerts per XGBoost flag, clinical context | [IMPLEMENTED] |
| `module4_explanations/module4_explanations.py` | `build_admin_dashboard` | 821 | `"admin_dashboard.json"` | Aggregated alert stats, model rankings, biometric/network breakdown | [IMPLEMENTED] |
| `module4_explanations/module4_explanations.py` | `export_feature_concepts` | 859 | `"feature_concepts.json"` | Feature-to-concept mapping (categories, interpretations) | [IMPLEMENTED] |
| `module4_explanations/module4_explanations.py` | `export_nlg_templates` | 909 | `"nlg_templates.json"` | Natural language generation templates for stakeholders | [IMPLEMENTED] |
| `module4_explanations/module4_explanations.py` | `generate_example_explanations` | 1123 | `"example_explanations.json"` | Multi-stakeholder views of 5 diverse alert examples | [IMPLEMENTED] |
| `module4_explanations/module4_explanations.py` | `validate_consistency` | 1184 | `"validation_consistency.json"` | SHAP vs native importances comparison | [IMPLEMENTED] |
| `module4_explanations/module4_explanations.py` | `validate_perturbation` | 1256 | `"validation_perturbation.json"` | Faithfulness check: F1 drop from masking top features | [IMPLEMENTED] |
| `module4_explanations/module4_explanations.py` | `validate_cross_model` | 1322 | `"validation_cross_model.json"` | SHAP ranking agreement across Track A models | [IMPLEMENTED] |

---

### 1.B — Signed/Hash-Verified Artifact Patterns

**[IMPLEMENTED] — Cryptographic provenance protection:**

| File | Function | Lines | Property protected | Status |
|------|----------|-------|-------------------|--------|
| `module2_detection/module2_train_models.py` | `train_track_a` | 265 | ECDSA signature on classifier pickle via `dumps_signed()` | [IMPLEMENTED] |
| `module2_detection/module2_train_models.py` | `train_track_a` | 280 | `"random_seed": int(RANDOM_STATE)` embedded in `{name}_final_report.json` | [IMPLEMENTED] |
| `module2_detection/module2_train_models.py` | `train_track_b_dae` | 447 | `"architecture": "raw_25dim"` embedded in DAE report (Phase B marker) | [IMPLEMENTED] |
| `module2_detection/module2_train_models.py` | `train_track_b_dae` | 457 | `"random_seed": int(RANDOM_STATE)` embedded in `dae_final_report.json` | [IMPLEMENTED] |
| `module4_explanations/module4_explanations.py` | `write_json_sync` | 53-57 | Atomic JSON write via tmp-and-rename (write to `.tmp`, then `tmp.replace(path)`) | [IMPLEMENTED] |
| `module4_explanations/module4_explanations.py` | `compute_tree_shap` | 228-230 | Signed-pickle load: `loads_signed()` refuses unsigned/tampered classifiers | [IMPLEMENTED] |
| `module4_explanations/module4_explanations.py` | `compute_dae_feature_errors` | 301 | DAE loaded via `DAEDetector.from_artefacts()` (JSON + H5, pickle-free) | [IMPLEMENTED] |

**[INCONSISTENCY FLAG]** — No explicit content-hash (SHA256) is computed and persisted for intermediate NPZ/JSON artifacts. ECDSA signing only applies to classifier pickles. Intermediate NPZ files (`risk_scores.npz`, `shap_values_*.npz`, etc.) and JSON reports rely on filesystem integrity, not cryptographic verification.

---

### 1.C — Frozen Artifact Consumption Guards

**[IMPLEMENTED] — Demo-split leakage prevention:**

| File | Function | Lines | Exception type | Guard message (verbatim) | Status |
|------|----------|-------|----------------|--------------------------|--------|
| `module2_detection/module2_train_models.py` | `_assert_no_demo_leakage` | 67-77 | `RuntimeError` | `"Module 2 training functions must not load demo_phase1.parquet. Strategy 1 invariant: the demo split is frozen and may only be touched at inference time (see module2_train_models.predict_demo). If you need demo predictions, run predict_demo() on already-fitted pipelines — do not refit on demo rows."` | [IMPLEMENTED] |
| `module2_detection/module2_train_models.py` | `load_data` | 84-85 | `RuntimeError` via `_assert_no_demo_leakage` | Same guard called on both `train_path` and `test_path` | [IMPLEMENTED] |

**Strategy 1 invariant comments:**
- Lines 57-63: Docstring explaining frozen-split enforcement: "demo split is FROZEN — never seen by any model during training"
- Line 288: Test split loaded separately for evaluation (no refit on test rows)
- Lines 622-627: Post-training `predict_demo()` loads demo split only after models are fitted

---

### 1.D — Configuration Externalization

#### `configs/composite_risk_weights.yaml`

**Weights (policy parameters, sum = 1.0):**
- `detection_confidence` (w_C): **0.40**
- `device_criticality` (w_dcrit): **0.25**
- `data_sensitivity` (w_sdata): **0.15**
- `clinical_tier` (w_dclin): **0.20**

**Tier boundaries:**
- `critical_min`: **0.80**
- `high_min`: **0.60**
- `medium_min`: **0.40**

**Calibration metadata:**
- `anchored_to`: `"EHMS-2020 4-way test split"`, `date`: `"2026-05-07"`, `validated_against_distribution`: `true`
- Expected distribution: CRITICAL ~5%, HIGH ~20%, MEDIUM ~40%, LOW ~35%

**Governance:** `reviewers: ["CISO", "Patient Safety Officer", "Clinical Engineering Director"]`, `review_period: "12 months"`

#### `configs/risk_adaptive_thresholds.yaml`

**Base threshold:** `base_threshold`: **0.50**

**By-criticality multiplier table:**

| Criticality | Patchable status | Multiplier |
|---|---|---|
| CRITICAL | unpatchable | **0.70** |
| CRITICAL | patchable | **0.80** |
| HIGH | unpatchable | **0.85** |
| HIGH | patchable | **0.90** |
| MEDIUM | unpatchable | **0.95** |
| MEDIUM | patchable | **1.00** |
| LOW | unpatchable/patchable | **1.00** |

**By-device-class overrides:**

| Device class | Unpatchable | Patchable |
|---|---|---|
| infusion_pump | **0.70** | **0.85** |
| ventilator | **0.70** | **0.85** |
| patient_monitor | **0.75** | **0.90** |
| ehr_workstation | **0.80** | **0.95** |
| unknown | **0.70** (conservative) | **0.80** |

**Track A per-device surfacing thresholds (F2-tuned, baseline 0.05):** default: **0.05**; infusion_pump/ventilator: **0.03**; imaging: **0.07**; ehr_workstation: **0.10**

**Similar-events campaign detection:** `threshold_count`: 5, `reduction`: 0.20, `floor`: 0.50, `time_window_minutes`: 60

**Governance:** `reviewers: ["CISO"]`, `review_period: "12 months"`

---

## AREA 2: Online Inference Path Evidence

### 2.A — Latency or Performance Specifications

| File | Function | Lines | Exact value/constraint | Enforcement | Status |
|------|----------|-------|------------------------|-------------|--------|
| `module4_explanations/module4_online_explainer.py` | Module docstring | 5 | `<150ms` SLA for per-alert explanations using TreeSHAP + DAE decomposition + NLG | Documentation only | [DOCUMENTED_ONLY] |
| `module4_explanations/module4_online_explainer.py` | `STABILITY_HIGH` constant | 148 | `0.90` (cutoff for `is_stable` boolean; ARCHITECTURE.md Step [11]) | Runtime decision logic used in `compute_shap_stability` (line 389) and `build_shap_context` (line 389) | [IMPLEMENTED] |
| `module4_explanations/module4_online_explainer.py` | `STABILITY_N_SAMPLES` constant | 144 | `10` (bootstrap samples for SHAP stability check) | Passed to `compute_shap_stability` (line 158) | [IMPLEMENTED] |
| `module4_explanations/module4_online_explainer.py` | `plot_latency_cdf()` | 774 | SLA thresholds: `[50, 100, 150]` ms; vectorised broadcast | Reporting/chart only; no runtime assertion | [DOCUMENTED_ONLY] |
| `module4_explanations/module4_online_explainer.py` | `main()` summary | 970 | `"<150ms per alert"` SLA pass/fail: `p95 < 150 else "FAIL"` | Post-hoc check after all samples run; no real-time enforcement | [DOCUMENTED_ONLY] |
| `src/preprocessing.py` | `sanitize_features()` | 49-50 | `NAN_RATE_DEGRADED = 0.05` (5%); `NAN_RATE_FAILED = 0.50` (50%) | Runtime comparisons at lines 117, 123; `DataQuality` flag assignment | [IMPLEMENTED] |

**[INCONSISTENCY FLAG] — Latency SLA enforcement gap:** `<150ms` SLA at `module4_online_explainer.py:5` is documentation-only. Runtime check at line 970 is post-hoc (after all samples complete) and does not raise an exception if p95 exceeds 150ms; no per-alert assertion exists.

---

### 2.B — Frozen Artifact Loading at Inference Time

| File | Function | Lines | Artifact filename | What it represents | Status |
|------|----------|-------|-------------------|--------------------|--------|
| `src/preprocessing.py` | `load_benign_medians()` | 66-72 | `data/processed/benign_medians.json` | Per-feature benign-median lookup table; computed once from training benign_only_train.parquet; replaces NaN/Inf at inference (EA-06 mitigation); lazy-loaded, cached globally in `_BENIGN_MEDIANS` | [IMPLEMENTED] |
| `src/risk_scorer.py` | `_load_thresholds_yaml()` | 32-38 | `configs/risk_adaptive_thresholds.yaml` | Risk-adaptive threshold multipliers; loaded once at module import; falls back to hardcoded `_RISK_MULT` and `_THRESHOLD_MULT` tables if YAML missing | [IMPLEMENTED] |
| `src/context_enrichment.py` | `_load_device_inventory()` | 129-155 | `configs/device_inventory.yaml` (fallback: `tests/fixtures/device_inventory.yaml`) | Device type inventory with device_criticality, patchable, data_sensitivity; cached globally in `_INVENTORY_CACHE` | [IMPLEMENTED] |
| `src/mve_generator.py` | `_load_role_forbidden_terms()` | 132-157 | `configs/role_action_authorization.yaml` | Per-role forbidden-action term lists (INVARIANT 6); loaded once at module import; layered on `_ROLE_FORBIDDEN_DEFAULTS` | [IMPLEMENTED] |
| `src/mve_generator.py` | `_load_llm_data_flow()` | 996-1015 | `configs/llm_data_flow.yaml` | PHI allow-list for LLM API filtering (Step [12] guard); cached via `@functools.lru_cache(maxsize=1)` | [IMPLEMENTED] |
| `module4_explanations/module4_online_explainer.py` | `_load_feature_categories()` | 102-123 | `configs/feature_categories.yaml` | Feature-to-narrative-phrase mapping; 8 clinical groupings; falls back to inline `_FEATURE_GROUPS` | [IMPLEMENTED] |
| `module4_explanations/module4_online_explainer.py` | `AlertExplainer.__init__()` classifier load | 302-309 | `results/models/xgboost_final_pipeline.pkl`, `random_forest_final_pipeline.pkl`, `decision_tree_final_pipeline.pkl` | Track A classifiers; loaded from process-scoped registry via `get_track_a_classifiers()`; ECDSA-signed | [IMPLEMENTED] |
| `module4_explanations/module4_online_explainer.py` | `AlertExplainer.__init__()` DAE load | 312 | DAE (Track B) via `get_dae()` registry | Denoising autoencoder; loaded once at instantiation; pickle-free singleton | [IMPLEMENTED] |

---

### 2.C — Offline Fallback Mechanisms

| File | Function | Lines | Trigger condition | Fallback behavior | Status |
|------|----------|-------|-------------------|-------------------|--------|
| `src/mve_generator.py` | `_generate_llm()` | 1084-1092 | `ANTHROPIC_API_KEY` absent OR `anthropic` package not installed | Return `None`; caller `generate_mve()` invokes `_generate_rule_based()` (Mode B) | [IMPLEMENTED] |
| `src/mve_generator.py` | `_generate_llm()` try/except | 1156-1195 | Any API error (network, invalid JSON, invalid severity label) | Log warning, return `None`; Mode B activated | [IMPLEMENTED] |
| `src/mve_generator.py` | `generate_mve()` primary fallback | 1271-1278 | LLM returns `None` | Always produces `MVEOutput` via Mode A or Mode B; no alert leaves without explanation | [IMPLEMENTED] |
| `src/preprocessing.py` | `sanitize_features()` | 109-112 | NaN/Inf cells detected in flow vector | Replace with per-feature benign median (not zero, preventing NaN-injection per EA-06) | [IMPLEMENTED] |
| `src/risk_scorer.py` | `_load_thresholds_yaml()` | 32-38 | YAML file missing | Return empty dict; caller uses hardcoded `_RISK_MULT` and `_THRESHOLD_MULT` | [IMPLEMENTED] |
| `src/context_enrichment.py` | `_load_device_inventory()` | 139-155 | `configs/device_inventory.yaml` not found | Try `tests/fixtures/device_inventory.yaml`; if neither exists, return empty dict; UNKNOWN-device fail-safe triggered | [IMPLEMENTED] |
| `src/mve_generator.py` | `_load_role_forbidden_terms()` | 139-157 | `configs/role_action_authorization.yaml` missing or malformed | Return copy of inline `_ROLE_FORBIDDEN_DEFAULTS` | [IMPLEMENTED] |
| `src/mve_generator.py` | `_load_llm_data_flow()` | 999-1008 | `configs/llm_data_flow.yaml` missing | Log warning, return `{"allowed": [], "forbidden": []}`; filtered payload is empty, LLM skipped, Mode B activated | [IMPLEMENTED] |
| `module4_explanations/module4_online_explainer.py` | `_load_feature_categories()` | 102-123 | `configs/feature_categories.yaml` missing | Return copy of inline `_FEATURE_GROUPS` | [IMPLEMENTED] |

---

### 2.D — Per-Alert Processing Contracts (Dataclasses)

#### `ScoredAlert` (`src/data_models.py`)

| Field | Type | Provenance/Audit | Notes |
|-------|------|-----------------|-------|
| `adjusted_score` | `float` | No | Anomaly score after risk multiplier. Range [0.0, 1.0]. |
| `threshold` | `float` | No | Surfacing threshold for device context. Rounded to 4 decimals (line 319). |
| `should_surface` | `bool` | No | `adjusted_score > threshold`. |
| `risk_multiplier` | `float` | No | Multiplier applied to raw anomaly score; >= 1.5 for CRITICAL+unpatchable (line 156); 1.0 for LOW+patchable (line 163). |
| `suppression_reason` | `Optional[str]` | Yes (audit) | Human-readable suppression reason (e.g., `"maintenance window — reduced confidence, verify with biomed"`). |
| `fusion_class` | `FusionClass` | Yes (provenance) | Two-stage fusion outcome: BENIGN, KNOWN_ATTACK, CONFIRMED_ANOMALY, NOVEL_ANOMALY, DISAGREEMENT_ANOMALY. |
| `data_quality` | `DataQuality` | Yes (provenance) | Per-row sanitization outcome: OK, IMPUTED_NAN, DEGRADED, FAILED. |

No `__post_init__` validation on `ScoredAlert`; all fields assigned in `score_alert()`.

---

#### `MVEOutput` (`src/data_models.py`)

| Field | Type | Provenance/Audit | Notes |
|-------|------|-----------------|-------|
| `layer_1` | `dict[str, str]` | No | Keys: baseline_behavior, deviation_description, confidence_indicator. Max 60 words combined. |
| `layer_2` | `dict[str, str]` | No | Keys: affected_system, patient_care_impact, phi_exposure, severity_label, severity_rationale. Max 50 words combined. |
| `layer_3` | `dict[str, str]` | No | Keys: immediate_action, clinical_constraint, escalation_path, timeframe. Max 60 words combined. |
| `alert_involves_clinical_system` | `bool` | Yes (classification) | True for CRITICAL/HIGH/MEDIUM; controls DO_NOT content. |
| `mode_used` | `str` | Yes (audit) | `"A_llm"` or `"B_rule"`. Persisted for audit reproducibility (Step [12]). |
| `llm_provider` | `Optional[str]` | Yes (audit) | Provider name (e.g., `"anthropic"`). `None` for Mode B. |
| `llm_model_version` | `Optional[str]` | Yes (reproducibility) | Model ID (e.g., `"claude-sonnet-4-6"`). `None` for Mode B. Required for post-hoc LLM replay. |
| `llm_full_prompt` | `Optional[str]` | Yes (audit/reproducibility) | Full prompt sent to API. `None` for Mode B. Persisted for Step [16] hash-chain. |
| `llm_full_response` | `Optional[str]` | Yes (audit/reproducibility) | Full raw API response (pre-truncation). `None` for Mode B. |

`@property total_word_count` (lines 293-304) sums word counts across all three layers. No `__post_init__` validation; layer structure and word budgets enforced at generation time.

---

#### `SHAPContext` (`src/data_models.py`)

| Field | Type | Provenance/Audit | Notes |
|-------|------|-----------------|-------|
| `top_category` | `str` | Yes (signal grouping) | Highest-|SHAP|-sum feature group; one of 8 clinical groupings. |
| `top_features` | `List[str]` | Yes (explainability) | Top 3 features by \|SHAP\| value; grounds MVE Layer 1 deviation_description (lines 1319-1337). |
| `shap_direction` | `str` | Yes (signal direction) | `'elevated'` or `'suppressed'`; sign of top-1 SHAP value (line 386). |
| `confidence_from_shap` | `str` | Yes (signal strength) | HIGH (top-1 \|SHAP\| > 0.3), MEDIUM (0.1-0.3), LOW (< 0.1). |
| `stability_score` | `float` | Yes (robustness) | Mean pairwise Jaccard similarity of top-k features across 10 perturbations. Range [0.0, 1.0]. |
| `is_stable` | `bool` | Yes (robustness flag) | `True` iff `stability_score >= 0.90` (line 389; ARCHITECTURE.md Step [11]). |
| `shap_source` | `str` | Yes (fidelity marker) | `"xgboost"` (faithful), `"xgboost_low_confidence"` (DAE drove alert; SHAP flagged not-faithful), or `"dae_recon"` (future). |

No `__post_init__`. `build_shap_context()` (lines 322-391) returns empty dict if `top_features` is empty.

---

#### `ResponseRecommendation` (`src/data_models.py`)

| Field | Type | Provenance/Audit | Notes |
|-------|------|-----------------|-------|
| `primary_action` | `str` | No | Human-readable action verb. |
| `primary_action_code` | `str` | Yes (canonicalization) | Machine-readable code (e.g., `"isolate_device"`, `"escalate_clinical"`). Used for cross-role consistency checks. |
| `rationale` | `str` | Yes (audit) | One-sentence justification (<=25 words); logged to audit trail. |
| `estimated_clinical_impact` | `str` | Yes (impact classification) | One of `"minimal"`, `"moderate"`, `"high"`. Validated at construction (lines 422-427). |
| `operator_decision_required` | `bool` | Yes (governance, INVARIANT 3) | **Always True**; constructor refuses `False` (lines 416-421). |
| `suggested_priority` | `int` | No | Integer [1, 5]; 1 = highest. Validated at construction (lines 428-432). |
| `do_not_actions` | `List[str]` | Yes (constraint list) | Forbidden actions; machine-readable mirror of MVE Layer 3 `clinical_constraint`. |

`__post_init__()` (lines 415-432): enforces `operator_decision_required == True` (INVARIANT 3), validates `estimated_clinical_impact` and `suggested_priority`; raises `ValueError` on violation.

---

#### `OperatorDecision` (`src/data_models.py`)

| Field | Type | Provenance/Audit | Notes |
|-------|------|-----------------|-------|
| `alert_id` | `str` | Yes (identity) | Links decision to alert record. Required. |
| `operator_role` | `str` | Yes (role audit) | One of `"IT_generalist"`, `"biomed_engineer"`, `"nurse_manager"`. |
| `operator_action_taken` | `str` | Yes (action audit) | Free-form action label. Required. |
| `decision_time_seconds` | `float` | Yes (SLA audit) | Time from alert presentation to decision. Non-negative. |
| `timestamp` | `str` | Yes (timing audit) | ISO 8601 UTC timestamp. Required. |
| `operator_confidence` | `Optional[int]` | Yes (subjective assessment) | Likert [1, 5]; validated when present (lines 533-536). |
| `operator_rationale` | `str` | No | Free-form explanation. Optional. |

`validate()` method (lines 525-536): enforces non-empty required fields, `decision_time_seconds >= 0`, `operator_confidence` in [1, 5] when present; raises `ValueError` on violations.

**[INCONSISTENCY FLAG] — SHAP faithfulness caveat not propagated to MVE:** `shap_source = "xgboost_low_confidence"` assigned at `data_models.py:222-226` and `module4_online_explainer.py:374-381` for NOVEL_ANOMALY/STRONG_NOVEL_ANOMALY fusion classes. Neither `_clinician_nlg` (line 423-465) nor MVE generation (lines 1319-1337) carries this flag forward into Layer 1 prose. Operator receives SHAP-derived narrative without awareness that SHAP faithfulness is flagged as "low confidence."

---

## AREA 3: Trade-off Documentation and Acknowledged Limitations

### 3.A — Streaming or Real-Time References

**[IMPLEMENTED]**

| File:line | Verbatim text |
|---|---|
| `module4_explanations/module4_online_explainer.py:2-5` | `"Online-capable per-alert explanation pipeline with latency profiling. Demonstrates that per-alert explanations can be generated within real-time SLA (<150ms) using TreeSHAP + DAE decomposition + NLG."` |
| `module4_explanations/module4_online_explainer.py:143` | `"150 ms latency budget when called only on MEDIUM+ severity alerts."` |
| `module4_explanations/_severity.py:3-4` | `"Used by both the offline batch explainer (module4_explanations.py) and the online streaming explainer (module4_online_explainer.py)."` |
| `module3_risk_scoring/module3_risk_scores.py:171` | `"phi_realtime": 1.0,  # real-time vital signs (SpO2, HR, BP)` |
| `module3_risk_scoring/module3_risk_scores.py:428` | `"Biometric features carry PHI real-time sensitivity (1.0)."` |
| `module5_responses/module5_pipeline.py:671` | `"M5-6: reads only the last 4 KB of the file instead of streaming"` |
| `module6_evaluation/module6_app.py:6` | `"6.3b  Online Simulation — Stream test samples through pipeline in near-real-time"` |
| `module6_evaluation/module6_app.py:834` | `# 6C.1  Streaming data simulator` |
| `module6_evaluation/module6_app.py:841` | `"Yields one alert dict at a time, simulating real-time arrival."` |
| `module6_evaluation/module6_app.py:2194` | `'st.title("IoMT IDS — Real-Time Dashboard")'` |
| `module6_evaluation/module6_app.py:2478` | `"a mock live data source, a real-time latency profile panel, and"` |
| `module6_evaluation/module6_app.py:2687` | `'st.markdown("#### Real-Time Latency Profile")'` |
| `src/harness.py:51` | `"Opt-10: generator-based streaming pipeline — O(1) memory regardless of"` |
| `src/harness.py:157` | `# Opt-10: use streaming generator when no pre-loaded dataset is supplied.` |

**[DOCUMENTED_ONLY]**

| File:line | Verbatim text |
|---|---|
| `src/data_models.py:224-225` | `'"dae_recon" (future work — per-feature DAE reconstruction-error attribution). Lets downstream UIs warn the operator when SHAP narrative may not reflect the actual triggering signal.'` |
| `src/harness.py:312` | `"verdict": "NOT_TESTED — requires longitudinal field deployment (Phase 3)"` |

---

### 3.B — Batch Processing Limitation References

**[IMPLEMENTED]**

| File:line | Verbatim text |
|---|---|
| `module4_explanations/_severity.py:3-4` | `"Used by both the offline batch explainer (module4_explanations.py) and the online streaming explainer (module4_online_explainer.py)."` |
| `module4_explanations/module4_online_explainer.py:8-9` | `"Design: online-capable, validated in batch mode on the test set. Global artifacts (importance rankings, templates) loaded once at startup"` |
| `module6_evaluation/module6_app.py:596` | `"reconstruction-error attribution is future work."` |
| `module6_evaluation/module6_app.py:1346` | `"""Draw a single latency sample for one stage, consistent with the recorded mean / p50 / p95 of the offline latency profile."""` |
| `module1_preprocessing/phase1/scaler.py:152` | `"downstream consumer cannot silently load a stale pickle."` |
| `module1_preprocessing/phase1/_sidecar_io.py:9` | `"silently load a stale, executable byte stream."` |

**[DOCUMENTED_ONLY]**

| File:line | Verbatim text |
|---|---|
| `ARCHITECTURE.md:3` | `"This repository implements an offline-first, explainable intrusion-detection workflow for healthcare and IoMT environments. The system separates batch data preparation and artifact generation from the online user interface. In practice, the pipeline produces scored alerts, explanations, and response guidance offline, and the Streamlit dashboard mainly reads those artifacts for browsing, study flows, and evaluation."` |
| `ARCHITECTURE.md:139-164` | Full OFFLINE section from workflow diagram: 4-way stratified split (train 60%, val 15%, test 15% frozen, demo 10% frozen); threshold calibration on val split; random_state=42 |

---

### 3.C — ARCHITECTURE.md Limitation Sections

#### C.1 — Offline-first design commitment

**ARCHITECTURE.md:1-3**

> This repository implements an offline-first, explainable intrusion-detection workflow for healthcare and IoMT environments. The system separates batch data preparation and artifact generation from the online user interface. In practice, the pipeline produces scored alerts, explanations, and response guidance offline, and the Streamlit dashboard mainly reads those artifacts for browsing, study flows, and evaluation.

---

#### C.2 — Step 17: Outcome Tracking (Future Work)

**ARCHITECTURE.md:730-735** — `[STEP 17] OUTCOME TRACKING (FUTURE WORK — acknowledged)`

```
│  [STEP 17] OUTCOME TRACKING (FUTURE WORK — acknowledged)                 │
│                                                                          │
│   Documented but NOT IMPLEMENTED (requires real deployment):             │
│      ├─ Outcome assessment (was it true positive?)                      │
│      ├─ Clinical follow-up tracking                                     │
│      └─ Operator decision quality assessment                            │
```

---

#### C.3 — Step 18: Continuous Improvement (Future Work)

**ARCHITECTURE.md:737-742** — `[STEP 18] CONTINUOUS IMPROVEMENT (FUTURE WORK — acknowledged)`

```
│  [STEP 18] CONTINUOUS IMPROVEMENT (FUTURE WORK — acknowledged)           │
│                                                                          │
│   Documented but NOT IMPLEMENTED:                                        │
│      ├─ Feedback into model retraining                                  │
│      ├─ Active learning architecture                                     │
│      └─ Threshold auto-tuning                                            │
```

---

#### C.4 — OFFLINE / ONLINE separation (workflow diagram)

**ARCHITECTURE.md:134-167** — Two explicitly separate diagram blocks:

```
┌──── OFFLINE (one-time training) ────┐
│  [1] Data Preparation               │
│       └─ WUSTL-EHMS-2020            │
│       └─ 4-way stratified split:    │
│          • train (60%)              │
│          • val (15%)                │
│          • test (15%, frozen)       │
│          • demo (10%, frozen)       │
│       └─ random_state=42            │
│  ...                                │
│  [4] Threshold Calibration          │
│       └─ Per-track thresholds (val) │
│       └─ Risk-adaptive parameters   │
└─────────────────────────────────────┘
```

```
┌──────────────── ONLINE INFERENCE (per alert) ────────────────┐
```

---

#### C.5 — Step 8: Context Enrichment — KEY LIMITATIONs

**ARCHITECTURE.md:321-329** (patient acuity proxy):

> ★ KEY LIMITATION (Section 11): D_clinical_tier reflects device class, NOT real-time patient acuity. The same infusion pump on a stable post-op patient and a coding ICU patient gets the same tier_1=1.0. Production deployment would integrate EHR acuity scores (NEWS2/MEWS) — documented as Phase-3 work.

**ARCHITECTURE.md:321-329** (MITRE mapping):

> ★ KEY LIMITATION: MITRE mapping is rule-based and static, validated against framework version X.Y. Production would benefit from automated framework synchronization.

---

#### C.6 — Step 9: Composite Risk Scoring — KEY LIMITATIONS (L1-L4)

**ARCHITECTURE.md:367-392**

```
┌─ KEY LIMITATIONS (Section 11) ─────────────────────────────────┐
│                                                                  │
│  L1. Linear weighted sum ≠ true multiplicative risk semantics  │
│      Standard security risk ≈ P(threat) × Impact, but linear    │
│      sum allows compensatory effects (e.g., high D_crit alone   │
│      pushes alert into HIGH tier even when C_detect = 0).      │
│      Production deployment would benefit from R = C_detect ×   │
│      V_asset structure. Linear retained for thesis: simpler,    │
│      bounded [0,1], easier to certify.                          │
│                                                                  │
│  L2. D_clinical_tier is device-class proxy for patient acuity  │
│      Same infusion pump on stable post-op vs coding ICU         │
│      patient gets the same tier_1 = 1.0. Production deployment │
│      with EHR integration (NEWS2/MEWS) would correct this      │
│      asymmetry. Reported FNR_critical averages across acuity    │
│      states; under-detects on unstable patients.                │
│                                                                  │
│  L3. D_crit and D_clinical_tier are correlated (r ≈ X reported │
│      in paper). Combined weight 0.45 effectively double-counts │
│      "device importance," exceeding C_detect's weight 0.40.    │
│      Acknowledged design choice (patient-safety bias); not bug.│
│                                                                  │
│  L4. Tier boundaries calibrated on test split distribution.    │
│      Different deployments (different device mix, different    │
│      attack distributions) may need recalibration.             │
└────────────────────────────────────────────────────────────────┘
```

---

### 3.D — Configuration Governance Metadata

**[IMPLEMENTED]** — All eight YAML configuration files contain `review.*` governance fields:

| Config file | `review.reviewers` | `review.review_period` | `review.last_reviewed` | Additional calibration metadata |
|---|---|---|---|---|
| `configs/composite_risk_weights.yaml` | `["CISO", "Patient Safety Officer", "Clinical Engineering Director"]` | `"12 months"` | not present | `anchored_to: "EHMS-2020 4-way test split"`, `date: "2026-05-07"`, `validated_against_distribution: true` |
| `configs/device_clinical_tier_mapping.yaml` | `["CISO", "Clinical Engineering Director", "Patient Safety Officer"]` | `"12 months"` | `"2026-05-08"` | — |
| `configs/attack_to_mitre_mapping.yaml` | `["Security Team", "Threat Intel"]` | `"quarterly + on MITRE framework release"` | `"2026-05-08"` | `mitre_framework_version: "v14.1"`, per-mapping `last_validated: "2026-05-08"` |
| `configs/risk_adaptive_thresholds.yaml` | `["CISO"]` | `"12 months"` | not present | — |
| `configs/role_action_authorization.yaml` | `["CISO", "Clinical Engineering Director"]` | `"12 months"` | `"2026-05-07"` | — |
| `configs/tier_routing.yaml` | `["CISO", "SOC Lead"]` | `"12 months"` | `"2026-05-07"` | — |
| `configs/hospital_capabilities.yaml` | `["CISO", "Hospital Operations Lead"]` | `"per deployment + on staffing change"` | `"2026-05-07"` | — |
| `configs/llm_data_flow.yaml` | `["Privacy Officer", "CISO"]` | `"annually + on schema change"` | `"2026-05-07"` | `sanitize_before_send: true`, `log_full_prompt: true`, `log_full_response: true`, `max_input_chars: 8192` |

**[DOCUMENTED_ONLY]** — ARCHITECTURE.md:991-993 references example calibration schema:

```yaml
calibration:
  anchored_to: "EHMS-2020 test split"
  date: "[date]"
  validated_against_distribution: true
```

---

## AREA 4: Cross-Module Data Flow Verification

### 4.A — Module 2 → Module 3 Handoff

| Producing file:line | Consuming file:line | Artifact name | Format | Classification |
|---|---|---|---|---|
| `module2_detection/module2_train_models.py:265` | Not consumed by M3 | `xgboost_final_pipeline.pkl` | PKL (ECDSA-signed) | [IMPLEMENTED] OFFLINE |
| `module2_detection/module2_train_models.py:300` | `module3_risk_scoring/module3_risk_scores.py:265` | `xgboost_test_predictions.npz` | NPZ | [IMPLEMENTED] OFFLINE |
| `module2_detection/module2_train_models.py:305` | Not consumed by M3 | `xgboost_oof_proba.npy` | NPY | [IMPLEMENTED] OFFLINE |
| `module2_detection/module2_train_models.py:325` | Not consumed by M3 | `xgboost_val_proba.npy` | NPY | [IMPLEMENTED] OFFLINE |
| `module2_detection/module2_train_models.py:439-442` | `module3_risk_scoring/module3_risk_scores.py:360` (via registry) | `dae_detector.json` + `dae_model.weights.h5` | JSON + H5 | [IMPLEMENTED] OFFLINE |
| `module2_detection/module2_train_models.py:540-543` | `module3_risk_scoring/module3_demo_scores.py:52` | `xgboost_demo_predictions.npz` | NPZ | [IMPLEMENTED] OFFLINE |
| `module2_detection/module2_train_models.py:553-557` | Not directly loaded in M3 | `dae_demo_predictions.npz` | NPZ | [IMPLEMENTED] OFFLINE |

**Notes:** M3 loads XGBoost probabilities via `np.load(raw_path)["y_proba"]` at line 265. DAE is loaded via `get_dae()` registry (online singleton), which internally loads from persisted `dae_detector.json` + `dae_model.weights.h5` (offline disk artifacts).

**OFFLINE/ONLINE boundary:** M2→M3 is entirely **OFFLINE** (NPZ predictions → disk → M3 loads).

---

### 4.B — Module 3 → Module 4 Handoff

| Producing file:line | Consuming file:line | Artifact name | Format | Classification |
|---|---|---|---|---|
| `module3_risk_scoring/module3_risk_scores.py:1222` | `module6_evaluation/module6_app.py:204` (via M6, not M4) | `risk_scores.npz` | NPZ | [IMPLEMENTED] OFFLINE |
| `module3_risk_scoring/module3_demo_scores.py:99-118` | `module4_explanations/module4_explanations.py:193` (raw parquet, not scores NPZ) | `demo_phase1.parquet` | PARQUET | [IMPLEMENTED] OFFLINE |
| `module3_risk_scoring/module3_demo_scores.py:99-118` | `module6_evaluation/module6_app.py:204` | `demo_scores.npz` | NPZ | [IMPLEMENTED] OFFLINE |

**Notes:** M4 does NOT load `risk_scores.npz` or `demo_scores.npz` directly. M4 (`module4_explanations.py:182-203`) loads raw parquets + re-runs models in-memory; M3's NPZ outputs are consumed by M6, not M4. M4's own outputs (`analyst_report.json`, `clinician_summaries.json`) are written to disk and then consumed by M6.

**OFFLINE/ONLINE boundary:** M3→M4 is **OFFLINE** (disk parquet), but M4's internal model inference is **in-memory (ONLINE)**; the NPZ risk scores bypass M4 and go directly to M6.

---

### 4.C — Module 4 → Module 5 Handoff

| Producing file:line | Consuming file:line | Artifact name | Format | Classification |
|---|---|---|---|---|
| `module4_explanations/module4_explanations.py:667-668` | `module6_evaluation/module6_app.py:213` | `analyst_report.json` | JSON | [IMPLEMENTED] OFFLINE |
| `module4_explanations/module4_explanations.py:731-732` | `module6_evaluation/module6_app.py:215` | `clinician_summaries.json` | JSON | [IMPLEMENTED] OFFLINE |
| `module4_explanations/module4_explanations.py:260-261` | Not consumed downstream | `shap_values_{model}.npz` | NPZ | [IMPLEMENTED] OFFLINE |
| `module4_explanations/module4_explanations.py:320-321` | Not consumed downstream | `dae_feature_errors.npz` | NPZ | [IMPLEMENTED] OFFLINE |
| `src/mve_generator.py:254, 674, 722, 773, 883, 975` | `module5_responses/module5_pipeline.py` (batch) OR `module5_responses/module5_pipeline.py` (online) | `MVEOutput` object | In-memory | [IMPLEMENTED] ONLINE |

**Notes:** M4→M5 is **HYBRID**:
- **Batch path (M6 dashboard):** `analyst_report.json` + `clinician_summaries.json` written to disk (lines 667-732); M6 loads at lines 213-215.
- **Online path (M5 closed-loop):** `MVEOutput` objects generated in-memory via `mve_generator.generate_mve()` (returned at lines 254, 674, 722, 773, 883, 975) and passed directly to response handler — never persisted on this path.

---

### 4.D — Module 5 → Module 6 Handoff

| Producing file:line | Consuming file:line | Artifact name | Format | Classification |
|---|---|---|---|---|
| `module5_responses/module5_responses.py:773-774` | `module6_evaluation/module6_app.py:1120` via `load_alert_responses()` | `alert_responses.json` | JSON | [IMPLEMENTED] OFFLINE |
| `module5_responses/module5_responses.py:777-778` | `module6_evaluation/module6_app.py:47,57,58` (audit logger reads/writes) | `audit_trail.json` | JSON | [IMPLEMENTED] OFFLINE |
| `module5_responses/module5_responses.py:781-782` | Not directly loaded by M6 | `effectiveness_analysis.json` | JSON | [IMPLEMENTED] OFFLINE |
| `module5_responses/module5_responses.py:797-798` | Not directly loaded by M6 | `response_report.json` | JSON | [IMPLEMENTED] OFFLINE |

**Notes:** `ResponseRecommendation` objects created in-memory within M5 but are NOT passed to M6; M6 reconstructs from persisted `alert_responses.json`. M5→M6 is entirely **OFFLINE**.

---

### Offline-Online Boundary Summary Table

| Handoff | Mechanism | Classification |
|---------|-----------|----------------|
| M2 → M3 | NPZ files (`xgboost_test_predictions.npz`, etc.) persisted to `results/models/` | **OFFLINE** |
| M3 → M4 | Raw parquet loaded by M4 (NPZ scores bypass M4 → M6 directly) | **OFFLINE** |
| M4 → M5 (batch) | `analyst_report.json` + `clinician_summaries.json` on disk | **OFFLINE** |
| M4 → M5 (online) | `MVEOutput` objects in-memory via `mve_generator.generate_mve()` | **ONLINE** |
| M5 → M6 | `alert_responses.json` + `audit_trail.json` on disk | **OFFLINE** |

**Hybrid operation confirmed at two boundaries:** (1) M3 produces offline `risk_scores.npz` consumed by M6, AND in-memory risk components consumed by M4 explanations; (2) M4 produces both offline `analyst_report.json` (disk, consumed by M6) AND online `MVEOutput` objects (in-memory to M5).

---

## Concluding Summary (≤200 words)

### (a) Three strongest evidential items for offline-first characterization

1. **Strategy 1 frozen-split invariant with runtime enforcement** ([`module2_detection/module2_train_models.py:67-77`](module2_detection/module2_train_models.py#L67-L77)): `_assert_no_demo_leakage()` raises `RuntimeError` with verbatim explanation if `demo_phase1.parquet` is loaded during training — implemented guard, not merely documented.

2. **ECDSA-signed classifier pickles + per-artifact JSON reports with embedded random seed** ([`module2_detection/module2_train_models.py:265, 280-285`](module2_detection/module2_train_models.py#L265)): classifiers written via `dumps_signed()`; reports embed `"random_seed": int(RANDOM_STATE)` and `"architecture": "raw_25dim"` — cryptographic and reproducibility provenance fully implemented.

3. **Configuration externalization with multi-stakeholder governance metadata** ([`configs/composite_risk_weights.yaml`](configs/composite_risk_weights.yaml), [`configs/risk_adaptive_thresholds.yaml`](configs/risk_adaptive_thresholds.yaml)): all policy weights, tier boundaries, and thresholds are YAML-externalized; `review.reviewers`, `review.review_period`, and `calibration.anchored_to` fields implemented across eight config files.

### (b) Three strongest evidential items for online inference path

1. **`AlertExplainer` process-scoped model registry with ECDSA load guard** ([`module4_explanations/module4_online_explainer.py:302-312`](module4_explanations/module4_online_explainer.py#L302-L312)): classifiers loaded once from signed pickles via `get_track_a_classifiers()`; DAE via `get_dae()` pickle-free singleton — frozen artifacts reused per process.

2. **Dual-mode MVE generation with guaranteed fallback** ([`src/mve_generator.py:1271-1278`](src/mve_generator.py#L1271-L1278)): `generate_mve()` always produces `MVEOutput` via Mode A (LLM) or Mode B (rule-based); no alert leaves the pipeline without an explanation — full offline fallback implemented.

3. **`ResponseRecommendation.__post_init__()` enforcing INVARIANT 3** ([`src/data_models.py:415-432`](src/data_models.py#L415-L432)): `operator_decision_required = True` enforced at construction with `ValueError` on violation — zero-auto-execution guarantee is code-level, not documentation-level.

### (c) Evidence gaps the thesis author may wish to address

1. **Latency SLA lacks runtime assertion:** The `<150ms` SLA ([`module4_online_explainer.py:5`](module4_explanations/module4_online_explainer.py#L5)) is documented but not enforced by a per-alert assertion; the only check is post-hoc at `line 970`. The thesis should either acknowledge this as a research-prototype limitation or add a runtime guard before drafting claims about SLA enforcement.

2. **No cryptographic chain-of-custody on intermediate NPZ artifacts:** ECDSA signing applies only to classifier pickles; `risk_scores.npz`, `shap_values_*.npz`, and JSON reports have no content-hash. The thesis should acknowledge that intermediate artifact integrity relies on filesystem trust, not cryptographic verification.

3. **SHAP faithfulness flag not surfaced to operator for NOVEL_ANOMALY alerts:** `shap_source = "xgboost_low_confidence"` is assigned but not propagated into MVE Layer 1 prose; the operator reads SHAP-derived narrative without a fidelity warning for DAE-driven alerts. The thesis should document this as a known gap (see `src/data_models.py:224-225` future-work note) or implement the propagation before describing the explanation subsystem as complete.
