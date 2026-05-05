# XAI-IDS-Healthcare Full Pipeline
# Read this file COMPLETELY before writing any code.
# Last updated: 2026-04-14

---

## WHAT THIS PROJECT IS

Research prototype for:
"Human-in-loop Explainable + Risk-Adaptive IDS for mid-sized
healthcare organizations (200–500 beds)."

Target user: IT security generalist (NOT SOC specialist).
Processes 10–50 alerts/day alongside EHR support and network admin.
Needs clinically contextualized explanations, not raw anomaly scores.

Full spec: research_spec.yaml (read before touching any module)

---

## ARCHITECTURE — 6 MODULES + 3 CROSS-CUTTING FILES

```
OFFLINE TRAINING (run once)
  Module 0  phase0/                   Dataset audit
  Module 1  phase1/                   Preprocessing + SMOTE
  Module 2  module2_train_models.py   XGB/RF/DT + DAE → models/

ONLINE INFERENCE (per alert)
  Module 3  module3_risk_scores.py    Risk-adaptive gate
  Module 4  module4_online_explainer.py  SHAP → shap_context
            src/mve_generator.py      3-layer MVE (≤150 words)
  Module 5  module5_responses.py      Output to IT Generalist

EVALUATION
  Module 6  module6_evaluation.py     M1–M8 + study_analysis
            src/harness.py            Thin wrapper (backward-compat)

CROSS-CUTTING
  drift_detection.py
  dynamic_threshold_sim.py
  feedback_loop_demo.py
```

Key invariant: Track B (DAE) only elevates anomaly_score, never suppresses.
Suppression happens at Module 3 only. mve_generator.py is NEVER called
on suppressed alerts.

---

## MODULE CONTRACTS

### Module 2 — Train Models
File: `module2_train_models.py`

Track A (supervised, SMOTE-balanced):
- XGBoost   → P_xgb(attack)
- RandomForest → P_rf(attack)
- DecisionTree → P_dt(attack)
- Input: 25 raw network features
- OOF probabilities fed into DAE

Track B (unsupervised):
- DAE input: concat([25 raw features, P_xgb, P_rf, P_dt])  # 28-dim
- DAE output: reconstruction_error → normalized [0, 1]
- Trained on benign-only traffic

Fusion: `anomaly_score = max(Track_A_score, Track_B_score)`

Serialize to: `models/xgb.pkl, rf.pkl, dt.pkl, dae.pkl, scaler.pkl`

---

### Module 3 — Risk-Adaptive Scoring
File: `module3_risk_scores.py`
(`src/risk_scorer.py` is the primary per-alert implementation; it wraps and extends batch logic from `module3_risk_scores.py` with a per-alert dict interface, patchability-aware thresholds, and a safety floor for CRITICAL+unpatchable devices. It is not a thin alias.)

Function: `score_alert(anomaly_score, device_context, event_context) → ScoredAlert`

Rules (non-negotiable):
- CRITICAL + unpatchable  → threshold lowered ≥30%, risk_multiplier ≥1.5
- Maintenance window + known vendor IP → suppress (should_surface=False)
- LOW + patchable         → default threshold, risk_multiplier=1.0
- similar_events > 5 in 30d → reduce risk_multiplier by 0.2

---

### Module 4 — SHAP Explainer
File: `module4_online_explainer.py`

Function: `explain_alert(feature_vector, model) → SHAPContext`

- Compute SHAP values against XGBoost model
- Map features to 7 clinical feature groups (see research_spec.yaml)
- Output: top_category, top_features (top 3), shap_direction, confidence_from_shap
- Called ONLY when should_surface=True

---

### MVE Generator — UPDATED SIGNATURE
File: `src/mve_generator.py`

```python
def generate_mve(
    raw_alert: dict,
    device_context: dict,
    behavioral_baseline: dict,
    user_context: dict | None = None,
    shap_context: dict | None = None,   # NEW in v2.0
) -> MVEOutput:
```

Layer 1 rules:
- If shap_context provided: deviation_description MUST mention
  shap_context.top_category AND at least 1 shap_context.top_feature
- If shap_context is None: fall back to rule-based deviation (v1.0 behavior)

Layer 3 rules (updated from mve_improvement_analysis.yaml):
- ALL EHR access alerts: always include force-reauth instruction,
  regardless of severity. Severity = urgency, NOT whether action is needed.
- IoMT CRITICAL/HIGH: clinical_constraint MUST distinguish between
  network isolation (safe) and device power-off/physical disconnect (prohibited).
  Example: "DO NOT power off ventilator. Blocking port 23 at switch is SAFE."
- Layer 3 immediate_action for data_exfiltration: specify exact scope
  (destination IP only), what is preserved, and why partial > full isolation.
- LOW-severity alerts with benign hypothesis: lead Layer 1 with benign
  explanation if destination is internal and matches known pattern.

Layer 1 rules (updated):
- For IoMT behavioral deviation: deviation_description MUST include
  numeric baseline comparison when available.
  Example: "Normal: 12 DNS queries/min. Observed: 310 queries/min."
- Add role_authorization_check to Layer 1 for unauthorized_ehr_access alerts:
  "Role authorization: CONFIRMED / UNCONFIRMED / DENIED"
  If UNCONFIRMED → Layer 3 must recommend force-reauth regardless of severity.

Output constraints (unchanged):
- Total ≤150 words. Layer 1 ≤60. Layer 2 ≤50. Layer 3 ≤60.
- No raw SHAP values in output text
- No CVSS scores — use clinical CRITICAL/HIGH/MEDIUM/LOW only
- No vague actions ("investigate further", "monitor closely")

Mode A (LLM): Use if ANTHROPIC_API_KEY in environment.
Mode B (rules): Always implement. Must pass all tests offline.

---

### Module 5 — Recommendation Output
File: `module5_responses.py`

- Format MVEOutput for IT Generalist display
- NEVER auto-execute any action
- Output: structured dict (CLI / API / Streamlit-ready)

---

### Module 6 — Evaluation

Files: `module6_evaluation/module6_evaluation.py`, `module6_evaluation/module6_app.py`,
       `module6_evaluation/_src_adapter.py`, `module6_evaluation/compute_rq2_metrics.py`,
       `module6_evaluation/study_loader.py`, `module6_evaluation/study_analysis.py`,
       `src/harness.py`

Submodule contracts:

#### _src_adapter.py — `scored_from_eval_alert(alert_data: dict) -> ScoredAlert`

- Bridges one `evaluation_alerts.json` record into `src.risk_scorer.score_alert()`
- Safe defaults: `patchable=True` (unknown device), `event_context=None`

#### compute_rq2_metrics.py (standalone script)

- Input: `results/reports/evaluation_alerts.json`
- Output: `results/rq2_metrics.json`
- Computes: `critical_alert_rate`, `fnr_critical`, `{TP,FN,FP,TN}`, `sensitivity`, `specificity`

#### study_loader.py

- `load_study_alerts(participant_id) -> list[AlertScenario]`
- Deterministic shuffle: `random.Random(int(md5(participant_id).hexdigest(), 16))`
- A/B assignment: counterbalanced by `pid_seed % 2`

#### study_analysis.py (standalone script)

- Input: `survey/study_responses_*.json`
- Output: `survey/m5_result.yaml`
- Primary test: Mann-Whitney U (one-tailed B > A); thresholds: target=0.30, minimum=0.15
- Verdict: PASS / WARN / FAIL

---

## FILE STRUCTURE

```text
ids-healthcare-cip/
├── module0_analysis/
│   └── phase0/                   # dataset audit
├── module1_preprocessing/
│   └── phase1/                   # preprocessing + SMOTE
├── module2_detection/
│   └── module2_train_models.py   # XGB/RF/DT/DAE → results/models/
├── module3_risk_scoring/
│   └── module3_risk_scores.py    # batch composite risk scoring
├── module4_explanations/
│   ├── module4_explanations.py   # batch SHAP + stakeholder outputs
│   └── module4_online_explainer.py
├── module5_responses/
│   ├── module5_responses.py      # recommendation output
│   └── module5_pipeline.py       # PolicyEngine + feedback loop
├── module6_evaluation/
│   ├── module6_evaluation.py     # build evaluation artifacts
│   ├── module6_app.py            # Streamlit dashboard (browse + study mode)
│   ├── _src_adapter.py           # bridges eval artifacts → src.risk_scorer
│   ├── compute_rq2_metrics.py    # RQ2 metrics → results/rq2_metrics.json
│   ├── study_loader.py           # A/B scenario loader (MD5 shuffle)
│   └── study_analysis.py         # M5 Mann-Whitney → survey/m5_result.yaml
├── src/
│   ├── __init__.py
│   ├── data_models.py            # MVEOutput, ScoredAlert, SHAPContext, ...
│   ├── mve_generator.py          # v2.0: +shap_context param
│   ├── risk_scorer.py            # per-alert scoring + patchability + safety floor
│   └── harness.py                # thin wrapper → module6_evaluation.py
├── common/
│   ├── model_registry.py
│   ├── phi.py
│   └── signed_pickle.py
├── analysis/
│   └── analyze_rq3.py            # final A/B study analysis
├── utils/
│   └── convert_legacy_survey.py  # legacy survey format conversion
├── tests/
│   ├── __init__.py
│   ├── acceptance_tests.py       # M1–M8 (includes M5 SHAP alignment)
│   ├── negative_tests.py
│   ├── test_safe_failure.py      # 5 failure-mode tests (added post-spec)
│   ├── test_coverage_mve.py      # MVE branch coverage (added post-spec)
│   └── fixtures/
│       ├── sample_alerts.yaml
│       ├── device_inventory.yaml
│       ├── behavioral_baselines.yaml
│       └── shap_stubs.yaml       # stub SHAPContext for offline tests
├── run_tests.py
├── run_all_modules.py
├── alignment_report.yaml         # generated
├── survey/m5_result.yaml         # generated
├── results/rq2_metrics.json      # generated
├── research_spec.yaml
└── CLAUDE.md
```

---

## DO NOT BUILD

- Device discovery / network scanning
- Automated enforcement / blocking (recommend only, never execute)
- RF / proprietary wireless protocol detection (non-IP IoMT)
- Ransomware early-detection claims
- UI / frontend (Streamlit in module6_app.py is evaluation-only)
- Database / persistence (in-memory + YAML fixtures only)
- Authentication / authorization

If asked to build any of the above: refuse and cite this file.

---

## DONE CONDITION

Prototype is COMPLETE when `run_tests.py` produces:

### Automated Tests (all must pass at ≥minimum)

| Test | Minimum | Target |
|---|---|---|
| test_mve_completeness (M1) | 85% | 95% |
| test_layer1_length_constraint (M1b) | 90% | 95% |
| test_clinical_relevance (M2) | 75% | 90% |
| test_actionability (M3) | 70% | 85% |
| test_clinical_constraint_awareness (M4) | 80% | 90% |
| test_shap_narrative_alignment (M5) | 75% | 85% |
| test_false_positive_rate (M6) | 20% FP reduction | 40% |
| test_risk_adaptive_threshold (M7) | 100% | 100% |
| test_severity_label_accuracy (M8) | 70%, 0 CRITICAL↔LOW | 80% |

M8 hard fail: any CRITICAL↔LOW mismatch = immediate BLOCKED.

### Negative Tests (all must pass, 0 violations)

- test_no_device_discovery_attempted
- test_no_automated_blocking
- test_no_rf_protocol_claims
- test_no_ransomware_dwell_time_claims
- test_severity_uses_clinical_not_cvss
- test_no_model_internals_exposed       ← SHAP values must NOT appear in MVE text

### Study Analysis

- m5_result.yaml generated from study_responses_A/B.json
- group_b_composite_accuracy ≥ 0.55
- relative_improvement ≥ 0.40

### Final Outputs

- alignment_report.yaml: recommendation = SHIP_TO_USER_STUDY / ITERATE / BLOCKED
- m5_result.yaml: verdict = PASS / WARN / FAIL

---

## BUILD ORDER

Do not skip steps. Tests define what "correct" means.
Implementation serves the tests, not the other way around.

```
1.  data_models.py          — add SHAPContext dataclass
2.  fixtures/shap_stubs.yaml — stub SHAPContext for each of 5 alert types
3.  tests/acceptance_tests.py — add M5 (shap_narrative_alignment)
4.  tests/negative_tests.py — verify test_no_model_internals_exposed covers SHAP
5.  module2_train_models.py — train + serialize to models/
6.  module3_risk_scores.py  — risk-adaptive scoring (verify = risk_scorer.py)
7.  module4_online_explainer.py — SHAP → SHAPContext
8.  mve_generator.py        — add shap_context param, update Layer 1 + Layer 3
9.  module5_responses.py    — format output
10. module6_evaluation.py   — wire M1–M8 + study_analysis
11. run_tests.py            — entry point
12. Run → fix until all pass
13. Generate alignment_report.yaml + m5_result.yaml
```

---

## STACK

Python 3.11+

Required: pyyaml, numpy, scikit-learn, xgboost, shap, dataclasses, typing
Optional: anthropic (Mode A MVE generation), imbalanced-learn (SMOTE), streamlit
Removed from v1.0: re (no longer needed)

Style:
- snake_case everywhere
- Type hints on all function signatures
- Docstrings on all public functions
- Functions over classes unless state management is needed
- MUST have offline/rule-based fallback for mve_generator.py (Mode B)
- All M1–M8 tests must pass without ANTHROPIC_API_KEY

---

## SEVERITY MAPPING (use this, not CVSS)

CRITICAL: Life-sustaining (active infusion, ventilator, surgical) → immediately
HIGH:     Active clinical care (EHR, active PACS, pharmacy, monitors) → within 1h
MEDIUM:   Clinical-support not immediate (scheduling, archived imaging) → within 4h
LOW:      Administrative, minimal PHI (guest Wi-Fi, marketing) → within 24h

---

## ALIGNMENT REPORT FORMAT (unchanged from v1.0)

```yaml
test_results:
  - metric_id: M1
    result_value: 0.0
    target: 0.95
    minimum: 0.85
    pass_fail: PASS / WARN / FAIL

claims_supported:
  - claim_id: C1
    supported_by: [M2, M8, M5]
    verdict: SUPPORTED / PARTIAL / NOT_SUPPORTED

claims_not_tested:
  - claim_id: C4
    reason: "A/B user study Phase 2"
  - claim_id: C5
    reason: "Field deployment Phase 3"

recommendation: SHIP_TO_USER_STUDY / ITERATE / BLOCKED
```

Recommendation logic:
- SHIP_TO_USER_STUDY: all M1–M8 PASS, all negative tests PASS,
                      ≥4/5 claims SUPPORTED, m5 PASS
- ITERATE:            any test WARN OR 1–2 claims PARTIAL OR m5 WARN
- BLOCKED:            any test FAIL OR negative test violation OR M8 hard_fail

---

## KNOWN DESIGN GAPS (from mve_improvement_analysis.yaml)

These must be addressed in mve_generator.py before paper submission:

IMP-01: EHR access alerts — severity ≠ action-required. Always force-reauth.
IMP-02: unauthorized_ehr_access — add role_authorization_check to Layer 1.
IMP-03: IoMT clinical constraint — distinguish network isolation vs device power-off.
IMP-04: LOW-severity benign alerts — lead with benign hypothesis, not anomaly framing.
IMP-05: IoMT behavioral deviation — include numeric baseline in deviation_description.
IMP-06: Data exfiltration — specify exact block scope in Layer 3 immediate_action.
IMP-07 and IMP-08 are paper limitations, not code fixes.

<!-- code-review-graph MCP tools -->
## MCP Tools: code-review-graph

**IMPORTANT: This project has a knowledge graph. ALWAYS use the
code-review-graph MCP tools BEFORE using Grep/Glob/Read to explore
the codebase.** The graph is faster, cheaper (fewer tokens), and gives
you structural context (callers, dependents, test coverage) that file
scanning cannot.

### When to use graph tools FIRST

- **Exploring code**: `semantic_search_nodes` or `query_graph` instead of Grep
- **Understanding impact**: `get_impact_radius` instead of manually tracing imports
- **Code review**: `detect_changes` + `get_review_context` instead of reading entire files
- **Finding relationships**: `query_graph` with callers_of/callees_of/imports_of/tests_for
- **Architecture questions**: `get_architecture_overview` + `list_communities`

Fall back to Grep/Glob/Read **only** when the graph doesn't cover what you need.

### Key Tools

| Tool | Use when |
|------|----------|
| `detect_changes` | Reviewing code changes — gives risk-scored analysis |
| `get_review_context` | Need source snippets for review — token-efficient |
| `get_impact_radius` | Understanding blast radius of a change |
| `get_affected_flows` | Finding which execution paths are impacted |
| `query_graph` | Tracing callers, callees, imports, tests, dependencies |
| `semantic_search_nodes` | Finding functions/classes by name or keyword |
| `get_architecture_overview` | Understanding high-level codebase structure |
| `refactor_tool` | Planning renames, finding dead code |

### Workflow

1. The graph auto-updates on file changes (via hooks).
2. Use `detect_changes` for code review.
3. Use `get_affected_flows` to understand impact.
4. Use `query_graph` pattern="tests_for" to check coverage.
