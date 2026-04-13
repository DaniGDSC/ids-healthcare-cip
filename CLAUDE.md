# CLAUDE.md
# XAI-IDS-Healthcare Prototype
# Read this file completely before writing any code.

## WHAT THIS PROJECT IS

Research prototype for:
"Human-centric Explainable + Risk-Adaptive IDS for mid-sized
healthcare organizations (200-500 beds)."

Target user: IT security generalist (NOT SOC specialist).
Processes 10-50 alerts/day alongside EHR support and network admin.
Needs clinically contextualized explanations, not raw anomaly scores.

Full spec: research_spec.yaml (read before implementing any component)

---

## BUILD EXACTLY 3 COMPONENTS — NO MORE

### Component 1: MVE Generator
File: src/mve_generator.py
Function: generate_mve(raw_alert, device_context, baseline, user_context, shap_context=None) → MVEOutput

Produces 3-layer Minimum Viable Explanation:
- Layer 1 (WHY anomalous): baseline vs deviation, max 60 words
- Layer 2 (CLINICAL SEVERITY): patient-care impact, CRITICAL/HIGH/MEDIUM/LOW
- Layer 3 (RECOMMENDED ACTION): specific step + DO NOT constraint + escalation

Total output: ≤150 words. No jargon. No SHAP values. No CVSS.

### Component 2: Risk-Adaptive Scoring Engine
File: src/risk_scorer.py
Function: score_alert(anomaly_score, device_context, event_context) → ScoredAlert

Rules (non-negotiable):
- CRITICAL + unpatchable → threshold lowered ≥30%, risk_multiplier ≥1.5
- Maintenance window + known vendor IP → suppress (should_surface=False)
- LOW + patchable → default threshold, risk_multiplier=1.0

### Component 3: Alert Simulation Harness
File: src/harness.py
Function: run_simulation(dataset) → TestReport
Purpose: testing only, not production.

---

## FILE STRUCTURE

```
xai-ids-healthcare/
├── src/
│   ├── __init__.py
│   ├── data_models.py
│   ├── mve_generator.py
│   ├── risk_scorer.py
│   └── harness.py
├── tests/
│   ├── __init__.py
│   ├── acceptance_tests.py
│   ├── negative_tests.py
│   └── fixtures/
│       ├── sample_alerts.yaml
│       ├── device_inventory.yaml
│       └── behavioral_baselines.yaml
├── run_tests.py
├── alignment_report.yaml  ← generated after tests pass
├── research_spec.yaml
└── CLAUDE.md
```

---

## DO NOT BUILD

- Device discovery / network scanning
- Automated enforcement / blocking (recommend only, never execute)
- RF protocol detection (non-IP IoMT)
- Ransomware early-detection claims
- UI / frontend
- Database / persistence (use in-memory + YAML fixtures)
- Authentication / authorization
- ML model training (assume anomaly scores given as input)

If asked to build any of the above: refuse and cite this file.

---

## DONE CONDITION

Prototype is COMPLETE when run_tests.py produces:

AUTOMATED TESTS (all must pass at ≥minimum):
- test_mve_completeness          ≥85% (target 95%)
- test_clinical_relevance        ≥75% (target 90%)
- test_actionability             ≥70% (target 85%)
- test_clinical_constraint       ≥80% (target 90%)
- test_severity_label_accuracy   ≥70%, 0 CRITICAL↔LOW mismatches
- test_layer1_length_constraint  ≥90% (target 95%)
- test_risk_adaptive_threshold   100% (binary)
- test_false_positive_rate       ≥20% FP reduction (target 40%)

NEGATIVE TESTS (all must pass, 0 violations):
- test_no_device_discovery_attempted
- test_no_automated_blocking
- test_no_rf_protocol_claims
- test_no_ransomware_dwell_time_claims
- test_severity_uses_clinical_not_cvss
- test_no_model_internals_exposed

FINAL OUTPUT:
- alignment_report.yaml generated
- recommendation: SHIP_TO_USER_STUDY / ITERATE / BLOCKED

---

## ML PIPELINE ARCHITECTURE (pipeline/)

The prototype's 3 components (src/) are backed by a full ML pipeline
that trains models, scores risk, and generates explanations.

### Detection: Cascaded Track A → Track B

```
Raw Features ──→ Track A (XGB, RF, DT) ──→ P(attack) per model
                      │                          │
                      └──→ [Features ∥ P(attack)] ──→ Track B (DAE)
                                                         ↓
                                               reconstruction error
```

- Track A: 3 supervised tree models (SMOTE-balanced), trained first
- Track B: DAE trained on [25 raw features || 3 Track A OOF probabilities]
- Fusion: C_detect = max(Track_A, Track_B) — DAE elevates, never suppresses
- Files: pipeline/module2_detection/module2_train_models.py
         pipeline/module3_risk_scoring/module3_risk_scores.py

### Explanations: Feature Group Narratives

SHAP features are mapped to 7 clinically meaningful categories before
the clinician summary is generated. This absorbs within-category
feature swaps (e.g., DIntPkt↔Sport both map to "network_timing")
and produces stable narratives (84.6% top-1 category agreement).

When SHAP top feature is biometric, generate_mve() receives
shap_context={"top_category": "biometric", "top_feature_narrative": ...}
and enriches Layer 1 with biometric context.

- Files: pipeline/module4_explanations/module4_online_explainer.py
         pipeline/module4_explanations/module4_explanations.py
         src/mve_generator.py (shap_context parameter)

### ML Validation Results

- DAE separation: PASS (AUROC 0.9374, cascaded architecture)
- Risk monotonicity: PASS (M7 gap 0.401, zero overlap)
- SHAP stability: FAIL cross-model / PASS narrative-level (0.846)
- XAI faithfulness: PASS (perturbation 0.901, consistency 1.0, coverage 0.950)
- Full results: ml_validation.yaml, xai_faithfulness.yaml

---

## STACK

Python 3.11+
Required: pyyaml, dataclasses, re, typing (all stdlib or pip)
Optional: anthropic (for LLM-based MVE generation)
MUST have offline/mock fallback if no API key.

Style:
- snake_case
- Type hints on all function signatures
- Docstrings on all public functions
- Functions over classes unless state needed

---

## BUILD ORDER

1. data_models.py — all dataclasses first
2. fixtures/ — generate 50 labeled alerts (10 per type × 5 types)
3. tests/acceptance_tests.py — write tests BEFORE implementation
4. tests/negative_tests.py — write tests BEFORE implementation
5. src/risk_scorer.py — Component 2 (simpler, no LLM)
6. src/mve_generator.py — Component 1 (LLM or rule-based)
7. src/harness.py — Component 3 (wires everything together)
8. run_tests.py — entry point
9. Run tests → fix until all pass
10. Generate alignment_report.yaml

DO NOT skip step 3 and 4.
Tests define what "correct" means.
Implementation serves the tests, not the other way around.

---

## SYNTHETIC DATASET (generate in fixtures/)

50 alerts total, 10 per type:
- T1: Anomalous outbound from clinical device subnet
- T2: Unauthorized EHR/EMR access outside normal patterns
- T3: Lateral movement between network segments
- T4: Data exfiltration indicator from clinical system
- T5: IoMT device behavioral deviation

Label distribution:
- true_positive: ~50% (25/50)
- false_positive: ~30% (15/50)
- legitimate_rare: ~20% (10/50)

Severity distribution:
- CRITICAL: ~15% (7-8 alerts) — active infusion pumps, ventilators
- HIGH: ~35% (17-18 alerts) — EHR violations, active PACS, lateral movement
- MEDIUM: ~30% (15 alerts) — archived systems, IoMT firmware patterns
- LOW: ~20% (10 alerts) — admin/guest network anomalies

Reference examples: see mve_specification.yaml (5 concrete examples).
Vary IPs, timestamps, device types, locations, user names.

---

## MVE GENERATION APPROACH

Option A — LLM-based (preferred if API key available):
  Call Anthropic API with structured prompt.
  System prompt enforces 3-layer format, word limits, clinical framing.
  Parse response into MVEOutput dataclass.

Option B — Rule-based (fallback, always implement):
  Template strings per alert type.
  Fill device_context and baseline fields into templates.
  Deterministic, testable offline.

Implement BOTH. Use Option A if ANTHROPIC_API_KEY in environment,
else fall back to Option B automatically.

---

## SEVERITY MAPPING (use this, not CVSS)

CRITICAL: Life-sustaining (active infusion, ventilator, surgical) → respond immediately
HIGH: Active clinical care (EHR, active PACS, pharmacy, monitors) → within 1 hour
MEDIUM: Clinical-support not immediate (scheduling, archived imaging) → within 4 hours
LOW: Administrative, minimal PHI (guest Wi-Fi, marketing) → within 24 hours

---

## ALIGNMENT REPORT FORMAT

After all tests pass, write alignment_report.yaml:

```yaml
test_results:
  - metric_id: M1
    result_value: 0.0
    target: 0.95
    minimum: 0.85
    pass_fail: PASS/WARN/FAIL

claims_supported:
  - claim_id: C1
    supported_by: [M2, M8]
    verdict: SUPPORTED/PARTIAL/NOT_SUPPORTED

claims_not_tested:
  - claim_id: C4
    reason: "Requires A/B user study Phase 2"
  - claim_id: C5
    reason: "Requires field deployment Phase 3"

recommendation: SHIP_TO_USER_STUDY / ITERATE / BLOCKED
```

Recommendation logic:
- SHIP_TO_USER_STUDY: all tests PASS, ≥4/5 claims SUPPORTED
- ITERATE: any test WARN or 1-2 claims PARTIAL
- BLOCKED: any test FAIL or any negative test violation
