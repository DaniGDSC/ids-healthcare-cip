# RQ2 — Compliance Mapping (literature ↔ MVE)

*Generated from `configs/rq2_compliance_manifest.yaml` on 2026-05-18T18:51:02.792095+00:00.*  
*Manifest last validated: 2026-05-19.*  
*Required evidence present: YES.*

| Requirement | Literature Term | MVE Implementation | Required Evidence | Pending |
|---|---|---|---|---|
| **REQ-FAITHFULNESS** | Faithfulness | Invariant 5: Layer 1 references SHAP top-3 features.  Verified end-to-end by tests/test_step12_mve_faithfulness.py. | `tests/test_step12_mve_faithfulness.py`<br>`results/rq2_mve_shap_alignment.json` | `analysis/compute_mve_shap_alignment.py` |
| **REQ-STABILITY** | Stability | SHAP stability score reported in results/rq2_shap_stability.json and gated by tests/test_step11_shap_stability.py. | `tests/test_step11_shap_stability.py`<br>`results/rq2_shap_stability.json` | `analysis/compute_shap_stability.py` |
| **REQ-COMPLETENESS** | Completeness | Three-layer structure enforced by MVEOutput dataclass (layer_1 WHY, layer_2 CLINICAL SEVERITY, layer_3 RECOMMENDED ACTION + DO NOT).  Coverage verified by tests/test_coverage_mve.py. | `src/mve_generator.py`<br>`src/data_models.py`<br>`tests/test_coverage_mve.py` | — |
| **REQ-BREVITY** | Brevity | Per-layer word budgets (Layer 1 ≤60, Layer 2 ≤50, Layer 3 ≤60) and total ≤150, enforced by the MVEOutput contract and audited by analysis/audit_word_budgets.py over the full surfaced-alert corpus. | `analysis/audit_word_budgets.py`<br>`results/rq2_word_budget_audit.json`<br>`tests/test_word_budgets.py`<br>`src/data_models.py` | — |
| **REQ-AUDIENCE_APPROPRIATENESS** | Audience appropriateness | Role views derived via derive_role_view.  Invariant 6 (Layer 2 severity identical across roles), Invariant 7 (DO NOT preserved on clinical alerts), Invariant 9 (shared anchor identical), plus a positive role-differentiation assertion in test_step13_cross_role_consistency.py. | `src/mve_generator.py`<br>`tests/test_step13_cross_role_consistency.py`<br>`tests/test_safe_failure.py`<br>`configs/role_action_authorization.yaml` | — |
| **REQ-PROVENANCE** | Provenance | MVEOutput carries mode_used, llm_provider, llm_model_version, llm_full_prompt, llm_full_response (populated only when Mode A runs).  The Module 5 batch sidecar (results/reports/mve_outputs.meta.json) fingerprints the corpus with SHA-256 of inputs.  When ANTHROPIC_API_KEY is configured, a per-call audit log accumulates at logs/llm_audit.jsonl (the historical scan in test_phi_not_in_llm_prompt.py reads it). | `src/data_models.py`<br>`src/mve_generator.py`<br>`tests/test_phi_not_in_llm_prompt.py`<br>`results/reports/mve_outputs.meta.json` | `logs/llm_audit.jsonl` |
| **REQ-FALLBACK** | Fallback / availability | Mode A returns None on any failure (missing key, missing package, API error, PHI red flag), and generate_mve falls back to the deterministic Mode B rule-based path.  Verified by test_mode_b_makes_no_external_calls (Phase 1) and the Mode B-only corpus produced by the current Module 5 batch. | `src/mve_generator.py`<br>`tests/test_safe_failure.py`<br>`tests/test_phi_not_in_llm_prompt.py` | — |
| **REQ-PHI_CONTROL** | Data minimization / HIPAA boundary | configs/llm_data_flow.yaml defines the strict allow-list (default-deny) and the forbidden field list with hard-fail semantics in _filter_for_llm.  Three layers of test coverage: static schema, live PHI honeypot, historical audit-log scan. | `configs/llm_data_flow.yaml`<br>`src/mve_generator.py`<br>`tests/test_phi_not_in_llm_prompt.py` | — |

---

## Detailed Descriptions

### REQ-FAITHFULNESS — Faithfulness

Explanations must reflect the actual decision logic of the underlying model — not post-hoc plausible-sounding text.  For MVE, Layer 1 must reference SHAP top-3 features (raw names or their human-readable mappings).

**MVE Implementation:** Invariant 5: Layer 1 references SHAP top-3 features.  Verified end-to-end by tests/test_step12_mve_faithfulness.py.

**Required Evidence:**
- ✅ `tests/test_step12_mve_faithfulness.py`
- ✅ `results/rq2_mve_shap_alignment.json`

**Pending Evidence (informational):**
- ⏳ `analysis/compute_mve_shap_alignment.py`

### REQ-STABILITY — Stability

Explanations should not change drastically under small input perturbations.  Operationalised as SHAP top-k overlap under a small additive perturbation.

**MVE Implementation:** SHAP stability score reported in results/rq2_shap_stability.json and gated by tests/test_step11_shap_stability.py.

**Required Evidence:**
- ✅ `tests/test_step11_shap_stability.py`
- ✅ `results/rq2_shap_stability.json`

**Pending Evidence (informational):**
- ⏳ `analysis/compute_shap_stability.py`

### REQ-COMPLETENESS — Completeness

Explanations should cover the why, the impact, and the recommended action — not just one of these.

**MVE Implementation:** Three-layer structure enforced by MVEOutput dataclass (layer_1 WHY, layer_2 CLINICAL SEVERITY, layer_3 RECOMMENDED ACTION + DO NOT).  Coverage verified by tests/test_coverage_mve.py.

**Required Evidence:**
- ✅ `src/mve_generator.py`
- ✅ `src/data_models.py`
- ✅ `tests/test_coverage_mve.py`

### REQ-BREVITY — Brevity

Explanations must be concise enough for time-pressured triage decisions (60-second nurse glance, 90-second IT first read).

**MVE Implementation:** Per-layer word budgets (Layer 1 ≤60, Layer 2 ≤50, Layer 3 ≤60) and total ≤150, enforced by the MVEOutput contract and audited by analysis/audit_word_budgets.py over the full surfaced-alert corpus.

**Required Evidence:**
- ✅ `analysis/audit_word_budgets.py`
- ✅ `results/rq2_word_budget_audit.json`
- ✅ `tests/test_word_budgets.py`
- ✅ `src/data_models.py`

### REQ-AUDIENCE_APPROPRIATENESS — Audience appropriateness

The same alert must be communicated differently to different stakeholders (IT generalist, biomed engineer, nurse manager) while preserving shared facts (anchor, severity).

**MVE Implementation:** Role views derived via derive_role_view.  Invariant 6 (Layer 2 severity identical across roles), Invariant 7 (DO NOT preserved on clinical alerts), Invariant 9 (shared anchor identical), plus a positive role-differentiation assertion in test_step13_cross_role_consistency.py.

**Required Evidence:**
- ✅ `src/mve_generator.py`
- ✅ `tests/test_step13_cross_role_consistency.py`
- ✅ `tests/test_safe_failure.py`
- ✅ `configs/role_action_authorization.yaml`

### REQ-PROVENANCE — Provenance

For LLM-generated explanations, the prompt, model version, and response must be auditable for reproducibility and accountability.

**MVE Implementation:** MVEOutput carries mode_used, llm_provider, llm_model_version, llm_full_prompt, llm_full_response (populated only when Mode A runs).  The Module 5 batch sidecar (results/reports/mve_outputs.meta.json) fingerprints the corpus with SHA-256 of inputs.  When ANTHROPIC_API_KEY is configured, a per-call audit log accumulates at logs/llm_audit.jsonl (the historical scan in test_phi_not_in_llm_prompt.py reads it).

**Required Evidence:**
- ✅ `src/data_models.py`
- ✅ `src/mve_generator.py`
- ✅ `tests/test_phi_not_in_llm_prompt.py`
- ✅ `results/reports/mve_outputs.meta.json`

**Pending Evidence (informational):**
- ⏳ `logs/llm_audit.jsonl`

### REQ-FALLBACK — Fallback / availability

The explanation system must degrade gracefully when external dependencies fail (e.g. LLM API unavailable, anthropic package absent, no API key configured).

**MVE Implementation:** Mode A returns None on any failure (missing key, missing package, API error, PHI red flag), and generate_mve falls back to the deterministic Mode B rule-based path.  Verified by test_mode_b_makes_no_external_calls (Phase 1) and the Mode B-only corpus produced by the current Module 5 batch.

**Required Evidence:**
- ✅ `src/mve_generator.py`
- ✅ `tests/test_safe_failure.py`
- ✅ `tests/test_phi_not_in_llm_prompt.py`

### REQ-PHI_CONTROL — Data minimization / HIPAA boundary

For deployments with patient context, no PHI may cross to external LLM providers; the data-flow contract must be auditable and tested with hard-fail CI gates.

**MVE Implementation:** configs/llm_data_flow.yaml defines the strict allow-list (default-deny) and the forbidden field list with hard-fail semantics in _filter_for_llm.  Three layers of test coverage: static schema, live PHI honeypot, historical audit-log scan.

**Required Evidence:**
- ✅ `configs/llm_data_flow.yaml`
- ✅ `src/mve_generator.py`
- ✅ `tests/test_phi_not_in_llm_prompt.py`

