# RQ2.a — Formal Explainability Requirements

**Status:** Mapping table fixed; faithfulness measurements moved from
"claimed" to "measured" (see RQ2.b).
**Last updated:** 2026-05-26

This table maps each formal explainability requirement from the
literature (Doshi-Velez & Kim 2017; Adadi & Berrada 2018; Tjoa & Guan
2020; HHS ONC 2024 AI/ML clinical decision support guidance) to the
specific MVE design choice that satisfies it, plus the test or measured
artifact that closes the loop.

---

## Requirement → MVE implementation → Evidence

| # | Requirement | MVE design choice | Evidence | Status |
|---|-------------|-------------------|----------|--------|
| 1 | **Faithfulness** — the explanation reflects what the model actually used | Layer 1 includes the top SHAP features (Mode B) or the LLM is conditioned on them (Mode A) | `tests/test_step12_mve_faithfulness.py` + `results/rq2_mve_shap_alignment.json` | ⚠ measured (gap: Mode B injects only top-1; ≥2 target not met) |
| 2 | **Stability** — small input perturbations don't reshuffle the explanation | Tree-based SHAP on a calibrated XGBoost pipeline (low variance under small noise) | `tests/test_step11_shap_stability.py` + `results/rq2_shap_stability.json` | ⚠ measured (mean 0.735 vs spec target 0.90 — pct stable 86% met) |
| 3 | **Completeness** — Why / Impact / Action all present | 3-layer structure: `layer_1` (Why anomalous), `layer_2` (Clinical severity), `layer_3` (Recommended action) | Branch coverage on `src/mve_generator.py::generate_mve` + 5 example explanations in `results/reports/example_explanations.json` | ✓ |
| 4 | **Brevity** — explanation fits the operator's reading budget | Hard word cap: 150 words total across layers; enforced post-generation in `MVEOutput.__post_init__` | `tests/test_coverage_mve.py` token-count assertions | ✓ |
| 5 | **Audience appropriateness** — different roles see different framings | Three role renderers in `module6_evaluation/module6_app.py` (`render_analyst`, `render_clinician`, `render_admin`) + a 4-th channel via `render_mve_layers` collapsible expander | RQ3 §1 capability table (cross-role consistency tests) | ✓ |
| 6 | **Provenance** — every produced explanation must be auditable | Mode A audit log includes prompt + raw response + provider key id; Mode B records `provider: "rule_based"` + template version | Audit log schema in `module5_responses/module5_pipeline.py::AuditLogger` + chain integrity test in `tests/test_step16_audit_integrity.py` | ✓ |
| 7 | **Fallback (safe failure)** — if Mode A (LLM) fails, Mode B (rule-based) takes over | `src/mve_generator.py::generate_mve(force_rule_based=False)` chain: OpenAI → Anthropic → rule-based; tripwire after first quota error per batch | Sentinel UI badge ("rule-based provider") + `tests/test_coverage_mve.py` fallback cases | ✓ |
| 8 | **MITRE grounding** — Layer 1 references the ATT&CK technique when applicable | `config/attack_to_mitre_mapping.yaml` (6 categories mapped to v14.1, no orphans) | `tools/rq2_audit_mitre_coverage.py` + `results/rq2_mitre_coverage.json` | ⚠ measured (config 100%; Layer 1 reference rate 0% — implementation gap) |
| 9 | **DO NOT constraint surfacing** — clinical safety boundary visible | Layer 3 `clinical_constraint` field; device-class fallbacks if LLM omits | `tests/test_safe_failure.py::test_inv7_critical_clinical_carries_do_not` | ✓ (34/34 critical clinical alerts carry DO NOT) |

---

## Architecture-side summary

Of the 9 requirements, **6 are fully done (✓)** and **3 are measured
with gaps (⚠)** — those gaps drive the RQ2.b paper section.

The 3 measured-with-gap items:

1. **Faithfulness alignment (#1)** — Mode B currently injects only the
   top-1 SHAP feature into Layer 1's "Primary signal" suffix. The spec
   target of ≥2 ≥ 95% requires `src.mve_generator` to extend that
   injection to top-2 / top-3. Gap is implementation, not architectural.

2. **Stability (#2)** — Mean stability score (0.735) sits below the
   aspirational 0.90 target because top-k SHAP rankings reshuffle when
   features have close magnitudes. The pct-stable target (>80%) IS met
   at 86.2%. The shortfall is a real model property — improving it
   needs feature-importance regularization at training time.

3. **MITRE grounding (#8)** — Config is 100% mapped (6/6 categories,
   v14.1 pinned). But Layer 1 narratives do not currently reference
   ATT&CK IDs/names — the MITRE injection logic isn't wired into
   `src.mve_generator`. Gap is implementation.

---

## Crosscut: which requirements depend on user study completion?

| Requirement | User-study dependent? |
|-------------|----------------------:|
| 1. Faithfulness | No (measured automatically) |
| 2. Stability | No (measured automatically) |
| 3. Completeness | No (structural) |
| 4. Brevity | No (structural) |
| 5. Audience appropriateness | Partially (validated via RQ2.c per-role MWU) |
| 6. Provenance | No (audit log integrity) |
| 7. Fallback | No (test coverage) |
| 8. MITRE grounding | No (config + reference rate audit) |
| 9. DO NOT surfacing | Partially (RQ2.d failure-mode signal) |

Most requirements close on automated evidence; user-study evidence
augments rather than replaces.

---

## References

- Doshi-Velez & Kim, "Towards a Rigorous Science of Interpretable Machine Learning" (2017)
- Adadi & Berrada, "Peeking Inside the Black-Box: A Survey on Explainable AI (XAI)" (2018)
- Tjoa & Guan, "A Survey on Explainable AI: Toward Medical XAI" (2020)
- HHS ONC, "HTI-1 — Decision Support Interventions Transparency Criterion" (2024)

---

## Linked artifacts

- `results/rq2_shap_stability.json` — stability per-sample + summary
- `results/rq2_mve_shap_alignment.json` — Mode A vs Mode B alignment
- `results/rq2_mitre_coverage.json` — coverage + reference rate
- `analysis/outputs/rq2c_per_role.json` — per-role user-study results
- `analysis/outputs/rq2d_failure_modes.json` — failure-mode catalog
- `config/attack_to_mitre_mapping.yaml` — MITRE mapping (v14.1)
- `results/figures/rq2_shap_stability_hist.png` — stability distribution
- `results/figures/rq2_mve_alignment.png` — alignment % bars
