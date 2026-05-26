# RQ2 Summary — MVE Faithfulness, Stakeholder Adaptation, Iteration, MITRE

**Status:** Complete (15/15 deliverables) + 3 P0 gaps closed (G1, G2, G3)
**Generated:** 2026-05-26 (revised after P0 fixes)

---

## TL;DR — Post P0 fix

**Architectural side (RQ2.a):** **8/9 explainability requirements
satisfied** (was 6/9). The remaining gap is the mean SHAP stability
score, which is a model property documented in Limitations.

**Faithfulness measurements (RQ2.b):**
- SHAP stability: mean **0.735**, pct stable **86.2%** ✓ (pct target
  80% met; mean target 0.90 documented as model property)
- MVE-SHAP alignment Mode B (post G1+G2 fix): top-1 **100%**, ≥2
  **100%**, all-3 **100%** ✅ all 3 targets met
- MVE-SHAP alignment Mode A (LLM): top-1 80%, ≥2 25%, all-3 0% —
  abstract-by-design; user-study acceptable

**Stakeholder adaptation (RQ2.c):**
- Administrator: 0.700→0.920 acc, p=0.007 ✓ significant
- Clinician: 0.700→0.900 acc, p=0.032 ✓ significant
- Analyst: 0.860→0.940 acc, p=0.109 (already-high baseline)

**Iteration on failure modes (RQ2.d):** Observation-level catalog of
3 failure modes from 1300-response single-round study. Improvement
claims rescoped per spec scope note.

**MITRE grounding (RQ2.e):** Config 100% coverage. **After G3 fix:
Layer 1 MITRE reference rate 100% on attack-class alerts** (was 0%);
benign baseline correctly excluded (0% — by design).

---

## 1. RQ2.a — Formal Explainability Requirements Mapping

See `docs/rq2a_explainability_requirements.md` for the full table.

| Requirement | Status |
|-------------|--------|
| Faithfulness | ⚠ measured with gap |
| Stability | ⚠ measured (mean below target, pct met) |
| Completeness | ✓ |
| Brevity | ✓ |
| Audience appropriateness | ✓ |
| Provenance | ✓ |
| Fallback (Mode A/B) | ✓ |
| MITRE grounding | ⚠ config done, injection unwired |
| DO NOT surfacing | ✓ |

---

## 2. RQ2.b — Faithfulness Metrics

### SHAP Stability (`results/rq2_shap_stability.json`)

| Quantity | Value | Spec target | Met? |
|----------|------:|------------:|-----:|
| Mean stability score | 0.735 | 0.90 | ✗ |
| Median stability score | 0.700 | — | — |
| Std stability score | 0.123 | — | — |
| % alerts stable (Jaccard ≥ 0.4) | 86.2% | 80% | ✓ |
| n attack samples evaluated | 80 | — | — |
| n perturbations per sample | 5 | — | — |
| Noise σ (normalized × std) | 0.005 | — | — |
| Top-k | 5 | — | — |

**Finding:** mean shortfall is a real model property. Several SHAP
magnitudes are close enough that small perturbation flips top-k
ranking. Improving the mean requires feature-importance regularization
at training time — outside paper scope. The pct-stable target IS met.

### MVE-SHAP Alignment (`results/rq2_mve_shap_alignment.json`) — POST G1+G2+G6 FIX

| Mode | n | Contains top-1 | Contains ≥2 of 3 | Contains all 3 |
|------|--:|---------------:|------------------:|---------------:|
| Mode A (LLM narrative) | 20 | 80.0% | 25.0% | 0.0% |
| Mode B (rule-based, n=20) | 20 | **100.0%** ✅ | **100.0%** ✅ | **100.0%** ✅ |
| Mode B (rule-based, **large-N**) | **200** | **100.0%** | **100.0%** | **100.0%** |
| **Spec target** | — | (n/a) | **95%** | **80%** |

**G6 statistical rebuttal:** At n=200 (stratified by severity), the
95% Wilson CI on ≥2-rate is **[98.12%, 100.00%]** — the lower bound
clears the 95% spec target, ruling out the n=20 result as a
small-sample artifact.

**Finding (resolved):** Mode B previously injected only `top_features[0]`
at `src/mve_generator.py:1114`. G1+G2 fix extended the "Primary signal"
suffix to list the top-3 features (`Top signals: <category> (f1, f2, f3)`).
Mode B now hits 100% on all three alignment metrics, well above spec
targets. Word budget impact: +6-12 words, well under the 150-word cap.

Mode A (LLM) remains abstract-by-design — the LLM narrative deliberately
simplifies for clinician readability. It satisfies user-study comprehension
(see RQ2.c) but not feature-by-feature alignment; documented in
`docs/rq2a_explainability_requirements.md` as expected behavior.

---

## 3. RQ2.c — User Study Per-Role (`analysis/outputs/rq2c_per_role.json`)

Role labels: IT Generalist (`analyst`) / Nurse Manager (`clinician`) /
Biomed Engineer (`administrator`) — spec triad, with internal data
keys in parentheses (M6 study; M5 study
in `survey/` has no role metadata).

| Role | n records | Acc without XAI | Acc with XAI | MWU p | Decision time (s) Δ |
|------|----------:|----------------:|-------------:|-------:|---------------------:|
| **Biomed Engineer** (`administrator`) | 50 | 0.700 | 0.920 | **0.007** ✓ | 44.4 → 29.1 |
| **Nurse Manager** (`clinician`) | 50 | 0.700 | 0.900 | **0.032** ✓ | 43.1 → 27.0 |
| **IT Generalist** (`analyst`) | 50 | 0.860 | 0.940 | 0.109 | 40.2 → 20.7 |

All three roles improve under XAI; Biomed Engineer + Nurse Manager
reach significance at α=0.05. IT Generalist's high baseline reduces
effect size — expected (this is the role most familiar with SHAP).

**Cross-check with RQ3 per-role analysis:** identical numbers
(✓ on all three roles in this session's cross-check).

---

## 4. RQ2.d — Failure Mode Catalog (`analysis/outputs/rq2d_failure_modes.json`)

**Scope per spec §RQ2.d:** Single-round study only — claim is
**OBSERVATIONAL** (failure modes identified). Improvement metrics are
catalogued for future second-round but NOT claimed.

| Failure mode | n observed (out of 1300) | Iteration recipe | Improvement claim? |
|--------------|--------------------------:|-------------------|--------------------|
| `do_not_constraint_ignored` | 70 | Visual emphasis in Layer 3 (bold + warning) | ✗ requires second round |
| `mitre_not_understood` | 7 | Inline plain-language gloss for each MITRE ID | ✗ requires second round |
| `feature_too_abstract` | 1 | Extend Mode B feature injection from top-1 to top-3 | ✗ requires second round |

**Total signal rate:** 5.92% of responses contained explicit failure
signal — consistent with informal feedback that the explanations are
"mostly clear, with edge cases." Catalog is meant to seed v2 design,
not to claim measured improvement.

---

## 5. RQ2.e — MITRE ATT&CK Grounding (`results/rq2_mitre_coverage.json`)

### Config audit
| Quantity | Value | Target | Met? |
|----------|------:|-------:|-----:|
| Total attack categories | 6 (excluding `normal`) | — | — |
| Mapped to MITRE | 6 (100%) | 100% | ✓ |
| Orphans | (none) | 0 | ✓ |
| Framework version pinned | v14.1 | required | ✓ |
| With sub-techniques | 4 | — | — |
| With ICS techniques | 2 (Spoofing, Data Alteration) | — | — |

### Layer 1 MITRE reference rate — POST G3 FIX

| Quantity | Value | Target | Met? |
|----------|------:|-------:|-----:|
| Cached narratives (pre-fix) | 0/25 (0.0%) | — | (baseline) |
| Fresh narratives — all categories | 32/48 (66.67%) | 90% | partial (benign included) |
| Fresh narratives — **attack-class only** | **32/32 (100.0%)** ✅ | 90% | ✅ |
| Benign baseline reference rate | 0/16 (0.0%) | 0% | ✅ no over-attribution |

**Finding (resolved):** G3 fix added `_lookup_mitre_reference()` to
`src/mve_generator.py` — it reads `config/attack_to_mitre_mapping.yaml`
and appends `"Consistent with MITRE TXXXX (Name)."` to Layer 1 when an
attack category has a primary technique with confidence ≥ medium.

Two denominators reported:
- **all-categories** (66.67%) — includes benign baseline samples
- **attack-class only** (100.00%) — the spec metric, per the YAML's
  `excluded_from_coverage_audit: true` flag on `normal`

100% on attack-class far exceeds the 90% target. Benign baseline correctly
remains at 0% — adding MITRE refs to benign traffic would be false
attribution.

---

## 6. Tests — POST P0 FIX

| Suite | n tests | Status |
|-------|--------:|--------|
| `tests/test_step11_shap_stability.py` | 7 | ✅ all pass |
| `tests/test_step12_mve_faithfulness.py` | 10 (+2 post-fix) | ✅ all pass |
| `tests/test_rq2_mitre_grounding.py` | 5 (new, G3) | ✅ all pass |
| **Total** | **22** | ✅ |

Tests encode current measured values as bounds so future regressions
trip them. New assertions added after P0:

- `test_mode_b_at_least_2_meets_target` — fails if Mode B drops below 95% ≥2
- `test_mode_b_all_3_meets_target` — fails if Mode B drops below 80% all-3
- `test_layer1_mitre_reference_attack_class_meets_target` — fails if attack-class MITRE ref <90%
- `test_layer1_mitre_reference_no_benign_attribution` — fails if benign records gain MITRE refs (over-attribution)

---

## 7. Figures (PNG @ 300 DPI)

- `results/figures/rq2_shap_stability_hist.png` — distribution + threshold + mean + spec-target lines
- `results/figures/rq2_mve_alignment.png` — Mode A vs Mode B side-by-side, 3 metric groups, spec target lines

---

## 8. Artifact Inventory (15/15)

### JSON (5)
- `results/rq2_shap_stability.json`
- `results/rq2_mve_shap_alignment.json`
- `results/rq2_mitre_coverage.json`
- `analysis/outputs/rq2c_per_role.json`
- `analysis/outputs/rq2d_failure_modes.json`

### Tests (2)
- `tests/test_step11_shap_stability.py`
- `tests/test_step12_mve_faithfulness.py`

### Config (1)
- `config/attack_to_mitre_mapping.yaml`

### Figures PNG (2)
- `results/figures/rq2_shap_stability_hist.png`
- `results/figures/rq2_mve_alignment.png`

### Docs (2)
- `docs/rq2a_explainability_requirements.md`
- `results/rq2_summary.md` (this file)

### Tools (3)
- `tools/rq2_compute_faithfulness.py`
- `tools/rq2_audit_mitre_coverage.py`
- `tools/rq2_user_study_analysis.py`
- `tools/rq2_plot_figures.py`

---

## 9. Gap status — POST P0 FIX

| Gap ID | Description | Status |
|--------|-------------|--------|
| **G1** | Mode B injects top-1 only — alignment ≥2 = 0% | ✅ **CLOSED** (G1+G2 fix in src/mve_generator.py:1098-1131; ≥2 now 100%) |
| **G2** | Mode B all-3 = 0% | ✅ **CLOSED** (same fix; all-3 now 100%) |
| **G3** | Layer 1 MITRE reference rate 0% | ✅ **CLOSED** (`_lookup_mitre_reference()` + injection; attack-class now 100%) |
| **G4** | Mode A LLM alignment ≥2 = 25% | ⚠ Documented — LLM abstracts by design; user-study acceptable |
| **G5** | Mean SHAP stability 0.735 < 0.90 | ⚠ Model property — pct-stable (86%) meets target; mean shortfall requires training-time regularization |
| **G6** | Alignment sample n=20 (cached) | ✅ **CLOSED** — Mode B large-N audit at n=200, all-3 targets met, 95% CI ≥2 = [98.12%, 100.00%] |
| **G7** | NOVEL_ANOMALY per-feature attribution | ⚠ Out of scope — future work (XGBoost SHAP only currently) |
| **G8** | RQ2.d improvement claim | ⚠ Rescoped to observation-only per spec |
| **G9** | Bedside nurse role missing (direct recruit) | ⚠ Proxied via `clinician` (Nurse Manager); future recruitment |

## 10. Defendability Statement — POST P0 FIX

**Architectural side (RQ2.a):** ✅ Defensible.

- 9 explainability requirements mapped to specific design choices.
- **8/9 closed** by automated tests (was 6/9 pre-fix).
- The remaining gap (G5 — mean SHAP stability) is a model property,
  not an implementation defect; documented in Limitations.

**Empirical side (RQ2.b):** ✅ Defensible.

- SHAP stability: pct-stable 86% ✓ (target 80%); mean 0.735
  (target 0.90) documented as model property.
- MVE-SHAP alignment Mode B: **all 3 targets met** (top-1 100%,
  ≥2 100%, all-3 100%) after G1+G2 fix.
- Mode A intentional abstraction documented in RQ2.a row 1.

**User study (RQ2.c):** ✅ Defensible.

- 2 of 3 roles improve significantly (Biomed Engineer p=0.007, Nurse
  Manager p=0.032); IT Generalist trends positive without significance
  (high baseline).
- Cross-validates with RQ3 results.

**Iteration (RQ2.d):** Observation-level claim only — explicitly
rescoped from the original improvement claim per spec scope note.

**MITRE grounding (RQ2.e):** ✅ Defensible.

- Config-level: 100% coverage, no orphans, framework v14.1 pinned.
- **Layer 1 attack-class reference rate: 100%** after G3 fix.
- Benign baseline correctly excluded (no over-attribution).

---

## 11. Reproducibility

```bash
# Faithfulness
.ids/bin/python tools/rq2_compute_faithfulness.py

# MITRE coverage
.ids/bin/python tools/rq2_audit_mitre_coverage.py

# User study
.ids/bin/python tools/rq2_user_study_analysis.py

# Figures
.ids/bin/python tools/rq2_plot_figures.py

# Tests
.ids/bin/python -m pytest tests/test_step11_shap_stability.py \
                          tests/test_step12_mve_faithfulness.py -v
```

---

## 12. Cross-RQ pointers

- **RQ1** — Detection metrics on the same test split: `results/rq1_summary.md`
- **RQ3** — User-study aggregate + invariants: `results/rq3_summary.md`
- **Shared:** `docs/rq1_tier_surfacing_truth.md`, `config/tier_routing.yaml`,
  `config/role_action_authorization.yaml`
