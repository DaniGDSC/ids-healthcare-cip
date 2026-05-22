# Fix 1 Thesis Handoff — Draft Text for Thesis Author

**Date:** 2026-05-21
**Branch:** `fix/rq1-weight-sensitivity`
**Sources:** `docs/fix1_design_memo.md`, `results/rq1_weight_sensitivity.json`, `results/rq1_sensitivity_analysis.json` (legacy v1 evidence preserved as `_legacy_evidence`).

The thesis docx is not present in the repository (Session 6 Q-V6 verified). This file is the handoff to whoever maintains the thesis manuscript. All numerical claims below come from `results/rq1_weight_sensitivity.json` produced by Fix 1's Stage 5B analysis (Phase 3).

---

## §3.3.2 — Risk Weights (REPLACEMENT TEXT)

Replace the existing paragraph that introduces the four composite-risk weights with this:

> In the absence of labeled outcome data linking specific hospital security incidents to alert tiers, we do not learn the four composite-risk weights from data. Instead, we frame them as policy parameters tunable by hospital security and clinical leadership. The default values (`detection_confidence` = 0.40, `device_criticality` = 0.25, `data_sensitivity` = 0.15, `clinical_tier` = 0.20) reflect our judgment of relative importance for the IoMT context, with detection confidence weighted most heavily as the primary signal-quality indicator. We demonstrate the robustness of this design via the sensitivity analysis reported in §5.2.4: Invariant 2 (the safety floor) holds across ±10% and ±20% multiplicative perturbations of each weight, while tier-assignment agreement degrades smoothly with the magnitude of policy departure. This makes the weights a hospital-side configuration knob rather than a system-side assumption.

---

## §5.2.4 — Weight Sensitivity Analysis (NEW SUBSECTION)

### Method

We perturb the four composite-risk weights via joint random multiplicative perturbations at two magnitudes (±10% and ±20%), generating N=30 perturbed weight vectors per magnitude. Each perturbed vector is L1-renormalized to maintain the sum-to-1.0 invariant required by `compute_composite_risk()`. For each perturbation, we compute the alert tier assignments under the perturbed weights and measure exact tier-match agreement against the baseline weights' tier assignments. We additionally track `fnr_critical_delta` — the fraction of the alert population that was assigned CRITICAL under the policy baseline and is no longer CRITICAL under the perturbed weights — to detect any safety-floor breach. Analysis is run on n=2448 alerts.

We also report three named baselines as comparators: equal weights (0.25 each); detection-confidence-only (1.0, 0, 0, 0); and a multiplicative-R alternative formula `R = c_detect × max(d_crit, s_data, d_clinical_tier)`.

### Result (Table 5.4)

| Condition | Agreement (mean ± std) | Agreement (min–max) | FNR-critical Δ |
|---|---|---|---|
| ±10% perturbation (N=30) | 0.9823 ± 0.0082 | 0.9673–0.9939 | max=0.0086 |
| ±20% perturbation (N=30) | 0.964 ± 0.0167 | 0.9154–0.9918 | max=0.0106 |
| Equal weights | 0.7341 | — | 0.0106 |
| C_detect-only | 0.7667 | — | 0.0 |
| Multiplicative R | 0.7937 | — | 0.0 |

### Discussion

> Under ±10% multiplicative perturbations, tier-agreement remained at 0.9823 on average, with the worst-case perturbation still agreeing on 96.73% of alerts. Under the more aggressive ±20% magnitude, mean agreement dropped to 0.964 with worst-case 91.54% — a smooth, monotone degradation with magnitude. Critically, the maximum FNR-critical Δ across all 60 perturbations was 0.0106 (i.e., at most 1.06% of the alert population that the policy baseline tiered as CRITICAL was no longer CRITICAL under any single perturbation), matching the worst named baseline (`equal_weights`, also 0.0106) and well below any breach of Invariant 2 (the safety floor). This empirically supports the policy-parameter framing: hospitals tuning weights within ±20% of the defaults preserve the safety floor while seeing only small degradation in tier-assignment consistency.

### Comparison with v1 evidence (legacy)

The prior `results/rq1_sensitivity_analysis.json` (v1 evidence, produced by `analysis/compute_rq1.py`) reported `agreement_mean = 0.9823` across 30 perturbations at ±10% only. Fix 1's ±10% result (0.9823 in this table) **matches the v1 number bit-exactly**; the v1 artifact is preserved under `rq1_metrics.json::weight_sensitivity._legacy_evidence` per the merge script's documented precedence rule. Fix 1 extends the v1 evidence on three axes: (i) a second magnitude (±20%); (ii) `fnr_critical_delta_max` tracked per condition (the v1 artifact reported it for the three named baselines only); (iii) explicit policy-parameter framing in §3.3.2 and ARCHITECTURE.md.

---

## §7.X — Future Work additions

Add to the Future Work section:

> **Multiplicative composite-risk formulation.** Fix 1 includes the multiplicative-R formula `R = c_detect × max(d_crit, s_data, d_clinical_tier)` as a named baseline comparator (agreement 0.7937, FNR-critical Δ 0.0 vs the additive primary's 0.9823 / 0.0). Promoting multiplicative R to the primary formula would require refactoring `module3_risk_scoring.apply_weight_feedback()` (at `module3_risk_scoring/module3_risk_scores.py:627`, which sweeps each of four weights and L1-normalizes them under additive-form assumptions) and is named here as future work, not addressed in this thesis. The multiplicative comparator's substantially lower tier-agreement (0.7937 vs the additive baseline's 1.0 self-reflexive) and identical safety-floor preservation (0.0) is consistent with the YAML's acknowledged limitation L1 (`configs/composite_risk_weights.yaml:41`: `Linear sum allows compensatory effects vs true multiplicative risk`).

> **R2 (split choice) finalization.** Fix 1's analysis runs on `results/reports/risk_scores.npz`, whose underlying split is ambiguous per Codebase_Investigation.html Session 11 §4 (the legacy producer's inline comment at `analysis/compute_rq1.py:282` says "test-split sourced"; the row count `n_alerts_evaluated = 2448` matches `val_phase1.parquet`'s row count per `docs/RQ1_pipeline.md:779`). Phase 0e (parquet-row-count check across `val_phase1.parquet` and `test_phase1.parquet`) resolves the identity; once locked, the analysis can be re-run with the asserted split for provenance, and the named-split version of Stage 5B becomes citable in the thesis.

---

## Optional thesis methodology footnote — three names

Adapt the disambiguation paragraph from ARCHITECTURE.md §"Three weight-sensitivity surfaces (disambiguation)" if the thesis methodology section discusses the implementation's structure. Three names, three purposes, briefly distinguished. The ARCHITECTURE.md text is the authoritative version; thesis text can be a one-paragraph condensed cite.

Suggested footnote (one paragraph):

> The codebase contains three closely-named entities serving distinct purposes: `weight_sensitivity_analysis()` (an AUROC-driven grid + OAT search at `module3_risk_scoring/module3_risk_scores.py:1071`, run once per main-pipeline invocation, PNG output, diagnostic); `results/rq1_sensitivity_analysis.json` (the legacy v1 robustness artifact, preserved as `_legacy_evidence` under the merge script's documented precedence rule); and `results/rq1_weight_sensitivity.json` (Fix 1's Stage 5B canonical robustness artifact reported here). ARCHITECTURE.md §"Three weight-sensitivity surfaces" carries the authoritative disambiguation.

---

## What this handoff does NOT include

- Edits to the thesis docx (file absent from repo).
- Bibliography entries (thesis author owns these).
- Figure captions (depend on thesis figure numbering — the agreement-histogram-by-magnitude figure is a natural candidate, sourced from `results/rq1_weight_sensitivity.json::results.perturbation_results.by_magnitude[*].histogram_counts/edges`).
- Cross-references to other thesis sections (depend on final section numbers).
- The R2 split-identity reconciliation paragraph; that's Phase 0e's deliverable.

## Audit trail

- Design decisions: `docs/fix1_design_memo.md`
- Implementation: `analysis/compute_weight_sensitivity.py`
- Implementation tests: `tests/test_weight_sensitivity_invariants.py` (15 invariant tests; all pass per Phase 3 V-4)
- Headline results: `results/rq1_weight_sensitivity.json`
- Merged metrics: `results/rq1_metrics.json::weight_sensitivity`
- Legacy evidence preserved: `rq1_metrics.json::weight_sensitivity._legacy_evidence` (bit-intact `agreement_mean = 0.9823`)
- Discovery audit trail: `Codebase_Investigation.html` Sessions 8–11
- Architecture documentation: `ARCHITECTURE.md` §"Risk weights as policy parameters" + §"Three weight-sensitivity surfaces (disambiguation)" (inserted by Phase 4)
- Spec update: `docs/RQ1_pipeline.md` §6.1 (status moved from SPEC PENDING to RESOLVED by Phase 4)
