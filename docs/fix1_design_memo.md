# Fix 1 Design Memo — Risk Weight Sensitivity Analysis

**Date:** 2026-05-21
**Branch:** `fix/rq1-weight-sensitivity` (HEAD at `10e26156dca76174ff1d155cf23211bb730abf3f`)
**Sources:** `Codebase_Investigation.html` Sessions 8–11
**Decision authority:** _USER NAME_
**Decision date:** _TO BE FILLED IN PHASE 1 STEP 7_

---

## Executive Summary

- **Design.** Fix 1 supersedes the legacy `results/rq1_sensitivity_analysis.json` (v1 evidence) with a new canonical `results/rq1_weight_sensitivity.json`. The new artifact extends the legacy's protocol shape (joint random N=30, exact tier match, three named baselines including multiplicative comparator) by adding a second magnitude — ±20% alongside the legacy's ±10%.
- **Methodology (D1–D4).** D1 = both ±10% and ±20%; D2 = joint random sampling with multiplicative-then-L1-renormalize; D3 = exact tier match (`np.mean(tier == tier_base)` per legacy); D4 = multiplicative R kept as named baseline comparator (formula `c_detect * max(d_crit, s_data, d_clinical_tier)`). The four weights are perturbed jointly for 30 perturbations per magnitude (60 perturbation conditions + 3 baselines in the output JSON).
- **Result vehicle (R1–R3).** R1 = keep all three "sensitivity" names with ARCHITECTURE.md + thesis disambiguation (zero file/code renames). R2 = `[USER REVIEW NEEDED]` pending Phase 0e parquet-row-count confirmation. R3 = supersede via the merge script's documented `_legacy_evidence` precedence rule at `analysis/merge_rq1_metrics.py:15-18`, with zero-LOC aggregator change.
- **Defense framing.** Weights are hospital-tunable policy parameters per `configs/composite_risk_weights.yaml:7-8`; the sensitivity analysis at ±10% (legacy continuity, citable `agreement_mean = 0.9823`) and ±20% (YAML §11 reference, stress test) demonstrates safety-floor robustness. This reframes the senior review §3.1 `unjustified weights` critique as transparency: weights are policy parameters reviewed annually, with Stage 5B as the robustness evidence.
- **Open item.** R2 (split choice) pending Phase 0e parquet-row-count result; all other six decisions are locked.

---

## Verified Findings (Inputs)

### The three-surface landscape

Three closely-named things touch the composite-risk weight space; each does something different. All three are pre-existing per Sessions 9–11.

| Surface | Location | What it does | Metric | Split | Output | Status |
|---|---|---|---|---|---|---|
| 1. Legacy JSON | `results/rq1_sensitivity_analysis.json` (Session 9 Q-V1; Session 11 Q-V5) | 30 perturbations + 3 named baselines under `"multiplicative ±10% then L1 renormalize to sum=1.0"` | Exact tier match: `np.mean(tier == tier_base)` on integer tier vectors from `_assign_tier` (Session 11 Q-V5.3) | Inline comment at `analysis/compute_rq1.py:282` says "test-split sourced"; row-count `n_alerts_evaluated = 2448` matches val_phase1 (contradiction surfaced Session 11 §4) | JSON written by `analysis/compute_rq1.py`; merged by `analysis/merge_rq1_metrics.py` | v1 evidence per merge script status string |
| 2. In-pipeline `weight_sensitivity_analysis()` | `module3_risk_scoring/module3_risk_scores.py:1071` (Session 10 Q-V3) | Docstring L1078: `"Grid search over weight space; evaluate AUROC of R as binary classifier."` — 5-point grid `np.array([0.10, 0.20, 0.30, 0.40, 0.50])` plus per-component OAT sweep `np.arange(0.05, 0.65, 0.05)` (12 values) | AUROC vs `y_true` (continuous classifier metric) | `y_test` argument at L1525 — test split | Returns dict `{grid_size, best_weights, best_auroc, default_weights, top_10, per_component_sensitivity}`; saves PNG at `CHARTS_DIR / "weight_sensitivity.png"` — no JSON | Live; invoked once per main-pipeline run at L1525 |
| 3. Calibration utility `apply_weight_feedback()` | `module3_risk_scoring/module3_risk_scores.py:627` (Session 10 Q-V2) | Docstring L637: `"Adjust Module 3 weights using AUROC as the optimization target."` — variance-based redistribution + per-weight ±`max_delta` hill-climb | AUROC hill-climb against caller-supplied `y_true` | Caller-supplied | Pure return: `dict` with sum-to-1.0 normalized weights; no module-level mutation, no file write | Live; invoked by `feedback_loop_demo.py:267` |

### The composite-risk implementation

- **Function:** `compute_composite_risk()` at `module3_risk_scoring/module3_risk_scores.py:552` (Session 8 Q-W2).
- **Signature (verbatim L552-558):**
  ```python
  def compute_composite_risk(
      c_detect: np.ndarray,
      d_crit: np.ndarray,
      s_data: np.ndarray,
      d_clinical_tier: np.ndarray,
      weights: dict | None = None,
  ) -> np.ndarray:
  ```
- **Weight injection (verbatim L560-565, Session 8 Q-W2.2):**
  ```python
  w = weights or WEIGHTS
  R = (w["w1"] * c_detect +
       w["w2"] * d_crit +
       w["w3"] * s_data +
       w["w4"] * d_clinical_tier)
  return np.clip(R, 0.0, 1.0)
  ```
- **YAML config:** `configs/composite_risk_weights.yaml` (Session 8 Q-W3). Weights: `detection_confidence: 0.40, device_criticality: 0.25, data_sensitivity: 0.15, clinical_tier: 0.20`. Tier boundaries: `critical_min: 0.80, high_min: 0.60, medium_min: 0.40`.
- **Sum-to-1.0 invariant (verbatim `module3_risk_scoring/module3_risk_scores.py:86-90`, Session 8 Q-W3):**
  ```python
  total = round(sum(weights.values()), 6)
  if abs(total - 1.0) > 1e-6:
      raise ValueError(
          f"Composite-risk weights must sum to 1.0, got {total}: {weights}"
      )
  ```
- **YAML policy framing (verbatim `configs/composite_risk_weights.yaml:7-8`):**
  `These are POLICY parameters — set by hospital security/clinical leadership, NOT learned from data. Reviewed annually.`

### The aggregator path

- **Merge script:** `analysis/merge_rq1_metrics.py`, 98 lines (Session 10 Q-V4). Self-identifies at L1-2 as `"Merge supporting analyses into rq1_metrics.json (RQ1_pipeline.md §6.4 / Stage 5E)."`
- **Module constants (verbatim L29-30):**
  ```python
  WS_NEW = REPO_ROOT / "results/rq1_weight_sensitivity.json"
  WS_LEGACY = REPO_ROOT / "results/rq1_sensitivity_analysis.json"
  ```
- **Precedence rule (verbatim docstring L15-18, Session 10 Q-V4.2):**
  `When both ``rq1_weight_sensitivity.json`` (new) and ``rq1_sensitivity_analysis.json`` (legacy) exist, the new one wins; the legacy block is preserved under ``weight_sensitivity._legacy_evidence`` for traceability.`
- **Zero-LOC aggregator contract (Session 10 Q-V4.3):** the merge script performs `metrics["weight_sensitivity"] = ws_new` (L55) — whole-dict assignment with no sub-key inspection. The new JSON's schema is not constrained by the merge script; Phase 1 has schema freedom for the new file's body.
- **Legacy handling (verbatim L62-74, Session 10 Q-V4.4):**
  ```python
  elif ws_legacy is not None:
      metrics["weight_sensitivity"] = {
          "_status": (
              "v1 evidence from legacy analysis/compute_rq1.py — "
              "perturbation protocol pending finalisation per "
              "RQ1_pipeline.md §6.1"
          ),
          "_source": WS_LEGACY.name,
          "_merged_at": now,
          **ws_legacy,
      }
  ```

### The legacy artifact

- **File:** `results/rq1_sensitivity_analysis.json` — 2,357 bytes, mtime `2026-05-16 17:07` (Session 9 Q-V1.1). Provenance currency verified Session 9 Q-V1.4: YAML `sha256: a8c5bb9d...` byte-matches current `configs/composite_risk_weights.yaml`; JSON `git_commit: 52521ee...` is an ancestor of HEAD `10e26156`.
- **Perturbation method (verbatim from `results.perturbation_method`):** `"multiplicative ±10% then L1 renormalize to sum=1.0"`
- **N perturbations:** `30` (Session 9 Q-V1.2; `results.perturbation_results.n_perturbations`).
- **Agreement distribution (verbatim `results.perturbation_results`):** `agreement_mean: 0.9823`, `agreement_std: 0.0082`, `agreement_min: 0.9669`, `agreement_max: 0.9947`, p25/p50/p75: `0.9755 / 0.9828 / 0.9905`. Histogram: all 30 perturbations land in the rightmost (0.9–1.0) bin (`histogram_counts: [0,0,0,0,0,0,0,0,0,30]`).
- **Baselines (verbatim `results.baselines`):**
  - `equal_weights: {"agreement": 0.7345, "fnr_critical_delta": 0.0106}`
  - `c_detect_only: {"agreement": 0.7659, "fnr_critical_delta": 0.0}`
  - `multiplicative: {"agreement": 0.7929, "fnr_critical_delta": 0.0}`
- **Multiplicative baseline formula (Session 11 §4; verbatim `analysis/compute_rq1.py:357-359`):** `R = comp["c_detect"] * np.maximum.reduce([comp["d_crit"], comp["s_data"], comp["d_clinical_tier"]])`
- **Agreement metric definition (Session 11 Q-V5.1, verbatim from `analysis/compute_rq1.py`):**
  - Perturbation-loop site L349: `agreements.append(float(np.mean(tier == tier_base)))`
  - Baseline site L363: `agreement = float(np.mean(tier == tier_base))`
  - Both operate on integer tier vectors from `_assign_tier(R, boundaries)` with `boundaries = (0.80, 0.60, 0.40)` (L280).
  - Histogram x-axis label L383 (verbatim): `"Tier agreement with baseline weights"`
  - Classification: outcome (a) — exact tier match (Session 11 Q-V5.3). Cohen's κ, continuous-R proxy, surfacing-match all ruled out.

### Spec status

- **`RQ1_pipeline.md:806` (verbatim section header, Session 8 Q-W6):** `### 6.1 Stage 5B — Weight sensitivity (SPEC PENDING, ``a_high`` half RESOLVED in §5b)`
- **`RQ1_pipeline.md:6` (status line):** `Phase 4 sub-stage 5B (weight sensitivity) and Phase 4 sub-stage 5D (MedSec sibling metrics) have remaining design questions flagged inline — implement the rest first.`
- **Four open questions in spec (verbatim `RQ1_pipeline.md:812-815`, mapping to D1–D4):**
  - L812 (→ D2): `Perturbation protocol: one-at-a-time vs joint sampling vs Dirichlet?`
  - L813 (→ D1 sub-question): `Number of perturbations per condition?`
  - L814 (→ D3): `Agreement metric: exact tier match, Cohen's κ, both?`
  - L815 (→ D4): `Multiplicative R: implement as a separate condition or skip?`
- **D1 magnitude is in the YAML, not the spec.** `configs/composite_risk_weights.yaml:10-11` (verbatim): `Sensitivity analysis under ±20% perturbation is reported in paper / Section 11; the weights here are the calibration baseline.` The spec itself does not name a magnitude.
- **Spec implementation pointer (verbatim `RQ1_pipeline.md:819`):** `When ready to implement, create ``analysis/compute_weight_sensitivity.py`` writing to ``results/rq1_weight_sensitivity.json``. The merge script (Stage 5E) folds it in.`
- **Stage 5E merge contract (verbatim `RQ1_pipeline.md:880-883`, §6.4):**
  - `**Inputs:** ``results/rq1_metrics.json`` (Phase 3 output), ``results/rq1_weight_sensitivity.json`` (5B), ``results/rq1_cascade_ablation.json`` (5C).`
  - `**Output:** updates ``results/rq1_metrics.json`` in place.`
  - `**Idempotent:** safe to re-run.`

---

## Decisions

### D1 — Perturbation magnitude

**Pick:** Both ±10% and ±20%

**Constraint source:** Session 8 Q-W6.1 (spec leaves magnitude open — L812-815 do not name a magnitude); Session 8 Q-W3 (YAML comment cites ±20% for paper §11); Session 9 Q-V1 (legacy artifact uses ±10%).

**Verified evidence:**
- Spec (verbatim `RQ1_pipeline.md:813`): `Number of perturbations per condition?` — the spec's §6.1 list does not specify a magnitude alongside the count question.
- YAML hint (verbatim `configs/composite_risk_weights.yaml:10-11`): `Sensitivity analysis under ±20% perturbation is reported in paper / Section 11; the weights here are the calibration baseline.`
- Legacy artifact (verbatim `results/rq1_sensitivity_analysis.json::results.perturbation_method`): `"multiplicative ±10% then L1 renormalize to sum=1.0"`
- Legacy saturation (Session 9 Q-V1.2 + Session 11 Q-V5.5): all 30 perturbations land in [0.9, 1.0] under ±10%; per Session 11, this saturation is consistent with exact-tier-match on small multiplicative perturbations and does not by itself distinguish "policy invariant" from "metric coarse."

**Options available:**

| Option | What it means | Trade-off considerations |
|---|---|---|
| ±10% multiplicative + L1 renormalize | Matches legacy artifact exactly; reuses N=30 distribution | Legacy results citable; agreement saturated (all 30 in [0.9,1.0]); doesn't stress the formula |
| ±20% multiplicative + L1 renormalize | Matches YAML's paper-§11 reference; more aggressive | Legacy numbers don't apply; new run required; larger spread expected |
| Both ±10% and ±20% | Report two magnitudes side-by-side | More text in thesis; clearer robustness story; two N=30 runs needed |
| Absolute ±0.05 / ±0.10 (additive) | Different framing (relative vs absolute) | Reframes the analysis; departs from both legacy and YAML conventions |
| OAT sweep (e.g., 0.05–0.60 per weight) | Matches in-pipeline `weight_sensitivity_analysis()` at L1071 | Per-weight isolation; loses joint-perturbation realism |

**Rationale:** The verified evidence carries two distinct magnitude signals: legacy artifact uses ±10% (`results.perturbation_method = "multiplicative ±10% then L1 renormalize to sum=1.0"`, Session 9 Q-V1.2), and YAML config cites ±20% for paper §11 (`configs/composite_risk_weights.yaml:10-11`, Session 8 Q-W3). Running both reconciles the two signals and preserves citability of the legacy `agreement_mean = 0.9823` while adding the magnitude referenced in the YAML. The legacy distribution saturates at [0.9669, 0.9947] under ±10% (Session 9 Q-V1.2 + Session 11 Q-V5.5); ±20% provides the additional stress test that ±10% does not. The incremental compute cost is one additional N=30 run, negligible against the audit value of having both magnitudes in the same JSON.

**Defense Q&A prep:**
- Q: Why did you pick this magnitude?
- A: Running both magnitudes covers the two signals in the verified evidence: legacy ±10% (preserves citability of `agreement_mean = 0.9823` from `results/rq1_sensitivity_analysis.json::results.perturbation_results`, Session 9 Q-V1.2) and YAML ±20% (matches the magnitude referenced for paper §11 in `configs/composite_risk_weights.yaml:10-11`, Session 8 Q-W3). The two magnitudes test different claims — ±10% as local robustness check, ±20% as the stress test that ±10%'s saturation at [0.9669, 0.9947] does not provide (Session 11 Q-V5.5).
- Q: How does this magnitude relate to the YAML's "±20%" comment and the legacy artifact's "±10%"?
- A: Running both reconciles the two signals. The YAML comment at `configs/composite_risk_weights.yaml:10-11` indicates paper §11 reports ±20% sensitivity; the legacy artifact at `results/rq1_sensitivity_analysis.json::results.perturbation_method` reports ±10%. Reporting both magnitudes makes the two pieces of pre-existing documentation mutually consistent and provides in-thesis evidence for both claims.

---

### D2 — Perturbation protocol

**Pick:** Joint random sampling (legacy framing)

**Constraint source:** Session 8 Q-W6.1 (spec lists OAT vs joint vs Dirichlet as open at `RQ1_pipeline.md:812`); Session 9 Q-V1.2 (legacy uses joint random N=30 under multiplicative+L1-renorm); Session 10 Q-V3.2 (in-pipeline uses 5-point grid + 12-point OAT).

**Verified evidence:**
- Spec (verbatim `RQ1_pipeline.md:812`): `Perturbation protocol: one-at-a-time vs joint sampling vs Dirichlet?`
- Legacy protocol (verbatim from JSON): `"multiplicative ±10% then L1 renormalize to sum=1.0"` over N=30; producer at `analysis/compute_rq1.py:343-349` does `rng.uniform(-0.10, 0.10, size=4)` then `pert / pert.sum()`.
- In-pipeline OAT (verbatim `module3_risk_scoring/module3_risk_scores.py:1151`): `for val in sweep:` where L1147 defines `sweep = np.arange(0.05, 0.65, 0.05)` (12 values from 0.05 to 0.60).
- In-pipeline grid (verbatim L1092): `grid_points = np.array([0.10, 0.20, 0.30, 0.40, 0.50])` with `w4 = 1.0 - w1 - w2 - w3` rounded and validity mask `(g4 >= 0.05) & (g4 <= 0.60)` at L1100.

**Options available:**

| Option | What it means | Trade-off considerations |
|---|---|---|
| Joint random sampling (legacy framing) | Sample all four weights jointly; N=30 or larger | Matches legacy; realistic combined perturbation; less per-weight interpretability |
| One-at-a-time (OAT) | Vary one weight, hold others, renormalize | Matches in-pipeline at L1071; per-weight curves; loses joint realism |
| Dirichlet sampling | Statistically principled simplex sampling | Bayesian flavor; may not match thesis's frequentist framing; more code |
| Hybrid: joint + OAT | Run both; report both | More work; richer defense; two outputs in the JSON |
| Grid search | Enumerate combinations exhaustively | In-pipeline already does 5-point grid; structured; combinatorial cost |

**Rationale:** The legacy producer at `analysis/compute_rq1.py:343-349` runs `rng.uniform(-0.10, 0.10, size=4)` then `pert / pert.sum()` — joint random sampling with L1 renormalize at N=30 (Session 9 Q-V1.2). Adopting the same protocol preserves the legacy distribution (`agreement_mean = 0.9823`, p25/p50/p75 = 0.9755/0.9828/0.9905) as continuing evidence rather than predecessor work. The in-pipeline `weight_sensitivity_analysis()` at `module3_risk_scoring/module3_risk_scores.py:1071` performs OAT for AUROC-driven weight search (Session 10 Q-V3), a different goal from Fix 1's tier-stability check; reusing its protocol conflates the two analyses. Joint sampling is explicitly named in the spec's open question list at `RQ1_pipeline.md:812` (`one-at-a-time vs joint sampling vs Dirichlet`).

**Defense Q&A prep:**
- Q: Why this protocol over the alternatives in the spec?
- A: Joint random sampling matches the legacy producer at `analysis/compute_rq1.py:343-349` (`rng.uniform(-0.10, 0.10, size=4)` then `pert / pert.sum()`). Adopting it preserves the legacy 30-perturbation distribution from `rq1_sensitivity_analysis.json` as continuing evidence rather than predecessor work. OAT conflates Fix 1 with the AUROC-driven weight search at `module3_risk_scoring/module3_risk_scores.py:1071` (Session 10 Q-V3); Dirichlet adds Bayesian framing inconsistent with the thesis's frequentist treatment.
- Q: How does this relate to the existing `weight_sensitivity_analysis()` function at L1071?
- A: The L1071 function performs an AUROC-driven grid + OAT search to find weights that maximize classifier performance (docstring L1078: `Grid search over weight space; evaluate AUROC of R as binary classifier`, Session 10 Q-V3). Fix 1 measures tier-agreement robustness around the policy-set baseline weights — different goal, different metric (exact tier match vs AUROC), different protocol (joint random vs grid + OAT). The two are complementary: the L1071 function answers `what weights maximize AUROC?`, Fix 1 answers `do the policy weights produce stable tiers under perturbation?`

---

### D3 — Agreement metric

**Pick:** Exact tier match

**Constraint source:** Session 8 Q-W6.1 (spec lists exact-match/κ/both as open at `RQ1_pipeline.md:814`); Session 9 Q-V1.3 (legacy reports scalar agreement; definition was [UNKNOWN] in Session 9); Session 11 Phase 0d Q-V5 (legacy's agreement metric definition resolved as outcome (a) — exact tier match).

**Verified evidence:**
- Spec (verbatim `RQ1_pipeline.md:814`): `Agreement metric: exact tier match, Cohen's κ, both?`
- Legacy reports (verbatim summary statistics, Session 9 Q-V1.2): `agreement_mean: 0.9823`; histogram all 30 in [0.9, 1.0]; baselines report a scalar `agreement` per baseline (0.7345 / 0.7659 / 0.7929).
- Phase 0d outcome (Session 11 Q-V5.3): exact tier match. Verbatim compute sites:
  - `analysis/compute_rq1.py:349`: `agreements.append(float(np.mean(tier == tier_base)))`
  - `analysis/compute_rq1.py:363`: `agreement = float(np.mean(tier == tier_base))`
  - Both operate on integer tier vectors from `_assign_tier(R, (0.80, 0.60, 0.40))`.
  - Histogram x-axis label `analysis/compute_rq1.py:383` (verbatim): `"Tier agreement with baseline weights"`.
- Phase 0d saturation framing (Session 11 Q-V5.5): the [0.9,1.0] saturation is consistent with exact-tier-match for small (±10% mult) perturbations because only rows whose `R_base` sits near a tier boundary (0.40, 0.60, 0.80) can flip tier under small shifts. A chance-adjusted (κ) measure would not saturate near 1.0 in the same way given the skewed tier distribution (LOW prevalent; YAML's expected distribution `LOW: "~35%"`, `MEDIUM: "~40%"`, `HIGH: "~20%"`, `CRITICAL: "~5%"`).

**Options available:**

_The following options are stated independently; Phase 0d's outcome (exact tier match) is the metric the legacy artifact's 0.9823 was computed under._

| Option | What it means | Trade-off considerations |
|---|---|---|
| Exact tier match | `(tiers_baseline == tiers_perturbed).mean()` | Simplest; defensible; granular per-alert |
| Cohen's κ | Chance-adjusted tier agreement | Statistically principled; chance correction; less intuitive |
| Both exact match + κ | Report both per condition | Belt-and-suspenders; one extra column; defensible |
| Surfacing-match | Agreement on binary surface/not-surface decisions | Production-relevant; coarser than tier-match |
| Continuous-R proxy | E.g., `1 - mean(abs(R_base - R_pert))` | Tier-bypassing; comparable to legacy if legacy used this |

**Rationale:** Phase 0d Q-V5.3 verified that the legacy artifact computes agreement as `np.mean(tier == tier_base)` on integer tier vectors from `_assign_tier(R, (0.80, 0.60, 0.40))` at `analysis/compute_rq1.py:349`. Adopting exact tier match for Fix 1 preserves direct citability of the legacy `agreement_mean = 0.9823` and the per-baseline values (0.7345 / 0.7659 / 0.7929 per Session 9 Q-V1.2). The metric maps to the property Fix 1 is testing — tier stability under policy-perturbation — the same property the legacy's `fnr_critical_delta` baseline-evaluation tracks (Session 9 Q-V1.2 — `equal_weights.fnr_critical_delta = 0.0106`). Cohen's κ on the skewed tier distribution at `configs/composite_risk_weights.yaml:31-34` (LOW ~35%, MEDIUM ~40%, HIGH ~20%, CRITICAL ~5%) carries a high chance-agreement baseline by construction; this is the answer to the chance-correction critique without requiring κ to be the headline statistic (Session 11 Q-V5.5).

**Defense Q&A prep:**
- Q: How does your agreement metric relate to the legacy artifact's 0.9823 value?
- A: The legacy artifact computed agreement as `np.mean(tier == tier_base)` on integer tier vectors (Phase 0d Q-V5.3, `analysis/compute_rq1.py:349`). Fix 1 uses the identical definition. The 0.9823 mean across 30 perturbations from `results/rq1_sensitivity_analysis.json::results.perturbation_results` remains directly citable as continuing evidence under Fix 1's ±10% magnitude condition.
- Q: Why this metric over the alternatives in `RQ1_pipeline.md` §6.1?
- A: Exact tier match measures the property Fix 1 is testing — tier stability under policy-perturbation — the property tracked by the legacy's `fnr_critical_delta` baseline evaluation (Session 9 Q-V1.2). Cohen's κ at the tier prevalence in `configs/composite_risk_weights.yaml:31-34` (LOW ~35%, MEDIUM ~40%, HIGH ~20%, CRITICAL ~5%) carries a high chance-agreement baseline by construction; this is the structural answer to the chance-correction critique without requiring κ to be the headline metric (Session 11 Q-V5.5).

---

### D4 — Multiplicative R alternative

**Pick:** Keep as comparator baseline (legacy framing)

**Constraint source:** Session 8 Q-W6.1 (spec lists "separate condition or skip" as open at `RQ1_pipeline.md:815`); Session 9 Q-V1.2 (legacy includes multiplicative baseline with agreement 0.7929, fnr_critical_delta 0.0); Session 10 Q-V2.1 (`apply_weight_feedback()` assumes additive form via per-weight sweep + L1 renormalize); YAML L41 acknowledged limitation L1.

**Verified evidence:**
- Spec (verbatim `RQ1_pipeline.md:815`): `Multiplicative R: implement as a separate condition or skip?`
- Legacy baseline values (verbatim `results.baselines.multiplicative`): `{"agreement": 0.7929, "fnr_critical_delta": 0.0}`.
- Legacy multiplicative formula (verbatim `analysis/compute_rq1.py:357-359`): `R = comp["c_detect"] * np.maximum.reduce([comp["d_crit"], comp["s_data"], comp["d_clinical_tier"]])` — specifically `c_detect · max(d_crit, s_data, d_clinical_tier)`, not a four-factor product.
- Implementation coupling (Session 10 Q-V2): `apply_weight_feedback()` at L627 sweeps each of four weights, builds an (11, 4) trial-weight matrix, and row-normalizes via `row_sums = w_matrix.sum(axis=1, keepdims=True); w_matrix /= row_sums`. Meaningful only for additive R with sum-to-1.0; would not transfer to multiplicative R without changes.
- YAML acknowledged limitation (verbatim `configs/composite_risk_weights.yaml:41`): `L1: Linear sum allows compensatory effects vs true multiplicative risk`.

**Options available:**

| Option | What it means | Trade-off considerations |
|---|---|---|
| Keep as comparator baseline (legacy framing) | Multiplicative R reported as a named baseline; primary formula stays additive | Matches legacy; preserves 0.7929 number; minimal new code |
| Promote to alternative formula | Run sensitivity on multiplicative R as the primary | Major design shift; calibration utility (`apply_weight_feedback`) no longer applies without changes |
| Skip / drop | Remove the multiplicative baseline entirely | Cleaner scope; loses an existing baseline; departs from legacy |
| Defer to future work | Acknowledge in thesis §7.X as named future work | Scope-controlled; honest about limits |

**Rationale:** The legacy artifact already reports `multiplicative` as a named baseline at `results.baselines.multiplicative = {"agreement": 0.7929, "fnr_critical_delta": 0.0}` (Session 9 Q-V1.2). Keeping the same framing preserves the 0.7929 number for direct citation and matches the legacy's protocol shape (Sessions 9 + 11). Promoting multiplicative R to primary requires refactoring `apply_weight_feedback()` at `module3_risk_scoring/module3_risk_scores.py:627` (Session 10 Q-V2), which sweeps each of four weights and L1-normalizes them under additive-form assumptions; that refactor is outside Fix 1's scope. The YAML's acknowledged limitation L1 at `configs/composite_risk_weights.yaml:41` (`Linear sum allows compensatory effects vs true multiplicative risk`) is preserved as transparent design choice with the comparator baseline as the in-thesis evidence.

**Defense Q&A prep:**
- Q: Why include / exclude multiplicative R?
- A: Multiplicative R is included as a named baseline comparator, retaining the legacy artifact's framing at `results.baselines.multiplicative` (agreement 0.7929, fnr_critical_delta 0.0; Session 9 Q-V1.2; formula at `analysis/compute_rq1.py:357-359` = `c_detect * np.maximum.reduce([d_crit, s_data, d_clinical_tier])`). The additive primary's perturbation distribution (`agreement_mean = 0.9823`) alongside the multiplicative comparator's 0.7929 gives the thesis concrete evidence on both formulations and directly addresses YAML L1's acknowledged limitation. Promoting multiplicative to primary requires refactoring `apply_weight_feedback()` at L627 (Session 10 Q-V2), outside Fix 1's scope.

---

### R3 — Legacy artifact disposition

**Pick:** Supersede

**Constraint source:** Session 10 Q-V4.2 (merge script's documented precedence rule); Session 11 Q-V5 (legacy methodology now fully characterized).

**Verified evidence:**
- Merge script behavior (verbatim docstring `analysis/merge_rq1_metrics.py:15-18`): `When both ``rq1_weight_sensitivity.json`` (new) and ``rq1_sensitivity_analysis.json`` (legacy) exist, the new one wins; the legacy block is preserved under ``weight_sensitivity._legacy_evidence`` for traceability.`
- Merge script status string (verbatim `analysis/merge_rq1_metrics.py:64-68`, used when only legacy is present): `"v1 evidence from legacy analysis/compute_rq1.py — perturbation protocol pending finalisation per RQ1_pipeline.md §6.1"`
- Legacy methodology fully characterized (Sessions 9 + 11): ±10% multiplicative joint random N=30 with L1 renormalize; exact tier match; three named baselines with fnr_critical_delta.

**Options available:**

| Option | What it means | Trade-off considerations |
|---|---|---|
| Formalize | Rename legacy file to `rq1_weight_sensitivity.json`; Fix 1 inherits its conditions | Lowest implementation cost; commits to legacy choices for D1/D2/D4 |
| Supersede | Write fresh new file; legacy auto-nested as `_legacy_evidence` per merge script | Clean break; preserves audit trail; freedom on D1–D4 |
| Coexist as comparator | Both files canonical; new file is primary, legacy cited as prior work | Documentation cost; richest evidence base; potential reader confusion |

**Rationale:** Writing Fix 1's output to `results/rq1_weight_sensitivity.json` triggers the merge script's documented precedence rule at `analysis/merge_rq1_metrics.py:15-18`: `the new one wins; the legacy block is preserved under weight_sensitivity._legacy_evidence` (Session 10 Q-V4.2). The aggregator scope is zero LOC because the merge script performs whole-dict assignment without sub-key inspection (Session 10 Q-V4.3). Fix 1 inherits the legacy's protocol shape on D2/D3/D4 (joint random, exact tier match, multiplicative comparator) and extends the magnitude axis from ±10% to both ±10% and ±20%; the legacy's numbers are a strict subset preserved as evidence under `_legacy_evidence`. The merge script's documented status string at `analysis/merge_rq1_metrics.py:64-68` (`v1 evidence from legacy analysis/compute_rq1.py — perturbation protocol pending finalisation per RQ1_pipeline.md §6.1`) is the pre-existing framing for exactly this supersession path.

**Defense Q&A prep:**
- Q: What is your relationship to the prior `rq1_sensitivity_analysis.json`?
- A: The prior `results/rq1_sensitivity_analysis.json` is preserved as v1 evidence via the merge script's documented `_legacy_evidence` mechanism (`analysis/merge_rq1_metrics.py:15-18`, Session 10 Q-V4.2). Fix 1's `rq1_weight_sensitivity.json` is the canonical Stage 5B artifact; the legacy is auto-nested by the merge script for traceability with no manual migration step. Methodologically, Fix 1 inherits the legacy's protocol (joint random) and metric (exact tier match per Phase 0d Q-V5.3) and extends magnitude coverage from ±10% alone to both ±10% and ±20%.

---

### R1 — Naming collision resolution

**Pick:** Keep all three names with documentation

**Constraint source:** Sessions 9 and 10 — three closely-named things refer to three distinct purposes (function, legacy JSON, planned JSON).

**Verified evidence:**
- `weight_sensitivity_analysis()` at `module3_risk_scoring/module3_risk_scores.py:1071` — AUROC grid + OAT search, PNG output, 1 caller at L1525 (Session 10 Q-V3).
- `rq1_sensitivity_analysis.json` at `results/` — legacy JSON, joint random ±10%, exact tier match, baselines including multiplicative comparator (Sessions 9 + 11).
- `rq1_weight_sensitivity.json` at `results/` — Fix 1's planned output, spec-named at `RQ1_pipeline.md:27` and `RQ1_pipeline.md:819` (Sessions 8 + 10).
- Migration touchpoints if renaming the legacy JSON: `analysis/build_thesis_results.py:51` and `:425` consume it; `analysis/merge_rq1_metrics.py:30` defines its path constant; `results/THESIS_RESULTS.md` references it textually (per Session 10 §4 grep).

**Options available:**

| Option | What it means | Trade-off considerations |
|---|---|---|
| Keep all three names with documentation | Memo + ARCHITECTURE.md note explaining the distinctions | No code/file changes; documentation cost; potential reader confusion |
| Rename legacy file | E.g., `results/rq1_legacy_sensitivity_analysis.json` | Migration cost (build_thesis_results.py reference); cleaner |
| Rename in-pipeline function | E.g., `weight_grid_auroc_search()` at L1071 | Code edit beyond doc-only scope; requires explicit memo authorization; clearer |
| Hybrid | Rename one, document the others | Balanced; specific cost depends on what's renamed |

**Rationale:** The merge script already names both JSON files at `analysis/merge_rq1_metrics.py:29-30` (`WS_NEW = results/rq1_weight_sensitivity.json`; `WS_LEGACY = results/rq1_sensitivity_analysis.json`); the precedence rule at L15-18 (Session 10 Q-V4) requires no file rename. Renaming `weight_sensitivity_analysis()` at `module3_risk_scoring/module3_risk_scores.py:1071` is a code edit outside Fix 1's analysis-and-doc scope; the function is invoked by the main pipeline at L1525 (Session 10 Q-V3.4) and a rename carries migration cost on that call site plus any imports. A one-paragraph note in ARCHITECTURE.md and the thesis methodology section disambiguates the three names against zero code-edit cost. The three distinct purposes are already documented in this memo's three-surface landscape table above.

**Defense Q&A prep:**
- Q: Why are there three "sensitivity" things in your codebase?
- A: The three serve distinct purposes. `weight_sensitivity_analysis()` at `module3_risk_scoring/module3_risk_scores.py:1071` is an AUROC-driven weight search that runs once per main-pipeline execution and saves a PNG (Session 10 Q-V3). `rq1_sensitivity_analysis.json` is the legacy robustness artifact preserved as v1 evidence under the merge script's `_legacy_evidence` key (Session 10 Q-V4.2). `rq1_weight_sensitivity.json` is Fix 1's robustness artifact and the canonical Stage 5B output per the spec at `RQ1_pipeline.md:819`. The disambiguation is documented in ARCHITECTURE.md and the thesis methodology section per the Phase 4 outline.

---

### R2 — Split choice

**Pick:** [USER REVIEW NEEDED: Phase 0e parquet-row-count result is not in conversation context. Decision logic: if `val_phase1` rows = 2448 and `test_phase1` rows ≠ 2448, pick `val_phase1` (matches legacy `n_alerts_evaluated = 2448` inference and preserves test split for headline metrics). If both splits = 2448 rows, the legacy producer's inline comment at `analysis/compute_rq1.py:282` ("test-split sourced") becomes the deciding signal — pick `test_phase1` to align with the comment and the in-pipeline function at `module3_risk_scores.py:1525`, or pick `val_phase1` to align with the row-count inference and avoid same-split sensitivity. User decides.]

**Constraint source:** Session 8 Q-W4 (both val and test parquets exist; `load_split_data` rejects `"val"`); Session 9 Q-V1.2 (legacy reports `n_alerts_evaluated: 2448`, matching val_phase1's row count from `RQ1_pipeline.md:779`); Session 10 Q-V3.4 (in-pipeline uses `y_test`); Session 11 §4 (split contradiction surfaced — legacy producer's inline comment says "test-split sourced" while row count matches val).

**Verified evidence:**
- Both parquets exist (Session 8 Q-W4): `data/processed/test_phase1.parquet` (131,925 bytes, May 7 16:28); `data/processed/val_phase1.parquet` (131,974 bytes, May 7 16:28).
- `load_split_data` restriction (verbatim `module3_risk_scoring/module3_risk_scores.py:221-222`): `if split not in ("test", "demo"): raise ValueError(f"split must be 'test' or 'demo', got {split!r}")`
- In-pipeline split (verbatim `module3_risk_scoring/module3_risk_scores.py:1525`): `sensitivity = weight_sensitivity_analysis(c_detect, d_crit, s_data, d_clinical_tier, y_test)`
- Legacy producer comment (verbatim `analysis/compute_rq1.py:282`): `# Load risk components from M3 risk_scores.npz (test-split sourced)` — contradicts the n=2448 ↔ val inference; Phase 1 must reconcile before claiming a split.
- val_phase1 row count from spec (verbatim `RQ1_pipeline.md:779`): `data/processed/val_phase1.parquet (2,448 rows; canonical Phase-1 held-out validation split; disjoint from train/test/demo per split_metadata.yaml)`.

**Options available:**

| Option | What it means | Trade-off considerations |
|---|---|---|
| val_phase1 | Sensitivity on val; preserves test split clean for headline metrics | Matches legacy (if n=2448 inference holds); requires 1-line extension to load_split_data |
| test_phase1 | Same split as headline metrics | Matches in-pipeline; no loader change; same-split sensitivity is borderline |
| Both | Run sensitivity on both; report each | Belt-and-suspenders; double the run time; cleanest defense |

**Rationale:** _TO BE FILLED AFTER R2 RESOLVES (pending Phase 0e parquet-row-count result)_

**Defense Q&A prep:**
- Q: Why this split for sensitivity analysis?
- A: _TO BE FILLED AFTER R2 RESOLVES_
- Q: How does this reconcile with the Phase 1 finding that the legacy producer's "test-split sourced" comment contradicts the n=2448 row-count inference?
- A: _TO BE FILLED AFTER R2 RESOLVES_

---

## Phase 2 Implementation Outline

### Files to create

- `analysis/compute_weight_sensitivity.py` (new, ~150–250 LOC) — the Stage 5B analysis script. Mirrors the legacy producer's protocol structure with the magnitude extension:
  - Function `run_perturbation_analysis(weights_base, magnitudes=[0.10, 0.20], n=30, split=<R2 PICK>)`.
  - For each magnitude: `n` joint random perturbations via `rng.uniform(-mag, mag, size=4)` then `pert / pert.sum()` (legacy protocol at `analysis/compute_rq1.py:343-349`).
  - Per perturbation: call `compute_composite_risk(c_detect, d_crit, s_data, d_clinical_tier, weights={"w1": w1, "w2": w2, "w3": w3, "w4": w4})` (signature per Session 8 Q-W2), assign tier via `_assign_tier(R, (0.80, 0.60, 0.40))` or equivalent, compute exact tier match vs baseline tier vector.
  - Three named baselines retained from legacy: `equal_weights`, `c_detect_only`, `multiplicative` (formula at `analysis/compute_rq1.py:357-359`: `c_detect * np.maximum.reduce([d_crit, s_data, d_clinical_tier])`).
  - Per condition: `agreement` (exact tier match) and `fnr_critical_delta` (legacy formula `analysis/compute_rq1.py:365-369`: `crit_base = (tier_base == 3); fnr_delta = mean(crit_base & (tier < 3))`).
  - Schema: `{provenance, results: {perturbation_results: {by_magnitude: {"0.10": {n_perturbations, agreement_mean, agreement_std, agreement_min, agreement_max, agreement_p25, agreement_p50, agreement_p75, histogram_counts, histogram_edges}, "0.20": {same}}}, baselines: {equal_weights, c_detect_only, multiplicative}, baseline_weights, tier_boundaries, perturbation_method, n_alerts_evaluated}}`.
  - Output: `results/rq1_weight_sensitivity.json`.

- `tests/test_weight_sensitivity_invariants.py` (new) — invariant tests:
  - `test_safety_floor_holds_across_perturbations`: assert max `fnr_critical_delta` across all 60 perturbations and 3 baselines stays below the Phase-3-determined threshold.
  - `test_sum_to_one_invariant`: assert every perturbed weight dict satisfies `abs(sum - 1.0) < 1e-6`, consistent with the production invariant at `module3_risk_scoring/module3_risk_scores.py:86-90` (Session 8 Q-W3).
  - `test_schema_conformance`: assert output JSON has expected top-level keys per the schema above.

### Files to modify

- None for production code. `compute_composite_risk()` at `module3_risk_scoring/module3_risk_scores.py:552` already accepts an injectable `weights: dict | None = None` parameter (Session 8 Q-W2.1); no refactor required.
- None for the aggregator. `analysis/merge_rq1_metrics.py:29-30` already names `WS_NEW = results/rq1_weight_sensitivity.json`, and L52-61 handles the new-wins-with-legacy-nested precedence (Session 10 Q-V4); zero-LOC change.

### Optional 1-line touch (conditional on R2 picking val_phase1)

- `module3_risk_scoring/module3_risk_scores.py:221-222`: extend `load_split_data()`'s allowed-set to include `"val"`, or have `analysis/compute_weight_sensitivity.py` call `pd.read_parquet` directly to bypass the loader's `if split not in ("test", "demo"): raise ValueError` (Session 8 Q-W4).

---

## Phase 3 Verification Outline

### Acceptance gates

- **V-1:** `python analysis/compute_weight_sensitivity.py` exits 0 and writes `results/rq1_weight_sensitivity.json` with a non-empty `results.perturbation_results.by_magnitude` block for both `0.10` and `0.20`.
- **V-2:** Python one-liner on the output confirms schema conformance: top-level `provenance` + `results`; `results.perturbation_results.by_magnitude` keys `["0.10", "0.20"]` each with `n_perturbations = 30`; three baselines `[equal_weights, c_detect_only, multiplicative]` each carrying `agreement` and `fnr_critical_delta`.
- **V-3:** `python analysis/merge_rq1_metrics.py` updates `results/rq1_metrics.json` with `weight_sensitivity` populated from the new file and `weight_sensitivity._legacy_evidence` populated from `results/rq1_sensitivity_analysis.json` (per merge script L52-61, Session 10 Q-V4.2).
- **V-4:** `pytest tests/test_weight_sensitivity_invariants.py` passes all three invariant tests.
- **V-5:** `pytest tests/acceptance_tests.py` (existing headline-target tests, per `RQ1_pipeline.md` Phase 3) shows no regression — no production code is modified per the Phase 2 outline.
- **V-6:** `git status --short` shows only `analysis/compute_weight_sensitivity.py` (new), `tests/test_weight_sensitivity_invariants.py` (new), modified `results/rq1_metrics.json`, and `results/rq1_weight_sensitivity.json` (new); no other files modified.
- **V-7:** legacy preservation check — Python one-liner asserts `d["weight_sensitivity"]["_legacy_evidence"]["results"]["perturbation_results"]["agreement_mean"] == 0.9823` confirms the legacy value remains accessible (Session 9 Q-V1.2).

---

## Phase 4 Documentation Outline

### ARCHITECTURE.md

- Module 3 section: add policy-parameter framing paragraph citing `configs/composite_risk_weights.yaml:7-8` (`POLICY parameters — set by hospital security/clinical leadership, NOT learned from data. Reviewed annually.`) plus a forward-reference to Stage 5B sensitivity as the robustness evidence for those policy parameters.
- New paragraph disambiguating the three names: `weight_sensitivity_analysis()` (AUROC search, PNG side-effect), `rq1_sensitivity_analysis.json` (v1 robustness evidence, legacy), `rq1_weight_sensitivity.json` (Stage 5B canonical robustness artifact).

### RQ1_pipeline.md §6.1

- Update L806 status header from `### 6.1 Stage 5B — Weight sensitivity (SPEC PENDING, ``a_high`` half RESOLVED in §5b)` to `### 6.1 Stage 5B — Weight sensitivity (RESOLVED per docs/fix1_design_memo.md, ``a_high`` half RESOLVED in §5b)`.
- Replace L810-815's `Still pending` block plus the four open questions with the locked D1–D4 picks from this memo (D1: ±10% and ±20%; D2: joint random with L1 renormalize; D3: exact tier match; D4: multiplicative comparator baseline).

### senior_engineer_review.md §3.1

- Annotate the §3.1 `unjustified weights` critique with `addressed via Stage 5B (docs/fix1_design_memo.md)`; cite the Fix 1 sensitivity result (`rq1_weight_sensitivity.json` per V-2) as the resolution evidence.

### Thesis sections (handoff text — `thesis_outline_latest.docx` absent per Session 6 Q-V6 / Session 7 Open Items)

- §3.3.2 (weights introduction): policy-parameter framing paragraph draft sourced from YAML L7-8.
- §5.2.4 (sensitivity result): sensitivity table draft with `[±10%, ±20%]` × `[30, 30]` perturbations, three baselines (`equal_weights`, `c_detect_only`, `multiplicative`), and defense-Q&A-grounded interpretation (`tier stability under policy-perturbation`, `multiplicative comparator at 0.7929`, `safety floor via fnr_critical_delta`).
- §11 (limitations) cross-reference: link the YAML's acknowledged limitation L1 at `configs/composite_risk_weights.yaml:41` (`Linear sum allows compensatory effects vs true multiplicative risk`) to the multiplicative comparator baseline as transparent in-thesis evidence.

---

## Open Items

**In Fix 1 scope (will be addressed in Phase 2/3/4):**

- Split contradiction (R2 dependency): pending Phase 0e resolution per the `[USER REVIEW NEEDED]` marker in R2's Pick cell. Rationale and Defense Q&A get filled once Phase 0e's parquet-row-count is in.
- Three-name disambiguation: addressed by R1 Pick — documentation paragraphs in ARCHITECTURE.md and the thesis methodology section per the Phase 4 outline.

**Out of Fix 1 scope (deferred to separate work):**

- Session 8 §4 follow-up #4: contents of the aggregator's `risk_weights` echo block at `module6_evaluation/compute_rq1_metrics.py:107` — aggregator detail; non-blocking for Fix 1.
- Session 10 §4: `dynamic_threshold_sim.py:52` imports `apply_weight_feedback` but no invocation surfaced in the grep window — hygiene check; non-blocking.
- Session 10 §4: `feedback_loop_demo.py` and `dynamic_threshold_sim.py` invocation context (demo / one-shot vs Makefile / CI) — hygiene check; non-blocking.
- Session 10 §4: `weight_sensitivity_analysis()` validity mask `(g4 >= 0.05) & (g4 <= 0.60)` upper bound 0.60 not cross-checked against YAML policy bounds — concern for the L1071 function's own design; non-blocking for Fix 1.
- Session 10 §4: plot at `CHARTS_DIR / "weight_sensitivity.png"` (L1179) was not checked for staleness on disk — artifact hygiene; non-blocking.

**Open (require user attention):**

- Thesis docx absence (Session 6 Q-V6 / Session 7 Open Items table): Phase 4 thesis edits land as handoff text for the human author rather than direct docx edits.

---

## Audit Trail

- Phase 0 discovery: `Codebase_Investigation.html` Session 8 (lines 1259–1523)
- Phase 0b discovery: Session 9 (lines 1524–1738) — Q-V1 STOP after outcome (a-fresh)
- Phase 0c discovery: Session 10 (lines 1739–2033) — Q-V4 + Q-V2 + Q-V3
- Phase 0d discovery: Session 11 (lines 2034–2229) — Q-V5 closed legacy agreement metric [UNKNOWN]
- This memo records decisions derived from the above. Picks, Rationales, and Defense Q&A answers are filled in subsequent Phase 1 steps (2–7), not by the bootstrap that wrote this template.
