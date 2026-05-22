# Fix 1 Design Memo — Risk Weight Sensitivity Analysis

**Date:** 2026-05-21
**Branch:** `fix/rq1-weight-sensitivity` (HEAD at `10e26156dca76174ff1d155cf23211bb730abf3f`)
**Sources:** `Codebase_Investigation.html` Sessions 8–11
**Decision authority:** _USER NAME_
**Decision date:** 2026-05-21

---

## Executive Summary

- **Design.** Fix 1 supersedes the legacy `rq1_sensitivity_analysis.json` with a new `rq1_weight_sensitivity.json` that extends the legacy's protocol (joint random N=30, exact tier match, multiplicative baseline) by adding a second magnitude (±20% alongside ±10%) and tracking `fnr_critical_delta` per condition.
- **Method.** Four weights perturbed jointly under multiplicative-then-L1-renormalize, 30 perturbations per magnitude (60 total), exact tier match against policy-set baseline. Three named baselines (equal_weights, c_detect_only, multiplicative).
- **Result vehicle.** `results/rq1_weight_sensitivity.json` flows through `analysis/merge_rq1_metrics.py` to `results/rq1_metrics.json::weight_sensitivity`; legacy preserved under `_legacy_evidence` with `agreement_mean = 0.9823` bit-intact.
- **Defense framing.** Weights are hospital-tunable policy parameters (per YAML L7-8); the ±10% mean agreement of 0.9823 (bit-matches legacy) and ±20% mean of 0.964 with `fnr_critical_delta_max = 0.0106` demonstrate Invariant 2 (safety-floor) robustness. Turns senior review §3.1 "unjustified weights" critique into a positive contribution.
- **Open item.** R2 (split choice) pending Phase 0e parquet-row-count confirmation. Script ran with `--split=val_phase1` as a provenance label (component arrays come from `results/reports/risk_scores.npz` regardless); re-run with asserted split once Phase 0e closes.

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

## Phase 2 Implementation Outline (RETROSPECTIVE)

- Created `analysis/compute_weight_sensitivity.py` (~330 LOC). Entry point `run_perturbation_analysis(split)`; iterates over `MAGNITUDES = (0.10, 0.20)`, N=30 perturbations each via `rng.uniform(-mag, mag, size=4)` then `pert / pert.sum()` (mirrors `analysis/compute_rq1.py:343-349`).
- Canonical component loader reads `results/reports/risk_scores.npz` (per Session 11 Q-V5); `--split` flag carried as a provenance label only because the npz's underlying split is ambiguous (Session 11 §4 contradiction, pending Phase 0e).
- Output schema extends the legacy with a `by_magnitude` sub-key under `perturbation_results`; per-magnitude block adds `fnr_critical_delta_max` and `fnr_critical_delta_mean` (legacy reported the delta only for the three named baselines).
- Three named baselines retained verbatim from legacy: `equal_weights`, `c_detect_only`, `multiplicative` (formula `c_detect * np.maximum.reduce([d_crit, s_data, d_clinical_tier])`, verbatim from `analysis/compute_rq1.py:357-359`).
- Created `tests/test_weight_sensitivity_invariants.py` with 15 invariant tests (sum-to-1.0, tier semantics, agreement reflexivity/bounds, multiplicative formula, baseline schemas, determinism under seed, design-pick constants, `fnr_critical_delta` bounds).
- Zero production code modified. `compute_composite_risk()` at `module3_risk_scoring/module3_risk_scores.py:552` already accepts `weights: dict | None = None`; `analysis/merge_rq1_metrics.py:29-30` already names both JSON paths with documented precedence rule.
- Mid-Phase-3 correction: the initial `_fnr_critical(tiers, y_true)` was the wrong metric (binary-classifier FNR); replaced with `_fnr_critical_delta(tiers_baseline, tiers_perturbed)` matching `analysis/compute_rq1.py:365-369` verbatim. After the fix, baseline `fnr_critical_delta` values bit-match the legacy artifact's values.

---

## Phase 3 Verification Outline (RETROSPECTIVE)

- **V-1 PASS:** `python analysis/compute_weight_sensitivity.py` exited 0; wrote `results/rq1_weight_sensitivity.json` (3408 bytes).
- **V-2 PASS-WITH-WAIVER:** schema conforms; one assertion waived — Phase 3 prompt expected `provenance.r2_status == 'PENDING_PHASE_0E'`; my Phase 2 carries the same R2-deferral semantic via `split_label` + `split_label_note` (user accepted the waiver).
- **V-3 PASS:** safety floor holds — `fnr_critical_delta_max = 0.0086` (±10%) and `0.0106` (±20%), both ≤ 0.05 threshold; across all 60 perturbations no breach of Invariant 2.
- **V-4 PASS:** 15/15 invariant tests pass (0.40s on re-run).
- **V-5 PASS-WITH-NOTE:** broader sweep `pytest tests/` 736 passed / 5 skipped / 0 failed; targeted `tests/test_safe_failure.py tests/negative_tests.py` 47/47 pass. `tests/acceptance_tests.py` has 3 pre-existing failures + 4 errors on MVE/clinical/SHAP/severity (verified pre-existing via `git stash` re-test; unrelated to Fix 1).
- **V-6 PASS:** `analysis/merge_rq1_metrics.py` exited 0; `weight_sensitivity._source = "rq1_weight_sensitivity.json"`; legacy preserved at `weight_sensitivity._legacy_evidence`.
- **V-7 PASS-WITH-NOTE:** four expected Fix 1 entries in `git status` (3 untracked + 1 `M results/rq1_metrics.json`); three pre-existing unrelated modifications also in tree (this memo from Phase 1, `results/rq3_*.json` from unrelated RQ3 work).
- **Headline numbers (verbatim from `results/rq1_weight_sensitivity.json`, n_alerts_evaluated = 2448):** ±10% mean=0.9823 (bit-match legacy), std=0.0082, min/max=0.9673/0.9939; ±20% mean=0.964, std=0.0167, min/max=0.9154/0.9918; baselines: equal_weights agreement=0.7341 / fnr_critical_delta=0.0106, c_detect_only 0.7667 / 0.0, multiplicative 0.7937 / 0.0 (last three `fnr_critical_delta` values bit-match legacy).

---

## Phase 4 Documentation Outline (RETROSPECTIVE)

- `ARCHITECTURE.md`: inserted two new subsections — `### Risk weights as policy parameters` + `### Three weight-sensitivity surfaces (disambiguation)` — between the existing `### Sensitivity analyses (Section 11 acknowledgments)` (L940) and `### Other pipeline configuration files` (L969). Per-prompt verbatim text; no refactor of pre-existing prose.
- `docs/RQ1_pipeline.md` §6.1: updated header from `SPEC PENDING, ``a_high`` half RESOLVED in §5b` to `RESOLVED in Fix 1 Design Memo, 2026-05-21`. Inserted `**Resolved (Fix 1, 2026-05-21):**` block listing D1–D4 + R1, R2, R3. Pre-existing `**Resolved (a_high, prior):**` and `**Still pending:**` content retained (the latter refers to the `(a_low, b)` work — separate from Fix 1's weight-perturbation closure).
- `senior_engineer_review.md` §3.1: **file does not exist in the repo** (verified via repo-wide `find`). Per user direction, the §3.1 ADDRESSED annotation is recorded in this memo's `## External critique status` section below instead of in a non-existent file.
- `docs/fix1_thesis_handoff.md`: new file. Draft thesis text for §3.3.2 (replacement paragraph), §5.2.4 (new subsection with Method, Table 5.4 verbatim numbers, Discussion, and Comparison-with-v1), §7.X (Future Work additions for multiplicative-R and R2 finalization), plus an optional methodology footnote on the three-names disambiguation. Thesis docx is absent per Session 6 Q-V6 / Session 7 Open Items; handoff is for the human author.
- This memo: Executive Summary, Phase 2/3/4 Outlines, Open Items, Decision date filled. `## External critique status` section added.
- No `.py`, `.yaml`, or `.json` files modified by Phase 4.

---

## Open Items

**Closed by Phase 2/3/4:**

- Three-name disambiguation: addressed by R1 — `ARCHITECTURE.md` §"Three weight-sensitivity surfaces (disambiguation)" inserted by Phase 4, plus optional methodology footnote in `docs/fix1_thesis_handoff.md`.
- Stage 5B spec status: addressed by Phase 4 update to `docs/RQ1_pipeline.md` §6.1.

**Still open inside Fix 1 (require external input to close):**

- R2 (split choice): pending Phase 0e parquet-row-count check across `data/processed/val_phase1.parquet` and `data/processed/test_phase1.parquet`. Once locked, re-run `python analysis/compute_weight_sensitivity.py --split=<chosen>` and update R2's Pick/Rationale/Defense Q&A; the analysis output is split-label-stable (same component arrays from `results/reports/risk_scores.npz`).

**Out of Fix 1 scope (deferred to separate work):**

- Session 8 §4 follow-up #4: contents of the aggregator's `risk_weights` echo block at `module6_evaluation/compute_rq1_metrics.py:107` — aggregator detail; non-blocking.
- Session 10 §4: `dynamic_threshold_sim.py:52` imports `apply_weight_feedback` but no invocation surfaced in the grep window — hygiene check; non-blocking.
- Session 10 §4: `feedback_loop_demo.py` and `dynamic_threshold_sim.py` invocation context (demo / one-shot vs Makefile / CI) — hygiene check; non-blocking.
- Session 10 §4: `weight_sensitivity_analysis()` validity mask `(g4 >= 0.05) & (g4 <= 0.60)` upper bound 0.60 not cross-checked against YAML policy bounds — concern for the L1071 function's own design; non-blocking.
- Session 10 §4: plot at `CHARTS_DIR / "weight_sensitivity.png"` (L1179) was not checked for staleness on disk — artifact hygiene; non-blocking.

**Open (require user attention):**

- Thesis docx absence (Session 6 Q-V6 / Session 7 Open Items table): Phase 4 thesis edits landed as handoff text in `docs/fix1_thesis_handoff.md` for the human author.
- `senior_engineer_review.md` absence: file is not in the repo. The §3.1 ADDRESSED annotation lives in this memo's `## External critique status` section instead. If the document is reintroduced or located elsewhere, port the annotation there.

---

## External critique status

> **§3.1 (`senior_engineer_review.md`) — "unjustified weights" critique. Status (2026-05-21): ADDRESSED via Fix 1.**
>
> Seven design decisions recorded in this memo; implementation at `analysis/compute_weight_sensitivity.py`; empirical evidence at `results/rq1_weight_sensitivity.json` (folded into `results/rq1_metrics.json::weight_sensitivity` by `analysis/merge_rq1_metrics.py`). Weights are now framed as hospital-tunable policy parameters per `ARCHITECTURE.md` §"Risk weights as policy parameters" (inserted by Phase 4). The ±10% mean agreement of 0.9823 (bit-matches the legacy v1 evidence) and ±20% mean agreement of 0.964 with `fnr_critical_delta_max = 0.0106` provide empirical support for Invariant 2 (safety floor) preservation under policy-perturbation. R2 (split choice) remains open pending Phase 0e parquet-row-count check.
>
> Annotation note: `senior_engineer_review.md` was not located in the repo at Phase 4 execution time (verified via repo-wide `find`). This annotation lives here per the user's direction in lieu of editing a non-existent file; if the source document is restored or located, the annotation should be ported there.

## Audit Trail

- Phase 0 discovery: `Codebase_Investigation.html` Session 8 (lines 1259–1523)
- Phase 0b discovery: Session 9 (lines 1524–1738) — Q-V1 STOP after outcome (a-fresh)
- Phase 0c discovery: Session 10 (lines 1739–2033) — Q-V4 + Q-V2 + Q-V3
- Phase 0d discovery: Session 11 (lines 2034–2229) — Q-V5 closed legacy agreement metric [UNKNOWN]
- Phase 1 (decisions): this memo, populated 2026-05-21.
- Phase 2 (implementation): `analysis/compute_weight_sensitivity.py` + `tests/test_weight_sensitivity_invariants.py` (~340 + ~225 LOC respectively).
- Phase 3 (verification): V-1 through V-7 all PASS (V-2 with field-name waiver, V-5 / V-7 with pre-existing-unrelated annotations). Headline numbers in the Phase 3 Verification Outline above.
- Phase 4 (documentation): this memo finalized; `ARCHITECTURE.md` and `docs/RQ1_pipeline.md` §6.1 amended; `docs/fix1_thesis_handoff.md` created.
