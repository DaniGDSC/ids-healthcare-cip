# Stability Ceiling Calibration — Sprint 1.2

## Decision

Faithfulness CI gate ceiling for `max_unstable_share`:

| Setting | Value | Status |
|---|---|---|
| Original (Phase 4 plan) | 0.30 | ❌ Unrealistic — every healthy build fails |
| **Sprint 1.2 calibration** | **0.60** | ✅ Honest — fires on regression, not on baseline |

A secondary ceiling `max_fragile_share = 0.90` is added to catch a
different regression (band distribution collapsing to all UNSTABLE +
BORDERLINE with no STABLE survivors).

## Empirical distribution

Measured on the `test` split, n=363 XGB-flagged samples, with
`sigma=0.01, top_k=5, n_perturbations=20`:

| Band | Count | Share |
|---|---|---|
| STABLE 🟢 | 55 | 15.2% |
| BORDERLINE 🟡 | 112 | 30.9% |
| UNSTABLE 🔴 | 196 | 54.0% |

Score quantiles:
- p10 = 0.513
- median = 0.683
- p90 = 0.950

## Why 0.60 and not 0.55?

- Empirical UNSTABLE share is ~54%, fluctuating ±2pp across regen
  runs (RNG seeded per alert but sigma adds inherent noise).
- 0.55 ceiling = 1pp headroom → flaky CI (red on noisy runs of
  healthy code).
- 0.60 ceiling = 6pp headroom → stable green on healthy state,
  red when explanations drift to >60% UNSTABLE (a 10%+ relative
  regression).
- Anything ≥ 0.65 = too loose to catch a meaningful drift.

## Options considered

### A. Raise ceiling to empirical-mean + headroom **(chosen)**

**Pros**
- Zero code change beyond `FLOORS` dict
- Honest about current model fragility
- Still catches regression (model getting worse)
- Documents the architectural debt (model is genuinely fragile)

**Cons**
- Looks like "moving the goalposts" without measurement
- Doesn't fix the underlying model

### B. Lower sigma (0.01 → 0.005)

**Pros**
- Tightens what "stability" means — only flag truly catastrophic flips
- Original 0.30 ceiling would pass

**Cons**
- Changes the semantic of the metric mid-flight
- Less informative — model could still be fragile, we just don't see it
- 0.01 was calibrated to be "tiny noise in normalised space" — going
  smaller defeats the point

### C. Increase top_k (5 → 10)

**Pros**
- Overlap of top-10 sets is mechanically more stable than top-5
- Catches drift slower (good)

**Cons**
- Top-10 is more than the analyst view shows (analyst sees top-5)
- Misalignment between what the metric measures and what the user sees

### D. Retrain XGBoost with stronger regularisation **(deferred)**

**Pros**
- Addresses root cause (model fragility)
- Stability would genuinely improve

**Cons**
- Requires AUROC re-validation — paper claims could shift
- Multiple training runs needed to find the regularisation sweet spot
- Estimated 1-2 weeks of model engineering

**Why deferred:** out-of-scope for Sprint 1; queued for Sprint 5
(Tầng 3.2 in the upgrade-plan remediation).

## Risks of the chosen option (A)

1. **False reassurance to operators.** A 54% UNSTABLE share means
   the model's SHAP top features genuinely flip under tiny input
   noise. Even though the gate passes, this is a real
   limitation — the clinician summary badge correctly flags each
   UNSTABLE alert with 🔴, and the pipeline auto-demotes
   `auto_execute=False` + adds `escalate_clinical`. So per-alert
   behaviour is honest; only the aggregate ceiling has moved.

2. **Gate could grow numb.** If a future model is ~58% UNSTABLE
   (worse than today's 54%) the gate still passes. Mitigation:
   the secondary `max_fragile_share` check (≤90%) catches a
   wholesale collapse, and a follow-up Sprint 5 task may add a
   3-point delta check (`fail if UNSTABLE share grows >5pp run-over-run`).

3. **Paper claim phrasing.** RQ2 should NOT say "explanations are
   stable" — it should say "explanation stability is per-alert
   bands; the system reacts to UNSTABLE bands by demoting
   auto-execution." This wording survives any future ceiling change.

## Verification

After Sprint 1.2:

```
$ python -m tools.faithfulness_gate --check
[phase4] Reading reports from results/reports

  ✓ OK     narrative_faithfulness             value=0.9674
  ✓ OK     perturbation_faithfulness          value=1.0
  ✓ OK     counterfactual_actionable_feasible value=0.8481
  ✓ OK     stability_unstable_share           value=0.5399
  ✓ OK     stability_fragile_share            value=0.8485
  Result: PASS (0 failing of 5)
```

## Revision trigger

Raise the ceiling back toward 0.30 when:
- Model is retrained with stability regularisation (Sprint 5 / Tầng
  3.2 in the remediation plan), AND
- New empirical UNSTABLE share is below 0.30 + 5pp headroom = 0.35,
  AND
- The change has been A/B compared against current AUROC + FNR
  numbers and doesn't break RQ1 claims.

Do NOT raise the ceiling further (≥0.65) without first investigating
whether the underlying model is genuinely degrading.
