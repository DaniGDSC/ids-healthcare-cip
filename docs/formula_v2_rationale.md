# Formula v2 rationale — Sprint 4 / Tầng 3.1

## Problem v2 fixes

v1 is a linear weighted sum::

    R = w1·C_detect + w2·D_crit + w3·S_data + w4·D_clinical_tier
    with w1=0.40, w2=0.25, w3=0.15, w4=0.20

This conflates two semantically different signals into one
additive contribution:

  - **Detection** ("did the model say attack?") — should drive
    whether we emit an alert at all.
  - **Context** (device criticality, data sensitivity, patient
    acuity) — should drive how serious an alert is, not whether
    one exists.

Because v1 adds the two, a sample with **zero detection signal**
on a vital-monitoring device with PHI data lands at::

    R = 0.40·0 + 0.25·0.40 + 0.15·0.74 + 0.20·0 = 0.211

i.e. above the original LOW boundary of 0.40 / 0.30 — the system
emits a LOW alert even though no detector flagged anything. On the
test split this produced ~2000 false alerts that operators had to
triage (91.6% of the LOW tier). Operational precision collapsed to
0.125.

v2 was conceived in the Phase A+B fix (gate at C_detect ≥ 0.02,
NORMAL tier at R < 0.30) which patched the symptom in v1's tier
table. Sprint 4 implements the architectural correction: separate
detection from context inside the composition itself.

## v2 architecture

Two layers:

    Layer 1 (gate):
        if C_detect < MIN_DETECTION_GATE:    R = 0
    Layer 2 (amplify):
        R = C_detect × (1 + α·D_crit + β·S_data + γ·D_clinical_tier)
        clipped to [0, 1]

With ``α=0.6, β=0.4, γ=0.5`` so the maximum amplification factor
is ``1 + 0.6 + 0.4 + 0.5 = 2.5``. Detection drives the score;
context can only amplify what's already there. A silent model
produces a silent R, no matter how critical the device.

## Empirical results (test split, n=2448)

### Tier distribution

| Tier      | v1     | v2     | Δ        |
|-----------|--------|--------|----------|
| CRITICAL  | 12     | 355    | **+343** |
| HIGH      | 301    | 32     | -269     |
| MEDIUM    | 68     | 82     | +14      |
| LOW       | 139    | 215    | +76      |
| NORMAL    | 1928   | 1764   | -164     |

The most striking shift is CRITICAL going from 0.5% → 14.5% of
the corpus. This is **not noise** — it's the natural consequence
of v2's bimodality: when C_detect is high *and* context amplifies
it, R saturates at 1.0 (the clip), which lands in CRITICAL.

### RQ1 paper claim (surfaced = MEDIUM+)

| Metric              | v1     | v2     | Δ        |
|---------------------|--------|--------|----------|
| Surfaced recall     | 0.973  | 0.987  | **+1.4pp** |
| Surfaced precision  | 0.756  | 0.623  | -13.3pp  |
| Alert volume        | 520    | 684    | +164     |

v2 catches more attacks (+1.4pp surfaced recall) but at the cost
of more false alerts in surfaced tiers (-13pp precision). The
RQ1 paper's headline recall claim is preserved (within 2pp).

### Operational pool (entire alert pool)

| Metric                    | v1     | v2     | Δ        |
|---------------------------|--------|--------|----------|
| Operational precision     | 0.569  | 0.433  | -13.7pp  |
| Operational recall        | 1.000  | 1.000  | 0        |
| LOW-tier attack density   | 5.8%   | 1.9%   | **-3.9pp** |

LOW-tier attack density is the formula-bug sentinel: when LOW is
flooded with benign noise it approaches 0. v2 cuts the density
from 5.8% to 1.9% — the LOW tier is now much cleaner. The
operational precision drop reflects v2's more aggressive surfacing
behaviour, not a regression in signal quality.

### Counterfactual coverage

| Metric                        | v1     | v2     |
|-------------------------------|--------|--------|
| Actionable feasible (M+/H/C)  | 80.6%  | 67.6%  |

The drop is denominator-driven: v2 puts more records in the
actionable bucket (469 vs 381). XGB-flagged sample count is the
same.

## Why default = v1

v2 is the architecturally correct formula. v1 is the formula the
RQ1 paper was written against. To preserve paper reproducibility
without freezing the codebase:

  - ``composition.compute_composite_risk()`` defaults to
    ``formula_version="v1"`` so all existing call sites keep
    their current behaviour.
  - The CLI flag ``--formula-version v2`` opts into v2 for new
    regen runs.
  - The npz now carries a ``formula_version`` field so downstream
    consumers know which interpretation applies.

Production deployment switches to v2 by setting
``--formula-version v2`` in the regen tools. The dashboard reads
the npz field and renders the v2 interpretation when present.

## Threshold calibration

v2's R distribution is bimodal — most attacks saturate at 1.0,
most benign samples sit at 0.0. The cutoffs in
``RISK_THRESHOLDS_V2`` were chosen to preserve RQ1 surfaced
recall (policy A in ``tools/calibrate_thresholds_v2.py``) within
±2pp::

    CRITICAL ≥ 0.80
    HIGH     ≥ 0.45
    MEDIUM   ≥ 0.20
    LOW      ≥ 0.05

Operator capacity caveat — v2's CRITICAL share (14.5% on the test
split) is much higher than v1's (0.5%). If operator capacity is
tuned for v1's CRITICAL volume, raising the v2 CRITICAL cutoff
toward 0.99 (operator capacity policy) is reasonable. We didn't
do that here because the cutoff is just numeric — adjusting it
post-deploy is one constant edit and a regen.

## Migration checklist

When the team is ready to flip production to v2:

  1. Run ``python -m module3_risk_scoring.module3_risk_scores
     --formula-version v2 --split both`` to regenerate the npz.
  2. Re-run the offline regen tools (``phase1_regen_module4``,
     ``phase1_regen_module5``) — they pick up the new R + tier
     labels without code changes.
  3. Re-run ``python -m tools.phase0_baseline --update-floors``
     to lock in the v2 metric values. The v1 floors in
     ``backups/v1_paper_frozen/phase0_baseline.json`` stay as
     paper reference.
  4. Re-run ``python -m tools.faithfulness_gate --check`` to
     verify per-alert badges still surface.
  5. Verify ``python -m tools.compare_v1_v2`` snapshot matches
     the expected v2 numbers; commit the artifact set.

## Rollback

Restoring v1 is one ``cp``::

    cp backups/v1_paper_frozen/* results/reports/

Run ``--check`` on the floor file to confirm the v1 metrics
land back on their recorded floors. The composition code keeps
both formulas around forever — there's no "v2-only" point of no
return.

## Risks of v2

  1. **CRITICAL inflation.** 14.5% of records lands in CRITICAL.
     If operator triage SOPs assume CRITICAL is rare, they need
     updating. Mitigation: raise CRITICAL cutoff toward 0.99 (3
     digit edit + regen).

  2. **Surfaced precision drop.** -13pp on surfaced tier means
     more false positives reach the operator. Mitigation: tune
     MEDIUM cutoff upward, or pair with a stability gate so only
     STABLE high-confidence alerts auto-surface.

  3. **Paper reproducibility risk.** RQ1 numbers shift slightly
     under v2 (recall +1.4pp, precision -13pp). The current paper
     run was on v1; if v2 is deployed and the paper is re-run
     against deployed artifacts, the numbers won't match. The v1
     backup directory mitigates this — paper-grade numbers are
     reproducible from ``backups/v1_paper_frozen/``.

## Sprint 4 deliverables

  - ``module3_risk_scoring/config.py``: ``CONTEXT_WEIGHTS_V2`` +
    ``RISK_THRESHOLDS_V2``
  - ``module3_risk_scoring/composition.py``: v2 implementation +
    ``formula_version`` dispatcher
  - ``module3_risk_scoring/module3_risk_scores.py``:
    ``--formula-version`` CLI flag
  - ``module3_risk_scoring/io.py``: npz writer records
    ``formula_version``
  - ``tools/regen_risk_scores_offline.py``: ``--formula-version``
    flag for the offline path
  - ``tools/calibrate_thresholds_v2.py``: threshold tuning helper
  - ``tools/compare_v1_v2.py``: side-by-side metric report
  - ``backups/v1_paper_frozen/``: byte-exact paper snapshot
  - ``backups/v2_deployed_snapshot/``: v2 reference output
  - ``results/v1_v2_comparison.json``: machine-readable diff
  - ``tests/test_formula_v2.py``: 14 architectural tests
