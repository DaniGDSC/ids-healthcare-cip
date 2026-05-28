# Round-2 User Study Protocol — Sprint 5 / Tầng 3.3

## Background

The Round-1 study (recorded in `tools/rq2_user_study_analysis.py`)
catalogued failure modes against the *pre-upgrade* output set: SHAP
narrative with no observation phrase, generic actions with no
extension/SLA, no counterfactual, no playbook, no stability badge.

The upgrade-plan execution (Phases 1.1 / 1.2 / 1.4 / 2 / 3.1 / 3.2 /
4.1) committed to addressing those failure modes. Sprint 5 / Tầng
3.3 is the round-2 study that measures whether the additions
*actually* shift comprehensibility / actionability / trust / accuracy
ratings as the simulator predicted.

This document is the **protocol**. The actual recruitment + IRB +
data collection happens outside the codebase — but every parameter
the study needs is committed to here so the analysis runs without
re-litigating the design.

## Study design

### Participants

| Role         | n  | Recruitment channel                          |
|--------------|----|----------------------------------------------|
| Analyst      | 8  | Hospital SOC / IT-security mailing list      |
| Clinician    | 8  | ICU charge-nurse + on-call physician roster  |
| Administrator| 6  | Biomed engineering + compliance officers     |

22 total. Each participant rates 20 alerts (same alert set across
participants for paired comparison).

### Conditions

- **Baseline**: pre-upgrade artifacts from `backups/v1_paper_frozen/`
  (recorded mid-upgrade so the baseline is itself faithful to the
  artifact state before the enrichments started landing).
- **Enriched**: current production artifacts after Sprint 1-4 land.

Each participant sees 10 alerts in each condition, randomised order,
between-subjects per alert. The same 20 alerts are sampled in
proportion to the production severity distribution::

    CRITICAL  1   (5%)
    HIGH      9   (45%)
    MEDIUM    2   (10%)
    LOW       3   (15%)
    NORMAL    5   (25%, presented but labelled as "logged, no action")

Sample IDs are pinned in `survey/round2_alert_ids.json` (to be
created when the study is scheduled).

### Tasks per alert

For each alert the participant answers:

1. **Decision task** — pick one action from a fixed menu of 5
   (isolate, restrict, escalate clinical, log+monitor, no action).
   Ground truth: derived from `response.actions`.
2. **Confidence** — Likert 1-5.
3. **Comprehensibility** — Likert 1-5 ("I understood why the system
   flagged this alert").
4. **Trust** — Likert 1-5 ("I would act on this without further
   verification").
5. **Actionability** — Likert 1-5 ("The next step was clear from
   the explanation").
6. **Free-text feedback** — optional, anonymised.

Time-to-decision is captured automatically.

### Primary outcomes

| Metric | Source | Target lift |
|---|---|---|
| Decision correctness | derived       | +5pp on clinician, +0pp on analyst (baseline already high) |
| Comprehensibility    | Likert mean   | +0.5 point |
| Trust                | Likert mean   | +0.3 point |
| Actionability        | Likert mean   | +0.5 point |
| Time-to-decision     | seconds       | -10s on clinician, -5s on admin |

All targets are pre-registered. The study **does not** look at
analyst time-to-decision as a primary — analysts read SHAP regardless,
the enrichments target non-ML audiences.

### Secondary outcomes (exploratory)

- Per-tier confidence delta (CRITICAL vs LOW)
- Counterfactual-driven action attempt rate (did the operator
  choose the `try_first_action` over the prescribed isolation?)
- Playbook step adherence
- Routing-warning compliance (did the operator reroute when flagged?)

## Statistical analysis

- Mixed-effects model: rating ~ condition + (1 | participant) +
  (1 | alert).
- Bonferroni-correct across the 5 Likert items; primary endpoints
  pre-registered above.
- Bootstrap CIs (1000 iter) on every paired mean delta.
- Effect sizes (Cohen's d) reported per primary.

## Simulated round-2 results

While the real study is scheduled, the **simulator** (Phase 4.3 in
the upgrade-plan implementation) gives a point-estimate prediction
based on each participant's likely response to the surface features
they actually see.

Simulator output (last run on the v1 production artifact):

| Metric              | baseline | enriched | Δ      | predicted vs target |
|---------------------|----------|----------|--------|---------------------|
| Decision correctness| 0.74     | 0.85     | +0.11  | exceeds (+5pp target)|
| Comprehensibility   | 3.10     | 3.95     | +0.85  | exceeds (+0.5)      |
| Trust               | 3.40     | 3.85     | +0.45  | exceeds (+0.3)      |
| Actionability       | 3.05     | 4.10     | +1.05  | exceeds (+0.5)      |
| Time-to-decision    | 38s      | 24s      | -14s   | exceeds (clin -10s) |

(Source: re-run `tools/rq2_user_study_analysis.py` after the v1
enrichment regen.)

## Limitations of the simulator

The simulator's "comprehensibility" is keyed on whether the
expected surface artefact (observation phrase, MITRE gloss,
playbook, etc.) is *present* in the rendered output. It does NOT
model whether the participant actually parses the text correctly.
Real humans:

- Skim playbooks (don't follow every conditional step)
- Anchor on the headline severity tier and ignore the body
- Override prescribed actions based on personal experience
- Get fatigue after the first ~5 alerts

The simulator overstates the effect by ~15-25pp in our experience.
Round-2 numbers will likely look like the simulator's prediction
*divided by ≈ 1.2*.

## Real-study scheduling

The actual round-2 study is gated on:

1. **IRB approval** — pending board review, est. 6-8 weeks
2. **Recruitment confirmation** — 22 participants across 3 sites
3. **Stipend funding** — $50/participant × 22 = $1,100
4. **Pre-registration** — OSF registry once the IRB greenlight lands

Until those are complete, the simulator's prediction is the only
empirical evidence we can cite. The simulator agrees with v1
(round-1) on the *baseline* numbers — it correctly reproduces the
74% decision correctness and 3.10 comprehensibility that the
round-1 study measured. This gives modest confidence its
prediction for the enriched condition is also in the right
ballpark.

## What to do until the study runs

1. Treat the simulator output as **upper-bound** on real effect.
2. Phrase paper claims as "the simulator predicts" or "we estimate";
   do NOT report the predicted values as observed.
3. The CI gates (`tools/faithfulness_gate.py`,
   `tools/phase0_baseline.py`, `tools/coverage_audit.py`) operate
   on the artefact level and don't need human ratings. They
   continue to enforce the enrichment-coverage invariants the
   upgrade-plan committed to.
4. Document this state in the paper's "limitations" section: the
   per-stakeholder UX improvement is **inferred from a simulator
   validated against round-1 baseline numbers, with real-human
   round-2 measurement pending IRB approval**.

## Sprint 5 deliverables

- This protocol document (`docs/user_study_protocol.md`)
- `survey/round2_alert_ids.json` placeholder (to be filled once
  sample selection is finalised)
- Simulator re-run reproducible via
  `python -m tools.rq2_user_study_analysis`
- Round-1 baseline numbers re-validated against simulator (pin in
  `results/round2_simulator_predictions.json`)
