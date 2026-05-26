# RQ1 — Tier × Surfacing Truth Table

**Artifact reference:** `results/rq1_metrics.json`, `results/reports/risk_scores.npz`
**Sample population:** n = 2,448 (frozen test split, IDS-HC-IoMT-2024)
**Last updated:** 2026-05-25

This table is the source of truth for what risk tier each sample lands in
and whether that tier triggers a surface-to-operator decision. It is the
ground truth that downstream evaluation references (Modules 5 / 6) and
the safety-floor invariant tests (Invariant 1, Invariant 2).

---

## 1. Tier boundary mapping

| Risk tier   | Composite R range (observed) | Surfacing action      | Operator visible |
|-------------|------------------------------|------------------------|------------------|
| LOW         | R < 0.30 (suppress band)     | suppress               | ✗ (audit log only) |
| MEDIUM      | 0.30 ≤ R < 0.45              | surface — informational | ✓ |
| HIGH        | 0.45 ≤ R < 0.65              | surface — actionable    | ✓ |
| CRITICAL    | R ≥ 0.65                     | surface — escalate      | ✓ |

> Note: the R boundaries above are *observed* on the test split. The
> canonical boundary definitions live in `src.risk_scorer`; this table
> is the empirical layout, not the policy spec.

## 2. Tier × ground-truth crosstab (test split, n=2,448)

| Tier     | Total | Attacks | Benign | Surface? | Notes |
|----------|------:|--------:|-------:|----------|-------|
| CRITICAL |    34 |      34 |      0 | ✓ surface | 100% attacks (perfect precision at tier) |
| HIGH     |   273 |     245 |     28 | ✓ surface | 89.7% attacks (28 FPs are HIGH-tier false alarms) |
| MEDIUM   |    85 |       9 |     76 | ✓ surface | 10.6% attacks (low precision — noise tier) |
| LOW      | 2,056 |      19 |  2,037 | ✗ suppress | 19 missed attacks at LOW — **all on non-critical devices** |
| **Total** | **2,448** | **307** | **2,141** | — | 12.5% attack prevalence |

### Derived counts

- **Surfaced (MED+):**  TP = 288, FP = 104, TN = 2,037, FN = 19
- **Sensitivity (surfacing):** 288 / 307 = **0.9381**
- **Specificity (surfacing):** 2,037 / 2,141 = **0.9514**
- **Precision (surfacing):** 288 / 392 = **0.7347**

## 3. Tier × device-criticality crosstab

| Tier     | All devices | Critical (d_crit≥0.8) | Non-critical (d_crit<0.8) |
|----------|------------:|----------------------:|--------------------------:|
| CRITICAL |          34 |                    34 |                         0 |
| HIGH     |         273 |                   104 |                       169 |
| MEDIUM   |          85 |                     0 |                        85 |
| LOW      |       2,056 |                     0 |                     2,056 |
| **Total** | **2,448** | **138** | **2,310** |

**Key invariant verified:** Every attack on a critical device (d_crit ≥ 0.8)
is surfaced (n=138, all in CRITICAL or HIGH tier). The safety floor
(Invariant 2) is empirically not needed on this split because the
detector already places life-critical-device attacks at MED+ tier;
the floor remains as defense-in-depth for edge cases not in the corpus.

## 4. The 19 LOW-tier missed attacks

All 19 are on **non-critical devices** (d_crit < 0.8). Breakdown:

- These are the only false negatives under the surfacing decision.
- None are on life-critical devices, so they do not contribute to
  FNR_critical (which is computed over `d_crit ≥ 0.8` attacks).
- Recovery path: the cascade re-baseline (Module 4 drift detection)
  catches distribution shift over time; per-alert recovery requires
  HITL review during periodic batch audit.

| Metric                        | Value     | Target  | Met? |
|-------------------------------|-----------|---------|------|
| FNR (overall)                 | 0.0619    | —       | n/a  |
| FNR_critical (d_crit ≥ 0.8)   | **0.000** | < 0.05  | ✓    |
| Sensitivity (surfacing)       | 0.9381    | > 0.90  | ✓    |
| Specificity (surfacing)       | 0.9514    | > 0.95  | ✓    |
| AUC (Track A)                 | 0.9947    | > 0.99  | ✓    |
| AUC (Track B)                 | 0.7569    | per-class breakdown | mixed |
| AUC (Composite R)             | 0.9838    | > 0.99  | ✗ (close) |

## 5. The 28 HIGH-tier benign samples (false alarms)

Of HIGH-tier samples, 28 / 273 = 10.3% are benign. These produce operator
alert volume. Under the response policy, HIGH-tier alerts route to a
T2 SOC analyst with no automatic action; the operator-time cost is
quantified in RQ3 (decision_time_sec metric on the HITL study split).

## 6. Auditability

For every sample in this table, the following are persisted:

- `results/reports/risk_scores.npz` — R, components, tier label
- `results/reports/alert_responses.json` — full response record (test split)
- `results/reports/audit_trail.json` — signed audit chain per alert

A reviewer can reproduce any row of this table by:

```bash
python3 -c "
import numpy as np
d = np.load('results/reports/risk_scores.npz', allow_pickle=True)
# e.g. all CRITICAL-tier attacks
import numpy as np
mask = (d['risk_levels'] == 'CRITICAL') & (d['y_true'] == 1)
print('n =', mask.sum())  # 34
"
```

---

## Appendix — How to read this table during defense

If a question asks *"What does the system do for tier X?"*:

- **CRITICAL** → surface immediately, route to T3 + biomed, audit-log
  required. Cannot be silently dismissed (Module 5 invariant C3).
- **HIGH** → surface to T2 SOC. Acknowledge/Escalate/Dismiss buttons
  enabled. Dismissal requires written rationale.
- **MEDIUM** → surface as informational. No mandatory action.
  Suitable for monitoring + investigation.
- **LOW** → suppress. Audit-logged but not surfaced.
  19 false negatives sit here — none on life-critical devices.

If a question asks *"What if d_crit is high but tier is LOW?"*:

In this split: empirically does not happen for attacks (all 138 attacks
on `d_crit ≥ 0.8` devices land in MED+ tier). The Module-5 safety floor
(Invariant 2) is the policy backstop if it ever does: it bumps such
alerts to HIGH regardless of composite score.
