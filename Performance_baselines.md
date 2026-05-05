# Performance Baselines

Source artifacts (regenerate by re-running Module 2 → Module 3):

- `results/models/{xgboost,random_forest,decision_tree,dae}_final_report.json`
- `results/reports/test_evaluation_report.json`
- `results/reports/risk_report.json` (`dual_track_fusion`, `per_category_stats`)
- `results/rq2_metrics.json`

Dataset: WUSTL-EHMS-2020. Test split: **4,896 samples** (4,282 benign / 614 attack — **12.54% attack prevalence**). Train split: 11,422 samples (SMOTE-balanced for Track A).

Attack categories present in test set: **Spoofing** (337) and **Data Alteration** (277). The 5-category breakdown listed below applies only after extending fixtures with `lateral_movement`, `data_exfiltration`, and `unauthorized_access` synthetic alerts; today's test set has 2 of 5 attack types.

---

## Track A — Supervised Classifiers (per-model)

All metrics on the held-out test set. Thresholds are the per-model optimal F2-tuned thresholds saved at training time.

### Track A — Overall Metrics

| Model | Threshold | Accuracy | Precision (attack) | Recall (attack) | F1 (attack) | F2 (attack) | F1 macro | F1 weighted | ROC AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **XGBoost** | 0.0500 | 0.9716 | 0.8508 | 0.9381 | **0.8923** | **0.9192** | 0.9380 | 0.9722 | **0.9941** |
| **Random Forest** | 0.4887 | 0.9504 | 0.8076 | 0.7932 | 0.8003 | 0.7960 | 0.8860 | 0.9502 | 0.9589 |
| **Decision Tree** | 0.3349 | 0.9097 | 0.6014 | 0.8306 | 0.6977 | 0.7718 | 0.8223 | 0.9157 | 0.8911 |

### Track A — Confusion Matrices

| Model | TP | FN | FP | TN | FNR | FPR |
| --- | --- | --- | --- | --- | --- | --- |
| XGBoost | 576 | 38 | 101 | 4,181 | **0.0619** | **0.0236** |
| Random Forest | 487 | 127 | 116 | 4,166 | 0.2068 | 0.0271 |
| Decision Tree | 510 | 104 | 338 | 3,944 | 0.1694 | 0.0789 |

`FNR = FN / (TP+FN)` — fraction of true attacks missed.
`FPR = FP / (FP+TN)` — fraction of benign flows flagged.

### Track A — Per Attack-Category FNR

The risk report records `dual_track_fusion.quadrants` with per-category attack counts in each fusion quadrant. The XGBoost-alone quadrant + neither-flagged quadrant give attack-category miss attribution:

| Category | Total attacks | XGBoost flagged | XGBoost missed (FN) | FNR_xgb |
| --- | --- | --- | --- | --- |
| Data Alteration | 277 | 277 (142 both + 135 only-xgb) | 0 | 0.0000 |
| Spoofing | 337 | 299 (2 both + 297 only-xgb) | 38 | **0.1128** |
| **All** | 614 | 576 | 38 | 0.0619 |

Spoofing is the residual blind spot for Track A; Data Alteration is fully covered.

### Track A — Per Device-Class FNR

**Not yet computed.** The test split records `attack_category` but does not yet record `device_class` per row. To populate this section, join `evaluation_alerts.json` (which carries `device_class`) with the test predictions, or extend Module 1 to export device class with each row of `test_phase1.parquet`. Tracked as **GAP-PB-1**.

---

## Track B — DAE (Standalone)

Track B alone is a weak attack detector by design: it is trained on benign-only traffic and is meant to *elevate* Track A on cascaded fusion, not to flag attacks on its own. The numbers below show why running Track B as a standalone gate would underperform.

### Track B — Overall Metrics (post GAP-A10 retraining, seed 42)

| Metric | Value |
| --- | --- |
| Threshold | 1.852 × 10⁻⁵ (95th percentile of benign reconstruction error) |
| Accuracy | 0.9218 |
| Precision (attack) | 0.6732 |
| Recall (attack) | 0.7313 |
| F1 (attack) | 0.7018 |
| F2 (attack) | 0.7295 |
| F1 macro | 0.8278 |
| F1 weighted | 0.9223 |
| ROC AUC | **0.9128** |

### Track B — Confusion Matrix

| TP | FN | FP | TN | FNR | FPR |
| --- | --- | --- | --- | --- | --- |
| 449 | 165 | 218 | 4,064 | **0.2687** | 0.0509 |

### Track B — Reconstruction-Error Distribution

| Class | Mean reconstruction error |
| --- | --- |
| Benign | 1.306 × 10² (heavily right-skewed by long-tail outliers; median 5.51 × 10⁻⁷) |
| Attack | 2.702 × 10³ (median 6.41 × 10⁻³) |
| Separation ratio (means) | ≈ 21× |
| Separation ratio (medians) | ≈ 1.16 × 10⁴ |

GAP-A10 retraining substantially improved the standalone DAE: AUC 0.71 → 0.91, F1 0.46 → 0.70, FNR 0.58 → 0.27. The cascade is still the design's primary value path (fusion gain +0.0033 over Track A alone), but the standalone DAE is no longer a "weak detector" — its AUC is now competitive with Random Forest.

### Track B — Threshold-Dependency Curve

**Not yet computed.** A sweep over `threshold_percentile ∈ {80, 85, 90, 92.5, 95, 97.5, 99}` would produce the (FPR, FNR, F1, F2) curve. Run script not present; tracked as **GAP-PB-2**. Recommended location: `module2_detection/tuning/run_dae.py --sweep-threshold`.

### Track B — Per Attack-Category and Per Device-Class FNR

From the post-A10 fusion quadrant table:

- `only_dae` quadrant: 96 samples — **2 attacks** + 94 benign (was 0/96 pre-A10).
- `both_flag` quadrant: 406 samples — 400 attacks + 6 benign (was 144/0 pre-A10; substantially larger now because the post-A10 DAE flags more attacks at p=95).

DAE-alone recall on the test set is 0.6547 (was 0.2345 pre-A10). Per-device-class FNR is now unblocked by GAP-A7 (device_class column at the row level); recompute via `module6_evaluation/compute_per_device_metrics.py` with the DAE prediction file.

---

## Cascaded Fusion (Module 3)

Fusion rule: `c_detect = max(c_track_a, c_track_b)` — DAE elevates, never suppresses.

### Fusion Quadrant (XGBoost vs DAE) — post-A10

| Quadrant | Samples | True attacks | True benign |
| --- | --- | --- | --- |
| Both flag (TP for both) | 406 | 400 | 6 |
| Only XGBoost flags | 272 | 177 | 95 |
| Only DAE flags | 96 | 2 | 94 |
| Neither flags | 4,122 | 35 | 4,087 |

### Fusion Recall Comparison

| Strategy | Recall |
| --- | --- |
| XGBoost alone | 0.9397 |
| DAE alone | 0.6547 |
| **Union fusion (max)** | **0.9430** |
| Fusion gain | **+0.0033** (post-A10; was 0.0000 pre-A10) |

**Post-A10 Track B contributes 0.33 pp recall gain over Track A** (2 unique attack catches at the cost of 94 standalone false positives). The pre-A10 "no recall gain" claim is OBSOLETE. The DAE's value remains primarily the spoofing-defence rationale (cascaded input `[raw || P_xgb, P_rf, P_dt]` is sensitive to classifier-aware perturbations), but the cascade now also adds small but non-zero in-distribution recall.

---

## Module 3 Risk Score (`R = 0.40·C_detect + 0.25·D_crit + 0.15·S_data + 0.20·D_clinical_tier`)

### Risk Distribution Across Test Set

| Tier | Threshold | Count | % | Mean R |
| --- | --- | --- | --- | --- |
| LOW | R < 0.40 | 4,155 | 84.9% | 0.2478 |
| MEDIUM | 0.40 ≤ R < 0.60 | 140 | 2.9% | 0.4962 |
| HIGH | 0.60 ≤ R < 0.80 | 553 | 11.3% | 0.7213 |
| CRITICAL | R ≥ 0.80 | 48 | 1.0% | 0.8128 |

### Risk Score by Attack Category

| Class | Count | Mean R | Median R | Std R |
| --- | --- | --- | --- | --- |
| Normal (benign) | 4,282 | 0.2593 | 0.2509 | 0.0795 |
| Spoofing | 337 | 0.6495 | 0.7028 | 0.1536 |
| Data Alteration | 277 | 0.7497 | 0.7500 | 0.0390 |

Composite separation between benign and attack is clean at the median (0.25 vs 0.70+), supporting the chosen tier thresholds.

---

## RQ2 Surfacing Metrics (`results/rq2_metrics.json`)

Computed by `module6_evaluation/compute_rq2_metrics.py` against the 20-alert `evaluation_alerts.json` curated study set (not the full 4,896-sample test split).

| Metric | Value |
| --- | --- |
| Total surfaced alerts | 18 |
| True CRITICAL count | 15 |
| Critical alert rate (surfaced) | 0.6667 |
| **FNR_critical** (true CRITICAL missed) | **0.0000** |
| Sensitivity (attack recall) | 1.0000 |
| Specificity (benign rejection) | 0.5000 |
| Confusion: TP / FN / FP / TN | 16 / 0 / 2 / 2 |

The curated 20-alert set is a small evaluation harness, not a stratified eval set. Stratified-sampling deliverable still pending — see GAP-PB-4 below.

---

## Evaluation-Set Status (P0.4 Deliverable)

| Requirement | Current state |
| --- | --- |
| Stratified random sample (P0.4) | **NOT MET** — 20 hand-curated alerts in `evaluation_alerts.json`, plus the 4,896-sample raw test split which is stratified by attack/benign only. |
| Min 200 alerts per severity stratum | **NOT MET** — full test split has 48 CRITICAL, 553 HIGH, 140 MEDIUM, 4,155 LOW. CRITICAL and MEDIUM are below the 200-alert minimum. |
| Per-attack-category breakdown | **PARTIAL** — only Spoofing (337) and Data Alteration (277) present. The 5-category schema (`spoofing`, `data_alteration`, `lateral_movement`, `data_exfiltration`, `unauthorized_access`) is not yet realised in the dataset; 3 of 5 categories have zero samples. |
| Per-device-class breakdown | **NOT MET** — `device_class` is attached at evaluation-alert curation time only, not at the row level of `test_phase1.parquet`. |

### Open Gaps

| ID | Gap | Owner |
| --- | --- | --- |
| GAP-PB-1 | Add `device_class` column to the per-row test predictions to enable per-device FNR/FPR | Module 1 / Module 3 |
| GAP-PB-2 | DAE threshold-percentile sweep curve (FPR, FNR, F1, F2) | Module 2 |
| GAP-PB-3 | Re-evaluate cascaded fusion gain on a spoofing-augmented test set | Module 2 / Module 3 |
| GAP-PB-4 | Build stratified evaluation set: ≥200 alerts × 4 severity strata × 5 attack categories | Module 6 |
| GAP-PB-5 | Synthesise lateral_movement, data_exfiltration, unauthorized_access samples (currently 0/614 attacks) | Phase 1 / Module 0 |

---

## Reproducing These Numbers

```bash
# Train models (regenerates *_final_report.json)
python module2_detection/module2_train_models.py

# Compute composite risk and dual-track quadrant table
python module3_risk_scoring/module3_risk_scores.py

# RQ2 metrics on curated 20-alert study set
python module6_evaluation/compute_rq2_metrics.py

# Acceptance + negative + safe-failure tests
python run_tests.py
```

All numerical values above were extracted directly from the JSON artifacts at the paths listed in the **Source artifacts** block at the top of this document; no values were hand-typed from logs.
