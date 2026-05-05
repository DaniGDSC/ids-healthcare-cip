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

### Track B — Overall Metrics

| Metric | Value |
| --- | --- |
| Threshold | 6.42 × 10⁻⁵ (95th percentile of benign reconstruction error) |
| Accuracy | 0.8770 |
| Precision (attack) | 0.5120 |
| Recall (attack) | 0.4186 |
| F1 (attack) | 0.4606 |
| F2 (attack) | 0.4344 |
| F1 macro | 0.6956 |
| F1 weighted | 0.8717 |
| ROC AUC | **0.7143** |

### Track B — Confusion Matrix

| TP | FN | FP | TN | FNR | FPR |
| --- | --- | --- | --- | --- | --- |
| 257 | 357 | 245 | 4,037 | **0.5814** | 0.0572 |

### Track B — Reconstruction-Error Distribution

| Class | Mean reconstruction error |
| --- | --- |
| Benign | 1.470 × 10⁻⁵ |
| Attack | 1.473 × 10⁻⁴ |
| Separation ratio | ≈ 10× |

The means differ by an order of magnitude, but the distributions overlap enough that the 95th-percentile threshold yields FNR=58%. This is the expected behaviour for a benign-only autoencoder on the WUSTL test set; the real value of Track B comes from the **cascaded fusion** with Track A, not from running it alone.

### Track B — Threshold-Dependency Curve

**Not yet computed.** A sweep over `threshold_percentile ∈ {80, 85, 90, 92.5, 95, 97.5, 99}` would produce the (FPR, FNR, F1, F2) curve. Run script not present; tracked as **GAP-PB-2**. Recommended location: `module2_detection/tuning/run_dae.py --sweep-threshold`.

### Track B — Per Attack-Category and Per Device-Class FNR

**Not directly computed in current artifacts.** From the fusion quadrant table:

- `only_dae` quadrant: 96 samples, **all benign** — DAE alone produced zero true-positive flags that XGBoost missed.
- `both_flag` quadrant: 144 samples, all attacks (142 Data Alteration + 2 Spoofing).

This means Track B's standalone attack detection on the test set is captured entirely inside the `both_flag` quadrant, contributing 144/614 = 23.5% recall — matching the `recall.dae_alone = 0.2345` field in `risk_report.json`. Per-device-class FNR is blocked on the same `device_class` join described in GAP-PB-1.

---

## Cascaded Fusion (Module 3)

Fusion rule: `c_detect = max(c_track_a, c_track_b)` — DAE elevates, never suppresses.

### Fusion Quadrant (XGBoost vs DAE)

| Quadrant | Samples | True attacks | True benign |
| --- | --- | --- | --- |
| Both flag (TP for both) | 144 | 144 | 0 |
| Only XGBoost flags | 533 | 432 | 101 |
| Only DAE flags | 96 | 0 | 96 |
| Neither flags | 4,123 | 38 | 4,085 |

### Fusion Recall Comparison

| Strategy | Recall |
| --- | --- |
| XGBoost alone | 0.9381 |
| DAE alone | 0.2345 |
| **Union fusion (max)** | **0.9381** |
| Fusion gain | 0.0000 |

**On this test set, Track B contributes no recall gain over Track A** because every attack the DAE catches (the 144 in `both_flag`) is also caught by XGBoost. The DAE's value remains the spoofing-defence rationale (cascaded input includes `[raw || P_xgb, P_rf, P_dt]` so a classifier-aware spoofing attack still triggers high reconstruction error), but on the WUSTL static test split the cascade does not improve recall numerically. Tracked as **GAP-PB-3** — re-evaluate fusion gain after a spoofing-augmented test set is produced.

---

## Module 3 Risk Score (`R = 0.40·C_detect + 0.25·D_crit + 0.15·S_data + 0.20·A_patient`)

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
