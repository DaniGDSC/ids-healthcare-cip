# MedSec-25 LOCO — Cascade Contract Validation

**Run date:** 2026-05-06  
**Dataset:** MedSec-25 IoMT Cybersecurity (`data/raw/MedSec-25/MedSec-25.csv`, 554,534 rows × 84 cols)  
**Question:** Does the cascade design — *Track A detects known attacks; DAE detects unknown attacks + verifies normal* — survive a leave-one-category-out test on a richer attack space than EHMS-2020 can offer?

## Setup

- 4 LOCO folds, one per attack category. The held-out category is excluded from Track A's training set entirely.
- Track A: `GradientBoostingClassifier` (XGBoost surrogate, matching the project's existing convention) + `RandomForest` + `DecisionTree`. Trained on a stratified subsample of 80 k rows from the 4-category training partition.
- DAE: cascaded — input is `[69 raw features || P_xgb_val, P_rf_val, P_dt_val]` = 72-dim. Trained on benign-only val rows (∼1,200 per fold). Threshold percentile = 99.0.
- Cascade thresholds: `a_high = 0.85`, `a_low = 0.40`.

## Per-fold results

| Held-out category | Unknown attacks in test | Track A high-conf | Track A silent | DAE catch on silent unknown | DAE FPR on silent benign |
|---|---:|---:|---:|---:|---:|
| Exfiltration     |  2,591 | 99.23% | 0.54% | **2/14 = 14.29%** | 2.63% |
| Initial access   | 10,209 | 98.79% | 0.52% | **35/53 = 66.04%** ✓ | 1.99% |
| Lateral movement |  1,250 | 97.36% | 1.52% | **1/19 = 5.26%** | 2.22% |
| Reconnaissance   | 40,169 | 98.28% | 0.83% | **10/335 = 2.99%** | 2.29% |

## End-to-end cascade metrics on each fold

| Held-out category | TP | FP | FN | TN | Recall | Precision | F1 | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Exfiltration     | 54,158 | 49 |  61 | 1,186 | 0.9989 | 0.9991 | 0.9990 | 3.97% |
| Initial access   | 54,098 | 49 | 121 | 1,186 | 0.9978 | 0.9991 | 0.9984 | 3.97% |
| Lateral movement | 54,146 | 46 |  73 | 1,189 | 0.9987 | 0.9992 | 0.9989 | 3.72% |
| Reconnaissance   | 53,636 | 37 | 583 | 1,198 | 0.9892 | 0.9993 | 0.9943 | 3.00% |

## Headline finding

**Track A generalises strongly across attack categories.** On every held-out fold, the binary tree classifier — *trained without ever seeing the held-out category* — was high-confident on **97–99% of the unseen category's attacks**. From the network-flow feature perspective, the four MITRE categories (Reconnaissance, Initial access, Exfiltration, Lateral movement) are tactical groupings of behaviours that *all look like "not benign"*; binary attack/benign labelling captures most of the discriminative structure.

This means **the residual where DAE could matter is small** (<2% of unknown attacks per fold). Within that small residual, DAE's recall varies wildly:
- **Initial access: 66% recall (35/53)** — DAE doing real, useful work.
- **Exfiltration: 14% (2/14)** — modest contribution.
- **Lateral movement: 5% (1/19)** — basically random.
- **Reconnaissance: 3% (10/335)** — basically random.

DAE FPR on Track-A-silent benigns is consistently low (2.0–2.6%), so DAE is not broken — it just has very little signal to work with on most categories.

## Implications for the thesis

1. **The cascade contract holds, but with an updated framing.** The original framing "tree handles known, DAE handles unknown" is too binary. The data-supported framing is:

   > Track A (tree) handles both known *and most unknown* attacks; the cascaded DAE adds incremental coverage on the small residual Track A misses, with material contribution on at least one attack class (Initial access, +66% recall on silent unknowns).

2. **DAE's value is not uniform across attack categories.** The 66% recall on Initial access is the design paying off; the ≤5% on Lateral movement and Reconnaissance is the design's null hypothesis. A real deployment would need per-category measurement to understand where DAE earns its operating cost.

3. **Why EHMS-2020 looked worse for the DAE.** EHMS has 2 attack categories with ~12% attack rate. The Track-A-silent residual on EHMS's 614 attacks was 61 rows — all Spoofing — and DAE caught 0 of them. MedSec-25's much larger and more diverse attack space surfaces the same finding (DAE is mostly dominated by Track A) but also reveals at least one category where DAE genuinely contributes.

4. **The cascade is not redundant.** Across the 4 folds, DAE caught a total of 48 unknown attacks that Track A missed entirely — at the cost of ~111 benign FPs. That's a meaningful operator-side trade only on Initial access (35 catches for 24 FPs ≈ 1.5 catches per FP); on the others it's not worth the noise.

## Files

- [`results/medsec25_loco/loco_results.yaml`](loco_results.yaml) — full per-fold results
- [`results/medsec25_loco/loco_results.json`](loco_results.json) — same, JSON
- `results/medsec25_loco/per_fold/<category>/test_predictions.npz` — raw arrays for each fold
- [`experiments/medsec25_loco/preprocess.py`](../../experiments/medsec25_loco/preprocess.py)
- [`experiments/medsec25_loco/run_loco.py`](../../experiments/medsec25_loco/run_loco.py)
- [`experiments/medsec25_loco/summarize_loco.py`](../../experiments/medsec25_loco/summarize_loco.py)
