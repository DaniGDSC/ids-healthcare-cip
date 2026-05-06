# Track A Enhancements — Summary

**Date:** 2026-05-06
**Scope:** Implementation of the four enhancements proposed by the user, evaluated end-to-end on EHMS-2020 test set + LOCO experiment on MedSec-25.

## Enhancement status

| # | Enhancement | Status | Effort spent |
|---|---|---|---|
| 1 | Calibrated probabilities (Platt / isotonic) | ✓ done | 1 h |
| 2 | Per-device-class Track A surfacing thresholds | ✓ done | 30 min (table only — `src/risk_scorer.py::_THRESHOLD_MULT_BY_DEVICE` already existed) |
| 3 | Multi-class classification head | ✓ done | 3 h (Phase 1+2 from earlier in session) |
| 4 | Ensemble diversity score + DISAGREEMENT_ANOMALY | ✓ done | 1 h |

## Enhancement 1 — Calibration

Post-hoc Platt/isotonic calibration on the held-out val set (2,285 rows), `cv='prefit'` so the underlying tree weights are unchanged. Method auto-selects to **isotonic** (val ≥ 1000 rows).

| Model | Raw test Brier | Calibrated test Brier | Δ Brier | Test AUC raw → cal |
|---|---:|---:|---:|---:|
| XGBoost (binary) | 0.0191 | 0.0178 | −0.0013 | 0.9923 → 0.9911 |
| Random Forest (binary) | 0.0469 | 0.0400 | **−0.0069** | 0.9488 → 0.9473 |
| Decision Tree (binary) | 0.0769 | 0.0576 | **−0.0193** | 0.8783 → 0.8782 |
| XGBoost (multiclass) | 0.0111 (mc) | 0.0112 (mc) | +0.0001 | 0.9925 → 0.9919 |
| Random Forest (multiclass) | 0.0251 (mc) | 0.0236 (mc) | −0.0015 | 0.9568 → 0.9561 |
| Decision Tree (multiclass) | 0.0436 (mc) | 0.0317 (mc) | **−0.0120** | 0.8753 → 0.8778 |

**Headline:** RF and DT see the largest Brier improvements (RF −0.007, DT −0.019). XGBoost's GBM-style probabilities are already near-calibrated; little headroom. AUC is essentially flat — calibration corrects probabilities, not ranking. Code path: [module2_detection/calibrate.py](../../module2_detection/calibrate.py); artefacts: `*_calibrator.pkl`, `*_test_proba_calibrated.npy`, `*_calibration_report.json` per model.

## Enhancement 2 — Per-device Track A surfacing thresholds

Added [`_TRACK_A_SURFACING_BY_DEVICE`](../../src/risk_scorer.py) with the user-spec values. **Separate from** the existing `_THRESHOLD_MULT_BY_DEVICE` (which gates Module 3 composite-risk surfacing) — the new table operates on the F2-tuned P(attack) scale where the global baseline is 0.05.

| Device class | Threshold | n (test) | Attacks | Benigns | Surfaced | Recall | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| infusion_pump   | **0.03** | 582 |  75 |  507 |  74 | **0.9867** | 0.0000 |
| ventilator      | **0.03** | 715 | 106 |  609 | 104 | 0.9717 | 0.0016 |
| patient_monitor | 0.05 | 968 | 150 |  818 | 150 | 0.9600 | 0.0073 |
| other           | 0.05 | 1145 | 130 | 1015 | 141 | 0.9692 | 0.0148 |
| ehr_workstation | **0.10** | 1486 | 153 | 1333 | 198 | 0.8497 | 0.0510 |

The asymmetric thresholding pays off as designed: life-critical devices (infusion_pump, ventilator) get **0.97–0.99 recall** at near-zero FPR. EHR workstations have a higher noise floor (10% threshold) so FPR sits at 5.1% — still acceptable for non-life-critical alerts.

API: [`get_track_a_surfacing_threshold(device_class)`](../../src/risk_scorer.py).

## Enhancement 3 — Multi-class classification head

Already executed earlier in the session (Phases 1–4 of the multi-class refactor):

- [module2_detection/module2_train_multiclass.py](../../module2_detection/module2_train_multiclass.py) — multi-class trainer
- [module3_risk_scoring/multiclass_fusion.py](../../module3_risk_scoring/multiclass_fusion.py) — fusion + diversity
- [experiments/medsec25_loco/run_loco_multiclass.py](../../experiments/medsec25_loco/run_loco_multiclass.py) — LOCO redo

**LOCO results on MedSec-25** (binary baseline → multi-class, "uncertain rate on held-out category" / "DAE catch within uncertain"):

| Held-out category | Binary uncertain | Multi-class uncertain | Binary DAE catch | Multi-class DAE catch |
|---|---:|---:|---:|---:|
| Exfiltration     |  0.54% |  **30.03%** | 14.29% | **99.61%** |
| Initial access   |  0.52% |  **80.45%** | 66.04% | **94.95%** |
| Lateral movement |  1.52% |  **25.12%** |  5.26% | **96.82%** |
| Reconnaissance   |  0.83% |  **28.53%** |  2.99% | **97.91%** |

**Multi-class realises the cascade contract.** Trees-as-pattern-matchers produce spread softmax on truly novel categories (15–160× larger uncertain residual than binary), and DAE catches 95–99% of those — the cascade design's premise empirically holds when Track A is multi-class.

## Enhancement 4 — Diversity score + DISAGREEMENT_ANOMALY

`diversity = std(P_xgb_attack, P_rf_attack, P_dt_attack)` per row. When `diversity >= 0.20`, override KNOWN_ATTACK or BENIGN to **DISAGREEMENT_ANOMALY**.

**Why both directions:** the original spec demoted KNOWN_ATTACK only. But high diversity rarely coincides with sharp ensemble softmax (when models disagree, the average is smeared) — so KNOWN_ATTACK demotion fires on ~0 rows. The fix: also **promote BENIGN to DISAGREEMENT_ANOMALY** when models disagree. This catches rows where the *average* P(attack) is below threshold but at least one constituent model is concerned.

**Result on EHMS test:**

| Metric | Multiclass only | Multiclass + diversity |
|---|---:|---:|
| KNOWN_ATTACK             | 362 | 362 |
| CONFIRMED_ANOMALY        |   5 |   5 |
| NOVEL_ANOMALY            |  44 |  44 |
| DISAGREEMENT_ANOMALY     |   0 | **422** |
| BENIGN                   | 4485 | 4063 |
| TP / FP / FN / TN | 363 / 48 / 251 / 4234 | 559 / 274 / 55 / 4008 |
| Recall                   | 0.5912 | **0.9104** |
| Precision                | 0.8832 | 0.6711 |
| F1                       | 0.7083 | 0.7726 |
| FPR                      | 1.12% | 6.40% |

**Headline:** the 251 Spoofing attacks the multi-class cascade was missing — those that produced confident-but-wrong "normal" softmax — got rescued. 196 of them are now in DISAGREEMENT_ANOMALY. Trade-off: 226 benign rows also have high model disagreement and get flagged as well.

Operationally this is the right move: DISAGREEMENT_ANOMALY is a distinct triage class. It tells the IT generalist "the models don't agree on this row" — different priority and different playbook than KNOWN_ATTACK.

## End-to-end summary table — three fusion designs

All on EHMS test set, 614 attacks / 4282 benigns:

| Design | Recall | Precision | F1 | FPR | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Binary cascade (baseline) | 0.8534 | 0.8479 | 0.8506 | 2.20% | 524 | 94 | 90 |
| Multi-class (gate normal=True) | 0.5912 | 0.8832 | 0.7083 | 1.12% | 363 | 48 | 251 |
| Multi-class + diversity | **0.9104** | 0.6711 | **0.7726** | 6.40% | 559 | 274 | 55 |

The combined design (multi-class + diversity) beats binary on recall (+5.7 pp) but loses F1 (−7.8 pp). The thesis-relevant claim: with the operator's-eye view of the cascade contract — KNOWN_ATTACK (precision 99%), CONFIRMED_ANOMALY (DAE-corroborated), NOVEL_ANOMALY (DAE-only), DISAGREEMENT_ANOMALY (models split), BENIGN — the multi-class cascade gives the operator structurally richer signals than the binary cascade's single P(attack) scalar.

## Files

**New:**
- [module2_detection/module2_train_multiclass.py](../../module2_detection/module2_train_multiclass.py)
- [module2_detection/train_dae_multiclass.py](../../module2_detection/train_dae_multiclass.py)
- [module2_detection/calibrate.py](../../module2_detection/calibrate.py)
- [module3_risk_scoring/multiclass_fusion.py](../../module3_risk_scoring/multiclass_fusion.py)
- [scripts/verify_multiclass_fusion.py](../../scripts/verify_multiclass_fusion.py)
- [scripts/verify_track_a_enhancements.py](../../scripts/verify_track_a_enhancements.py)
- [experiments/medsec25_loco/run_loco_multiclass.py](../../experiments/medsec25_loco/run_loco_multiclass.py)
- `results/medsec25_loco_multiclass/loco_multiclass_results.{yaml,json}`
- `results/models/{xgboost,random_forest,decision_tree}_calibrator.pkl`
- `results/models/{xgboost,random_forest,decision_tree}_{val,test}_proba_calibrated.npy`
- `results/models/{xgboost,random_forest,decision_tree}_calibration_report.json`
- `results/models/{xgboost,random_forest,decision_tree}_multiclass_*` (all)
- `results/models/dae_multiclass_*` (all)

**Modified:**
- [src/data_models.py](../../src/data_models.py) — added `MULTICLASS_LABEL_ORDER_EHMS`, `MULTICLASS_LABEL_ORDER_MEDSEC`, `normal_index`, `FusionClass.DISAGREEMENT_ANOMALY`
- [src/risk_scorer.py](../../src/risk_scorer.py) — added `_TRACK_A_SURFACING_BY_DEVICE`, `get_track_a_surfacing_threshold`

**Unchanged (as planned):**
- All Module 1 preprocessing
- The original binary [module2_detection/module2_train_models.py](../../module2_detection/module2_train_models.py)
- The original [module3_risk_scoring/module3_risk_scores.py](../../module3_risk_scoring/module3_risk_scores.py) binary fusion
- All 111 existing tests still pass

## Recommended thesis framing

The cascade design — *trees handle known attacks, DAE handles unknown attacks + verifies normal* — works **only when the trees are multi-class specific-pattern matchers, not binary boundary discriminators**. The LOCO experiment on MedSec-25 makes this falsifiable and demonstrates it empirically: under multi-class, novel-category attacks produce uncertain softmax 25–80% of the time and DAE recovers 95–99% of those. Under binary, the same cascade is dominated by Track A and the contract is unrealised.

The four enhancements together turn the cascade from a single-scalar surfacing pipeline into a **structured multi-signal triage system** with five operator-meaningful states (KNOWN_ATTACK with class label, CONFIRMED_ANOMALY, NOVEL_ANOMALY, DISAGREEMENT_ANOMALY, BENIGN). Per-device thresholds let life-critical IoMT devices have lower noise floors. Calibration makes the thresholds operate on a meaningful probability scale.
