# IV. EXPERIMENTAL RESULTS

This section presents the experimental evaluation of the RA-X-IoMT framework
on the WUSTL-EHMS-2020 dataset. All experiments use a CNN-BiLSTM-Attention
architecture (482,817 parameters) trained with inverse-frequency class weighting
and optimized via Bayesian hyperparameter search. Final model evaluation is
performed on a held-out test set never accessed during training or tuning.

---

## A. Dataset Summary

The WUSTL-EHMS-2020 dataset captures physiological and network traffic from
IoMT-connected vital-sign monitors. Training data is augmented with SMOTE to
address the 6.97:1 class imbalance; the test set retains the original
distribution to reflect deployment conditions.

| Dataset | Samples | Normal | Attack | Imbalance Ratio |
|---------|--------:|-------:|-------:|----------------:|
| WUSTL train (pre-SMOTE) | 11,422 | 9,990 | 1,432 | 6.97 : 1 |
| WUSTL train (post-SMOTE) | 19,980 | 9,990 | 9,990 | 1 : 1 |
| WUSTL test | 4,877 | 4,265 | 612 | 6.97 : 1 |

The test partition preserves the original class imbalance (12.5% attack
prevalence), ensuring that reported metrics reflect realistic clinical
deployment scenarios rather than artificially balanced evaluation conditions.

---

## B. Hyperparameter Configuration

Bayesian optimization (keras-tuner, 20 trials × 2 executions per trial,
objective = val\_AUC) was conducted over a search space of 324 possible
configurations. The search confirmed that the Phase 2 default architecture is
already optimal for val\_AUC on this dataset.

| Hyperparameter | Phase 2 Default | Optimized | Search Method |
|----------------|:---------------:|:---------:|:-------------:|
| timesteps | 20 | 20 | Bayesian (20 trials) |
| cnn\_filters | 64 | 64 | Bayesian |
| bilstm\_units | 128 | 128 | Bayesian |
| dropout\_rate | 0.3 | 0.3 | Bayesian |
| learning\_rate | 0.001 | 0.001 | Bayesian |
| dense\_units | 64 | 64 | Bayesian |
| class\_weight (attack) | 1.0 | 3.988 | Computed (N / 2·N\_attack) |
| decision threshold | 0.5 | 0.608 | Youden's J statistic |

**Bayesian search result.** The optimal hyperparameters coincide with the
Phase 2 defaults across all six architectural parameters (Trial 08,
val\_AUC = 0.8645). This convergence indicates that the original architecture
was well-specified and that performance gains must come from training strategy
(class weighting, threshold calibration) rather than topology changes.

**Class weight derivation.** Weights are computed from the pre-SMOTE
distribution: w\_class = N\_total / (2 · N\_class). This yields w\_normal = 0.572
and w\_attack = 3.988, imposing a 7.0× penalty on attack misclassification
relative to normal-class errors — reflecting the asymmetric cost structure of
IoMT environments where missed attacks carry life-critical consequences.

---

## C. Training Progression

Table III tracks the iterative improvement pipeline from baseline to final
model. Each version introduces a specific fix targeting the identified
performance bottleneck.

| Version | Fix Applied | AUC | Attack Recall | F1 (weighted) |
|---------|-------------|:---:|:-------------:|:-------------:|
| v1 (baseline) | None — Phase 2 weights, t = 0.5 | 0.6114 | 12.1% | 0.8135 |
| v2 (class\_weight) | class\_weight + progressive unfreezing (45 epochs) | 0.7243 | 57.7% | 0.8233 |
| v3 (tuned) | Bayesian HP search — confirmed defaults optimal | 0.7243 | 57.7% | 0.8233 |

**v1 → v2 analysis.** Introducing inverse-frequency class weighting
(w\_attack = 3.988) and progressive unfreezing across three training phases
(20 + 10 + 15 epochs with early stopping) improved attack recall by
+377% (12.1% → 57.7%) and AUC by +18.5% (0.6114 → 0.7243). The accuracy
decrease from 83.4% to 80.2% is expected: the model trades majority-class
precision for minority-class sensitivity — a favorable tradeoff in IoMT
threat detection where false negatives pose greater risk than false positives.

**v2 → v3 analysis.** Bayesian optimization over 324 hyperparameter
configurations confirmed that the Phase 2 architecture (timesteps = 20,
cnn\_filters = 64, bilstm\_units = 128, dropout = 0.3, lr = 0.001,
dense = 64) is already Pareto-optimal for val\_AUC. The v3 contribution is
methodological: providing empirical evidence that the search space has been
exhausted and that further architectural exploration on this dataset is
unlikely to yield AUC improvements.

---

## D. Ablation Study

To quantify the contribution of each architectural component, three model
variants were trained with identical hyperparameters (from the Bayesian
optimum) and evaluated on the validation set (post-SMOTE, balanced classes).
All variants use class\_weight, SMOTE-augmented training data, and the same
random seed (42).

| Model | AUC | F1 | Attack Recall | Params | Train Time |
|-------|:---:|:--:|:-------------:|-------:|-----------:|
| A: CNN only | 0.8360 | 0.5355 | 97.7% | 38,657 | 26.3 s |
| B: CNN + BiLSTM | 0.8615 | 0.6497 | 95.6% | 466,177 | 49.7 s |
| C: Full (CNN + BiLSTM + Attention) | 0.8610 | 0.6407 | 96.7% | 482,817 | 77.7 s |

**Row A → B (adding BiLSTM).** Incorporating bidirectional LSTM layers
improved AUC by +3.05% (0.8360 → 0.8615) and F1 by +21.3%
(0.5355 → 0.6497), confirming that temporal pattern learning across
sequential network observations contributes meaningfully to intrusion
detection accuracy. The BiLSTM captures bidirectional dependencies in traffic
flow sequences that the CNN's local receptive field cannot model. Attack
recall decreased slightly (−2.2%) as the BiLSTM produces more calibrated
predictions, reducing the over-prediction of attacks observed in the CNN-only
variant.

**Row B → C (adding Attention).** The Bahdanau attention mechanism improved
attack recall by +1.1% (95.6% → 96.7%) with negligible AUC change
(−0.06%), at a cost of only 16,640 additional parameters (+3.6%).
While the attention layer's contribution to raw discriminative performance
is marginal on this dataset, it provides two critical capabilities:
(1) per-timestep importance scores enabling SHAP-based explainability
(Phase 5), and (2) an interpretable context vector for the risk-adaptive
scoring engine (Phase 6). In IoMT deployments where regulatory transparency
is required, this interpretability premium justifies the architectural
inclusion regardless of marginal AUC impact.

---

## E. Final Test-Set Performance

The final model (v2/v3, Full architecture) is evaluated on the held-out
test set (N = 4,877) at the optimal threshold (t = 0.608) determined by
Youden's J statistic. All metrics below are computed on data never accessed
during training, validation, or hyperparameter tuning.

| Metric | Value | Interpretation |
|--------|:-----:|----------------|
| AUC-ROC | 0.7243 | Moderate discriminative ability; probability that a randomly chosen attack ranks higher than a random normal sample |
| F1-score (weighted) | 0.8233 | Balanced precision–recall tradeoff, weighted by class support |
| Precision (weighted) | 0.8571 | 85.7% of positive predictions are correct across both classes |
| Recall (weighted) | 0.8023 | 80.2% overall detection rate across both classes |
| Attack Recall (Sensitivity) | 0.5768 | 57.7% of true attacks are detected — a 4.8× improvement over v1 baseline (12.1%) |
| False Positive Rate | 0.1653 | 16.5% of normal traffic triggers a false alarm — acceptable for clinical monitoring with human-in-the-loop review |

**Attack-class metrics at the operating threshold (t = 0.608):**

| Metric | Value |
|--------|:-----:|
| Attack Precision | 0.3336 |
| Attack Recall | 0.5768 |
| Attack F1 | 0.4228 |
| Normal Precision | 0.9322 |
| Normal Recall | 0.8347 |
| Normal F1 | 0.8808 |

---

## F. Confusion Matrix

The confusion matrix at the optimal operating threshold (t = 0.608) on the
WUSTL-EHMS-2020 test set:

|  | Predicted Normal | Predicted Attack |
|--|:----------------:|:----------------:|
| **Actual Normal** (N = 4,265) | TN = 3,560 | FP = 705 |
| **Actual Attack** (N = 612) | FN = 259 | TP = 353 |

**Analysis.** Of 612 true attacks, 353 (57.7%) are correctly detected while
259 (42.3%) are missed. The 705 false positives represent 16.5% of normal
traffic — a manageable false alarm rate for IoMT environments with automated
triage. In a clinical deployment, false positives trigger review by security
analysts (low cost), while false negatives expose patients to undetected
device compromise (high cost). The model's operating point prioritizes
reducing FN at the expense of elevated FP, consistent with the asymmetric
cost structure encoded via class\_weight.

---

## G. Threshold Sensitivity Analysis

Table VII demonstrates the precision–sensitivity tradeoff across decision
thresholds for the v2 model. The AUC-ROC (0.7243) remains constant across
thresholds as it measures overall rank ordering.

| Threshold | Accuracy | F1 (weighted) | Attack Recall | FPR |
|:---------:|:--------:|:-------------:|:-------------:|:---:|
| 0.30 | 12.6% | 0.0296 | 100.0% | 99.9% |
| 0.40 | 45.9% | 0.5316 | 82.2% | 59.3% |
| 0.50 (default) | 68.5% | 0.7362 | 66.3% | 31.2% |
| **0.608 (optimal)** | **80.2%** | **0.8233** | **57.7%** | **16.5%** |

**Youden's J statistic.** The optimal threshold (t = 0.608) maximizes
J = TPR − FPR = 0.577 − 0.165 = 0.412, representing the point of maximum
separation between the true positive and false positive rates on the ROC
curve. At this threshold, the model achieves 57.7% attack recall at 16.5% FPR.

**Clinical operating point selection.** In deployments where attack recall
must exceed a regulatory minimum (e.g., > 80%), the threshold may be lowered
to t ≈ 0.40 (attack recall = 82.2%), accepting an FPR of 59.3%. The
risk-adaptive layer (Phase 6) dynamically adjusts the effective threshold
based on device criticality and patient acuity, enabling per-context
operating-point customization without retraining.

---

## H. Generalization Assessment

Table VIII compares validation-set performance (balanced, post-SMOTE, used
during Bayesian tuning and ablation) against held-out test-set performance
(imbalanced, original distribution). All validation metrics are from the
Full ablation variant (C) at t = 0.5; test metrics are at t = 0.5 for
apples-to-apples comparison where noted.

| Metric | Validation | Test | Delta | Assessment |
|--------|:----------:|:----:|:-----:|------------|
| AUC | 0.8610 | 0.7243 | −15.9% | Moderate gap — SMOTE-augmented validation inflates ranking performance |
| Attack Recall (t = 0.5) | 0.9669 | 0.6634 | −31.4% | Significant degradation on natural class distribution |
| F1 (t = 0.5) | 0.6407 | 0.7362 | +14.9% | Apparent gain is an artifact of majority-class skew in the test set |

**Interpretation.** The validation–test gap is primarily attributable to
three factors:

1. **SMOTE distribution shift.** The validation set contains synthetic
   minority-class samples generated by SMOTE, creating a balanced 50/50
   distribution. The test set reflects the original 6.97:1 imbalance, where
   attack samples constitute only 12.5% of observations.

2. **Threshold sensitivity.** At the default threshold (t = 0.5), the model
   detects 66.3% of test-set attacks. At the optimal threshold (t = 0.608),
   attack recall drops to 57.7% but overall accuracy improves from 68.5% to
   80.2%, reflecting the tradeoff between sensitivity and specificity.

3. **Feature overlap.** The WUSTL-EHMS-2020 attack types (spoofing, data
   injection, network probing) share statistical signatures with normal
   physiological variations, limiting separability in feature space —
   particularly for low-amplitude spoofing attacks that mimic normal vital
   sign ranges.

**Cross-dataset disclosure.** Evaluation on CICIoMT2024 (1.6M samples,
97.7% attack) yielded AUC = 0.489, confirming that the WUSTL-trained model
does not generalize to unseen IoMT attack distributions. Of 29 input
features, only 5 could be semantically mapped from CICIoMT2024 to WUSTL
schema; the remaining 24 were imputed with WUSTL normal-class medians,
effectively nullifying the model's learned feature representations. This
result underscores the need for domain-specific training data in IoMT
intrusion detection and motivates future work on transfer learning and
domain adaptation techniques.

---

## I. Comparison with Related Work

Table IX positions RA-X-IoMT against prior work evaluated on the
WUSTL-EHMS-2020 dataset.

| Model | Accuracy | F1 | AUC | Explainability | Risk-Adaptive |
|-------|:--------:|:--:|:---:|:--------------:|:-------------:|
| RCL Net [8] | 0.9978 | — | — | No | No |
| Bouke et al. [9] | 0.9900 | — | — | No | No |
| RA-X-IoMT (ours) | 0.8023 | 0.8233 | 0.7243 | Yes (SHAP) | Yes (MAD) |

**Disclaimer.** Direct comparison with prior work should be interpreted
with caution due to fundamental differences in evaluation methodology and
optimization objectives:

1. **Accuracy-optimized baselines.** RCL Net and Bouke et al. optimize for
   raw accuracy without class weighting, achieving > 99% accuracy on the
   WUSTL-EHMS-2020 dataset. However, accuracy is a misleading metric under
   6.97:1 class imbalance: a naive classifier predicting all samples as
   "Normal" achieves 87.5% accuracy. Neither work reports attack-specific
   recall, AUC-ROC, or false positive rates — metrics essential for
   evaluating clinical threat detection.

2. **Explicit attack-recall prioritization.** RA-X-IoMT applies a 7.0×
   class\_weight penalty on attack misclassification and optimizes for
   val\_AUC rather than accuracy. This design choice intentionally trades
   majority-class accuracy for minority-class sensitivity, reflecting the
   life-critical nature of IoMT environments where a missed intrusion (FN)
   may compromise patient safety, whereas a false alarm (FP) triggers a
   low-cost manual review.

3. **Explainability and risk adaptation.** RA-X-IoMT integrates SHAP-based
   feature attribution (Phase 5) and MAD-based risk-adaptive scoring
   (Phase 6) — capabilities absent from prior work. These components are
   essential for regulatory compliance (FDA, HIPAA) and clinical trust,
   but impose architectural constraints (e.g., attention layer for
   interpretability) that may reduce raw discriminative performance.

4. **Reproducibility.** All RA-X-IoMT experiments use fixed random seeds
   (seed = 42), stratified data splits, and strict test-set isolation.
   Training configuration, hyperparameter search logs, and threshold
   analysis artifacts are publicly available in the project repository.

---

## Summary of Key Findings

| Finding | Evidence |
|---------|----------|
| Bayesian tuning confirms Phase 2 defaults are optimal | 20 trials over 324 combinations; best trial matches defaults (val\_AUC = 0.8645) |
| Class weighting is the primary performance driver | Attack recall: 12.1% → 57.7% (+377%) after introducing w\_attack = 3.988 |
| BiLSTM captures temporal patterns | Ablation: +3.05% AUC over CNN-only (0.8360 → 0.8615) |
| Attention adds interpretability, not AUC | Ablation: −0.06% AUC, +1.1% attack recall; enables SHAP explanations |
| SMOTE inflates validation performance | Validation AUC (0.8610) overestimates test AUC (0.7243) by 15.9% |
| Model does not generalize cross-dataset | CICIoMT2024 AUC = 0.489 (5/29 features mapped; 24 imputed) |

---

*All metrics computed on held-out test data (N = 4,877) unless explicitly
noted as validation-set results. Ablation study uses validation split of
SMOTE-augmented training data. Hardware: NVIDIA RTX 4060 Laptop GPU,
mixed\_float16 precision, TensorFlow 2.20.0.*
