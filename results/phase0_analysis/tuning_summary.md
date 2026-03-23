## Bayesian Hyperparameter Optimization — Summary

Generated: 2026-03-20T21:05:13.800115+00:00

### Search Configuration

- Objective: val_AUC (maximize)
- Max trials: 20
- Executions per trial: 2
- Hardware: /physical_device:GPU:0
- Mixed precision: float16
- Tuning time: 3276.0s (~55 min, GPU mode)

### Best Hyperparameters

| Parameter | Value |
|-----------|-------|
| timesteps | 20 |
| cnn_filters | 64 |
| bilstm_units | 128 |
| dropout_rate | 0.3 |
| learning_rate_a | 0.001 |
| dense_units | 64 |

**Best val_AUC: 0.8645**

Beats v2 baseline (0.7243): YES

### Ablation Study

| Model | AUC | F1 | Attack Recall | Params | Time |
|-------|-----|----|---------------|--------|------|
| A: CNN | 0.8360 | 0.5355 | 0.9774 | 38657 | 26.3s |
| B: CNN+BiLSTM | 0.8615 | 0.6497 | 0.9563 | 466177 | 49.7s |
| C: Full | 0.8610 | 0.6407 | 0.9669 | 482817 | 77.7s |

### Disclosure

- Test set never accessed during tuning
- class_weight applied in every trial
- random_state=42 for reproducibility
- Validation split: 20% of training data
- Models trained from scratch (no Phase 2 weight transfer during tuning)
- val_AUC baseline comparison is approximate (v2 baseline measured on test set)

---

**Reviewer signature:** ____________________  **Date:** ____/____/______
