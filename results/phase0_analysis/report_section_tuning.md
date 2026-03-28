## 5.3 Hyperparameter Fine-Tuning & Ablation Study

This section documents the Phase 2.5 hyperparameter search and component
ablation study for the CNN-BiLSTM-Attention detection architecture.

### 5.3.1 Search Configuration

| Property | Value |
|----------|-------|
| Strategy | bayesian |
| Target metric | attack_f1 (maximize) |
| Total trials | 30 |
| Completed trials | 9 |
| Pruned trials | 21 |
| Failed trials | 0 |

### 5.3.2 Best Hyperparameters

| Parameter | Value |
|-----------|-------|
| `attention_units` | 64 |
| `batch_size` | 512 |
| `bilstm_units_1` | 256 |
| `bilstm_units_2` | 64 |
| `cnn_filters_1` | 64 |
| `cnn_filters_2` | 256 |
| `cnn_kernel_size` | 3 |
| `dropout_rate` | 0.218436 |
| `focal_alpha` | 0.612284 |
| `focal_gamma` | 1.32709 |
| `learning_rate` | 0.0017815 |
| `phase_a_lr` | 0.0026529 |
| `phase_b_lr` | 0.000202097 |
| `phase_c_lr` | 6.22874e-06 |
| `timesteps` | 30 |
| `unfreezing_epochs` | 8 |

### 5.3.3 Best Trial Metrics

| Metric | Value |
|--------|-------|
| accuracy | 0.9211 |
| attack_f1 | 0.5889 |
| attack_precision | 0.8488 |
| attack_recall | 0.4508 |
| auc_roc | 0.7439 |
| f1_score | 0.9103 |
| macro_f1 | 0.7726 |
| macro_precision | 0.8875 |
| macro_recall | 0.7197 |
| precision | 0.9165 |
| recall | 0.9211 |
| threshold | 0.9000 |

### 5.3.4 Top-5 Trials

| Trial | F1 | AUC-ROC | Accuracy | Params | Duration |
|-------|-----|---------|----------|--------|----------|
| 22 | 0.9103 | 0.7439 | 0.9211 | 1,417,729 | 6.3s |
| 1 | 0.9107 | 0.7368 | 0.9223 | 1,442,689 | 6.6s |
| 24 | 0.9098 | 0.7496 | 0.9205 | 1,417,729 | 6.0s |
| 27 | 0.8888 | 0.7354 | 0.8940 | 1,417,729 | 6.3s |
| 26 | 0.8845 | 0.7273 | 0.8893 | 1,417,729 | 6.1s |

### 5.3.5 Multi-Seed Validation

Top-3 configs retrained with 5 seeds
(8 epochs each) for confidence intervals.

| Rank | Original attack_f1 | Mean | Std | Seeds |
|------|------------|------|-----|-------|
| 1 | 0.5889 | 0.5667 | 0.0069 | 5 |
| 2 | 0.5876 | 0.5828 | 0.0054 | 5 |
| 3 | 0.5870 | 0.5635 | 0.0181 | 5 |

### 5.3.6 Parameter Importance (optuna_fanova)

| Parameter | Importance | |
|-----------|-----------|---|
| `phase_b_lr` | 0.2884 | ########### |
| `dropout_rate` | 0.2573 | ########## |
| `cnn_kernel_size` | 0.1106 | #### |
| `learning_rate` | 0.0836 | ### |
| `focal_gamma` | 0.0554 | ## |
| `cnn_filters_2` | 0.0486 | # |
| `focal_alpha` | 0.0370 | # |
| `phase_c_lr` | 0.0325 | # |
| `batch_size` | 0.0253 | # |
| `bilstm_units_1` | 0.0197 |  |

### 5.3.7 Ablation Study Results

Component ablation reveals the contribution of each architectural block.
Negative delta indicates performance degradation when the component is
removed or modified.

| Variant | F1 | AUC-ROC | Accuracy | Params | delta F1 | delta AUC |
|---------|-----|---------|----------|--------|---------|----------|
| baseline (full) | 0.8889 | 0.7374 | 0.8942 | 1,417,729 | +0.0000 | +0.0000 |
| no_attention | 0.9051 | 0.7417 | 0.9149 | 1,409,409 | +0.0162 | +0.0043 |
| no_bilstm2 | 0.9049 | 0.7457 | 0.9139 | 1,171,457 | +0.0160 | +0.0083 |
| no_cnn2 | 0.9152 | 0.7447 | 0.9256 | 975,105 | +0.0262 | +0.0073 |
| unidirectional_lstm | 0.8357 | 0.7114 | 0.8262 | 670,977 | -0.0532 | -0.0260 |
| timesteps_10 | 0.8965 | 0.7510 | 0.9053 | 1,417,729 | +0.0075 | +0.0135 |
| timesteps_30 | 0.9110 | 0.7374 | 0.9234 | 1,417,729 | +0.0221 | -0.0000 |
| low_dropout | 0.9108 | 0.7434 | 0.9223 | 1,417,729 | +0.0219 | +0.0060 |
| high_dropout | 0.9082 | 0.7393 | 0.9203 | 1,417,729 | +0.0193 | +0.0019 |

### 5.3.8 Execution Summary

| Property | Value |
|----------|-------|
| Device | see finetuned_results.json |
| TensorFlow | — |
| CUDA | — |
| Python | — |
| Platform | — |
| Duration | 359.89s |
| Git commit | `see finetune` |
| Config file | `config/phase2_5_config.yaml` (version-controlled) |

---

**Reviewer signature:** ____________________  **Date:** ____/____/______
