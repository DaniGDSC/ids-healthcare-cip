## 5.3 Hyperparameter Fine-Tuning & Ablation Study

### 5.3.1 Search Configuration

| Property | Value |
|----------|-------|
| Strategy | Bayesian TPE (Optuna) |
| Target metric | attack_f2 (maximize) |
| Total trials | 30 |
| Completed | 30 |
| Failed | 0 |

### 5.3.2 Best Hyperparameters

| Parameter | Value |
|-----------|-------|
| `cw_attack` | 7.58795 |
| `finetune_lr` | 7.96945e-05 |
| `ft_epochs` | 5 |
| `head_epochs` | 3 |
| `head_lr` | 0.00118443 |

### 5.3.3 Best Trial Metrics

| Metric | Value |
|--------|-------|
| accuracy | 0.9415 |
| attack_f1 | 0.9699 |
| attack_f2 | 0.9877 |
| attack_precision | 0.9415 |
| attack_recall | 1.0000 |
| auc_roc | 0.8876 |
| f1_score | 0.9132 |
| macro_f1 | 0.4849 |
| threshold | 0.1000 |

### 5.3.4 Top-5 Trials

| Trial | Attack F1 | Attack Recall | AUC-ROC | Accuracy | Duration |
|-------|-----------|---------------|---------|----------|----------|
| 0 | 0.9699 | 1.0000 | 0.8876 | 0.9415 | 8.4s |
| 1 | 0.9699 | 1.0000 | 0.8926 | 0.9415 | 5.1s |
| 2 | 0.9699 | 1.0000 | 0.9056 | 0.9415 | 6.2s |
| 3 | 0.9699 | 1.0000 | 0.8917 | 0.9415 | 5.6s |
| 4 | 0.9699 | 1.0000 | 0.8750 | 0.9415 | 4.8s |

### 5.3.6 Parameter Importance (optuna_fanova)

| Parameter | Importance | |
|-----------|-----------|---|
| `head_lr` | 0.3440 | ############# |
| `cw_attack` | 0.3328 | ############# |
| `finetune_lr` | 0.2957 | ########### |
| `ft_epochs` | 0.0275 | # |
| `head_epochs` | 0.0000 |  |

### 5.3.7 Ablation Study Results

| Variant | Attack F1 | AUC-ROC | Accuracy | delta F1 | delta AUC |
|---------|-----------|---------|----------|---------|----------|
| baseline (full) | FAILED | — | — | — | — |
| no_attention | FAILED | — | — | — | — |
| no_bilstm2 | FAILED | — | — | — | — |
| no_cnn2 | FAILED | — | — | — | — |
| unidirectional_lstm | FAILED | — | — | — | — |
| timesteps_10 | FAILED | — | — | — | — |
| timesteps_30 | FAILED | — | — | — | — |
| low_dropout | FAILED | — | — | — | — |
| high_dropout | FAILED | — | — | — | — |

### 5.3.8 Execution Summary

| Property | Value |
|----------|-------|
| Device | see finetuned_results.json |
| TensorFlow | — |
| Duration | 239.79s |
| Git commit | `see finetune` |

---

**Reviewer signature:** ____________________  **Date:** ____/____/______
