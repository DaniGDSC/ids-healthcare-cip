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
| `cw_attack` | 5.47623 |
| `finetune_lr` | 1.3113e-05 |
| `ft_epochs` | 4 |
| `head_epochs` | 3 |
| `head_lr` | 0.00299369 |

### 5.3.3 Best Trial Metrics

| Metric | Value |
|--------|-------|
| accuracy | 0.9403 |
| attack_f1 | 0.9692 |
| attack_f2 | 0.9875 |
| attack_precision | 0.9403 |
| attack_recall | 1.0000 |
| auc_roc | 0.8423 |
| f1_score | 0.9118 |
| macro_f1 | 0.4914 |
| threshold | 0.9000 |

### 5.3.4 Top-5 Trials

| Trial | Attack F1 | Attack Recall | AUC-ROC | Accuracy | Duration |
|-------|-----------|---------------|---------|----------|----------|
| 10 | 0.9692 | 1.0000 | 0.8423 | 0.9403 | 6.5s |
| 7 | 0.9691 | 1.0000 | 0.8420 | 0.9401 | 6.4s |
| 0 | 0.9690 | 1.0000 | 0.8441 | 0.9399 | 7.9s |
| 1 | 0.9690 | 1.0000 | 0.8422 | 0.9399 | 5.5s |
| 2 | 0.9690 | 1.0000 | 0.8442 | 0.9399 | 6.7s |

### 5.3.6 Parameter Importance (optuna_fanova)

| Parameter | Importance | |
|-----------|-----------|---|
| `head_lr` | 0.4961 | ################### |
| `cw_attack` | 0.2138 | ######## |
| `finetune_lr` | 0.1561 | ###### |
| `ft_epochs` | 0.0874 | ### |
| `head_epochs` | 0.0467 | # |

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
| Duration | 278.92s |
| Git commit | `see finetune` |

---

**Reviewer signature:** ____________________  **Date:** ____/____/______
