## Cross-Dataset Validation Summary

Primary dataset: WUSTL-EHMS-2020
Validation dataset: CICIoMT2024

### Sampling Disclosure

Full dataset: 1,612,117 samples
Evaluated sample: 10,000 (stratified, random_state=42)
Class ratio preserved: Normal=2.33%, Attack=97.67%

### Alignment Disclosure

Mapped features: 5 (dur, load, srcload, totbytes, packet_num)
Imputed features: 24 (WUSTL Normal medians)
dstload: excluded — drate=0.0 in CICIoMT2024

### Results

| Metric | WUSTL test | CICIoMT2024 | Delta |
|--------|-----------|-------------|-------|
| Accuracy | 0.8341 | 0.0579 | 93.1% |
| F1-score | 0.8135 | 0.0731 | 91.0% |
| Precision | 0.7976 | 0.9071 | 13.7% |
| Recall | 0.8341 | 0.0579 | 93.1% |
| AUC-ROC | 0.6114 | 0.4888 | 20.1% |

Interpretation: Limited generalization

### Confusion Matrix — CICIoMT2024

| | Predicted Normal | Predicted Attack |
|---|---|---|
| **Actual Normal** | TN=203 | FP=29 |
| **Actual Attack** | FN=9374 | TP=375 |

### Sample Counts

| Dataset | Samples | Normal | Attack |
|---------|---------|--------|--------|
| WUSTL test | 4,877 | — | — |
| CICIoMT2024 (full) | 1,612,117 | 37,607 | 1,574,510 |
| CICIoMT2024 (sample) | 10,000 | 233 | 9,767 |

### Note

Results represent conservative lower bound due to feature
imputation and stratified sampling. The 5 mapped features
carry real CICIoMT2024 signal; the 24 imputed features
contribute no discriminative information and act as constant baselines.

---

**Reviewer signature:** ____________________  **Date:** ____/____/______
