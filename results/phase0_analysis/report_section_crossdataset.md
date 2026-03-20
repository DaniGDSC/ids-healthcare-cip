## 6.2 Cross-Dataset Validation (CICIoMT2024)

This section documents the cross-dataset generalization evaluation
using CICIoMT2024 as a validation dataset against the WUSTL-EHMS-2020
trained classification model.

### 6.2.1 Dataset Preparation

- **Primary dataset:** WUSTL-EHMS-2020 — training + primary evaluation
- **Validation dataset:** CICIoMT2024 — generalization evaluation only
- **CICIoMT2024 samples:** 200
- **WUSTL test samples:** 4877

CICIoMT2024 does not contain biometric features.
Missing values imputed using median values from WUSTL-EHMS-2020
Normal traffic samples — conservative approach to avoid
inflating cross-dataset results.

### 6.2.2 Biometric Feature Imputation

| Feature | Imputed Value | Source |
|---------|--------------|--------|
| `DIA` | 0.1452 | WUSTL Normal (y=0) |
| `Heart_rate` | 0.0000 | WUSTL Normal (y=0) |
| `Pulse_Rate` | -0.2531 | WUSTL Normal (y=0) |
| `Resp_Rate` | 0.0000 | WUSTL Normal (y=0) |
| `ST` | 0.0000 | WUSTL Normal (y=0) |
| `SYS` | 0.0367 | WUSTL Normal (y=0) |
| `SpO2` | 0.0000 | WUSTL Normal (y=0) |
| `Temp` | -0.0120 | WUSTL Normal (y=0) |

All biometric features were absent in CICIoMT2024 and imputed
using median values from WUSTL-EHMS-2020 Normal (benign) traffic
samples only (Label=0). This ensures imputed values represent
baseline physiological readings, not attack-influenced values.

### 6.2.3 Scaler Application

The `RobustScaler` fitted on WUSTL-EHMS-2020 training data was
applied to CICIoMT2024 features via `transform()` only — the
scaler was **NOT** refit on CICIoMT2024. This preserves the
original feature scaling learned from the training distribution.

### 6.2.4 Metric Comparison

| Metric | WUSTL test | CICIoMT2024 | Delta | Delta % |
|--------|-----------|-------------|-------|---------|
| Accuracy | 0.8341 | 0.7348 | 0.0993 | 11.9% |
| F1-score | 0.8135 | 0.6413 | 0.1722 | 21.2% |
| Precision | 0.7976 | 0.6198 | 0.1778 | 22.3% |
| Recall | 0.8341 | 0.7348 | 0.0993 | 11.9% |
| AUC-ROC | 0.6114 | 0.5013 | 0.1101 | 18.0% |

### 6.2.5 Generalization Assessment

Delta up to 22.3% indicates the model has limited generalization across these specific IoMT dataset domains. Further domain adaptation may be needed.

### 6.2.6 Confusion Matrix — WUSTL Test Set

| | Predicted Normal | Predicted Attack |
|---|---|---|
| **Actual Normal** | TN=3994 | FP=271 |
| **Actual Attack** | FN=538 | TP=74 |

### 6.2.7 Confusion Matrix — CICIoMT2024

| | Predicted Normal | Predicted Attack |
|---|---|---|
| **Actual Normal** | TN=132 | FP=3 |
| **Actual Attack** | FN=45 | TP=1 |

### 6.2.8 Limitations

Cross-dataset evaluation is limited by biometric feature
imputation — CICIoMT2024 results should be interpreted
as a **conservative lower bound** of true generalization.

Specific limitations:

1. **Biometric imputation:** All 8 biometric features are
   constant-valued after imputation, reducing feature variance
   that the Attention mechanism may rely on.
2. **Feature mapping:** Network feature names may not perfectly
   correspond between WUSTL-EHMS (Argus-based) and CICIoMT2024
   (CICFlowMeter-based) datasets.
3. **Label semantics:** Attack categories may differ between
   datasets. Binary classification (Normal vs Attack) provides
   the fairest comparison.

### 6.2.9 Cross-Dataset Disclosure

> CICIoMT2024 biometric features imputed using WUSTL-EHMS-2020
> Normal sample medians. Scaler fitted on WUSTL train set only.
> Results reflect conservative generalization estimate.

### 6.2.10 References

- S. Dadkhah et al., "CICIoMT2024: Attack Vectors in
  Healthcare Devices," Internet of Things, 2024.
- A. A. Hady et al., "WUSTL-EHMS-2020," IEEE Access, 2020.

---

*Generated: Phase 3 Classification Engine — Cross-Dataset Validation*
