## 3.1 Dataset Characterisation

The experiments in this study employ the WUSTL-EHMS-2020 dataset [WUSTL-EHMS-2020], a publicly available benchmark for evaluating intrusion detection systems in Internet of Medical Things (IoMT) environments. The dataset comprises **16,318** samples described by **45** attributes: **35** network-traffic features, **8** physiological (biometric) features, a binary class label, and an attack-category descriptor.

### 3.1.1 Data Partitioning

A stratified 70%/30% train/test split preserves the original class distribution in both partitions. This confirms that evaluation metrics are computed on unseen data that faithfully represents the deployment-time class prior.

### 3.1.2 Class Distribution

| Class    | Count   | Percentage   |
|----------|--------:|-------------:|
| Normal   | 14,272  | 87.4617% |
| Attack   | 2,046  | 12.5383% |
| **Total** | **16,318** | **100.0000%** |

The imbalance ratio is **6.9756:1** (Normal : Attack). This demonstrates a pronounced class imbalance that necessitates resampling (SMOTE) and class-weighted loss functions to prevent majority-class bias during model training.

### 3.1.3 Feature Variance Analysis

Table 2 lists the top 5 features ranked by sample variance, identifying the attributes that carry the most discriminative signal prior to any feature-selection stage.

| Rank | Feature       | Variance         |
|-----:|--------------:|-----------------:|
| 1    | Load          | 12,393,250,127.2434 |
| 2    | SrcLoad       | 6,309,105,848.1125 |
| 3    | DstLoad       | 2,052,824,481.3813 |
| 4    | Packet_num    |  22,181,609.7461 |
| 5    | SrcJitter     |   1,603,704.8657 |

This confirms that network-flow volume features (byte counts, packet counts) exhibit the highest variance, consistent with the heterogeneous traffic patterns observed during attack scenarios in IoMT networks [WUSTL-EHMS-2020].

### 3.1.4 Missing Value Assessment

The dataset contains **zero missing values** across all 45 attributes.

This confirms that the WUSTL-EHMS-2020 dataset is acquisition-complete, requiring no imputation and thereby eliminating a potential source of information leakage during preprocessing.

### 3.1.5 Multicollinearity Analysis

Pearson correlation analysis identifies **7** feature pairs with |*r*| > 0.95:

| Feature A       | Feature B       | *r*       |
|-----------------|-----------------|----------:|
| SIntPktAct      | SrcJitter       | +0.997285 |
| Loss            | pLoss           | +0.985985 |
| DstLoad         | Rate            | +0.977307 |
| DIntPkt         | DstJitter       | +0.971512 |
| SIntPktAct      | Loss            | +0.954233 |
| SrcJitter       | Loss            | +0.953454 |
| DstBytes        | TotPkts         | +0.952941 |

This demonstrates the presence of redundant feature pairs that inflate dimensionality without contributing independent discriminative information. Phase 1 redundancy elimination retains one member of each pair, reducing the feature space from 45 to 29 attributes before feature-selection in Phase 2.

### 3.1.6 Descriptive Statistics Summary

Descriptive statistics (mean, median, standard deviation, min, max) were computed for all 37 numeric features. Full per-feature results are available in `stats_report.json`. Key observations:

- Biometric features exhibit low variance and narrow physiological ranges, consistent with stable patient vital signs under normal operating conditions.
- Network-traffic features span multiple orders of magnitude, motivating the use of RobustScaler (IQR-based) normalisation in Phase 1 to mitigate the influence of extreme outliers in flow-volume attributes.
