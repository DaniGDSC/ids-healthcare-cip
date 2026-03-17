## 3.2 Data Quality Assessment

This section presents a systematic data-quality evaluation of the WUSTL-EHMS-2020 dataset [WUSTL-EHMS-2020] conducted prior to any preprocessing transformation. Each subsection documents a specific quality dimension with quantitative evidence and interpretation.

### 3.2.1 Outlier Analysis (IQR Method)

Outliers are identified using the Interquartile Range (IQR) method with a fence multiplier of *k* = 1.5. A sample is flagged as an outlier if it falls outside [Q1 − 1.5·IQR, Q3 + 1.5·IQR].

| Feature | Outlier Count | Outlier (%) | Lower Bound | Upper Bound |
|---------|-------------:|------------:|------------:|------------:|
| SpO2            |         6,585 |     40.3542 |     98.0000 |     98.0000 |
| SIntPkt         |         2,716 |     16.6442 |      2.0054 |      6.7714 |
| Dur             |         2,715 |     16.6381 |      0.0060 |      0.0203 |
| SrcJitter       |         2,568 |     15.7372 |      1.5153 |      4.9797 |
| SrcLoad         |         2,304 |     14.1194 | 105298.2500 | 355312.2500 |
| DstLoad         |         2,304 |     14.1194 |  35098.0000 | 118450.0000 |
| Load            |         2,304 |     14.1194 | 140384.0000 | 473808.0000 |
| Rate            |         2,304 |     14.1194 |    212.2750 |    716.4430 |
| Label           |         2,046 |     12.5383 |      0.0000 |      0.0000 |
| DIntPkt         |         1,930 |     11.8274 |      0.8604 |      4.5019 |
| DstJitter       |         1,920 |     11.7661 |      0.7997 |      4.4577 |
| Resp_Rate       |         1,289 |      7.8993 |      9.0000 |     33.0000 |
| Heart_rate      |           979 |      5.9995 |     64.0000 |     88.0000 |
| Temp            |           894 |      5.4786 |     25.1000 |     29.1000 |
| Pulse_Rate      |           886 |      5.4296 |     64.0000 |     88.0000 |
| ST              |           698 |      4.2775 |      0.0500 |      0.4500 |
| SYS             |           507 |      3.1070 |    133.0000 |    157.0000 |
| TotPkts         |            92 |      0.5638 |      7.0000 |      7.0000 |
| TotBytes        |            92 |      0.5638 |    682.0000 |    682.0000 |
| SrcBytes        |            90 |      0.5515 |    496.0000 |    496.0000 |
| DstBytes        |            90 |      0.5515 |    186.0000 |    186.0000 |
| DIA             |            36 |      0.2206 |     64.0000 |     96.0000 |
| Loss            |            18 |      0.1103 |      0.0000 |      0.0000 |
| pLoss           |            18 |      0.1103 |      0.0000 |      0.0000 |
| pSrcLoss        |            15 |      0.0919 |      0.0000 |      0.0000 |
| pDstLoss        |            10 |      0.0613 |      0.0000 |      0.0000 |
| SIntPktAct      |             7 |      0.0429 |      0.0000 |      0.0000 |
| sMaxPktSz       |             7 |      0.0429 |    310.0000 |    310.0000 |
| sMinPktSz       |             4 |      0.0245 |     60.0000 |     60.0000 |
| dMaxPktSz       |             1 |      0.0061 |     66.0000 |     66.0000 |

**30** of 37 numeric features contain at least one outlier. This confirms that network-traffic features exhibit heavy-tailed distributions characteristic of bursty IoMT traffic, justifying the use of RobustScaler (IQR-based normalisation) in Phase 1 rather than Z-score or min–max scaling.

### 3.2.2 Class Imbalance Analysis

| Class  | Count   | Percentage   |
|--------|--------:|-------------:|
| Normal |  14,272 | 87.4617% |
| Attack |   2,046 | 12.5383% |

The imbalance ratio of **6.9756:1** (Normal : Attack) justifies the use of SMOTE (Synthetic Minority Oversampling Technique) applied exclusively to the training partition. Without resampling, classifiers trained on the raw distribution would achieve high accuracy by trivially predicting the majority class, yielding unacceptable recall on attack samples. SMOTE is applied before feature scaling to generate synthetic samples in the original feature space.

### 3.2.3 Feature Correlation Analysis

Pearson correlation analysis identifies **7** feature pairs with |*r*| > 0.95. These pairs represent redundant linear relationships that inflate dimensionality without contributing independent discriminative information.

| # | Feature A       | Feature B       | *r*       | Interpretation |
|--:|-----------------|-----------------|----------:|----------------|
| 1 | SIntPktAct      | SrcJitter       | +0.997285 | Timing jitter derives from inter-packet intervals |
| 2 | Loss            | pLoss           | +0.985985 | Absolute and proportional loss are co-determined |
| 3 | DstLoad         | Rate            | +0.977307 | Destination load is a rate-normalised throughput |
| 4 | DIntPkt         | DstJitter       | +0.971512 | Destination jitter derives from inter-packet intervals |
| 5 | SIntPktAct      | Loss            | +0.954233 | Packet timing correlates with loss under congestion |
| 6 | SrcJitter       | Loss            | +0.953454 | Jitter and loss co-occur during network degradation |
| 7 | DstBytes        | TotPkts         | +0.952941 | Byte volume scales linearly with packet count |

The correlation heatmap reveals two dominant clusters of collinearity: (1) inter-arrival timing features (SIntPktAct ↔ SrcJitter ↔ Loss), and (2) volume-rate features (DstLoad ↔ Rate, DstBytes ↔ TotPkts). Phase 1 redundancy elimination retains one member of each pair (threshold |*r*| > 0.95), reducing the feature space by 6 columns while preserving the full information content.

### 3.2.4 Missing Value Summary

The dataset contains **zero missing values** across all 45 attributes. This confirms that the WUSTL-EHMS-2020 capture pipeline produced acquisition-complete records, eliminating imputation as a potential source of information bias.

### 3.2.5 Data Leakage Risk Assessment

Features dropped due to leakage risk: [`SrcAddr`, `DstAddr`, `Sport`, `Dport`, `SrcMac`, `DstMac`, `Dir`, `Flgs`]

**Justification:** These 8 columns encode network identifiers (IP addresses, MAC addresses, port numbers, and direction/flag fields) that are environment-specific artefacts of the capture topology. A model trained on these features would memorise source/destination pairs rather than learning generalisable intrusion signatures, resulting in artificially inflated test performance that does not transfer to unseen network environments. Their removal is also required for HIPAA Safe Harbor de-identification compliance.

### 3.2.6 Reproducibility Statement

All experiments use `random_state=42` and stratified split 70/30 to ensure deterministic partitioning and reproducible results across independent runs. Stratification preserves the original class prior in both the training and test partitions, preventing evaluation bias due to sampling variance. The complete analysis pipeline is version-controlled and the configuration is externalised in `config.yaml` to enable exact replication by independent researchers.
