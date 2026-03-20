"""Cross-dataset comparison report renderer — generates section 6.2 markdown.

Pure function with no I/O: the pipeline writes the returned string
to disk.
"""

from __future__ import annotations

from typing import Any, Dict, List


def build_comparison_report(
    wustl_metrics: Dict[str, Any],
    ciciomt_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Build per-metric delta comparison for JSON export.

    Args:
        wustl_metrics: Evaluation metrics from WUSTL test set.
        ciciomt_metrics: Evaluation metrics from CICIoMT2024.

    Returns:
        Dict with per-metric comparison including absolute and
        percentage deltas.
    """
    metric_keys: List[str] = [
        "accuracy",
        "f1_score",
        "precision",
        "recall",
        "auc_roc",
    ]
    comparison: Dict[str, Any] = {}
    for key in metric_keys:
        w_val = wustl_metrics[key]
        c_val = ciciomt_metrics[key]
        delta = abs(w_val - c_val)
        pct = (delta / w_val * 100) if w_val != 0 else 0.0
        comparison[key] = {
            "wustl": round(w_val, 4),
            "ciciomt2024": round(c_val, 4),
            "delta": round(delta, 4),
            "delta_pct": round(pct, 2),
        }
    return comparison


def render_cross_dataset_report(
    wustl_metrics: Dict[str, Any],
    ciciomt_metrics: Dict[str, Any],
    load_report: Dict[str, Any],
    comparison: Dict[str, Any],
) -> str:
    """Render section 6.2 Cross-Dataset Validation report as markdown.

    Args:
        wustl_metrics: Evaluation metrics from WUSTL test set.
        ciciomt_metrics: Evaluation metrics from CICIoMT2024.
        load_report: CICIoMTLoader report dict.
        comparison: Per-metric delta dict from build_comparison_report.

    Returns:
        Complete markdown string.
    """
    # Comparison table
    comp_rows = ""
    for key in ["accuracy", "f1_score", "precision", "recall", "auc_roc"]:
        label = {
            "accuracy": "Accuracy",
            "f1_score": "F1-score",
            "precision": "Precision",
            "recall": "Recall",
            "auc_roc": "AUC-ROC",
        }[key]
        c = comparison[key]
        comp_rows += (
            f"| {label} | {c['wustl']:.4f} "
            f"| {c['ciciomt2024']:.4f} "
            f"| {c['delta']:.4f} | {c['delta_pct']:.1f}% |\n"
        )

    # Generalization assessment
    deltas = [comparison[k]["delta_pct"] for k in comparison]
    max_delta = max(deltas)
    if max_delta < 5.0:
        assessment = (
            "Delta < 5% across all metrics indicates **strong "
            "generalization**. RA-X-IoMT maintains performance across "
            "heterogeneous IoMT environments despite dataset domain "
            "differences."
        )
    elif max_delta < 10.0:
        assessment = (
            "Delta < 10% indicates **moderate generalization**. "
            "Some performance degradation observed when transferring "
            "across IoMT dataset domains."
        )
    else:
        assessment = (
            f"Delta up to {max_delta:.1f}% indicates the model "
            "has limited generalization across these specific "
            "IoMT dataset domains. Further domain adaptation may "
            "be needed."
        )

    # CICIoMT2024 confusion matrix
    cm_ciciomt = ciciomt_metrics.get("confusion_matrix", [])
    if len(cm_ciciomt) == 2:
        tn = cm_ciciomt[0][0]
        fp = cm_ciciomt[0][1]
        fn = cm_ciciomt[1][0]
        tp = cm_ciciomt[1][1]
        cm_table = (
            "| | Predicted Normal | Predicted Attack |\n"
            "|---|---|---|\n"
            f"| **Actual Normal** | TN={tn} | FP={fp} |\n"
            f"| **Actual Attack** | FN={fn} | TP={tp} |"
        )
    else:
        cm_table = "See `confusion_matrix_ciciomt.csv` for full matrix."

    # WUSTL confusion matrix
    cm_wustl = wustl_metrics.get("confusion_matrix", [])
    if len(cm_wustl) == 2:
        wtn = cm_wustl[0][0]
        wfp = cm_wustl[0][1]
        wfn = cm_wustl[1][0]
        wtp = cm_wustl[1][1]
        cm_wustl_table = (
            "| | Predicted Normal | Predicted Attack |\n"
            "|---|---|---|\n"
            f"| **Actual Normal** | TN={wtn} | FP={wfp} |\n"
            f"| **Actual Attack** | FN={wfn} | TP={wtp} |"
        )
    else:
        cm_wustl_table = "See `confusion_matrix.csv` for full matrix."

    # Biometric imputation table
    medians = load_report.get("biometric_medians", {})
    bio_rows = ""
    for col, val in sorted(medians.items()):
        bio_rows += f"| `{col}` | {val:.4f} | WUSTL Normal (y=0) |\n"

    # Sample counts
    wustl_n = wustl_metrics.get("test_samples", "N/A")
    ciciomt_n = load_report.get("samples", "N/A")

    report = f"""## 6.2 Cross-Dataset Validation (CICIoMT2024)

This section documents the cross-dataset generalization evaluation
using CICIoMT2024 as a validation dataset against the WUSTL-EHMS-2020
trained classification model.

### 6.2.1 Dataset Preparation

- **Primary dataset:** WUSTL-EHMS-2020 — training + primary evaluation
- **Validation dataset:** CICIoMT2024 — generalization evaluation only
- **CICIoMT2024 samples:** {ciciomt_n}
- **WUSTL test samples:** {wustl_n}

CICIoMT2024 does not contain biometric features.
Missing values imputed using median values from WUSTL-EHMS-2020
Normal traffic samples — conservative approach to avoid
inflating cross-dataset results.

### 6.2.2 Biometric Feature Imputation

| Feature | Imputed Value | Source |
|---------|--------------|--------|
{bio_rows}
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
{comp_rows}
### 6.2.5 Generalization Assessment

{assessment}

### 6.2.6 Confusion Matrix — WUSTL Test Set

{cm_wustl_table}

### 6.2.7 Confusion Matrix — CICIoMT2024

{cm_table}

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
"""
    return report
