"""Page 5 - Dataset Overview: class distribution, ablation, feature importance."""

from __future__ import annotations

from typing import Any, Dict, Optional

import plotly.graph_objects as go
import streamlit as st

from dashboard.utils.loader import (
    load_ablation_results,
    load_best_hyperparams,
    load_explanation_report,
    load_preprocessing_report,
)


def render_class_distribution(preprocess: Dict[str, Any]) -> None:
    """Render class distribution bar chart (pre/post SMOTE)."""
    smote = preprocess.get("smote", {})
    before = smote.get("class_counts_before", {"0": 9990, "1": 1432})
    after = smote.get("class_counts_after", {"0": 9990, "1": 9990})

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Normal", "Attack"],
        y=[before.get("0", before.get(0, 9990)), before.get("1", before.get(1, 1432))],
        name="Pre-SMOTE",
        marker_color="#e74c3c",
        text=[before.get("0", before.get(0, 9990)), before.get("1", before.get(1, 1432))],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        x=["Normal", "Attack"],
        y=[after.get("0", after.get(0, 9990)), after.get("1", after.get(1, 9990))],
        name="Post-SMOTE",
        marker_color="#2ecc71",
        text=[after.get("0", after.get(0, 9990)), after.get("1", after.get(1, 9990))],
        textposition="outside",
    ))
    fig.update_layout(
        title="Class Distribution (Training Set)",
        barmode="group",
        height=380,
        yaxis_title="Sample Count",
        margin=dict(t=50, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e0e0e0",
        yaxis=dict(gridcolor="#333"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_ablation_study(ablation: Dict[str, Any]) -> None:
    """Render ablation study table and grouped bar chart."""
    results = ablation.get("results", [])
    if not results:
        st.info("No ablation results available.")
        return

    st.markdown("#### Ablation Study — Component Contribution")

    # Table
    import pandas as pd
    rows = []
    for r in results:
        rows.append({
            "Model": r["variant"],
            "AUC": f"{r['auc_roc']:.4f}",
            "F1": f"{r['f1_score']:.4f}",
            "Attack Recall": f"{r['attack_recall']:.1%}",
            "Params": f"{r['params']:,}",
            "Train Time": f"{r['train_time_s']:.1f}s",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Grouped bar chart: AUC + Attack Recall
    variants = [r["variant"] for r in results]
    auc_vals = [r["auc_roc"] for r in results]
    recall_vals = [r["attack_recall"] for r in results]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=variants, y=auc_vals,
        name="AUC-ROC",
        marker_color="#3498db",
        text=[f"{v:.4f}" for v in auc_vals],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        x=variants, y=recall_vals,
        name="Attack Recall",
        marker_color="#e67e22",
        text=[f"{v:.1%}" for v in recall_vals],
        textposition="outside",
    ))
    fig.update_layout(
        title="Ablation: AUC-ROC and Attack Recall per Architecture",
        barmode="group",
        height=400,
        yaxis=dict(range=[0, 1.1], gridcolor="#333"),
        margin=dict(t=50, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e0e0e0",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_feature_ranking(explanation: Optional[Dict[str, Any]]) -> None:
    """Render feature importance ranking from SHAP analysis."""
    if explanation is None:
        st.info("Feature importance not available (no explanation report).")
        return

    importance = explanation.get("feature_importance", [])
    if not importance:
        return

    st.markdown("#### Feature Importance Ranking (SHAP)")

    import pandas as pd
    rows = []
    for f in importance:
        feat = f["feature"]
        category = "Biometric" if feat in {
            "Temp", "SpO2", "Pulse_Rate", "SYS", "DIA",
            "Heart_rate", "Resp_Rate", "ST",
        } else "Network"
        rows.append({
            "Rank": f["rank"],
            "Feature": feat,
            "Category": category,
            "Mean |SHAP|": f"{f['mean_abs_shap']:.6f}",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_dataset_stats(preprocess: Dict[str, Any]) -> None:
    """Render dataset statistics summary."""
    st.markdown("#### Dataset Summary")

    ingestion = preprocess.get("ingestion", {})
    split = preprocess.get("split", {})
    redundancy = preprocess.get("redundancy", {})
    hipaa = preprocess.get("hipaa", {})

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Raw Samples", f"{ingestion.get('raw_rows', 16318):,}")
    with c2:
        st.metric("Features (final)", preprocess.get("output", {}).get("n_features", 29))
    with c3:
        st.metric("Train / Test Split", f"{split.get('train_ratio', 0.7):.0%} / {split.get('test_ratio', 0.3):.0%}")
    with c4:
        st.metric("HIPAA Columns Removed", len(hipaa.get("columns_dropped", [])))

    import pandas as pd
    pipeline_df = pd.DataFrame([
        {"Step": "Raw ingestion", "Detail": f"{ingestion.get('raw_rows', 0):,} rows x {ingestion.get('raw_columns', 0)} cols"},
        {"Step": "HIPAA de-identification", "Detail": f"-{len(hipaa.get('columns_dropped', []))} quasi-identifiers"},
        {"Step": "Redundancy removal", "Detail": f"-{redundancy.get('n_dropped', 0)} features (|r| > {redundancy.get('threshold', 0.95)})"},
        {"Step": "Stratified split", "Detail": f"Train: {split.get('train_samples', 0):,} / Test: {split.get('test_samples', 0):,}"},
        {"Step": "SMOTE oversampling", "Detail": f"{preprocess.get('smote', {}).get('samples_before', 0):,} -> {preprocess.get('smote', {}).get('samples_after', 0):,} (+{preprocess.get('smote', {}).get('synthetic_added', 0):,} synthetic)"},
        {"Step": "Scaling", "Detail": f"RobustScaler (median + IQR)"},
    ])
    st.dataframe(pipeline_df, use_container_width=True, hide_index=True)


def render() -> None:
    """Render the full Dataset Overview page."""
    st.header("Dataset Overview")
    st.caption("WUSTL-EHMS-2020 dataset analysis, ablation study, and feature importance")

    preprocess = load_preprocessing_report()
    if preprocess is None:
        st.error("Preprocessing report not available.")
        return

    render_dataset_stats(preprocess)
    st.divider()

    col1, col2 = st.columns([1, 1])
    with col1:
        render_class_distribution(preprocess)
    with col2:
        ablation = load_ablation_results()
        if ablation:
            render_ablation_study(ablation)
        else:
            st.warning("Ablation results not available.")

    st.divider()
    explanation = load_explanation_report()
    render_feature_ranking(explanation)
