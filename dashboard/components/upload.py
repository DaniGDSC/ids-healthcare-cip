"""Page 4 - Upload & Predict: CSV upload, HIPAA check, inference, risk scoring."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.utils.loader import load_feature_names
from dashboard.utils.predictor import (
    detect_pii_columns,
    predict,
    preprocess_upload,
    score_risk,
)

RISK_COLORS = {
    "NORMAL": "#2ecc71",
    "LOW": "#3498db",
    "MEDIUM": "#f39c12",
    "HIGH": "#e67e22",
    "CRITICAL": "#e74c3c",
}


def render_pii_warning(pii_cols: List[str]) -> None:
    """Display HIPAA PII warning for detected columns."""
    st.error(
        f"**HIPAA Warning:** {len(pii_cols)} potential PII column(s) detected: "
        f"**{', '.join(pii_cols)}**. These columns will NOT be processed. "
        "Please remove PII before uploading."
    )


def render_results_table(results: List[Dict[str, Any]]) -> None:
    """Display prediction results as a styled DataFrame."""
    df = pd.DataFrame(results)

    # Summary counts
    counts = df["risk_level"].value_counts().to_dict()
    n_attacks = sum(v for k, v in counts.items() if k != "NORMAL")
    n_normal = counts.get("NORMAL", 0)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Predictions", len(df))
    with c2:
        st.metric("Attacks Detected", n_attacks)
    with c3:
        st.metric("Normal", n_normal)

    # Risk distribution mini chart
    levels = ["NORMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
    level_counts = [counts.get(l, 0) for l in levels]
    colors = [RISK_COLORS[l] for l in levels]

    fig = go.Figure(go.Bar(
        x=levels,
        y=level_counts,
        marker_color=colors,
        text=level_counts,
        textposition="outside",
    ))
    fig.update_layout(
        title="Risk Distribution (Uploaded Data)",
        height=280,
        margin=dict(t=50, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e0e0e0",
        yaxis=dict(gridcolor="#333"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Detailed Results")
    st.dataframe(
        df[["sample_index", "anomaly_score", "threshold", "distance", "risk_level"]],
        use_container_width=True,
        hide_index=True,
        height=400,
    )

    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name="rax_iomt_predictions.csv",
        mime="text/csv",
    )


def render() -> None:
    """Render the full Upload & Predict page."""
    st.header("Upload & Predict")
    st.caption("Upload a CSV with 29 WUSTL-EHMS features for real-time inference")

    st.markdown(
        "**Required features:** " + ", ".join(f"`{f}`" for f in load_feature_names())
    )

    uploaded = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        help="CSV must contain the 29 WUSTL-EHMS features (case-insensitive column matching)",
    )

    if uploaded is None:
        st.info("Upload a CSV file to begin prediction.")
        return

    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return

    st.success(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Step 1: HIPAA PII check
    pii_cols = detect_pii_columns(df)
    if pii_cols:
        render_pii_warning(pii_cols)
        df = df.drop(columns=pii_cols, errors="ignore")
        st.info(f"Dropped {len(pii_cols)} PII column(s) before processing.")

    # Step 2-3: Preprocess and reshape
    with st.spinner("Preprocessing: scaling and reshaping..."):
        X_windows = preprocess_upload(df)

    if X_windows is None:
        return

    st.info(f"Created {len(X_windows)} windows of shape (20, 29)")

    # Step 4: Predict
    with st.spinner("Running model inference..."):
        probs = predict(X_windows)

    # Step 5: Risk scoring
    with st.spinner("Computing risk scores (MAD-based)..."):
        results = score_risk(probs)

    # Step 6: Display
    render_results_table(results)
