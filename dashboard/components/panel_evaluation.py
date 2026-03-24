"""Panel 3 — Evaluation Results.

Confusion matrix, threshold sensitivity analysis,
and performance metric overview.
"""

from __future__ import annotations

from typing import Any, Dict

import plotly.graph_objects as go
import streamlit as st


def render_core_metrics(gt: Dict[str, Any]) -> None:
    """Render core performance metric cards with v1 deltas.

    Args:
        gt: Ground truth data.
    """
    perf = gt.get("performance", {})
    prog = gt.get("training_progression", {})

    if perf.get("status") not in ("VERIFIED", "PRESENT_UNVERIFIED"):
        st.info("Performance metrics not available — artifact not found")
        return

    if perf.get("status") == "PRESENT_UNVERIFIED":
        st.warning("Artifact present but hash unverified")

    v1 = prog.get("v1_baseline", {})

    # Compute deltas from ground truth
    auc_delta = prog.get("auc_delta")
    v1_recall = v1.get("attack_recall", 0) or 0
    v2_recall = perf.get("attack_recall", 0) or 0
    recall_delta = v2_recall - v1_recall if v1_recall else None

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "AUC-ROC",
            f"{perf.get('auc_roc', 0):.4f}",
            delta=f"+{auc_delta:.4f} vs v1" if auc_delta else None,
        )
    with col2:
        st.metric(
            "Attack Recall",
            f"{v2_recall:.1%}",
            delta=f"+{recall_delta:.1%} vs v1" if recall_delta else None,
        )
    with col3:
        st.metric(
            "F1 (weighted)",
            f"{perf.get('f1_weighted', 0):.4f}",
        )
    with col4:
        st.metric(
            "Optimal Threshold",
            f"{perf.get('optimal_threshold', 0):.3f}",
            delta="Youden's J",
            delta_color="off",
        )

    # Naive baseline comparison
    dataset = gt.get("dataset", {})
    naive = dataset.get("naive_baseline_accuracy", 0)
    model_acc = perf.get("accuracy", 0)
    beats_naive = prog.get("beats_naive_baseline", model_acc > naive)

    st.caption(
        f"Naive baseline (always-Normal): {naive:.1%} accuracy | "
        f"Model accuracy: {model_acc:.1%} | "
        f"Model beats naive: {'Yes' if beats_naive else 'No'}"
    )
    if not beats_naive:
        st.warning(
            "Model does not beat naive baseline on raw accuracy — "
            "this reflects intentional class-weight optimization "
            "for attack recall."
        )


def render_confusion_matrix(gt: Dict[str, Any]) -> None:
    """Render annotated confusion matrix heatmap.

    Args:
        gt: Ground truth data.
    """
    perf = gt.get("performance", {})
    cm = perf.get("confusion_matrix", {})

    if not cm:
        st.info("Confusion matrix data not available")
        return

    tn = cm.get("TN", 0)
    fp = cm.get("FP", 0)
    fn = cm.get("FN", 0)
    tp = cm.get("TP", 0)

    z = [[tn, fp], [fn, tp]]
    text = [[f"TN={tn}", f"FP={fp}"], [f"FN={fn}", f"TP={tp}"]]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=["Predicted Normal", "Predicted Attack"],
        y=["Actual Normal", "Actual Attack"],
        text=text,
        texttemplate="%{text}",
        textfont={"size": 16},
        colorscale=[
            [0.0, "rgba(46,204,113,0.27)"],
            [0.5, "rgba(243,156,18,0.27)"],
            [1.0, "rgba(231,76,60,0.27)"],
        ],
        showscale=False,
    ))

    fig.update_layout(
        title=f"Confusion Matrix (N={perf.get('test_samples', '?')} | "
              f"Threshold={perf.get('optimal_threshold', 0):.3f})",
        height=350,
        margin=dict(t=60, b=40, l=120),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e0e0e0",
        yaxis=dict(autorange="reversed"),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_threshold_sensitivity(gt: Dict[str, Any]) -> None:
    """Render threshold sensitivity multi-line chart.

    Args:
        gt: Ground truth data.
    """
    perf = gt.get("performance", {})
    sensitivity = perf.get("threshold_sensitivity", [])

    if not sensitivity:
        st.info("Threshold sensitivity data not available")
        return

    thresholds = [s["threshold"] for s in sensitivity]
    recalls = [s.get("attack_recall", 0) for s in sensitivity]
    f1s = [s.get("f1", 0) for s in sensitivity]
    accuracies = [s.get("accuracy", 0) for s in sensitivity]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=thresholds, y=recalls,
        mode="lines+markers", name="Attack Recall",
        line=dict(color="#e74c3c", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=thresholds, y=f1s,
        mode="lines+markers", name="F1 Score",
        line=dict(color="#3498db", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=thresholds, y=accuracies,
        mode="lines+markers", name="Accuracy",
        line=dict(color="#2ecc71", width=2),
    ))

    # Optimal threshold line
    opt = perf.get("optimal_threshold", 0.608)
    fig.add_vline(
        x=opt, line_dash="dash", line_color="#f39c12",
        annotation_text=f"Optimal: {opt:.3f}",
    )

    fig.update_layout(
        title="Threshold Sensitivity Analysis",
        xaxis_title="Threshold",
        yaxis_title="Score",
        height=400,
        margin=dict(t=60, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e0e0e0",
        yaxis=dict(range=[0, 1.05]),
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Method: {perf.get('threshold_method', 'Youdens J statistic')}")


def render(gt: Dict[str, Any]) -> None:
    """Render the full Evaluation Results panel.

    Args:
        gt: Ground truth data.
    """
    st.header("Evaluation Results")
    render_core_metrics(gt)

    st.divider()
    col1, col2 = st.columns([1, 1])
    with col1:
        render_confusion_matrix(gt)
    with col2:
        render_threshold_sensitivity(gt)
