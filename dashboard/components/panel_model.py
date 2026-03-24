"""Panel 2 — Model Architecture and Training.

Architecture summary, training progression (v1 vs v2),
hyperparameter configuration, and ablation study results.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import plotly.graph_objects as go
import streamlit as st


def render_architecture_summary(gt: Dict[str, Any]) -> None:
    """Render model architecture summary.

    Args:
        gt: Ground truth data.
    """
    arch = gt.get("model_architecture", {})

    if arch.get("status") not in ("VERIFIED", "PRESENT_UNVERIFIED"):
        st.info("Model architecture data not available — artifact not found")
        return

    if arch.get("status") == "PRESENT_UNVERIFIED":
        st.warning("Artifact present but hash unverified")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Parameters", f"{arch.get('total_params', 0):,}")
    with col2:
        has_att = arch.get("has_attention", False)
        st.metric("Attention Layer", "Present" if has_att else "Not found")
    with col3:
        has_bi = arch.get("has_bilstm", False)
        st.metric("BiLSTM Layer", "Present" if has_bi else "Not found")

    st.markdown(f"**Input shape:** `{arch.get('input_shape', 'N/A')}`")
    st.markdown(f"**Output shape:** `{arch.get('output_shape', 'N/A')}`")
    st.markdown(f"**Layers:** {arch.get('n_layers', 'N/A')}")

    # Layer table
    layers = arch.get("layers", [])
    if layers:
        with st.expander("Layer Details"):
            import pandas as pd
            df = pd.DataFrame(layers)
            st.dataframe(df, use_container_width=True, hide_index=True)


def render_training_progression(gt: Dict[str, Any]) -> None:
    """Render v1 vs v2 training progression chart.

    Args:
        gt: Ground truth data.
    """
    prog = gt.get("training_progression", {})

    if prog.get("status") not in ("VERIFIED", "PRESENT_UNVERIFIED"):
        st.info("Training progression data not available")
        return

    v1 = prog.get("v1_baseline", {})
    v2 = prog.get("v2_fixed", {})

    metrics = ["AUC", "Attack Recall", "F1"]
    v1_vals = [v1.get("auc", 0) or 0, v1.get("attack_recall", 0) or 0,
               v1.get("f1", 0) or 0]
    v2_vals = [v2.get("auc", 0) or 0, v2.get("attack_recall", 0) or 0,
               v2.get("f1", 0) or 0]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=metrics, y=v1_vals,
        name="v1 Baseline", marker_color="#e74c3c",
        text=[f"{v:.4f}" for v in v1_vals],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        x=metrics, y=v2_vals,
        name="v2 Class-Weighted", marker_color="#2ecc71",
        text=[f"{v:.4f}" for v in v2_vals],
        textposition="outside",
    ))

    fig.update_layout(
        title="Training Progression: v1 Baseline vs v2 Class-Weighted",
        yaxis_title="Score",
        barmode="group",
        height=400,
        margin=dict(t=60, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e0e0e0",
        yaxis=dict(range=[0, 1.05]),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Delta annotations
    hp = gt.get("hyperparameters", {})
    val_auc = hp.get("val_auc", 0)
    if val_auc:
        st.caption(
            f"Bayesian search confirmed Phase 2 defaults optimal "
            f"(val_AUC={val_auc})"
        )

    recall_improvement = prog.get("attack_recall_improvement_pct", 0)
    if recall_improvement:
        st.caption(f"Attack recall improvement: +{recall_improvement:.1f}%")


def render_hyperparameter_config(gt: Dict[str, Any]) -> None:
    """Render hyperparameter configuration table.

    Args:
        gt: Ground truth data.
    """
    hp = gt.get("hyperparameters", {})

    if hp.get("status") not in ("VERIFIED", "PRESENT_UNVERIFIED"):
        st.info("Hyperparameter data not available — artifact not found")
        return

    optimal = hp.get("optimal", {})
    prog = gt.get("training_progression", {})
    class_weight = prog.get("class_weight", {})
    perf = gt.get("performance", {})
    threshold = perf.get("optimal_threshold", 0.608)

    import pandas as pd
    params = [
        {"Parameter": "timesteps", "Value": str(optimal.get("timesteps", "N/A")),
         "Source": "Bayesian/Default"},
        {"Parameter": "cnn_filters", "Value": str(optimal.get("cnn_filters", "N/A")),
         "Source": "Bayesian/Default"},
        {"Parameter": "bilstm_units", "Value": str(optimal.get("bilstm_units", "N/A")),
         "Source": "Bayesian/Default"},
        {"Parameter": "dropout_rate", "Value": str(optimal.get("dropout_rate", "N/A")),
         "Source": "Bayesian/Default"},
        {"Parameter": "learning_rate", "Value": str(optimal.get("learning_rate", "N/A")),
         "Source": "Bayesian/Default"},
        {"Parameter": "class_weight", "Value": str(class_weight.get("1", "N/A")),
         "Source": "Computed"},
        {"Parameter": "threshold", "Value": f"{threshold:.3f}",
         "Source": "Youden's J"},
    ]

    df = pd.DataFrame(params)
    st.dataframe(df, use_container_width=True, hide_index=True)

    conclusion = hp.get("tuning_conclusion", "")
    if conclusion:
        st.caption(conclusion)

    # GPU info
    if hp.get("gpu_used"):
        st.caption(
            f"Tuned on GPU ({hp.get('gpu_name', 'N/A')}) | "
            f"{hp.get('tuning_trials', 20)} trials | "
            f"{hp.get('tuning_time_s', 0):.0f}s"
        )


def render_ablation_study(gt: Dict[str, Any]) -> None:
    """Render ablation study results chart.

    Args:
        gt: Ground truth data.
    """
    abl = gt.get("ablation", {})

    if abl.get("status") not in ("VERIFIED", "PRESENT_UNVERIFIED"):
        st.info("Ablation data not available — artifact not found")
        return

    a = abl.get("model_a_cnn", {})
    b = abl.get("model_b_bilstm", {})
    c = abl.get("model_c_full", {})

    variants = ["A: CNN only", "B: CNN+BiLSTM", "C: CNN+BiLSTM+Attention"]
    aucs = [a.get("auc", 0), b.get("auc", 0), c.get("auc", 0)]
    recalls = [a.get("attack_recall", 0), b.get("attack_recall", 0),
               c.get("attack_recall", 0)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=variants, y=aucs,
        name="AUC", marker_color="#3498db",
        text=[f"{v:.4f}" for v in aucs],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        x=variants, y=recalls,
        name="Attack Recall", marker_color="#e67e22",
        text=[f"{v:.4f}" for v in recalls],
        textposition="outside",
    ))

    fig.update_layout(
        title="Ablation Study",
        yaxis_title="Score",
        barmode="group",
        height=400,
        margin=dict(t=60, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e0e0e0",
        yaxis=dict(range=[0, 1.1]),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Transition annotations
    bilstm_gain = abl.get("bilstm_auc_gain", 0)
    att_gain = abl.get("attention_auc_gain", 0)
    att_recall = abl.get("attention_recall_delta", 0)

    st.caption(
        f"A→B: BiLSTM adds +{bilstm_gain:.4f} AUC | "
        f"B→C: Attention adds +{att_gain:.4f} AUC "
        f"+ {att_recall:+.4f} attack recall + XAI capability"
    )


def render(gt: Dict[str, Any]) -> None:
    """Render the full Model Architecture and Training panel.

    Args:
        gt: Ground truth data.
    """
    st.header("Model Architecture & Training")

    render_architecture_summary(gt)

    st.divider()
    st.subheader("Training Progression")
    render_training_progression(gt)

    st.divider()
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Hyperparameter Configuration")
        render_hyperparameter_config(gt)
    with col2:
        st.subheader("Ablation Study")
        render_ablation_study(gt)
