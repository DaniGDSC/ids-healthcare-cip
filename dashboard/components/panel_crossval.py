"""Panel — Cross-Dataset Validation & Live Simulation.

CICIoMT2024 results, MedSec-25 status, generalization gap analysis,
and live framework simulation injection controls.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import plotly.graph_objects as go
import streamlit as st


def render_ciciomt_section(gt: Dict[str, Any]) -> None:
    """Render CICIoMT2024 discontinued validation section.

    Args:
        gt: Ground truth data.
    """
    cross = gt.get("cross_dataset", {})
    cic = cross.get("ciciomt2024", {})

    status = cic.get("status", "NOT_AVAILABLE")
    st.markdown(
        f'<span style="background:#e74c3c; color:white; padding:2px 8px; '
        f'border-radius:3px; font-size:0.8em; font-weight:bold;">'
        f'{status}</span>',
        unsafe_allow_html=True,
    )

    reason = cic.get("reason", "Feature mismatch — details unavailable")
    st.caption(f"Reason: {reason}")

    if cic.get("auc") is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("AUC", f"{cic['auc']:.4f}")
        with col2:
            recall = cic.get("attack_recall", 0)
            st.metric("Attack Recall", f"{recall:.4f}" if recall else "N/A")
        with col3:
            st.metric("Test Samples", f"{cic.get('test_samples', 'N/A'):,}")

        mapped = cic.get("mapped_features", 5)
        imputed = cic.get("imputed_features", 24)
        st.caption(
            f"Feature overlap: {mapped}/29 mapped, {imputed}/29 imputed"
        )


def render_medsec25_section(gt: Dict[str, Any]) -> None:
    """Render MedSec-25 validation status section.

    Args:
        gt: Ground truth data.
    """
    cross = gt.get("cross_dataset", {})
    ms = cross.get("medsec25", {})

    status = ms.get("status", "PENDING")
    color = "#3498db" if status == "PENDING" else "#f39c12"
    st.markdown(
        f'<span style="background:{color}; color:white; padding:2px 8px; '
        f'border-radius:3px; font-size:0.8em; font-weight:bold;">'
        f'{status}</span>',
        unsafe_allow_html=True,
    )

    if ms.get("csv_available"):
        st.success("MedSec-25.csv available (554,534 flows)")
    else:
        st.warning("MedSec-25.csv not found in data/external/")

    note = ms.get("note", "")
    if note:
        st.caption(note)


def render_generalization_gap(gt: Dict[str, Any]) -> None:
    """Render generalization gap analysis table and chart.

    Args:
        gt: Ground truth data.
    """
    ablation = gt.get("ablation", {})
    perf = gt.get("performance", {})

    if ablation.get("status") not in ("VERIFIED", "PRESENT_UNVERIFIED"):
        st.info("Ablation data not available for gap analysis")
        return

    full = ablation.get("model_c_full", {})
    val_auc = full.get("auc", 0)
    val_recall = full.get("attack_recall", 0)
    test_auc = perf.get("auc_roc", 0)
    test_recall = perf.get("attack_recall", 0)

    import pandas as pd
    gap_data = pd.DataFrame([
        {
            "Metric": "AUC",
            "SMOTE Validation": f"{val_auc:.4f}",
            "Test Set": f"{test_auc:.4f}",
            "Delta": f"{(test_auc - val_auc) / val_auc * 100:+.1f}%"
            if val_auc > 0 else "N/A",
        },
        {
            "Metric": "Attack Recall",
            "SMOTE Validation": f"{val_recall:.4f}",
            "Test Set": f"{test_recall:.4f}",
            "Delta": f"{(test_recall - val_recall) / val_recall * 100:+.1f}%"
            if val_recall > 0 else "N/A",
        },
    ])

    st.dataframe(gap_data, use_container_width=True, hide_index=True)

    st.caption(
        "Validation gap attributable to: "
        "(1) SMOTE distribution shift, "
        "(2) threshold sensitivity, "
        "(3) feature space overlap constraints"
    )


def render_simulation_controls(
    sim_status: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Render live simulation injection controls.

    Args:
        sim_status: Current simulator status.

    Returns:
        Dictionary of user-selected simulation parameters.
    """
    st.markdown("##### Live Framework Simulation")

    result: Dict[str, Any] = {"action": None}

    if sim_status and sim_status.get("medsec25_available"):
        st.success("MedSec-25.csv ready for injection")

        mode = st.selectbox(
            "Timing Mode",
            ["ACCELERATED", "REALTIME", "STRESS"],
            index=0,
            help="ACCELERATED=10x speed, REALTIME=1x, STRESS=max throughput",
        )
        result["mode"] = mode

        scenario = st.selectbox(
            "Scenario",
            ["A: Benign Only", "B: Gradual Attack",
             "C: Abrupt Attack", "D: Mixed Cycle"],
            index=2,
        )
        result["scenario"] = scenario[0]  # Extract letter

        is_running = sim_status.get("running", False)

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Start Simulation", disabled=is_running,
                         type="primary"):
                result["action"] = "start"
        with col2:
            if st.button("Stop Simulation", disabled=not is_running):
                result["action"] = "stop"
        with col3:
            if st.button("Reset"):
                result["action"] = "reset"

        if is_running:
            flows = sim_status.get("flows_injected", 0)
            phase = sim_status.get("current_phase", "")
            st.info(
                f"Simulating: Scenario {sim_status.get('scenario', '?')} | "
                f"{sim_status.get('mode', 'ACCELERATED')} | "
                f"{flows:,} flows\n\n"
                f"Phase: {phase}"
            )
    else:
        st.info(
            "MedSec-25.csv not found. Place the dataset in "
            "`data/external/MedSec-25.csv` to enable live simulation."
        )

    return result


def render(
    gt: Dict[str, Any],
    sim_status: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Render the full Cross-Dataset Validation panel.

    Args:
        gt: Ground truth data.
        sim_status: Simulator status.

    Returns:
        Simulation control actions.
    """
    st.header("Cross-Dataset Validation")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("CICIoMT2024")
        render_ciciomt_section(gt)
    with col2:
        st.subheader("MedSec-25")
        render_medsec25_section(gt)

    st.divider()
    st.subheader("Generalization Gap Analysis")
    render_generalization_gap(gt)

    st.divider()
    return render_simulation_controls(sim_status)
