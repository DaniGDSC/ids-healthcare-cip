"""Panel — System & Compliance (merged static reference panels).

Tabs: Model Architecture | Evaluation Metrics | Compliance & Audit
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def render(gt: Dict[str, Any]) -> None:
    """Render system reference panel with tabs."""
    st.header("System & Compliance")

    tab_model, tab_eval, tab_compliance = st.tabs([
        "Model Architecture", "Evaluation Metrics", "Compliance & Audit",
    ])

    with tab_model:
        _render_model(gt)

    with tab_eval:
        _render_evaluation(gt)

    with tab_compliance:
        _render_compliance(gt)


def _render_model(gt: Dict[str, Any]) -> None:
    """Model architecture and training summary."""
    arch = gt.get("model_architecture", {})
    st.markdown("##### CNN-BiLSTM-Attention Architecture")
    st.markdown(f"- **Parameters:** {arch.get('total_params', 0):,}")
    st.markdown(f"- **Input shape:** (batch, 20, 24)")
    st.markdown(f"- **Output:** Sigmoid (binary classification)")

    # Training info from Phase 2.5
    ft_path = PROJECT_ROOT / "data" / "phase2_5" / "finetuned_results.json"
    if ft_path.exists():
        with open(ft_path) as f:
            ft = json.load(f)
        st.markdown("##### Training (Phase 2.5)")
        hp = ft.get("best_hyperparameters", {})
        st.json(hp)

        history = ft.get("training_history", [])
        if history:
            st.markdown("##### Training History")
            for h in history:
                st.markdown(f"- **{h['phase']}**: {h['epochs_run']} epochs, "
                            f"val_loss={h.get('final_val_loss', 0):.4f}")


def _render_evaluation(gt: Dict[str, Any]) -> None:
    """Evaluation metrics and baseline comparison."""
    ft_path = PROJECT_ROOT / "data" / "phase2_5" / "finetuned_results.json"
    if not ft_path.exists():
        st.info("Phase 2.5 results not available")
        return

    with open(ft_path) as f:
        ft = json.load(f)

    st.markdown("##### Test Set Metrics")
    test = ft.get("test_metrics", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AUC-ROC", f"{test.get('auc_roc', 0):.4f}")
    c2.metric("Attack Recall", f"{test.get('attack_recall', 0):.4f}")
    c3.metric("Attack Precision", f"{test.get('attack_precision', 0):.4f}")
    c4.metric("Attack F1", f"{test.get('attack_f1', 0):.4f}")

    # Baseline comparison
    comparison = ft.get("baseline_comparison", {})
    if comparison:
        st.markdown("##### vs Baseline")
        for metric, vals in comparison.items():
            bl = vals.get("baseline", 0)
            ft_val = vals.get("finetuned", 0)
            delta = ft_val - bl
            st.markdown(f"- **{metric}**: {bl:.4f} → {ft_val:.4f} ({delta:+.4f})")


def _render_compliance(gt: Dict[str, Any]) -> None:
    """Compliance status and audit trail."""
    st.markdown("##### HIPAA Compliance")
    st.markdown("- PHI columns removed (SrcAddr, DstAddr, Sport, Dport, SrcMac, DstMac)")
    st.markdown("- Device IDs pseudonymized (SHA-256, 12-char prefix)")
    st.markdown("- Email bodies contain risk level + timestamp only")
    st.markdown("- Biometric values never transmitted in alerts")

    st.markdown("##### FDA 21 CFR Part 11 Audit")
    audit_path = PROJECT_ROOT / "data" / "audit" / "fda_audit.jsonl"
    if audit_path.exists():
        from src.production.audit_logger import FDAAuditLogger
        logger = FDAAuditLogger(audit_path)
        result = logger.verify_chain()
        if result["is_valid"]:
            st.success(f"Audit chain valid: {result['entries_checked']} entries verified")
        else:
            st.error(f"Audit chain BROKEN at entry {result.get('first_broken', '?')}")
        recent = logger.get_recent(5)
        if recent:
            st.markdown("##### Recent Audit Entries")
            for entry in recent:
                st.caption(f"[{entry['ts']}] {entry['event']} by {entry['actor']}")
    else:
        st.info("No audit log yet — will be created during streaming")

    # Artifact integrity
    st.markdown("##### Artifact Integrity")
    inv = gt.get("artifact_inventory", {})
    verified = inv.get("verified", 0)
    total = inv.get("total_artifacts_checked", 0)
    missing = inv.get("missing", 0)
    st.markdown(f"- **Verified:** {verified}/{total}")
    st.markdown(f"- **Missing:** {missing}")
