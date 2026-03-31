"""Panel — System & Compliance (merged static reference panels).

Tabs: Model Architecture | Evaluation Metrics | Compliance & Audit | History
"""

from __future__ import annotations

import csv
import io
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def render(gt: Dict[str, Any]) -> None:
    """Render system reference panel with tabs."""
    st.header("System & Compliance")

    tab_model, tab_eval, tab_compliance, tab_history = st.tabs([
        "Model Architecture", "Evaluation Metrics", "Compliance & Audit", "History",
    ])

    with tab_model:
        _render_model(gt)

    with tab_eval:
        _render_evaluation(gt)

    with tab_compliance:
        _render_compliance(gt)

    with tab_history:
        _render_history()


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


def _render_history() -> None:
    """Historical alert query panel backed by SQLite database."""
    db = st.session_state.get("database")
    if db is None:
        st.info("Database not initialized — history unavailable")
        return

    st.markdown("##### Alert History")

    # Filters
    col_date, col_risk = st.columns(2)
    with col_date:
        today = date.today()
        week_ago = today - timedelta(days=7)
        date_range = st.date_input("Date range", value=(week_ago, today))
    with col_risk:
        risk_filter = st.multiselect(
            "Risk level",
            ["HIGH", "CRITICAL"],
            default=["HIGH", "CRITICAL"],
        )

    # Build query params
    since_str = None
    until_str = None
    if isinstance(date_range, tuple) and len(date_range) == 2:
        since_str = datetime.combine(date_range[0], datetime.min.time()).isoformat() + "Z"
        until_str = datetime.combine(date_range[1], datetime.max.time()).isoformat() + "Z"

    # Query
    all_alerts: List[Dict] = []
    for level in (risk_filter or ["HIGH", "CRITICAL"]):
        alerts = db.query_alerts(since=since_str, until=until_str, risk_level=level, limit=500)
        all_alerts.extend(alerts)
    all_alerts.sort(key=lambda a: a.get("time", ""), reverse=True)

    # Summary metrics
    total = len(all_alerts)
    ack_count = sum(1 for a in all_alerts if a.get("acknowledged"))
    pending = total - ack_count

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Alerts", total)
    m2.metric("Acknowledged", ack_count)
    m3.metric("Pending", pending)

    if not all_alerts:
        st.info("No alerts found for the selected filters")
        return

    # Results table
    for a in all_alerts[:100]:
        time_str = a.get("time", "")[:19]
        risk = a.get("risk_level", "?")
        sev = a.get("clinical_severity", "?")
        device = a.get("device_id_hash", "—")[:8]
        ack = "Yes" if a.get("acknowledged") else "No"
        ack_by = a.get("acknowledged_by", "—") if a.get("acknowledged") else "—"

        c1, c2, c3, c4, c5, c6 = st.columns([3, 2, 1, 2, 1, 2])
        c1.caption(time_str)
        c2.caption(risk)
        c3.caption(str(sev))
        c4.caption(device)
        c5.caption(ack)
        c6.caption(ack_by)

    # Export CSV
    if all_alerts:
        csv_buf = io.StringIO()
        fields = ["time", "risk_level", "clinical_severity", "device_id_hash",
                   "alert_emit", "alert_reason", "acknowledged", "acknowledged_by"]
        writer = csv.DictWriter(csv_buf, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for a in all_alerts:
            writer.writerow(a)

        st.download_button(
            label="Export Alerts (CSV)",
            data=csv_buf.getvalue(),
            file_name="alert_history.csv",
            mime="text/csv",
        )
