#!/usr/bin/env python3
"""Module 6 — Evaluation Interface (Tasks 6.3a/b/c, 6.4, 6.5, 6.9).

Three modes:
  6.3a  Offline — Browse/Study pre-computed alerts with Likert questionnaires
  6.3b  Online Simulation — Stream test samples through pipeline in near-real-time
  6.3c  Dashboard — Risk gauge, alert feed, SHAP waterfall, NLG panel, response panel,
        admin heatmap, tier distribution chart

Usage:
    streamlit run pipeline/module6_evaluation/module6_app.py
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = PROJECT_ROOT / "results/reports"
CHARTS_DIR = PROJECT_ROOT / "results/charts"
MODELS_DIR = PROJECT_ROOT / "results/models"

ROLES = ["Security Analyst", "Clinician", "Administrator"]
ACTIONS = ["dismiss", "monitor", "investigate", "isolate", "escalate"]

TIER_COLORS = {"CRITICAL": "#8e44ad", "HIGH": "#e74c3c", "MEDIUM": "#e67e22", "LOW": "#2ecc71"}

BIOMETRIC_FEATURES = {"Temp", "SpO2", "Pulse_Rate", "SYS", "DIA", "Heart_rate", "Resp_Rate", "ST"}


# ═══════════════════════════════════════════════════════════════════════
# 6A.7  Audit Trail Writer (JSONL, immutable with integrity hashes)
# ═══════════════════════════════════════════════════════════════════════

class AuditTrailWriter:
    """Append-only JSONL audit log for every user interaction."""

    def __init__(self, path: Path | None = None):
        self.path = path or (EVAL_DIR / "audit_trail.jsonl")
        self.prev_hash = "0" * 64

    def log(self, event_type: str, **kwargs) -> None:
        record = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            **kwargs,
            "prev_hash": self.prev_hash,
        }
        payload = json.dumps(record, sort_keys=True)
        record["integrity_hash"] = hashlib.sha256(payload.encode()).hexdigest()
        self.prev_hash = record["integrity_hash"]
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")


# Singleton audit writer for the Streamlit session
_audit_writer = AuditTrailWriter()


def audit_log(event_type: str, **kwargs) -> None:
    """Log an interaction event to the audit trail."""
    _audit_writer.log(event_type, **kwargs)


# ═══════════════════════════════════════════════════════════════════════
# 6A.5  Reusable Likert questionnaire component
# ═══════════════════════════════════════════════════════════════════════

def likert_form(alert_id: str, form_key: str) -> dict | None:
    """Reusable 5-point Likert × 4 dimensions + action + free-text.

    Returns a dict of responses on submit, or None if not yet submitted.
    """
    with st.form(form_key):
        st.markdown("#### Your Response")
        action = st.selectbox("What action would you take?", ACTIONS,
                              format_func=lambda x: x.capitalize())
        confidence = st.slider("Confidence in your decision (1–5)", 1, 5, 3)

        st.markdown("#### Rate the alert presentation (1 = strongly disagree, 5 = strongly agree)")
        trust = st.slider("I trust this classification", 1, 5, 3,
                           key=f"lt_{form_key}")
        usefulness = st.slider("The information helps me respond appropriately", 1, 5, 3,
                                key=f"lu_{form_key}")
        comprehensibility = st.slider("I understand why this alert was triggered", 1, 5, 3,
                                       key=f"lc_{form_key}")
        actionability = st.slider("I know what action to take", 1, 5, 3,
                                   key=f"la_{form_key}")

        feedback = st.text_area("Free-text feedback (optional)", key=f"fb_{form_key}")
        reclass = st.selectbox("Reclassify tier?",
                               ["No change", "CRITICAL", "HIGH", "MEDIUM", "LOW", "Benign/Dismiss"],
                               key=f"rc_{form_key}")

        if st.form_submit_button("Submit & Next"):
            return {
                "alert_id": alert_id,
                "chosen_action": action,
                "confidence": confidence,
                "likert_trust": trust,
                "likert_usefulness": usefulness,
                "likert_comprehensibility": comprehensibility,
                "likert_actionability": actionability,
                "feedback": feedback,
                "reclassification": reclass if reclass != "No change" else None,
            }
    return None


# ═══════════════════════════════════════════════════════════════════════
# 6B.3  A/B condition assignment (counterbalanced)
# ═══════════════════════════════════════════════════════════════════════

def assign_ab_conditions(n_alerts: int, participant_id: str) -> list[bool]:
    """Counterbalanced A/B assignment: half with XAI, half without.

    Uses participant_id as seed so the same participant always gets
    the same assignment, but different participants get different
    orderings.  Latin-square style: even PIDs get XAI-first,
    odd PIDs get no-XAI-first.
    """
    seed = int(hashlib.md5(participant_id.encode()).hexdigest(), 16) % (2**31)
    rng = random.Random(seed)

    # Build balanced list: exactly half True, half False
    half = n_alerts // 2
    conditions = [True] * half + [False] * (n_alerts - half)

    # Determine block order from PID parity
    pid_num = sum(ord(c) for c in participant_id)
    if pid_num % 2 == 0:
        # XAI-first block
        pass
    else:
        # Reverse: no-XAI first
        conditions = conditions[::-1]

    # Shuffle within each block to avoid position effects
    block1 = conditions[:half]
    block2 = conditions[half:]
    rng.shuffle(block1)
    rng.shuffle(block2)
    return block1 + block2


# ═══════════════════════════════════════════════════════════════════════
# 6C.1  Streaming data simulator
# ═══════════════════════════════════════════════════════════════════════

def stream_simulator(responses: list, delay: float = 1.0):
    """Generator yielding test samples with configurable delay.

    Yields one alert dict at a time, simulating real-time arrival.
    """
    for r in responses:
        yield r
        time.sleep(delay)


# ═══════════════════════════════════════════════════════════════════════
# 6C.9  Online interaction capture (JSONL)
# ═══════════════════════════════════════════════════════════════════════

def capture_online_interaction(
    participant_id: str,
    alert_id: str | int,
    action_type: str,
    details: dict | None = None,
) -> None:
    """Log confirm/reject, reclassifications, feedback with timestamps."""
    record = {
        "timestamp": datetime.now().isoformat(),
        "participant_id": participant_id,
        "alert_id": alert_id,
        "action_type": action_type,
        "details": details or {},
    }
    path = EVAL_DIR / "online_interactions.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
    # Also write to audit trail
    audit_log("online_interaction", **record)


# ═══════════════════════════════════════════════════════════════════════
# 6A.3  process_alert() — end-to-end sample → structured alert object
# ═══════════════════════════════════════════════════════════════════════

def process_alert(sample_index: int, alert_data: dict) -> dict:
    """Take a raw alert record and produce a fully structured alert object.

    In production, this would run Modules 2-5 live. Here it assembles
    from pre-computed artifacts (risk scores, SHAP, NLG, responses).
    """
    xai = alert_data.get("xai_explanation", {})
    return {
        "sample_index": sample_index,
        "prediction": 1 if alert_data.get("risk_score", 0) >= 0.4 else 0,
        "confidence": alert_data.get("risk_score", 0),
        "risk_score": alert_data.get("risk_score", 0),
        "tier": alert_data.get("risk_level", "LOW"),
        "attack_category": alert_data.get("attack_category", "unknown"),
        "ground_truth": alert_data.get("ground_truth", "unknown"),
        "shap_top_features": xai.get("xgboost_top_features", []),
        "dae_top_features": xai.get("dae_top_features", []),
        "nlg_text": xai.get("clinician_summary", ""),
        "consensus": xai.get("consensus", ""),
        "response_action": alert_data.get("correct_action", "monitor"),
    }


# ═══════════════════════════════════════════════════════════════════════
# 6A.4  Stakeholder view renderers
# ═══════════════════════════════════════════════════════════════════════

def render_analyst(alert: dict):
    """Analyst view: SHAP plots + feature table + classification detail."""
    st.markdown("#### Security Analyst View")

    # SHAP waterfall
    idx = alert.get("sample_index", 0)
    chart = CHARTS_DIR / f"waterfall_xgboost_sample_{idx:04d}.png"
    if chart.exists():
        st.image(str(chart), caption="SHAP Waterfall", use_container_width=True)

    # Force plot
    force = CHARTS_DIR / f"force_xgboost_sample_{idx:04d}.png"
    if force.exists():
        st.image(str(force), caption="SHAP Force Plot", use_container_width=True)

    # Top features table
    feats = alert.get("shap_top_features", [])
    if feats:
        st.markdown("**Top SHAP Features:**")
        rows = []
        for f in feats[:5]:
            rows.append({
                "Feature": f["feature"],
                "SHAP Value": f"{f.get('shap_value', 0):+.4f}",
                "Direction": f.get("direction", ""),
                "Type": "Biometric" if f["feature"] in BIOMETRIC_FEATURES else "Network",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # DAE indicators
    dae = alert.get("dae_top_features", [])
    if dae:
        st.markdown("**DAE Anomaly Features:**")
        for f in dae[:3]:
            st.text(f"  {f['feature']}: {f.get('pct_contribution', 0):.1f}% contribution")

    st.markdown(f"**Consensus:** {alert.get('consensus', 'N/A')}")


def render_clinician(alert: dict):
    """Clinician view: plain-language NLG summary + biometric safety notes."""
    st.markdown("#### Clinician View")

    nlg = alert.get("nlg_text", "")
    if nlg:
        st.warning(nlg)
    else:
        st.info("No clinician summary available for this alert.")

    # Highlight biometric features
    bio_feats = [f["feature"] for f in alert.get("shap_top_features", [])
                 if f["feature"] in BIOMETRIC_FEATURES]
    if bio_feats:
        st.error(f"Patient safety note: Biometric features affected: {', '.join(bio_feats)}")
    else:
        st.success("Patient vitals are not among the primary alert indicators.")

    st.metric("Risk Score", f"{alert.get('risk_score', 0):.2f}")
    st.markdown(f"**Recommended action:** {alert.get('response_action', 'N/A')}")


def render_admin(alert: dict):
    """Administrator view: summary statistics + risk breakdown."""
    st.markdown("#### Administrator View")

    col1, col2, col3 = st.columns(3)
    col1.metric("Risk Score", f"{alert.get('risk_score', 0):.3f}")
    col2.metric("Tier", alert.get("tier", "N/A"))
    col3.metric("Category", alert.get("attack_category", "N/A"))

    st.markdown(f"**Consensus:** {alert.get('consensus', 'N/A')}")
    st.markdown(f"**Recommended Action:** {alert.get('response_action', 'N/A')}")

    # Global charts
    gc1, gc2 = st.columns(2)
    gi = CHARTS_DIR / "global_importance_xgboost.png"
    bs = CHARTS_DIR / "beeswarm_xgboost.png"
    if gi.exists():
        gc1.image(str(gi), caption="Global Feature Importance", use_container_width=True)
    if bs.exists():
        gc2.image(str(bs), caption="SHAP Beeswarm", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════

@st.cache_data
def load_alerts() -> list:
    path = EVAL_DIR / "evaluation_alerts.json"
    if not path.exists():
        st.error("Run `python pipeline/module6_evaluation/module6_evaluation.py` first.")
        st.stop()
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_all_responses() -> list:
    path = EVAL_DIR / "alert_responses.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


@st.cache_data
def load_risk_scores():
    path = EVAL_DIR / "risk_scores.npz"
    if not path.exists():
        return None
    return dict(np.load(path, allow_pickle=True))


@st.cache_data
def load_admin_dashboard() -> dict:
    path = EVAL_DIR / "admin_dashboard.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


@st.cache_data
def load_clinician_summaries() -> dict:
    path = EVAL_DIR / "clinician_summaries.json"
    if path.exists():
        with open(path) as f:
            return {s["sample_index"]: s for s in json.load(f)}
    return {}


@st.cache_data
def load_response_policy() -> dict:
    path = EVAL_DIR / "response_policy.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


# ═══════════════════════════════════════════════════════════════════════
# Session state
# ═══════════════════════════════════════════════════════════════════════

def init_session():
    defaults = {
        "participant_id": "", "participant_role": "", "current_alert": 0,
        "responses": [], "alert_start_time": None,
        "study_started": False, "study_complete": False,
        "ab_conditions": [],
        "app_mode": "dashboard", "sim_index": 0, "sim_running": False,
        "sim_history": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ═══════════════════════════════════════════════════════════════════════
# 6.3c  Dashboard Components
# ═══════════════════════════════════════════════════════════════════════

def dashboard_mode():
    """Full dashboard with risk gauge, alert feed, SHAP, NLG, responses, heatmap."""
    st.title("IoMT IDS — Real-Time Dashboard")

    responses = load_all_responses()
    admin = load_admin_dashboard()
    clin_summaries = load_clinician_summaries()
    risk_data = load_risk_scores()
    policy = load_response_policy()

    if not responses:
        st.warning("No alert data found. Run Modules 3-5 first.")
        return

    # ── Row 1: Summary metrics ──
    st.markdown("### System Overview")
    c1, c2, c3, c4, c5 = st.columns(5)

    tier_counts = Counter(r["risk_level"] for r in responses)
    total = len(responses)
    c1.metric("Total Alerts", total)
    c2.metric("CRITICAL", tier_counts.get("CRITICAL", 0), delta_color="inverse")
    c3.metric("HIGH", tier_counts.get("HIGH", 0), delta_color="inverse")
    c4.metric("MEDIUM", tier_counts.get("MEDIUM", 0))
    c5.metric("LOW", tier_counts.get("LOW", 0))

    # ── Row 2: Alert tier distribution + Risk heatmap ──
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Alert Tier Distribution")
        tiers = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        counts = [tier_counts.get(t, 0) for t in tiers]
        chart_df = pd.DataFrame({"Tier": tiers, "Count": counts})
        st.bar_chart(chart_df.set_index("Tier"), color="#3274A1")

    with col_right:
        st.markdown("#### Risk Score Heatmap (by Attack Category)")
        if admin:
            cat_stats = admin.get("alerts_by_attack_category", {}) if admin else {}
            if cat_stats:
                st.bar_chart(pd.Series(cat_stats), color="#e74c3c")
            else:
                st.info("No category data available")
        elif risk_data is not None:
            st.info("Admin dashboard not loaded")

    # ── Row 3: Risk gauge (latest alert) + Alert feed ──
    st.markdown("---")
    col_gauge, col_feed = st.columns([1, 2])

    with col_gauge:
        st.markdown("#### Risk Score Gauge")
        # Show gauge for selected/latest alert
        alert_idx = st.selectbox("Select alert", range(min(20, len(responses))),
                                 format_func=lambda i: f"#{responses[i]['sample_index']} ({responses[i]['risk_level']})")
        selected = responses[alert_idx]
        score = selected["risk_score"]
        level = selected["risk_level"]

        # Gauge visualization using progress bar + color
        st.metric("Risk Score", f"{score:.3f}", delta=level)
        st.progress(min(score, 1.0))

        # Component breakdown
        comps = selected.get("risk_components", {})
        if comps:
            st.markdown("**Components:**")
            for k, v in comps.items():
                st.text(f"  {k}: {v:.4f}")

    with col_feed:
        st.markdown("#### Alert Feed (latest 15)")
        feed_data = []
        for r in responses[:15]:
            feed_data.append({
                "Sample": r["sample_index"],
                "Level": r["risk_level"],
                "Score": round(r["risk_score"], 3),
                "Category": r.get("attack_category", ""),
                "Actions": "|".join(r.get("response", {}).get("actions", [])),
            })
        st.dataframe(pd.DataFrame(feed_data), use_container_width=True, hide_index=True)

    # ── Row 4: SHAP waterfall + NLG clinician alert ──
    st.markdown("---")
    col_shap, col_nlg = st.columns(2)

    with col_shap:
        st.markdown("#### SHAP Waterfall Plot")
        sample_idx = selected["sample_index"]
        chart_path = CHARTS_DIR / f"waterfall_xgboost_sample_{sample_idx:04d}.png"
        if chart_path.exists():
            st.image(str(chart_path), use_container_width=True)
        else:
            # Try force plot
            force_path = CHARTS_DIR / f"force_xgboost_sample_{sample_idx:04d}.png"
            if force_path.exists():
                st.image(str(force_path), use_container_width=True)
            else:
                st.info(f"No SHAP chart for sample {sample_idx}")

    with col_nlg:
        st.markdown("#### Clinician Alert")
        clin = clin_summaries.get(sample_idx, {})
        if clin:
            severity = clin.get("severity", "LOW")
            color = TIER_COLORS.get(severity, "#999")
            st.markdown(f"**Severity:** <span style='color:{color}'>{severity}</span>",
                        unsafe_allow_html=True)
            st.warning(clin.get("summary", "No summary available"))
        else:
            st.info("No clinician summary for this sample")

    # ── Row 5: Response recommendation panel ──
    st.markdown("---")
    st.markdown("#### Response Recommendation")
    resp = selected.get("response", {})
    if resp:
        rc1, rc2, rc3 = st.columns(3)
        rc1.markdown(f"**Actions:** {', '.join(resp.get('actions', []))}")
        rc2.markdown(f"**Max Response:** {resp.get('max_response_min', 'N/A')} min")
        rc3.markdown(f"**Priority:** {resp.get('priority', 'N/A')}")

        rationale = resp.get("rationale", "")
        if rationale:
            st.caption(f"Rationale: {rationale[:200]}")

        escalation = resp.get("escalation_chain", {})
        if escalation and escalation.get("primary"):
            st.markdown(f"**Escalation:** {escalation['primary']}"
                        f"{' → ' + escalation['secondary'] if escalation.get('secondary') else ''}")

    # ── Row 6: Global SHAP summary ──
    st.markdown("---")
    st.markdown("#### Global Feature Importance (XGBoost)")
    global_chart = CHARTS_DIR / "global_importance_xgboost.png"
    beeswarm_chart = CHARTS_DIR / "beeswarm_xgboost.png"
    gc1, gc2 = st.columns(2)
    with gc1:
        if global_chart.exists():
            st.image(str(global_chart), use_container_width=True)
    with gc2:
        if beeswarm_chart.exists():
            st.image(str(beeswarm_chart), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
# 6.3b  Online Simulation Mode
# ═══════════════════════════════════════════════════════════════════════

def simulation_mode():
    """Stream test samples through pipeline, auto-refresh with new alerts.

    Includes 6C.1 streaming simulator, 6C.3 risk gauge, 6C.8 role switcher,
    6C.9 interaction capture, and 6C.10 dynamic threshold display.
    """
    st.title("IoMT IDS — Online Simulation")

    responses = load_all_responses()
    clin_summaries = load_clinician_summaries()

    if not responses:
        st.warning("No alert data. Run Modules 3-5 first.")
        return

    # 6C.8 Role switcher
    st.sidebar.markdown("---")
    st.sidebar.markdown("## Stakeholder View")
    sim_role = st.sidebar.selectbox(
        "View as:",
        ["Security Analyst", "Clinician", "Administrator"],
        key="sim_role",
    )

    # Controls
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
    with col_ctrl1:
        speed = st.slider("Simulation speed (alerts/batch)", 1, 10, 3)
    with col_ctrl2:
        auto_refresh = st.toggle("Auto-refresh", value=False)
    with col_ctrl3:
        if st.button("Step Forward"):
            st.session_state.sim_index = min(
                st.session_state.sim_index + speed, len(responses) - 1)

    idx = st.session_state.sim_index
    current_batch = responses[max(0, idx - speed + 1):idx + 1]
    history = responses[:idx + 1]

    # ── Row 1: Summary metrics ──
    st.markdown("---")
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("Samples Processed", idx + 1)
    tier_counts = Counter(r["risk_level"] for r in history)
    mc2.metric("CRITICAL", tier_counts.get("CRITICAL", 0))
    mc3.metric("HIGH", tier_counts.get("HIGH", 0))
    attacks = sum(1 for r in history if r.get("ground_truth") == "attack")
    mc4.metric("True Attacks Seen", attacks)
    if history:
        mc5.metric("Latest Risk Score", f"{history[-1]['risk_score']:.3f}")

    # ── 6C.3  Risk score gauge (latest alert) ──
    if current_batch:
        latest = current_batch[-1]
        latest_score = latest["risk_score"]
        latest_level = latest["risk_level"]

        col_gauge, col_components = st.columns([1, 2])
        with col_gauge:
            st.markdown("#### Risk Score Gauge")
            color = TIER_COLORS.get(latest_level, "#999")
            st.metric("Current Alert", f"{latest_score:.3f}",
                       delta=latest_level, delta_color="inverse" if latest_level in ("CRITICAL", "HIGH") else "normal")
            st.progress(min(latest_score, 1.0))

        with col_components:
            st.markdown("#### 4-Component Breakdown")
            comps = latest.get("risk_components", {})
            if comps:
                comp_df = pd.DataFrame({
                    "Component": list(comps.keys()),
                    "Value": [float(v) for v in comps.values()],
                })
                st.bar_chart(comp_df.set_index("Component"), color="#3274A1")
            else:
                st.caption("Component breakdown not available for this alert")

    # ── 6C.10  Dynamic threshold display ──
    st.markdown("---")
    col_thresh, col_drift = st.columns(2)
    with col_thresh:
        st.markdown("#### Adaptive Threshold Monitor")
        # Load dynamic threshold results if available
        dyn_path = EVAL_DIR / "dynamic_threshold_results.json"
        if dyn_path.exists():
            with open(dyn_path) as f:
                dyn = json.load(f)
            b1 = dyn.get("b1_static_vs_adaptive", {})
            fm = b1.get("final_metrics", {})
            if fm:
                thc1, thc2 = st.columns(2)
                thc1.metric("Static F1", f"{fm.get('static', {}).get('f1', 0):.4f}")
                thc2.metric("Adaptive F1", f"{fm.get('adaptive', {}).get('f1', 0):.4f}")
            # Show threshold chart if exists
            thresh_chart = CHARTS_DIR / "threshold_over_time.png"
            if thresh_chart.exists():
                st.image(str(thresh_chart), use_container_width=True,
                         caption="DAE threshold: static vs adaptive")
        else:
            st.info("Run `dynamic_threshold_sim.py` to enable adaptive threshold monitoring")

    with col_drift:
        st.markdown("#### Drift Detection Status")
        drift_path = EVAL_DIR / "drift_detection_results.json"
        if drift_path.exists():
            with open(drift_path) as f:
                drift = json.load(f)
            psi = drift.get("psi_summary", {})
            ks = drift.get("ks_summary", {})
            n_events = len(drift.get("drift_events", []))
            dc1, dc2 = st.columns(2)
            dc1.metric("Drift Events", n_events)
            dc2.metric("PSI (max)", f"{psi.get('max', 0):.4f}",
                       delta="DRIFT" if psi.get("max", 0) > 0.1 else "OK",
                       delta_color="inverse" if psi.get("max", 0) > 0.1 else "normal")
            psi_chart = CHARTS_DIR / "drift_psi.png"
            if psi_chart.exists():
                st.image(str(psi_chart), use_container_width=True,
                         caption="PSI over time")
        else:
            st.info("Run `drift_detection.py` to enable drift monitoring")

    # ── Current batch display ──
    st.markdown("---")
    st.markdown("### Current Batch")
    for r in current_batch:
        sample_idx = r["sample_index"]
        level = r["risk_level"]
        score = r["risk_score"]
        color = TIER_COLORS.get(level, "#999")

        with st.expander(f"Alert #{sample_idx} — :{color.replace('#','')}[{level}] R={score:.3f}", expanded=(level in ("CRITICAL", "HIGH"))):
            # Build structured alert for role-specific rendering
            clin = clin_summaries.get(sample_idx, {})
            alert_obj = process_alert(sample_idx, {
                "risk_score": score, "risk_level": level,
                "attack_category": r.get("attack_category", "unknown"),
                "xai_explanation": {
                    "xgboost_top_features": r.get("explanation", {}).get("analyst", {}).get("xgboost_top_features", []),
                    "dae_top_features": r.get("explanation", {}).get("analyst", {}).get("dae_top_features", []),
                    "clinician_summary": clin.get("summary", ""),
                    "consensus": r.get("explanation", {}).get("analyst", {}).get("consensus", ""),
                },
            })

            # 6C.8 Role-switched rendering
            if sim_role == "Security Analyst":
                render_analyst(alert_obj)
            elif sim_role == "Clinician":
                render_clinician(alert_obj)
            else:
                render_admin(alert_obj)

            # Response recommendation always shown
            resp = r.get("response", {})
            if resp:
                st.markdown(f"**Response:** {', '.join(resp.get('actions', []))}")

            # 6C.9 Interaction buttons
            btn_col1, btn_col2, btn_col3 = st.columns(3)
            with btn_col1:
                if st.button("Confirm", key=f"confirm_{sample_idx}"):
                    capture_online_interaction(
                        st.session_state.get("participant_id", "anon"),
                        sample_idx, "confirm",
                        {"tier": level, "score": score},
                    )
                    st.success("Confirmed")
            with btn_col2:
                if st.button("Reject", key=f"reject_{sample_idx}"):
                    capture_online_interaction(
                        st.session_state.get("participant_id", "anon"),
                        sample_idx, "reject",
                        {"tier": level, "score": score},
                    )
                    st.warning("Rejected — logged for feedback loop")
            with btn_col3:
                note = st.text_input("Note", key=f"note_{sample_idx}", label_visibility="collapsed",
                                      placeholder="Add feedback note...")
                if note:
                    capture_online_interaction(
                        st.session_state.get("participant_id", "anon"),
                        sample_idx, "feedback_note",
                        {"note": note, "tier": level},
                    )

    # ── Tier distribution over time ──
    st.markdown("### Alert Tier Distribution (cumulative)")
    tier_over_time = {"LOW": [], "MEDIUM": [], "HIGH": [], "CRITICAL": []}
    for i in range(min(len(history), 50)):
        window = history[:i + 1]
        counts = Counter(r["risk_level"] for r in window)
        for t in tier_over_time:
            tier_over_time[t].append(counts.get(t, 0))
    st.line_chart(pd.DataFrame(tier_over_time))

    # Auto-refresh
    if auto_refresh and idx < len(responses) - 1:
        time.sleep(1.5)
        st.session_state.sim_index = min(idx + speed, len(responses) - 1)
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════
# 6.3a  Offline Evaluation (Browse + Study)
# ═══════════════════════════════════════════════════════════════════════

def display_alert(alert: dict, show_xai: bool):
    """Display an alert with or without XAI explanation."""
    st.markdown(f"### Alert: {alert['alert_id']}")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Risk Score", f"{alert['risk_score']:.2f}")
    with col2:
        level_colors = {"CRITICAL": "red", "HIGH": "orange", "MEDIUM": "blue", "LOW": "green"}
        st.markdown(f"**Risk Level:** :{level_colors.get(alert['risk_level'], 'gray')}[{alert['risk_level']}]")

    if show_xai:
        st.markdown("---")
        st.markdown("#### XAI Explanation")

        xai = alert.get("xai_explanation", {})
        top_feats = xai.get("xgboost_top_features", [])
        if top_feats:
            st.markdown("**Top Contributing Features (SHAP):**")
            for f in top_feats[:5]:
                direction = "increases" if f.get("shap_value", 0) > 0 else "decreases"
                st.markdown(f"- **{f['feature']}**: {direction} risk (SHAP: {f.get('shap_value', 0):+.3f})")

        dae_feats = xai.get("dae_top_features", [])
        if dae_feats:
            st.markdown("**DAE Anomaly Indicators:**")
            for f in dae_feats[:3]:
                st.markdown(f"- **{f['feature']}**: {f.get('pct_contribution', 0):.1f}% of anomaly score")

        consensus = xai.get("consensus", "")
        if consensus:
            st.info(f"Model consensus: {consensus}")

        clin = xai.get("clinician_summary", "")
        if clin:
            st.warning(clin[:500])

        chart_path = CHARTS_DIR / f"waterfall_xgboost_sample_{alert['sample_index']:04d}.png"
        if chart_path.exists():
            st.image(str(chart_path), caption="SHAP Waterfall Plot", use_container_width=True)
    else:
        st.markdown("---")
        st.info("No explanation available. Decide based on risk score and level only.")


def response_form(alert: dict, alert_index: int, show_xai: bool) -> dict | None:
    """Capture participant response — delegates to reusable likert_form()."""
    result = likert_form(alert["alert_id"], f"response_{alert_index}")
    if result:
        elapsed = round(time.time() - st.session_state.alert_start_time, 1)
        result.update({
            "participant_id": st.session_state.participant_id,
            "participant_role": st.session_state.participant_role,
            "condition": "with_xai" if show_xai else "without_xai",
            "correct_action": alert.get("correct_action", ""),
            "decision_correct": result["chosen_action"] == alert.get("correct_action", ""),
            "decision_time_sec": elapsed,
            "timestamp": datetime.now().isoformat(),
        })
        audit_log("response_submit", participant_id=st.session_state.participant_id,
                  alert_id=alert["alert_id"], action=result["chosen_action"],
                  decision_time=elapsed)
        return result
    return None


def browse_mode():
    """6.3a — Free browsing with XAI toggle."""
    alerts = load_alerts()
    n = len(alerts)

    st.sidebar.markdown("## Browse Controls")
    show_xai = st.sidebar.toggle("Show XAI Explanation", value=True)
    idx = st.sidebar.slider("Alert #", 0, n - 1, 0)
    alert = alerts[idx]

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Ground Truth:** `{alert['ground_truth']}`")
    st.sidebar.markdown(f"**Attack Type:** `{alert['attack_category']}`")
    st.sidebar.markdown(f"**Correct Action:** `{alert.get('correct_action', 'N/A')}`")

    st.title("IoMT Alert Browser")
    st.caption(f"Alert {idx + 1} of {n} — {'With XAI' if show_xai else 'Without XAI'}")
    display_alert(alert, show_xai)


def study_mode():
    """6.3a — Formal A/B study with Likert questionnaires.

    Uses counterbalanced A/B assignment, reusable likert_form(),
    audit trail logging, and auto-timer.
    """
    if not st.session_state.study_started:
        st.title("IoMT IDS Explanation Evaluation")
        st.markdown("This study evaluates how XAI explanations affect "
                    "your ability to respond to IoMT intrusion detection alerts.")
        with st.form("registration"):
            pid = st.text_input("Participant ID (e.g., P01)")
            role = st.selectbox("Your Role", ROLES)
            consent = st.checkbox("I consent to participate")
            if st.form_submit_button("Start") and pid and consent:
                st.session_state.participant_id = pid
                st.session_state.participant_role = role
                st.session_state.study_started = True
                st.session_state.alert_start_time = time.time()
                # Pre-compute A/B conditions
                alerts = load_alerts()
                st.session_state.ab_conditions = assign_ab_conditions(len(alerts), pid)
                audit_log("study_start", participant_id=pid, role=role)
                st.rerun()
        return

    if st.session_state.study_complete:
        responses = st.session_state.responses
        n = len(responses)
        correct = sum(1 for r in responses if r["decision_correct"])
        st.title("Evaluation Complete")
        st.success(f"Thank you, {st.session_state.participant_id}!")
        st.metric("Alerts", n)
        st.metric("Accuracy", f"{correct/n*100:.0f}%" if n > 0 else "N/A")
        save_path = EVAL_DIR / f"responses_{st.session_state.participant_id}.json"
        save_path.write_text(json.dumps(responses, indent=2))
        st.info(f"Saved to `{save_path}`")
        audit_log("study_complete", participant_id=st.session_state.participant_id,
                  n_responses=n, accuracy=correct / n if n else 0)
        return

    alerts = load_alerts()
    n = len(alerts)
    current = st.session_state.current_alert
    if current >= n:
        st.session_state.study_complete = True
        st.rerun()
        return

    alert = alerts[current]
    # Use counterbalanced A/B conditions
    ab = st.session_state.get("ab_conditions", [True] * (n // 2) + [False] * (n - n // 2))
    show_xai = ab[current] if current < len(ab) else current < (n // 2)

    st.progress(current / n, text=f"Alert {current+1}/{n} ({'With XAI' if show_xai else 'Without XAI'})")
    audit_log("alert_view", participant_id=st.session_state.participant_id,
              alert_id=alert["alert_id"], condition="with_xai" if show_xai else "without_xai")
    display_alert(alert, show_xai)

    # Use reusable likert_form component
    likert_result = likert_form(alert["alert_id"], f"study_{current}")
    if likert_result:
        elapsed = round(time.time() - st.session_state.alert_start_time, 1)
        response = {
            "participant_id": st.session_state.participant_id,
            "participant_role": st.session_state.participant_role,
            "condition": "with_xai" if show_xai else "without_xai",
            "correct_action": alert.get("correct_action", ""),
            "decision_correct": likert_result["chosen_action"] == alert.get("correct_action", ""),
            "decision_time_sec": elapsed,
            "timestamp": datetime.now().isoformat(),
            **likert_result,
        }
        st.session_state.responses.append(response)
        audit_log("response_submit", participant_id=st.session_state.participant_id,
                  alert_id=alert["alert_id"], action=likert_result["chosen_action"],
                  decision_time=elapsed)
        st.session_state.current_alert += 1
        st.session_state.alert_start_time = time.time()
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def pcap_replay_stub():
    """6C.10 — PCAP replay placeholder (optional, future work)."""
    st.title("PCAP Replay Mode")
    st.info(
        "**PCAP Replay** loads a `.pcap` file, extracts network flows via ARGUS/Scapy, "
        "and feeds them through the full preprocessing + detection + risk scoring + "
        "explanation pipeline in real time.\n\n"
        "**Status:** Stub — not yet implemented. This requires:\n"
        "- `scapy` for packet parsing\n"
        "- ARGUS for flow extraction\n"
        "- Integration with Phase 1 preprocessing pipeline\n\n"
        "For the thesis prototype, the **Online Simulation** mode demonstrates "
        "the same end-to-end flow using pre-processed test set samples."
    )
    uploaded = st.file_uploader("Upload .pcap file (future)", type=["pcap", "pcapng"], disabled=True)
    if uploaded:
        st.warning("PCAP processing not yet implemented.")


def main():
    st.set_page_config(page_title="IoMT IDS Dashboard", layout="wide")
    init_session()

    st.sidebar.title("IoMT IDS")
    mode = st.sidebar.radio(
        "Mode:",
        ["Dashboard", "Online Simulation", "Browse Alerts", "Study (A/B)", "PCAP Replay"],
    )

    if mode == "Dashboard":
        dashboard_mode()
    elif mode == "Online Simulation":
        simulation_mode()
    elif mode == "Browse Alerts":
        browse_mode()
    elif mode == "Study (A/B)":
        study_mode()
    elif mode == "PCAP Replay":
        pcap_replay_stub()


if __name__ == "__main__":
    main()
