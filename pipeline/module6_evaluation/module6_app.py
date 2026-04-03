#!/usr/bin/env python3
"""Module 6 — Evaluation Interface (Tasks 6.3-6.5, 6.9).

Streamlit web app for human evaluation of XAI explanations.
Presents alerts with/without explanations, captures Likert scores,
decision accuracy, time, and feedback.

Usage:
    streamlit run evaluation_app.py
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = PROJECT_ROOT / "results/reports"
CHARTS_DIR = PROJECT_ROOT / "results/charts"

ROLES = ["Security Analyst", "Clinician", "Administrator"]
ACTIONS = ["dismiss", "monitor", "investigate", "isolate", "escalate"]


def load_alerts() -> list:
    """Load curated evaluation alerts."""
    path = EVAL_DIR / "evaluation_alerts.json"
    if not path.exists():
        st.error("Run `python build_evaluation.py` first to generate evaluation alerts.")
        st.stop()
    with open(path) as f:
        return json.load(f)


def init_session():
    """Initialize session state."""
    if "participant_id" not in st.session_state:
        st.session_state.participant_id = ""
        st.session_state.participant_role = ""
        st.session_state.current_alert = 0
        st.session_state.responses = []
        st.session_state.alert_start_time = None
        st.session_state.study_started = False
        st.session_state.study_complete = False


def registration_page():
    """Participant registration."""
    st.title("IoMT IDS Explanation Evaluation")
    st.markdown("### Welcome")
    st.markdown(
        "This study evaluates how explainable AI (XAI) explanations affect "
        "your ability to respond to intrusion detection alerts in a healthcare IoMT setting."
    )

    with st.form("registration"):
        pid = st.text_input("Participant ID (e.g., P01)")
        role = st.selectbox("Your Role", ROLES)
        consent = st.checkbox("I consent to participate in this evaluation")
        submitted = st.form_submit_button("Start Evaluation")

        if submitted and pid and consent:
            st.session_state.participant_id = pid
            st.session_state.participant_role = role
            st.session_state.study_started = True
            st.session_state.alert_start_time = time.time()
            st.rerun()


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

        # SHAP top features
        xai = alert.get("xai_explanation", {})
        top_feats = xai.get("xgboost_top_features", [])
        if top_feats:
            st.markdown("**Top Contributing Features (SHAP):**")
            for f in top_feats[:5]:
                direction = "increases" if f.get("shap_value", 0) > 0 else "decreases"
                st.markdown(f"- **{f['feature']}**: {direction} risk (SHAP: {f.get('shap_value', 0):+.3f})")

        # DAE features
        dae_feats = xai.get("dae_top_features", [])
        if dae_feats:
            st.markdown("**DAE Anomaly Indicators:**")
            for f in dae_feats[:3]:
                st.markdown(f"- **{f['feature']}**: {f.get('pct_contribution', 0):.1f}% of anomaly score")

        # Consensus
        consensus = xai.get("consensus", "")
        if consensus:
            st.info(f"Model consensus: {consensus}")

        # Clinician summary
        clin = xai.get("clinician_summary", "")
        if clin:
            st.warning(clin[:500])

        # Show waterfall chart if available
        chart_path = CHARTS_DIR / f"waterfall_xgboost_sample_{alert['sample_index']:04d}.png"
        if chart_path.exists():
            st.image(str(chart_path), caption="SHAP Waterfall Plot", use_container_width=True)
    else:
        st.markdown("---")
        st.info("No explanation available for this alert. Please make your decision based on the risk score and level only.")


def response_form(alert: dict, alert_index: int, show_xai: bool) -> dict | None:
    """Capture participant response."""
    with st.form(f"response_{alert_index}"):
        st.markdown("#### Your Response")

        action = st.selectbox(
            "What action would you take?",
            ACTIONS,
            format_func=lambda x: x.capitalize(),
        )

        confidence = st.slider("How confident are you? (1=Not at all, 5=Very confident)", 1, 5, 3)

        st.markdown("#### Rate the alert presentation")
        trust = st.slider("I trust this alert classification is correct", 1, 5, 3, key=f"trust_{alert_index}")
        usefulness = st.slider("The information helps me respond appropriately", 1, 5, 3, key=f"useful_{alert_index}")
        comprehensibility = st.slider("I understand why this alert was triggered", 1, 5, 3, key=f"comp_{alert_index}")
        actionability = st.slider("I know what action to take", 1, 5, 3, key=f"action_{alert_index}")

        # 6.9 Feedback collection
        feedback = st.text_area("Any feedback? (optional)", key=f"feedback_{alert_index}")
        reclass = st.selectbox(
            "Would you reclassify this alert? (optional)",
            ["No change", "CRITICAL", "HIGH", "MEDIUM", "LOW", "Benign/Dismiss"],
            key=f"reclass_{alert_index}",
        )

        submitted = st.form_submit_button("Submit & Next")

        if submitted:
            elapsed = round(time.time() - st.session_state.alert_start_time, 1)
            return {
                "participant_id": st.session_state.participant_id,
                "participant_role": st.session_state.participant_role,
                "alert_id": alert["alert_id"],
                "condition": "with_xai" if show_xai else "without_xai",
                "chosen_action": action,
                "correct_action": alert.get("correct_action", ""),
                "decision_correct": action == alert.get("correct_action", ""),
                "decision_time_sec": elapsed,
                "confidence": confidence,
                "likert_trust": trust,
                "likert_usefulness": usefulness,
                "likert_comprehensibility": comprehensibility,
                "likert_actionability": actionability,
                "feedback": feedback,
                "reclassification": reclass if reclass != "No change" else None,
                "timestamp": datetime.now().isoformat(),
            }
    return None


def results_page():
    """Show completion summary and save responses."""
    st.title("Evaluation Complete")
    st.success(f"Thank you, {st.session_state.participant_id}! You completed all alerts.")

    responses = st.session_state.responses
    n = len(responses)
    correct = sum(1 for r in responses if r["decision_correct"])

    st.metric("Alerts Evaluated", n)
    st.metric("Decision Accuracy", f"{correct/n*100:.0f}%" if n > 0 else "N/A")

    # Save responses
    save_path = EVAL_DIR / f"responses_{st.session_state.participant_id}.json"
    save_path.write_text(json.dumps(responses, indent=2))
    st.info(f"Responses saved to `{save_path}`")

    # Append to combined file
    combined_path = EVAL_DIR / "all_responses.json"
    existing = []
    if combined_path.exists():
        with open(combined_path) as f:
            existing = json.load(f)
    existing.extend(responses)
    combined_path.write_text(json.dumps(existing, indent=2))


def main():
    st.set_page_config(page_title="IoMT XAI Evaluation", layout="wide")
    init_session()

    if not st.session_state.study_started:
        registration_page()
        return

    if st.session_state.study_complete:
        results_page()
        return

    alerts = load_alerts()
    n_alerts = len(alerts)
    current = st.session_state.current_alert

    if current >= n_alerts:
        st.session_state.study_complete = True
        st.rerun()
        return

    alert = alerts[current]
    show_xai = current < (n_alerts // 2)  # first half with XAI, second without

    # Progress bar
    st.progress(current / n_alerts,
                text=f"Alert {current + 1} of {n_alerts} "
                     f"({'With XAI' if show_xai else 'Without XAI'})")

    # Display alert
    display_alert(alert, show_xai)

    # Response form
    response = response_form(alert, current, show_xai)
    if response:
        st.session_state.responses.append(response)
        st.session_state.current_alert += 1
        st.session_state.alert_start_time = time.time()
        st.rerun()


if __name__ == "__main__":
    main()
