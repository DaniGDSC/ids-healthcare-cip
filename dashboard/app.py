"""RA-X-IoMT Security Monitoring Dashboard v3.

Production prototype with live streaming inference, role-based access
control, network traffic visualization, and real-time alerts.

Run:
    python extract_project_reality.py   # Generate ground truth
    streamlit run dashboard/app.py      # Launch dashboard
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import streamlit as st

# ── Path Setup ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

GROUND_TRUTH_PATH = PROJECT_ROOT / "project_ground_truth.json"

# ── RBAC: Role-Based Access Control ──────────────────────────────────────
ROLES: Dict[str, list] = {
    "IT Security Analyst": [
        "live_monitor", "alerts", "explanations", "performance",
        "stakeholder", "system",
    ],
    "Clinical IT Administrator": [
        "live_monitor", "alerts", "explanations", "stakeholder", "system",
    ],
    "Attending Physician": [
        "live_monitor", "alerts", "explanations", "stakeholder",
    ],
    "Hospital Manager": [
        "live_monitor", "explanations", "stakeholder", "performance", "system",
    ],
    "Regulatory Auditor": [
        "live_monitor", "explanations", "stakeholder", "system",
    ],
}

PAGE_LABELS: Dict[str, str] = {
    "live_monitor": "Live Monitor",
    "alerts": "Alert Feed",
    "explanations": "Explanations",
    "performance": "Performance",
    "stakeholder": "Stakeholder Intelligence",
    "system": "System & Compliance",
}


# ── Ground Truth Loader ─────────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_ground_truth() -> Dict[str, Any]:
    try:
        with open(GROUND_TRUTH_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        logger.error("Failed to load ground truth: %s", exc)
        return {}


# ── Session State ────────────────────────────────────────────────────────
def init_session_state() -> None:
    """Initialize streaming engine, inference service, database, and simulator."""
    from dashboard.streaming.window_buffer import WindowBuffer

    # Database (persistent storage)
    if "database" not in st.session_state:
        try:
            from src.production.database import Database
            db_path = _cfg("database.path", "data/production/iomt_ids.db")
            st.session_state.database = Database(str(PROJECT_ROOT / db_path))
        except Exception as exc:
            logger.error("Failed to init database: %s", exc)
            st.session_state.database = None

    if "buffer" not in st.session_state:
        st.session_state.buffer = WindowBuffer(
            window_size=20, calibration_threshold=20,
        )

    if "inference_service" not in st.session_state:
        try:
            from src.production.inference_service import InferenceService
            service = InferenceService(
                PROJECT_ROOT,
                database=st.session_state.database,
            )
            with st.spinner("Loading model..."):
                service.load()
            st.session_state.inference_service = service
        except Exception as exc:
            logger.error("Failed to load inference service: %s", exc)
            st.session_state.inference_service = None

    if "simulator" not in st.session_state:
        from dashboard.simulation.wustl_simulator import WUSTLFlowSimulator
        st.session_state.simulator = WUSTLFlowSimulator(
            flows_dir=str(PROJECT_ROOT / "data" / "streaming" / "wustl_flows"),
            buffer=st.session_state.buffer,
            inference_service=st.session_state.inference_service,
            database=st.session_state.database,
        )

    if "selected_alert" not in st.session_state:
        st.session_state.selected_alert = 0

    # Feedback loop (human-in-the-loop management)
    if "feedback_loop" not in st.session_state:
        try:
            from src.production.feedback_loop import FeedbackLoop
            calibrator_fn = None
            if st.session_state.get("inference_service"):
                calibrator_fn = st.session_state.inference_service.recalibrate_from_feedback
            st.session_state.feedback_loop = FeedbackLoop(
                database=st.session_state.get("database"),
                calibrator_fn=calibrator_fn,
            )
        except Exception as exc:
            logger.error("Failed to init feedback loop: %s", exc)
            st.session_state.feedback_loop = None


# ── Page Config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RA-X-IoMT Dashboard",
    page_icon="shield",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS
CSS_PATH = Path(__file__).parent / "assets" / "style.css"
if CSS_PATH.exists():
    st.markdown(f"<style>{CSS_PATH.read_text()}</style>", unsafe_allow_html=True)

# HIPAA banner
st.markdown(
    '<div class="hipaa-banner">'
    "This dashboard is for <b>research purposes only</b>. "
    "No Protected Health Information (PHI) is stored or transmitted."
    "</div>",
    unsafe_allow_html=True,
)

# Initialize
init_session_state()

# ── Authentication Gate ────────────────────────────────────────────────
from config.production_loader import cfg as _cfg

_auth_mode = _cfg("auth.mode", "open")

if _auth_mode != "open":
    if "auth_session" not in st.session_state:
        st.markdown("### Login")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
        if submitted and username and password:
            from src.production.auth import AuthProvider
            if "auth_provider" not in st.session_state:
                st.session_state.auth_provider = AuthProvider.from_config()
            session = st.session_state.auth_provider.authenticate(username, password)
            if session:
                st.session_state.auth_session = session
                st.rerun()
            else:
                st.error("Invalid credentials")
        st.stop()

# Load ground truth
gt = load_ground_truth()
if not gt:
    st.error(
        "Ground truth not found. Run `python extract_project_reality.py` "
        "to generate `project_ground_truth.json`."
    )
    st.stop()

# ── Sidebar ──────────────────────────────────────────────────────────────
sim = st.session_state.simulator
buffer = st.session_state.buffer

with st.sidebar:
    st.markdown("### RA-X-IoMT")
    st.caption("Risk-Adaptive Explainable IDS for IoMT")
    st.markdown("---")

    # System info
    arch = gt.get("model_architecture", {})
    params = arch.get("total_params", 0)
    if params:
        st.markdown(f"**Model:** CNN-BiLSTM-Attention ({params:,} params)")
    st.markdown("**Features:** 24 (post-variance filtering)")
    st.markdown("**Dataset:** WUSTL-EHMS-2020")

    model_status = "Loaded" if st.session_state.inference_service else "Not loaded"
    st.markdown(f"**Inference:** {model_status}")
    st.markdown("---")

    # RBAC: Role from auth or selector
    if "auth_session" in st.session_state:
        auth = st.session_state.auth_session
        role = auth.role
        st.markdown(f"**User:** {auth.username}")
        st.markdown(f"**Role:** {role}")
        if st.button("Logout"):
            del st.session_state.auth_session
            st.rerun()
    else:
        role = st.selectbox("View as", list(ROLES.keys()), index=0)
    allowed_panels = ROLES[role]

    # Navigation
    pages_for_role = [PAGE_LABELS[p] for p in allowed_panels if p in PAGE_LABELS]
    page = st.radio("Navigation", pages_for_role, index=0)

    # Access logging (HIPAA audit)
    db = st.session_state.get("database")
    if db and _cfg("database.log_access", True):
        _user = "anonymous"
        if "auth_session" in st.session_state:
            _user = st.session_state.auth_session.username
        try:
            db.insert_access(_user, role, f"view_{page}", None)
        except Exception:
            pass

    st.markdown("---")

    # ── Streaming Control ──
    st.markdown("##### Streaming Control")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("START", type="primary", disabled=sim.running or sim.exhausted,
                      width="stretch"):
            sim.start()
            st.rerun()
    with col2:
        if st.button("STOP", disabled=not sim.running, width="stretch"):
            sim.stop()
            st.rerun()
    with col3:
        if st.button("RESET", width="stretch"):
            sim.reset()
            buffer.reset()
            st.rerun()

    sim_status = sim.get_status()
    progress = sim_status.get("progress_pct", 0)

    if sim.exhausted:
        st.error("Dataset exhausted — update dataset to continue streaming.")
    elif sim.running:
        st.progress(progress / 100, text=f"{sim_status.get('current_phase', '')} — "
                     f"{sim_status['flows_injected']:,}/{sim_status['total_files']:,} "
                     f"({progress:.1f}%)")
    elif sim_status["flows_injected"] > 0:
        st.caption(f"Paused at {sim_status['flows_injected']:,}/{sim_status['total_files']:,}")

    # Data export
    _db = st.session_state.get("database")
    if _db and _db.get_prediction_count() > 0:
        st.markdown("---")
        st.markdown("##### Data Export")
        import csv as _csv
        import io as _io

        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            alerts_data = _db.query_alerts(limit=5000)
            if alerts_data:
                buf = _io.StringIO()
                fields = ["time", "risk_level", "clinical_severity", "device_id_hash",
                           "acknowledged", "acknowledged_by"]
                w = _csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
                w.writeheader()
                for a in alerts_data:
                    w.writerow(a)
                st.download_button("Alerts CSV", buf.getvalue(),
                                   "alerts.csv", "text/csv", width="stretch")
        with col_exp2:
            preds_data = _db.query_predictions(limit=5000)
            if preds_data:
                buf = _io.StringIO()
                fields = ["time", "sample_index", "anomaly_score", "risk_level",
                           "clinical_severity", "latency_ms", "ground_truth"]
                w = _csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
                w.writeheader()
                for p in preds_data:
                    w.writerow(p)
                st.download_button("Predictions CSV", buf.getvalue(),
                                   "predictions.csv", "text/csv", width="stretch")

    st.markdown("---")
    st.warning(
        "**RESEARCH PROTOTYPE** — MedSec-2026 streaming simulation. "
        "No PHI stored or transmitted."
    )

# ── Page Routing (RBAC-filtered) ─────────────────────────────────────────
# Live panels use @st.fragment(run_every=N) for partial updates
# without reloading the entire page. Static panels render once.

if page == "Live Monitor":
    @st.fragment(run_every=2)
    def _live_monitor_fragment():
        from dashboard.components.panel_live_monitor import render
        render(
            buffer.get_status(),
            sim.get_status(),
            buffer.get_prediction_timeseries(200),
            buffer.get_flow_vectors(50),
        )
    _live_monitor_fragment()

elif page == "Alert Feed":
    @st.fragment(run_every=2)
    def _alerts_fragment():
        from dashboard.components.panel_alerts import render
        s = sim.get_status()
        live = buffer.get_alerts() if s.get("running") or buffer.flow_count > 0 else None
        selected = render(gt, live)
        if selected is not None:
            st.session_state.selected_alert = selected
    _alerts_fragment()

elif page == "Explanations":
    from dashboard.components.panel_shap import render
    live_alerts = buffer.get_alerts() if buffer.flow_count > 0 else None
    render(gt, st.session_state.selected_alert, role=role, live_alerts=live_alerts)

elif page == "Performance":
    @st.fragment(run_every=3)
    def _performance_fragment():
        from dashboard.components.panel_performance import render
        render(
            buffer.get_prediction_timeseries(200),
            sim.get_status(),
            buffer.get_status(),
        )
    _performance_fragment()

elif page == "Stakeholder Intelligence":
    from dashboard.components.panel_stakeholder import render
    live_stakeholder = buffer.get_alerts() if buffer.flow_count > 0 else None
    render(gt, role=role, live_alerts=live_stakeholder)

elif page == "System & Compliance":
    from dashboard.components.panel_system import render
    render(gt)
