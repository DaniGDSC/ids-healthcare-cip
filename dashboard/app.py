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
        "live_monitor", "alerts", "stakeholder", "system",
    ],
    "Attending Physician": [
        "live_monitor", "alerts", "stakeholder",
    ],
    "Hospital Manager": [
        "live_monitor", "stakeholder", "performance", "system",
    ],
    "Regulatory Auditor": [
        "live_monitor", "stakeholder", "system",
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
    """Initialize streaming engine, inference service, and simulator."""
    from dashboard.streaming.window_buffer import WindowBuffer

    if "buffer" not in st.session_state:
        st.session_state.buffer = WindowBuffer(
            window_size=20, calibration_threshold=20,
        )

    if "inference_service" not in st.session_state:
        try:
            from src.production.inference_service import InferenceService
            service = InferenceService(PROJECT_ROOT)
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
        )

    if "selected_alert" not in st.session_state:
        st.session_state.selected_alert = 0


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

    # RBAC: Role selector
    role = st.selectbox("View as", list(ROLES.keys()), index=0)
    allowed_panels = ROLES[role]

    # Navigation
    pages_for_role = [PAGE_LABELS[p] for p in allowed_panels if p in PAGE_LABELS]
    page = st.radio("Navigation", pages_for_role, index=0)

    st.markdown("---")

    # ── Demo Controls ──
    st.markdown("##### Streaming Control")
    from dashboard.simulation.scenarios import ScenarioID, SimMode

    scenario_options = [
        "RANDOM: All Scenarios",
        "A: Benign Only",
        "B: Gradual Attack",
        "C: Abrupt Attack",
        "D: Mixed Cycle",
        "E: Novelty Attacks",
    ]
    scenario_choice = st.selectbox("Scenario", scenario_options, index=0)
    mode = st.selectbox("Speed", ["ACCELERATED", "REALTIME", "STRESS"], index=0)

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("START", type="primary", disabled=sim.running, use_container_width=True):
            if scenario_choice.startswith("RANDOM"):
                sid = ScenarioID.RANDOM
            else:
                sid = ScenarioID(scenario_choice[0])
            sim.start(sid, SimMode[mode])
            st.rerun()
    with col2:
        if st.button("STOP", disabled=not sim.running, use_container_width=True):
            sim.stop()
            st.rerun()
    with col3:
        if st.button("RESET", use_container_width=True):
            sim.reset()
            buffer.reset()
            st.rerun()

    if sim.running:
        sim_status = sim.get_status()
        st.info(
            f"Scenario {sim_status.get('scenario', '?')} | "
            f"{sim_status.get('current_phase', '')} | "
            f"{sim_status.get('flows_injected', 0):,} flows"
        )

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
    render(gt, st.session_state.selected_alert)

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
    render(gt, role=role)

elif page == "System & Compliance":
    from dashboard.components.panel_system import render
    render(gt)
