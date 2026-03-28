"""RA-X-IoMT Security Monitoring Dashboard v2.

Autonomous real-time monitoring dashboard with MedSec-25 streaming
simulation, role-based views, and verified empirical results.

Run:
    python extract_project_reality.py   # Generate ground truth
    streamlit run dashboard/app.py      # Launch dashboard
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st

# ── Path Setup ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

# ── Named Constants ──────────────────────────────────────────────────────
WINDOW_SIZE: int = 20
CALIBRATION_THRESHOLD: int = 100
MAX_ALERT_DISPLAY: int = 50
SHAP_TIMEOUT_SECONDS: float = 30.0
CRITICAL_LATENCY_BUDGET: float = 1.0
HIPAA_HASH_PREFIX_LENGTH: int = 8

GROUND_TRUTH_PATH = PROJECT_ROOT / "project_ground_truth.json"

# ── Role Definitions ────────────────────────────────────────────────────
ROLES: Dict[str, list] = {
    "IT Security Analyst": [
        "operational", "alerts", "stakeholder", "shap", "evaluation",
        "model", "risk", "devices", "crossval", "compliance",
    ],
    "Clinical IT Administrator": [
        "operational", "stakeholder", "risk", "evaluation", "model", "compliance",
    ],
    "Attending Physician": [
        "operational", "stakeholder", "risk", "devices",
    ],
    "Hospital Manager": [
        "operational", "stakeholder", "risk", "compliance",
    ],
    "Regulatory Auditor": [
        "operational", "stakeholder", "compliance",
    ],
}

PAGE_LABELS: Dict[str, str] = {
    "operational": "Operational Status",
    "stakeholder": "Stakeholder Intelligence",
    "alerts": "Alert Feed",
    "shap": "SHAP Explanations",
    "evaluation": "Evaluation Results",
    "model": "Model & Training",
    "risk": "Risk Analytics",
    "devices": "Device Inventory",
    "crossval": "Cross-Dataset & Simulation",
    "compliance": "Compliance & Audit",
}


# ── Ground Truth Loader ─────────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_ground_truth() -> Dict[str, Any]:
    """Load the consolidated ground truth JSON.

    Returns:
        Ground truth dictionary, or empty dict on failure.
    """
    try:
        with open(GROUND_TRUTH_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        logger.error("Failed to load ground truth: %s", exc)
        return {}


def render_or_unavailable(
    status: Optional[str],
    render_fn: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Conditionally render a panel or show unavailable message.

    Args:
        status: Artifact status string.
        render_fn: Function to call if available.
        *args: Positional args for render_fn.
        **kwargs: Keyword args for render_fn.

    Returns:
        Result of render_fn or None.
    """
    if status in ("VERIFIED", "PRESENT_UNVERIFIED"):
        if status == "PRESENT_UNVERIFIED":
            st.warning("Artifact present but hash unverified")
        return render_fn(*args, **kwargs)
    else:
        st.info("NOT AVAILABLE — artifact not found in project directory")
        return None


# ── Session State Initialization ─────────────────────────────────────────
def init_session_state() -> None:
    """Initialize session state for streaming and simulation."""
    if "buffer" not in st.session_state:
        from dashboard.streaming.window_buffer import WindowBuffer
        st.session_state.buffer = WindowBuffer(
            window_size=WINDOW_SIZE,
            calibration_threshold=CALIBRATION_THRESHOLD,
        )

    if "simulator" not in st.session_state:
        from dashboard.simulation.medsec25_simulator import MedSec25StreamSimulator
        st.session_state.simulator = MedSec25StreamSimulator()

    if "watcher" not in st.session_state:
        st.session_state.watcher = None

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
    st.markdown(
        f"<style>{CSS_PATH.read_text()}</style>",
        unsafe_allow_html=True,
    )

# HIPAA disclaimer banner
st.markdown(
    '<div class="hipaa-banner">'
    "This dashboard is for <b>research purposes only</b>. "
    "No Protected Health Information (PHI) is stored or transmitted."
    "</div>",
    unsafe_allow_html=True,
)

# Initialize state
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
with st.sidebar:
    st.markdown("### RA-X-IoMT")
    st.caption("Risk-Adaptive Explainable IDS for IoMT")

    st.markdown("---")

    # System info from ground truth
    arch = gt.get("model_architecture", {})
    params = arch.get("total_params", 482817)
    perf = gt.get("performance", {})
    threshold = perf.get("optimal_threshold", 0.608)

    st.markdown(f"**Model:** CNN-BiLSTM-Attention v2")
    st.markdown(f"**Dataset:** WUSTL-EHMS-2020")
    st.markdown(f"**Params:** {params:,}")
    st.markdown(f"**Threshold:** {threshold:.3f} (Youden's J)")

    # Artifact status
    inv = gt.get("artifact_inventory", {})
    verified = inv.get("verified", 0)
    total = inv.get("total_artifacts_checked", 0)
    missing = inv.get("missing", 0)
    st.markdown(f"**Status:** {verified}/{total} verified, {missing} missing")

    st.markdown("---")

    # Role selector
    role = st.selectbox("View as", list(ROLES.keys()), index=0)
    allowed_panels = ROLES[role]

    # Navigation — filtered by role
    pages_for_role = [
        PAGE_LABELS[p] for p in allowed_panels
        if p in PAGE_LABELS
    ]
    page = st.radio("Navigation", pages_for_role, index=0)

    st.markdown("---")

    # Ground truth timestamp
    ts = gt.get("extraction_timestamp", "")
    if ts:
        st.caption(f"Ground Truth: {ts[:19]}")

    # Re-extract button
    if st.button("Re-extract Ground Truth"):
        import subprocess
        with st.spinner("Extracting..."):
            result = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "extract_project_reality.py")],
                capture_output=True, text=True, cwd=str(PROJECT_ROOT),
            )
        if result.returncode == 0:
            st.success("Extraction complete")
            load_ground_truth.clear()
            st.rerun()
        else:
            st.error(f"Extraction failed: {result.stderr[-200:]}")

    # Auto-refresh
    auto_refresh = st.checkbox("Auto-refresh (5s)", value=False)

    st.markdown("---")

    # Simulation controls in sidebar
    sim = st.session_state.simulator
    sim_status = sim.get_status()

    if sim_status.get("medsec25_available"):
        st.markdown("##### Simulation")

        sim_mode = st.selectbox(
            "Mode", ["ACCELERATED", "REALTIME", "STRESS"],
            index=0, key="sim_mode_sidebar",
        )
        sim_scenario = st.selectbox(
            "Scenario",
            ["A: Benign Only", "B: Gradual Attack",
             "C: Abrupt Attack", "D: Mixed Cycle"],
            index=2, key="sim_scenario_sidebar",
        )

        scol1, scol2, scol3 = st.columns(3)
        with scol1:
            if st.button("Start", disabled=sim.running, key="sim_start"):
                from dashboard.simulation.scenarios import ScenarioID, SimMode
                scenario_id = ScenarioID(sim_scenario[0])
                mode = SimMode[sim_mode]
                sim.start(scenario_id, mode)
                st.rerun()
        with scol2:
            if st.button("Stop", disabled=not sim.running, key="sim_stop"):
                sim.stop()
                st.rerun()
        with scol3:
            if st.button("Reset", key="sim_reset"):
                sim.reset()
                st.session_state.buffer.reset()
                st.rerun()

        if sim.running:
            st.info(
                f"Scenario {sim_status.get('scenario', '?')} | "
                f"{sim_status.get('mode', '')} | "
                f"{sim_status.get('flows_injected', 0):,} flows"
            )

    # Disclosure banner
    st.markdown("---")
    st.warning(
        "**RESEARCH PROTOTYPE**\n\n"
        "This dashboard operates in MedSec-25 simulation mode. "
        "No PHI is stored or transmitted. "
        "Not validated for clinical deployment. "
        "Results reflect conservative lower bound estimates "
        "attributable to feature imputation methodology."
    )


# ── Page Routing ─────────────────────────────────────────────────────────
buffer_status = st.session_state.buffer.get_status()
sim_status = st.session_state.simulator.get_status()

if page == "Operational Status":
    from dashboard.components.panel_operational import render
    render(gt, buffer_status)

elif page == "Stakeholder Intelligence":
    from dashboard.components.panel_stakeholder import render
    render(gt, role=role)

elif page == "Alert Feed":
    from dashboard.components.panel_alerts import render
    live_alerts = st.session_state.buffer.get_alerts() if sim_status.get("running") else None
    selected = render(gt, live_alerts)
    if selected is not None:
        st.session_state.selected_alert = selected

elif page == "SHAP Explanations":
    from dashboard.components.panel_shap import render
    render(gt, st.session_state.selected_alert)

elif page == "Evaluation Results":
    from dashboard.components.panel_evaluation import render
    render(gt)

elif page == "Model & Training":
    from dashboard.components.panel_model import render
    render(gt)

elif page == "Risk Analytics":
    from dashboard.components.panel_risk import render
    render(gt, buffer_status, sim_status)

elif page == "Device Inventory":
    from dashboard.components.panel_devices import render
    render(gt)

elif page == "Cross-Dataset & Simulation":
    from dashboard.components.panel_crossval import render
    sim_actions = render(gt, sim_status)

    if sim_actions and sim_actions.get("action"):
        action = sim_actions["action"]
        if action == "start":
            from dashboard.simulation.scenarios import ScenarioID, SimMode
            sid = ScenarioID(sim_actions.get("scenario", "C"))
            mode = SimMode[sim_actions.get("mode", "ACCELERATED")]
            st.session_state.simulator.start(sid, mode)
            st.rerun()
        elif action == "stop":
            st.session_state.simulator.stop()
            st.rerun()
        elif action == "reset":
            st.session_state.simulator.reset()
            st.session_state.buffer.reset()
            st.rerun()

elif page == "Compliance & Audit":
    from dashboard.components.panel_compliance import render
    render(gt, sim_active=sim_status.get("running", False))

# ── Auto-refresh ─────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(5)
    st.rerun()
