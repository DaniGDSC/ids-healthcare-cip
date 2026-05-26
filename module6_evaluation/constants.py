"""Module 6 constants — role names, tier colors, action priority maps.

All module-level dicts that were defined inside ``module6_app.py``. Pure
data, no Streamlit dependencies, importable from tests without bootstrapping
session state.
"""
from __future__ import annotations

# ── Role display ──────────────────────────────────────────────────────────
# Canonical display names (spec triad). Internal data keys remain lowercase
# analyst / clinician / administrator to preserve backward compatibility with
# ``participant_responses.json``; user-visible labels use the spec triad below.
#
# Mapping:
#   analyst       ↔ IT Generalist   (SOC analyst, IT support)
#   administrator ↔ Biomed Engineer (biomedical engineering, service line owner)
#   clinician     ↔ Nurse Manager   (bedside clinician, charge nurse)
ROLE_DISPLAY_NAMES = {
    "analyst": "IT Generalist",
    "administrator": "Biomed Engineer",
    "clinician": "Nurse Manager",
}
ROLE_INTERNAL_KEY = {v: k for k, v in ROLE_DISPLAY_NAMES.items()}
ROLE_ORDER = ("analyst", "administrator", "clinician")
ROLE_DISPLAY_LIST = [ROLE_DISPLAY_NAMES[k] for k in ROLE_ORDER]
ROLES = ROLE_DISPLAY_LIST  # legacy alias
ROLE_SHORT_LABELS = {
    "analyst": "IT",
    "administrator": "Biomed",
    "clinician": "Nurse",
}

# ── Actions ──────────────────────────────────────────────────────────────
ACTIONS = ["dismiss", "monitor", "investigate", "isolate", "escalate"]

# ── Tier coloring ────────────────────────────────────────────────────────
TIER_COLORS = {
    "CRITICAL": "#8e44ad", "HIGH": "#e74c3c",
    "MEDIUM": "#e67e22", "LOW": "#2ecc71",
}
TIER_STREAMLIT_COLORS = {
    "CRITICAL": "violet", "HIGH": "red",
    "MEDIUM": "orange", "LOW": "green",
}

DETECTOR_CONSENSUS_LABEL = "Detector consensus"

# ── Device class inference fallback ──────────────────────────────────────
_CATEGORY_TO_DEVICE = {
    "Spoofing": "iomt_device",
    "Data Alteration": "iomt_device",
    "iomt_deviation": "iomt_device",
    "anomalous_outbound": "iomt_device",
    "lateral_movement": "workstation",
    "data_exfiltration": "ehr_workstation",
    "ehr_access": "ehr_workstation",
}

# ── Action display ordering for consensus + priority bucketing ───────────
_ACTION_DISPLAY = {
    "isolate_device":        (1, "\U0001f534", "Isolate device"),
    "escalate_incident":     (2, "\U0001f7e0", "Escalate to security lead"),
    "escalate_clinical":     (2, "\U0001f7e0", "Escalate to clinical engineering"),
    "restrict_traffic":      (3, "\U0001f7e1", "Restrict suspicious traffic"),
    "re_authenticate":       (3, "\U0001f7e1", "Force re-authentication"),
    "forensic_snapshot":     (4, "\U0001f535", "Capture forensic snapshot"),
    "enhanced_monitoring":   (5, "\U0001f7e2", "Enable enhanced monitoring"),
    "log_event":             (6, "⚪", "Log event"),
}
_ACTION_DISPLAY_MISS = (99, "⚪", "")

_CRIT_COLOR_HEX = {
    "CRITICAL": "#d32f2f", "HIGH": "#f57c00",
    "MEDIUM": "#1976d2", "LOW": "#388e3c",
}

_PA_MAP = {
    "isolate_device":     "Isolate device",
    "escalate_incident":  "Escalate to security lead",
    "escalate_clinical":  "Escalate to clinical engineering",
    "restrict_traffic":   "Restrict suspicious traffic",
    "re_authenticate":    "Force re-authentication",
    "enhanced_monitoring": "Enhanced monitoring",
    "forensic_snapshot":  "Capture forensic snapshot",
    "log_event":          "Log and monitor",
}

# Bucketing used by process_alert() to roll up policy actions to coarse
# operator categories (isolate / escalate / investigate / monitor).
_ACTION_PRIORITY = {
    "isolate_device":    "isolate",
    "escalate_incident": "escalate",
    "escalate_clinical": "escalate",
    "restrict_traffic":  "investigate",
    "forensic_snapshot": "investigate",
    "re_authenticate":   "investigate",
    "enhanced_monitoring": "monitor",
    "log_event":         "monitor",
}

# ── Page → split routing ─────────────────────────────────────────────────
# Each operator-facing page reads a *specific* frozen split.
PAGE_SPLIT: dict[str, str | None] = {
    "Dashboard":         "test",
    "Online Simulation": "demo",
    "Browse Alerts":     "test",
    "Study (A/B)":       None,
    "PCAP Replay":       None,
}

_SPLIT_FILES = {
    "test": "",
    "demo": "_demo",
}


def resolve_suffix(split: str | None) -> str:
    """Resolve the file suffix for a split, with strict validation.

    ``None`` → ``""`` (legacy test fallback for dashboard callers that
    haven't migrated to passing split explicitly). Any other value is
    delegated to :func:`common.split_paths.suffix`, which raises
    ``ValueError`` on typos.
    """
    if split is None:
        return ""
    from common import split_paths as sp
    return sp.suffix(split)


__all__ = [
    "ROLE_DISPLAY_NAMES", "ROLE_INTERNAL_KEY", "ROLE_ORDER",
    "ROLE_DISPLAY_LIST", "ROLES", "ROLE_SHORT_LABELS",
    "ACTIONS",
    "TIER_COLORS", "TIER_STREAMLIT_COLORS", "DETECTOR_CONSENSUS_LABEL",
    "_CATEGORY_TO_DEVICE",
    "_ACTION_DISPLAY", "_ACTION_DISPLAY_MISS",
    "_CRIT_COLOR_HEX", "_PA_MAP", "_ACTION_PRIORITY",
    "PAGE_SPLIT", "_SPLIT_FILES", "resolve_suffix",
]
