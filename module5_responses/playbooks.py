"""Conditional action playbooks for stakeholder views (Phase 3.1).

Each playbook is an ordered list of ``PlaybookStep`` items describing a
short, branching decision tree the operator can follow:

    Step 1: <check>
        if YES → <action_yes>  (terminal)
        if NO  → continue to Step 2
    Step 2: <check>
        if YES → <action_yes>  (terminal)
        if NO  → continue to Step 3
    Step 3: <terminal action>

The aim is to give a non-ML operator a *procedure* rather than a single
prescribed action. The branches are deliberately rooted in things the
operator can *actually verify* — manual vital readings, backup monitor
agreement, network connectivity — not in SHAP magnitudes or model
confidences. Picking the right playbook is keyed on the SHAP top
category from ``module4_explanations.feature_groups`` plus the
canonical risk_level.

The render functions emit Markdown (for clinician_summaries.json) and
structured dicts (for alert_responses.json) so the same data drives
both surface forms.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable


# ── Step + Playbook dataclasses ────────────────────────────────────


@dataclass(frozen=True)
class PlaybookStep:
    """One node in a decision tree.

    A step either branches on a check (when ``check`` is non-empty) or
    is a terminal action (when ``check`` is empty — the operator is
    instructed to perform ``action_yes`` regardless).
    """
    check: str
    action_yes: str
    action_no: str = ""    # empty → "continue to next step"

    def is_terminal(self) -> bool:
        return not self.check


@dataclass(frozen=True)
class Playbook:
    name: str
    description: str
    steps: tuple[PlaybookStep, ...]
    severity_floor: str = "LOW"   # severity tier at/above which this fires

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "steps": [asdict(s) for s in self.steps],
        }


# ── Reusable step builders ────────────────────────────────────────


# Reusable verification checks that recur across multiple playbooks.
# Phrasings are clinician-friendly — no SHAP, no model jargon.
_CHECK_VITAL_BACKUP   = ("Compare vital reading on the IDS-flagged "
                         "device against the bedside backup monitor "
                         "or a manual measurement")
_CHECK_TRAFFIC_VENDOR = ("Confirm whether the destination IP belongs "
                         "to a known vendor maintenance host or appears "
                         "in the approved-destinations list")
_CHECK_DEVICE_HEALTH  = ("Inspect the device for physical disconnection, "
                         "cable damage, or sensor displacement at the "
                         "bedside")
_CHECK_REMEDIATION    = ("Apply the counterfactual-derived remediation "
                         "(`response.try_first_action`) and observe "
                         "whether the alert clears within 60 seconds")


# ── Per-category playbooks ────────────────────────────────────────


_PLAYBOOK_BIOMETRIC = Playbook(
    name="biometric_anomaly",
    description=(
        "Network alert with biometric SHAP contributors — verify the "
        "vital sign before treating the alert as a device compromise."
    ),
    steps=(
        PlaybookStep(
            check=_CHECK_VITAL_BACKUP,
            action_yes="Acknowledge and log (likely IDS noise; vital confirmed normal).",
        ),
        PlaybookStep(
            check=_CHECK_DEVICE_HEALTH,
            action_yes="Resolve the physical issue first; suppress the alert and re-monitor.",
        ),
        PlaybookStep(
            check="",
            action_yes=("Escalate to clinical staff for independent patient assessment "
                        "AND notify Biomed Engineering to inspect device integrity."),
        ),
    ),
    severity_floor="LOW",
)


_PLAYBOOK_NETWORK_VOLUME = Playbook(
    name="network_volume_anomaly",
    description=(
        "Unusual byte volume / load on the device — try the cheapest "
        "remediation (port-level throttle) before isolating the device."
    ),
    steps=(
        PlaybookStep(
            check=_CHECK_TRAFFIC_VENDOR,
            action_yes="Acknowledge as scheduled maintenance; log destination + timestamp.",
        ),
        PlaybookStep(
            check=_CHECK_REMEDIATION,
            action_yes="Counterfactual remediation succeeded — log and monitor.",
        ),
        PlaybookStep(
            check="",
            action_yes=("Isolate the device and capture forensic snapshot. "
                        "Notify SOC + Biomed Engineering."),
        ),
    ),
    severity_floor="LOW",
)


_PLAYBOOK_NETWORK_PROTOCOL = Playbook(
    name="network_protocol_anomaly",
    description=(
        "Unusual port / flag pattern — likely scan or protocol drift. "
        "Block the offending port before broader isolation."
    ),
    steps=(
        PlaybookStep(
            check=_CHECK_TRAFFIC_VENDOR,
            action_yes="Acknowledge as known maintenance traffic.",
        ),
        PlaybookStep(
            check=_CHECK_REMEDIATION,
            action_yes="Port-block remediation succeeded — log and monitor.",
        ),
        PlaybookStep(
            check="",
            action_yes=("Restrict device to clinical-essential traffic and "
                        "open SOC incident."),
        ),
    ),
    severity_floor="LOW",
)


_PLAYBOOK_NETWORK_TIMING = Playbook(
    name="network_timing_anomaly",
    description=(
        "Packet timing drift — consistent with covert channel or "
        "link congestion. Investigate before isolating."
    ),
    steps=(
        PlaybookStep(
            check=("Check whether the timing anomaly correlates with a "
                   "known link-saturation event or scheduled backup window"),
            action_yes="Acknowledge as expected network behaviour.",
        ),
        PlaybookStep(
            check=_CHECK_REMEDIATION,
            action_yes="Timing remediation succeeded — log and monitor.",
        ),
        PlaybookStep(
            check="",
            action_yes=("Capture packet trace for the affected flow and "
                        "open SOC incident."),
        ),
    ),
    severity_floor="LOW",
)


_PLAYBOOK_DEFAULT = Playbook(
    name="default",
    description=(
        "Fallback when no category-specific playbook matches the alert. "
        "Falls back to the prescribed actions from the response engine."
    ),
    steps=(
        PlaybookStep(
            check=_CHECK_REMEDIATION,
            action_yes="Remediation succeeded — log and monitor.",
        ),
        PlaybookStep(
            check="",
            action_yes=("Follow the prescribed action set "
                        "(`response.action_descriptions`)."),
        ),
    ),
    severity_floor="LOW",
)


# ── Category → playbook table ─────────────────────────────────────


_CATEGORY_PLAYBOOKS: dict[str, Playbook] = {
    "biometric":         _PLAYBOOK_BIOMETRIC,
    "network_volume":    _PLAYBOOK_NETWORK_VOLUME,
    "network_protocol":  _PLAYBOOK_NETWORK_PROTOCOL,
    "network_timing":    _PLAYBOOK_NETWORK_TIMING,
    "network_packet":    _PLAYBOOK_NETWORK_PROTOCOL,  # similar surface area
    "network_loss":      _PLAYBOOK_NETWORK_TIMING,    # similar
    "unknown":           _PLAYBOOK_DEFAULT,
}


_SEVERITY_RANK = {"NORMAL": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}


def select_playbook(top_category: str, severity: str) -> Playbook:
    """Pick the playbook for a given (category, severity) pair.

    Returns the category-specific playbook when the severity meets or
    exceeds the playbook's ``severity_floor``; otherwise falls back to
    the default playbook (a single "follow the prescribed actions"
    step). Unknown categories also fall through to the default.

    The default playbook itself has ``severity_floor="LOW"``, so this
    function always returns a usable playbook for any non-NORMAL alert.
    """
    pb = _CATEGORY_PLAYBOOKS.get(top_category, _PLAYBOOK_DEFAULT)
    if _SEVERITY_RANK.get(severity, 0) < _SEVERITY_RANK.get(pb.severity_floor, 0):
        return _PLAYBOOK_DEFAULT
    return pb


# ── Markdown renderer ─────────────────────────────────────────────


def render_markdown(playbook: Playbook) -> str:
    """Render a playbook as a numbered Markdown checklist.

    Format::

        **Playbook: <name>**
        1. **Check:** <check phrase>
           - if YES → <action_yes>
        2. **Check:** <check phrase>
           - if YES → <action_yes>
        3. **Action:** <terminal action>
    """
    out: list[str] = [f"**Playbook: {playbook.name}**"]
    n = 1
    for step in playbook.steps:
        if step.is_terminal():
            out.append(f"{n}. **Action:** {step.action_yes}")
        else:
            out.append(f"{n}. **Check:** {step.check}")
            out.append(f"   - if YES → {step.action_yes}")
            out.append(f"   - if NO  → continue to next step")
        n += 1
    return "\n".join(out)


__all__ = [
    "Playbook",
    "PlaybookStep",
    "select_playbook",
    "render_markdown",
]
