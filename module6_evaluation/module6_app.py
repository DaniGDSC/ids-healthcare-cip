#!/usr/bin/env python3
"""Module 6 — Evaluation Interface (Tasks 6.3a/b/c, 6.4, 6.5, 6.9).

Three modes:
  6.3a  Offline — Browse/Study pre-computed alerts with Likert questionnaires
  6.3b  Online Simulation — Stream test samples through pipeline in near-real-time
  6.3c  Dashboard — Risk gauge, alert feed, SHAP waterfall, NLG panel, response panel,
        admin heatmap, tier distribution chart

Usage:
    streamlit run module6_evaluation/module6_app.py
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections import ChainMap, Counter, deque
from datetime import datetime
from pathlib import Path

import sys

import numpy as np
import pandas as pd
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:  # pragma: no cover - optional dependency

    def st_autorefresh(*args, **kwargs) -> int:
        return 0


# When invoked via `streamlit run module6_evaluation/module6_app.py`
# the project root is NOT on sys.path (streamlit treats the file as a
# script, not a package). Prepend it so the absolute import below works.
_PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT_FOR_IMPORT))

# Hardened audit logger from Module 5 — used to bind reviewer attribution
# (participant_id / role / timestamp) from st.session_state to a signed,
# hash-chained record in results/reports/audit_log.jsonl.
from module5_responses.module5_pipeline import AuditLogger as HardenedAuditLogger  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "results/reports"
CHARTS_DIR = PROJECT_ROOT / "results/charts"
MODELS_DIR = PROJECT_ROOT / "results/models"
# Singleton hardened logger for reviewer-attributed events. The existing
# AuditTrailWriter (audit_trail.jsonl) is kept for backward compatibility
# with offline study mode; reviewer-attributed alert decisions ALSO get
# logged to the signed audit_log.jsonl chain.
_hardened_audit = HardenedAuditLogger(EVAL_DIR / "audit_log.jsonl")

ACTIONS = ["dismiss", "monitor", "investigate", "isolate", "escalate"]

TIER_COLORS = {"CRITICAL": "#8e44ad", "HIGH": "#e74c3c", "MEDIUM": "#e67e22", "LOW": "#2ecc71"}

from common.phi import BIOMETRIC_COLUMNS as BIOMETRIC_FEATURES  # noqa: E402

# Wires the dashboard's per-alert processing to the research prototype's
# Risk-Adaptive Scoring Engine (research_spec.yaml component_2) so tier
# assignment uses the same logic the prototype tests enforce (M7, M6).
from module6_evaluation.study_loader import assign_ab_condition, load_study_alerts  # noqa: E402

# v4 visual helpers (Layer 5 v4.0 — 9-class badges, confidence dots, mode A/B).
# Pure metadata module; safe to import at app startup.
from src.data_models import AlertType, Confidence, OperatorRole  # noqa: E402
from module6_evaluation.presentation_v4 import (  # noqa: E402
    BADGE_FOR_ALERT_TYPE,
    CONFIDENCE_INDICATOR,
    MODE_INDICATOR,
    MODE_A_LLM,
    MODE_B_RULE_BASED,
    anomalous_dims_markdown,
    badge_for_alert_type,
    confidence_display,
    mode_display,
)
# v4 MITRE per-role formatting (built once in module4_explanations,
# reused here so the Dashboard / Sim / Browse all surface the same
# technique copy per role).
from module4_explanations.triage_v4_adapter import (  # noqa: E402
    format_mitre_for_alert_type,
    format_mitre_for_role,
)


# ═══════════════════════════════════════════════════════════════════════
# MVE Display Helpers (Gap 1, Gap 2, Gap 3 fixes)
# ═══════════════════════════════════════════════════════════════════════

# FIX B: Infer device class from attack category when asset lookup fails
_CATEGORY_TO_DEVICE = {
    "Spoofing": "iomt_device",
    "Data Alteration": "iomt_device",
    "iomt_deviation": "iomt_device",
    "anomalous_outbound": "iomt_device",
    "lateral_movement": "workstation",
    "data_exfiltration": "ehr_workstation",
    "ehr_access": "ehr_workstation",
}

# FIX C: Action priority ordering for consensus display
_ACTION_DISPLAY = {
    "isolate_device": (1, "\U0001f534", "Isolate device"),
    "escalate_incident": (2, "\U0001f7e0", "Escalate to security lead"),
    "escalate_clinical": (2, "\U0001f7e0", "Escalate to clinical engineering"),
    "restrict_traffic": (3, "\U0001f7e1", "Restrict suspicious traffic"),
    "re_authenticate": (3, "\U0001f7e1", "Force re-authentication"),
    "forensic_snapshot": (4, "\U0001f535", "Capture forensic snapshot"),
    "enhanced_monitoring": (5, "\U0001f7e2", "Enable enhanced monitoring"),
    "log_event": (6, "\u26aa", "Log event"),
}

_CRIT_COLOR_HEX = {
    "CRITICAL": "#d32f2f",
    "HIGH": "#f57c00",
    "MEDIUM": "#1976d2",
    "LOW": "#388e3c",
}

# Module-level policy action label map — avoids rebuilding this dict on every
# render_mve_layers() call (issue 4 / render_mve_layers locality fix).
_PA_MAP = {
    "isolate_device": "Isolate device",
    "escalate_incident": "Escalate to security lead",
    "escalate_clinical": "Escalate to clinical engineering",
    "restrict_traffic": "Restrict suspicious traffic",
    "re_authenticate": "Force re-authentication",
    "enhanced_monitoring": "Enhanced monitoring",
    "forensic_snapshot": "Capture forensic snapshot",
    "log_event": "Log and monitor",
}

# Module-level sentinel for _ACTION_DISPLAY misses — avoids {} allocation
# per sort-key lambda invocation (issue 9).
_ACTION_DISPLAY_MISS = (99, "\u26aa", "")

# M6-A1: hoist _ACTION_PRIORITY to module level — was rebuilt as a dict
# literal on every process_alert() call (one call per alert per simulation tick).
_ACTION_PRIORITY = {
    "isolate_device": "isolate",
    "escalate_incident": "escalate",
    "escalate_clinical": "escalate",
    "restrict_traffic": "investigate",
    "forensic_snapshot": "investigate",
    "re_authenticate": "investigate",
    "enhanced_monitoring": "monitor",
    "log_event": "monitor",
}


def render_device_criticality(alert: dict) -> None:
    """Gap 2 + UX-X-02: Render device class criticality badge + context."""
    criticality = str(alert.get("device_criticality", "")).upper()
    if not criticality or criticality not in _CRIT_COLOR_HEX:
        criticality = str(alert.get("risk_level", "UNKNOWN")).upper()
    hex_c = _CRIT_COLOR_HEX.get(criticality, "#757575")

    # FIX B: infer device class when missing
    device_cls, was_inferred = infer_device_class(alert)
    affected = alert.get("affected_system", "")

    st.markdown(
        f'<span style="background:{hex_c};color:white;'
        f'padding:3px 10px;border-radius:4px;font-weight:bold;">'
        f"Device: {criticality}</span>",
        unsafe_allow_html=True,
    )
    if device_cls or affected:
        st.caption(f"{device_cls}{' — ' + affected if affected else ''}")
    if was_inferred:
        st.caption(
            f"\u26a0\ufe0f Device class inferred from attack category "
            f"({alert.get('attack_category', '?')}) — asset inventory lookup unavailable."
        )

    # UX-X-01: Patient impact warning
    impact = alert.get("patient_care_impact", "")
    active = alert.get("active_device", False)
    if active and impact:
        st.warning(f"\U0001f3e5 Active device — {impact}")
    elif impact:
        st.info(f"\u2139\ufe0f {impact}")


def infer_device_class(alert: dict) -> tuple[str, bool]:
    """FIX B: Return (device_class, was_inferred) with fallback from attack category."""
    device_cls = alert.get("device_class", "")
    if device_cls and device_cls not in ("", "other", "unknown"):
        return device_cls, False
    attack_cat = alert.get("attack_category", "")
    inferred = _CATEGORY_TO_DEVICE.get(attack_cat, "")
    if inferred:
        return inferred, True
    return device_cls or "unknown", not bool(device_cls)


def render_prioritized_actions(actions: list) -> None:
    """FIX C: Render response actions in priority order with icons.

    Issue 9 fix: sort key uses _ACTION_DISPLAY_MISS sentinel (module-level
    constant) instead of creating a throwaway (99, "", a) tuple on every
    miss. The label fallback uses the action string directly from `act`.
    """
    if not actions:
        return
    sorted_actions = sorted(
        actions,
        key=lambda a: _ACTION_DISPLAY.get(a, _ACTION_DISPLAY_MISS)[0],
    )
    st.markdown("**Response (in priority order):**")
    for act in sorted_actions:
        entry = _ACTION_DISPLAY.get(act)
        if entry:
            priority, icon, label = entry
        else:
            priority, icon, label = 99, "\u26aa", act
        if priority == 1:
            st.error(f"{icon} **Primary: {label}**")
        elif priority <= 3:
            st.warning(f"{icon} {label}")
        else:
            st.info(f"{icon} {label}")


def render_do_not_constraint(layer3_text: str, severity: str = "") -> None:
    """Extract and render DO NOT constraint from Layer 3 text."""
    if not layer3_text:
        return
    for line in layer3_text.replace("\n", " ").split(". "):
        stripped = line.strip()
        if "DO NOT" in stripped.upper():
            st.warning(f"\u26a0\ufe0f {stripped.rstrip('.')}.")
            return
    # No DO NOT found — silence is correct.
    # Device-class fallbacks are handled by render_mve_layers().


# Text matches src/mve_generator.py T5 device-specific constraints exactly
_DO_NOT_FALLBACKS = {
    "infusion_pump": "DO NOT power-cycle pump during active infusion. SAFE: NAC quarantine blocking non-HTTPS preserves controller.",
    "ventilator": "DO NOT power off or disconnect ventilator. SAFE: block port at switch — clinical traffic on 443 unaffected.",
    "patient_monitor": "DO NOT isolate from EHR gateway — vitals must continue. SAFE: DNS rate-limit or port block — HL7 on 443 unaffected.",
    "insulin_pump": "DO NOT disrupt wireless control loop. SAFE: destination-specific block only if IP-connected.",
    "ehr_workstation": "DO NOT suspend account without verifying role — disrupts active clinical documentation.",
    "pacs_server": "DO NOT shut down PACS — active radiology reads depend on image delivery.",
    "pharmacy_system": "DO NOT disable dispensing system — automated drug delivery depends on availability.",
    "iomt_device": "DO NOT isolate without contacting Biomed Engineering — clinical function unconfirmed.",
    "workstation": "DO NOT lock workstation without verifying active clinical sessions.",
}


def render_mve_layers(alert: dict) -> None:
    """Render alert content as explicit Layer 1/2/3 sections.

    Searches alert top-level, nested xai_explanation, and nested
    explanation dicts. Falls back to clinician summary, response
    policy fields, and device-class-based DO NOT fallbacks.

    Issue 4 fix: replaced nested _get() loop (O(sources × keys) per field
    lookup) with a single ChainMap merge at entry. ChainMap holds views of
    the 4 source dicts with no copy; lookups are O(1) average against the
    merged namespace. All _get() calls become simple dict.get() on _cm.
    """
    xai = alert.get("xai_explanation") or {}
    expl = alert.get("explanation") or {}
    resp = alert.get("response") or {}

    # Merge once — O(1) view construction, O(1) per key lookup thereafter.
    _cm = ChainMap(
        alert,
        xai if isinstance(xai, dict) else {},
        expl if isinstance(expl, dict) else {},
        resp if isinstance(resp, dict) else {},
    )

    def _get(*keys) -> str:
        """First non-empty string value for any key across the merged view."""
        for k in keys:
            v = _cm.get(k)
            if v and isinstance(v, str) and v.strip():
                return v
        return ""

    # ── Layer 1: Why Anomalous ──
    l1 = _get(
        "why_anomalous",
        "layer_1",
        "baseline_behavior",
        "deviation_description",
        "confidence_indicator",
        "clinician_summary",
        "nlg_text",
    )
    consensus = _get("consensus")

    with st.expander("\U0001f50d Layer 1 \u2014 Why Anomalous", expanded=True):
        if l1:
            st.write(l1)
            if consensus:
                st.caption(f"Model consensus: {consensus}")
        else:
            st.caption("Baseline deviation detected. See SHAP features below.")

        # Day 2: MITRE per role. The technique is derived from the v4
        # ``alert_type`` (which we derive heuristically from the legacy
        # alert fields) so the line goes live without waiting for Layer 3
        # to plumb ``mitre_technique_id`` into the alert schema.
        try:
            alert_type, _, _ = derive_v4_fields(alert)
            op_role = get_current_operator_role()
            mitre_line = format_mitre_for_alert_type(alert_type, op_role)
        except Exception:  # never break Layer 1 over a MITRE lookup miss
            mitre_line = ""
        if mitre_line:
            st.markdown(f"**Threat intelligence:** {mitre_line}")

    # ── Layer 2: Clinical Severity ──
    affected = _get("affected_system")
    impact = _get("patient_care_impact")
    severity = _get("severity_label", "severity", "risk_level", "tier")
    device_tier = _get("device_tier", "device_class")

    with st.expander("\U0001f3e5 Layer 2 \u2014 Clinical Severity", expanded=True):
        if severity:
            color = TIER_COLORS.get(severity.upper(), "#999")
            st.markdown(
                f"**Severity:** <span style='color:{color}'>{severity}</span>",
                unsafe_allow_html=True,
            )
        if affected:
            st.write(f"**Affected system:** {affected}")
        elif device_tier:
            st.write(f"**Device tier:** {device_tier}")
        if impact:
            st.write(f"**Patient impact:** {impact}")
        if not affected and not impact and not severity:
            st.caption("Layer 2 data not available for this alert.")

    # ── Layer 3: Recommended Action ──
    action = _get(
        "recommended_action", "layer_3", "immediate_action", "response_action", "correct_action"
    )
    constraint = _get("clinical_constraint")
    rationale = _get("rationale")

    # If no explicit action field, derive from response.actions policy list
    if not action and isinstance(resp, dict):
        policy_actions = resp.get("actions", [])
        for pa in reversed(policy_actions):
            if pa in _PA_MAP:
                action = _PA_MAP[pa]
                break

    with st.expander("\u26a1 Layer 3 \u2014 Recommended Action", expanded=True):
        if action:
            st.write(f"**Recommended:** {action}")
        else:
            actions_list = resp.get("actions", []) if isinstance(resp, dict) else []
            if actions_list:
                st.write(f"**Actions:** {', '.join(actions_list)}")
            elif rationale:
                st.write(rationale[:200])
            else:
                st.caption("Layer 3 data not available for this alert.")

        # DO NOT constraint — explicit field, fallback, or silence
        full_l3 = f"{action} {constraint} {rationale}"
        if "DO NOT" in full_l3.upper():
            render_do_not_constraint(full_l3, severity)
        else:
            # FIX B: device-class fallback for HIGH/CRITICAL
            device_cls, _ = infer_device_class(alert)
            if not device_cls:
                device_cls = _get("device_tier")
            sev_upper = severity.upper() if severity else ""
            if sev_upper in ("CRITICAL", "HIGH"):
                fallback = _DO_NOT_FALLBACKS.get(device_cls, "")
                if fallback:
                    st.warning(f"\u26a0\ufe0f {fallback}")
                else:
                    st.warning(
                        "\u26a0\ufe0f DO NOT isolate or power off without "
                        "contacting Biomed Engineering \u2014 clinical function unknown."
                    )
            # MEDIUM/LOW: show nothing (silence is correct)


# ═══════════════════════════════════════════════════════════════════════
# v4 visual helpers (Layer 5 v4.0 — 9-class badges, confidence, mode A/B)
# ═══════════════════════════════════════════════════════════════════════

# Heuristic mapping: cat∈{spoofing, data alteration} ⇒ "known" Track-A category.
# Anything else (or empty) is treated as a novel/non-cataloged anomaly.
_KNOWN_ATTACK_CATEGORIES: frozenset[str] = frozenset({"spoofing", "data alteration"})


def derive_v4_fields(alert: dict) -> tuple[AlertType, Confidence, str]:
    """Best-effort heuristic mapping legacy alert schema → v4 fields.

    The Layer 3 v4 fusion does not yet write ``alert_type`` /
    ``confidence`` / ``generation_mode`` into ``evaluation_alerts.json``.
    Until that plumbing lands, the dashboard derives them from the
    fields that *are* present (``ground_truth``, ``attack_category``,
    ``risk_level``, ``risk_score``) so the v4 visual treatment is
    exercised end-to-end. Replace this call with a direct field read
    once the upstream pipeline emits these fields natively.

    Args:
        alert: A row from ``evaluation_alerts.json`` /
            ``alert_responses.json`` (loose schema; missing keys are
            tolerated and degrade to BENIGN / LOW / Mode B).

    Returns:
        ``(alert_type, confidence, generation_mode)``.
    """
    gt = (alert.get("ground_truth") or "").lower()
    cat = (alert.get("attack_category") or "").lower()
    lvl = (alert.get("risk_level") or "").upper()
    try:
        score = float(alert.get("risk_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        score = 0.0

    # Alert type
    if gt == "benign" or cat in {"normal", "benign", ""}:
        if lvl in {"CRITICAL", "HIGH"}:
            alert_type = AlertType.DISAGREEMENT_ANOMALY
        else:
            alert_type = AlertType.BENIGN_WATCH
    elif cat in _KNOWN_ATTACK_CATEGORIES:
        if lvl == "CRITICAL" and score >= 0.80:
            alert_type = AlertType.KNOWN_ATTACK
        elif lvl in {"CRITICAL", "HIGH"}:
            alert_type = AlertType.KNOWN_ATTACK_UNCERTAIN
        elif lvl == "MEDIUM":
            alert_type = AlertType.CONFIRMED_ANOMALY
        else:
            alert_type = AlertType.SUSPICIOUS_PATTERN
    else:
        if lvl == "CRITICAL":
            alert_type = AlertType.STRONG_NOVEL_ANOMALY
        elif lvl == "HIGH":
            alert_type = AlertType.NOVEL_ANOMALY
        elif lvl == "MEDIUM":
            alert_type = AlertType.CONFIRMED_ANOMALY
        else:
            alert_type = AlertType.SUSPICIOUS_PATTERN

    # Confidence
    if score >= 0.85:
        conf = Confidence.VERY_HIGH
    elif score >= 0.70:
        conf = Confidence.HIGH
    elif score >= 0.50:
        conf = Confidence.MEDIUM
    else:
        conf = Confidence.LOW

    # Mode (precedence: explicit field on alert, then env, then Mode B fallback)
    raw_mode = (
        alert.get("generation_mode")
        or alert.get("mve_mode")
        or (alert.get("mve_structured") or {}).get("generation_mode")
    )
    if raw_mode in (MODE_A_LLM, MODE_B_RULE_BASED):
        mode = raw_mode
    else:
        import os
        mode = MODE_A_LLM if os.environ.get("ANTHROPIC_API_KEY") else MODE_B_RULE_BASED
    return alert_type, conf, mode


def render_alert_type_badge(alert_or_type) -> None:
    """Render the 9-class alert-type pill badge.

    Accepts an :class:`AlertType`, the raw string value, or a full
    alert dict (in which case the type is derived via
    :func:`derive_v4_fields`). Unknown strings fall back to ``BENIGN``
    via :func:`badge_for_alert_type`.
    """
    if isinstance(alert_or_type, dict):
        alert_type, _, _ = derive_v4_fields(alert_or_type)
    else:
        alert_type = alert_or_type
    badge = badge_for_alert_type(alert_type)
    st.markdown(
        f"<span style='background:{badge['color']};color:white;"
        f"padding:4px 10px;border-radius:4px;font-weight:600;font-size:13px;'>"
        f"{badge['icon']} {badge['label']}</span>",
        unsafe_allow_html=True,
    )


def render_confidence_indicator(confidence) -> None:
    """Render the 4-level confidence dots indicator."""
    style = confidence_display(confidence)
    label = confidence.value if hasattr(confidence, "value") else str(confidence)
    st.markdown(
        f"<span style='color:{style['color']};font-weight:600;font-size:14px;'>"
        f"{style['symbol']}</span> "
        f"<span style='font-size:12px;color:#374151;'>{label}</span>",
        unsafe_allow_html=True,
    )


def render_mode_indicator(generation_mode: str) -> None:
    """Render the Mode A (LLM) / Mode B (rule-based) badge.

    Side effect: also updates the top-bar mode indicator via
    :func:`set_current_mode` so the global state reflects the most
    recently rendered alert's mode.
    """
    style = mode_display(generation_mode)
    if generation_mode == MODE_A_LLM:
        st.success(style["badge"])
    else:
        st.warning(style["badge"])
    set_current_mode(generation_mode)


def render_dae_anomalous_dims(
    anomalous_dims, feature_names=()
) -> None:
    """Render Layer 2 v4 anomalous dims as a collapsible expander.

    No-op when ``anomalous_dims`` is empty (the helper from
    ``presentation_v4`` returns an empty string in that case).
    """
    md = anomalous_dims_markdown(anomalous_dims, feature_names)
    if not md:
        return
    n = len(list(anomalous_dims))
    with st.expander(f"\U0001f52c DAE anomaly details ({n} dims)", expanded=False):
        st.markdown(md)


def render_dae_top_features(items) -> None:
    """Render the alert's ``xai_explanation.dae_top_features`` list.

    The pipeline already serialises DAE per-dim attribution as a list of
    ``{"feature": str, "weighted_error": float, "pct_contribution": float}``
    dicts (Layer 2 v4 emits the indices, Layer 3/4 enriches them with the
    feature name and contribution share). This helper renders that
    shape directly so the Dashboard does not need to round-trip through
    ``anomalous_dims_markdown`` (which expects ``(int_indices,
    name_table)``).

    No-op on empty lists / non-list inputs so the caller can pass
    ``alert.get("xai_explanation", {}).get("dae_top_features", [])``
    unconditionally and skip the empty-state expander.
    """
    if not isinstance(items, list) or not items:
        return
    n = len(items)
    with st.expander(f"\U0001f52c DAE Anomaly Details ({n} dim{'s' if n != 1 else ''})", expanded=False):
        for it in items:
            if not isinstance(it, dict):
                continue
            name = it.get("feature", "?")
            pct = it.get("pct_contribution")
            err = it.get("weighted_error")
            parts = [f"`{name}`"]
            if isinstance(pct, (int, float)):
                parts.append(f"**{pct:.1f}%** of reconstruction error")
            if isinstance(err, (int, float)):
                parts.append(f"weighted error = {err:.2e}")
            st.markdown("- " + "  ·  ".join(parts))


def _is_safety_floor_alert(alert: dict) -> bool:
    """True iff the alert triggers the safety floor (INVARIANT 2).

    Safety floor = ``risk_level == 'CRITICAL'`` AND device cannot be
    patched. The Sim playback auto-pauses on the first tick that surfaces
    such an alert so the operator (or examiner) sees the explicit
    "do-not-isolate, escalate-instead" branch before the next alert
    advances.

    The schema check looks at the canonical ``device_patchable`` boolean
    (top-level on every ``evaluation_alerts.json`` row). When the field
    is missing we default to ``True`` (patchable) — the *safer* default,
    because a missing field shouldn't manufacture a pause.
    """
    if (alert.get("risk_level") or "").upper() != "CRITICAL":
        return False
    patchable = alert.get("device_patchable", True)
    return patchable is False


def display_alert_header_v4(alert: dict) -> None:
    """Render the v4 alert header strip.

    Layout: [severity] [alert-type badge] [risk-level pill] [confidence] [mode]
    """
    alert_type, conf, mode = derive_v4_fields(alert)
    cols = st.columns([1.5, 2.5, 1.2, 1.4, 1.6])

    with cols[0]:
        severity = (alert.get("true_severity")
                    or alert.get("severity")
                    or alert.get("risk_level")
                    or "LOW")
        color = TIER_COLORS.get(str(severity).upper(), "#999")
        st.markdown(
            f"<span style='background:{color};color:white;"
            f"padding:4px 10px;border-radius:4px;font-weight:600;font-size:13px;'>"
            f"{severity}</span>",
            unsafe_allow_html=True,
        )

    with cols[1]:
        render_alert_type_badge(alert_type)

    with cols[2]:
        risk_level = alert.get("risk_level", "—")
        st.markdown(f"\U0001f4cb **Triage:** {risk_level}")

    with cols[3]:
        render_confidence_indicator(conf)

    with cols[4]:
        render_mode_indicator(mode)


# ═══════════════════════════════════════════════════════════════════════
# 6A.7  Audit Trail Writer (JSONL, immutable with integrity hashes)
# ═══════════════════════════════════════════════════════════════════════


class AuditTrailWriter:
    """Append-only JSONL audit log for every user interaction.

    Issue 11 fix: write buffer defers disk I/O so that high-frequency
    callers (simulation auto-advance at 4× speed = 0.5 s tick) do not
    open+write+close the file on every event.  The buffer flushes
    automatically when it reaches _FLUSH_AFTER records or when flush()
    is called explicitly (e.g. on study completion).

    The hash chain is computed eagerly on every .log() call (so the
    in-memory chain stays consistent) but the bytes hit disk only on flush.
    This preserves integrity while eliminating the per-event file open.
    """

    _FLUSH_AFTER = 10  # records buffered before an automatic disk write

    def __init__(self, path: Path | None = None):
        self.path = path or (EVAL_DIR / "audit_trail.jsonl")
        self.prev_hash = "0" * 64
        self._buffer: list[str] = []

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
        self._buffer.append(json.dumps(record) + "\n")
        if len(self._buffer) >= self._FLUSH_AFTER:
            self.flush()

    def flush(self) -> None:
        """Write all buffered records to disk in a single open/write/close."""
        if not self._buffer:
            return
        with open(self.path, "a", encoding="utf-8") as f:
            f.writelines(self._buffer)
        self._buffer.clear()


# Singleton audit writer for the Streamlit session
_audit_writer = AuditTrailWriter()

# M6-A4: Buffered writer for online_interactions.jsonl — amortises the
# open+write+close that fired on every Confirm/Reject/Note button click.
# Shares the same _FLUSH_AFTER=10 policy as AuditTrailWriter.
_online_writer = AuditTrailWriter(EVAL_DIR / "online_interactions.jsonl")


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
        action = st.selectbox(
            "What action would you take?", ACTIONS, format_func=lambda x: x.capitalize()
        )
        confidence = st.slider("Confidence in your decision (1–5)", 1, 5, 3)

        st.markdown("#### Rate the alert presentation (1 = strongly disagree, 5 = strongly agree)")
        trust = st.slider("I trust this classification", 1, 5, 3, key=f"lt_{form_key}")
        usefulness = st.slider(
            "The information helps me respond appropriately", 1, 5, 3, key=f"lu_{form_key}"
        )
        comprehensibility = st.slider(
            "I understand why this alert was triggered", 1, 5, 3, key=f"lc_{form_key}"
        )
        actionability = st.slider("I know what action to take", 1, 5, 3, key=f"la_{form_key}")

        feedback = st.text_area("Free-text feedback (optional)", key=f"fb_{form_key}")
        reclass = st.selectbox(
            "Reclassify tier?",
            ["No change", "CRITICAL", "HIGH", "MEDIUM", "LOW", "Benign/Dismiss"],
            key=f"rc_{form_key}",
        )

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
    """Log confirm/reject, reclassifications, feedback with timestamps.

    Writes to three sinks:
      1. online_interactions.jsonl  — flat per-interaction log (buffered)
      2. AuditTrailWriter           — local hash-chained eval-app trail
      3. HardenedAuditLogger        — signed, reviewer-attributed entry
                                       in the Module 5 audit_log.jsonl

    M6-A4: sink 1 uses _online_writer (AuditTrailWriter buffer) instead of
    open+write+close on every button click — I/O is amortised over
    _FLUSH_AFTER=10 events.
    """
    record = {
        "timestamp": datetime.now().isoformat(),
        "participant_id": participant_id,
        "alert_id": alert_id,
        "action_type": action_type,
        "details": details or {},
    }
    # M6-A4: buffered write — no per-event file open
    _online_writer.log(action_type, **record)
    # Local eval-app audit trail. Day 6 enrichment: carry the operator's
    # current top-bar role into the audit record so the panel can show
    # who decided what without separately joining session state.
    audit_log(
        "online_interaction",
        role=st.session_state.get("role"),
        **record,
    )
    _mark_decision_submitted()
    # Signed Module 5 audit chain with reviewer attribution
    _hardened_audit.log(
        {"event_type": "reviewer_interaction", "alert_id": alert_id, "details": details or {}},
        reviewer_id=participant_id or st.session_state.get("participant_id") or "anon",
        reviewer_role=st.session_state.get("participant_role") or st.session_state.get("sim_role"),
        review_action=action_type,
    )


# ═══════════════════════════════════════════════════════════════════════
# 6A.3  process_alert() — end-to-end sample → structured alert object
# ═══════════════════════════════════════════════════════════════════════


def process_alert(sample_index: int, alert_data: dict) -> dict:
    """Takes a pre-computed JSON alert record and prepares it for UI components.

    Under Hướng B, this no longer calls src.risk_scorer.score_alert live.
    It simply maps the nested JSON properties (e.g. should_surface, adjusted_score)
    into the flattened flat-dict format that render_analyst / render_clinician expect.
    """
    xai = alert_data.get("xai_explanation", {})
    expl = alert_data.get("explanation", {})
    clinician_summary = xai.get("clinician_summary", "") or (
        expl.get("clinician_summary", "") if isinstance(expl, dict) else ""
    )

    # Derive recommended action
    resp = alert_data.get("response", {})
    action = alert_data.get("correct_action", "")
    if not action and isinstance(resp, dict):
        policy_actions = resp.get("actions", [])
        for pa in reversed(policy_actions):
            mapped = _ACTION_PRIORITY.get(pa)
            if mapped:
                action = mapped
                break
    if not action:
        level = alert_data.get("risk_level", "LOW")
        action = {
            "CRITICAL": "isolate",
            "HIGH": "escalate",
            "MEDIUM": "investigate",
            "LOW": "monitor",
        }.get(level, "monitor")

    return {
        "sample_index": sample_index,
        "prediction": 1 if alert_data.get("should_surface", True) else 0,
        "confidence": alert_data.get("adjusted_score", alert_data.get("risk_score", 0)),
        "risk_score": alert_data.get("adjusted_score", alert_data.get("risk_score", 0)),
        "raw_risk_score": alert_data.get("risk_score", 0),
        "threshold": alert_data.get("threshold", 0.5),
        "risk_multiplier": alert_data.get("risk_multiplier", 1.0),
        "suppression_reason": alert_data.get("suppression_reason", ""),
        "tier": alert_data.get("risk_level", "LOW"),
        "attack_category": alert_data.get("attack_category", "unknown"),
        "ground_truth": alert_data.get("ground_truth", "unknown"),
        "shap_top_features": xai.get("xgboost_top_features", []),
        "dae_top_features": xai.get("dae_top_features", []),
        "nlg_text": clinician_summary,
        "consensus": xai.get("consensus", "")
        or (expl.get("consensus", "") if isinstance(expl, dict) else ""),
        "response_action": action,
        "response": resp,
        # Device context
        "device_class": alert_data.get("device_class", ""),
        "device_criticality": alert_data.get("device_criticality", ""),
        "affected_system": alert_data.get("affected_system", ""),
        "patient_care_impact": alert_data.get("patient_care_impact", ""),
        "active_device": alert_data.get("active_device", False),
        # MVE layer content
        "clinician_summary": clinician_summary,
        "severity_label": alert_data.get("risk_level", ""),
        "group_a_display": alert_data.get("group_a_display", ""),
        "group_b_display": alert_data.get("group_b_display", ""),
        "mve_structured": alert_data.get("mve_structured", {}),
    }


# ═══════════════════════════════════════════════════════════════════════
# 6A.4  Stakeholder view renderers
# ═══════════════════════════════════════════════════════════════════════


def render_analyst(alert: dict):
    """Analyst view: SHAP plots + feature table + classification detail."""
    st.markdown("#### Security Analyst View")

    # Day 3: v4 visual treatment for cross-page consistency with Dashboard.
    display_alert_header_v4(alert)

    # Gap 2: Device criticality badge
    render_device_criticality(alert)

    # SHAP waterfall
    idx = alert.get("sample_index", 0)
    chart_bytes = _cached_png_bytes(str(CHARTS_DIR / f"waterfall_xgboost_sample_{idx:04d}.png"))
    if chart_bytes:
        st.image(chart_bytes, caption="SHAP Waterfall", width="stretch")

    # Force plot
    force_bytes = _cached_png_bytes(str(CHARTS_DIR / f"force_xgboost_sample_{idx:04d}.png"))
    if force_bytes:
        st.image(force_bytes, caption="SHAP Force Plot", width="stretch")

    # Top features table
    feats = alert.get("shap_top_features", [])
    if feats:
        st.markdown("**Top SHAP Features:**")
        rows = []
        for f in feats[:5]:
            rows.append(
                {
                    "Feature": f["feature"],
                    "SHAP Value": f"{f.get('shap_value', 0):+.4f}",
                    "Direction": f.get("direction", ""),
                    "Type": "Biometric" if f["feature"] in BIOMETRIC_FEATURES else "Network",
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

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

    # Day 3: v4 visual treatment for cross-page consistency with Dashboard.
    display_alert_header_v4(alert)

    # Gap 2: Device criticality badge
    render_device_criticality(alert)

    nlg = alert.get("nlg_text", "")
    if nlg:
        st.warning(nlg)
    else:
        st.info("No clinician summary available for this alert.")

    # Highlight biometric features
    bio_feats = [
        f["feature"]
        for f in alert.get("shap_top_features", [])
        if f["feature"] in BIOMETRIC_FEATURES
    ]
    if bio_feats:
        st.error(f"Patient safety note: Biometric features affected: {', '.join(bio_feats)}")
    else:
        st.success("Patient vitals are not among the primary alert indicators.")

    st.metric("Risk Score", f"{alert.get('risk_score', 0):.2f}")
    st.markdown(f"**Recommended action:** {alert.get('response_action', 'N/A')}")

    # Gap 1: DO NOT constraint
    action_text = alert.get("response_action", "") or alert.get("clinical_constraint", "")
    render_do_not_constraint(action_text, alert.get("tier", ""))

    # Gap 3: MVE layers (collapsed in clinician view)
    with st.expander("View full MVE explanation"):
        render_mve_layers(alert)


def render_admin(alert: dict):
    """Administrator view: summary statistics + risk breakdown."""
    st.markdown("#### Administrator View")

    # Day 3: v4 visual treatment for cross-page consistency with Dashboard.
    display_alert_header_v4(alert)

    # Gap 2: Device criticality badge
    render_device_criticality(alert)

    col1, col2, col3 = st.columns(3)
    col1.metric("Risk Score", f"{alert.get('risk_score', 0):.3f}")
    col2.metric("Tier", alert.get("tier", "N/A"))
    col3.metric("Category", alert.get("attack_category", "N/A"))

    st.markdown(f"**Consensus:** {alert.get('consensus', 'N/A')}")
    st.markdown(f"**Recommended Action:** {alert.get('response_action', 'N/A')}")

    # Global charts
    gc1, gc2 = st.columns(2)
    gi_bytes = _cached_png_bytes(str(CHARTS_DIR / "global_importance_xgboost.png"))
    bs_bytes = _cached_png_bytes(str(CHARTS_DIR / "beeswarm_xgboost.png"))
    if gi_bytes:
        gc1.image(gi_bytes, caption="Global Feature Importance", width="stretch")
    if bs_bytes:
        gc2.image(bs_bytes, caption="SHAP Beeswarm", width="stretch")


# ═══════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════


@st.cache_data
def load_alerts() -> list:
    path = EVAL_DIR / "evaluation_alerts.json"
    if not path.exists():
        st.error("Run `python module6_evaluation/module6_evaluation.py` first.")
        st.stop()
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_alerts_dict() -> dict:
    """Return evaluation alerts keyed by sample_index, enriched for UI renderers."""
    alerts = load_alerts()
    return {
        int(alert["sample_index"]): {
            **alert,
            **process_alert(int(alert["sample_index"]), alert),
        }
        for alert in alerts
        if "sample_index" in alert
    }


@st.cache_data
def load_all_responses() -> list:
    """Load alert_responses.json, enriched with device context from evaluation_alerts.json."""
    path = EVAL_DIR / "alert_responses.json"
    if not path.exists():
        return []
    with open(path) as f:
        responses = json.load(f)

    # M4: Join with evaluation_alerts.json for device context fields
    eval_path = EVAL_DIR / "evaluation_alerts.json"
    if eval_path.exists():
        with open(eval_path) as f:
            eval_alerts = {a["sample_index"]: a for a in json.load(f)}
        _ENRICH_KEYS = (
            "device_class",
            "device_criticality",
            "affected_system",
            "patient_care_impact",
            "active_device",
            "correct_action",
        )
        for r in responses:
            ea = eval_alerts.get(r.get("sample_index"))
            # Issue 5 fix: guard with `if ea` before iterating — avoids
            # creating a throwaway {} on every miss via .get(key, {}).
            if ea:
                for k in _ENRICH_KEYS:
                    if k in ea and k not in r:
                        r[k] = ea[k]

    return responses


@st.cache_data
def load_risk_scores():
    path = EVAL_DIR / "risk_scores.npz"
    if not path.exists():
        return None
    return dict(np.load(path, allow_pickle=True))


def _load_json_or(filename: str, default, transform=None):
    """Read ``EVAL_DIR/filename`` as JSON, returning *default* on absence.

    Single source for the "best-effort artefact load" pattern used by
    the Streamlit dashboard: the optional artefacts (admin dashboard,
    clinician summaries, response policy, latency profile, …) are
    produced by upstream pipeline runs and may not exist in every
    environment. Returning *default* keeps the dashboard renderable
    even when an artefact is missing.

    Args:
        filename: Basename inside ``EVAL_DIR`` (e.g. ``"admin_dashboard.json"``).
        default: Value returned when the file is absent.
        transform: Optional post-parse transformer applied to the JSON
            payload before returning (e.g. dict-by-key indexing).

    Returns:
        Parsed JSON (optionally transformed) or *default*.
    """
    path = EVAL_DIR / filename
    if not path.exists():
        return default
    with open(path) as f:
        data = json.load(f)
    return transform(data) if transform is not None else data


@st.cache_data
def load_admin_dashboard() -> dict:
    return _load_json_or("admin_dashboard.json", default={})


@st.cache_data
def load_clinician_summaries() -> dict:
    return _load_json_or(
        "clinician_summaries.json",
        default={},
        transform=lambda data: {s["sample_index"]: s for s in data},
    )


@st.cache_data
def load_response_policy() -> dict:
    return _load_json_or("response_policy.json", default={})


@st.cache_data(max_entries=64, show_spinner=False)
def _cached_png_bytes(path_str: str) -> bytes | None:
    """Cache PNG file bytes across reruns.

    `st.image(str)` re-reads and re-decodes the file on every script
    rerun. With several charts visible per simulation tick this is the
    single largest piece of waste in the page. Caching the raw bytes
    lets Streamlit reuse them across reruns; the decoded image is then
    streamed to the browser exactly once per session per file.
    """
    p = Path(path_str)
    if not p.exists():
        return None
    return p.read_bytes()


@st.cache_data
def load_audit_trail() -> dict:
    """Module 5 FDA-style audit records, keyed by sample index parsed from alert_id."""
    path = EVAL_DIR / "audit_trail.json"
    if not path.exists():
        return {}
    with open(path) as f:
        records = json.load(f)
    out: dict[int, dict] = {}
    for rec in records:
        aid = rec.get("alert_id", "")
        # alert_id format: "ALERT-00042"
        # Issue 7 fix: rsplit("-", 1) allocates a 2-element list instead of
        # splitting the full string into all "-"-delimited segments.
        try:
            idx = int(aid.rsplit("-", 1)[-1])
            out[idx] = rec
        except (ValueError, IndexError):
            continue
    return out


@st.cache_data
def _compute_tier_counts(responses_tuple: tuple) -> dict:
    """Pre-aggregate risk-level counts from the full response list.

    Issue 1 fix: Counter(r["risk_level"] for r in responses) ran on every
    Streamlit render (every widget click, sidebar toggle, expander open).
    Wrapping in @st.cache_data with a hashable tuple arg means the O(n)
    scan runs once per data load, not once per render.

    Args:
        responses_tuple: tuple of (sample_index, risk_level) pairs — the
            minimal hashable slice of responses needed for hashing.
    Returns:
        dict with keys CRITICAL / HIGH / MEDIUM / LOW → int count.
    """
    counts: dict[str, int] = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for _, level in responses_tuple:
        if level in counts:
            counts[level] += 1
    return counts


@st.cache_data
def _build_feed_dataframe(responses_head: tuple) -> pd.DataFrame:
    """Build the alert-feed DataFrame once per data load.

    Issue 6 fix: the list-of-dicts -> pd.DataFrame construction ran on
    every render inside dashboard_mode(). Pre-computing it with a hashable
    key (tuple of the first 15 records' key fields) means the O(15) dict
    construction + DataFrame ctor run once per cache key, not per render.

    Args:
        responses_head: tuple of (sample_index, risk_level, risk_score,
            device_class, attack_category, correct_action, ground_truth)
            for the first 15 responses - fully hashable for st.cache_data.
            ``ground_truth`` is required so the v4 type heuristic can
            distinguish benign rows from attacks.

    Returns:
        pd.DataFrame ready for st.dataframe().
    """
    rows = []
    for item in responses_head:
        sample, level, score, device, category, action, ground_truth = item
        synthetic = {
            "ground_truth": ground_truth,
            "attack_category": category,
            "risk_level": level,
            "risk_score": score,
        }
        atype, conf, mode = derive_v4_fields(synthetic)
        badge = badge_for_alert_type(atype)
        rows.append(
            {
                "Sample": sample,
                "Level": level,
                "Score": round(score, 3),
                "Type": f"{badge['icon']} {badge['label']}",
                "Confidence": conf.value,
                "Mode": "✓ AI" if mode == MODE_A_LLM else "⚠ Rule",
                "Device": device or "—",
                "Category": category or "",
                "Action": action or "—",
            }
        )
    return pd.DataFrame(rows)


@st.cache_data
def load_latency_profile() -> dict:
    """Module 4 online_latency_profile.json — aggregate per-stage latency stats."""
    return _load_json_or("online_latency_profile.json", default={})


@st.cache_data
def load_live_stream_source() -> pd.DataFrame | None:
    """Mock 'live data source' — reads the test parquet directly and attaches a
    synthetic arrival timestamp per row.

    This simulates a feature-extracted flow stream without requiring a real
    network TAP. Each row is one timestep of mock 'arrived data'; the
    timestamps are anchored to a fixed start instant so the stream is
    reproducible across reruns.
    """
    path = PROJECT_ROOT / "data/processed/test_phase1.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    # Synthetic arrival timestamps: 1 second between rows, anchored at the
    # session start. Keeping this deterministic per session is intentional
    # so timestamps stay stable when Streamlit reruns the script.
    base = datetime(2026, 4, 9, 8, 0, 0)
    df = df.reset_index(drop=True)
    # Issue 8 fix: pd.date_range() generates n timestamps in one C-level call
    # instead of n Python timedelta + datetime additions in a list comprehension.
    df["arrived_at"] = pd.date_range(start=base, periods=len(df), freq="1s").strftime(
        "%Y-%m-%dT%H:%M:%S"
    )
    df["sample_index"] = df.index
    return df


# ═══════════════════════════════════════════════════════════════════════
# 6C.11  Synthetic per-sample latency series
# ═══════════════════════════════════════════════════════════════════════


def _draw_latency_sample(stage_stats: dict, rng: random.Random) -> float:
    """Draw a single latency sample for one stage, consistent with the
    recorded mean / p50 / p95 of the offline latency profile.

    Uses a lognormal fit: μ = ln(p50), σ derived from p95 ≈ exp(μ + 1.645·σ).
    Falls back to a clamped normal if percentiles are missing.
    """
    p50 = stage_stats.get("p50") or stage_stats.get("mean") or 1.0
    p95 = stage_stats.get("p95") or (p50 * 1.5)
    if p50 <= 0:
        return max(0.0, stage_stats.get("mean", 1.0))
    try:
        mu = np.log(p50)
        sigma = max(1e-3, (np.log(p95) - mu) / 1.645)
        # Use the provided rng so the series is reproducible per session.
        sample = float(np.exp(rng.gauss(mu, sigma)))
        lo = stage_stats.get("min", 0.0)
        hi = stage_stats.get("max", sample * 4)
        return max(lo, min(hi, sample))
    except (ValueError, TypeError):
        return float(p50)


def push_latency_sample(profile: dict) -> dict | None:
    """Append one synthetic per-stage latency sample to the rolling deque
    held in session state. Returns the new sample, or None if the profile
    is empty.
    """
    if not profile:
        return None
    stages = profile.get("all_alerts", {})
    if not stages:
        return None

    rng = st.session_state.setdefault("_latency_rng", random.Random(42))
    history = st.session_state.setdefault("latency_history", deque(maxlen=120))

    sample = {stage: _draw_latency_sample(stats, rng) for stage, stats in stages.items()}
    sample["arrival_idx"] = len(history)
    history.append(sample)
    return sample


# ═══════════════════════════════════════════════════════════════════════
# Session state
# ═══════════════════════════════════════════════════════════════════════


def init_session():
    defaults = {
        "participant_id": "",
        "participant_role": "",
        "participant_years": 1,
        "participant_ids_exp": "No",
        "current_alert": 0,
        "responses": [],
        "alert_start_time": None,
        "study_started": False,
        "study_complete": False,
        "study_alerts": [],
        "sim_index": 0,
        "sim_running": True,
        "sim_history": [],
        "sim_speed": 1.0,  # 0.5x / 1x / 2x / 4x
        "sim_source": "alerts",  # "alerts" or "live_parquet"
        "latency_history": deque(maxlen=120),
        "alerts_dict": {},
        # Top-bar global state (persists across page changes)
        "role": "IT Generalist",        # one of: IT Generalist | Biomed | Nurse
        "demo_mode": False,             # curated-demo subset toggle
        "latest_mve_mode": None,        # last seen MVE generator mode (A_llm / B_rule_based)
        # Day 3 — Sim polish
        "researcher_mode": False,       # show research-only export buttons (Sim sidebar)
        "auto_paused_at_index": None,   # sim_index where the last safety-floor pause fired
        "safety_floor_banner": False,   # banner flag set by fragment, rendered outside it
        # Day 5 — Study mode demo bypass
        "study_demo_bypass_active": False,
        # Day 6 — Audit panel auto-expand on decision submit
        "audit_panel_just_submitted": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ═══════════════════════════════════════════════════════════════════════
# Top bar (global, sticky)
# ═══════════════════════════════════════════════════════════════════════

ROLES: tuple[str, ...] = ("IT Generalist", "Biomed", "Nurse")
DEMO_ALERT_LIMIT: int = 5

# Display labels with role-affordance icons (selectbox / log / etc.).
ROLE_DISPLAY_LABEL: dict[str, str] = {
    "IT Generalist": "🖥️ IT Generalist",
    "Biomed":        "⚕️ Biomed",
    "Nurse":         "👩‍⚕️ Nurse",
}

# Map the top-bar role labels to the existing per-role renderers.
# IT Generalist ≈ Security Analyst (network/IDS framing).
# Biomed       ≈ Administrator (device-fleet / biomed-engineering view).
# Nurse        ≈ Clinician (patient-care framing).
_ROLE_TO_LEGACY_VIEW: dict[str, str] = {
    "IT Generalist": "Security Analyst",
    "Biomed":        "Administrator",
    "Nurse":         "Clinician",
}


# ── Public accessors ──────────────────────────────────────────────────
# Wrap session-state reads/writes so callers don't depend on key names.
# When/if the top bar moves to ``components/top_bar.py``, these stay
# import-stable.

def get_current_role() -> str:
    """Return the currently selected top-bar role.

    Falls back to ``"IT Generalist"`` if state is uninitialised.
    """
    return st.session_state.get("role", ROLES[0])


def get_demo_mode() -> bool:
    """Return whether Demo Mode is on."""
    return bool(st.session_state.get("demo_mode", False))


def set_current_mode(generation_mode: str) -> None:
    """Update the top-bar Mode A/B indicator from a page renderer.

    Pages with per-alert MVE rendering call this so the global indicator
    reflects the most recently shown alert's mode.
    """
    if generation_mode in (MODE_A_LLM, MODE_B_RULE_BASED):
        st.session_state["latest_mve_mode"] = generation_mode


# Bridge between the top-bar's display-string roles and the project's
# canonical ``OperatorRole`` enum (used by the v4 MVE generator and
# the MITRE per-role formatter).
_TOPBAR_TO_OPERATOR_ROLE: dict[str, OperatorRole] = {
    "IT Generalist": OperatorRole.IT_GENERALIST,
    "Biomed":        OperatorRole.BIOMED_ENGINEER,
    "Nurse":         OperatorRole.NURSE_MANAGER,
}


def get_current_operator_role() -> OperatorRole:
    """Return the current top-bar role as an :class:`OperatorRole` enum.

    Used by code paths that talk to the v4 Layer 4 helpers (MITRE per
    role, role-lensed MVE templates). Defaults to ``IT_GENERALIST``.
    """
    return _TOPBAR_TO_OPERATOR_ROLE.get(get_current_role(), OperatorRole.IT_GENERALIST)


# ── Day 4: Demo playlist + synthetic alert loaders ───────────────────

PLAYLIST_PATH = Path(__file__).resolve().parent.parent / "configs" / "demo_playlist.yaml"
SYNTHETIC_ALERTS_PATH = (
    Path(__file__).resolve().parent.parent / "configs" / "synthetic_demo_alerts.yaml"
)


@st.cache_data(ttl=60)
def load_demo_playlist() -> dict:
    """Load ``configs/demo_playlist.yaml``.

    Returns the parsed playlist with ``alerts`` ordered by
    ``narrative_position``. Returns ``{"alerts": []}`` on missing /
    malformed file so the dashboard never crashes on a fresh checkout.
    """
    import yaml
    if not PLAYLIST_PATH.exists():
        return {"alerts": []}
    try:
        data = yaml.safe_load(PLAYLIST_PATH.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return {"alerts": []}
    alerts = sorted(
        data.get("alerts", []),
        key=lambda a: a.get("narrative_position", 999),
    )
    data["alerts"] = alerts
    return data


@st.cache_data(ttl=60)
def load_synthetic_demo_alerts() -> list:
    """Load ``configs/synthetic_demo_alerts.yaml`` (Demo Mode only).

    Each entry is decorated with ``is_synthetic_demo=True`` so callers can
    filter synthetic rows out of any non-demo path.
    """
    import yaml
    if not SYNTHETIC_ALERTS_PATH.exists():
        return []
    try:
        data = yaml.safe_load(SYNTHETIC_ALERTS_PATH.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return []
    out = []
    for a in data.get("synthetic_alerts", []) or []:
        a = dict(a)
        a["is_synthetic_demo"] = True
        out.append(a)
    return out


def _playlist_alert_ids() -> list:
    """Playlist alert_ids in narrative order."""
    return [a["alert_id"] for a in load_demo_playlist().get("alerts", []) if "alert_id" in a]


def _get_alerts_for_demo_mode() -> list:
    """Return the alert list the page should render right now.

    Demo Mode OFF → all 20 real alerts.
    Demo Mode ON  → playlist alerts in narrative order; missing IDs are
    surfaced via ``st.warning`` so the operator notices a drift between
    playlist and eval set.
    """
    real_alerts = load_alerts()
    if not st.session_state.get("demo_mode"):
        return real_alerts
    playlist_ids = _playlist_alert_ids()
    if not playlist_ids:
        return real_alerts
    real_by_id = {a.get("alert_id"): a for a in real_alerts}
    syn_by_id = {a.get("alert_id"): a for a in load_synthetic_demo_alerts()}
    out: list = []
    missing: list = []
    for aid in playlist_ids:
        if aid in real_by_id:
            out.append(real_by_id[aid])
        elif aid in syn_by_id:
            out.append(syn_by_id[aid])
        else:
            missing.append(aid)
    if missing:
        st.warning(
            "Demo playlist references unknown alert_id(s): " + ", ".join(missing)
        )
    return out


def _filter_responses_for_demo_mode(responses: list) -> list:
    """Filter a Sim-style ``responses`` list to playlist alerts.

    Synthetic alerts have no precomputed response artefact, so they're
    silently excluded — Sim demos at most ``len(playlist) - synthetics``
    alerts. Real-alert order matches the playlist.
    """
    if not st.session_state.get("demo_mode"):
        return responses
    playlist_ids = _playlist_alert_ids()
    if not playlist_ids:
        return responses
    by_id = {r.get("alert_id"): r for r in responses}
    return [by_id[aid] for aid in playlist_ids if aid in by_id]


def _show_demo_mode_indicator() -> None:
    """Render a uniform Demo-Mode banner at the top of demo-affected pages.

    Skipped on Study Mode (locked schema — Demo Mode does not change
    study_responses_*.json behaviour).
    """
    if st.session_state.get("demo_mode"):
        n = len(_playlist_alert_ids())
        st.info(
            f"\U0001f3ac **Demo Mode** — showing {n}-alert playlist "
            "(curated narrative beats). Toggle off in the top bar to see "
            "the full evaluation set."
        )


# ── Day 5: Study Mode demo bypass + Group B alert lookup ─────────────


def _study_alert_dict_for(alert_id: str) -> dict | None:
    """Look up the dashboard alert dict matching a study alert_id.

    Study Mode loads :class:`AlertScenario` (locked schema, only carries
    the Phase-2 stimulus text). The v4 visual helpers expect the dict
    shape from ``evaluation_alerts.json`` (``risk_level``,
    ``attack_category``, ``device_patchable``, ``mve_structured`` …).
    This helper bridges the two by ``alert_id`` so the Group B chrome
    can render without disturbing the AlertScenario contract.

    Returns ``None`` when no matching alert is found — callers then fall
    back to the legacy rendering (no v4 chrome) so Group B never crashes
    on a missing entry.
    """
    real = {a.get("alert_id"): a for a in load_alerts()}
    syn = {a.get("alert_id"): a for a in load_synthetic_demo_alerts()}
    return real.get(alert_id) or syn.get(alert_id)


def _render_demo_bypass_offer() -> None:
    """Render the Skip-Registration affordance on the Study registration page.

    Visible only when the top-bar Demo Mode toggle is ON. Click flips
    ``study_demo_bypass_active`` so ``study_mode`` reroutes to the
    bypass view on the next rerun.
    """
    if not st.session_state.get("demo_mode"):
        return
    st.warning(
        "\U0001f3ac **Demo Mode is on.** Skip the 30-min protocol to show "
        "A/B differentiation directly. **No data will be saved.**"
    )
    if st.button(
        "⏭ Skip Registration (Demo Only)",
        type="primary",
        key="study_skip_registration",
        help="Bypasses registration and shows ONE curated alert in both Group A and Group B styles.",
    ):
        st.session_state.study_demo_bypass_active = True
        st.rerun()


def _render_demo_bypass_view() -> None:
    """A/B comparison view for the defense demo.

    Renders one curated alert (the playlist's ``ab_comparison`` beat)
    in two tabs — Group A (control / raw IDS) and Group B (treatment /
    full v4 chrome). No Likert form is shown and no
    ``study_responses_*.json`` is written. Designed as the strongest
    single visual moment of the 10-min defense demo.
    """
    st.title("\U0001f4cb Study Mode — A/B Demo Bypass")
    st.error(
        "⚠ **DEMO ONLY** — registration bypassed; no study data is "
        "collected. Click **Exit Demo** to return to the full protocol."
    )
    if st.button("← Exit Demo", type="secondary", key="study_exit_demo"):
        st.session_state.study_demo_bypass_active = False
        st.rerun()

    st.markdown("---")

    # Pick the playlist's A/B beat (Beat 5). Fall back to first beat.
    playlist = load_demo_playlist().get("alerts", [])
    ab_entry = next(
        (e for e in playlist if e.get("narrative_beat") == "ab_comparison"),
        playlist[0] if playlist else None,
    )
    if ab_entry is None:
        st.error("Demo playlist is empty — check `configs/demo_playlist.yaml`.")
        return
    target_id = ab_entry["alert_id"]
    alert_dict = _study_alert_dict_for(target_id)
    if alert_dict is None:
        st.error(f"Demo A/B alert `{target_id}` not found in evaluation set.")
        return

    st.markdown(f"### Same alert: `{target_id}`")
    st.caption(
        "Group A = control (raw IDS).  Group B = treatment (with MVE). "
        "The text content of each is identical to what Phase-2 study "
        "participants saw; Group B simply gets the v4 visual chrome."
    )

    tab_a, tab_b = st.tabs(
        ["Group A — Raw IDS (control)", "Group B — With MVE (treatment)"]
    )
    with tab_a:
        st.caption("Operator sees this in the Group A condition.")
        group_a_text = alert_dict.get("group_a_display") or "(no Group A text)"
        st.code(group_a_text, language=None)

    with tab_b:
        st.caption("Operator sees this in the Group B condition.")
        display_alert_header_v4(alert_dict)
        try:
            atype, _, _ = derive_v4_fields(alert_dict)
            op_role = get_current_operator_role()
            mitre_line = format_mitre_for_alert_type(atype, op_role)
        except Exception:
            mitre_line = ""
        if mitre_line:
            st.markdown(f"**Threat intelligence:** {mitre_line}")
        st.markdown("---")
        group_b_text = alert_dict.get("group_b_display") or "(no Group B text)"
        _render_group_b_highlighted(group_b_text)

    st.markdown("---")
    st.markdown(
        "**Talking points**\n"
        "- Group A: minimal information — operator must reason alone.\n"
        "- Group B: 9-class badge, MITRE per role, 3-layer MVE with prominent DO_NOT.\n"
        "- Method-1 LLM simulation (M5): +60.8 % composite-accuracy improvement for IT generalist; "
        "Wilcoxon p < 1e-6, Cohen's h = 0.43 (medium-large)."
    )


# ── Day 6: Last-5-Decisions audit panel (INVARIANT 4 visualisation) ──
#
# The local audit log written by ``AuditTrailWriter`` carries a
# hash-chained record of every UI interaction. The panel surfaces only
# the records that represent operator *decisions* (not mechanical
# playback events) so the operator/examiner sees the HITL loop closing
# without scrolling past sim_pause / sim_jump noise.

_AUDIT_TRAIL_PATH = EVAL_DIR / "audit_trail.jsonl"

DECISION_EVENT_TYPES: frozenset[str] = frozenset(
    {"response_submit", "alert_response", "online_interaction"}
)

_ROLE_COMPACT: dict[str, str] = {
    "IT Generalist":   "🖥️ IT",
    "Biomed":          "⚕️ Bio",
    "Nurse":           "👩‍⚕️ RN",
    # Legacy display labels also seen in older audit records:
    "IT_generalist":   "🖥️ IT",
    "biomed_engineer": "⚕️ Bio",
    "nurse_manager":   "👩‍⚕️ RN",
}

_EVENT_LABEL: dict[str, str] = {
    "response_submit":    "Submit",
    "alert_response":     "Study",
    "online_interaction": "Sim",
}


def _mark_decision_submitted() -> None:
    """Flush the audit buffer and set the panel auto-expand flag.

    Called after every audit_log() decision-write so the on-disk file
    reflects the new entry by the time the panel reads it (the buffered
    AuditTrailWriter only writes after _FLUSH_AFTER=10 records). The
    flag tells ``render_last_5_decisions_panel`` to render expanded on
    the next rerun.
    """
    try:
        _audit_writer.flush()
        _online_writer.flush()
    except Exception:
        pass  # never break a submit over a flush hiccup
    st.session_state["audit_panel_just_submitted"] = True


def _iter_audit_records():
    """Yield records from ``audit_trail.jsonl`` (skips malformed lines).

    Defensive: returns nothing on missing file, JSON decode errors, or
    a non-existent EVAL_DIR. Never raises.
    """
    if not _AUDIT_TRAIL_PATH.exists():
        return
    try:
        text = _AUDIT_TRAIL_PATH.read_text(encoding="utf-8")
    except OSError:
        return
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


@st.cache_data(ttl=2)
def load_recent_decisions(n: int = 5) -> list[dict]:
    """Return the ``n`` most recent operator decisions from the audit
    trail, ordered most-recent-first.

    Filters by :data:`DECISION_EVENT_TYPES` so playback mechanics
    (``sim_pause`` / ``sim_jump`` / ``study_start`` / etc.) don't mask
    real decisions in the panel.
    """
    records = [
        r for r in _iter_audit_records()
        if r.get("event_type") in DECISION_EVENT_TYPES
    ]
    records.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    return records[:n]


@st.cache_data(ttl=2)
def _count_total_decisions() -> int:
    """Total decision-event count in the on-disk audit trail."""
    return sum(
        1 for r in _iter_audit_records()
        if r.get("event_type") in DECISION_EVENT_TYPES
    )


_CHAIN_SEED = "0" * 64


@st.cache_data(ttl=2)
def verify_audit_chain_integrity() -> bool:
    """Verify every chain *segment* in ``audit_trail.jsonl`` links cleanly.

    ``AuditTrailWriter`` instantiates with ``prev_hash = '0'*64`` at the
    start of each Streamlit session, so the on-disk file is the
    concatenation of one segment per session - each starting with the
    canonical 64-zero seed. A "broken chain" is therefore a record
    whose ``prev_hash`` is **neither** the seed (segment start)
    **nor** the previous record's ``integrity_hash`` (continuation).

    Returns ``True`` for an empty / missing log and for any file whose
    every segment links cleanly. Returns ``False`` on the first broken
    link or a record missing the hash fields.
    """
    expected_prev = None  # None = start-of-file or start-of-segment
    for r in _iter_audit_records():
        prev = r.get("prev_hash")
        ihash = r.get("integrity_hash")
        if not isinstance(prev, str) or not isinstance(ihash, str):
            return False
        if prev == _CHAIN_SEED:
            # Start of a new session segment - always valid here.
            pass
        elif expected_prev is None or prev != expected_prev:
            return False
        expected_prev = ihash
    return True


def _decision_summary(record: dict) -> dict:
    """Project a raw audit record onto the panel's compact display fields.

    Tolerates heterogeneous event-type schemas:
      * ``response_submit`` carries ``action`` (chosen action label).
      * ``alert_response`` carries ``condition`` (with_mve / without_mve)
        and a ``composite_score``.
      * ``online_interaction`` carries ``action_type`` (confirm / reject
        / note) and a ``details`` dict with ``tier`` / ``score``.
    """
    ts = record.get("timestamp") or ""
    try:
        time_disp = datetime.fromisoformat(ts.replace("Z", "+00:00")).strftime("%H:%M:%S")
    except Exception:
        time_disp = ts[-8:] if len(ts) >= 8 else "—"

    raw_aid = str(record.get("alert_id", "—"))
    # 10 chars keeps the ``EVAL-XXXX`` prefix intact while still cutting
    # off freeform IDs (uuid-style) that would blow up the column width.
    alert_disp = raw_aid if len(raw_aid) <= 10 else raw_aid[-10:]

    raw_role = record.get("role", "")
    role_disp = _ROLE_COMPACT.get(raw_role, raw_role or "—")
    role_disp = role_disp[:10]

    event_disp = _EVENT_LABEL.get(record.get("event_type", ""), record.get("event_type", "—"))

    # Action picks whichever field actually has content
    action = (
        record.get("action")
        or record.get("action_type")
        or record.get("condition")
        or "—"
    )
    action_disp = action if len(str(action)) <= 28 else str(action)[:25] + "…"

    conf = record.get("confidence")
    if isinstance(conf, (int, float)):
        conf_disp = str(int(conf))
    else:
        conf_disp = "—"

    return {
        "Time":   time_disp,
        "Alert":  alert_disp,
        "Role":   role_disp,
        "Event":  event_disp,
        "Action": action_disp,
        "Conf":   conf_disp,
        "_record": record,
    }


def render_last_5_decisions_panel() -> None:
    """Render the Last-5-Decisions panel at the bottom of a page.

    Default: collapsed. Auto-expands once after a decision submission
    via the ``audit_panel_just_submitted`` session-state flag (set by
    :func:`_mark_decision_submitted`). The flag is cleared the moment
    we render expanded so the next page navigation collapses again.
    """
    auto_expand = bool(st.session_state.get("audit_panel_just_submitted"))
    if auto_expand:
        st.session_state["audit_panel_just_submitted"] = False

    decisions = load_recent_decisions(5)
    label = f"\U0001f4cb Last 5 Decisions ({len(decisions)})"

    with st.expander(label, expanded=auto_expand):
        if not decisions:
            st.caption(
                "No operator decisions logged yet. Submit a Likert form "
                "(Sim or Study) or a Confirm/Reject in Sim to see the "
                "audit trail populate here."
            )
            return

        st.caption(
            "Append-only audit trail (INVARIANT 4) — most recent first. "
            "Mechanical events (sim playback, study registration) are "
            "filtered out so this view is decisions only."
        )

        rows = [_decision_summary(d) for d in decisions]
        df = pd.DataFrame(
            [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
        )
        st.dataframe(df, width="stretch", hide_index=True)

        with st.expander("\U0001f50d Full record details", expanded=False):
            for i, r in enumerate(rows, start=1):
                rec = r["_record"]
                st.markdown(f"**Decision {i}** — `{rec.get('event_type','?')}`")
                st.text(f"timestamp:       {rec.get('timestamp','—')}")
                st.text(f"alert_id:        {rec.get('alert_id','—')}")
                if rec.get("participant_id"):
                    st.text(f"participant_id:  {rec['participant_id']}")
                if rec.get("role"):
                    st.text(f"role:            {rec['role']}")
                if "details" in rec:
                    st.text(f"details:         {rec['details']}")
                ihash = rec.get("integrity_hash") or ""
                if ihash:
                    st.text(f"integrity_hash:  {ihash[:16]}…")
                st.markdown("---")

        col_count, col_chain = st.columns([3, 1])
        with col_count:
            st.caption(
                f"Total decisions in audit trail: {_count_total_decisions()} "
                "(append-only, hash-chained)."
            )
        with col_chain:
            if verify_audit_chain_integrity():
                st.success("✓ Chain valid")
            else:
                st.warning("⚠ Chain check failed")


@st.cache_data(ttl=60)
def _aggregate_study_decision_quality() -> dict:
    """Aggregate ``study_responses_*.json`` into a Dashboard metric.

    The legacy ``all_responses.json`` stream the dashboard's feed table
    runs on does not record operator confidence or follow-rate. The
    Phase 2 user study does, in ``results/reports/study_responses_*.json``
    (one file per participant). This helper aggregates *real* decisions
    across whichever participant files are present so the Dashboard's
    "Decision Quality" tile shows real-data evidence of the protocol
    rather than a synthetic placeholder.

    Returns:
        ``{"n": int, "avg_confidence": float | None,
            "followed_pct": float | None}``. ``n == 0`` means no study
        files have been written yet (e.g. fresh checkout); the caller
        should render a "no study data yet" tile in that case.
    """
    n = 0
    confidences: list[float] = []
    followed = 0
    for path in EVAL_DIR.glob("study_responses_*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, list):
            continue
        for r in data:
            if not isinstance(r, dict):
                continue
            n += 1
            c = r.get("confidence")
            if isinstance(c, (int, float)):
                confidences.append(float(c))
            # Heuristic for "followed recommendation": chosen_action equals
            # correct_action. The study schema does not record an explicit
            # follow-rate field; this is the closest defensible proxy.
            if (
                r.get("chosen_action")
                and r.get("chosen_action") == r.get("correct_action")
            ):
                followed += 1
    avg_confidence = (sum(confidences) / len(confidences)) if confidences else None
    followed_pct = (100.0 * followed / n) if n else None
    return {"n": n, "avg_confidence": avg_confidence, "followed_pct": followed_pct}


def _inject_top_bar_css() -> None:
    """Inject sticky-bar CSS once per page load."""
    if st.session_state.get("_topbar_css_injected"):
        return
    st.markdown(
        """
        <style>
        div[data-testid="stAppViewBlockContainer"] > div:first-child > div[data-testid="stHorizontalBlock"]:first-of-type {
            position: sticky;
            top: 0;
            z-index: 999;
            background: #F8FAFC;
            border-bottom: 1px solid #E5E7EB;
            padding: 12px 24px;
            margin-bottom: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.session_state["_topbar_css_injected"] = True


def _render_top_bar() -> None:
    """Render the global top bar (title / role / mode / demo toggle).

    Layout:  [3 : 2 : 2 : 1] columns
    - Title:        "🔒 IoMT IDS"
    - Role:         persistent selectbox (IT Generalist / Biomed / Nurse)
    - Mode status:  info-only badge from last seen MVE generator mode
    - Demo toggle:  st.toggle bound to ``st.session_state.demo_mode``
    """
    _inject_top_bar_css()
    title_col, role_col, mode_col, demo_col = st.columns([3, 2, 2, 1])

    with title_col:
        st.markdown("### 🔒 IoMT IDS")

    with role_col:
        st.selectbox(
            "Role",
            options=ROLES,
            index=ROLES.index(get_current_role()),
            key="role",
            label_visibility="collapsed",
            format_func=lambda r: ROLE_DISPLAY_LABEL.get(str(r), str(r)),
        )

    with mode_col:
        latest = st.session_state.get("latest_mve_mode")
        if latest == "A_llm":
            st.markdown(
                "<div style='padding-top:6px;color:#15803D;'>Mode: ✓ AI Mode (LLM)</div>",
                unsafe_allow_html=True,
            )
        elif latest == "B_rule_based":
            st.markdown(
                "<div style='padding-top:6px;color:#B45309;'>Mode: ⚠ Rule-based</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='padding-top:6px;color:#6B7280;'>Mode: —</div>",
                unsafe_allow_html=True,
            )

    with demo_col:
        st.toggle(
            "Demo Mode",
            key="demo_mode",
            help=(
                "Curated subset for the defense demo: Dashboard shows the "
                "first 5 alerts; Study auto-starts as participant DEMO; "
                "Browse and Sim show a Demo badge."
            ),
        )


# ═══════════════════════════════════════════════════════════════════════
# 6C.12  FDA-style audit record export
# ═══════════════════════════════════════════════════════════════════════


def build_fda_record_for_alert(
    sample_idx: int,
    alert: dict,
    audit_trail: dict[int, dict],
) -> dict:
    """Return the FDA-style audit record for a sample.

    Prefers the canonical Module 5 record from `audit_trail.json` when
    available; otherwise constructs an equivalent record from whatever
    fields are present on the alert object so the export still works
    when Module 5 hasn't been run.
    """
    canonical = audit_trail.get(sample_idx)
    if canonical is not None:
        return canonical

    # Fallback: rebuild a Module-5-shaped record from what we have on hand.
    response = alert.get("response", {}) or {}
    explanation = (
        alert.get("explanation", {}).get("analyst", {}).get("consensus", "")
        if isinstance(alert.get("explanation"), dict)
        else ""
    )
    payload = json.dumps(
        {
            "idx": sample_idx,
            "risk_score": alert.get("risk_score"),
            "risk_level": alert.get("risk_level"),
            "actions": response.get("actions", []),
        },
        sort_keys=True,
    )
    integrity_hash = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return {
        "alert_id": f"ALERT-{sample_idx:05d}",
        "timestamp": datetime.now().isoformat(),
        "device_tier": response.get("device_tier", "unknown"),
        "attack_category": alert.get("attack_category", "unknown"),
        "risk_score": round(float(alert.get("risk_score", 0.0)), 4),
        "risk_level": alert.get("risk_level", "LOW"),
        "recommended_actions": response.get("actions", []),
        "action_rationale": response.get("rationale", ""),
        "escalation_chain": response.get("escalation_chain", {}),
        "explanation_summary": explanation[:200],
        "simulated_outcome": {
            "outcome": "n/a — synthesized at export time",
            "action_effective": None,
            "time_to_effectiveness_sec": None,
            "ground_truth": alert.get("ground_truth", "unknown"),
        },
        "integrity_hash": integrity_hash,
        "_source": "fallback (audit_trail.json not found)",
    }


# ═══════════════════════════════════════════════════════════════════════
# 6.3c  Dashboard Components
# ═══════════════════════════════════════════════════════════════════════


def dashboard_mode():
    """Full dashboard with risk gauge, alert feed, SHAP, NLG, responses, heatmap.

    Day 2: 3-column metric strip (was 5), DAE expander wired below SHAP,
    MITRE per role in MVE Layer 1, auto-refresh toggle removed (data is
    static), alert selection via table row-click with selectbox fallback.
    """
    st.title("\U0001f512 IoMT IDS \u2014 Real-Time Dashboard")
    _show_demo_mode_indicator()

    # Day 2: removed auto-refresh toggle. The dashboard runs on static
    # pre-computed JSON; auto-refresh just re-rendered the same data and
    # implied liveness that does not exist. Remove the misleading affordance.

    responses = load_all_responses()
    admin = load_admin_dashboard()
    clin_summaries = load_clinician_summaries()
    risk_data = load_risk_scores()
    policy = load_response_policy()

    if not responses:
        st.warning("No alert data found. Run Modules 3-5 first.")
        return

    # Day 4: filter responses to the demo playlist when Demo Mode is on.
    # (Day 1 stub used a "first 5" slice; Day 4 swaps in the curated set.)
    responses = _filter_responses_for_demo_mode(responses)
    if not responses:
        st.warning(
            "Demo Mode is on but the playlist resolved to zero responses. "
            "Toggle Demo Mode off or check `configs/demo_playlist.yaml`."
        )
        return

    # Best-effort: surface the latest alert's MVE generator mode in the
    # top-bar indicator. The field is not written by any pipeline stage
    # today; this is a forward-compatible hook so the indicator goes
    # live as soon as ``mve_generator`` plumbs ``generation_mode`` into
    # ``alert_responses.json`` / ``evaluation_alerts.json``.
    if responses:
        last = responses[-1]
        mode = (
            last.get("mve_mode")
            or last.get("generation_mode")
            or last.get("mve_structured", {}).get("generation_mode")
        )
        if mode:
            st.session_state["latest_mve_mode"] = mode

    # ── Row 1: Summary metrics (3-column strip — projector-readable) ──
    # Day 2: was 5 columns × ~350 px each on 1920×1080 — too dense at 2 m+
    # viewing distance. Now 3 columns × ~600 px focused on the operator's
    # actual decision triad: how bad is it, how much is queued, and how
    # well are decisions tracking the recommendation.
    st.markdown("### System Overview")

    # Issue 1 fix: O(n) Counter pre-computed once per data load via cache.
    _resp_key = tuple((r.get("sample_index"), r.get("risk_level")) for r in responses)
    tier_counts = _compute_tier_counts(_resp_key)
    total = len(responses)
    critical_count = tier_counts.get("CRITICAL", 0)
    high_count = tier_counts.get("HIGH", 0)
    medium_count = tier_counts.get("MEDIUM", 0)
    low_count = tier_counts.get("LOW", 0)
    pending_review = critical_count + high_count
    quality = _aggregate_study_decision_quality()

    col_critical, col_pending, col_quality = st.columns(3)
    with col_critical:
        st.metric(
            label="🔴 CRITICAL Alerts",
            value=critical_count,
            delta=f"of {total} total",
            delta_color="off",
            help="Highest-severity alerts requiring immediate attention.",
        )
    with col_pending:
        st.metric(
            label="📋 Pending Review",
            value=pending_review,
            delta=f"H:{high_count}  M:{medium_count}  L:{low_count}",
            delta_color="off",
            help="CRITICAL+HIGH awaiting triage; deltas show the rest of the queue.",
        )
    with col_quality:
        if quality["n"] and quality["avg_confidence"] is not None:
            st.metric(
                label="📊 Decision Quality",
                value=f"{quality['avg_confidence']:.1f}/5",
                delta=f"{quality['followed_pct']:.0f}% followed rec.  (n={quality['n']})",
                delta_color="off",
                help="Average operator confidence + recommendation-follow rate, "
                     "aggregated across study_responses_*.json.",
            )
        else:
            st.metric(
                label="📊 Decision Quality",
                value="N/A",
                delta="No study data yet",
                delta_color="off",
                help="Run the Study (A/B) page to capture participant decisions.",
            )

    # ── Row 2: Alert distribution (collapsed to reduce cognitive load) ──
    with st.expander("\U0001f4ca Alert Distribution", expanded=False):
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

    # ── Row 3: Alert feed (row-click drill-down) ──
    st.markdown("---")
    st.markdown("#### Alert Feed — click a row to inspect")

    # Issue 6 fix: DataFrame built once per data load via cache;
    # not rebuilt on every render triggered by widget interactions.
    _feed_key = tuple(
        (
            r.get("sample_index"),
            r.get("risk_level"),
            r.get("risk_score", 0.0),
            r.get("device_class"),
            r.get("attack_category"),
            r.get("correct_action"),
            r.get("ground_truth", ""),
        )
        for r in responses[:15]
    )
    feed_df = _build_feed_dataframe(_feed_key)

    # Day 2: try row-click drill-down (Streamlit ≥1.35). On older versions
    # or if any error surfaces, fall back to the legacy selectbox so the
    # demo never blocks on a UX upgrade.
    alert_idx = 0
    try:
        event = st.dataframe(
            feed_df,
            width="stretch",
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="dash_alert_table",
        )
        rows = getattr(getattr(event, "selection", None), "rows", None) or []
        if rows:
            alert_idx = rows[0]
    except (TypeError, AttributeError) as exc:
        st.warning(
            f"Row-click selection unavailable ({type(exc).__name__}); "
            "using selectbox fallback."
        )
        st.dataframe(feed_df, width="stretch", hide_index=True)
        alert_idx = st.selectbox(
            "Select alert",
            range(min(20, len(responses))),
            format_func=lambda i: f"#{responses[i]['sample_index']} ({responses[i]['risk_level']})",
            key="dash_alert_idx_fallback",
        )

    selected = responses[alert_idx]
    score = selected["risk_score"]
    level = selected["risk_level"]
    display_alert_header_v4(selected)

    # ── Row 4: Risk gauge ──
    col_gauge, _col_pad = st.columns([1, 2])
    with col_gauge:
        st.markdown("#### Risk Score Gauge")
        # Gap 2: Device criticality badge
        render_device_criticality(selected)
        # Gauge visualization using progress bar + color
        st.metric("Risk Score", f"{score:.3f}", delta=level)
        st.progress(min(score, 1.0))
        # Component breakdown
        comps = selected.get("risk_components", {})
        if comps:
            st.markdown("**Components:**")
            for k, v in comps.items():
                st.text(f"  {k}: {v:.4f}")

    # ── Row 4: SHAP waterfall + NLG clinician alert ──
    st.markdown("---")
    col_shap, col_nlg = st.columns(2)

    with col_shap:
        st.markdown("#### SHAP Waterfall Plot")
        sample_idx = selected["sample_index"]
        wf_bytes = _cached_png_bytes(
            str(CHARTS_DIR / f"waterfall_xgboost_sample_{sample_idx:04d}.png")
        )
        if wf_bytes:
            st.image(wf_bytes, width="stretch")
        else:
            force_bytes = _cached_png_bytes(
                str(CHARTS_DIR / f"force_xgboost_sample_{sample_idx:04d}.png")
            )
            if force_bytes:
                st.image(force_bytes, width="stretch")
            else:
                st.info(f"No SHAP chart for sample {sample_idx}")

        # Day 2: DAE per-dim attribution below the SHAP waterfall.
        # Hidden entirely when the alert carries no DAE features.
        render_dae_top_features(
            (selected.get("xai_explanation") or {}).get("dae_top_features", [])
        )

    with col_nlg:
        st.markdown("#### Clinician Alert")
        clin = clin_summaries.get(sample_idx, {})
        if clin:
            severity = clin.get("severity", "LOW")
            color = TIER_COLORS.get(severity, "#999")
            st.markdown(
                f"**Severity:** <span style='color:{color}'>{severity}</span>",
                unsafe_allow_html=True,
            )
            st.warning(clin.get("summary", "No summary available"))
        else:
            st.info("No clinician summary for this sample")

        # Gap 3: MVE layer rendering for the selected alert
        render_mve_layers(selected)

    # ── Row 5: Response recommendation panel ──
    st.markdown("---")
    st.markdown("#### Response Recommendation")
    resp = selected.get("response", {})
    if resp:
        rc1, rc2 = st.columns([2, 1])
        with rc1:
            render_prioritized_actions(resp.get("actions", []))
        with rc2:
            st.metric("Max Response", f"{resp.get('max_response_min', 'N/A')} min")
            st.metric("Priority", resp.get("priority", "N/A"))

        rationale = resp.get("rationale", "")
        if rationale:
            st.caption(f"Rationale: {rationale[:200]}")

        escalation = resp.get("escalation_chain", {})
        if escalation and escalation.get("primary"):
            st.markdown(
                f"**Escalation:** {escalation['primary']}"
                f"{' → ' + escalation['secondary'] if escalation.get('secondary') else ''}"
            )

        # Gap 1: DO NOT constraint from response policy
        constraint = resp.get("clinical_constraint", "") or resp.get("rationale", "")
        render_do_not_constraint(constraint, level)

    # ── Row 6: Global SHAP (collapsed to reduce cognitive load) ──
    st.markdown("---")
    with st.expander("\U0001f52c Global Feature Importance", expanded=False):
        global_bytes = _cached_png_bytes(str(CHARTS_DIR / "global_importance_xgboost.png"))
        beeswarm_bytes = _cached_png_bytes(str(CHARTS_DIR / "beeswarm_xgboost.png"))
        gc1, gc2 = st.columns(2)
        with gc1:
            if global_bytes:
                st.image(global_bytes, width="stretch")
        with gc2:
            if beeswarm_bytes:
                st.image(beeswarm_bytes, width="stretch")


# ═══════════════════════════════════════════════════════════════════════
# 6.3b  Online Simulation Mode
# ═══════════════════════════════════════════════════════════════════════
    # Day 6: Last 5 Decisions audit panel — INVARIANT 4.
    st.markdown("---")
    render_last_5_decisions_panel()



def simulation_mode():
    """Stream test samples through pipeline with smooth playback controls,
    a mock live data source, a real-time latency profile panel, and
    per-alert FDA-style audit record export.

    Includes 6C.1 streaming simulator, 6C.3 risk gauge, 6C.8 role switcher,
    6C.9 interaction capture, 6C.10 dynamic threshold display,
    6C.11 latency profile panel, 6C.12 FDA-record export.
    """
    # Step 0 instrumentation — measure end-to-end render time per script
    # rerun. Persisted to a JSONL file for offline analysis. Disabled by
    # default; toggle from the sidebar.
    _render_t0 = time.perf_counter()

    st.title("\U0001f4e1 IoMT IDS \u2014 Online Simulation")
    _show_demo_mode_indicator()

    responses = load_all_responses()
    clin_summaries = load_clinician_summaries()
    audit_trail = load_audit_trail()
    latency_profile = load_latency_profile()
    live_df = load_live_stream_source()
    if not st.session_state.alerts_dict:
        st.session_state.alerts_dict = load_alerts_dict()

    if not responses:
        st.warning("No alert data. Run Modules 3-5 first.")
        return

    # Day 4: filter playback sequence to the demo playlist when Demo Mode
    # is on. Synthetic alerts have no precomputed response artefact, so
    # they are silently skipped — the synthetic adversarial belongs in
    # Browse (deep-dive view), not Sim (streaming view).
    if st.session_state.get("demo_mode"):
        filtered = _filter_responses_for_demo_mode(responses)
        if filtered:
            responses = filtered
            # Reset playhead if it points past the new (smaller) end.
            if st.session_state.sim_index >= len(responses):
                st.session_state.sim_index = 0

    # 6C.8 Stakeholder view — driven by the global top-bar role.
    sim_role = _ROLE_TO_LEGACY_VIEW.get(get_current_role(), "Security Analyst")
    # Mirror the legacy key so any downstream readers (audit log, etc.)
    # that look at ``st.session_state.sim_role`` still see a value.
    st.session_state["sim_role"] = sim_role

    # ── Step 0 instrumentation toggle ──
    st.sidebar.markdown("---")
    st.sidebar.markdown("## Debug")
    st.session_state["_render_caption_enabled"] = st.sidebar.toggle(
        "Show render time", value=True, key="dbg_render_caption"
    )
    st.session_state["_render_log_enabled"] = st.sidebar.toggle(
        "Log render time to /tmp/sim_render_timings.jsonl",
        value=False,
        key="dbg_render_log",
    )

    # ── Day 3: Researcher Mode (toggles research-only export buttons) ──
    st.sidebar.markdown("---")
    st.sidebar.markdown("## Researcher Mode")
    st.sidebar.toggle(
        "Show researcher tools",
        key="researcher_mode",
        help=(
            "When enabled, surfaces research-only export controls "
            "(FDA-style audit record download per alert). Default is "
            "OFF for an examiner-friendly demo view."
        ),
    )

    # ── Data source toggle (6C.11 mock live source) ──
    st.sidebar.markdown("---")
    st.sidebar.markdown("## Data Source")
    source_label = st.sidebar.radio(
        "Stream from:",
        ["Pre-computed alerts (Module 5)", "Live parquet (mock TAP)"],
        index=0 if st.session_state.sim_source == "alerts" else 1,
        help=(
            "Pre-computed alerts replays Module 5 outputs.\n\n"
            "Live parquet reads data/processed/test_phase1.parquet row by row "
            "and attaches synthetic arrival timestamps, simulating a feature-"
            "extracted flow stream from a network TAP. Alert metadata is "
            "joined from Module 5 by sample index where available."
        ),
    )
    st.session_state.sim_source = "live_parquet" if "Live" in source_label else "alerts"
    using_live = st.session_state.sim_source == "live_parquet"

    if using_live and live_df is None:
        st.sidebar.warning(
            "data/processed/test_phase1.parquet not found — falling back to pre-computed alerts."
        )
        using_live = False
        st.session_state.sim_source = "alerts"

    # ── Day 3: Auto-pause on safety-floor alerts (INVARIANT 2) ──
    # Checked on every rerun (including after each fragment tick) so the
    # operator never blows past a CRITICAL+unpatchable alert without
    # explicit acknowledgement. Latched on sim_index to avoid pause-thrash:
    # after the user resumes, the next index either is non-safety-floor
    # (no pause) or a fresh safety-floor case (pause again — correct).
    _cur_idx = st.session_state.sim_index
    if (
        st.session_state.sim_running
        and 0 <= _cur_idx < len(responses)
        and st.session_state.auto_paused_at_index != _cur_idx
        and _is_safety_floor_alert(responses[_cur_idx])
    ):
        st.session_state.sim_running = False
        st.session_state.auto_paused_at_index = _cur_idx
        st.session_state.safety_floor_banner = True
        audit_log("sim_auto_pause_safety_floor", sim_index=_cur_idx)

    if st.session_state.safety_floor_banner:
        st.warning(
            "⚠ **Safety Floor Invoked** — Auto-paused on a "
            "CRITICAL + unpatchable device alert. "
            "Review the recommended action / DO NOT constraint, "
            "then click **▶ Resume** to continue."
        )

    # ── Smoother playback controls ──
    ctrl_a, ctrl_b, ctrl_c, ctrl_d, ctrl_e = st.columns([1.2, 1, 1, 1, 1.4])

    with ctrl_a:
        speed_label = st.selectbox(
            "Speed",
            ["0.5x", "1x", "2x", "4x"],
            index=(
                ["0.5x", "1x", "2x", "4x"].index(f"{st.session_state.sim_speed:g}x")
                if f"{st.session_state.sim_speed:g}x" in ["0.5x", "1x", "2x", "4x"]
                else 1
            ),
            help="Playback speed multiplier for the auto-advance loop.",
        )
        st.session_state.sim_speed = float(speed_label.rstrip("x"))

    with ctrl_b:
        if st.session_state.sim_running:
            if st.button("⏸ Pause", width="stretch"):
                st.session_state.sim_running = False
                audit_log("sim_pause", sim_index=st.session_state.sim_index)
        else:
            # Day 3: highlight Resume when the pause was an automatic
            # safety-floor pause so examiners notice the explicit ack.
            resume_label = (
                "▶ Resume (Safety Floor)"
                if st.session_state.safety_floor_banner
                else "▶ Resume"
            )
            resume_kind = "primary" if st.session_state.safety_floor_banner else "secondary"
            if st.button(resume_label, width="stretch", type=resume_kind):
                st.session_state.sim_running = True
                st.session_state.safety_floor_banner = False
                audit_log("sim_resume", sim_index=st.session_state.sim_index)

    with ctrl_c:
        if st.button("⏭ Step", width="stretch", help="Advance one alert (works while paused)."):
            st.session_state.sim_index = min(st.session_state.sim_index + 1, len(responses) - 1)
            push_latency_sample(latency_profile)

    with ctrl_d:
        if st.button("⟲ Reset", width="stretch"):
            st.session_state.sim_index = 0
            st.session_state.latency_history.clear()
            # Day 3: reset clears safety-floor latch so the pause can fire
            # again on the same data on a fresh playback.
            st.session_state.auto_paused_at_index = None
            st.session_state.safety_floor_banner = False
            audit_log("sim_reset")

    with ctrl_e:
        jump_target = st.number_input(
            "Jump to alert #",
            min_value=0,
            max_value=max(0, len(responses) - 1),
            value=int(st.session_state.sim_index),
            step=1,
            help="Jump the playhead to a specific alert index.",
        )
        if jump_target != st.session_state.sim_index:
            st.session_state.sim_index = int(jump_target)
            audit_log("sim_jump", target=int(jump_target))

    # ─────────────────────────────────────────────────────────────────
    # Phase 2 — STATIC ANALYTICS (rendered once per script run)
    # ─────────────────────────────────────────────────────────────────
    # These panels do not depend on the playhead. They are derived from
    # cached JSON files and are placed BEFORE the playhead fragment so
    # they re-render only when something forces a full script rerun
    # (sidebar change, control click, page reload). The fragment below
    # then handles tick-driven updates without re-running these panels.

    # ── System Diagnostics (Day 3: 3 tabs in one collapsed expander) ──
    # Latency / threshold / drift used to be a stacked group inside one
    # expander; the three are now split across explicit tabs so an
    # examiner doesn't have to scroll through a 600 px column to find the
    # one they want, and the Sim alert flow above stays visually clean.
    st.markdown("---")
    with st.expander("\U0001f527 System Diagnostics", expanded=False):
        st.caption(
            "Performance, threshold, and drift telemetry. Default hidden — "
            "for researcher inspection."
        )
        diag_tab_lat, diag_tab_thr, diag_tab_drift = st.tabs(
            ["\U0001f4ca Latency", "\U0001f3af Thresholds", "\U0001f4c9 Drift"]
        )

        with diag_tab_lat:
            st.markdown("#### Real-Time Latency Profile")
            if not latency_profile:
                st.info(
                    "No `online_latency_profile.json` found. Run "
                    "`python -m module4_explanations.module4_online_explainer` "
                    "to generate it."
                )
            else:
                all_stages = latency_profile.get("all_alerts", {})
                startup_ms = latency_profile.get("startup_ms")
                n_total = latency_profile.get("n_alerts_total", 0)

                lc1, lc2, lc3 = st.columns(3)
                lc1.metric("Profile samples", n_total)
                if startup_ms is not None:
                    lc2.metric("Startup", f"{startup_ms:.0f} ms")
                if "total_ms" in all_stages:
                    lc3.metric(
                        "End-to-end p95",
                        f"{all_stages['total_ms'].get('p95', 0):.0f} ms",
                    )

                lat_left, lat_right = st.columns(2)

                with lat_left:
                    st.markdown("**Per-stage mean latency (ms)**")
                    stage_rows = [
                        {"stage": s, "mean_ms": float(stats.get("mean", 0.0))}
                        for s, stats in all_stages.items()
                        if s != "total_ms"
                    ]
                    if stage_rows:
                        stage_df = pd.DataFrame(stage_rows).set_index("stage")
                        st.bar_chart(stage_df, color="#3274A1")

                with lat_right:
                    st.markdown("**Percentiles per stage**")
                    pct_rows = [
                        {
                            "stage": s,
                            "p50": round(stats.get("p50", 0.0), 2),
                            "p95": round(stats.get("p95", 0.0), 2),
                            "p99": round(stats.get("p99", 0.0), 2),
                            "max": round(stats.get("max", 0.0), 2),
                        }
                        for s, stats in all_stages.items()
                    ]
                    if pct_rows:
                        st.dataframe(
                            pd.DataFrame(pct_rows),
                            hide_index=True,
                            width="stretch",
                        )

        with diag_tab_thr:
            st.markdown("#### Adaptive Threshold Monitor")
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
                thresh_bytes = _cached_png_bytes(str(CHARTS_DIR / "threshold_over_time.png"))
                if thresh_bytes:
                    st.image(
                        thresh_bytes, width="stretch", caption="DAE threshold: static vs adaptive"
                    )
            else:
                st.info("Run `dynamic_threshold_sim.py` to enable adaptive threshold monitoring")

        with diag_tab_drift:
            st.markdown("#### Drift Detection Status")
            drift_path = EVAL_DIR / "drift_detection_results.json"
            if drift_path.exists():
                with open(drift_path) as f:
                    drift = json.load(f)
                psi = drift.get("psi_summary", {})
                n_events = len(drift.get("drift_events", []))
                dc1, dc2 = st.columns(2)
                dc1.metric("Drift Events", n_events)
                dc2.metric(
                    "PSI (max)",
                    f"{psi.get('max', 0):.4f}",
                    delta="DRIFT" if psi.get("max", 0) > 0.1 else "OK",
                    delta_color="inverse" if psi.get("max", 0) > 0.1 else "normal",
                )
                psi_bytes = _cached_png_bytes(str(CHARTS_DIR / "drift_psi.png"))
                if psi_bytes:
                    st.image(psi_bytes, width="stretch", caption="PSI over time")
            else:
                st.info("Run `drift_detection.py` to enable drift monitoring")

    # ─────────────────────────────────────────────────────────────────
    # Phase 2 — PLAYHEAD FRAGMENT (auto-ticks at speed-derived interval)
    # ─────────────────────────────────────────────────────────────────
    # Speed multiplier maps to fragment tick interval:
    #   0.5x → 4.0s, 1x → 2.0s, 2x → 1.0s, 4x → 0.5s
    # When sim_running is False, run_every is None — the fragment
    # renders once but does not auto-tick. Pause/Resume/Step/Reset/Jump
    # buttons live OUTSIDE the fragment, so a click triggers a full
    # script rerun which re-defines and re-calls the fragment with
    # fresh closure values.
    interval_s = 2.0 / max(0.25, st.session_state.sim_speed)
    fragment_interval = interval_s if st.session_state.sim_running else None

    @st.fragment(run_every=fragment_interval)
    def _playhead():
        # Auto-advance: only when running and not at the end. Step/Reset/
        # Jump live in the control row above and mutate sim_index there.
        if st.session_state.sim_running and st.session_state.sim_index < len(responses) - 1:
            st.session_state.sim_index = min(st.session_state.sim_index + 1, len(responses) - 1)
            push_latency_sample(latency_profile)

        idx_local = st.session_state.sim_index
        # Issue 3 fix: avoid O(n) history_local slice — use direct index
        # access throughout. history_local is only needed for the tier
        # distribution chart which uses the incremental _tier_history state.
        # current_batch_local is a bounded O(3) slice — kept as-is.
        window_size = 3
        current_batch_local = responses[max(0, idx_local - window_size + 1) : idx_local + 1]

        # ── Issues 2 & 3: incremental accumulators ──────────────────────
        # Replace O(n) Counter + sum(1 for ...) on growing history_local
        # with O(1) session-state accumulators updated on each tick delta.
        # On a playhead jump backward, rebuild is O(k) where k = new_idx.
        _acc = st.session_state.setdefault(
            "_sim_acc",
            {
                "idx": -1,
                "tier": {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0},
                "attacks": 0,
            },
        )
        if idx_local < _acc["idx"]:
            # Jumped backward — rebuild from scratch up to idx_local
            _acc["tier"] = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
            _acc["attacks"] = 0
            for _r in responses[: idx_local + 1]:
                _lv = _r.get("risk_level", "LOW")
                if _lv in _acc["tier"]:
                    _acc["tier"][_lv] += 1
                if _r.get("ground_truth") == "attack":
                    _acc["attacks"] += 1
            _acc["idx"] = idx_local
        elif idx_local > _acc["idx"]:
            # Advanced forward — only process new records (delta)
            for _i in range(_acc["idx"] + 1, idx_local + 1):
                _r = responses[_i]
                _lv = _r.get("risk_level", "LOW")
                if _lv in _acc["tier"]:
                    _acc["tier"][_lv] += 1
                if _r.get("ground_truth") == "attack":
                    _acc["attacks"] += 1
            _acc["idx"] = idx_local
        # _acc is always consistent with idx_local at this point

        # Status + progress
        status_col, prog_col = st.columns([1, 4])
        with status_col:
            running_local = st.session_state.sim_running
            st.markdown(
                f"**Status:** {'🟢 Running' if running_local else '🟡 Paused'} "
                f"&nbsp;&nbsp; `{st.session_state.sim_speed:g}x`",
                unsafe_allow_html=True,
            )
        with prog_col:
            st.progress(
                (idx_local + 1) / max(1, len(responses)),
                text=f"Alert {idx_local + 1} / {len(responses)}",
            )

        # Mock live source preview (only when live mode active)
        if using_live and live_df is not None and idx_local < len(live_df):
            live_row = live_df.iloc[idx_local]
            with st.expander(
                f"📡 Live source — row {idx_local} arrived at {live_row['arrived_at']}",
                expanded=False,
            ):
                st.caption(
                    "Mock TAP: feature-extracted flow read directly from "
                    "data/processed/test_phase1.parquet."
                )
                preview_cache = st.session_state.setdefault("_live_preview_cache", {})
                if idx_local not in preview_cache:
                    preview_cols = [c for c in live_row.index if c != "arrived_at"][:8]
                    preview_cache[idx_local] = pd.DataFrame(
                        {"feature": preview_cols, "value": [live_row[c] for c in preview_cols]}
                    )
                    if len(preview_cache) > 256:
                        preview_cache.pop(next(iter(preview_cache)))
                st.dataframe(
                    preview_cache[idx_local],
                    hide_index=True,
                    width="stretch",
                )

        # Summary metrics — O(1) reads from incremental accumulators
        st.markdown("---")
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("Samples Processed", idx_local + 1)
        mc2.metric("CRITICAL", _acc["tier"].get("CRITICAL", 0))
        mc3.metric("HIGH", _acc["tier"].get("HIGH", 0))
        mc4.metric("True Attacks Seen", _acc["attacks"])
        mc5.metric("Latest Risk Score", f"{responses[idx_local]['risk_score']:.3f}")

        # Risk gauge + 4-component breakdown
        if current_batch_local:
            latest = current_batch_local[-1]
            latest_score = latest["risk_score"]
            latest_level = latest["risk_level"]
            col_gauge, col_components = st.columns([1, 2])
            with col_gauge:
                st.markdown("#### Risk Score Gauge")
                st.metric(
                    "Current Alert",
                    f"{latest_score:.3f}",
                    delta=latest_level,
                    delta_color="inverse" if latest_level in ("CRITICAL", "HIGH") else "normal",
                )
                st.progress(min(latest_score, 1.0))
            with col_components:
                st.markdown("#### 4-Component Breakdown")
                comps = latest.get("risk_components", {})
                if comps:
                    comp_df = pd.DataFrame(
                        {
                            "Component": list(comps.keys()),
                            "Value": [float(v) for v in comps.values()],
                        }
                    )
                    st.bar_chart(comp_df.set_index("Component"), color="#3274A1")
                else:
                    st.caption("Component breakdown not available for this alert")

        # Rolling per-sample latency line chart (the playhead-driven part
        # of the latency profile — the static panels are rendered above,
        # outside the fragment).
        history_deque = st.session_state.get("latency_history")
        if history_deque is not None and len(history_deque) > 0:
            st.markdown(
                "**Rolling per-sample latency** "
                "(synthetic draws consistent with the recorded p50/p95)"
            )
            # Issue 3 fix: only reconstruct DataFrame when deque length changes.
            # pd.DataFrame(list(deque)) ran every tick even when nothing changed.
            _lat_cache = st.session_state.setdefault("_latency_df_cache", {"n": 0, "df": None})
            if len(history_deque) != _lat_cache["n"]:
                _lat_cache["df"] = pd.DataFrame(list(history_deque))
                _lat_cache["n"] = len(history_deque)
            roll_df = _lat_cache["df"]
            chart_cols = [c for c in roll_df.columns if c != "arrival_idx"]
            st.line_chart(roll_df[chart_cols])
            if "total_ms" in roll_df.columns:
                last_total = float(roll_df["total_ms"].iloc[-1])
                sla_warn = " ⚠️ exceeds 150 ms SLA" if last_total > 150 else " ✅ within 150 ms SLA"
                st.caption(f"Latest total latency: {last_total:.1f} ms{sla_warn}")
        else:
            st.caption("Step or resume the simulation to populate the rolling latency chart.")

        # Current batch (per-alert expanders with role render + interactions + FDA export)
        st.markdown("---")
        st.markdown("### Current Batch")
        alerts_cache = st.session_state.setdefault("_processed_alerts", {})
        fda_cache = st.session_state.setdefault("_fda_payload_cache", {})
        fda_filename_cache = st.session_state.setdefault("_fda_filename_cache", {})

        for r in current_batch_local:
            sample_idx = r["sample_index"]
            level = r["risk_level"]
            score = r["risk_score"]
            color = TIER_COLORS.get(level, "#999")

            with st.expander(
                f"Alert #{sample_idx} — :{color.replace('#', '')}[{level}] R={score:.3f}",
                expanded=(level in ("CRITICAL", "HIGH")),
            ):
                if sample_idx not in alerts_cache:
                    alerts_cache[sample_idx] = st.session_state.alerts_dict.get(sample_idx, r)
                alert_obj = alerts_cache[sample_idx]

                if sim_role == "Security Analyst":
                    render_analyst(alert_obj)
                elif sim_role == "Clinician":
                    render_clinician(alert_obj)
                else:
                    render_admin(alert_obj)

                resp = r.get("response", {})
                if resp:
                    render_prioritized_actions(resp.get("actions", []))

                # Day 3: FDA-style audit record export is researcher-only.
                # The cache fill is cheap (build once per sample, reuse) and
                # is kept outside the toggle so the download button shows up
                # immediately when an examiner flips Researcher Mode on.
                if sample_idx not in fda_cache:
                    fda_record = build_fda_record_for_alert(sample_idx, r, audit_trail)
                    fda_cache[sample_idx] = json.dumps(fda_record, indent=2).encode("utf-8")
                    fda_filename_cache[sample_idx] = f"audit_{fda_record['alert_id']}.json"
                if st.session_state.get("researcher_mode"):
                    st.download_button(
                        label="⬇ Export FDA-style Audit Record",
                        data=fda_cache[sample_idx],
                        file_name=fda_filename_cache[sample_idx],
                        mime="application/json",
                        key=f"fda_{sample_idx}",
                        help=(
                            "Download this alert as a Module-5 FDA-style audit "
                            "record (alert_id, timestamp, risk, actions, "
                            "rationale, simulated outcome, integrity hash). "
                            "Researcher tool — hidden by default."
                        ),
                    )

                btn_col1, btn_col2, btn_col3 = st.columns(3)
                with btn_col1:
                    if st.button("Confirm", key=f"confirm_{sample_idx}"):
                        capture_online_interaction(
                            st.session_state.get("participant_id", "anon"),
                            sample_idx,
                            "confirm",
                            {"tier": level, "score": score},
                        )
                        st.success("Confirmed")
                with btn_col2:
                    if st.button("Reject", key=f"reject_{sample_idx}"):
                        capture_online_interaction(
                            st.session_state.get("participant_id", "anon"),
                            sample_idx,
                            "reject",
                            {"tier": level, "score": score},
                        )
                        st.warning("Rejected — logged for feedback loop")
                with btn_col3:
                    note = st.text_input(
                        "Note",
                        key=f"note_{sample_idx}",
                        label_visibility="collapsed",
                        placeholder="Add feedback note...",
                    )
                    if note:
                        capture_online_interaction(
                            st.session_state.get("participant_id", "anon"),
                            sample_idx,
                            "feedback_note",
                            {"note": note, "tier": level},
                        )

        # Cumulative tier distribution (incremental, O(1) per advance)
        # Issue 3 fix: target_len uses idx_local+1 directly — no history_local.
        # The while-loop body accesses responses[i] directly instead of
        # history_local[i], eliminating the O(n) slice dependency.
        st.markdown("### Alert Tier Distribution (cumulative)")
        TIERS = ("LOW", "MEDIUM", "HIGH", "CRITICAL")
        tier_state = st.session_state.setdefault(
            "_tier_history",
            {"len": 0, "data": {t: [] for t in TIERS}},
        )
        target_len = idx_local + 1
        if target_len < tier_state["len"]:
            # Jumped backwards or reset → rebuild from scratch (O(k) once)
            tier_state["data"] = {t: [] for t in TIERS}
            tier_state["len"] = 0
        while tier_state["len"] < target_len:
            i = tier_state["len"]
            new_level = responses[i]["risk_level"]  # O(1) direct access
            for t in TIERS:
                prev = tier_state["data"][t][-1] if tier_state["data"][t] else 0
                tier_state["data"][t].append(prev + (1 if new_level == t else 0))
            tier_state["len"] += 1
        if tier_state["len"] > 0:
            DISPLAY_LIMIT = 200
            if tier_state["len"] <= DISPLAY_LIMIT:
                display = tier_state["data"]
            else:
                display = {t: tier_state["data"][t][-DISPLAY_LIMIT:] for t in TIERS}
            st.line_chart(pd.DataFrame(display))

    # Drive the playhead fragment. The first call renders the playhead
    # panels using the current sim_index. Subsequent ticks (via the
    # fragment's run_every timer) re-execute ONLY this function — none
    # of the static analytics above re-render until the next full script
    # rerun (sidebar/control change, button click outside the fragment).
    # Time the fragment call separately so the render-time caption can
    # report both the page-setup cost and the playhead cost.
    _playhead_t0 = time.perf_counter()
    _playhead()
    st.session_state["_last_playhead_ms"] = (time.perf_counter() - _playhead_t0) * 1000.0

    # Compute the equivalent autorefresh interval for the render-time
    # caption below — this is what the previous code reported as
    # `interval_ms`. Phase 2 no longer uses st_autorefresh, but the
    # number is still meaningful as the fragment tick interval.
    interval_ms = int(2000 / max(0.25, st.session_state.sim_speed))

    # Step 0 instrumentation tail — measure and record end-to-end render
    # cost of this rerun. Writes to /tmp/sim_render_timings.jsonl when
    # the sidebar toggle is on; otherwise no-op except for the rolling
    # in-memory deque used for the live caption.
    _render_ms = (time.perf_counter() - _render_t0) * 1000.0
    _hist = st.session_state.setdefault("_render_ms_history", deque(maxlen=50))
    _hist.append(_render_ms)
    _mean50 = sum(_hist) / len(_hist) if _hist else 0.0

    if st.session_state.get("_render_log_enabled"):
        try:
            with open("/tmp/sim_render_timings.jsonl", "a") as _rf:
                _rf.write(
                    json.dumps(
                        {
                            "ts": datetime.now().isoformat(),
                            "render_ms": round(_render_ms, 3),
                            "sim_index": int(st.session_state.sim_index),
                            "sim_speed": float(st.session_state.sim_speed),
                            "sim_running": bool(st.session_state.sim_running),
                            "sim_source": st.session_state.sim_source,
                        }
                    )
                    + "\n"
                )
        except Exception:  # noqa: BLE001
            pass

    if st.session_state.get("_render_caption_enabled", True):
        playhead_ms = st.session_state.get("_last_playhead_ms", 0.0)
        static_ms = _render_ms - playhead_ms
        st.caption(
            f"render: {_render_ms:.0f} ms (mean50={_mean50:.0f} ms, "
            f"interval={interval_ms} ms)  •  "
            f"playhead fragment: {playhead_ms:.0f} ms  •  "
            f"static page setup: {static_ms:.0f} ms"
        )


# ═══════════════════════════════════════════════════════════════════════
# 6.3a  Offline Evaluation (Browse + Study)
# ═══════════════════════════════════════════════════════════════════════
    # Day 6: Last 5 Decisions audit panel — INVARIANT 4.
    st.markdown("---")
    render_last_5_decisions_panel()



def display_alert(alert: dict, show_xai: bool):
    """Display an alert with or without XAI explanation."""
    st.markdown(f"### Alert: {alert['alert_id']}")

    # Gap 2: Device criticality badge at the top
    render_device_criticality(alert)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Risk Score", f"{alert['risk_score']:.2f}")
    with col2:
        level_colors = {"CRITICAL": "red", "HIGH": "orange", "MEDIUM": "blue", "LOW": "green"}
        st.markdown(
            f"**Risk Level:** :{level_colors.get(alert['risk_level'], 'gray')}[{alert['risk_level']}]"
        )

    if show_xai:
        st.markdown("---")

        # Gap 3: MVE 3-layer format
        render_mve_layers(alert)

        st.markdown("---")
        st.markdown("#### XAI Feature Detail")

        xai = alert.get("xai_explanation", {})
        top_feats = xai.get("xgboost_top_features", [])
        if top_feats:
            st.markdown("**Top Contributing Features (SHAP):**")
            for f in top_feats[:5]:
                direction = "increases" if f.get("shap_value", 0) > 0 else "decreases"
                st.markdown(
                    f"- **{f['feature']}**: {direction} risk (SHAP: {f.get('shap_value', 0):+.3f})"
                )

        dae_feats = xai.get("dae_top_features", [])
        if dae_feats:
            st.markdown("**DAE Anomaly Indicators:**")
            for f in dae_feats[:3]:
                st.markdown(
                    f"- **{f['feature']}**: {f.get('pct_contribution', 0):.1f}% of anomaly score"
                )

        consensus = xai.get("consensus", "")
        if consensus:
            st.info(f"Model consensus: {consensus}")

        wf_bytes = _cached_png_bytes(
            str(CHARTS_DIR / f"waterfall_xgboost_sample_{alert['sample_index']:04d}.png")
        )
        if wf_bytes:
            st.image(wf_bytes, caption="SHAP Waterfall Plot", width="stretch")
    else:
        st.markdown("---")
        st.info("No explanation available. Decide based on risk score and level only.")


def response_form(alert: dict, alert_index: int, show_xai: bool) -> dict | None:
    """Capture participant response — delegates to reusable likert_form()."""
    result = likert_form(alert["alert_id"], f"response_{alert_index}")
    if result:
        elapsed = round(time.time() - st.session_state.alert_start_time, 1)
        result.update(
            {
                "participant_id": st.session_state.participant_id,
                "participant_role": st.session_state.participant_role,
                "condition": "with_xai" if show_xai else "without_xai",
                "correct_action": alert.get("correct_action", ""),
                "decision_correct": result["chosen_action"] == alert.get("correct_action", ""),
                "decision_time_sec": elapsed,
                "timestamp": datetime.now().isoformat(),
            }
        )
        audit_log(
            "response_submit",
            participant_id=st.session_state.participant_id,
            alert_id=alert["alert_id"],
            action=result["chosen_action"],
            confidence=result.get("confidence"),
            role=st.session_state.get("role"),
            decision_time=elapsed,
        )
        _mark_decision_submitted()
        return result
    return None


def _render_demo_playlist_sidebar(alerts: list, current_idx: int) -> None:
    """Render the Demo Playlist jump-button block in the Browse sidebar.

    Each button writes ``_browse_target_idx`` and triggers a rerun; the
    main slider reads that value as its initial position. The latch
    clears after one consumption so manual slider edits remain
    responsive.
    """
    playlist = load_demo_playlist().get("alerts", [])
    if not playlist:
        return
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎬 Demo Playlist")
    st.sidebar.caption("Click a beat to jump.")
    current_id = alerts[current_idx].get("alert_id") if 0 <= current_idx < len(alerts) else None
    by_id = {a.get("alert_id"): i for i, a in enumerate(alerts)}
    for entry in playlist:
        aid = entry["alert_id"]
        beat_n = entry.get("narrative_position", "?")
        label = entry.get("narrative_label", "—")
        desc = entry.get("narrative_short_desc", "")
        target = by_id.get(aid)
        is_current = (aid == current_id)
        button_label = f"**{beat_n}. {label}**"
        if is_current:
            button_label = "▶ " + button_label
        if target is None:
            st.sidebar.button(
                button_label, key=f"playlist_{beat_n}_disabled",
                width="stretch", disabled=True,
                help=f"{desc}\n\n(Not in current view — toggle Demo Mode on to load.)",
            )
            continue
        clicked = st.sidebar.button(
            button_label,
            key=f"playlist_jump_{beat_n}",
            width="stretch",
            type="primary" if is_current else "secondary",
            help=desc,
        )
        if clicked and target != current_idx:
            st.session_state["_browse_target_idx"] = target
            st.rerun()
    total_s = load_demo_playlist().get("total_demo_time_seconds", 0)
    if total_s:
        st.sidebar.caption(f"Total budget: {total_s} s ({total_s // 60} min)")


def browse_mode():
    """6.3a — Free browsing with XAI toggle.

    Day 4 polish:
      * Honours Demo Mode — slider range follows the curated 5-alert
        playlist when the toggle is on.
      * Sidebar carries the playlist jump buttons so each narrative
        beat is one click away during the defense demo.
      * v4 alert-header strip surfaces above the existing detail body
        for visual consistency with Dashboard / Sim.
    """
    alerts = _get_alerts_for_demo_mode()
    n = len(alerts)
    if n == 0:
        st.title("\U0001f4c2 IoMT Alert Browser")
        st.warning("No alerts to browse.")
        return

    st.sidebar.markdown("## Browse Controls")
    show_xai = st.sidebar.toggle("Show XAI Explanation", value=True)

    # Slider initial value — honours an in-flight playlist jump request.
    initial_idx = st.session_state.pop("_browse_target_idx", None)
    if initial_idx is None or not (0 <= initial_idx < n):
        initial_idx = 0
    idx = st.sidebar.slider("Alert #", 0, max(n - 1, 0), value=initial_idx)
    alert = alerts[idx]

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Ground Truth:** `{alert.get('ground_truth', 'N/A')}`")
    st.sidebar.markdown(f"**Attack Type:** `{alert.get('attack_category', 'N/A')}`")
    st.sidebar.markdown(f"**Correct Action:** `{alert.get('correct_action', 'N/A')}`")
    if alert.get("is_synthetic_demo"):
        st.sidebar.warning(
            "⚠ **Synthetic alert** — for demo visualisation only; "
            "not part of the evaluation set."
        )

    _render_demo_playlist_sidebar(alerts, idx)

    st.title("\U0001f4c2 IoMT Alert Browser")
    _show_demo_mode_indicator()
    st.caption(
        f"Alert {idx + 1} of {n} — "
        f"{'With XAI' if show_xai else 'Without XAI'}"
    )
    # Day 4: v4 visual header for cross-page consistency. The existing
    # display_alert body keeps its SHAP / DAE / waterfall content and
    # already calls render_mve_layers (which carries Day 2's MITRE-per-
    # role line in Layer 1).
    display_alert_header_v4(alert)
    st.markdown("---")
    display_alert(alert, show_xai)

    # UX-B-01: Action affordance — show recommended action
    st.divider()
    st.subheader("\u26a1 Recommended Action")
    correct_action = alert.get("correct_action", "")
    _ACTION_GUIDANCE = {
        "isolate": (
            "\U0001f534 Isolate device from network",
            "Block all non-essential connections while preserving clinical paths.",
        ),
        "escalate": (
            "\U0001f7e0 Escalate immediately",
            "Notify security lead and clinical engineering on-call.",
        ),
        "investigate": (
            "\U0001f7e1 Investigate before acting",
            "Gather more information. Check with Biomed for scheduled maintenance.",
        ),
        "monitor": (
            "\U0001f7e2 Monitor — no immediate action",
            "Watch for escalation. Set alert for threshold change.",
        ),
        "dismiss": (
            "\u26aa Dismiss — expected behavior",
            "Verify with asset owner. Document reason for dismissal.",
        ),
    }
    label, guidance = _ACTION_GUIDANCE.get(
        correct_action,
        ("\u2139\ufe0f Review recommended", "Check response policy for this alert type."),
    )
    st.markdown(f"**{label}**")
    st.caption(guidance)
    # Day 6: Last 5 Decisions audit panel — INVARIANT 4.
    st.markdown("---")
    render_last_5_decisions_panel()



def _render_proxy_questions():
    """
    Q21 + Q22: proxy validation for clinical staff
    and management stakeholders.
    Shown once after all 20 alerts are completed.
    """
    st.title("Two Final Questions")
    st.markdown("Based on the alerts you reviewed, please answer these two questions.")

    with st.form("proxy_questions"):
        st.markdown("#### Q21 — Clinical Staff")
        q21 = st.radio(
            "If you forwarded one of these alerts to a nurse "
            "or physician, would they have enough information "
            "to understand the patient safety risk?",
            [
                "Yes — the information is clear for clinical staff",
                "Partially — some alerts were clear, others were not",
                "No — clinical staff would need more explanation",
            ],
            index=1,
        )
        q21_note = st.text_input(
            "What was missing for clinical staff? (optional)",
            placeholder="e.g. patient impact was unclear, too technical...",
        )

        st.markdown("---")
        st.markdown("#### Q22 — Management / Security Lead")
        q22 = st.radio(
            "If you reported these alerts to your manager "
            "or security lead, would the information be "
            "sufficient to justify your recommended action?",
            [
                "Yes — the explanation justifies the action clearly",
                "Partially — for some alerts yes, others no",
                "No — I would need to add more context myself",
            ],
            index=1,
        )
        q22_note = st.text_input(
            "What additional context would management need? (optional)",
            placeholder="e.g. business impact unclear, risk level hard to explain...",
        )

        if st.form_submit_button(
            "Submit & Complete Study", type="primary", use_container_width=True
        ):
            proxy = {
                "participant_id": st.session_state.participant_id,
                "q21_clinical_clarity": q21,
                "q21_note": q21_note,
                "q22_management_justification": q22,
                "q22_note": q22_note,
                "timestamp": datetime.now().isoformat(),
            }
            # Append to existing responses
            st.session_state.responses.append(proxy)
            st.session_state.proxy_done = True
            audit_log(
                "proxy_questions_submitted",
                participant_id=st.session_state.participant_id,
                q21=q21,
                q22=q22,
            )
            st.rerun()


_SEV_COLORS = {
    "CRITICAL": "#d32f2f",
    "HIGH": "#f57c00",
    "MEDIUM": "#1976d2",
    "LOW": "#388e3c",
}

# Issue 10 fix: compile patterns once at module load instead of running
# repeated `in` substring scans on every line of every Group B render.
import re as _re

_DO_NOT_RE = _re.compile(r"DO NOT", _re.IGNORECASE)
# Matches "SEVERITY: CRITICAL", "► HIGH", "SEVERITY HIGH" etc.
_SEV_LINE_RE = _re.compile(r"(?:SEVERITY[:\s]+|►\s*)(CRITICAL|HIGH|MEDIUM|LOW)", _re.IGNORECASE)


def _render_group_b_highlighted(display_text: str) -> None:
    """FIX 8: Render Group B display with severity color + DO NOT highlight.

    Issue 10 fix: module-level compiled regexes replace 3–5 per-line
    `in upper` substring scans with a single regex match per line per check.
    """
    lines = display_text.split("\n")
    regular: list[str] = []

    def _flush_regular():
        if regular:
            st.code("\n".join(regular), language=None)
            regular.clear()

    for line in lines:
        # DO NOT constraint → warning box
        if _DO_NOT_RE.search(line):
            _flush_regular()
            clean = line.strip().lstrip("\u2502").strip()
            st.warning(f"\u26a0\ufe0f {clean}")
            continue

        # Severity label line → colored banner
        m = _SEV_LINE_RE.search(line)
        if m:
            detected_sev = m.group(1).upper()
            _flush_regular()
            hex_c = _SEV_COLORS.get(detected_sev, "#757575")
            st.markdown(
                f'<div style="background:{hex_c};color:white;'
                f"padding:4px 10px;border-radius:4px;"
                f'font-family:monospace;margin:2px 0;">'
                f"{line.strip()}</div>",
                unsafe_allow_html=True,
            )
            continue

        # Regular line → batch for code block
        regular.append(line)

    _flush_regular()


def study_mode():
    """
    Phase 2 User Study — A/B design validating C4.
    Group A: raw IDS output only
    Group B: raw IDS + MVE (3-layer explanation)
    """
    from module6_evaluation.study_loader import load_study_alerts, assign_ab_condition

    # Day 5: Skip Registration demo bypass.
    # When the top-bar Demo Mode is ON the examiner can flip into a
    # dedicated A/B-comparison view (one curated alert, two tabs, no
    # Likert form, NO writes to ``study_responses_*.json``) without
    # touching the locked study-protocol flow. The Day 1 auto-DEMO
    # short-circuit (which silently registered participant_id="DEMO")
    # is removed in favour of this explicit, opt-in path.
    if st.session_state.get("study_demo_bypass_active"):
        _render_demo_bypass_view()
        return

    # ── Registration ──────────────────────────────────────────
    if not st.session_state.study_started:
        _render_demo_bypass_offer()
        if st.session_state.get("pid_conflict_check"):
            st.warning(
                f"Participant ID '{st.session_state.conflict_pid}' already has saved progress."
            )
            col1, col2, col3 = st.columns(3)

            if col1.button("Resume saved progress"):
                st.session_state.participant_id = st.session_state.conflict_pid
                pid = st.session_state.conflict_pid
                checkpoint_file = EVAL_DIR / f"study_checkpoint_{pid}.json"
                final_file = EVAL_DIR / f"study_responses_{pid}.json"

                saved = []
                if checkpoint_file.exists():
                    try:
                        with open(checkpoint_file, "r", encoding="utf-8") as f:
                            saved = json.load(f)
                    except Exception:
                        pass
                elif final_file.exists():
                    try:
                        with open(final_file, "r", encoding="utf-8") as f:
                            saved = json.load(f)
                    except Exception:
                        pass

                st.session_state.responses = saved
                st.session_state.current_alert = len(saved)
                st.session_state.study_started = True
                st.session_state.alert_start_time = time.time()
                st.session_state.study_alerts = load_study_alerts(pid)
                st.session_state.pid_conflict_check = False
                st.rerun()

            if col2.button("Start fresh (overwrite)"):
                st.session_state.participant_id = st.session_state.conflict_pid
                st.session_state.responses = []
                st.session_state.current_alert = 0
                st.session_state.study_started = True
                st.session_state.alert_start_time = time.time()
                st.session_state.study_alerts = load_study_alerts(st.session_state.participant_id)
                st.session_state.pid_conflict_check = False
                audit_log(
                    "study_start",
                    participant_id=st.session_state.conflict_pid,
                    role=st.session_state.participant_role,
                    years=st.session_state.participant_years,
                    ids_exp=st.session_state.participant_ids_exp,
                )
                st.rerun()

            if col3.button("Cancel & use different PID"):
                st.session_state.pid_conflict_check = False
                st.rerun()
            return

        st.title("\U0001f4cb Healthcare IDS Alert Evaluation Study")
        st.markdown(
            """
        **Purpose:** Evaluate how security alert information helps
        IT staff make response decisions.

        **Time required:** 30–40 minutes

        **What you will do:** Review 20 security alerts and decide
        how to respond to each one.
        """
        )

        with st.form("registration"):
            pid = st.text_input(
                "Participant ID", placeholder="e.g. P01, P02 ...", help="Assigned by researcher"
            )
            role = st.selectbox(
                "Your current role",
                [
                    "IT Security Generalist",
                    "Network/System Administrator",
                    "Healthcare IT Support",
                    "Other IT Role",
                ],
            )
            years_exp = st.slider("Years in current role", 1, 15, 3)
            has_ids_exp = st.radio("Have you worked with IDS/SIEM alerts before?", ["Yes", "No"])
            consent = st.checkbox(
                "I agree to participate in this research study "
                "and understand my responses will be anonymized."
            )

            if st.form_submit_button("Begin Study") and pid and consent:
                st.session_state.participant_role = role
                st.session_state.participant_years = years_exp
                st.session_state.participant_ids_exp = has_ids_exp

                checkpoint_file = EVAL_DIR / f"study_checkpoint_{pid}.json"
                final_file = EVAL_DIR / f"study_responses_{pid}.json"

                if checkpoint_file.exists() or final_file.exists():
                    st.session_state.pid_conflict_check = True
                    st.session_state.conflict_pid = pid
                else:
                    st.session_state.participant_id = pid
                    st.session_state.study_started = True
                    st.session_state.responses = []
                    st.session_state.current_alert = 0
                    st.session_state.alert_start_time = time.time()
                    st.session_state.study_alerts = load_study_alerts(pid)
                    audit_log(
                        "study_start",
                        participant_id=pid,
                        role=role,
                        years=years_exp,
                        ids_exp=has_ids_exp,
                    )
                st.rerun()
        return

    # ── Study complete ─────────────────────────────────────────
    if st.session_state.study_complete:
        st.title("\U0001f4cb Study Complete")
        st.success(f"Thank you for participating!")

        responses = st.session_state.responses
        n = len(responses)

        # Save responses
        save_path = (
            PROJECT_ROOT
            / "results"
            / "reports"
            / f"study_responses_{st.session_state.participant_id}.json"
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(responses, indent=2), encoding="utf-8")

        checkpoint_file = EVAL_DIR / f"study_checkpoint_{st.session_state.participant_id}.json"
        if checkpoint_file.exists():
            checkpoint_file.unlink()

        st.metric("Alerts Reviewed", n)
        st.info(
            f"Your responses have been saved. Results will be shared after the study concludes."
        )

        audit_log("study_complete", participant_id=st.session_state.participant_id, n_responses=n)
        return

    # ── Main study loop ────────────────────────────────────────
    alerts = st.session_state.study_alerts
    n_total = len(alerts)
    current_idx = st.session_state.current_alert

    if current_idx >= n_total:
        # Check if proxy questions already answered
        if not st.session_state.get("proxy_done", False):
            _render_proxy_questions()
            return
        st.session_state.study_complete = True
        st.rerun()
        return

    alert = alerts[current_idx]
    pid = st.session_state.participant_id
    show_mve = assign_ab_condition(pid, current_idx, n_total)

    # Progress bar
    progress = current_idx / n_total
    st.progress(progress, text=f"Alert {current_idx + 1} of {n_total}")

    # ── Alert display ──────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"### Alert {current_idx + 1}")
    st.markdown(
        "_You are the on-call IT security staff at a 300-bed hospital. "
        "Review the alert below and decide how to respond._"
    )

    # Show Group A or Group B content.
    # Day 5: Group B gets the v4 visual chrome (9-class badge / confidence
    # / Mode A-B / triage / severity strip + MITRE-per-role line) above
    # the existing locked stimulus prose. The ``alert.group_b_display``
    # text is the locked Phase-2 study material — unchanged. Only the
    # frame around it is enhanced. Group A stays raw (st.code) — that
    # IS the control-condition feel.
    if show_mve:
        alert_dict = _study_alert_dict_for(alert.alert_id)
        if alert_dict is not None:
            display_alert_header_v4(alert_dict)
            try:
                op_role = get_current_operator_role()
                atype, _, _ = derive_v4_fields(alert_dict)
                mitre_line = format_mitre_for_alert_type(atype, op_role)
            except Exception:
                mitre_line = ""
            if mitre_line:
                st.markdown(f"**Threat intelligence:** {mitre_line}")
            st.markdown("---")
        _render_group_b_highlighted(alert.group_b_display)
    else:
        st.code(alert.group_a_display, language=None)

    # ── Response form ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Your Decision")

    with st.form(f"alert_form_{current_idx}"):
        severity = st.radio(
            "1. How severe is this alert? *(select one)*",
            [
                "CRITICAL — Respond immediately",
                "HIGH — Respond within 1 hour",
                "MEDIUM — Respond within 4 hours",
                "LOW — Review within 24 hours",
            ],
            index=None,
        )

        action = st.radio(
            "2. What action would you take? *(select one)*",
            [
                "Isolate the device/system from the network",
                "Escalate to clinical staff / senior management",
                "Investigate further before taking action",
                "Monitor closely but no immediate action",
                "Dismiss — this is likely a false alarm",
            ],
            index=None,
        )

        confidence = st.select_slider(
            "3. How confident are you in this decision?",
            options=[1, 2, 3, 4, 5],
            value=3,
            format_func=lambda x: {
                1: "1 — Guessing",
                2: "2 — Uncertain",
                3: "3 — Somewhat confident",
                4: "4 — Confident",
                5: "5 — Very confident",
            }[x],
        )

        submitted = st.form_submit_button(
            "Submit & Next Alert →", type="primary", use_container_width=True
        )

        if submitted:
            # Validate selections (FIX 7: no default → must select)
            if severity is None or action is None:
                if severity is None:
                    st.error("Please select a severity level before submitting.")
                if action is None:
                    st.error("Please select an action before submitting.")
                st.stop()

            elapsed = round(time.time() - st.session_state.alert_start_time, 1)

            # Map display values to scoring values
            severity_map = {
                "CRITICAL — Respond immediately": "CRITICAL",
                "HIGH — Respond within 1 hour": "HIGH",
                "MEDIUM — Respond within 4 hours": "MEDIUM",
                "LOW — Review within 24 hours": "LOW",
            }
            action_map = {
                "Isolate the device/system from the network": "isolate",
                "Escalate to clinical staff / senior management": "escalate",
                "Investigate further before taking action": "investigate",
                "Monitor closely but no immediate action": "monitor",
                "Dismiss — this is likely a false alarm": "dismiss",
            }

            chosen_severity = severity_map[severity]
            chosen_action = action_map[action]

            # Score response
            severity_correct = chosen_severity == alert.correct_severity
            action_correct = chosen_action == alert.correct_action

            # Partial credit for severity
            LEVEL = {"CRITICAL": 3, "HIGH": 2, "MEDIUM": 1, "LOW": 0}
            sev_diff = abs(LEVEL.get(chosen_severity, -1) - LEVEL.get(alert.correct_severity, -1))
            severity_score = 1.0 if sev_diff == 0 else (0.5 if sev_diff == 1 else 0.0)
            catastrophic = sev_diff == 3  # CRITICAL↔LOW mismatch

            composite_score = (severity_score + (1.0 if action_correct else 0.0)) / 2

            response = {
                "participant_id": pid,
                "participant_role": st.session_state.participant_role,
                "alert_id": alert.alert_id,
                "alert_type": alert.alert_type,
                "alert_index": current_idx,
                "condition": "with_mve" if show_mve else "without_mve",
                "chosen_severity": chosen_severity,
                "correct_severity": alert.correct_severity,
                "severity_correct": severity_correct,
                "severity_score": severity_score,
                "catastrophic_miss": catastrophic,
                "chosen_action": chosen_action,
                "correct_action": alert.correct_action,
                "action_correct": action_correct,
                "composite_score": composite_score,
                "confidence": confidence,
                "decision_time_sec": elapsed,
                "ground_truth_label": alert.ground_truth_label,
                "timestamp": datetime.now().isoformat(),
            }

            st.session_state.responses.append(response)
            st.session_state.current_alert += 1

            if st.session_state.current_alert % 5 == 0:
                checkpoint_file = EVAL_DIR / f"study_checkpoint_{pid}.json"
                with open(checkpoint_file, "w", encoding="utf-8") as f:
                    json.dump(st.session_state.responses, f, indent=2)

            st.session_state.alert_start_time = time.time()

            audit_log(
                "alert_response",
                participant_id=pid,
                alert_id=alert.alert_id,
                condition=response["condition"],
                composite_score=composite_score,
                confidence=response.get("confidence"),
                role=st.session_state.get("role"),
                decision_time=elapsed,
            )
            _mark_decision_submitted()
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
    # Day 6: Last 5 Decisions audit panel — INVARIANT 4.
    st.markdown("---")
    render_last_5_decisions_panel()




# ─────────────────────────────────────────────────────────────────
# Day 7: Projector typography
# ─────────────────────────────────────────────────────────────────
# Bumps the base font to 17 px and tightens a handful of component
# selectors so the dashboard reads cleanly on a 1920×1080 projector
# at 2 m viewing distance. The CSS is scoped to ``[data-testid=...]``
# selectors (Streamlit's stable hooks) rather than auto-generated
# CSS class names, so it should survive Streamlit minor-version bumps.

PROJECTOR_CSS = """
<style>
html, body, [data-testid="stAppViewContainer"] {
    font-size: 17px;
    line-height: 1.5;
}
h1, h2, h3, h4 { font-weight: 600; line-height: 1.3; }
h1 { font-size: 2.0rem; }
h2 { font-size: 1.6rem; }
h3 { font-size: 1.3rem; }
[data-testid="stMarkdownContainer"] p { font-size: 17px; line-height: 1.5; }
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 2.0rem !important; font-weight: 600;
}
[data-testid="stMetric"] [data-testid="stMetricLabel"] {
    font-size: 1.0rem !important; font-weight: 500;
}
[data-testid="stMetric"] [data-testid="stMetricDelta"] { font-size: 0.95rem !important; }
[data-testid="stSelectbox"] label,
[data-testid="stRadio"] label,
[data-testid="stCheckbox"] label { font-size: 17px; font-weight: 500; }
[data-testid="stButton"] button { font-size: 16px; font-weight: 500; padding: 8px 16px; }
[data-testid="stDataFrame"] { font-size: 14px; }
[data-testid="stExpander"] summary { font-size: 16px; font-weight: 500; }
[data-testid="stCaptionContainer"] { font-size: 13px; color: #6B7280; }
[data-testid="stTabs"] button[aria-selected="true"] {
    border-bottom: 3px solid #2563EB; font-weight: 600;
}
.do-not-box {
    border: 2px solid #DC2626; background-color: #FEF2F2;
    padding: 12px 16px; border-radius: 6px;
    font-size: 16px; font-weight: 500;
}
</style>
"""


def main():
    st.set_page_config(page_title="IoMT IDS Dashboard", layout="wide")
    st.markdown(PROJECTOR_CSS, unsafe_allow_html=True)
    init_session()

    _render_top_bar()

    st.sidebar.title("IoMT IDS")
    mode = st.sidebar.radio(
        "Mode:",
        ["Dashboard", "Online Simulation", "Browse Alerts", "Study (A/B)"],
    )

    if mode == "Dashboard":
        dashboard_mode()
    elif mode == "Online Simulation":
        simulation_mode()
    elif mode == "Browse Alerts":
        browse_mode()
    elif mode == "Study (A/B)":
        study_mode()


if __name__ == "__main__":
    main()
