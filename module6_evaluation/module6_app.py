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
from streamlit_autorefresh import st_autorefresh


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

ROLES = ["Security Analyst", "Clinician", "Administrator"]
ACTIONS = ["dismiss", "monitor", "investigate", "isolate", "escalate"]

TIER_COLORS = {"CRITICAL": "#8e44ad", "HIGH": "#e74c3c", "MEDIUM": "#e67e22", "LOW": "#2ecc71"}

from common.phi import BIOMETRIC_COLUMNS as BIOMETRIC_FEATURES  # noqa: E402

# Wires the dashboard's per-alert processing to the research prototype's
# Risk-Adaptive Scoring Engine (research_spec.yaml component_2) so tier
# assignment uses the same logic the prototype tests enforce (M7, M6).
from module6_evaluation._src_adapter import scored_from_eval_alert  # noqa: E402


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
    "isolate_device":        (1, "\U0001f534", "Isolate device"),
    "escalate_incident":     (2, "\U0001f7e0", "Escalate to security lead"),
    "escalate_clinical":     (2, "\U0001f7e0", "Escalate to clinical engineering"),
    "restrict_traffic":      (3, "\U0001f7e1", "Restrict suspicious traffic"),
    "re_authenticate":       (3, "\U0001f7e1", "Force re-authentication"),
    "forensic_snapshot":     (4, "\U0001f535", "Capture forensic snapshot"),
    "enhanced_monitoring":   (5, "\U0001f7e2", "Enable enhanced monitoring"),
    "log_event":             (6, "\u26aa", "Log event"),
}

_CRIT_COLOR_HEX = {
    "CRITICAL": "#d32f2f", "HIGH": "#f57c00",
    "MEDIUM": "#1976d2", "LOW": "#388e3c",
}

# Module-level policy action label map — avoids rebuilding this dict on every
# render_mve_layers() call (issue 4 / render_mve_layers locality fix).
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

# Module-level sentinel for _ACTION_DISPLAY misses — avoids {} allocation
# per sort-key lambda invocation (issue 9).
_ACTION_DISPLAY_MISS = (99, "\u26aa", "")

# M6-A1: hoist _ACTION_PRIORITY to module level — was rebuilt as a dict
# literal on every process_alert() call (one call per alert per simulation tick).
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
        f'Device: {criticality}</span>',
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
    xai  = alert.get("xai_explanation") or {}
    expl = alert.get("explanation") or {}
    resp = alert.get("response") or {}

    # Merge once — O(1) view construction, O(1) per key lookup thereafter.
    _cm = ChainMap(
        alert,
        xai  if isinstance(xai,  dict) else {},
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
    l1 = _get("why_anomalous", "layer_1", "baseline_behavior",
              "deviation_description", "confidence_indicator",
              "clinician_summary", "nlg_text")
    consensus = _get("consensus")

    with st.expander("\U0001f50d Layer 1 \u2014 Why Anomalous", expanded=True):
        if l1:
            st.write(l1)
            if consensus:
                st.caption(f"Model consensus: {consensus}")
        else:
            st.caption("Baseline deviation detected. See SHAP features below.")

    # ── Layer 2: Clinical Severity ──
    affected = _get("affected_system")
    impact = _get("patient_care_impact")
    severity = _get("severity_label", "severity", "risk_level", "tier")
    device_tier = _get("device_tier", "device_class")

    with st.expander("\U0001f3e5 Layer 2 \u2014 Clinical Severity", expanded=True):
        if severity:
            color = TIER_COLORS.get(severity.upper(), "#999")
            st.markdown(
                f"**Severity:** <span style='color:{color}'>"
                f"{severity}</span>",
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
    action = _get("recommended_action", "layer_3", "immediate_action",
                  "response_action", "correct_action")
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
    # Local eval-app audit trail
    audit_log("online_interaction", **record)
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
    """Take a raw alert record and produce a fully structured alert object.

    In production, this would run Modules 2-5 live. Here it assembles
    from pre-computed artifacts (risk scores, SHAP, NLG, responses) and
    runs the scoring step through the research prototype's Risk-Adaptive
    Scoring Engine (src.risk_scorer.score_alert) so the dashboard's
    should-surface decision matches the logic M7/M6 enforce.
    """
    xai = alert_data.get("xai_explanation", {})
    expl = alert_data.get("explanation", {})
    clinician_summary = (
        xai.get("clinician_summary", "")
        or (expl.get("clinician_summary", "") if isinstance(expl, dict) else "")
    )

    # Derive recommended action: correct_action > response.actions > risk-level default
    resp = alert_data.get("response", {})
    action = alert_data.get("correct_action", "")
    if not action and isinstance(resp, dict):
        # Use the highest-priority policy action (last in list = most severe).
        # M6-A1: _ACTION_PRIORITY is now a module-level constant — no per-call alloc.
        policy_actions = resp.get("actions", [])
        for pa in reversed(policy_actions):
            mapped = _ACTION_PRIORITY.get(pa)
            if mapped:
                action = mapped
                break
    if not action:
        # Last resort: derive from risk_level
        level = alert_data.get("risk_level", "LOW")
        action = {"CRITICAL": "isolate", "HIGH": "escalate",
                  "MEDIUM": "investigate", "LOW": "monitor"}.get(level, "monitor")

    # Risk-adaptive scoring via the prototype's Component 2. Any mismatch
    # between dashboard and test-harness outputs is now a bug in one place.
    scored = scored_from_eval_alert(alert_data)

    return {
        "sample_index": sample_index,
        "prediction": 1 if scored.should_surface else 0,
        "confidence": scored.adjusted_score,
        "risk_score": scored.adjusted_score,
        "raw_risk_score": alert_data.get("risk_score", 0),
        "threshold": scored.threshold,
        "risk_multiplier": scored.risk_multiplier,
        "suppression_reason": scored.suppression_reason,
        "tier": alert_data.get("risk_level", "LOW"),
        "attack_category": alert_data.get("attack_category", "unknown"),
        "ground_truth": alert_data.get("ground_truth", "unknown"),
        "shap_top_features": xai.get("xgboost_top_features", []),
        "dae_top_features": xai.get("dae_top_features", []),
        "nlg_text": clinician_summary,
        "consensus": xai.get("consensus", "") or (expl.get("consensus", "") if isinstance(expl, dict) else ""),
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
    }


# ═══════════════════════════════════════════════════════════════════════
# 6A.4  Stakeholder view renderers
# ═══════════════════════════════════════════════════════════════════════


def render_analyst(alert: dict):
    """Analyst view: SHAP plots + feature table + classification detail."""
    st.markdown("#### Security Analyst View")

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
            "device_class", "device_criticality", "affected_system",
            "patient_care_impact", "active_device", "correct_action",
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

    Issue 6 fix: the list-of-dicts → pd.DataFrame construction ran on
    every render inside dashboard_mode(). Pre-computing it with a hashable
    key (tuple of the first 15 records' key fields) means the O(15) dict
    construction + DataFrame ctor run once per cache key, not per render.

    Args:
        responses_head: tuple of (sample_index, risk_level, risk_score,
            device_class, attack_category, correct_action) for the first
            15 responses — fully hashable for st.cache_data.
    Returns:
        pd.DataFrame ready for st.dataframe().
    """
    rows = [
        {
            "Sample":   item[0],
            "Level":    item[1],
            "Score":    round(item[2], 3),
            "Device":   item[3] or "\u2014",
            "Category": item[4] or "",
            "Action":   item[5] or "\u2014",
        }
        for item in responses_head
    ]
    return pd.DataFrame(rows)


@st.cache_data
def load_latency_profile() -> dict:
    """Module 4 online_latency_profile.json — aggregate per-stage latency stats."""
    path = EVAL_DIR / "online_latency_profile.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


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
    df["arrived_at"] = (
        pd.date_range(start=base, periods=len(df), freq="1s")
        .strftime("%Y-%m-%dT%H:%M:%S")
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
        "ab_conditions": [],
        "app_mode": "dashboard",
        "sim_index": 0,
        "sim_running": True,
        "sim_history": [],
        "sim_speed": 1.0,  # 0.5x / 1x / 2x / 4x
        "sim_source": "alerts",  # "alerts" or "live_parquet"
        "latency_history": deque(maxlen=120),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


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
    """Triage view — three-column Sentinel layout per `docs/sentinel_dashboard.html`.

    D1=A (Streamlit refactor in place). Visual direction locked to the
    prototype; ~85% fidelity envelope. See `docs/dashboard_design_memo.md`
    Phase 3 Plan for the implementation contract.
    """
    from module6_evaluation.sentinel_theme import inject_theme
    from module6_evaluation import components as ui

    inject_theme()

    responses = load_all_responses()
    if not responses:
        st.warning("No alert data found. Run Modules 3-5 first.")
        return

    # Visual queue: top 50 by tier-then-score (CRITICAL first, then HIGH...)
    _TIER_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    sorted_resp = sorted(
        responses,
        key=lambda r: (_TIER_ORDER.get(r.get("risk_level", "LOW"), 9),
                       -r.get("risk_score", 0.0)),
    )
    visible = sorted_resp[:50]

    counts = Counter(r.get("risk_level", "LOW") for r in responses)

    # Selection state — single source of truth: st.session_state["selected_alert_id"]
    if "selected_alert_id" not in st.session_state:
        st.session_state["selected_alert_id"] = visible[0].get("sample_index")
    selected = next(
        (r for r in responses
         if r.get("sample_index") == st.session_state["selected_alert_id"]),
        visible[0],
    )

    with st.container(key="sentinel-triage"):
        col_q, col_inv, col_act = st.columns([1.3, 3.0, 1.7], gap="small")
        with col_q:
            _triage_queue_column(visible, counts, selected, ui)
        with col_inv:
            _triage_investigation_column(selected, ui)
        with col_act:
            _triage_actions_column(selected, ui)

    _triage_status_strip(ui)


def _floor_elevated(alert: dict) -> bool:
    """Approximate the Module-5 safety-floor invariant for visual purposes.

    The canonical floor logic is in `module5_responses/module5_pipeline.py`
    (not opened in Phase 0 per Q-W5 Section 4 follow-up). This proxy flags
    alerts where a life-critical device (D_crit ≥ 0.9) is paired with a
    high-tier composite — the same conditions the prototype illustrates at
    `docs/sentinel_dashboard.html:974` (Invariant 2).
    """
    d_crit = (alert.get("risk_components") or {}).get("D_crit", 0.0)
    tier = alert.get("risk_level", "LOW")
    return tier in ("CRITICAL", "HIGH") and d_crit >= 0.9


def _triage_queue_column(visible, counts, selected, ui):
    sel_id = selected.get("sample_index")
    total_open = sum(counts.values())

    st.markdown(
        f'<div style="padding:16px 16px 4px;">'
        f'  <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:12px;">'
        f'    <h2 class="font-display" style="font-size:1.5rem;margin:0;letter-spacing:-0.02em;color:var(--text-primary);">Active queue</h2>'
        f'    <span class="font-mono" style="font-size:11px;color:var(--text-tertiary);">{total_open} open</span>'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # 4-up tier-count tile grid
    tiles = '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px;padding:0 16px 12px;">'
    for tier in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
        tiles += ui.render_tier_count_tile(tier, counts.get(tier, 0))
    tiles += '</div>'
    st.markdown(tiles, unsafe_allow_html=True)

    # Functional selectbox — source of truth for selection (click-on-row
    # is the prototype's behavior; Streamlit can't bind clicks to arbitrary
    # HTML, so the selectbox is the workable substitute under D1=A).
    options = [r.get("sample_index") for r in visible]
    def _fmt(idx):
        a = next(a for a in visible if a.get("sample_index") == idx)
        return f"{a.get('risk_level', 'LOW')[:4]}  ·  A-{idx:04d}  ·  {a.get('attack_category', '?')}"

    chosen = st.selectbox(
        "Select alert",
        options,
        index=options.index(sel_id) if sel_id in options else 0,
        format_func=_fmt,
        key="alert_selectbox",
        label_visibility="collapsed",
    )
    if chosen != sel_id:
        st.session_state["selected_alert_id"] = chosen
        st.rerun()

    # Visual queue, grouped by tier
    queue_html = ""
    for tier in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
        tier_alerts = [a for a in visible if a.get("risk_level") == tier]
        if not tier_alerts:
            continue
        queue_html += ui.render_tier_header(tier, counts.get(tier, 0))
        for a in tier_alerts:
            aid = a.get("sample_index", 0)
            queue_html += ui.render_alert_row(
                alert_id=f"A-{aid:04d}",
                title=a.get("attack_category", "Alert"),
                subtitle=f"sample {aid} · score {a.get('risk_score', 0.0):.2f}",
                tier=tier,
                age="",
                floor_elevated=_floor_elevated(a),
                active=(aid == sel_id),
            )
    st.markdown(queue_html, unsafe_allow_html=True)


def _triage_investigation_column(selected, ui):
    from html import escape

    aid = selected.get("sample_index", 0)
    tier = selected.get("risk_level", "LOW")
    components = selected.get("risk_components", {}) or {}
    floor = _floor_elevated(selected)
    composite = selected.get("risk_score", 0.0)
    raw = max(0.0, composite - 0.15) if floor else composite
    floor_delta = (composite - raw) if floor else None

    subtitle_html = (
        f'Attack <span class="font-mono" style="color:var(--text-primary);">'
        f'{escape(selected.get("attack_category", "?"))}</span> · '
        f'<span class="font-mono" style="color:var(--text-primary);">sample {aid}</span> · '
        f'ground truth <span class="font-mono">{escape(selected.get("ground_truth", "?"))}</span>'
    )

    st.markdown(
        ui.render_investigation_header(
            alert_id=f"A-{aid:04d}",
            tier=tier,
            title=selected.get("attack_category") or f"Alert {aid}",
            subtitle_html=subtitle_html,
            composite_risk=composite,
            raw_risk=raw if floor else None,
            floor_delta=floor_delta,
            floor_elevated=floor,
            invariant_label="Invariant 2",
        ),
        unsafe_allow_html=True,
    )

    # 4-column metric grid mapping risk_components to the prototype's
    # detection/criticality/sensitivity/clinical-tier breakdown.
    metric_grid = (
        '<div style="padding:20px 32px;display:grid;grid-template-columns:repeat(4,1fr);'
        'gap:24px;border-bottom:1px solid var(--border-subtle);">'
    )
    for label, key, color, sub in (
        ("Detection confidence", "C_detect",  "--accent",        "calibrated"),
        ("Device criticality",   "D_crit",    "--tier-critical", "life-critical" if components.get("D_crit", 0) >= 0.9 else "device class"),
        ("Data sensitivity",     "S_data",    "--tier-medium",   "data scope"),
        ("Patient acuity",       "A_patient", "--tier-high",     "active-care"),
    ):
        v = float(components.get(key, 0.0))
        metric_grid += ui.render_metric_with_bar(
            label, v, sub, color, bar_value=v, with_ticks=(key == "C_detect"),
        )
    metric_grid += '</div>'
    st.markdown(metric_grid, unsafe_allow_html=True)

    # Risk-component contribution rows (substitute for SHAP under
    # current data contract — SHAP top-features live on evaluation_alerts,
    # not alert_responses; tying the two requires the device-context join
    # already done in load_all_responses).
    st.markdown(
        '<div style="padding:24px 32px 8px;">'
        '  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">'
        '    <h3 class="font-display" style="font-size:1.25rem;margin:0;letter-spacing:-0.01em;color:var(--text-primary);">Risk component contributions</h3>'
        '    <span class="font-mono" style="font-size:10px;text-transform:uppercase;letter-spacing:0.08em;color:var(--text-tertiary);">6 components</span>'
        '  </div>'
        '</div>',
        unsafe_allow_html=True,
    )
    factor_html = '<div style="padding:0 32px;">'
    component_labels = (
        ("C_detect",  "Detection confidence",   "consensus across detectors"),
        ("C_track_a", "Track-A consistency",    "behavior vs cohort baseline"),
        ("C_track_b", "Track-B consistency",    "behavior vs device self-baseline"),
        ("D_crit",    "Device criticality",     "device-class weight"),
        ("S_data",    "Data sensitivity",       "what this device touches"),
        ("A_patient", "Patient acuity",         "active-care weight"),
    )
    for key, label, sub in component_labels:
        v = float(components.get(key, 0.0))
        factor_html += ui.render_factor_row(label, sub, int(round(v * 100)), v)
    factor_html += '</div>'
    st.markdown(factor_html, unsafe_allow_html=True)

    # Recommended-actions card (from response policy)
    resp = selected.get("response", {}) or {}
    actions = resp.get("actions", [])
    if actions:
        body = '<div style="display:flex;flex-direction:column;gap:6px;">'
        for a in actions[:5]:
            body += (
                f'<div style="font-size:0.875rem;color:var(--text-primary);">'
                f'<span class="font-mono" style="color:var(--accent);">→ </span>{escape(str(a))}'
                f'</div>'
            )
        body += '</div>'
        meta_bits = []
        if resp.get("max_response_min") is not None:
            meta_bits.append(f'Max response <span style="color:var(--text-primary);">{resp.get("max_response_min")} min</span>')
        if resp.get("priority"):
            meta_bits.append(f'Priority <span style="color:var(--text-primary);">{escape(str(resp.get("priority")))}</span>')
        if meta_bits:
            body += (
                '<div class="font-mono" style="margin-top:12px;padding-top:12px;'
                'border-top:1px solid var(--border-subtle);font-size:11px;color:var(--text-tertiary);">'
                + ' · '.join(meta_bits)
                + '</div>'
            )
        st.markdown(
            f'<div style="padding:20px 32px 32px;">'
            f'{ui.render_card("Response policy · device-aware", body)}</div>',
            unsafe_allow_html=True,
        )


def _triage_actions_column(selected, ui):
    from html import escape
    aid = selected.get("sample_index", 0)
    tier = selected.get("risk_level", "LOW")

    # Role pills (Step 3) — replaces the sidebar selectbox at the legacy L1300.
    st.markdown('<div data-sentinel-role-pills="1" style="padding:20px 20px 8px;">', unsafe_allow_html=True)
    st.markdown(
        '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">'
        '  <h3 class="font-display" style="font-size:1.25rem;margin:0;letter-spacing:-0.01em;color:var(--text-primary);">Why this fired</h3>'
        '</div>',
        unsafe_allow_html=True,
    )
    role_pick = st.pills(
        "Role",
        ["SOC", "Clinical", "Admin"],
        default=st.session_state.get("sim_role_pill", "SOC"),
        selection_mode="single",
        key="sim_role_pill",
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Role-adaptive body (delegates to existing render_* functions — Step 3.5
    # body review parked; reuse keeps behavior stable).
    role = role_pick or "SOC"
    # Map pill -> existing sim_role string so the legacy renderers work unchanged.
    st.session_state["sim_role"] = {
        "SOC": "Security Analyst",
        "Clinical": "Clinician",
        "Admin": "Administrator",
    }[role]

    with st.container(border=False):
        st.markdown('<div style="padding:0 20px 16px;">', unsafe_allow_html=True)
        if role == "SOC":
            clin = (selected.get("explanation") or {}).get("clinician_summary", "")
            st.markdown(
                f'<p style="font-size:0.875rem;line-height:1.55;color:var(--text-primary);margin:0 0 8px;">'
                f'Attack category <span class="font-mono" style="color:var(--accent);">{escape(selected.get("attack_category", "?"))}</span> '
                f'flagged at risk <span class="font-mono">{selected.get("risk_score", 0.0):.2f}</span> (tier {escape(tier)}).'
                f'</p>'
                f'<p style="font-size:0.875rem;line-height:1.55;color:var(--text-secondary);margin:0;">'
                f'Six risk components contributed; see the contribution panel. The composite reflects detector consensus,'
                f' device-class weight, and active-care weighting.</p>',
                unsafe_allow_html=True,
            )
        elif role == "Clinical":
            clin = (selected.get("explanation") or {}).get("clinician_summary", "")
            st.markdown(
                f'<p style="font-size:0.875rem;line-height:1.55;color:var(--text-primary);margin:0 0 8px;">'
                f'{escape(clin) if clin else "No clinician summary on file for this alert."}</p>',
                unsafe_allow_html=True,
            )
        else:  # Admin
            comps = selected.get("risk_components", {}) or {}
            top = sorted(comps.items(), key=lambda kv: -kv[1])[:3]
            top_html = " · ".join(
                f'<span class="font-mono">{escape(k)}={v:.2f}</span>' for k, v in top
            )
            st.markdown(
                f'<p style="font-size:0.875rem;line-height:1.55;color:var(--text-primary);margin:0 0 8px;">'
                f'Aggregate: tier <span class="font-mono">{escape(tier)}</span>, composite '
                f'<span class="font-mono">{selected.get("risk_score", 0.0):.2f}</span>. Top components: {top_html}.</p>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    # Action buttons (Step 4)
    st.markdown(
        '<div style="padding:8px 20px 4px;">'
        '  <div style="font-size:10px;font-weight:500;letter-spacing:0.08em;text-transform:uppercase;color:var(--text-tertiary);margin-bottom:10px;">Recommended actions · human-required</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    btn_col = st.container()
    with btn_col:
        st.markdown('<div data-sentinel-action="acknowledge" style="padding:0 20px 6px;">', unsafe_allow_html=True)
        if st.button("✓  Acknowledge — taking ownership", key=f"ack_{aid}", width="stretch"):
            _capture_dashboard_action(aid, "acknowledge", details={"tier": tier, "role": role})
            st.toast(f"Alert acknowledged · A-{aid:04d}", icon="✅")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div data-sentinel-action="escalate" style="padding:0 20px 6px;">', unsafe_allow_html=True)
        if st.button("↑  Escalate — pull in T3 + biomed", key=f"esc_{aid}", width="stretch"):
            _capture_dashboard_action(aid, "escalate", details={"tier": tier, "role": role})
            st.toast(f"Escalated · A-{aid:04d}", icon="⚠️")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div data-sentinel-action="dismiss" style="padding:0 20px 6px;">', unsafe_allow_html=True)
        if st.button("✕  Dismiss — requires reason", key=f"dis_{aid}", width="stretch"):
            _dismiss_dialog(aid, tier, role)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(
            f'<div style="padding:6px 20px 16px;">{ui.render_actions_disclaimer()}</div>',
            unsafe_allow_html=True,
        )

    # Audit timeline — derived from audit_trail.json for this sample
    audit = load_audit_trail()
    rec = audit.get(aid, {}) if isinstance(audit, dict) else {}
    events = rec.get("events", []) if isinstance(rec, dict) else []
    if not events:
        events = [
            {"kind": "system", "label": "Alert raised", "ts": "", "body": f"Sentinel · risk {selected.get('risk_score', 0.0):.2f} · tier {tier}"},
        ]
    timeline_html = (
        '<div style="padding:8px 20px 24px;">'
        f'  <div style="font-size:10px;font-weight:500;letter-spacing:0.08em;text-transform:uppercase;color:var(--text-tertiary);margin-bottom:12px;">Audit trail · {len(events)} entr{"y" if len(events) == 1 else "ies"}</div>'
    )
    for i, ev in enumerate(events[:8]):
        kind = ev.get("kind") or ("human" if "operator" in ev else "system")
        timeline_html += ui.render_timeline_item(
            kind=kind,
            label=str(ev.get("label", ev.get("event_type", "event"))),
            timestamp=str(ev.get("ts", ev.get("timestamp", ""))),
            body=str(ev.get("body", ev.get("rationale", ""))),
            is_last=(i == len(events) - 1 or i == 7),
        )
    timeline_html += '</div>'
    st.markdown(timeline_html, unsafe_allow_html=True)


def _capture_dashboard_action(sample_idx: int, action: str, details: dict | None = None) -> None:
    """Route a Triage-view action through the existing triple-sink audit fan-out.

    Mirrors the contract of `capture_interaction` (L520) but for the dashboard
    page (where no participant_id is set in study mode). Action vocabulary
    extension: adds "acknowledge" as the explicit ownership signal.
    """
    payload = {
        "timestamp": datetime.now().isoformat(),
        "participant_id": st.session_state.get("participant_id", "dashboard_user"),
        "alert_id": f"ALERT-{sample_idx:05d}",
        "action_type": action,
        "details": details or {},
    }
    try:
        _online_writer.log(action, **payload)
    except Exception:
        pass
    audit_log("dashboard_action", **payload)
    try:
        _hardened_audit.log(
            {"event_type": "dashboard_action",
             "alert_id": payload["alert_id"],
             "action": action,
             "details": details or {}},
            reviewer_id=payload["participant_id"],
            reviewer_role=st.session_state.get("sim_role", ""),
        )
    except Exception:
        pass


@st.dialog("Dismiss alert")
def _dismiss_dialog(sample_idx: int, tier: str, role: str):
    """Required-rationale dismiss flow.

    Implements C3's no-silent-suppression invariant at the UI layer: the
    dialog cannot complete without a logged rationale. Mirrors the prototype's
    L1022-1049 markup contract.
    """
    st.markdown(
        f'<p style="font-size:0.875rem;color:var(--text-secondary);margin:0 0 16px;">'
        f'Dismissing A-{sample_idx:04d} ({tier}) requires a recorded reason. Your operator '
        f'ID, timestamp, and rationale persist to the audit log and export in FDA-record format.</p>',
        unsafe_allow_html=True,
    )
    category = st.radio(
        "Reason category",
        ["False positive", "Scheduled maintenance", "Known vendor activity", "Other"],
        horizontal=False,
        key=f"dismiss_cat_{sample_idx}",
    )
    rationale = st.text_area(
        "Rationale · required",
        placeholder="What did your investigation find? What did you confirm and with whom?",
        height=100,
        key=f"dismiss_rat_{sample_idx}",
    )
    cancel_col, confirm_col = st.columns(2)
    with cancel_col:
        if st.button("Cancel", key=f"dismiss_cancel_{sample_idx}", width="stretch"):
            st.rerun()
    with confirm_col:
        if st.button("Confirm dismissal",
                     key=f"dismiss_confirm_{sample_idx}",
                     type="primary",
                     width="stretch"):
            if not rationale.strip():
                st.error("Rationale is required. Dismissal not recorded.")
                return
            _capture_dashboard_action(
                sample_idx, "dismiss",
                details={"tier": tier, "role": role, "category": category, "rationale": rationale.strip()},
            )
            st.toast(f"Dismissed A-{sample_idx:04d} · audit recorded", icon="📝")
            st.rerun()


def _triage_status_strip(ui):
    """Fixed-position status strip footer (prototype L996-1019)."""
    latency = load_latency_profile()
    p95_ms = ""
    if isinstance(latency, dict):
        stages = latency.get("stages") or latency
        if isinstance(stages, dict):
            agg = stages.get("module4") or stages.get("aggregate") or {}
            if isinstance(agg, dict):
                p95 = agg.get("p95_ms") or agg.get("p95")
                if p95 is not None:
                    p95_ms = f"{float(p95):.0f}ms"

    build = "feature/dashboard-design"
    try:
        import subprocess
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=2, check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            build = f"feature/dashboard-design@{out.stdout.strip()}"
    except Exception:
        pass

    metrics = {
        "system": "System nominal",
        "p95_ms": p95_ms or "—",
        "build": build,
    }
    st.markdown(ui.render_status_strip(metrics), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# 6.3b  Online Simulation Mode
# ═══════════════════════════════════════════════════════════════════════


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

    st.title("IoMT IDS — Online Simulation")

    responses = load_all_responses()
    clin_summaries = load_clinician_summaries()
    audit_trail = load_audit_trail()
    latency_profile = load_latency_profile()
    live_df = load_live_stream_source()

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

    # ── Smoother playback controls ──
    ctrl_a, ctrl_b, ctrl_c, ctrl_d, ctrl_e = st.columns([1.2, 1, 1, 1, 1.4])

    with ctrl_a:
        speed_label = st.selectbox(
            "Speed",
            ["0.5x", "1x", "2x", "4x"],
            index=["0.5x", "1x", "2x", "4x"].index(f"{st.session_state.sim_speed:g}x")
            if f"{st.session_state.sim_speed:g}x" in ["0.5x", "1x", "2x", "4x"]
            else 1,
            help="Playback speed multiplier for the auto-advance loop.",
        )
        st.session_state.sim_speed = float(speed_label.rstrip("x"))

    with ctrl_b:
        if st.session_state.sim_running:
            if st.button("⏸ Pause", width="stretch"):
                st.session_state.sim_running = False
                audit_log("sim_pause", sim_index=st.session_state.sim_index)
        else:
            if st.button("▶ Resume", width="stretch"):
                st.session_state.sim_running = True
                audit_log("sim_resume", sim_index=st.session_state.sim_index)

    with ctrl_c:
        if st.button("⏭ Step", width="stretch", help="Advance one alert (works while paused)."):
            st.session_state.sim_index = min(st.session_state.sim_index + 1, len(responses) - 1)
            push_latency_sample(latency_profile)

    with ctrl_d:
        if st.button("⟲ Reset", width="stretch"):
            st.session_state.sim_index = 0
            st.session_state.latency_history.clear()
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

    # ── System health panels (UX-S-01: collapsed to reduce triage distraction) ──
    st.markdown("---")
    with st.expander("\U0001f4c8 System Health (Latency / Threshold / Drift)", expanded=False):
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

        st.markdown("---")
        col_thresh, col_drift = st.columns(2)
        with col_thresh:
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
                    st.image(thresh_bytes, width="stretch", caption="DAE threshold: static vs adaptive")
            else:
                st.info("Run `dynamic_threshold_sim.py` to enable adaptive threshold monitoring")

        with col_drift:
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
        _acc = st.session_state.setdefault("_sim_acc", {
            "idx": -1,
            "tier": {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0},
            "attacks": 0,
        })
        if idx_local < _acc["idx"]:
            # Jumped backward — rebuild from scratch up to idx_local
            _acc["tier"] = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
            _acc["attacks"] = 0
            for _r in responses[:idx_local + 1]:
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
                    clin = clin_summaries.get(sample_idx, {})
                    alerts_cache[sample_idx] = process_alert(
                        sample_idx,
                        {
                            "risk_score": score,
                            "risk_level": level,
                            "attack_category": r.get("attack_category", "unknown"),
                            "xai_explanation": {
                                "xgboost_top_features": r.get("explanation", {})
                                .get("analyst", {})
                                .get("xgboost_top_features", []),
                                "dae_top_features": r.get("explanation", {})
                                .get("analyst", {})
                                .get("dae_top_features", []),
                                "clinician_summary": clin.get("summary", ""),
                                "consensus": r.get("explanation", {})
                                .get("analyst", {})
                                .get("consensus", ""),
                            },
                        },
                    )
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

                if sample_idx not in fda_cache:
                    fda_record = build_fda_record_for_alert(sample_idx, r, audit_trail)
                    fda_cache[sample_idx] = json.dumps(fda_record, indent=2).encode("utf-8")
                    fda_filename_cache[sample_idx] = f"audit_{fda_record['alert_id']}.json"
                st.download_button(
                    label="⬇ Export FDA-style Audit Record",
                    data=fda_cache[sample_idx],
                    file_name=fda_filename_cache[sample_idx],
                    mime="application/json",
                    key=f"fda_{sample_idx}",
                    help=(
                        "Download this alert as a Module-5 FDA-style audit "
                        "record (alert_id, timestamp, risk, actions, "
                        "rationale, simulated outcome, integrity hash)."
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
            new_level = responses[i]["risk_level"]   # O(1) direct access
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
            decision_time=elapsed,
        )
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

    # UX-B-01: Action affordance — show recommended action
    st.divider()
    st.subheader("\u26a1 Recommended Action")
    correct_action = alert.get("correct_action", "")
    _ACTION_GUIDANCE = {
        "isolate": ("\U0001f534 Isolate device from network",
                    "Block all non-essential connections while preserving clinical paths."),
        "escalate": ("\U0001f7e0 Escalate immediately",
                     "Notify security lead and clinical engineering on-call."),
        "investigate": ("\U0001f7e1 Investigate before acting",
                        "Gather more information. Check with Biomed for scheduled maintenance."),
        "monitor": ("\U0001f7e2 Monitor — no immediate action",
                    "Watch for escalation. Set alert for threshold change."),
        "dismiss": ("\u26aa Dismiss — expected behavior",
                    "Verify with asset owner. Document reason for dismissal."),
    }
    label, guidance = _ACTION_GUIDANCE.get(
        correct_action,
        ("\u2139\ufe0f Review recommended",
         "Check response policy for this alert type."),
    )
    st.markdown(f"**{label}**")
    st.caption(guidance)


def _render_proxy_questions():
    """
    Q21 + Q22: proxy validation for clinical staff
    and management stakeholders.
    Shown once after all 20 alerts are completed.
    """
    st.title("Two Final Questions")
    st.markdown(
        "Based on the alerts you reviewed, "
        "please answer these two questions."
    )

    with st.form("proxy_questions"):

        st.markdown("#### Q21 — Clinical Staff")
        q21 = st.radio(
            "If you forwarded one of these alerts to a nurse "
            "or physician, would they have enough information "
            "to understand the patient safety risk?",
            ["Yes — the information is clear for clinical staff",
             "Partially — some alerts were clear, others were not",
             "No — clinical staff would need more explanation"],
            index=1
        )
        q21_note = st.text_input(
            "What was missing for clinical staff? (optional)",
            placeholder="e.g. patient impact was unclear, too technical..."
        )

        st.markdown("---")
        st.markdown("#### Q22 — Management / Security Lead")
        q22 = st.radio(
            "If you reported these alerts to your manager "
            "or security lead, would the information be "
            "sufficient to justify your recommended action?",
            ["Yes — the explanation justifies the action clearly",
             "Partially — for some alerts yes, others no",
             "No — I would need to add more context myself"],
            index=1
        )
        q22_note = st.text_input(
            "What additional context would management need? (optional)",
            placeholder="e.g. business impact unclear, risk level hard to explain..."
        )

        if st.form_submit_button("Submit & Complete Study",
                                 type="primary",
                                 use_container_width=True):
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
            audit_log("proxy_questions_submitted",
                     participant_id=st.session_state.participant_id,
                     q21=q21, q22=q22)
            st.rerun()


_SEV_COLORS = {
    "CRITICAL": "#d32f2f", "HIGH": "#f57c00",
    "MEDIUM": "#1976d2", "LOW": "#388e3c",
}

# Issue 10 fix: compile patterns once at module load instead of running
# repeated `in` substring scans on every line of every Group B render.
import re as _re

_DO_NOT_RE = _re.compile(r"DO NOT", _re.IGNORECASE)
# Matches "SEVERITY: CRITICAL", "► HIGH", "SEVERITY HIGH" etc.
_SEV_LINE_RE = _re.compile(
    r"(?:SEVERITY[:\s]+|►\s*)(CRITICAL|HIGH|MEDIUM|LOW)", _re.IGNORECASE
)


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
                f'padding:4px 10px;border-radius:4px;'
                f'font-family:monospace;margin:2px 0;">'
                f'{line.strip()}</div>',
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
    from module6_evaluation.study_loader import (
        load_study_alerts, assign_ab_condition
    )

    # ── Registration ──────────────────────────────────────────
    if not st.session_state.study_started:
        st.title("Healthcare IDS Alert Evaluation Study")
        st.markdown("""
        **Purpose:** Evaluate how security alert information helps
        IT staff make response decisions.

        **Time required:** 30–40 minutes

        **What you will do:** Review 20 security alerts and decide
        how to respond to each one.
        """)

        with st.form("registration"):
            pid = st.text_input(
                "Participant ID",
                placeholder="e.g. P01, P02 ...",
                help="Assigned by researcher"
            )
            role = st.selectbox(
                "Your current role",
                ["IT Security Generalist",
                 "Network/System Administrator",
                 "Healthcare IT Support",
                 "Other IT Role"]
            )
            years_exp = st.slider(
                "Years in current role", 1, 15, 3
            )
            has_ids_exp = st.radio(
                "Have you worked with IDS/SIEM alerts before?",
                ["Yes", "No"]
            )
            consent = st.checkbox(
                "I agree to participate in this research study "
                "and understand my responses will be anonymized."
            )

            if st.form_submit_button("Begin Study") and pid and consent:
                st.session_state.participant_id = pid
                st.session_state.participant_role = role
                st.session_state.participant_years = years_exp
                st.session_state.participant_ids_exp = has_ids_exp
                st.session_state.study_started = True
                st.session_state.current_alert = 0
                st.session_state.responses = []
                st.session_state.alert_start_time = time.time()
                st.session_state.study_alerts = load_study_alerts()
                audit_log("study_start",
                         participant_id=pid,
                         role=role,
                         years=years_exp,
                         ids_exp=has_ids_exp)
                st.rerun()
        return

    # ── Study complete ─────────────────────────────────────────
    if st.session_state.study_complete:
        st.title("Study Complete")
        st.success(f"Thank you for participating!")

        responses = st.session_state.responses
        n = len(responses)

        # Save responses
        save_path = (
            PROJECT_ROOT / "results" / "reports" /
            f"study_responses_{st.session_state.participant_id}.json"
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(
            json.dumps(responses, indent=2), encoding="utf-8"
        )

        st.metric("Alerts Reviewed", n)
        st.info(f"Your responses have been saved. "
                f"Results will be shared after the study concludes.")

        audit_log("study_complete",
                 participant_id=st.session_state.participant_id,
                 n_responses=n)
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
    st.progress(progress,
                text=f"Alert {current_idx + 1} of {n_total}")

    # ── Alert display ──────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"### Alert {current_idx + 1}")
    st.markdown(
        "_You are the on-call IT security staff at a 300-bed hospital. "
        "Review the alert below and decide how to respond._"
    )

    # Show Group A or Group B content
    if show_mve:
        _render_group_b_highlighted(alert.group_b_display)
    else:
        st.code(alert.group_a_display, language=None)

    # ── Response form ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Your Decision")

    with st.form(f"alert_form_{current_idx}"):

        severity = st.radio(
            "1. How severe is this alert? *(select one)*",
            ["CRITICAL — Respond immediately",
             "HIGH — Respond within 1 hour",
             "MEDIUM — Respond within 4 hours",
             "LOW — Review within 24 hours"],
            index=None,
        )

        action = st.radio(
            "2. What action would you take? *(select one)*",
            ["Isolate the device/system from the network",
             "Escalate to clinical staff / senior management",
             "Investigate further before taking action",
             "Monitor closely but no immediate action",
             "Dismiss — this is likely a false alarm"],
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
                5: "5 — Very confident"
            }[x]
        )

        submitted = st.form_submit_button(
            "Submit & Next Alert →",
            type="primary",
            use_container_width=True
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
            severity_correct = (chosen_severity == alert.correct_severity)
            action_correct = (chosen_action == alert.correct_action)

            # Partial credit for severity
            LEVEL = {"CRITICAL": 3, "HIGH": 2, "MEDIUM": 1, "LOW": 0}
            sev_diff = abs(
                LEVEL.get(chosen_severity, -1) -
                LEVEL.get(alert.correct_severity, -1)
            )
            severity_score = 1.0 if sev_diff == 0 else (
                0.5 if sev_diff == 1 else 0.0
            )
            catastrophic = (sev_diff == 3)  # CRITICAL↔LOW mismatch

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
            st.session_state.alert_start_time = time.time()

            audit_log("alert_response",
                     participant_id=pid,
                     alert_id=alert.alert_id,
                     condition=response["condition"],
                     composite_score=composite_score,
                     decision_time=elapsed)
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════


def pcap_replay_stub():
    """6C.10 — PCAP replay placeholder (optional, future work)."""
    st.title("\U0001f4e6 PCAP Replay")

    st.info(
        "**Phase 3 Feature \u2014 Not yet implemented**\n\n"
        "This module will allow upload of raw .pcap / .pcapng "
        "network capture files for offline replay through the "
        "full IoMT IDS pipeline (DAE anomaly detection \u2192 "
        "risk scoring \u2192 MVE explanation generation).\n\n"
        "**Planned for:** Phase 3 (hospital pilot deployment)"
    )

    st.markdown("#### Planned capabilities:")
    st.markdown(
        "- Upload PCAP files from network taps or span ports\n"
        "- Replay packet-by-packet through the detection pipeline\n"
        "- Generate MVE explanations for each detected anomaly\n"
        "- Export audit trail in FDA-compatible format\n"
        "- Compare replay results against known attack signatures"
    )

    st.caption(
        "For live demo: use the **Online Simulation** page which "
        "replays pre-processed test data through the same pipeline."
    )


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
