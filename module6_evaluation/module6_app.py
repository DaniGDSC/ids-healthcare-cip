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


# When invoked via `streamlit run module6_evaluation/module6_app.py`
# the project root is NOT on sys.path (streamlit treats the file as a
# script, not a package). Prepend it so the absolute import below works.
# C5 follow-up: kept for the `streamlit run` invocation path; the package
# itself is importable without this because of __init__.py.
_PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT_FOR_IMPORT))

# Lazy hardened audit (Y5/Y8 fix) — was eagerly constructed at import time.
from module6_evaluation.audit_writer import get_hardened_audit  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "results/reports"
CHARTS_DIR = PROJECT_ROOT / "results/charts"
MODELS_DIR = PROJECT_ROOT / "results/models"

# Dataset routing + suffix files now live in module6_evaluation.constants
# (re-imported below). PAGE_SPLIT carries the test=paper-clean /
# demo=operator-clean per-page routing the legacy code documented here.


# Constants + suffix resolution now live in module6_evaluation.constants.
# Re-exported here for back-compat with any callers reading them off the
# app module directly.
from module6_evaluation.constants import (  # noqa: E402
    PAGE_SPLIT,
    _SPLIT_FILES,
    resolve_suffix as _resolve_suffix,
)

# Role names + tier colors + action priority maps live in
# module6_evaluation.constants. Re-imported here for back-compat with any
# caller (test or module) still grabbing them off the app module path.
from module6_evaluation.constants import (  # noqa: E402, F401
    ACTIONS,
    DETECTOR_CONSENSUS_LABEL,
    ROLE_DISPLAY_LIST,
    ROLE_DISPLAY_NAMES,
    ROLE_INTERNAL_KEY,
    ROLE_ORDER,
    ROLE_SHORT_LABELS,
    ROLES,
    TIER_COLORS,
    TIER_STREAMLIT_COLORS,
)


def _parse_consensus(consensus_str: str) -> tuple[int, int] | None:
    """Parse a Module 4 consensus string into (n_flagged, total).

    Format produced by `module4_explanations.py:772` is `"N/M models flagged"`.
    Returns None if the string doesn't match (e.g. empty or unexpected
    legacy shape) — callers should treat None as "consensus unavailable".
    """
    if not consensus_str:
        return None
    try:
        head = consensus_str.split(" ", 1)[0]  # "N/M"
        n_str, m_str = head.split("/")
        return int(n_str), int(m_str)
    except (ValueError, IndexError):
        return None


def _analyst_state(alert: dict) -> str:
    """Return analyst-data availability state for a rendered alert.

    Three states drive consistent empty-state messaging across the role
    renderers:
      * "available"   — analyst data joined and consensus present
      * "pending"     — flag says analyst_available but the join missed
                        (pipeline mismatch — Module 4/5 desynced)
      * "unavailable" — Module 4 didn't process this sample
    """
    consensus = alert.get("consensus") or ""
    if consensus:
        return "available"
    expl = alert.get("explanation") or {}
    flag = bool(expl.get("analyst_available")) if isinstance(expl, dict) else False
    return "pending" if flag else "unavailable"

from common.phi import BIOMETRIC_COLUMNS as BIOMETRIC_FEATURES  # noqa: E402

# Wires the dashboard's per-alert processing to the research prototype's
# Risk-Adaptive Scoring Engine (research_spec.yaml component_2) so tier
# assignment uses the same logic the prototype tests enforce (M7, M6).
from module6_evaluation._src_adapter import scored_from_eval_alert  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════
# MVE Display Helpers — all action/device/tier maps now live in
# module6_evaluation.constants; re-exported below for back-compat.
# ═══════════════════════════════════════════════════════════════════════

from module6_evaluation.constants import (  # noqa: E402, F401
    _ACTION_DISPLAY,
    _ACTION_DISPLAY_MISS,
    _ACTION_PRIORITY,
    _CATEGORY_TO_DEVICE,
    _CRIT_COLOR_HEX,
    _PA_MAP,
)


def render_device_criticality(alert: dict) -> None:
    """Gap 2 + UX-X-02: Render device class criticality badge + context.

    BA-15: badge now uses Sentinel tier tokens (`--tier-*-bg`) when the
    criticality maps to a known tier, instead of a hardcoded hex that
    silently diverged from the Dashboard's palette. Falls back to a neutral
    surface when criticality isn't a tier label.
    """
    criticality = str(alert.get("device_criticality", "")).upper()
    if not criticality or criticality not in _CRIT_COLOR_HEX:
        criticality = str(alert.get("risk_level", "UNKNOWN")).upper()

    # Prefer Sentinel tier token; fall back to a neutral surface for
    # non-tier strings (e.g. "ROUTINE", "UNKNOWN").
    if criticality in TIER_STREAMLIT_COLORS:
        token = criticality.lower()
        bg = f"var(--tier-{token}-bg)"
        fg = f"var(--tier-{token})"
        border = f"1px solid {fg}"
    else:
        bg = "var(--surface-2)"
        fg = "var(--text-secondary)"
        border = "1px solid var(--border)"

    # FIX B: infer device class when missing
    device_cls, was_inferred = infer_device_class(alert)
    affected = alert.get("affected_system", "")

    st.markdown(
        f'<span style="background:{bg};color:{fg};border:{border};'
        f'padding:4px 12px;border-radius:4px;font-weight:500;'
        f'font-family:JetBrains Mono,monospace;font-size:12px;'
        f'letter-spacing:0.04em;">'
        f'Device · {criticality}</span>',
        unsafe_allow_html=True,
    )
    if device_cls or affected:
        st.caption(f"{device_cls}{' — ' + affected if affected else ''}")
    if was_inferred:
        st.caption(
            f"Device class inferred from attack category "
            f"({alert.get('attack_category', '?')}) — asset inventory lookup unavailable."
        )

    # UX-X-01: Patient impact warning
    impact = alert.get("patient_care_impact", "")
    active = alert.get("active_device", False)
    if active and impact:
        st.warning(f"Active device — {impact}")
    elif impact:
        st.info(impact)


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

    # MVE payload (Option 4) — nested under explanation.mve when Module 5
    # generated one. layer_1/layer_2/layer_3 sub-dicts contain string fields
    # the existing _get() lookup chain expects (baseline_behavior, severity,
    # immediate_action, ...); `mve` itself carries the joined why_anomalous.
    mve = (expl.get("mve") if isinstance(expl, dict) else None) or {}
    mve_l1 = mve.get("layer_1") or {}
    mve_l2 = mve.get("layer_2") or {}
    mve_l3 = mve.get("layer_3") or {}

    # Merge once — O(1) view construction, O(1) per key lookup thereafter.
    # MVE sub-dicts come AFTER the source dicts so an explicit alert-level
    # field still wins; they come BEFORE the implicit fallback to legacy
    # clinician_summary inside _get's key list.
    _cm = ChainMap(
        alert,
        xai  if isinstance(xai,  dict) else {},
        expl if isinstance(expl, dict) else {},
        resp if isinstance(resp, dict) else {},
        mve_l1 if isinstance(mve_l1, dict) else {},
        mve_l2 if isinstance(mve_l2, dict) else {},
        mve_l3 if isinstance(mve_l3, dict) else {},
        mve    if isinstance(mve,    dict) else {},
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

    # BA-9: Layer headers without emoji. Layer numbers + serif typography
    # already establish the hierarchy; emojis render differently across
    # browsers and force-vocalize on screen readers.
    with st.expander("Layer 1 \u00b7 Why anomalous", expanded=True):
        if l1:
            st.write(l1)
            if consensus:
                # Unified label + visual treatment (T10). When the consensus
                # string is parseable as "N/M models flagged" we render the
                # visual badge; otherwise fall back to a caption with the
                # canonical label so all 4 surfaces use the same wording.
                parsed = _parse_consensus(consensus)
                if parsed:
                    from module6_evaluation import components as _ui
                    n_flagged, total = parsed
                    st.markdown(
                        _ui.render_consensus_badge(
                            n_flagged, total, label=DETECTOR_CONSENSUS_LABEL,
                        ),
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption(f"{DETECTOR_CONSENSUS_LABEL}: {consensus}")
        else:
            # BA-17: empty-state with explicit styling \u2014 was a quiet caption
            # that operators tended to skip past.
            st.info(
                "Baseline deviation detected, but no narrative explanation is "
                "available for this alert. See SHAP features below for the "
                "feature-level signal."
            )

    # ── Layer 2: Clinical Severity ──
    affected = _get("affected_system")
    impact = _get("patient_care_impact")
    severity = _get("severity_label", "severity", "risk_level", "tier")
    device_tier = _get("device_tier", "device_class")

    with st.expander("Layer 2 \u00b7 Clinical severity", expanded=True):
        if severity:
            # BA-15: use Sentinel tier token instead of inline hex from
            # TIER_COLORS. Severity may arrive as either tier name
            # (CRITICAL/HIGH/...) or a free-form label; only token-map the
            # former.
            sev_upper = severity.upper()
            if sev_upper in TIER_STREAMLIT_COLORS:
                token = sev_upper.lower()
                st.markdown(
                    f"**Severity:** <span style='color:var(--tier-{token});"
                    f"font-family:JetBrains Mono,monospace;font-weight:500;'>"
                    f"{severity}</span>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(f"**Severity:** {severity}")
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

    with st.expander("Layer 3 \u00b7 Recommended action", expanded=True):
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
    get_hardened_audit().log(
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
        # Per-model ensemble breakdown — surfaced by render_model_breakdown.
        # Empty dict when Module 4 didn't process this sample (analyst
        # unavailable). Pulled from the joined analyst_report payload.
        "models": xai.get("models", {}) or {},
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


def _render_provider_badge(alert: dict) -> None:
    """Path B · commit 4 — surface a degradation banner when the MVE came
    from the rule-based fallback (Mode B) instead of an LLM (Mode A).

    Reads ``alert["mve"]["provider"]`` (or legacy ``alert["mve_provider"]``).
    Anchors §3.4's "degradation badge" claim in code an operator sees at
    triage time.
    """
    mve = alert.get("mve") or {}
    provider = mve.get("provider") or alert.get("mve_provider")
    if provider == "rule_based":
        st.warning(
            "⚠ Rule-based explanation (LLM unavailable) — "
            "verify alert details independently before acting."
        )
    elif provider in ("openai", "anthropic"):
        st.caption(f"Explanation provider: {provider}")


def render_analyst(alert: dict):
    """Analyst view: SHAP plots + feature table + classification detail.

    Consensus + per-model breakdown now leads the render so the operator
    sees the ensemble agreement signal *before* diving into any one
    model's SHAP. Empty-state messaging is gated by `_analyst_state` so a
    sample with no analyst data renders an explicit caption rather than
    silently empty rows.
    """
    from module6_evaluation import components as ui_mod

    st.markdown(f"#### {ROLE_DISPLAY_NAMES['analyst']} view")

    # Gap 2: Device criticality badge
    render_device_criticality(alert)

    # Path B · commit 4 — degradation banner when MVE used the rule-based fallback
    _render_provider_badge(alert)

    # Consensus + per-model breakdown surfaced at the top of the analyst
    # view — this is the meta-explanation that should frame how to read
    # the individual SHAP plots below. Three-state handling (T6):
    state = _analyst_state(alert)
    if state == "available":
        parsed = _parse_consensus(alert.get("consensus", ""))
        if parsed:
            n_flagged, total = parsed
            st.markdown(
                ui_mod.render_consensus_badge(
                    n_flagged, total, label=DETECTOR_CONSENSUS_LABEL,
                ),
                unsafe_allow_html=True,
            )
        models = alert.get("models") or {}
        if models:
            st.markdown(ui_mod.render_model_breakdown(models), unsafe_allow_html=True)
    elif state == "pending":
        st.info(
            "Analyst payload pending — `analyst_available=True` on this "
            "alert but no joined entry from `analyst_report.json`. "
            "Re-run `module4_explanations` to refresh the demo split."
        )
    else:  # unavailable
        st.caption(
            "Analyst data not generated for this sample (Module 4 didn't "
            "process it; SHAP charts below may also be missing)."
        )

    # SHAP waterfall
    idx = alert.get("sample_index", 0)
    chart_bytes = _cached_png_bytes(str(CHARTS_DIR / f"waterfall_xgboost_sample_{idx:04d}.png"))
    if chart_bytes:
        st.image(chart_bytes, caption="SHAP Waterfall", width="stretch")

    # Force plot
    force_bytes = _cached_png_bytes(str(CHARTS_DIR / f"force_xgboost_sample_{idx:04d}.png"))
    if force_bytes:
        st.image(force_bytes, caption="SHAP Force Plot", width="stretch")

    # Top features table — only when join landed data.
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


def render_clinician(alert: dict):
    """Clinician view: plain-language NLG summary + biometric safety notes."""
    st.markdown(f"#### {ROLE_DISPLAY_NAMES['clinician']} view")

    # Gap 2: Device criticality badge
    render_device_criticality(alert)

    # Path B · commit 4 — degradation banner when MVE used the rule-based fallback
    _render_provider_badge(alert)

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
        # Informational note — biometric features being implicated is
        # context for the clinician's decision, not an error. The previous
        # st.error (red) overstated the signal and contributed to alarm
        # fatigue.
        st.warning(
            f"Patient safety note · biometric features in the alert: "
            f"{', '.join(bio_feats)}"
        )
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
    from module6_evaluation import components as ui_mod

    st.markdown(f"#### {ROLE_DISPLAY_NAMES['administrator']} view")

    # Gap 2: Device criticality badge
    render_device_criticality(alert)

    # Path B · commit 4 — degradation banner when MVE used the rule-based fallback
    _render_provider_badge(alert)

    col1, col2, col3 = st.columns(3)
    col1.metric("Risk Score", f"{alert.get('risk_score', 0):.3f}")
    col2.metric("Tier", alert.get("tier", "N/A"))
    col3.metric("Category", alert.get("attack_category", "N/A"))

    # Detector consensus — admin doesn't need the full per-model breakdown
    # (that's analyst territory), just the badge for situational awareness.
    state = _analyst_state(alert)
    if state == "available":
        parsed = _parse_consensus(alert.get("consensus", ""))
        if parsed:
            n_flagged, total = parsed
            st.markdown(
                ui_mod.render_consensus_badge(
                    n_flagged, total, label=DETECTOR_CONSENSUS_LABEL,
                ),
                unsafe_allow_html=True,
            )

    # Recommended action — consensus is rendered as a badge above by
    # the new render_consensus_badge component; the plain `**Detector
    # consensus:** ...` text row was the T3 stopgap fix and is now
    # superseded.
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


_ENRICH_KEYS = (
    "device_class", "device_criticality", "affected_system",
    "patient_care_impact", "active_device", "correct_action",
)


def _enrich_with_device_context(responses: list, split: str | None = None) -> list:
    """Join responses with evaluation_alerts{suffix}.json for device-context fields.

    ``split`` selects the split-specific curated file. Test = paper-clean
    (legacy unsuffixed); demo = operator-clean (suffix '_demo'). When
    ``None`` (legacy callers), falls back to the test-suffixed file —
    safe for the dashboard Test page, dangerous if a demo-side caller
    forgets to pass split because device_class is derived from per-row
    biometric features and so is sample-specific to the split.

    Mutates `responses` in place AND returns it. Tolerates a missing
    evaluation_alerts file silently (degrades to no enrichment).
    """
    suffix = _resolve_suffix(split)
    eval_path = EVAL_DIR / f"evaluation_alerts{suffix}.json"
    if not eval_path.exists():
        return responses
    with open(eval_path) as f:
        eval_alerts = {a["sample_index"]: a for a in json.load(f)}
    for r in responses:
        ea = eval_alerts.get(r.get("sample_index"))
        # Issue 5 fix: guard with `if ea` before iterating — avoids creating
        # a throwaway {} on every miss via .get(key, {}).
        if ea:
            for k in _ENRICH_KEYS:
                if k in ea and k not in r:
                    r[k] = ea[k]
    return responses


@st.cache_data
def load_responses_for(split: str | None) -> list:
    """Load `alert_responses_<split>.json` (or legacy `alert_responses.json`
    for test) enriched with device-context fields.

    `split` MUST be a key in `_SPLIT_FILES` (either "test" or "demo"), OR
    `None` for pages that don't use parquet-derived alerts (Study / PCAP).
    Any other value raises RuntimeError — guard against accidental reads
    from a non-frozen split.

    Accepts BOTH the new envelope shape
    ``{"_provenance": {...}, "records": [...]}`` and the legacy
    bare-list shape (returned as-is). When the envelope is present,
    the records are validated against
    :class:`common.alert_response_schema.AlertResponsesEnvelope` and a
    clean ``st.error`` + ``st.stop`` replaces what would otherwise be a
    ``KeyError`` 200 lines deep inside a Streamlit render.
    """
    from pydantic import ValidationError

    from common.alert_response_schema import AlertResponsesEnvelope

    if split is None:
        return []
    if split not in _SPLIT_FILES:
        raise RuntimeError(
            f"Refusing to load alert_responses for split={split!r}: must be "
            f"one of {sorted(_SPLIT_FILES)} or None."
        )
    suffix = _SPLIT_FILES[split]
    path = EVAL_DIR / f"alert_responses{suffix}.json"
    if not path.exists():
        return []
    with open(path) as f:
        raw = json.load(f)

    if isinstance(raw, list):
        # Legacy bare-list shape — pre-envelope artefact. No provenance,
        # no schema validation. Kept for backward compatibility with
        # demo files that haven't been regenerated yet.
        responses = raw
    elif isinstance(raw, dict) and "records" in raw:
        try:
            envelope = AlertResponsesEnvelope.model_validate(raw)
        except ValidationError as e:
            st.error(
                f"Schema mismatch in {path.name}:\n\n{e}\n\n"
                f"Run `python -m module5_responses.module5_responses "
                f"--split={split}` to regenerate."
            )
            st.stop()
        responses = [r.model_dump() for r in envelope.records]
    else:
        st.error(
            f"{path.name} is neither a bare list nor an envelope "
            f"({{'_provenance': ..., 'records': ...}}). Unknown shape."
        )
        st.stop()

    return _enrich_with_device_context(responses, split=split)


@st.cache_data
def load_simulation_stream(split: str | None) -> list:
    """Cached wrapper around
    :func:`module6_evaluation.loaders.load_simulation_stream_inner`.

    Used by the Online Simulation "Full stream" data-source mode — the
    array has one entry per arrival timestamp (1632 for demo / 2448 for
    test). NORMAL rows carry ``alert=None``; LOW+ rows embed the M5
    payload. Empty list when the artefact is missing (operator told to
    run :mod:`tools.build_simulation_stream`).
    """
    from module6_evaluation.loaders import (
        LoaderError, load_simulation_stream_inner,
    )
    try:
        return load_simulation_stream_inner(split)
    except LoaderError as e:
        st.error(
            f"{e}\n\nRun `python -m tools.build_simulation_stream "
            f"--split={split}` to rebuild."
        )
        st.stop()


@st.cache_data
def load_simulation_stream_meta(split: str | None) -> dict | None:
    """Cached wrapper for the simulation_stream ``_meta`` block.

    Returns the dict with split label, counts, and stream anchor — or
    None when the artefact is absent.
    """
    from module6_evaluation.loaders import load_simulation_stream_meta_inner
    return load_simulation_stream_meta_inner(split)


@st.cache_data
def load_provenance_for(split: str | None) -> dict | None:
    """Return the ``_provenance`` block from an envelope-formatted
    responses file, or ``None`` if the file is legacy bare-list shape.

    Used by the freshness banner in :func:`dashboard_mode` to detect
    when an upstream artefact (risk_scores.npz, analyst_report.json,
    test_phase1.parquet, clinician_summaries.json) has been
    regenerated since this responses file was built.
    """
    if split is None or split not in _SPLIT_FILES:
        return None
    suffix = _SPLIT_FILES[split]
    path = EVAL_DIR / f"alert_responses{suffix}.json"
    if not path.exists():
        return None
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "_provenance" in raw:
        return raw["_provenance"]
    return None


def _render_freshness_banner(split: str) -> None:
    """Warn the operator if upstream inputs are newer than this file.

    Compares each input's recorded ``mtime_iso`` (stamped at Module 5
    build time) against the file's live mtime on disk. A 1-second
    tolerance avoids spurious warnings on the same-second rewrite case.
    Silent when provenance is absent (legacy file).
    """
    prov = load_provenance_for(split)
    if prov is None:
        return
    stale = []
    for key, meta in prov.get("inputs", {}).items():
        if meta is None:
            continue
        live = (PROJECT_ROOT / meta["path"])
        if not live.exists():
            continue
        live_mtime = live.stat().st_mtime
        prov_mtime = datetime.fromisoformat(meta["mtime_iso"]).timestamp()
        if live_mtime > prov_mtime + 1.0:
            stale.append(meta["path"])
    if stale:
        st.warning(
            "⚠️ Module 5 outputs are stale. Newer inputs on disk: "
            f"{', '.join(stale)}. Run "
            f"`python -m module5_responses.module5_responses "
            f"--split={prov['split']}` to refresh."
        )


@st.cache_data
def load_all_responses() -> list:
    """Legacy compatibility shim — defaults to the test split.

    Kept so existing call sites that haven't migrated to `load_responses_for`
    continue to work. New code should call `load_responses_for("test"|"demo")`
    explicitly so the data-routing intent is visible at the call site.
    """
    return load_responses_for("test")


def _risk_scores_cache_key() -> tuple[str, int, int, str, int, int] | None:
    """Hash of (npz, meta, sig) mtime+size so cache busts when artefacts change.

    Tier 2 F5: previously the cache was unkeyed so a single malicious
    npz loaded once at server start served every subsequent visitor.
    The key now mixes mtime + size of every file in the signed pair.
    """
    npz = EVAL_DIR / "risk_scores.npz"
    meta = npz.with_suffix(".meta.json")
    sig = meta.with_suffix(meta.suffix + ".sig")
    if not (npz.exists() and meta.exists() and sig.exists()):
        return None
    return (
        str(npz),
        npz.stat().st_size, int(npz.stat().st_mtime_ns),
        str(meta),
        meta.stat().st_size, int(meta.stat().st_mtime_ns),
    )


@st.cache_data(ttl=60)
def _load_risk_scores_cached(cache_key: tuple) -> dict | None:
    """Inner cached loader — key controls cache invalidation.

    Tier 2 F1: load via the verified-pair loader; no allow_pickle.
    Returns a plain dict view so legacy callers continue to work.
    """
    from common.risk_scores_loader import load_risk_scores as _verified_load
    npz_path = EVAL_DIR / "risk_scores.npz"
    art = _verified_load(npz_path)
    return {
        "R": art.R,
        "c_detect": art.c_detect,
        "c_track_a": art.c_track_a,
        "c_track_b": art.c_track_b,
        "d_crit": art.d_crit,
        "s_data": art.s_data,
        "d_clinical_tier": art.d_clinical_tier,
        "y_true": art.y_true,
        "risk_levels": art.risk_levels,
        "schema_version": art.schema_version,
        "formula_version": art.formula_version,
    }


def load_risk_scores() -> dict | None:
    """Verified, hash-keyed risk-score loader for the dashboard.

    Returns None when the pair is absent; raises SignedSidecarError when
    the pair is present but signature verification fails (Tier 2 F1) —
    the dashboard surfaces that as an error rather than silently
    rendering tampered values.
    """
    key = _risk_scores_cache_key()
    if key is None:
        return None
    return _load_risk_scores_cached(key)


@st.cache_data
def load_admin_dashboard() -> dict:
    path = EVAL_DIR / "admin_dashboard.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


@st.cache_data
def load_clinician_summaries(split: str | None = None) -> dict:
    """Load clinician summaries for a frozen split.

    ``split`` MUST be a key in ``_SPLIT_FILES`` (``"test"`` or ``"demo"``)
    or ``None`` (legacy = test, paper-clean). Each page passes its own
    ``PAGE_SPLIT`` value so the operator-clean demo doesn't render
    test-split summaries on top of demo alerts (which silently mis-maps
    by ``sample_index``).
    """
    suffix = _resolve_suffix(split)
    path = EVAL_DIR / f"clinician_summaries{suffix}.json"
    if path.exists():
        with open(path) as f:
            return {s["sample_index"]: s for s in json.load(f)}
    return {}


@st.cache_data
def load_analyst_report_for(split: str | None = None) -> dict:
    """Load Module 4 analyst reports for a frozen split.

    Module 5's `alert_responses{suffix}.json` only carries the boolean
    `analyst_available` flag — the full `consensus / models / top_features`
    payload is dropped during serialization. This loader fills that gap by
    reading `analyst_report{suffix}.json` directly, returning a dict keyed
    by `sample_index` so the simulation playhead can join the two streams
    at render time.

    Returns `{}` when the file is missing (graceful fallback — UI is
    expected to handle the analyst-unavailable case anyway).
    """
    suffix = _resolve_suffix(split)
    path = EVAL_DIR / f"analyst_report{suffix}.json"
    if path.exists():
        with open(path) as f:
            return {a["sample_index"]: a for a in json.load(f)}
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
def load_audit_trail(split: str | None = None) -> dict:
    """Module 5 FDA-style audit records, keyed by sample index parsed from alert_id.

    ``split`` MUST be a key in ``_SPLIT_FILES`` (``"test"`` or ``"demo"``)
    or ``None`` (legacy = test). The Online Simulation page passes
    ``"demo"`` so its audit timeline doesn't render test-split FDA
    records on top of demo alerts (mis-mapped by ``sample_index``).
    """
    suffix = _resolve_suffix(split)
    path = EVAL_DIR / f"audit_trail{suffix}.json"
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
    """Mock 'live data source' for Online Simulation — reads the **demo**
    split parquet (per PAGE_SPLIT["Online Simulation"]) and attaches a
    synthetic arrival timestamp per row.

    This simulates a feature-extracted flow stream without requiring a real
    network TAP. Each row is one timestep of mock 'arrived data'; the
    timestamps are anchored to a fixed start instant so the stream is
    reproducible across reruns. Demo (not test) is used because operators
    interact with this page — keeping operator-touch off test protects the
    paper-clean held-out set.
    """
    split = PAGE_SPLIT["Online Simulation"]  # "demo" — single source of truth
    path = PROJECT_ROOT / "data/processed" / f"{split}_phase1.parquet"
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
    # Previously this stuffed the consensus string ("N/4 models flagged")
    # into `explanation_summary`, which is a semantic mismatch — FDA
    # records expect a human-readable narrative, not a model-vote count.
    # Pull the narrative from the clinician NLG or response rationale
    # instead; fall back to empty string if neither is available.
    response = alert.get("response", {}) or {}
    explanation = (
        alert.get("nlg_text")
        or alert.get("clinician_summary")
        or response.get("rationale", "")
        or ""
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
    # Path B · commit 5 — 16-char per-record fingerprint, renamed from
    # ``integrity_hash``. The signed 64-char chained audit hash keeps
    # the ``integrity_hash`` name in ``audit_log.jsonl``.
    record_fingerprint = hashlib.sha256(payload.encode()).hexdigest()[:16]
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
        "record_fingerprint": record_fingerprint,
        "_source": "fallback (audit_trail.json not found)",
    }


# ═══════════════════════════════════════════════════════════════════════
# 6.3c  Dashboard Components
# ═══════════════════════════════════════════════════════════════════════


_TIER_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
_DASHBOARD_PAGE_SIZE = 50


def dashboard_mode():
    """Triage view — three-column Sentinel layout per `docs/sentinel_dashboard.html`.

    D1=A (Streamlit refactor in place). Visual direction locked to the
    prototype; ~85% fidelity envelope. See `docs/dashboard_design_memo.md`
    Phase 3 Plan for the implementation contract.

    Render structure:
      * Theme + data load are unconditional (cheap; CSS is precomputed and
        loaders are @st.cache_data).
      * The three triage columns live inside an @st.fragment so changes to
        the selectbox / filter chips / search field rerun only that scope
        — not the full page, and not the status strip.
      * The status strip renders once per page and stays put.
    """
    from module6_evaluation.sentinel_theme import inject_theme
    from module6_evaluation import components as ui

    inject_theme()

    # P0-1: warn the operator if any Module 5 input has been
    # regenerated since alert_responses.json was built. Silent when
    # provenance is absent (legacy file) — so this is purely additive
    # and safe to ship even before Module 5 has been rerun.
    _render_freshness_banner(PAGE_SPLIT["Dashboard"])

    # Triage page reads the paper-clean test split per PAGE_SPLIT.
    # alert_responses.json (no suffix) is the test-split file produced by
    # `python module5_responses/module5_responses.py --split=test`.
    responses = load_responses_for(PAGE_SPLIT["Dashboard"])
    if not responses:
        st.warning(
            "No test-split alerts found at "
            "`results/reports/alert_responses.json`. "
            "Run `python module5_responses/module5_responses.py --split=test`."
        )
        return

    # Filter / search / page state — initialized once per session.
    st.session_state.setdefault("dashboard_filter", "All")
    st.session_state.setdefault("dashboard_search", "")
    st.session_state.setdefault("dashboard_page", 0)

    @st.fragment
    def _triage_body():
        # Apply filter chips + search to the full response list.
        filtered = _apply_dashboard_filters(responses)

        sorted_resp = sorted(
            filtered,
            key=lambda r: (
                _TIER_ORDER.get(r.get("risk_level", "LOW"), 9),
                -r.get("risk_score", 0.0),
            ),
        )

        page = st.session_state["dashboard_page"]
        start = page * _DASHBOARD_PAGE_SIZE
        end = start + _DASHBOARD_PAGE_SIZE
        visible = sorted_resp[start:end]
        if not visible and page > 0:
            # Filter narrowed past current page — reset to page 0.
            st.session_state["dashboard_page"] = 0
            visible = sorted_resp[:_DASHBOARD_PAGE_SIZE]

        # Counts: total = unfiltered (true situation), filtered = post-filter
        # (drives the queue tier-tiles so they reflect what's visible).
        counts_total = Counter(r.get("risk_level", "LOW") for r in responses)
        counts_filtered = Counter(r.get("risk_level", "LOW") for r in filtered)

        # Pick a sensible default selection if nothing is set yet, or if
        # the previously-selected alert was filtered out.
        if visible:
            default_id = visible[0].get("sample_index")
        else:
            default_id = sorted_resp[0].get("sample_index") if sorted_resp else responses[0].get("sample_index")
        st.session_state.setdefault("selected_alert_id", default_id)

        with st.container(key="sentinel-triage"):
            col_q, col_inv, col_act = st.columns([1.3, 3.0, 1.7], gap="small")
            with col_q:
                sel_id = _triage_queue_column(
                    responses=responses,
                    filtered=filtered,
                    sorted_resp=sorted_resp,
                    visible=visible,
                    counts_total=counts_total,
                    counts_filtered=counts_filtered,
                    page=page,
                    ui=ui,
                )
            # Resolve selection AFTER queue column has run — that's where
            # the selectbox lives. No st.rerun() needed; reading the
            # widget's current value within the same fragment render is
            # the natural Streamlit pattern.
            selected = next(
                (r for r in responses if r.get("sample_index") == sel_id),
                visible[0] if visible else responses[0],
            )
            with col_inv:
                _triage_investigation_column(selected, ui)
            with col_act:
                _triage_actions_column(selected, ui)

    _triage_body()
    _triage_status_strip(ui)


def _apply_dashboard_filters(responses: list) -> list:
    """Apply the active filter chip + search string to the response list.

    Filters:
      * "All" — pass through.
      * "Floor-elevated" — keep only alerts the visual floor proxy flags.
      * "Critical + High" — keep only top-two tiers (operator focus mode).

    Search matches case-insensitively against `attack_category` and the
    formatted alert ID (A-XXXX).
    """
    f = st.session_state.get("dashboard_filter", "All")
    if f == "Floor-elevated":
        responses = [r for r in responses if _floor_elevated(r)]
    elif f == "Critical + High":
        responses = [r for r in responses if r.get("risk_level") in ("CRITICAL", "HIGH")]

    q = (st.session_state.get("dashboard_search") or "").strip().lower()
    if q:
        def _match(r: dict) -> bool:
            cat = (r.get("attack_category") or "").lower()
            aid = f"a-{int(r.get('sample_index', 0)):04d}"
            return q in cat or q in aid
        responses = [r for r in responses if _match(r)]
    return responses


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


def _triage_queue_column(
    *,
    responses,
    filtered,
    sorted_resp,
    visible,
    counts_total,
    counts_filtered,
    page,
    ui,
):
    """Left column: header, tier tiles, filters, search, selector, queue list.

    Returns the currently-selected `sample_index` so the caller can resolve
    the alert dict in the same render pass (no st.rerun needed). The
    selectbox is the source of truth for selection; the rendered queue
    rows are display-only (see sentinel_theme._ALERT_ROW_CSS for why).
    """
    sel_id = st.session_state.get("selected_alert_id")
    total_open = sum(counts_total.values())
    filtered_open = sum(counts_filtered.values())
    total_pages = max(1, (filtered_open + _DASHBOARD_PAGE_SIZE - 1) // _DASHBOARD_PAGE_SIZE)

    # Header: title + "N open" (true total, not the filtered count — this
    # tells the operator how many live alerts are in the system regardless
    # of the current view).
    count_label = f"{total_open} open"
    if filtered_open != total_open:
        count_label = f"{filtered_open} shown · {total_open} open"
    st.markdown(
        f'<div style="padding:16px 16px 4px;">'
        f'  <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:12px;">'
        f'    <h2 class="font-display" style="font-size:1.5rem;margin:0;letter-spacing:-0.02em;color:var(--text-primary);">Active queue</h2>'
        f'    <span class="font-mono" style="font-size:11px;color:var(--text-tertiary);">{count_label}</span>'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # 4-up tier-count tile grid — reflects the unfiltered totals so the
    # operator's situational awareness isn't masked by their own filter.
    tiles = '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px;padding:0 16px 12px;">'
    for tier in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
        tiles += ui.render_tier_count_tile(tier, counts_total.get(tier, 0))
    tiles += '</div>'
    st.markdown(tiles, unsafe_allow_html=True)

    # Filter chips. "Unassigned" is omitted because alert_responses.json
    # carries no owner field; the chip would always match everything.
    st.markdown(
        '<div style="padding:0 16px 6px;font-size:10px;font-weight:500;'
        'letter-spacing:0.08em;text-transform:uppercase;color:var(--text-tertiary);">'
        'Filter'
        '</div>',
        unsafe_allow_html=True,
    )
    st.pills(
        "Filter",
        ["All", "Critical + High", "Floor-elevated"],
        default=st.session_state.get("dashboard_filter", "All"),
        selection_mode="single",
        key="dashboard_filter",
        label_visibility="collapsed",
    )

    # Search input — placeholder mentions ⌘K so the prototype's affordance
    # at least visually carries through. Native browser focus shortcut is
    # not bound (would require JS injection); the affordance is the search
    # box itself.
    st.text_input(
        "Search",
        placeholder="Search attack or A-#### (⌘K)",
        key="dashboard_search",
        label_visibility="collapsed",
    )

    # Selectbox is the actual control. Label is now visible so users
    # understand this is THE selector and the queue below is display.
    options = [r.get("sample_index") for r in visible]
    if not options:
        st.info(
            "No alerts match the current filter and search. "
            "Adjust the chips above or clear the search box."
        )
        return sel_id

    def _fmt(idx):
        a = next(a for a in visible if a.get("sample_index") == idx)
        return f"{a.get('risk_level', 'LOW')[:4]}  ·  A-{idx:04d}  ·  {a.get('attack_category', '?')}"

    # Default selection: if previously-selected alert is still visible
    # under the current filter, keep it. Otherwise fall back to first.
    if sel_id in options:
        default_index = options.index(sel_id)
    else:
        default_index = 0

    chosen = st.selectbox(
        "Select alert (↑/↓ when focused)",
        options,
        index=default_index,
        format_func=_fmt,
        key=f"alert_selectbox_p{page}",
    )
    # Persist selection without forcing a manual rerun — the fragment
    # naturally re-renders on widget change.
    st.session_state["selected_alert_id"] = chosen
    sel_id = chosen

    # Pagination controls — only render when there's >1 page.
    if total_pages > 1:
        st.markdown(
            f'<div style="padding:6px 16px 4px;font-family:JetBrains Mono,monospace;'
            f'font-size:10px;color:var(--text-tertiary);">'
            f'Page {page + 1} of {total_pages}</div>',
            unsafe_allow_html=True,
        )
        pcol1, pcol2 = st.columns(2)
        with pcol1:
            if st.button("← Prev", key="dashboard_prev", width="stretch",
                         disabled=(page == 0)):
                st.session_state["dashboard_page"] = max(0, page - 1)
                st.rerun(scope="fragment")
        with pcol2:
            if st.button("Next →", key="dashboard_next", width="stretch",
                         disabled=(page >= total_pages - 1)):
                st.session_state["dashboard_page"] = min(total_pages - 1, page + 1)
                st.rerun(scope="fragment")

    # Visual queue, grouped by tier — display-only. Cursor + hover removed
    # at the CSS layer (sentinel_theme._ALERT_ROW_CSS) to keep the
    # affordance honest.
    queue_html = ""
    for tier in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
        tier_alerts = [a for a in visible if a.get("risk_level") == tier]
        if not tier_alerts:
            continue
        queue_html += ui.render_tier_header(tier, counts_filtered.get(tier, 0))
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
    return sel_id


def _triage_investigation_column(selected, ui):
    from html import escape

    aid = selected.get("sample_index", 0)
    tier = selected.get("risk_level", "LOW")
    components = selected.get("risk_components", {}) or {}
    floor = _floor_elevated(selected)
    composite = selected.get("risk_score", 0.0)

    # Use the actual raw score if it's been persisted to the response
    # record (e.g. by Module 5 in a future iteration). Otherwise show
    # only the floor *badge* and skip the raw / delta sub-line — fabricating
    # a fixed 0.15 delta misled operators about the policy's real impact.
    raw_risk_real = selected.get("raw_risk_score")
    if raw_risk_real is not None and floor:
        raw_display = float(raw_risk_real)
        delta_display = composite - raw_display
    else:
        raw_display = None
        delta_display = None

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
            raw_risk=raw_display,
            floor_delta=delta_display,
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
        ("Clinical tier",        "D_clinical_tier", "--tier-high",     "active-care"),
    ):
        v = float(components.get(key, 0.0))
        metric_grid += ui.render_metric_with_bar(
            label, v, sub, color, bar_value=v, with_ticks=(key == "C_detect"),
        )
    metric_grid += '</div>'
    st.markdown(metric_grid, unsafe_allow_html=True)

    # Risk-component panel. Renamed "contributions" → "weights" because
    # alert_responses.json carries positive-only component values, not
    # signed SHAP-style contributions. Calling them "contributions" while
    # only showing positive bars over-promises the explanation depth
    # operators should expect from this panel. Real SHAP top-features live
    # on evaluation_alerts and would require the device-context join.
    st.markdown(
        '<div style="padding:24px 32px 8px;">'
        '  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">'
        '    <h3 class="font-display" style="font-size:1.25rem;margin:0;letter-spacing:-0.01em;color:var(--text-primary);">Risk component weights</h3>'
        '    <span class="font-mono" style="font-size:10px;text-transform:uppercase;letter-spacing:0.08em;color:var(--text-tertiary);">6 components</span>'
        '  </div>'
        '  <p style="font-size:11px;color:var(--text-tertiary);margin:0 0 8px;line-height:1.5;">'
        '    Positive-only weights from the Risk-Adaptive Scoring Engine. '
        '    For signed SHAP contributions, see the Online Simulation page.'
        '  </p>'
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
        ("D_clinical_tier", "Clinical tier",          "active-care weight"),
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
    # Short pill labels = IT / Biomed / Nurse (compact) per ROLE_SHORT_LABELS.
    # Maps internally to analyst / administrator / clinician keys; the legacy
    # sim_role state takes the long display label so other surfaces stay
    # consistent.
    _short_labels = [ROLE_SHORT_LABELS[k] for k in ROLE_ORDER]
    role_pick = st.pills(
        "Role",
        _short_labels,
        default=st.session_state.get("sim_role_pill", _short_labels[0]),
        selection_mode="single",
        key="sim_role_pill",
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Role-adaptive body (delegates to existing render_* functions — Step 3.5
    # body review parked; reuse keeps behavior stable).
    role = role_pick or _short_labels[0]
    # Short label -> internal role key -> long display label for sim_role.
    _short_to_key = {ROLE_SHORT_LABELS[k]: k for k in ROLE_ORDER}
    _role_key = _short_to_key.get(role, "analyst")
    st.session_state["sim_role"] = ROLE_DISPLAY_NAMES[_role_key]

    with st.container(border=False):
        st.markdown('<div style="padding:0 20px 16px;">', unsafe_allow_html=True)
        # Dispatch on the internal role key (analyst / clinician /
        # administrator) — `role` is the user-facing short label
        # (IT / Nurse / Biomed) which we map back via _short_to_key
        # above. Pre-rename branches said SOC / Clinical / Admin.
        if _role_key == "analyst":
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
        elif _role_key == "clinician":
            clin = (selected.get("explanation") or {}).get("clinician_summary", "")
            st.markdown(
                f'<p style="font-size:0.875rem;line-height:1.55;color:var(--text-primary);margin:0 0 8px;">'
                f'{escape(clin) if clin else "No clinician summary on file for this alert."}</p>',
                unsafe_allow_html=True,
            )
        else:  # administrator (Biomed Engineer)
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
    #
    # Vocabulary note: the Triage view exposes a deliberate 3-action subset
    # (Acknowledge / Escalate / Dismiss). The full Module-5 vocabulary
    # (`module6_app.ACTIONS` — monitor / investigate / isolate / escalate /
    # dismiss) is available on the Online Simulation page, where the
    # operator is in active investigation rather than first-touch triage.
    # The caption below makes this routing explicit so operators don't
    # think the missing actions are a bug.
    st.markdown(
        '<div style="padding:8px 20px 4px;">'
        '  <div style="font-size:10px;font-weight:500;letter-spacing:0.08em;text-transform:uppercase;color:var(--text-tertiary);margin-bottom:10px;">Recommended actions · human-required</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    btn_col = st.container()
    with btn_col:
        # Unicode glyph prefixes were dropped from button labels for two
        # reasons: (1) screen readers read them aloud as e.g. "check mark"
        # / "upwards arrow" / "multiplication sign", which is noise; (2) the
        # action color is already encoded by the sentinel-btn-* classes via
        # data-sentinel-action selectors in sentinel_theme._BTN_CSS. Color
        # + label carry the affordance; the glyph was decorative.
        # Audit-log payload: persist BOTH the canonical internal role key
        # (analyst/administrator/clinician — stable across UI renames) AND
        # the human-readable display label so old + new records remain
        # interpretable side by side.
        _audit_role_payload = {
            "role_key": _role_key,
            "role_display": ROLE_DISPLAY_NAMES.get(_role_key, _role_key),
        }
        st.markdown('<div data-sentinel-action="acknowledge" style="padding:0 20px 6px;">', unsafe_allow_html=True)
        if st.button("Acknowledge — taking ownership", key=f"ack_{aid}", width="stretch",
                     help="Records you as the owner of this alert. Audit log gets a signed acknowledge event."):
            _capture_dashboard_action(aid, "acknowledge",
                                       details={"tier": tier, **_audit_role_payload})
            st.toast(f"Alert acknowledged · A-{aid:04d}", icon="✅")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div data-sentinel-action="escalate" style="padding:0 20px 6px;">', unsafe_allow_html=True)
        if st.button("Escalate — pull in T3 + biomed", key=f"esc_{aid}", width="stretch",
                     help="Routes the alert to Tier-3 SOC and the biomedical engineering on-call."):
            _capture_dashboard_action(aid, "escalate",
                                       details={"tier": tier, **_audit_role_payload})
            st.toast(f"Escalated · A-{aid:04d}", icon="⚠️")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div data-sentinel-action="dismiss" style="padding:0 20px 6px;">', unsafe_allow_html=True)
        if st.button("Dismiss — requires reason", key=f"dis_{aid}", width="stretch",
                     help="Closes the alert. A written rationale is required (no silent suppression)."):
            _dismiss_dialog(aid, tier, ROLE_DISPLAY_NAMES.get(_role_key, _role_key))
        st.markdown('</div>', unsafe_allow_html=True)

        # Routing note: clarifies that monitor/investigate/isolate live on
        # Online Simulation, not in the Triage view's 3-action set.
        st.markdown(
            '<div style="padding:0 20px 6px;font-size:10px;line-height:1.5;'
            'color:var(--text-tertiary);">'
            'Need <span class="font-mono">monitor</span> / '
            '<span class="font-mono">investigate</span> / '
            '<span class="font-mono">isolate</span>? Switch to '
            '<strong>Online Simulation</strong> in the sidebar — the '
            'Triage view is intentionally a 3-action first-touch surface.'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div style="padding:6px 20px 16px;">{ui.render_actions_disclaimer()}</div>',
            unsafe_allow_html=True,
        )

    # Audit timeline — derived from audit_trail.json for this sample.
    audit = load_audit_trail()
    rec = audit.get(aid, {}) if isinstance(audit, dict) else {}
    events = rec.get("events", []) if isinstance(rec, dict) else []

    if not events:
        # Empty state — was previously a fabricated "Alert raised" entry
        # that misled operators into thinking the audit chain had a record
        # when it didn't. Honest empty state: say so, and remind that the
        # chain starts the moment an action is taken.
        st.markdown(
            '<div style="padding:8px 20px 24px;">'
            '  <div style="font-size:10px;font-weight:500;letter-spacing:0.08em;'
            'text-transform:uppercase;color:var(--text-tertiary);margin-bottom:12px;">'
            'Audit trail · empty'
            '  </div>'
            '  <p style="font-size:12px;line-height:1.5;color:var(--text-secondary);'
            'margin:0;padding:8px 0 0;border-top:1px dashed var(--border-subtle);">'
            'No operator events recorded for this alert yet. The signed audit '
            'chain (<span class="font-mono">audit_log.jsonl</span>) appends '
            'on the first Acknowledge / Escalate / Dismiss action below.'
            '  </p>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
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
        get_hardened_audit().log(
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


@st.dialog("Reset simulation?")
def _confirm_sim_reset():
    """Reset wipes the latency history and rewinds the playhead to 0.

    In a long demo session this is an expensive loss — the static analytics
    panels above are derived from `latency_history`, and clearing it
    erases the rolling per-sample chart. Required confirmation prevents
    a stray click from destroying state.
    """
    st.markdown(
        '<p style="font-size:0.875rem;color:var(--text-secondary);margin:0 0 16px;">'
        'Resetting will rewind the playhead to <span class="font-mono">position 1</span> '
        'and clear the latency history accumulated this session. The audit chain '
        'and the demo split data are not affected.'
        '</p>',
        unsafe_allow_html=True,
    )
    cancel_col, confirm_col = st.columns(2)
    with cancel_col:
        if st.button("Cancel", key="sim_reset_cancel", width="stretch"):
            st.rerun()
    with confirm_col:
        if st.button("Confirm reset", key="sim_reset_confirm",
                     type="primary", width="stretch"):
            st.session_state.sim_index = 0
            if "latency_history" in st.session_state:
                st.session_state.latency_history.clear()
            # Reset incremental accumulators so tier counts and tier
            # distribution chart start from zero too.
            st.session_state.pop("_sim_acc", None)
            st.session_state.pop("_tier_history", None)
            audit_log("sim_reset")
            st.toast("Simulation reset", icon="🔄")
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
        "system": "Snapshot · test split",
        "p95_ms": p95_ms or "—",
        "build": build,
    }
    # is_live=False swaps the pulsing dot for a static one — the Triage
    # page reads a frozen JSON snapshot, not a live stream. Pulsing here
    # would mis-signal liveness the page doesn't have.
    st.markdown(ui.render_status_strip(metrics, is_live=False), unsafe_allow_html=True)

    # Keyboard-navigation hint (P3-17). Streamlit doesn't natively bind
    # arbitrary key shortcuts to widgets, but the selectbox in the queue
    # column supports ↑/↓ when focused — clicking the selectbox once gives
    # operators arrow-key navigation through the visible alerts. A small
    # helper banner makes the affordance discoverable; we don't inject JS
    # to do something more aggressive because Streamlit reruns reset DOM
    # state on every interaction and the resulting brittleness isn't worth
    # the marginal UX gain.
    st.markdown(
        '<div style="position:fixed;left:20px;bottom:44px;z-index:998;'
        'font-family:JetBrains Mono,monospace;font-size:10px;'
        'color:var(--text-quaternary);pointer-events:none;">'
        'Tip · click the alert selector and use ↑/↓ to scan'
        '</div>',
        unsafe_allow_html=True,
    )


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

    # Inject the Sentinel design tokens so the Online Simulation page
    # shares the Dashboard's visual identity. Without this, the two pages
    # use entirely different palettes / fonts and operators switching
    # between them feel like they've entered a different app.
    from module6_evaluation.sentinel_theme import inject_theme
    inject_theme()

    # Page header — display serif title + one-line context so a new
    # operator knows this is a replay tool, not a live tap.
    st.markdown(
        '<div style="margin:8px 0 4px;">'
        '  <h1 class="font-display" style="font-size:2rem;letter-spacing:-0.025em;'
        'margin:0 0 6px;color:var(--text-primary);">IoMT IDS · Online Simulation</h1>'
        '  <p style="font-size:0.875rem;color:var(--text-secondary);margin:0;line-height:1.5;">'
        'Replays the <span class="font-mono">demo</span> split through the live '
        'detection pipeline at adjustable speed. Operator interactions on this '
        'page feed the audit chain and the feedback loop — the '
        '<span class="font-mono">test</span> split stays paper-clean for the '
        'thesis metrics.'
        '  </p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Online Simulation reads the operator-clean demo split per PAGE_SPLIT.
    # Operator interactions on this page (Confirm/Reject/Note) feed the audit
    # chain and potentially the feedback loop — keeping them off the test
    # split protects the thesis's paper-clean metrics.
    responses = load_responses_for(PAGE_SPLIT["Online Simulation"])
    clin_summaries = load_clinician_summaries(PAGE_SPLIT["Online Simulation"])
    audit_trail = load_audit_trail(PAGE_SPLIT["Online Simulation"])
    latency_profile = load_latency_profile()
    live_df = load_live_stream_source()
    # Full-stream artefact (1632 entries for demo) — built by
    # tools.build_simulation_stream. Empty list when the file is missing,
    # in which case the new "Full stream" radio option silently falls
    # back to alerts-only.
    full_stream = load_simulation_stream(PAGE_SPLIT["Online Simulation"])
    full_stream_meta = load_simulation_stream_meta(PAGE_SPLIT["Online Simulation"])
    # Join analyst_report data — Module 5 strips the full analyst payload
    # (consensus / models / top_features) before serializing the response
    # records; we restore it here so render_analyst / render_admin can
    # actually show what's been computed. Empty dict when the file is
    # missing, in which case _analyst_state will report "unavailable".
    analyst_report_by_idx = load_analyst_report_for(PAGE_SPLIT["Online Simulation"])

    if not responses:
        st.warning(
            "No demo-split alerts found at "
            "`results/reports/alert_responses_demo.json`. "
            "Run `python module5_responses/module5_responses.py --split=demo`."
        )
        return

    # 6C.8 Role switcher — pills match the Dashboard's role-toggle pattern
    # (see _triage_actions_column). Selectbox cost an extra click and was
    # inconsistent with the rest of the app.
    #
    # Labels = spec triad (IT Generalist / Biomed Engineer / Nurse Manager).
    # Internal key resolution via ROLE_INTERNAL_KEY maps the display label
    # back to analyst/administrator/clinician for downstream branching.
    st.sidebar.divider()
    st.sidebar.markdown("## Stakeholder View")
    sim_role = st.sidebar.pills(
        "View as",
        ROLE_DISPLAY_LIST,
        default=st.session_state.get("sim_role", ROLE_DISPLAY_LIST[0]),
        selection_mode="single",
        key="sim_role",
        label_visibility="collapsed",
    )
    # Fallback when user clears the pill (pills allow none-selected by
    # spec); the rest of the page assumes a non-None role string.
    if not sim_role:
        sim_role = ROLE_DISPLAY_LIST[0]

    # ── Debug instrumentation (hidden behind SENTINEL_DEV env var) ──
    # Render-time captions and timing JSONL exports are useful for
    # performance work but they're noise for thesis demos and RQ3
    # participants. Toggle visibility from the shell:
    #   SENTINEL_DEV=1 streamlit run module6_evaluation/module6_app.py
    import os as _os
    _dev_mode = bool(_os.environ.get("SENTINEL_DEV"))
    if _dev_mode:
        st.sidebar.divider()
        st.sidebar.markdown("## Debug")
        st.session_state["_render_caption_enabled"] = st.sidebar.toggle(
            "Show render time", value=False, key="dbg_render_caption"
        )
        st.session_state["_render_log_enabled"] = st.sidebar.toggle(
            "Log render time to /tmp/sim_render_timings.jsonl",
            value=False,
            key="dbg_render_log",
        )
    else:
        # Default-off when SENTINEL_DEV isn't set — even if some other
        # render flow toggled them on previously in this session.
        st.session_state.setdefault("_render_caption_enabled", False)
        st.session_state.setdefault("_render_log_enabled", False)

    # ── Data source toggle (6C.11 mock live source + Full-stream B3) ──
    st.sidebar.divider()
    st.sidebar.markdown("## Data Source")
    # Full-stream is the third option — added so the operator can see
    # the entire demo dataset as a stream (NORMAL placeholders + LOW+
    # alerts), not just the 320 LOW+ alerts surfaced by M5. The new mode
    # is silently hidden when the artefact is absent so a clean clone
    # behaves identically to the pre-B3 build.
    source_options = ["Pre-computed alerts (Module 5)", "Live parquet (mock TAP)"]
    if full_stream:
        source_options.append("Full stream (1632 samples)")
    source_index = {
        "alerts": 0, "live_parquet": 1, "full_stream": 2,
    }.get(st.session_state.sim_source, 0)
    # Coerce stored state to a valid index when full_stream artefact
    # disappeared between reruns.
    if source_index >= len(source_options):
        source_index = 0
    source_label = st.sidebar.radio(
        "Stream from:",
        source_options,
        index=source_index,
        help=(
            "Pre-computed alerts replays Module 5 demo-split outputs "
            "(320 LOW+ alerts).\n\n"
            "Live parquet reads data/processed/demo_phase1.parquet row by row "
            "and attaches synthetic arrival timestamps, simulating a feature-"
            "extracted flow stream from a network TAP. Alert metadata is "
            "joined from Module 5 by sample index where available.\n\n"
            "Full stream replays the entire demo split (1632 entries — "
            "320 alerts + 1312 NORMAL placeholders that just tick the "
            "stream clock). NORMAL rows are not auditable; Confirm/Reject "
            "buttons are disabled on them. Run "
            "`python -m tools.build_simulation_stream --split demo` to "
            "rebuild after Module 3/5 regen."
        ),
    )
    if source_label.startswith("Live"):
        st.session_state.sim_source = "live_parquet"
    elif source_label.startswith("Full"):
        st.session_state.sim_source = "full_stream"
    else:
        st.session_state.sim_source = "alerts"
    using_live = st.session_state.sim_source == "live_parquet"
    using_full_stream = st.session_state.sim_source == "full_stream"

    if using_live and live_df is None:
        st.sidebar.warning(
            f"data/processed/{PAGE_SPLIT['Online Simulation']}_phase1.parquet "
            "not found — falling back to pre-computed alerts."
        )
        using_live = False
        st.session_state.sim_source = "alerts"

    # iteration_list is the single iteration backbone: 320 LOW+ alerts in
    # the legacy two modes, 1632 stream entries in full_stream mode. The
    # rest of the page (sim_index clamps, position display, tier counters,
    # end-of-stream callout) reads len(iteration_list) so the same code
    # serves both shapes — the only divergence is the per-entry render
    # path (NORMAL placeholder vs full alert card).
    iteration_list: list = full_stream if using_full_stream else responses
    n_iteration = len(iteration_list)
    if using_full_stream and st.session_state.sim_index >= n_iteration:
        # Stored sim_index from a prior alerts-mode session may exceed the
        # new bound when switching back to alerts (320 vs 1632). Clamp.
        st.session_state.sim_index = 0

    # ── Smoother playback controls ──
    # Speed is now pills (was selectbox — 2 clicks → 1 click for frequent
    # control). Emoji glyphs in button labels are wrapped via the
    # `:material/` icon convention where supported; raw unicode glyphs are
    # kept for buttons whose meaning is universal (▶/⏸).
    ctrl_a, ctrl_b, ctrl_c, ctrl_d, ctrl_e = st.columns([1.5, 1, 1, 1, 1.4])

    with ctrl_a:
        st.markdown(
            '<div style="font-size:11px;font-weight:500;letter-spacing:0.04em;'
            'text-transform:uppercase;color:var(--text-tertiary);margin-bottom:4px;">'
            'Speed</div>',
            unsafe_allow_html=True,
        )
        speed_label = st.pills(
            "Speed",
            ["0.5x", "1x", "2x", "4x"],
            default=f"{st.session_state.sim_speed:g}x" if f"{st.session_state.sim_speed:g}x" in ["0.5x", "1x", "2x", "4x"] else "1x",
            selection_mode="single",
            key="sim_speed_pill",
            label_visibility="collapsed",
        )
        if speed_label:
            st.session_state.sim_speed = float(speed_label.rstrip("x"))

    with ctrl_b:
        if st.session_state.sim_running:
            if st.button("Pause", width="stretch", icon=":material/pause:",
                         help="Halt auto-advance. Step still works."):
                st.session_state.sim_running = False
                audit_log("sim_pause", sim_index=st.session_state.sim_index)
        else:
            if st.button("Resume", width="stretch", icon=":material/play_arrow:",
                         help="Resume auto-advance at the current speed."):
                st.session_state.sim_running = True
                audit_log("sim_resume", sim_index=st.session_state.sim_index)

    with ctrl_c:
        if st.button("Step", width="stretch", icon=":material/skip_next:",
                     help="Advance one alert (works while paused)."):
            st.session_state.sim_index = min(st.session_state.sim_index + 1, n_iteration - 1)
            push_latency_sample(latency_profile)

    with ctrl_d:
        if st.button("Reset", width="stretch", icon=":material/restart_alt:",
                     help="Rewind to alert 0 and clear the latency history. Requires confirmation."):
            _confirm_sim_reset()

    with ctrl_e:
        # Range hint surfaced in the label so operators don't have to
        # trial-and-error the bounds. The number_input enforces them at
        # widget level anyway, but the label is the first place a user
        # looks. 1-based to match the progress bar and "Stream position"
        # metric — see corresponding comment near the playhead.
        max_n = n_iteration
        jump_target = st.number_input(
            f"Jump to stream position # (1–{max_n})",
            min_value=1,
            max_value=max_n,
            value=int(st.session_state.sim_index) + 1,
            step=1,
            help="Jump the playhead to a specific position in the alert "
                 "stream (1-based). This is the order alerts arrived, "
                 "not the alert ID — see expander labels below for IDs.",
        )
        new_idx = int(jump_target) - 1
        if new_idx != st.session_state.sim_index:
            st.session_state.sim_index = new_idx
            audit_log("sim_jump", position=int(jump_target), sim_index=new_idx)

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
                    # Sentinel accent token. The previous #3274A1 was the
                    # matplotlib default blue and didn't match the rest of
                    # the app.
                    st.bar_chart(stage_df, color="#7BA7BC")
                    # SLA threshold annotation — st.bar_chart can't draw a
                    # horizontal reference line, so the value is surfaced
                    # in a caption directly under the chart instead.
                    st.caption(
                        "Reference: 150 ms end-to-end SLA. Stage means above this "
                        "value are likely the dominant latency contributor."
                    )

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
            # Online Simulation is the demo split — read the demo-suffixed
            # output so this panel reflects the operator-clean stream.
            sim_suffix = _resolve_suffix(PAGE_SPLIT["Online Simulation"])
            dyn_path = EVAL_DIR / f"dynamic_threshold_results{sim_suffix}.json"
            if dyn_path.exists():
                with open(dyn_path) as f:
                    dyn = json.load(f)
                b1 = dyn.get("b1_static_vs_adaptive", {})
                fm = b1.get("final_metrics", {})
                if fm:
                    thc1, thc2 = st.columns(2)
                    thc1.metric("Static F1", f"{fm.get('static', {}).get('f1', 0):.4f}")
                    thc2.metric("Adaptive F1", f"{fm.get('adaptive', {}).get('f1', 0):.4f}")
                thresh_bytes = _cached_png_bytes(
                    str(CHARTS_DIR / f"threshold_over_time{sim_suffix}.png")
                )
                if thresh_bytes:
                    st.image(thresh_bytes, width="stretch", caption="DAE threshold: static vs adaptive")
            else:
                st.info(
                    f"Run `python tools/diagnostics/dynamic_threshold_sim.py --split={PAGE_SPLIT['Online Simulation']}` "
                    "to enable adaptive threshold monitoring"
                )

        with col_drift:
            st.markdown("#### Drift Detection Status")
            # Online Simulation is the demo split — read the demo-suffixed
            # output so this panel reflects the operator-clean stream.
            drift_path = EVAL_DIR / f"drift_detection_results{sim_suffix}.json"
            if drift_path.exists():
                with open(drift_path) as f:
                    drift = json.load(f)
                psi = drift.get("psi_summary", {})
                n_events = len(drift.get("drift_events", []))
                dc1, dc2 = st.columns(2)
                dc1.metric(
                    "Drift Events", n_events,
                    help=(
                        "Number of times the drift detector flagged a "
                        "distribution shift since the last calibration. Each "
                        "event triggers a re-baseline check downstream."
                    ),
                )
                psi_max = psi.get("max", 0)
                dc2.metric(
                    "PSI (max)",
                    f"{psi_max:.4f}",
                    delta="DRIFT" if psi_max > 0.1 else "OK",
                    delta_color="inverse" if psi_max > 0.1 else "normal",
                    help=(
                        "Population Stability Index — measures how much each "
                        "feature's distribution has shifted vs. the calibration "
                        "baseline.\n\n"
                        "• PSI < 0.1 = no significant drift\n"
                        "• 0.1 ≤ PSI < 0.25 = moderate drift (monitor)\n"
                        "• PSI ≥ 0.25 = significant drift (re-baseline)"
                    ),
                )
                psi_bytes = _cached_png_bytes(str(CHARTS_DIR / f"drift_psi{sim_suffix}.png"))
                if psi_bytes:
                    st.image(psi_bytes, width="stretch", caption="PSI over time")
            else:
                st.info(
                    f"Run `python tools/diagnostics/drift_detection.py --split={PAGE_SPLIT['Online Simulation']}` "
                    "to enable drift monitoring"
                )

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
        if st.session_state.sim_running and st.session_state.sim_index < n_iteration - 1:
            st.session_state.sim_index = min(st.session_state.sim_index + 1, n_iteration - 1)
            push_latency_sample(latency_profile)

        idx_local = st.session_state.sim_index
        # Issue 3 fix: avoid O(n) history_local slice — use direct index
        # access throughout. history_local is only needed for the tier
        # distribution chart which uses the incremental _tier_history state.
        # current_batch_local is a bounded O(3) slice — kept as-is.
        window_size = 3
        current_batch_local = iteration_list[max(0, idx_local - window_size + 1) : idx_local + 1]

        # ── Issues 2 & 3: incremental accumulators ──────────────────────
        # Replace O(n) Counter + sum(1 for ...) on growing history_local
        # with O(1) session-state accumulators updated on each tick delta.
        # On a playhead jump backward, rebuild is O(k) where k = new_idx.
        # NORMAL bucket is included so the full-stream mode's tier strip
        # is faithful to the population the operator is reviewing.
        _acc = st.session_state.setdefault("_sim_acc", {
            "idx": -1,
            "tier": {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "NORMAL": 0},
            "attacks": 0,
        })
        # Backfill NORMAL bucket on legacy session states that pre-date
        # full_stream mode — without this, a switch into full_stream on
        # the same session would silently drop NORMAL hits.
        _acc["tier"].setdefault("NORMAL", 0)
        if idx_local < _acc["idx"]:
            # Jumped backward — rebuild from scratch up to idx_local
            _acc["tier"] = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "NORMAL": 0}
            _acc["attacks"] = 0
            for _r in iteration_list[:idx_local + 1]:
                _lv = _r.get("risk_level", "LOW")
                if _lv in _acc["tier"]:
                    _acc["tier"][_lv] += 1
                if _r.get("ground_truth") == "attack":
                    _acc["attacks"] += 1
            _acc["idx"] = idx_local
        elif idx_local > _acc["idx"]:
            # Advanced forward — only process new records (delta)
            for _i in range(_acc["idx"] + 1, idx_local + 1):
                _r = iteration_list[_i]
                _lv = _r.get("risk_level", "LOW")
                if _lv in _acc["tier"]:
                    _acc["tier"][_lv] += 1
                if _r.get("ground_truth") == "attack":
                    _acc["attacks"] += 1
            _acc["idx"] = idx_local
        # _acc is always consistent with idx_local at this point

        # Status + progress. Status indicator uses both color *and* text
        # (was emoji-only — color-blind operators / non-emoji fonts lose
        # the signal). Pulse-live class comes from the Sentinel theme.
        status_col, prog_col = st.columns([1, 4])
        with status_col:
            running_local = st.session_state.sim_running
            at_end = idx_local >= n_iteration - 1
            if at_end:
                state_label = "Complete"
                state_dot = "pulse-static"
                state_color = "var(--text-secondary)"
            elif running_local:
                state_label = "Running"
                state_dot = "pulse-live"
                state_color = "var(--success)"
            else:
                state_label = "Paused"
                state_dot = "pulse-static"
                state_color = "var(--warning)"
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;'
                f'font-size:0.875rem;color:var(--text-primary);">'
                f'<span class="{state_dot}" style="background:{state_color};"></span>'
                f'<span style="font-weight:500;">{state_label}</span>'
                f'<span class="font-mono" style="color:var(--text-tertiary);font-size:11px;">'
                f'{st.session_state.sim_speed:g}x</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with prog_col:
            st.progress(
                (idx_local + 1) / max(1, n_iteration),
                text=f"Position {idx_local + 1} / {n_iteration}",
            )

        # End-of-stream callout (OS-12). Without this the simulation just
        # quietly stops auto-advancing and the operator has no signal that
        # they've reviewed everything. Full-stream mode adds the alert /
        # NORMAL breakdown so the operator sees the population they just
        # scrubbed through (not just the count of cells advanced).
        if idx_local >= n_iteration - 1:
            if using_full_stream and full_stream_meta:
                n_alerts = int(full_stream_meta.get("n_surfaced", 0))
                n_normal_m = int(full_stream_meta.get("n_normal", 0))
                st.success(
                    f"🏁 End of stream — processed all {n_iteration} samples "
                    f"({n_alerts} alerts, {n_normal_m} NORMAL skipped). Use "
                    f"**Reset** to start over, or **Jump to stream position #** "
                    f"to revisit a specific position."
                )
            else:
                st.success(
                    f"🏁 End of stream — reviewed all {n_iteration} alerts in "
                    f"the demo split. Use **Reset** to start over, or **Jump to "
                    f"stream position #** to revisit a specific position."
                )

        # Mock live source preview (only when live mode active).
        # B5: index live_df by the alert's sample_index, NOT by idx_local —
        # idx_local was the alert position (0..n-1), so live_df.iloc[idx_local]
        # was previewing the first n parquet rows (mostly NORMAL) instead of
        # the actual parquet row that produced the alert at the playhead.
        if using_live and live_df is not None:
            current_entry = iteration_list[idx_local] if idx_local < n_iteration else {}
            sample_index = int(current_entry.get("sample_index", idx_local))
            if 0 <= sample_index < len(live_df):
                live_row = live_df.iloc[sample_index]
                with st.expander(
                    f"📡 Live source — row {sample_index} arrived at "
                    f"{live_row['arrived_at']}",
                    expanded=False,
                ):
                    st.caption(
                        "Mock TAP: feature-extracted flow read directly from "
                        f"data/processed/{PAGE_SPLIT['Online Simulation']}_phase1.parquet."
                    )
                    preview_cache = st.session_state.setdefault("_live_preview_cache", {})
                    if sample_index not in preview_cache:
                        preview_cols = [c for c in live_row.index if c != "arrived_at"][:8]
                        preview_cache[sample_index] = pd.DataFrame(
                            {"feature": preview_cols, "value": [live_row[c] for c in preview_cols]}
                        )
                        if len(preview_cache) > 256:
                            preview_cache.pop(next(iter(preview_cache)))
                    st.dataframe(
                        preview_cache[sample_index],
                        hide_index=True,
                        width="stretch",
                    )

        # Summary metrics — O(1) reads from incremental accumulators.
        # All four tiers are shown so the operator sees the full tier
        # distribution at a glance (was CRITICAL+HIGH only, hiding the
        # MEDIUM/LOW volume that contextualizes how noisy the stream is).
        st.markdown("---")
        # Full-stream mode adds a NORMAL column so the tier strip totals
        # to the population the operator is reviewing. Alerts-only mode
        # keeps the original 6-column layout (no NORMAL since the iter
        # list is already pre-filtered LOW+).
        if using_full_stream:
            mc1, mc2, mc3, mc4, mc5, mc6, mc7 = st.columns(7)
            mc1.metric(
                "Stream position",
                f"{idx_local + 1} / {n_iteration}",
                help="Current playhead position in the full stream (1-based). "
                     "NORMAL rows are included; the iteration covers the "
                     "entire demo dataset.",
            )
            mc2.metric("CRITICAL", _acc["tier"].get("CRITICAL", 0))
            mc3.metric("HIGH", _acc["tier"].get("HIGH", 0))
            mc4.metric("MEDIUM", _acc["tier"].get("MEDIUM", 0))
            mc5.metric("LOW", _acc["tier"].get("LOW", 0))
            mc6.metric("NORMAL", _acc["tier"].get("NORMAL", 0),
                       help="Stream ticks where Module 3 assigned NORMAL "
                            "(no operator action required).")
            mc7.metric("True attacks", _acc["attacks"],
                       help="Ground-truth attack count among samples processed so far.")
        else:
            mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
            mc1.metric(
                "Stream position",
                f"{idx_local + 1} / {n_iteration}",
                help="Current playhead position in the alert stream (1-based). "
                     "This is NOT the alert ID — see expander labels below.",
            )
            mc2.metric("CRITICAL", _acc["tier"].get("CRITICAL", 0))
            mc3.metric("HIGH", _acc["tier"].get("HIGH", 0))
            mc4.metric("MEDIUM", _acc["tier"].get("MEDIUM", 0))
            mc5.metric("LOW", _acc["tier"].get("LOW", 0))
            mc6.metric("True attacks", _acc["attacks"],
                       help="Ground-truth attack count among samples processed so far.")

        # Latest risk score + 4-component breakdown
        if current_batch_local:
            latest = current_batch_local[-1]
            latest_score = latest["risk_score"]
            latest_level = latest["risk_level"]
            col_gauge, col_components = st.columns([1, 2])
            with col_gauge:
                # Renamed from "Risk Score Gauge" — st.metric + st.progress
                # is a number plus a bar, not a true gauge. The old name
                # over-promised on the visualization.
                st.markdown("#### Latest risk score")
                # st.metric.delta needs a numeric sign to honor delta_color.
                # The previous code passed a string ("CRITICAL"), so the
                # inverse-color logic never kicked in. Render the tier as
                # a colored badge instead, and use a tone-appropriate help
                # tooltip for context.
                tier_color = TIER_STREAMLIT_COLORS.get(latest_level, "gray")
                st.metric(
                    "Current Alert",
                    f"{latest_score:.3f}",
                    help="Risk tier classification for this sample.",
                )
                st.markdown(
                    f"**Tier:** :{tier_color}[**{latest_level}**]",
                    help="Tier color reflects severity (violet=CRITICAL, red=HIGH, "
                    "orange=MEDIUM, green=LOW).",
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
                    # Sentinel accent token (#7BA7BC). The previous
                    # hardcoded #3274A1 (matplotlib default blue) was
                    # off-brand and didn't match the Dashboard's palette.
                    st.bar_chart(comp_df.set_index("Component"), color="#7BA7BC")
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
                # SLA badge — styled HTML pill rather than emoji-prefixed
                # caption. SLA breach is an incident-level signal; muted
                # caption tone under-weights it.
                if last_total > 150:
                    badge_bg = "var(--tier-high-bg)"
                    badge_border = "rgba(224, 122, 95, 0.3)"
                    badge_color = "var(--tier-high)"
                    badge_label = "SLA breach"
                else:
                    badge_bg = "var(--success-bg)"
                    badge_border = "rgba(95, 158, 123, 0.3)"
                    badge_color = "var(--success)"
                    badge_label = "Within SLA"
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:10px;'
                    f'margin-top:6px;font-size:12px;color:var(--text-secondary);">'
                    f'<span style="display:inline-flex;align-items:center;gap:6px;'
                    f'padding:3px 10px;border-radius:3px;background:{badge_bg};'
                    f'border:1px solid {badge_border};color:{badge_color};'
                    f'font-family:JetBrains Mono,monospace;font-size:11px;'
                    f'font-weight:500;letter-spacing:0.04em;">{badge_label}</span>'
                    f'<span>Latest total latency: <span class="font-mono" '
                    f'style="color:var(--text-primary);">{last_total:.1f} ms</span> '
                    f'(SLA = 150 ms)</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("Step or resume the simulation to populate the rolling latency chart.")

        # Current batch (per-alert expanders with role render + interactions + FDA export)
        st.markdown("---")
        st.markdown("### Current Batch")
        alerts_cache = st.session_state.setdefault("_processed_alerts", {})
        fda_cache = st.session_state.setdefault("_fda_payload_cache", {})
        fda_filename_cache = st.session_state.setdefault("_fda_filename_cache", {})

        for offset, r in enumerate(current_batch_local):
            sample_idx = r["sample_index"]
            level = r["risk_level"]
            score = r["risk_score"]
            # Streamlit's `:color[text]` syntax only supports named colors —
            # the previous `:{hex}[level]` rendered as literal `:e74c3c[HIGH]`.
            named_color = TIER_STREAMLIT_COLORS.get(level, "gray")

            # current_batch_local is the trailing window ending at idx_local,
            # so this entry's 1-based stream position is:
            #   idx_local - (window_len - 1 - offset) + 1
            # Edge: when idx_local < window-1, the slice is shorter and the
            # formula still holds since len(current_batch_local) reflects
            # the actual slice length.
            position = idx_local - (len(current_batch_local) - 1 - offset) + 1

            # Full-stream NORMAL row: render a compact placeholder and skip
            # the M5 alert pipeline entirely — process_alert expects the
            # alert-record shape, and NORMAL entries have alert=None.
            # B4: visible but un-actionable; no operator buttons.
            if level == "NORMAL":
                arrived_at = r.get("arrived_at", "")
                st.markdown(
                    f'<div style="padding:10px 12px;border-left:3px solid '
                    f'var(--text-tertiary);background:var(--surface-2);'
                    f'opacity:0.65;margin-bottom:6px;">'
                    f'<div class="font-mono" style="font-size:11px;'
                    f'color:var(--text-tertiary);">'
                    f'Position {position} · sample #{sample_idx} · {arrived_at}'
                    f'</div>'
                    f'<div style="font-size:13px;color:var(--text-secondary);'
                    f'margin-top:4px;">'
                    f'<strong>NORMAL</strong> — no operator action required '
                    f'<span class="font-mono" style="margin-left:8px;'
                    f'font-size:11px;color:var(--text-tertiary);">'
                    f'R={score:.3f}</span>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                continue

            # Full-stream LOW+ row: unwrap the embedded M5 alert payload so
            # the downstream render path sees the alert-record shape it
            # expects (explanation, response, risk_components ...). In
            # alerts-only mode, r is already the M5 record — no unwrap.
            if using_full_stream and isinstance(r.get("alert"), dict):
                r = r["alert"]

            with st.expander(
                f"Position {position} · Alert A-{sample_idx:04d} — "
                f":{named_color}[{level}] R={score:.3f}",
                expanded=(level in ("CRITICAL", "HIGH")),
            ):
                if sample_idx not in alerts_cache:
                    clin = clin_summaries.get(sample_idx, {})
                    # Module 5 strips analyst payload from alert_responses
                    # records (keeping only the `analyst_available` flag),
                    # so we join from `analyst_report_by_idx` loaded above.
                    # The `r.get("explanation", {}).get("analyst", {})`
                    # chain that lived here used to always return {} and
                    # silently break the analyst view.
                    analyst_data = analyst_report_by_idx.get(sample_idx, {})
                    analyst_models = analyst_data.get("models") or {}
                    xgb_top = (analyst_models.get("xgboost") or {}).get("top_features", [])
                    dae_top = (analyst_models.get("dae") or {}).get("top_features", [])
                    alerts_cache[sample_idx] = process_alert(
                        sample_idx,
                        {
                            "risk_score": score,
                            "risk_level": level,
                            "attack_category": r.get("attack_category", "unknown"),
                            "xai_explanation": {
                                "xgboost_top_features": xgb_top,
                                "dae_top_features": dae_top,
                                "clinician_summary": clin.get("summary", ""),
                                "consensus": analyst_data.get("consensus", ""),
                                # Per-model breakdown — surfaced by the new
                                # render_model_breakdown component. Empty
                                # dict when analyst data is unavailable.
                                "models": analyst_models,
                            },
                        },
                    )
                    # Mirror analyst_available flag onto the cached alert
                    # so _analyst_state() can distinguish "Module 4 didn't
                    # process this sample" from "pipeline desynced". Preserve
                    # the MVE payload (Option 4) and any other fields Module 5
                    # attached to the explanation — earlier this branch built
                    # a fresh dict and silently dropped explanation.mve, so
                    # render_mve_layers fell back to clinician_summary even
                    # after Module 5 produced an LLM-generated Layer 1.
                    src_expl = r.get("explanation") or {}
                    alerts_cache[sample_idx]["explanation"] = {
                        **src_expl,
                        "analyst_available": bool(src_expl.get("analyst_available")),
                    }
                alert_obj = alerts_cache[sample_idx]

                # Dispatch on internal role key — the sidebar pill returns
                # a display label (e.g. "IT Generalist"); map it back to
                # the canonical key (analyst/clinician/administrator)
                # before branching.
                _role_key = ROLE_INTERNAL_KEY.get(sim_role, "analyst")
                if _role_key == "analyst":
                    render_analyst(alert_obj)
                elif _role_key == "clinician":
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

                # Label-feedback section (OS-6): distinct from the Triage
                # Dashboard's Acknowledge/Escalate/Dismiss workflow actions.
                # Confirm/Reject here are *labels* fed back into the
                # feedback-loop training data, not operational decisions.
                st.markdown(
                    '<div style="font-size:10px;font-weight:500;letter-spacing:0.08em;'
                    'text-transform:uppercase;color:#6A6F7B;margin:8px 0 6px;">'
                    'Label feedback · trains the next model iteration'
                    '</div>',
                    unsafe_allow_html=True,
                )

                # Buttons on top row, note input below — the previous
                # 3-column layout made the text input visually overpower the
                # two buttons. Stacking gives each control its natural width.
                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    if st.button("Confirm", key=f"confirm_{sample_idx}", width="stretch"):
                        capture_online_interaction(
                            st.session_state.get("participant_id", "anon"),
                            sample_idx,
                            "confirm",
                            {"tier": level, "score": score},
                        )
                        # st.toast lives ~3s in a global slot — it survives
                        # the fragment's next tick, unlike st.success which
                        # vanishes the moment the fragment re-renders.
                        st.toast(f"Confirmed · A-{sample_idx:04d}", icon="✅")
                with btn_col2:
                    if st.button("Reject", key=f"reject_{sample_idx}", width="stretch"):
                        capture_online_interaction(
                            st.session_state.get("participant_id", "anon"),
                            sample_idx,
                            "reject",
                            {"tier": level, "score": score},
                        )
                        st.toast(
                            f"Rejected · A-{sample_idx:04d} — added to feedback loop",
                            icon="⚠️",
                        )

                # Note input. The auto-tick fragment re-runs every speed
                # interval — a naive `if note: capture_online_interaction(...)`
                # used to log on every tick, spamming the audit chain with
                # duplicates of the same note. We compare against the
                # last-logged value per sample and only log on change.
                note = st.text_input(
                    "Feedback note",
                    key=f"note_{sample_idx}",
                    placeholder="Add feedback note (optional)…",
                )
                _last_logged = st.session_state.setdefault("_last_logged_notes", {})
                if note and _last_logged.get(sample_idx) != note:
                    capture_online_interaction(
                        st.session_state.get("participant_id", "anon"),
                        sample_idx,
                        "feedback_note",
                        {"note": note, "tier": level},
                    )
                    _last_logged[sample_idx] = note

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
                window_caption = None
            else:
                display = {t: tier_state["data"][t][-DISPLAY_LIMIT:] for t in TIERS}
                # Surface the windowing so the operator doesn't think the
                # chart "froze" once they cross 200 samples — the chart is
                # a rolling window, not a complete history past that point.
                window_caption = (
                    f"Showing the most recent {DISPLAY_LIMIT} of "
                    f"{tier_state['len']} processed samples. Earlier samples "
                    f"are still counted in the tier metrics above."
                )
            st.line_chart(pd.DataFrame(display))
            if window_caption:
                st.caption(window_caption)

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

    level = alert.get("risk_level", "LOW")
    # BA-1: Use the single tier-token source of truth — the previous local
    # `level_colors` had CRITICAL=red which conflicted with the rest of
    # the app (CRITICAL=violet). Four mapping tables had silently diverged
    # across pages; consolidating here removes the last in-tree exception.
    # tier_token maps to Sentinel CSS vars (--tier-critical, --tier-high, ...).
    tier_token = level.lower()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Risk Score", f"{alert['risk_score']:.2f}")
    with col2:
        # BA-12: Render as a proper pill badge instead of plain colored
        # text. Pill matches the visual weight of the Sentinel investigation
        # header so cross-page recognition is preserved. The dot inherits
        # color from the parent span via currentColor.
        st.markdown(
            f'<div style="font-size:10px;font-weight:500;letter-spacing:0.08em;'
            f'text-transform:uppercase;color:var(--text-tertiary);margin-bottom:4px;">'
            f'Risk Level</div>'
            f'<span style="display:inline-flex;align-items:center;gap:6px;'
            f'padding:4px 12px;border-radius:4px;'
            f'background:var(--tier-{tier_token}-bg);'
            f'color:var(--tier-{tier_token});'
            f'font-family:JetBrains Mono,monospace;font-size:13px;'
            f'font-weight:500;letter-spacing:0.04em;">'
            f'<span style="width:8px;height:8px;border-radius:50%;'
            f'background:currentColor;display:inline-block;"></span>'
            f'{level}'
            f'</span>',
            unsafe_allow_html=True,
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
            # BA-10: SHAP features as bar visualization. The previous
            # bullet list buried the SHAP magnitude in body text — operators
            # had to mentally rank 5 numbers. The bar makes the dominant
            # contributors visually obvious; the sign drives the bar color
            # (positive = accent, negative = tier-high tone).
            st.markdown("**Top contributing features (SHAP)**")
            max_abs = max(abs(f.get("shap_value", 0)) for f in top_feats[:5]) or 1.0
            from module6_evaluation import components as ui_mod
            shap_html = '<div style="padding:4px 0 8px;">'
            for f in top_feats[:5]:
                v = float(f.get("shap_value", 0))
                weight_pct = int(round(abs(v) / max_abs * 100))
                feat_type = "biometric" if f["feature"] in BIOMETRIC_FEATURES else "network feature"
                shap_html += ui_mod.render_factor_row(
                    label=f["feature"],
                    sublabel=feat_type,
                    weight_pct=weight_pct,
                    contribution=v,
                    negative=(v < 0),
                )
            shap_html += '</div>'
            st.markdown(shap_html, unsafe_allow_html=True)

        dae_feats = xai.get("dae_top_features", [])
        if dae_feats:
            st.markdown("**DAE anomaly indicators**")
            dae_html = '<div style="padding:4px 0 8px;">'
            from module6_evaluation import components as ui_mod
            for f in dae_feats[:3]:
                pct = float(f.get("pct_contribution", 0))
                dae_html += ui_mod.render_factor_row(
                    label=f["feature"],
                    sublabel="DAE reconstruction error",
                    weight_pct=int(round(min(100, pct))),
                    contribution=pct / 100,
                )
            dae_html += '</div>'
            st.markdown(dae_html, unsafe_allow_html=True)

        consensus = xai.get("consensus", "")
        if consensus:
            # Unified visual treatment (T10) — was st.info with the
            # legacy "Model consensus" label.
            parsed = _parse_consensus(consensus)
            if parsed:
                n_flagged, total = parsed
                st.markdown(
                    ui_mod.render_consensus_badge(
                        n_flagged, total, label=DETECTOR_CONSENSUS_LABEL,
                    ),
                    unsafe_allow_html=True,
                )
            else:
                st.info(f"{DETECTOR_CONSENSUS_LABEL}: {consensus}")

        # BA-19: Graceful fallback when the waterfall PNG is missing.
        # Previously the section was invisible — operator couldn't tell
        # whether SHAP plotting failed or was just unavailable.
        wf_path = CHARTS_DIR / f"waterfall_xgboost_sample_{alert['sample_index']:04d}.png"
        wf_bytes = _cached_png_bytes(str(wf_path))
        if wf_bytes:
            st.image(wf_bytes, caption="SHAP Waterfall Plot", width="stretch")
        else:
            st.caption(
                f"SHAP waterfall plot unavailable for this sample · "
                f"expected at `results/charts/{wf_path.name}`."
            )
    else:
        st.markdown("---")
        # BA-8: The original "Decide based on risk score and level only"
        # implied a decision flow that browse_mode does not have (response
        # forms live in study_mode). The new copy explains the XAI toggle
        # honestly and points operators to the right surface for decisions.
        st.info(
            "XAI explanation hidden via the sidebar toggle. Re-enable it to "
            "see the 3-layer MVE breakdown and SHAP contributions. "
            "For decision capture, switch to the **Study (A/B)** or "
            "**Online Simulation** page."
        )


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
    """6.3a — Free browsing with XAI toggle.

    Refactored to match Dashboard / Online Simulation visual identity:
      * Sentinel theme injected (BA-3)
      * Title unified with sidebar nav label (BA-5)
      * Page intro caption explains the surface honestly (BA-13)
      * Sidebar restructured with dividers (BA-14)
      * Answer key gated behind facilitator toggle (BA-2)
      * Prev/Next navigation buttons + kbd hint (BA-4, BA-20)
      * Filter chips + search (BA-7)
      * Selectbox shows tier + visited mark (BA-6, BA-18)
      * Recommended action uses Sentinel tier glyphs (BA-11)
    """
    from module6_evaluation.sentinel_theme import inject_theme
    from module6_evaluation import components as ui_mod
    inject_theme()

    alerts = load_alerts()
    n = len(alerts)

    # ── Sidebar: structured with dividers ─────────────────────────────
    st.sidebar.divider()
    st.sidebar.markdown("## Browse Controls")
    show_xai = st.sidebar.toggle(
        "Show XAI Explanation",
        value=True,
        help="Toggle the 3-layer MVE breakdown and SHAP visualisation.",
    )

    # ── Filter chips + search (BA-7) ──────────────────────────────────
    st.sidebar.divider()
    st.sidebar.markdown("### Filter")
    st.session_state.setdefault("browse_filter", "All")
    st.session_state.setdefault("browse_search", "")
    st.sidebar.pills(
        "Filter",
        ["All", "Critical + High", "Attacks only"],
        default=st.session_state.get("browse_filter", "All"),
        selection_mode="single",
        key="browse_filter",
        label_visibility="collapsed",
    )
    st.sidebar.text_input(
        "Search",
        placeholder="Search attack / alert ID",
        key="browse_search",
        label_visibility="collapsed",
    )

    f = st.session_state.get("browse_filter", "All")
    q = (st.session_state.get("browse_search") or "").strip().lower()

    def _matches(a: dict) -> bool:
        if f == "Critical + High" and a.get("risk_level") not in ("CRITICAL", "HIGH"):
            return False
        if f == "Attacks only" and a.get("ground_truth") != "attack":
            return False
        if q:
            aid = str(a.get("alert_id", "")).lower()
            cat = str(a.get("attack_category", "")).lower()
            if q not in aid and q not in cat:
                return False
        return True

    visible_indices = [i for i, a in enumerate(alerts) if _matches(a)]
    if not visible_indices:
        st.sidebar.warning("No alerts match the current filter + search.")
        visible_indices = list(range(n))

    # ── Alert selector (BA-6, BA-18) ──────────────────────────────────
    st.sidebar.divider()
    st.sidebar.markdown("### Select alert")

    visited: set = st.session_state.setdefault("browse_visited", set())
    prev_idx = st.session_state.get("browse_idx", visible_indices[0])
    if prev_idx not in visible_indices:
        prev_idx = visible_indices[0]

    def _option_fmt(i: int) -> str:
        a = alerts[i]
        lvl = a.get("risk_level", "LOW")[:4]
        mark = "* " if i in visited else "  "
        return f"{mark}{lvl}  -  {a.get('alert_id', f'A-{i:04d}')}  -  {a.get('attack_category', '?')}"

    chosen_idx = st.sidebar.selectbox(
        "Alert",
        visible_indices,
        index=visible_indices.index(prev_idx),
        format_func=_option_fmt,
        key="browse_alert_selectbox",
        label_visibility="collapsed",
    )
    st.session_state["browse_idx"] = chosen_idx
    idx = chosen_idx
    alert = alerts[idx]
    visited.add(idx)

    # ── Facilitator-only answer key (BA-2) ────────────────────────────
    # Previously the sidebar exposed ground truth + correct action right
    # away — fine for facilitator review, but a spoiler for any
    # operator-learning use of this page. Gated behind a toggle that
    # defaults OFF.
    st.sidebar.divider()
    show_answer = st.sidebar.toggle(
        "Show facilitator answer key",
        value=False,
        help="Reveals ground truth + correct action. Hidden by default to "
        "preserve the browse-as-learning use case. Facilitators reviewing "
        "decisions should flip this on.",
    )
    if show_answer:
        st.sidebar.markdown(
            '<div style="margin-top:8px;padding:10px 12px;border:1px solid '
            'rgba(212,164,69,0.3);background:var(--tier-medium-bg);'
            'border-radius:4px;font-size:11px;">'
            '<div style="font-weight:500;color:var(--tier-medium);'
            'text-transform:uppercase;letter-spacing:0.08em;font-size:10px;'
            'margin-bottom:6px;">Answer key - facilitator only</div>'
            f'<div><strong>Ground truth:</strong> <code>{alert["ground_truth"]}</code></div>'
            f'<div><strong>Attack type:</strong> <code>{alert["attack_category"]}</code></div>'
            f'<div><strong>Correct action:</strong> <code>{alert.get("correct_action", "N/A")}</code></div>'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── Page header (BA-5, BA-13) ─────────────────────────────────────
    st.markdown(
        '<div style="margin:8px 0 4px;">'
        '  <h1 class="font-display" style="font-size:2rem;letter-spacing:-0.025em;'
        'margin:0 0 6px;color:var(--text-primary);">Browse Alerts</h1>'
        '  <p style="font-size:0.875rem;color:var(--text-secondary);margin:0;line-height:1.5;">'
        'Free-browsing view over the <span class="font-mono">evaluation</span> '
        'alert set. No decisions are captured here - switch to the '
        '<strong>Study (A/B)</strong> or <strong>Online Simulation</strong> '
        'page to record responses. Use the filter chips in the sidebar to '
        'narrow scope; the answer-key toggle reveals ground truth for '
        'facilitator review.'
        '  </p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Position indicator + visited count
    visible_position = visible_indices.index(idx) + 1
    visible_total = len(visible_indices)
    st.markdown(
        f'<div style="display:flex;gap:18px;align-items:center;margin:12px 0 8px;'
        f'font-family:JetBrains Mono,monospace;font-size:11px;'
        f'color:var(--text-tertiary);">'
        f'<span>Alert <span style="color:var(--text-primary);">{visible_position}</span> '
        f'of {visible_total}{" (filtered)" if visible_total != n else ""}</span>'
        f'<span>|</span>'
        f'<span>XAI <span style="color:var(--text-primary);">'
        f'{"on" if show_xai else "off"}</span></span>'
        f'<span>|</span>'
        f'<span>Visited <span style="color:var(--text-primary);">{len(visited)}</span> '
        f'/ {n}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Prev / Next navigation (BA-4) ─────────────────────────────────
    nav_prev, nav_next, nav_unvisit = st.columns([1, 1, 1.4])
    cur_pos = visible_indices.index(idx)
    with nav_prev:
        if st.button("< Previous", key="browse_prev", width="stretch",
                     disabled=(cur_pos == 0),
                     help="Move to the previous alert in the current filter."):
            st.session_state["browse_idx"] = visible_indices[max(0, cur_pos - 1)]
            st.rerun()
    with nav_next:
        if st.button("Next >", key="browse_next", width="stretch",
                     disabled=(cur_pos >= visible_total - 1),
                     help="Move to the next alert in the current filter."):
            st.session_state["browse_idx"] = visible_indices[min(visible_total - 1, cur_pos + 1)]
            st.rerun()
    with nav_unvisit:
        if st.button("Clear visited markers", key="browse_clear_visited",
                     width="stretch",
                     help="Forget which alerts you've already opened."):
            st.session_state["browse_visited"] = set()
            st.rerun()

    display_alert(alert, show_xai)

    # ── Recommended Action (BA-11) ────────────────────────────────────
    # Only renders when the facilitator toggle is on - the action *is* the
    # answer for the test split, so leaving it visible by default leaks
    # the same information BA-2 just hid.
    if not show_answer:
        # Keyboard shortcut hint (BA-20) only — recommended action is
        # gated; show the hint here so it sits below display_alert.
        st.markdown(
            '<div style="margin-top:24px;padding-top:12px;border-top:1px dashed var(--border-subtle);'
            'font-family:JetBrains Mono,monospace;font-size:10px;color:var(--text-quaternary);">'
            'Tip - click the Alert selector in the sidebar and use the up/down keys to scan. '
            'Use the Prev/Next buttons above to step through the filtered set.'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    st.divider()
    st.subheader("Recommended Action")
    correct_action = alert.get("correct_action", "")
    # Tier-mapped guidance — emoji color circles (the previous design)
    # replaced with the Sentinel tier glyph + token color, so the visual
    # carries through consistent with Dashboard / Online Simulation
    # severity coding.
    _ACTION_GUIDANCE = {
        "isolate":     ("critical", "Isolate device from network",
                        "Block all non-essential connections while preserving clinical paths."),
        "escalate":    ("high", "Escalate immediately",
                        "Notify security lead and clinical engineering on-call."),
        "investigate": ("high", "Investigate before acting",
                        "Gather more information. Check with Biomed for scheduled maintenance."),
        "monitor":     ("medium", "Monitor - no immediate action",
                        "Watch for escalation. Set alert for threshold change."),
        "dismiss":     ("low", "Dismiss - expected behavior",
                        "Verify with asset owner. Document reason for dismissal."),
    }
    tier_key, label, guidance = _ACTION_GUIDANCE.get(
        correct_action,
        ("low", "Review recommended",
         "Check response policy for this alert type."),
    )
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:12px;padding:14px 16px;'
        f'background:var(--tier-{tier_key}-bg);border-left:3px solid var(--tier-{tier_key});'
        f'border-radius:4px;margin-bottom:6px;">'
        f'{ui_mod.render_tier_glyph(tier_key.upper(), size_px=12)}'
        f'<div style="flex:1;"><div style="font-weight:500;color:var(--text-primary);'
        f'font-size:0.95rem;margin-bottom:2px;">{label}</div>'
        f'<div style="font-size:0.8rem;color:var(--text-secondary);">{guidance}</div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    # Keyboard shortcut hint (BA-20)
    st.markdown(
        '<div style="margin-top:24px;padding-top:12px;border-top:1px dashed var(--border-subtle);'
        'font-family:JetBrains Mono,monospace;font-size:10px;color:var(--text-quaternary);">'
        'Tip - click the Alert selector in the sidebar and use the up/down keys to scan. '
        'Use the Prev/Next buttons above to step through the filtered set.'
        '</div>',
        unsafe_allow_html=True,
    )


def _render_proxy_questions():
    """
    Q21 + Q22: proxy validation for clinical staff
    and management stakeholders.
    Shown once after all 20 alerts are completed.

    Sentinel restyle (S5): chrome only; Q21/Q22 wording and option text are
    spec-controlled and unchanged.
    """
    st.markdown(
        '<div style="padding:24px 0 8px;">'
        '<h1 class="font-display" style="font-size:2.25rem;margin:0 0 8px;'
        'letter-spacing:-0.025em;color:var(--text-primary);">Two final questions</h1>'
        '<p style="font-size:0.95rem;color:var(--text-secondary);margin:0;max-width:640px;">'
        'Based on the alerts you reviewed, please answer these two questions.</p>'
        '</div>',
        unsafe_allow_html=True,
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
import re as _re  # noqa: E402  — late import; pattern compiled near use site

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


def _render_scenario_context_header() -> None:
    """A1: persistent scenario-context banner (Sentinel card style).

    Replaces the per-alert italic prefix. Wording verbatim from the
    original prompt; styling changed to a stable session banner so it
    reads as orientation rather than per-alert filler.
    """
    role = st.session_state.get("participant_role", "")
    role_html = (
        f'<span class="font-mono" style="color:var(--accent);"> · role:</span> '
        f'<span class="font-mono">{role}</span>'
    ) if role else ""
    st.markdown(
        '<div style="background:var(--surface-1);border:1px solid var(--border-subtle);'
        'border-left:2px solid var(--accent);border-radius:4px;padding:12px 16px;'
        'margin-bottom:8px;">'
        '<div style="font-size:10px;font-weight:500;letter-spacing:0.08em;'
        'text-transform:uppercase;color:var(--text-tertiary);margin-bottom:4px;">Scenario context</div>'
        '<p style="font-size:0.875rem;color:var(--text-primary);margin:0;line-height:1.5;">'
        'You are the on-call IT security staff at a 300-bed hospital. Review the alert '
        'below and decide how to respond.'
        f'{role_html}'
        '</p></div>',
        unsafe_allow_html=True,
    )


def _render_action_reference_panel() -> None:
    """B1: action-vocabulary reference shown identically to both conditions.

    Isolates the IV (MVE explanation in Group B) from action-vocabulary
    literacy. Content lifted verbatim from `_ACTION_GUIDANCE` at L2260-2271
    of `browse_mode`. No severity hints in the reference text.
    """
    with st.expander("ⓘ What does each action mean? (reference)", expanded=False):
        st.markdown(
            '<div style="font-size:0.8125rem;line-height:1.55;color:var(--text-secondary);">'
            '<p style="margin:0 0 6px;color:var(--text-tertiary);font-size:11px;'
            'font-family:JetBrains Mono,monospace;">'
            'Reference only — shown to all participants. Pick the action that matches '
            'what you would do in real operations.</p>'
            '<ul style="margin:8px 0 0;padding-left:20px;">'
            '<li><strong style="color:var(--text-primary);">Isolate</strong> — block the '
            'device/system from the network. Preserves clinical paths where possible.</li>'
            '<li><strong style="color:var(--text-primary);">Escalate</strong> — notify security '
            'lead and clinical engineering on-call. Hand off ownership.</li>'
            '<li><strong style="color:var(--text-primary);">Investigate</strong> — gather '
            'more information before acting. Check with biomed for scheduled maintenance.</li>'
            '<li><strong style="color:var(--text-primary);">Monitor</strong> — no immediate '
            'action. Watch for escalation; set an alert on threshold changes.</li>'
            '<li><strong style="color:var(--text-primary);">Dismiss</strong> — expected '
            'behavior. Verify with the asset owner; document the reason for dismissal.</li>'
            '</ul></div>',
            unsafe_allow_html=True,
        )


def _study_sidebar_strip() -> None:
    """S6: Sentinel-styled participant strip in the sidebar.

    Additive only — shows participant_id, role, and session-elapsed minutes.
    No behavioral change; lets the facilitator confirm participant state
    at a glance.
    """
    pid = st.session_state.get("participant_id", "?")
    role = st.session_state.get("participant_role", "?")
    start = st.session_state.get("study_session_start")
    elapsed_min = int((time.time() - start) / 60) if start else 0
    st.sidebar.markdown(
        f'<div style="padding:12px 8px;border-top:1px solid var(--border);margin-top:12px;">'
        f'<div style="font-size:10px;font-weight:500;letter-spacing:0.08em;'
        f'text-transform:uppercase;color:var(--text-tertiary);margin-bottom:6px;">Participant</div>'
        f'<div class="font-mono" style="font-size:0.75rem;color:var(--text-primary);line-height:1.4;">'
        f'{pid}<br>'
        f'<span style="color:var(--text-tertiary);">{role}</span><br>'
        f'<span style="color:var(--text-tertiary);">session {elapsed_min} min</span>'
        f'</div></div>',
        unsafe_allow_html=True,
    )


def study_mode():
    """
    Phase 2 User Study — A/B design validating C4.
    Group A: raw IDS output only
    Group B: raw IDS + MVE (3-layer explanation)

    Sentinel restyle (Phase 3 Step 6.S): cosmetic chrome only. All behavioral
    contracts preserved — assign_ab_condition, load_study_alerts order, form
    fields, scoring keys, audit event names. See
    `docs/dashboard_design_memo.md` Step 6.S for the hard-preserve list.
    """
    from module6_evaluation.sentinel_theme import inject_theme
    from module6_evaluation.study_loader import (
        load_study_alerts, assign_ab_condition
    )

    inject_theme()  # S1: Sentinel palette + fonts inherited from theme module

    # S6: Sidebar participant strip — additive once registered.
    if st.session_state.get("study_started") and st.session_state.get("participant_id"):
        _study_sidebar_strip()

    # ── Registration ──────────────────────────────────────────
    if not st.session_state.study_started:
        # S4: registration card with Sentinel palette
        st.markdown(
            '<div style="padding:24px 0 8px;">'
            '<h1 class="font-display" style="font-size:2.25rem;margin:0 0 8px;'
            'letter-spacing:-0.025em;color:var(--text-primary);">'
            'Healthcare IDS Alert Evaluation Study</h1>'
            '<p style="font-size:0.95rem;color:var(--text-secondary);margin:0;max-width:640px;">'
            'Evaluate how security alert information helps IT staff make response '
            'decisions. Time required: 30–40 minutes. You will review 20 security '
            'alerts and decide how to respond to each one.</p>'
            '</div>',
            unsafe_allow_html=True,
        )

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

            if st.form_submit_button("Begin Study", type="primary") and pid and consent:
                st.session_state.participant_id = pid
                st.session_state.participant_role = role
                st.session_state.participant_years = years_exp
                st.session_state.participant_ids_exp = has_ids_exp
                st.session_state.study_started = True
                st.session_state.current_alert = 0
                st.session_state.responses = []
                st.session_state.alert_start_time = time.time()
                st.session_state.study_session_start = time.time()
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
        # S7: Sentinel completion card (replaces st.success / st.info)
        responses = st.session_state.responses
        n = len(responses)

        save_path = (
            PROJECT_ROOT / "results" / "reports" /
            f"study_responses_{st.session_state.participant_id}.json"
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(
            json.dumps(responses, indent=2), encoding="utf-8"
        )

        st.markdown(
            f'<div style="padding:48px 0 24px;text-align:center;">'
            f'<h1 class="font-display" style="font-size:2.5rem;margin:0 0 8px;'
            f'letter-spacing:-0.025em;color:var(--text-primary);">Session complete</h1>'
            f'<p style="font-size:0.95rem;color:var(--text-secondary);margin:0 0 24px;">'
            f'Thank you for participating. Your responses have been captured to disk; '
            f'aggregate results will be shared after the study concludes.</p>'
            f'<div style="display:inline-block;padding:20px 32px;background:var(--surface-1);'
            f'border:1px solid var(--border-subtle);border-radius:4px;text-align:left;">'
            f'<div style="font-size:10px;font-weight:500;letter-spacing:0.08em;'
            f'text-transform:uppercase;color:var(--text-tertiary);margin-bottom:8px;">Session summary</div>'
            f'<div class="font-mono" style="font-size:0.875rem;color:var(--text-primary);">'
            f'participant <span style="color:var(--accent);">{st.session_state.participant_id}</span>'
            f' · alerts reviewed <span style="color:var(--accent);">{n}</span>'
            f' · responses captured to <span style="color:var(--text-tertiary);">'
            f'{save_path.name}</span></div></div></div>',
            unsafe_allow_html=True,
        )

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

    # S3: Sentinel-styled progress strip. Behavior unchanged (same text;
    # same N-of-20 numerator); only the chrome changes. The condition label
    # (with/without MVE) is intentionally NOT shown to the participant —
    # condition is between-subjects and revealing it could bias the
    # response. Facilitator sees it via the audit log.
    pct = int(round((current_idx / n_total) * 100))
    st.markdown(
        f'<div style="padding:12px 0 16px;">'
        f'  <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:6px;">'
        f'    <span class="font-mono" style="font-size:11px;text-transform:uppercase;'
        f'letter-spacing:0.08em;color:var(--text-tertiary);">Alert {current_idx + 1} of {n_total}</span>'
        f'    <span class="font-mono" style="font-size:11px;color:var(--text-tertiary);">{pct}%</span>'
        f'  </div>'
        f'  <div style="height:3px;background:var(--surface-3);border-radius:2px;overflow:hidden;">'
        f'    <div style="height:100%;width:{pct}%;background:var(--accent);transition:width 240ms ease;"></div>'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── A1: Persistent scenario-context header ──────────────────
    # Replaces the per-alert italic prefix. Wording unchanged verbatim;
    # styled as a Sentinel card so it reads as a stable session banner
    # rather than per-alert filler.
    _render_scenario_context_header()

    # ── Alert display ──────────────────────────────────────────
    st.markdown(
        f'<h3 class="font-display" style="font-size:1.5rem;margin:16px 0 12px;'
        f'letter-spacing:-0.02em;color:var(--text-primary);">Alert {current_idx + 1}</h3>',
        unsafe_allow_html=True,
    )

    # Show Group A or Group B content (contract unchanged)
    if show_mve:
        _render_group_b_highlighted(alert.group_b_display)
    else:
        st.code(alert.group_a_display, language=None)

    # ── Response form ──────────────────────────────────────────
    st.markdown(
        '<div style="height:1px;background:var(--border-subtle);margin:20px 0 16px;"></div>'
        '<h4 class="font-display" style="font-size:1.125rem;margin:0 0 12px;'
        'letter-spacing:-0.01em;color:var(--text-primary);">Your decision</h4>',
        unsafe_allow_html=True,
    )

    # B2: bypass st.form so that the submit button can be disabled until
    # both required selections are made. Widget keys are scoped per
    # current_idx so values don't bleed across alerts.
    severity = st.radio(
        "1. How severe is this alert? *(select one)*",
        ["CRITICAL — Respond immediately",
         "HIGH — Respond within 1 hour",
         "MEDIUM — Respond within 4 hours",
         "LOW — Review within 24 hours"],
        index=None,
        key=f"sev_{current_idx}",
    )

    action = st.radio(
        "2. What action would you take? *(select one)*",
        ["Isolate the device/system from the network",
         "Escalate to clinical staff / senior management",
         "Investigate further before taking action",
         "Monitor closely but no immediate action",
         "Dismiss — this is likely a false alarm"],
        index=None,
        key=f"act_{current_idx}",
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
        }[x],
        key=f"conf_{current_idx}",
    )

    # B1: Action-vocabulary reference expander. Shown identically to both
    # conditions to isolate the IV from action-vocabulary literacy.
    _render_action_reference_panel()

    # B2 (continued): inline validation — button stays disabled until both
    # required radios are answered. The current value is `None` when the
    # participant has not yet clicked an option (because index=None above).
    ready = severity is not None and action is not None
    submit_hint = (
        '<p class="font-mono" style="font-size:10px;color:var(--text-tertiary);'
        'margin:8px 0 0;text-align:right;">'
        '— select a severity and an action to enable submission</p>'
    ) if not ready else ""
    if submit_hint:
        st.markdown(submit_hint, unsafe_allow_html=True)

    submitted = st.button(
        "Submit & Next Alert →",
        type="primary",
        width="stretch",
        disabled=not ready,
        key=f"submit_{current_idx}",
    )

    if submitted and ready:
        elapsed = round(time.time() - st.session_state.alert_start_time, 1)

        # Map display values to scoring values (verbatim)
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

        # Score response (logic unchanged)
        severity_correct = (chosen_severity == alert.correct_severity)
        action_correct = (chosen_action == alert.correct_action)

        LEVEL = {"CRITICAL": 3, "HIGH": 2, "MEDIUM": 1, "LOW": 0}
        sev_diff = abs(
            LEVEL.get(chosen_severity, -1) -
            LEVEL.get(alert.correct_severity, -1)
        )
        severity_score = 1.0 if sev_diff == 0 else (
            0.5 if sev_diff == 1 else 0.0
        )
        catastrophic = (sev_diff == 3)

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

        # C1: post-submit toast (st.toast survives one st.rerun by design)
        next_n = current_idx + 2  # human-friendly next-alert number
        if next_n <= n_total:
            st.toast(f"Response captured · advancing to alert {next_n}", icon="✅")
        else:
            st.toast("All 20 alerts complete · loading final questions", icon="✅")
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
