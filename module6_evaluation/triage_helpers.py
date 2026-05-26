"""Pure helpers used by the dashboard triage column.

Extracted from ``module6_app.py`` so they're testable without bootstrapping
Streamlit's session machinery.
"""
from __future__ import annotations

from collections import Counter

import pandas as pd

from .constants import _ACTION_DISPLAY, _ACTION_DISPLAY_MISS


def floor_elevated(alert: dict) -> bool:
    """Return True if the alert tier should be floored to MEDIUM-or-higher.

    Floor-elevation is the dashboard's RQ3 invariant: any CRITICAL alert on
    a clinical/critical device, OR any HIGH alert with elevated patient
    acuity, gets a non-dismissable badge so triage queue ordering stays
    safety-aligned.
    """
    tier = str(alert.get("risk_level", "")).upper()
    if tier not in ("CRITICAL", "HIGH"):
        return False
    comps = alert.get("risk_components") or {}
    d_crit = float(comps.get("D_crit", 0.0) or 0.0)
    d_clin = float(comps.get("D_clinical_tier", 0.0) or 0.0)
    if tier == "CRITICAL":
        return d_crit >= 0.5 or d_clin >= 0.4
    return d_crit >= 0.7 or d_clin >= 0.5


def apply_dashboard_filters(
    responses: list,
    *,
    severity_floor: str | None = None,
    attack_category: str | None = None,
    search_text: str | None = None,
) -> list:
    """Return the subset of responses matching the dashboard filter bar."""
    if not responses:
        return []

    tier_rank = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
    floor = tier_rank.get(str(severity_floor or "").upper(), 0)
    cat = (attack_category or "").strip().lower()
    needle = (search_text or "").strip().lower()

    out = []
    for r in responses:
        tier = str(r.get("risk_level", "")).upper()
        if floor and tier_rank.get(tier, 0) < floor:
            continue
        if cat and str(r.get("attack_category", "")).lower() != cat:
            continue
        if needle:
            hay = " ".join(str(r.get(k, "")) for k in ("alert_id", "attack_category", "risk_level")).lower()
            if needle not in hay:
                continue
        out.append(r)
    return out


def compute_tier_counts(responses_tuple: tuple) -> dict:
    """Counter of alerts per tier. Tuple input is for ``@st.cache_data`` keying."""
    counts: Counter = Counter()
    for r in responses_tuple:
        counts[str(r.get("risk_level", "UNKNOWN")).upper()] += 1
    return dict(counts)


def build_feed_dataframe(responses_head: tuple) -> pd.DataFrame:
    """Compact DataFrame rendered into the dashboard's alert-feed table."""
    rows = []
    for r in responses_head:
        actions = r.get("response", {}).get("actions") if isinstance(r.get("response"), dict) else []
        rows.append({
            "sample_index": r.get("sample_index"),
            "alert_id": r.get("alert_id", ""),
            "risk_level": r.get("risk_level", ""),
            "risk_score": r.get("risk_score", 0.0),
            "attack_category": r.get("attack_category", ""),
            "actions": "|".join(actions) if isinstance(actions, list) else "",
        })
    return pd.DataFrame(rows)


def primary_action(actions: list) -> str:
    """Pick the operator-meaningful primary action from a policy action set.

    Highest-cost action wins; falls back to ``log_event`` for empty/unknown.
    """
    if not actions:
        return "log_event"
    return max(
        actions,
        key=lambda a: -_ACTION_DISPLAY.get(a, _ACTION_DISPLAY_MISS)[0],
    )


__all__ = [
    "floor_elevated",
    "apply_dashboard_filters",
    "compute_tier_counts",
    "build_feed_dataframe",
    "primary_action",
]
