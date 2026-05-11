"""Layer 5 v4.0 — presentation metadata + lookup helpers.

Pure-function module that returns the visual metadata
(colour, icon, label, urgency, symbol) the Streamlit dashboard at
``module6_evaluation/module6_app.py`` needs to render the v4 deltas:

  * 9-class :class:`AlertType` badges with the prompt's prescribed
    palette (see ``BADGE_FOR_ALERT_TYPE``)
  * 4-level :class:`Confidence` indicator
  * Mode A (LLM) / Mode B (rule-based) generation indicator
  * DAE anomalous-dims markdown line for Layer 1 of the alert card

This module imports nothing from Streamlit so the metadata can be
unit-tested without a UI runtime. The Streamlit app calls into these
helpers and renders the returned dicts as it sees fit.

Stability contract
------------------
The colour values are part of the dashboard's documented design
(``docs/9_alert_type_visualization.md``). Tests pin the exact hex
codes; if the design changes, update both the constants and the
tests in lockstep.
"""
from __future__ import annotations

from enum import Enum
from typing import Iterable, Sequence, TypedDict, TypeVar

from src.data_models import AlertType, Confidence


# ── Type aliases for the metadata dicts ────────────────────────────────


class BadgeStyle(TypedDict):
    """Visual metadata for an :class:`AlertType` badge."""
    color: str            # hex string, e.g. "#DC2626"
    icon: str             # single emoji
    label: str            # human-readable label rendered next to icon
    urgency: str          # "HIGH" | "MEDIUM" | "LOW" | "INFO"


class ConfidenceStyle(TypedDict):
    """Visual metadata for a :class:`Confidence` indicator."""
    symbol: str           # rendered next to the level name
    color: str            # plain colour name (Streamlit-friendly)


class ModeStyle(TypedDict):
    """Visual metadata for the Mode A/B indicator."""
    badge: str            # short human-readable badge text
    color: str            # plain colour name


# ── 9-type AlertType badges (prompt-mandated palette) ──────────────────

BADGE_FOR_ALERT_TYPE: dict[AlertType, BadgeStyle] = {
    AlertType.KNOWN_ATTACK: {
        "color": "#DC2626",
        "icon": "🔴",
        "label": "KNOWN ATTACK",
        "urgency": "HIGH",
    },
    AlertType.KNOWN_ATTACK_UNCERTAIN: {
        "color": "#DC2626",
        "icon": "🔴",
        "label": "KNOWN ATTACK (Uncertain)",
        "urgency": "HIGH",
    },
    AlertType.DISAGREEMENT_ANOMALY: {
        "color": "#9333EA",   # purple — distinct from threat-tone reds/oranges
        "icon": "🟣",
        "label": "ADVERSARIAL DETECTED",
        "urgency": "HIGH",
    },
    AlertType.STRONG_NOVEL_ANOMALY: {
        "color": "#EA580C",
        "icon": "🟠",
        "label": "STRONG NOVEL",
        "urgency": "MEDIUM",
    },
    AlertType.NOVEL_ANOMALY: {
        "color": "#F97316",
        "icon": "🟠",
        "label": "NOVEL ANOMALY",
        "urgency": "MEDIUM",
    },
    AlertType.CONFIRMED_ANOMALY: {
        "color": "#EAB308",
        "icon": "🟡",
        "label": "CONFIRMED ANOMALY",
        "urgency": "MEDIUM",
    },
    AlertType.SUSPICIOUS_PATTERN: {
        # Day 7: changed from #FACC15 to #F59E0B (amber) so the badge is
        # distinguishable from CONFIRMED_ANOMALY's #EAB308 yellow on a
        # defense-room projector at 2 m viewing distance — the two
        # tones collapse to "looks the same" when projector contrast is
        # reduced. Tests pin the new value (see Day 7 invariants).
        "color": "#F59E0B",
        "icon": "🟡",
        "label": "SUSPICIOUS",
        "urgency": "LOW",
    },
    AlertType.BENIGN_WATCH: {
        "color": "#94A3B8",
        "icon": "⚪",
        "label": "BENIGN WATCH",
        "urgency": "INFO",
    },
    AlertType.BENIGN: {
        "color": "#94A3B8",
        "icon": "⚪",
        "label": "BENIGN",
        "urgency": "INFO",
    },
}


_E = TypeVar("_E", bound=Enum)
_S = TypeVar("_S")


def _resolve_style(
    value: _E | str,
    enum_cls: type[_E],
    table: dict[_E, _S],
    default: _E,
) -> _S:
    """Look up *value* in *table*, defaulting on unknown enum strings.

    Shared resolver used by every "render-side" lookup that accepts
    either a typed enum or its raw string name.  Unknown strings fall
    back to *default* so the renderer never crashes on stale data while
    still emitting a recognisable cue the operator can interpret.
    """
    if isinstance(value, str):
        try:
            value = enum_cls(value)
        except ValueError:
            return table[default]
    return table[value]


def badge_for_alert_type(alert_type: AlertType | str) -> BadgeStyle:
    """Return the badge metadata for a v4 alert type.

    Unknown strings fall back to the BENIGN badge — see
    :func:`_resolve_style` for the shared default-on-unknown rule.
    """
    return _resolve_style(alert_type, AlertType, BADGE_FOR_ALERT_TYPE, AlertType.BENIGN)


# ── 4-level Confidence indicator ───────────────────────────────────────

CONFIDENCE_INDICATOR: dict[Confidence, ConfidenceStyle] = {
    Confidence.VERY_HIGH: {"symbol": "●●●●", "color": "green"},
    Confidence.HIGH:      {"symbol": "●●●",  "color": "green"},
    Confidence.MEDIUM:    {"symbol": "●●",   "color": "orange"},
    Confidence.LOW:       {"symbol": "●",    "color": "gray"},
}


def confidence_display(confidence: Confidence | str) -> ConfidenceStyle:
    """Return the visual metadata for a :class:`Confidence` level.

    Unknown strings fall back to the LOW indicator — see
    :func:`_resolve_style` for the shared default-on-unknown rule.
    """
    return _resolve_style(confidence, Confidence, CONFIDENCE_INDICATOR, Confidence.LOW)


# ── Mode A / Mode B indicator ──────────────────────────────────────────

MODE_A_LLM = "A_llm"
MODE_B_RULE_BASED = "B_rule_based"

MODE_INDICATOR: dict[str, ModeStyle] = {
    MODE_A_LLM:        {"badge": "✓ AI Mode (LLM)",        "color": "green"},
    MODE_B_RULE_BASED: {"badge": "⚠ Rule-based Fallback", "color": "orange"},
}


def mode_display(generation_mode: str) -> ModeStyle:
    """Visual metadata for the ``generation_mode`` field on an MVE.

    Anything other than the two canonical strings is treated as Mode B
    so the operator gets the more conservative cue.
    """
    return MODE_INDICATOR.get(
        generation_mode, MODE_INDICATOR[MODE_B_RULE_BASED],
    )


# ── DAE anomalous-dims rendering ───────────────────────────────────────

def anomalous_dims_markdown(
    anomalous_dims: Iterable[int],
    feature_names: Sequence[str],
    *,
    max_features: int = 5,
) -> str:
    """Render ``anomalous_dims`` from Layer 2 v4 as a markdown bullet
    list suitable for the alert-card "Show DAE anomaly details"
    expander.

    Returns an empty string when no dims are anomalous so the caller
    can decide whether to render the expander at all.
    """
    dims = [int(i) for i in anomalous_dims if 0 <= int(i) < len(feature_names)]
    if not dims:
        return ""
    head, tail = dims[:max_features], dims[max_features:]
    lines = [f"- `{feature_names[i]}` (dim {i})" for i in head]
    if tail:
        lines.append(f"- … and **{len(tail)}** more")
    count_line = (
        f"DAE flagged **{len(dims)}** anomalous "
        f"dimension{'s' if len(dims) != 1 else ''}:"
    )
    return count_line + "\n" + "\n".join(lines)


__all__ = [
    "BadgeStyle",
    "ConfidenceStyle",
    "ModeStyle",
    "BADGE_FOR_ALERT_TYPE",
    "CONFIDENCE_INDICATOR",
    "MODE_INDICATOR",
    "MODE_A_LLM",
    "MODE_B_RULE_BASED",
    "badge_for_alert_type",
    "confidence_display",
    "mode_display",
    "anomalous_dims_markdown",
]
