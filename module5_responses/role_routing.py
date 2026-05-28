"""Role-mismatch detection (Phase 3.2).

When an alert's SHAP top-category implies one stakeholder (e.g. a
biometric anomaly → clinical staff) but the routing engine assigned a
different primary role (e.g. IT Security via ``ATTACK_ROUTING``), the
alert is at risk of being silently mis-handled. This module derives
the "expected" primary role from the SHAP category and emits a
``RoutingWarning`` when it disagrees with what the response engine
proposed, so the dashboard / UI can offer a one-click reroute.

The mapping from SHAP category → expected primary role is intentional
about *who can act on what*:

  biometric        → Clinical Engineering   (verify patient + device sensor)
  network_volume   → IT Security            (network throttle / firewall)
  network_protocol → IT Security            (port block / IDS tune)
  network_timing   → IT Security            (packet capture)
  network_packet   → IT Security
  network_loss     → IT Security
  unknown          → IT Security            (default escalation)

This module is consumed by ``module5_responses.pipeline`` (and the
phase1 regen tool) when assembling each alert record, so the warning
sits on ``Response`` alongside ``escalation_chain``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


# ── SHAP category → expected role ──────────────────────────────────


_CATEGORY_TO_EXPECTED_ROLE: dict[str, str] = {
    "biometric":         "Clinical Engineering",
    "network_volume":    "IT Security",
    "network_protocol":  "IT Security",
    "network_timing":    "IT Security",
    "network_packet":    "IT Security",
    "network_loss":      "IT Security",
    "unknown":           "IT Security",
}


# Roles considered equivalent for mismatch purposes — a routing primary
# of "Security lead" or "SOC" is operationally the same audience as
# "IT Security" and should NOT trigger a warning. Same for the various
# clinical engineering names.
_ROLE_EQUIVALENTS: dict[str, set[str]] = {
    "IT Security": {
        "it security", "security lead", "soc", "ciso", "incident commander",
        "network admin", "department it admin", "privacy officer",
    },
    "Clinical Engineering": {
        "clinical engineering", "biomedical engineering", "biomed engineering",
        "charge nurse", "on-call physician", "icu charge nurse",
        "floor charge nurse", "icu/floor charge nurse",
    },
}


def _normalise_role(role: str) -> str | None:
    """Map a routing-table role string to one of the two canonical
    audiences (``"IT Security"`` or ``"Clinical Engineering"``), or
    None when no match — None means we can't classify, so the caller
    will skip the warning rather than emit a noisy false positive.
    """
    if not role:
        return None
    rl = role.lower().strip()
    for canonical, aliases in _ROLE_EQUIVALENTS.items():
        if rl in aliases or any(a in rl for a in aliases):
            return canonical
    return None


# ── Warning dataclass ─────────────────────────────────────────────


@dataclass(frozen=True)
class RoutingWarning:
    """Single per-alert routing warning.

    ``suggested_role`` is the canonical audience the SHAP-top category
    implies. ``current_primary`` is what the response engine actually
    set. ``reason`` is a one-sentence explanation a non-ML user can
    understand without reading the SHAP table.
    """
    mismatch: bool
    current_primary: str
    suggested_role: str
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


# ── Detection ─────────────────────────────────────────────────────


def detect_routing_mismatch(
    top_category: str,
    routing_primary: str | None,
) -> RoutingWarning:
    """Compare the SHAP-derived expected role with the routing engine's
    primary, and return a ``RoutingWarning``.

    Args:
        top_category: SHAP top-category for the alert (e.g.
            ``"biometric"``, ``"network_volume"``). Sourced from
            ``module4_explanations.feature_groups._feature_to_narrative``.
        routing_primary: The ``primary`` field from the response
            engine's ``escalation_chain``. May be ``None`` (no routing
            applied — e.g. NORMAL alert), in which case no mismatch is
            flagged.

    Returns:
        ``RoutingWarning`` with ``mismatch=False`` when:
          - ``routing_primary`` is None / empty (nothing to compare)
          - the SHAP category is unknown (can't infer expected)
          - the routing primary normalises to the same canonical
            audience as the expected role

        ``RoutingWarning`` with ``mismatch=True`` otherwise, carrying
        the suggested alternate role and a non-ML-friendly reason.
    """
    expected = _CATEGORY_TO_EXPECTED_ROLE.get(top_category)
    if not expected or not routing_primary:
        return RoutingWarning(
            mismatch=False,
            current_primary=routing_primary or "",
            suggested_role="",
            reason="",
        )

    current_canonical = _normalise_role(routing_primary)
    if current_canonical == expected:
        return RoutingWarning(
            mismatch=False,
            current_primary=routing_primary,
            suggested_role=expected,
            reason="",
        )

    reason = (
        f"Top SHAP signal is in category '{top_category}', which is "
        f"operationally handled by {expected}. The response engine "
        f"routed the alert to '{routing_primary}' ({current_canonical or 'unknown audience'}). "
        f"Recommend rerouting so the alert reaches an operator who "
        f"can act on the underlying signal."
    )
    return RoutingWarning(
        mismatch=True,
        current_primary=routing_primary,
        suggested_role=expected,
        reason=reason,
    )


__all__ = [
    "RoutingWarning",
    "detect_routing_mismatch",
]
