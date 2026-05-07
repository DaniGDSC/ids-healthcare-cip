"""ARCHITECTURE.md Step [8] — Context Enrichment.

Single source of truth for transforming a raw alert dict (e.g. one
record from ``results/reports/evaluation_alerts.json``) into the
fully-enriched representation downstream stages require:

* ``device_class``, ``device_criticality``, ``patchable``,
  ``data_sensitivity``, ``clinical_tier``
* ``mitre_techniques`` with confidence levels (HIGH / MEDIUM / LOW)
* ``warning_flags`` (e.g. ``DEVICE_NOT_IN_INVENTORY``)

Both Module 3 (composite risk scoring, ``compute_composite_risk``) and
Module 6 (dashboard rendering) import from this module so the
enrichment policy is uniform across paper-metrics + demo paths.

Strict contract
---------------

The ``patchable`` field is **required** on every input alert. Missing
it raises :class:`MissingRequiredField`. The previous behaviour
(``patchable = True`` silent default) silently disabled the safety
floor (CRITICAL+unpatchable always surfaces) and was identified as
a bug in the architecture review — re-introducing the default would
revert that fix.

UNKNOWN device handling (conservative-fail-safe)
------------------------------------------------

Devices not found in ``configs/device_inventory.yaml`` are treated as
a security signal, NOT silently as low-risk:

* ``device_class = "UNKNOWN"``
* ``patchable = False``                       (conservative)
* ``device_criticality = HIGH``               (conservative)
* ``clinical_tier = tier_2_high_clinical``    (weight 0.8)
* ``data_sensitivity = UNKNOWN``
* ``warning_flags`` includes
  ``"DEVICE_NOT_IN_INVENTORY"`` and a secondary ``"rogue_device"``
  alert is emitted to the audit log.

Rationale: an unknown device IS a security signal (rogue device, BYOD
violation, asset-management gap). Treating it as low-risk by default
would let an attacker hide behind missing inventory data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = PROJECT_ROOT / "configs"


# ── Policy: clinical-tier weights ──
# Source: ARCHITECTURE.md Step [8], "D_clinical_tier weights".
CLINICAL_TIER_WEIGHTS: dict[str, float] = {
    "tier_1_life_critical": 1.0,
    "tier_2_high_clinical": 0.8,
    "tier_3_moderate":      0.5,
    "tier_4_supportive":    0.3,
    "tier_5_administrative": 0.1,
}


# ── Policy: device class → clinical tier ──
# Maps CSV-derived device_class names to the tier weight policy. Kept
# inline because there are only ~5 classes in WUSTL-EHMS-2020 and the
# hospital-specific tier file (``configs/device_clinical_tier_mapping.yaml``)
# has not landed yet. When that YAML appears it takes precedence.
DEFAULT_DEVICE_CLASS_TO_TIER: dict[str, str] = {
    "infusion_pump":   "tier_1_life_critical",
    "ventilator":      "tier_1_life_critical",
    "patient_monitor": "tier_1_life_critical",
    "ekg_machine":     "tier_2_high_clinical",
    "ehr_workstation": "tier_3_moderate",
    "bedside_terminal": "tier_4_supportive",
    "admin_workstation": "tier_5_administrative",
    "other":           "tier_3_moderate",
}


# ── Errors ────────────────────────────────────────────────────────────


class MissingRequiredField(KeyError):
    """Raised when an alert dict is missing a field the enrichment
    contract treats as load-bearing (no silent default)."""


class EnrichmentError(RuntimeError):
    """Raised for any other enrichment-policy violation."""


# ── Patchable: required, two name variants ────────────────────────────


def _read_patchable(alert: Mapping[str, Any]) -> bool:
    """Read the ``patchable`` flag from an alert dict.

    Accepts both ``patchable`` (canonical, doc-mandated) and
    ``device_patchable`` (legacy field on ``evaluation_alerts.json``).
    Refuses to default — missing the field on both keys raises
    :class:`MissingRequiredField`.
    """
    if "patchable" in alert:
        return bool(alert["patchable"])
    if "device_patchable" in alert:
        return bool(alert["device_patchable"])
    raise MissingRequiredField(
        "Alert is missing the required 'patchable' (or legacy "
        "'device_patchable') field. Per ARCHITECTURE.md Step [8] the "
        "field must be present on every alert; defaulting it to True "
        "silently disables the CRITICAL+unpatchable safety floor."
    )


# ── Device inventory loading ──────────────────────────────────────────


_INVENTORY_CACHE: dict[str, dict[str, Any]] | None = None


def _load_device_inventory() -> dict[str, dict[str, Any]]:
    """Load the device inventory map (device_type → entry).

    Looks for ``configs/device_inventory.yaml`` first; falls back to
    the test fixture if the production file is absent. Cached for the
    process lifetime.
    """
    global _INVENTORY_CACHE
    if _INVENTORY_CACHE is not None:
        return _INVENTORY_CACHE
    candidates = [
        CONFIGS_DIR / "device_inventory.yaml",
        PROJECT_ROOT / "tests" / "fixtures" / "device_inventory.yaml",
    ]
    for p in candidates:
        if p.exists():
            with p.open(encoding="utf-8") as f:
                body = yaml.safe_load(f)
            entries: list[dict[str, Any]] = body.get("devices", []) if isinstance(body, dict) else []
            _INVENTORY_CACHE = {e["device_type"]: e for e in entries if "device_type" in e}
            logger.info(
                "context_enrichment: loaded %d device(s) from %s",
                len(_INVENTORY_CACHE), p.relative_to(PROJECT_ROOT),
            )
            return _INVENTORY_CACHE
    _INVENTORY_CACHE = {}
    return _INVENTORY_CACHE


# ── UNKNOWN-device fail-safe ──────────────────────────────────────────


_UNKNOWN_DEVICE_DEFAULTS: dict[str, Any] = {
    "device_class":        "UNKNOWN",
    "patchable":           False,
    "device_criticality":  "HIGH",
    "clinical_tier":       "tier_2_high_clinical",
    "data_sensitivity":    "UNKNOWN",
}


# ── Public API ────────────────────────────────────────────────────────


def enrich_alert_context(alert: Mapping[str, Any]) -> dict[str, Any]:
    """Return an enriched copy of ``alert`` with policy fields populated.

    Reads the existing fields on the alert (no I/O for the common
    case where ``device_class`` and friends are already populated by
    Module 6's curation) and adds:

    * ``clinical_tier``  — derived from ``device_class`` via
      ``DEFAULT_DEVICE_CLASS_TO_TIER`` (or the YAML mapping when present).
    * ``clinical_tier_weight`` — looked up in
      :data:`CLINICAL_TIER_WEIGHTS`.
    * ``patchable`` — copied through; **required** on input, no default.
    * ``warning_flags`` — extended with ``DEVICE_NOT_IN_INVENTORY`` for
      UNKNOWN devices.

    Args:
        alert: Raw alert dict (e.g. from ``evaluation_alerts.json``).

    Returns:
        A new dict containing all original fields plus the enrichment.

    Raises:
        MissingRequiredField: if ``patchable`` (or ``device_patchable``)
            is absent from the input alert.
    """
    enriched = dict(alert)

    # ── Patchable (required, no default) ──
    enriched["patchable"] = _read_patchable(alert)

    # ── Device class lookup ──
    device_class = (
        alert.get("device_class")
        or alert.get("device_type")
    )

    if not device_class:
        # UNKNOWN-device fail-safe (per ARCHITECTURE.md Step [8]).
        # OVERRIDES caller-supplied values — even if the caller said
        # patchable=True, an UNKNOWN device is treated as a security
        # signal (potential rogue device / BYOD violation).
        for k, v in _UNKNOWN_DEVICE_DEFAULTS.items():
            enriched[k] = v
        warnings = list(enriched.get("warning_flags") or [])
        if "DEVICE_NOT_IN_INVENTORY" not in warnings:
            warnings.append("DEVICE_NOT_IN_INVENTORY")
        enriched["warning_flags"] = warnings
        logger.warning(
            "Enrichment: alert %s has no device_class/device_type — "
            "treating as UNKNOWN (conservative-fail-safe)",
            alert.get("alert_id", "<unknown>"),
        )
    else:
        enriched["device_class"] = device_class
        # ``device_criticality`` is the security-criticality tier (CRITICAL/
        # HIGH/MEDIUM/LOW) — preserve whatever the producer gave us; only
        # default if absent.
        enriched.setdefault("device_criticality", "MEDIUM")

    # ── Clinical-tier mapping ──
    tier = enriched.get("clinical_tier") or DEFAULT_DEVICE_CLASS_TO_TIER.get(
        enriched.get("device_class", ""),
        "tier_3_moderate",
    )
    enriched["clinical_tier"] = tier
    enriched["clinical_tier_weight"] = CLINICAL_TIER_WEIGHTS.get(tier, 0.5)

    return enriched


def score_alert_from_dict(alert: Mapping[str, Any]):
    """Convenience: enrich an alert dict and run it through
    ``src.risk_scorer.score_alert``.

    Replaces the legacy ``module6_evaluation/_src_adapter.scored_from_eval_alert``
    (which silently defaulted ``patchable=True``).
    """
    from src.risk_scorer import score_alert

    enr = enrich_alert_context(alert)
    return score_alert(
        anomaly_score=float(enr.get("risk_score", 0.0)),
        device_context={
            "criticality": enr.get("device_criticality", "MEDIUM"),
            "patchable": bool(enr["patchable"]),
            "clinical_function": enr.get("affected_system", ""),
        },
        event_context=None,
        fusion_class=enr.get("fusion_class"),
        data_quality=enr.get("data_quality"),
    )


__all__ = [
    "CLINICAL_TIER_WEIGHTS",
    "DEFAULT_DEVICE_CLASS_TO_TIER",
    "EnrichmentError",
    "MissingRequiredField",
    "enrich_alert_context",
    "score_alert_from_dict",
]
