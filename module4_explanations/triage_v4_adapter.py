"""Layer 4 v4.0 — adapter from the v4 enriched triage typology to the
existing 5-template MVE generator.

The MVE generator in ``src.mve_generator`` was built around the legacy
5-class alert type vocabulary (``T1``…``T5``) — the templates
(rule-based and LLM prompts) and the role-lens helpers all key off
those strings. Layer 3 v4 introduced a 9-class :class:`AlertType` and
a 4-level :class:`Confidence` indicator
(see ``module3_risk_scoring/triage_v4.py``); this module is the thin
shim that lets the v4 outputs flow through the existing generator
without duplicating any of the template machinery.

Responsibilities
----------------

1. **Template routing** — map every v4 ``AlertType`` to a legacy
   ``T1``…``T5`` so callers can keep using ``generate_mve`` and the
   role-lens helpers without modification.

2. **Adversarial flag** — recognise the v4-only
   ``DISAGREEMENT_ANOMALY`` so callers can surface adversarial-input
   wording in Layer 1 even though the underlying template is reused.

3. **Per-role MITRE visibility** — render the same MITRE technique
   differently for IT generalist (full ID + technique name), biomed
   engineer (threat-type prose), and nurse manager (plain-language
   prose). Honest naming of an existing requirement, not a behaviour
   change.

4. **DAE per-dim Layer 1 enrichment** — turn a list of anomalous
   feature indices (from Layer 2's per-dim DAE errors) into a one-
   sentence Layer 1 clause naming the features.

5. **Confidence rendering** — turn the v4 ``Confidence`` enum into a
   one-line string suitable for Layer 1's ``confidence_indicator``
   field.

This module performs no I/O and loads no models.
"""
from __future__ import annotations

from typing import Iterable, Sequence

from src.data_models import AlertType, Confidence, OperatorRole
from src.mve_generator import attck_for_alert_type


# ── 1. v4 AlertType → legacy template id (T1..T5) ───────────────────────
#
# ``T1`` (anomalous outbound) is the broadest "known attack" template
# in the legacy generator and is the right home for any alert type the
# v4 typology classifies as a known/confirmed attack. The
# adversarial-disagreement and novelty branches map onto ``T1`` for now
# (the prompt allows reusing the NOVEL template) and the special
# behaviours are layered on via the ``adversarial`` flag and the
# anomalous-dims clause; we deliberately do NOT introduce new template
# ids here because the rule-based generator's templates would have to
# be re-implemented for them.

_V4_TO_LEGACY_TEMPLATE: dict[AlertType, str] = {
    AlertType.KNOWN_ATTACK: "T1",
    AlertType.KNOWN_ATTACK_UNCERTAIN: "T1",
    AlertType.DISAGREEMENT_ANOMALY: "T1",   # adversarial flag on top
    AlertType.STRONG_NOVEL_ANOMALY: "T1",
    AlertType.NOVEL_ANOMALY: "T1",
    AlertType.CONFIRMED_ANOMALY: "T1",
    AlertType.SUSPICIOUS_PATTERN: "T1",
    AlertType.BENIGN_WATCH: "T1",
    AlertType.BENIGN: "T1",                 # caller decides whether to surface
}


def alert_type_v4_to_legacy(alert_type: AlertType | str) -> str:
    """Return the legacy ``T1``…``T5`` template id for a v4 alert type.

    Accepts either an :class:`AlertType` or the raw string for ergonomics
    (audit-log payloads round-trip through string).
    """
    if isinstance(alert_type, str):
        try:
            alert_type = AlertType(alert_type)
        except ValueError:
            return "T1"
    return _V4_TO_LEGACY_TEMPLATE.get(alert_type, "T1")


def is_adversarial(alert_type: AlertType | str) -> bool:
    """True iff the v4 alert type signals a potential adversarial
    input (currently only ``DISAGREEMENT_ANOMALY``).
    """
    if isinstance(alert_type, str):
        try:
            alert_type = AlertType(alert_type)
        except ValueError:
            return False
    return alert_type == AlertType.DISAGREEMENT_ANOMALY


# ── 2. Confidence rendering ─────────────────────────────────────────────

_CONFIDENCE_PROSE: dict[Confidence, str] = {
    Confidence.VERY_HIGH:
        "Confidence: VERY HIGH — Track A high probability and ensemble agreement.",
    Confidence.HIGH:
        "Confidence: HIGH — multi-signal corroboration.",
    Confidence.MEDIUM:
        "Confidence: MEDIUM — partial signal, manual review recommended.",
    Confidence.LOW:
        "Confidence: LOW — marginal indicators only.",
}


def confidence_clause(confidence: Confidence | str) -> str:
    """Render the v4 ``Confidence`` enum as a one-line Layer 1 clause."""
    if isinstance(confidence, str):
        try:
            confidence = Confidence(confidence)
        except ValueError:
            return "Confidence: UNKNOWN."
    return _CONFIDENCE_PROSE[confidence]


# ── 3. Per-role MITRE visibility ────────────────────────────────────────

_TECHNIQUE_PLAIN_LANGUAGE: dict[str, str] = {
    "T1071": "Equipment may be communicating with an unauthorized external system.",
    "T1078": "An authorized account may be compromised or being misused.",
    "T1021": "Equipment may be accessed by unauthorized parties on the network.",
    "T1041": "Patient information may be transmitted to unauthorized recipients.",
    "T1565": "Patient measurements or device data could be incorrect.",
}

_TECHNIQUE_BIOMED_PROSE: dict[str, str] = {
    "T1071": "Network communication consistent with attacker remote control.",
    "T1078": "Authorised credentials being misused on the device.",
    "T1021": "Unauthorised access to the device via legitimate channels.",
    "T1041": "Patient or device data may be leaving the organisation.",
    "T1565": "Patient data or device readings may be altered.",
}


def format_mitre_for_role(
    technique_id: str,
    technique_name: str,
    role: OperatorRole | str,
) -> str:
    """Render a MITRE technique in the format appropriate to a role.

    * IT generalist     — ``"T1071 (Application Layer Protocol)"``
    * Biomed engineer   — short threat-type prose
    * Nurse manager     — plain-language sentence, no jargon

    Unknown technique ids fall through to the technique name (or the
    raw id) so the renderer is total — it never raises.
    """
    if isinstance(role, OperatorRole):
        role_value = role.value
    else:
        role_value = str(role)

    if role_value == OperatorRole.IT_GENERALIST.value:
        if technique_id and technique_name:
            return f"{technique_id} ({technique_name})"
        return technique_id or technique_name or "no MITRE grounding"

    if role_value == OperatorRole.BIOMED_ENGINEER.value:
        return _TECHNIQUE_BIOMED_PROSE.get(
            technique_id,
            technique_name or "Unusual device behaviour — investigate.",
        )

    if role_value == OperatorRole.NURSE_MANAGER.value:
        return _TECHNIQUE_PLAIN_LANGUAGE.get(
            technique_id,
            "Equipment behaviour appears unusual — monitor and escalate.",
        )

    # Unknown role — default to the IT-generalist format.
    if technique_id and technique_name:
        return f"{technique_id} ({technique_name})"
    return technique_id or technique_name or "no MITRE grounding"


def format_mitre_for_alert_type(
    alert_type: AlertType | str,
    role: OperatorRole | str,
) -> str:
    """Convenience wrapper: derive (technique_id, technique_name) from
    the v4 alert type via the legacy template mapping, then format for
    the role.
    """
    legacy = alert_type_v4_to_legacy(alert_type)
    technique_id, technique_name = attck_for_alert_type(legacy)
    return format_mitre_for_role(technique_id, technique_name, role)


# ── 4. DAE per-dim Layer 1 enrichment ───────────────────────────────────

def anomalous_dims_clause(
    anomalous_dims: Iterable[int],
    feature_names: Sequence[str],
    *,
    max_features: int = 3,
) -> str:
    """Turn a list of DAE-anomalous feature indices into a Layer 1
    sentence naming up to ``max_features`` of them.

    Returns an empty string when ``anomalous_dims`` is empty so the
    caller can decide whether to drop the clause entirely; that is a
    cleaner failure mode than rendering "Anomalous features: none."

    Args:
        anomalous_dims: Indices into ``feature_names`` (typically the
            cascade-feature names list, length 28 = 25 raw + 3 Track A
            probas, but the function is agnostic to that).
        feature_names: Names matching the index space.
        max_features: Cap the rendered list — Layer 1 has a 60-word
            budget, so naming 3 features is the practical sweet spot.
    """
    dims = [int(i) for i in anomalous_dims if 0 <= int(i) < len(feature_names)]
    if not dims:
        return ""
    names = [feature_names[i] for i in dims[:max_features]]
    if len(dims) > max_features:
        suffix = f" and {len(dims) - max_features} more"
    else:
        suffix = ""
    if len(names) == 1:
        return f"DAE flagged feature {names[0]}{suffix}."
    return (
        "DAE flagged features "
        + ", ".join(names[:-1])
        + f" and {names[-1]}{suffix}."
    )


# ── 5. Adversarial wording for DISAGREEMENT_ANOMALY ─────────────────────

ADVERSARIAL_LAYER_1_HINT: str = (
    "Model disagreement detected — potential adversarial input. "
    "Verify input integrity before acting on the alert."
)


def adversarial_clause(alert_type: AlertType | str) -> str:
    """Return the adversarial Layer 1 hint for ``DISAGREEMENT_ANOMALY``;
    empty string for anything else.
    """
    return ADVERSARIAL_LAYER_1_HINT if is_adversarial(alert_type) else ""


__all__ = [
    "alert_type_v4_to_legacy",
    "is_adversarial",
    "confidence_clause",
    "format_mitre_for_role",
    "format_mitre_for_alert_type",
    "anomalous_dims_clause",
    "adversarial_clause",
    "ADVERSARIAL_LAYER_1_HINT",
]
