"""Regression guard for the MVE-Tier wording alignment.

Background: before Option 4, the dashboard's Layer 1 ("Why anomalous")
fell back to ``clinician_summary`` whose template prefixes the text with
``"CRITICAL ALERT"`` / ``"HIGH ALERT"`` / ``"MODERATE ALERT"`` / ``"LOW
ALERT"`` based on Module 4's *detection-consensus* severity (how many
models flagged). Layer 2's Tier badge, in contrast, reads Module 3's
*composite risk* tier (which weights device criticality and data
sensitivity). Those two severities can legitimately disagree, but mixing
them inside a single alert view was a UX bug — operators reading "Tier:
CRITICAL" above "MODERATE ALERT" body text concluded the data was wrong.

Option 4 fixed this by attaching a 3-layer ``MVEPayload`` from
``src.mve_generator`` to each record (under ``explanation.mve``) so the
dashboard's Layer 1 sources its text from the device-aware MVE generator
instead of the consensus-prefixed clinician summary.

These tests guard the fix:

* Every record in ``alert_responses{,_demo}.json`` carries an ``mve``
  payload (no silent regression to ``None``).
* ``mve.why_anomalous`` and the per-layer fields never contain a
  consensus-severity prefix that would clash with the Tier badge.
* ``mve.layer_2.severity_label`` is one of the four valid tiers (not the
  consensus label that previously leaked through).

Tests skip cleanly when the artefact files are absent so a fresh clone
without a Module 5 run does not red the CI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.alert_response_schema import AlertResponsesEnvelope

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS = PROJECT_ROOT / "results/reports"

_SPLIT_ARTIFACTS = [
    ("test", REPORTS / "alert_responses.json"),
    ("demo", REPORTS / "alert_responses_demo.json"),
]

# Tokens that, when found in Layer 1 text, indicate the old
# consensus-prefixed clinician_summary leaked through. Order-sensitive
# substring match — must appear as a standalone "ALERT" prefix, not in
# arbitrary prose like "the alert was triaged".
_CONSENSUS_PREFIXES = (
    "CRITICAL ALERT",
    "HIGH ALERT",
    "MODERATE ALERT",
    "LOW ALERT",
)

_VALID_TIERS = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}


def _load_records(path: Path) -> list[dict]:
    """Load and validate the envelope, returning records as plain dicts."""
    if not path.exists():
        pytest.skip(f"{path.name} not present — run module5_responses first")
    with open(path) as f:
        raw = json.load(f)
    envelope = AlertResponsesEnvelope.model_validate(raw)
    return [r.model_dump() for r in envelope.records]


@pytest.mark.parametrize(("split", "path"), _SPLIT_ARTIFACTS, ids=[s for s, _ in _SPLIT_ARTIFACTS])
def test_every_record_has_mve(split: str, path: Path) -> None:
    """Every alert record must carry an MVE payload after Module 5 ran.

    The schema marks ``mve`` as ``Optional[MVEPayload]`` so a missing
    field validates; this test enforces the stronger invariant that
    no record actually falls through to ``None`` on a real pipeline run.
    """
    records = _load_records(path)
    missing = [r["sample_index"] for r in records if r["explanation"].get("mve") is None]
    assert not missing, (
        f"{split}: {len(missing)} records have no MVE payload (sample_index "
        f"head: {missing[:10]}). build_all_records should always produce one — "
        f"check the try/except wrapping generate_mve in module5_responses."
    )


@pytest.mark.parametrize(("split", "path"), _SPLIT_ARTIFACTS, ids=[s for s, _ in _SPLIT_ARTIFACTS])
def test_layer_1_has_no_consensus_prefix(split: str, path: Path) -> None:
    """Layer 1 text must not start with CLINICIAN_TEMPLATES severity prefix.

    This is the regression that motivated Option 4 — guard against a
    future change in mve_generator that pulls clinician_summary back into
    Layer 1, or a template revision that adds an ``"… ALERT"`` prefix.
    """
    records = _load_records(path)
    violations = []
    for r in records:
        mve = r["explanation"]["mve"]
        if mve is None:
            continue
        why = mve.get("why_anomalous", "")
        for prefix in _CONSENSUS_PREFIXES:
            if prefix in why:
                violations.append(
                    (r["sample_index"], r["risk_level"], prefix, why[:80])
                )
                break
    assert not violations, (
        f"{split}: {len(violations)} records have a consensus-severity prefix "
        f"in Layer 1 text. First 5: {violations[:5]}"
    )


@pytest.mark.parametrize(("split", "path"), _SPLIT_ARTIFACTS, ids=[s for s, _ in _SPLIT_ARTIFACTS])
def test_layer_2_severity_label_is_valid_tier(split: str, path: Path) -> None:
    """Layer 2's ``severity_label`` must be a known tier string.

    The MVE's Layer 2 severity_label comes from the device-criticality
    branch of mve_generator. The dashboard's Layer 2 badge prefers
    Module 3's ``risk_level`` (set on the alert dict by process_alert),
    but mve.layer_2.severity_label is the fallback when risk_level is
    absent — so it must still parse as a tier.
    """
    records = _load_records(path)
    violations = []
    for r in records:
        mve = r["explanation"]["mve"]
        if mve is None:
            continue
        sev = mve.get("layer_2", {}).get("severity_label", "")
        if sev not in _VALID_TIERS:
            violations.append((r["sample_index"], sev))
    assert not violations, (
        f"{split}: {len(violations)} records have invalid Layer 2 "
        f"severity_label (must be one of {sorted(_VALID_TIERS)}). "
        f"First 5: {violations[:5]}"
    )


@pytest.mark.parametrize(("split", "path"), _SPLIT_ARTIFACTS, ids=[s for s, _ in _SPLIT_ARTIFACTS])
def test_mve_layers_have_required_keys(split: str, path: Path) -> None:
    """Each layer dict must carry the keys the dashboard's ChainMap expects.

    ``render_mve_layers._get`` looks up the per-layer field names
    (``baseline_behavior``, ``severity_label``, ``immediate_action``, …)
    directly via ChainMap. A missing key is a silent display gap, not a
    crash — this test makes the gap loud.
    """
    required = {
        "layer_1": {"baseline_behavior", "deviation_description", "confidence_indicator"},
        "layer_2": {"affected_system", "patient_care_impact", "severity_label",
                    "severity_rationale", "phi_exposure"},
        "layer_3": {"immediate_action", "clinical_constraint", "escalation_path",
                    "timeframe"},
    }
    records = _load_records(path)
    violations = []
    for r in records:
        mve = r["explanation"]["mve"]
        if mve is None:
            continue
        for layer_name, req_keys in required.items():
            present = set(mve.get(layer_name, {}).keys())
            missing = req_keys - present
            if missing:
                violations.append((r["sample_index"], layer_name, sorted(missing)))
                break
    assert not violations, (
        f"{split}: {len(violations)} records missing required MVE layer "
        f"keys. First 5: {violations[:5]}"
    )


@pytest.mark.parametrize(("split", "path"), _SPLIT_ARTIFACTS, ids=[s for s, _ in _SPLIT_ARTIFACTS])
def test_provider_is_known(split: str, path: Path) -> None:
    """The MVE provider tag must be one of the three documented values."""
    records = _load_records(path)
    valid = {"openai", "anthropic", "rule_based"}
    violations = []
    for r in records:
        mve = r["explanation"]["mve"]
        if mve is None:
            continue
        provider = mve.get("provider", "")
        if provider not in valid:
            violations.append((r["sample_index"], provider))
    assert not violations, (
        f"{split}: {len(violations)} records have unknown provider. "
        f"Valid: {sorted(valid)}. First 5: {violations[:5]}"
    )
