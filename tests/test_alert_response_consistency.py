"""Cross-source severity consistency for ``alert_responses{,_demo}.json``.

Module 3 emits ``risk_level`` (CRITICAL / HIGH / MEDIUM / LOW) from the
composite risk score; this is the canonical severity the response engine
acts on. Module 4 historically computed a parallel severity from
``n_models_flagged`` and used it to pick the clinician-summary template
prefix ("CRITICAL ALERT", "HIGH ALERT", "MODERATE ALERT", "LOW ALERT"),
and the MVE generator's ``layer_2.severity_label`` was free-text LLM
output validated only to be one of the four tiers — neither was checked
against ``risk_level``.

Result: alerts could ship with three disagreeing severities (e.g.
sample 61 of demo: risk_level=HIGH, clinician_summary=MODERATE,
mve.layer_2.severity_label=CRITICAL) — a safety-critical UX bug because
the clinician's "no immediate action required" text disagreed with the
response engine's HIGH-priority isolate_device action.

These tests enforce that all three severity surfaces of a single record
agree, treating ``risk_level`` as the single source of truth.

Tests skip when the artefact files are absent so a fresh clone without a
module 5 run does not red the CI.
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

# Map the clinician-summary template's first word back to a risk tier.
# CLINICIAN_TEMPLATES uses "MODERATE" instead of "MEDIUM" because
# "MODERATE" reads more naturally in plain-language clinical prose.
_PREFIX_TO_TIER = {
    "CRITICAL": "CRITICAL",
    "HIGH":     "HIGH",
    "MODERATE": "MEDIUM",
    "LOW":      "LOW",
}


def _load_records(path: Path) -> list[dict]:
    if not path.exists():
        pytest.skip(f"{path.name} not present — run module5_responses first")
    with open(path) as f:
        raw = json.load(f)
    envelope = AlertResponsesEnvelope.model_validate(raw)
    return [r.model_dump() for r in envelope.records]


def _summary_tier(summary: str) -> str | None:
    """Extract the tier word from the start of a clinician summary."""
    if not summary:
        return None
    first = summary.strip().split(None, 1)[0].upper()
    return _PREFIX_TO_TIER.get(first)


@pytest.mark.parametrize(
    ("split", "path"), _SPLIT_ARTIFACTS, ids=[s for s, _ in _SPLIT_ARTIFACTS]
)
def test_clinician_summary_severity_matches_risk_level(
    split: str, path: Path,
) -> None:
    """The summary's leading severity word must equal ``risk_level``.

    This is the test that would have caught sample 61 of demo:
    risk_level=HIGH but clinician_summary started with "MODERATE ALERT".
    """
    records = _load_records(path)
    violations = []
    for r in records:
        summary = r["explanation"]["clinician_summary"]
        tier = _summary_tier(summary)
        if tier is None:
            # Skip records whose summary doesn't start with a tier word
            # (e.g., legacy minimal summaries).
            continue
        if tier != r["risk_level"]:
            violations.append({
                "sample_index": r["sample_index"],
                "risk_level": r["risk_level"],
                "summary_tier": tier,
                "summary_head": summary[:80],
            })
    assert not violations, (
        f"{split}: {len(violations)} records have a clinician_summary "
        f"severity word that disagrees with risk_level. "
        f"First 5: {violations[:5]}"
    )


@pytest.mark.parametrize(
    ("split", "path"), _SPLIT_ARTIFACTS, ids=[s for s, _ in _SPLIT_ARTIFACTS]
)
def test_mve_severity_label_matches_risk_level(split: str, path: Path) -> None:
    """``mve.layer_2.severity_label`` must equal ``risk_level``.

    The MVE generator's post-coerce step (mve_generator.generate_mve)
    overwrites ``layer_2.severity_label`` with the canonical risk_level
    when it differs from the LLM/rule-based output. This test guards
    that the coerce is wired up end-to-end through module 5.
    """
    records = _load_records(path)
    violations = []
    for r in records:
        mve = r["explanation"].get("mve")
        if mve is None:
            continue
        label = str(mve.get("layer_2", {}).get("severity_label", "")).upper()
        if not label:
            continue
        if label != r["risk_level"].upper():
            violations.append({
                "sample_index": r["sample_index"],
                "risk_level": r["risk_level"],
                "mve_severity_label": label,
                "provider": mve.get("provider"),
            })
    assert not violations, (
        f"{split}: {len(violations)} records have an mve.layer_2."
        f"severity_label that disagrees with risk_level. "
        f"First 5: {violations[:5]}"
    )
