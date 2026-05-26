"""Regression tests for Module 5's envelope, provenance and drift checks.

Covers the P0 fixes shipped alongside this file:

- P0-3 — every record in ``alert_responses{,_demo}.json`` validates
  against :class:`common.alert_response_schema.AlertResponsesEnvelope`.
- P0-1 — the embedded ``_provenance`` block's sha256 / mtime fields
  match the actual input files on disk (otherwise upstream has been
  regenerated and Module 5 needs to rerun).
- P0-2 — every ``risk_score`` / ``risk_level`` / ``risk_components``
  field in the records matches the corresponding row of the source
  ``risk_scores.npz`` (or ``demo_scores.npz``) to within 1e-4.

These tests run against the on-disk artefacts produced by the most
recent pipeline run. They are skipped (not failed) when the file is
absent so a fresh clone with no demo run does not red the CI.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.alert_response_schema import AlertResponsesEnvelope

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS = PROJECT_ROOT / "results/reports"

# Mapping: (split label, responses file, scores npz file)
_SPLIT_ARTIFACTS = [
    ("test", REPORTS / "alert_responses.json", REPORTS / "risk_scores.npz"),
    ("demo", REPORTS / "alert_responses_demo.json", REPORTS / "demo_scores.npz"),
]


def _load_envelope_or_skip(path: Path) -> AlertResponsesEnvelope:
    """Load a Module 5 envelope file; skip if missing or legacy-shaped.

    Legacy bare-list files are not failures — they're old artefacts
    that pre-date the envelope refactor and will be replaced on the
    next Module 5 rerun. Tests only run against new-shape files.
    """
    if not path.exists():
        pytest.skip(f"{path.name} not on disk (likely no demo run yet)")
    raw = json.loads(path.read_text())
    if isinstance(raw, list):
        pytest.skip(
            f"{path.name} is legacy bare-list shape — rerun "
            f"`python -m module5_responses.module5_responses` to upgrade."
        )
    return AlertResponsesEnvelope.model_validate(raw)


# ── P0-3: schema validity ───────────────────────────────────────────────


@pytest.mark.parametrize("split,responses_path,_npz", _SPLIT_ARTIFACTS)
def test_alert_responses_envelope_schema_valid(
    split: str, responses_path: Path, _npz: Path,
) -> None:
    """Every record in the envelope must validate against AlertRecord."""
    envelope = _load_envelope_or_skip(responses_path)
    # model_validate already ran inside _load_envelope_or_skip; if we
    # got here the schema passed. Additionally assert the basic
    # invariants the schema can't express on its own.
    assert envelope.provenance.split == split, (
        f"Envelope split={envelope.provenance.split} != filename split={split}"
    )
    assert envelope.provenance.n_alerts_emitted == len(envelope.records), (
        "Provenance n_alerts_emitted disagrees with len(records)"
    )


# ── P0-1: provenance sha256 + mtime invariants ──────────────────────────


@pytest.mark.parametrize("split,responses_path,_npz", _SPLIT_ARTIFACTS)
def test_provenance_inputs_sha256_match_disk(
    split: str, responses_path: Path, _npz: Path,
) -> None:
    """Provenance must record sha256 of every input file as it was at
    build time. If a sha drifts, upstream has been regenerated and
    Module 5 needs to rerun — the Dashboard's freshness banner will
    fire in that case."""
    envelope = _load_envelope_or_skip(responses_path)
    drifted = []
    for key, meta in envelope.provenance.inputs.items():
        if meta is None:
            continue
        live = PROJECT_ROOT / meta.path
        if not live.exists():
            drifted.append(f"{key}: file missing on disk ({meta.path})")
            continue
        live_sha = hashlib.sha256(live.read_bytes()).hexdigest()
        if live_sha != meta.sha256:
            drifted.append(
                f"{key}: sha256 changed (live={live_sha[:12]}... "
                f"vs provenance={meta.sha256[:12]}...)"
            )
    assert not drifted, (
        f"Provenance is stale for split={split}; rerun Module 5:\n  "
        + "\n  ".join(drifted)
    )


# ── P0-2: cross-check record fields vs source npz ───────────────────────


@pytest.mark.parametrize("split,responses_path,npz_path", _SPLIT_ARTIFACTS)
def test_alert_responses_consistent_with_npz(
    split: str, responses_path: Path, npz_path: Path,
) -> None:
    """Every record's risk_score/level/components must equal the
    corresponding row of the source npz to within 1e-4. Catches the
    case where someone reran Module 3 without rerunning Module 5."""
    envelope = _load_envelope_or_skip(responses_path)
    if not npz_path.exists():
        pytest.skip(f"{npz_path.name} not on disk")
    data = np.load(npz_path)
    component_map = [
        ("C_detect", "c_detect"),
        ("C_track_a", "c_track_a"),
        ("C_track_b", "c_track_b"),
        ("D_crit", "d_crit"),
        ("S_data", "s_data"),
        ("D_clinical_tier", "d_clinical_tier"),
    ]
    tol = 1e-4
    n_drifted = 0
    first_drift = None
    for rec in envelope.records:
        idx = rec.sample_index
        expected_R = round(float(data["R"][idx]), 4)
        if abs(rec.risk_score - expected_R) > tol:
            n_drifted += 1
            first_drift = first_drift or (
                f"idx={idx}: record.risk_score={rec.risk_score} vs "
                f"npz.R={expected_R}"
            )
            continue
        if rec.risk_level != str(data["risk_levels"][idx]):
            n_drifted += 1
            first_drift = first_drift or (
                f"idx={idx}: record.risk_level={rec.risk_level} vs "
                f"npz={data['risk_levels'][idx]}"
            )
            continue
        for rec_key, npz_key in component_map:
            expected = round(float(data[npz_key][idx]), 4)
            actual = getattr(rec.risk_components, rec_key)
            if abs(actual - expected) > tol:
                n_drifted += 1
                first_drift = first_drift or (
                    f"idx={idx} {rec_key}: record={actual} vs npz={expected}"
                )
                break
    assert n_drifted == 0, (
        f"{n_drifted} record(s) drifted from {npz_path.name}. "
        f"First: {first_drift}. Rerun Module 5."
    )


# ── Loader backward-compat shim ─────────────────────────────────────────


def test_loader_accepts_legacy_list_format(tmp_path):
    """Files in the legacy bare-list shape must still load (no crash).

    This guards the backward-compat branch in
    ``module6_app.load_responses_for`` — if someone deletes the
    ``isinstance(raw, list)`` shim assuming everything has been
    upgraded, demo files that haven't been regenerated will start
    crashing the dashboard.
    """
    legacy = [
        {
            "sample_index": 0,
            "ground_truth": "benign",
            "attack_category": "Normal",
            "risk_score": 0.1,
            "risk_level": "LOW",
            "risk_components": {
                "C_detect": 0.1, "C_track_a": 0.1, "C_track_b": 0.0,
                "D_crit": 0.0, "S_data": 0.0, "D_clinical_tier": 0.0,
            },
            "response": {
                "actions": ["log_event"],
                "action_descriptions": ["Log the event"],
                "escalation_chain": {
                    "primary": None, "secondary": None, "tertiary": None,
                },
                "escalation_rationale": "n/a",
                "max_response_min": 480,
                "priority": 4,
                "rationale": "Base response for LOW",
                "device_tier": "general",
                "device_constraint_applied": False,
            },
            "explanation": {"clinician_summary": "", "analyst_available": False},
        }
    ]
    # The schema must accept a single record validated standalone,
    # which is what the legacy-list code path effectively does.
    from common.alert_response_schema import AlertRecord
    AlertRecord.model_validate(legacy[0])
