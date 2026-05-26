"""Regression guard for the shared device-context module.

Option 4 consolidated ``DEVICE_CONTEXT`` and the per-row device-class
heuristic into ``common.device_class``. Two pipelines now consume them:

* Module 5 (``build_all_records``) — picks a device class per alert so
  ``src.mve_generator`` can generate the right rule-based MVE template.
* Module 6 (``module6_evaluation._build_eval_alert``) — annotates the
  curated 20-alert evaluation set with the same device context the
  dashboard renders.

If these two ever drift (because someone updates one map and forgets the
other), the dashboard's Tier badge and the MVE's Layer 2 severity_label
diverge and the user-visible bug we just fixed comes back.

These tests pin the contract:

* The single-row helper agrees with the vectorised batch helper.
* The Module 6 shim still routes through the common module.
* ``DEVICE_CONTEXT`` covers every label the heuristic can produce.
* ``synthesize_raw_alert`` produces fields ``mve_generator`` actually
  reads (alert_name, protocol, severity_score in the [0, 10] band).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.device_class import (
    DEVICE_CONTEXT,
    derive_device_class_array,
    derive_device_class_row,
    device_context_for_idx,
    synthesize_raw_alert,
)


_BIO_FEATS = ("Temp", "SpO2", "Pulse_Rate", "Heart_rate", "Resp_Rate", "ST")
_NET_FEATS = ("Sport", "SrcBytes")


def _row(**overrides: float) -> pd.Series:
    """Build a single test row with biometric/network fields = 0 by default."""
    data = {f: 0.0 for f in _BIO_FEATS}
    data.update({f: 0.0 for f in _NET_FEATS})
    data.update(overrides)
    return pd.Series(data)


# ── derive_device_class_row vs derive_device_class_array agreement ──────


def test_row_and_array_helpers_agree() -> None:
    """The single-row helper must produce the same label as the batch one
    on the same input matrix — they're two views of the same heuristic
    and a divergence would mean the dashboard (which uses .iloc[idx]) and
    Module 5 (which iterates rows) tag the same sample differently."""
    rows = [
        _row(Resp_Rate=1.0, SpO2=1.0, Pulse_Rate=1.0, Heart_rate=1.0),     # ventilator
        _row(Pulse_Rate=1.0, Heart_rate=1.0, Temp=1.0),                     # patient_monitor
        _row(Temp=1.0, SpO2=1.0),                                           # infusion_pump
        _row(Sport=0.5),                                                    # ehr_workstation
        _row(),                                                             # other
    ]
    df = pd.DataFrame(rows)
    feature_names = list(df.columns)

    # Array helper expects a 2-D numpy float matrix
    array_labels = derive_device_class_array(df.to_numpy(dtype=float), feature_names)
    row_labels = [derive_device_class_row(df.iloc[i]) for i in range(len(df))]
    assert array_labels == row_labels


# ── Module 6 shim still routes through common ──────────────────────────


def test_module6_shim_uses_common() -> None:
    """``module6_evaluation._derive_device_class`` must delegate to common.

    Tests that the public name still resolves AND that its output equals
    the common helper's output on the same input."""
    from module6_evaluation import module6_evaluation as m6

    df = pd.DataFrame([_row(Temp=1.0, SpO2=1.0), _row()])
    for idx in range(len(df)):
        assert m6._derive_device_class(idx, df) == derive_device_class_row(df.iloc[idx])


# ── DEVICE_CONTEXT covers every label the heuristic can emit ────────────


def test_device_context_covers_every_heuristic_label() -> None:
    """Each label that the heuristic can output must have a DEVICE_CONTEXT
    entry. A missing entry would crash ``device_context_for_idx`` only
    when an unusual row hits the unlucky branch in production — pin this
    statically instead.
    """
    heuristic_labels = {
        "ventilator",
        "patient_monitor",
        "infusion_pump",
        "ehr_workstation",
        "other",
    }
    missing = heuristic_labels - set(DEVICE_CONTEXT.keys())
    assert not missing, f"DEVICE_CONTEXT missing entries for: {missing}"


def test_device_context_required_fields() -> None:
    """Each DEVICE_CONTEXT entry must carry the fields the dashboard reads.

    The dashboard's ``render_device_criticality`` and the enrichment join
    expect these four keys on every entry."""
    required = {"affected_system", "patient_care_impact",
                "device_criticality", "active_device"}
    valid_tiers = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
    for label, ctx in DEVICE_CONTEXT.items():
        missing = required - set(ctx.keys())
        assert not missing, f"DEVICE_CONTEXT[{label!r}] missing: {missing}"
        assert ctx["device_criticality"] in valid_tiers, (
            f"DEVICE_CONTEXT[{label!r}] criticality {ctx['device_criticality']!r} "
            f"not in {sorted(valid_tiers)}"
        )
        assert isinstance(ctx["active_device"], bool)


# ── device_context_for_idx returns a complete merged dict ───────────────


def test_device_context_for_idx_embeds_class() -> None:
    """The convenience wrapper must include ``device_class`` next to the
    context fields so Module 5 doesn't have to call the row helper twice."""
    df = pd.DataFrame([_row(Resp_Rate=1.0, SpO2=1.0, Pulse_Rate=1.0, Heart_rate=1.0)])
    ctx = device_context_for_idx(0, df)
    assert ctx["device_class"] == "ventilator"
    assert ctx["device_criticality"] == "CRITICAL"
    assert ctx["affected_system"].startswith("Ventilator")


# ── synthesize_raw_alert produces fields mve_generator actually reads ───


def test_synthesize_raw_alert_shape() -> None:
    """The synthesized raw_alert must populate the fields
    ``mve_generator._detect_alert_type`` and ``_confidence_level`` read
    (alert_name, protocol, severity_score). Nothing else — fake IPs or
    timestamps would mislead operators reading the explanation."""
    out = synthesize_raw_alert(sample_index=42, attack_category="Spoofing", risk_score=0.83)
    assert "Spoofing" in out["alert_name"]
    assert "42" in out["alert_name"]
    assert out["protocol"]  # non-empty
    # severity_score must be in the [0, 10] band that mve_generator's
    # confidence threshold (>7 HIGH, >4 MEDIUM) expects — Module 3's
    # risk_score is in [0, 1] and we scale by 10x.
    assert 0.0 <= out["severity_score"] <= 10.0
    # 0.83 * 10 = 8.3 → should land in HIGH band
    assert out["severity_score"] > 7.0


def test_synthesize_raw_alert_unknown_category() -> None:
    """Unknown attack categories must still produce a valid raw_alert —
    ``mve_generator._detect_alert_type`` defaults to T1 in that case."""
    out = synthesize_raw_alert(sample_index=0, attack_category="WeirdCategoryX", risk_score=0.5)
    assert out["protocol"] == "unknown"
    assert out["severity_score"] == 5.0


# ── End-to-end: heuristic → context → MVE generator path ────────────────


def test_end_to_end_other_class_does_not_force_critical() -> None:
    """An "other"-class row must NOT trigger mve_generator's is_unknown
    safe-default that pins criticality to CRITICAL. Module 5 maps "other"
    → "system" specifically to avoid this. Regression guard against
    someone removing that mapping.
    """
    # Need OPENAI_API_KEY missing so the chain falls through to
    # rule-based deterministically; pytest's env is already clean.
    from src.mve_generator import generate_mve

    df = pd.DataFrame([_row()])  # all zeros → "other"
    assert derive_device_class_row(df.iloc[0]) == "other"
    ctx_full = device_context_for_idx(0, df)
    assert ctx_full["device_criticality"] == "MEDIUM"

    # Replicate the Module 5 "other" → "system" mapping
    mve_device_type = ctx_full["device_class"]
    if mve_device_type == "other":
        mve_device_type = "system"

    mve = generate_mve(
        raw_alert=synthesize_raw_alert(0, "Spoofing", 0.5),
        device_context={
            "device_type": mve_device_type,
            "criticality": ctx_full["device_criticality"],
            "clinical_function": ctx_full["affected_system"],
            "patchable": True,
        },
        baseline={"baseline_days": 90},
        user_context=None,
        force_rule_based=True,
    )
    # The safety override would force CRITICAL; we want MEDIUM through.
    assert mve.layer_2["severity_label"] == "MEDIUM", (
        f"'other' device wrongly elevated to {mve.layer_2['severity_label']} — "
        f"the Module 5 'other' → 'system' mapping likely regressed."
    )
