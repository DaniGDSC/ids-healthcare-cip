"""Smoke + invariants tests for tools.build_simulation_stream.

The builder is the single source for the M6 dashboard's "Full stream"
mode — these tests pin its contract so a regression in the tier-join or
timestamp anchoring fails loudly here rather than as a quiet UI glitch.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.build_simulation_stream import _STREAM_START, build_stream


# ── Fixtures: lazily skip if upstream artefacts missing ─────────────────


@pytest.fixture(scope="module")
def demo_payload():
    parquet = PROJECT_ROOT / "data/processed/demo_phase1.parquet"
    scores = PROJECT_ROOT / "results/reports/demo_scores.npz"
    if not parquet.exists() or not scores.exists():
        pytest.skip("demo upstream artefacts missing — run M2/M3 demo first.")
    return build_stream("demo")


# ── _meta invariants ─────────────────────────────────────────────────────


def test_meta_split_label_is_demo(demo_payload):
    assert demo_payload["_meta"]["split"] == "demo"


def test_meta_counts_add_up(demo_payload):
    meta = demo_payload["_meta"]
    assert meta["n_total"] == meta["n_surfaced"] + meta["n_normal"]


def test_meta_stream_start_matches_constant(demo_payload):
    """Stream anchor must match the dashboard's load_live_stream_source
    so the two clocks line up if both modes are active."""
    assert demo_payload["_meta"]["stream_start"] == _STREAM_START.isoformat()


# ── stream array invariants ──────────────────────────────────────────────


def test_stream_length_matches_meta(demo_payload):
    assert len(demo_payload["stream"]) == demo_payload["_meta"]["n_total"]


def test_stream_indices_dense_and_ordered(demo_payload):
    indices = [e["sample_index"] for e in demo_payload["stream"]]
    assert indices == list(range(len(indices)))


def test_normal_entries_have_null_alert(demo_payload):
    """NORMAL placeholders are minimal — no alert payload, ever."""
    leaked = [
        e["sample_index"] for e in demo_payload["stream"]
        if e["risk_level"] == "NORMAL" and e["alert"] is not None
    ]
    assert not leaked, (
        f"{len(leaked)} NORMAL entries carried an alert payload "
        f"(first 3 indices: {leaked[:3]}). The contract is alert=None "
        f"for NORMAL — they exist only to tick the stream clock."
    )


def test_surfaced_entries_have_alert_when_m5_ran(demo_payload):
    """LOW+ entries must embed the M5 alert payload when alert_responses
    is present. Skips when the builder reports responses_joined=False
    so a pre-M5 build doesn't red the test."""
    if not demo_payload["_meta"]["responses_joined"]:
        pytest.skip("alert_responses_demo.json absent at build time.")
    missing = [
        e["sample_index"] for e in demo_payload["stream"]
        if e["risk_level"] != "NORMAL" and e["alert"] is None
    ]
    assert not missing, (
        f"{len(missing)} LOW+ entries have alert=None despite M5 having "
        f"run. Index gap between demo_scores.npz tiers and "
        f"alert_responses_demo.json sample_index — first 3: {missing[:3]}"
    )


def test_arrived_at_is_iso_seconds(demo_payload):
    """Timestamps are 1-second increments anchored at _STREAM_START."""
    first = demo_payload["stream"][0]["arrived_at"]
    assert first == _STREAM_START.strftime("%Y-%m-%dT%H:%M:%S")
    # Sample 60 should be exactly 60 seconds later.
    if len(demo_payload["stream"]) > 60:
        s60 = demo_payload["stream"][60]["arrived_at"]
        assert s60.endswith(":01:00")


def test_entry_carries_risk_components(demo_payload):
    """Every entry (both NORMAL and LOW+) carries the 4 component scores
    so the dashboard can render the risk-decomposition strip without
    needing a second artefact join."""
    needed = {"c_detect", "d_crit", "s_data", "d_clinical_tier"}
    for e in demo_payload["stream"][:5]:
        assert needed.issubset(set(e["risk_components"].keys())), (
            f"sample {e['sample_index']} missing component keys: "
            f"{needed - set(e['risk_components'].keys())}"
        )


def test_ground_truth_is_attack_or_benign(demo_payload):
    bad = {
        e["sample_index"]: e["ground_truth"]
        for e in demo_payload["stream"]
        if e["ground_truth"] not in ("attack", "benign")
    }
    assert not bad, f"ground_truth must be attack|benign — got {bad}"


# ── Builder fails fast on missing inputs ─────────────────────────────────


def test_build_unknown_split_raises():
    with pytest.raises(ValueError, match="unknown split"):
        build_stream("validation")
