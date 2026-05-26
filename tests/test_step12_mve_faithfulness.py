"""RQ2.b Step 12 — MVE faithfulness tests (MVE-SHAP alignment).

Verifies that the precomputed alignment artifact:
  1. Exists and reports BOTH modes (LLM narrative + rule-based).
  2. Reports per-feature hit rates (so reviewers can see which features
     are mentioned often vs rarely).
  3. The Mode B top-1 rate is at the documented 100% (rule-based always
     injects the primary SHAP feature).
  4. Layer 1 text is non-empty for all samples (no blank narratives).

This file deliberately ENCODES the current measured values as soft
lower bounds so a regression that drops Mode A top-1 below ~50% would
fail the suite. Aspirational targets (≥2 ≥ 95%) are documented in the
summary report but not enforced — those are open improvement items.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = PROJECT_ROOT / "results/rq2_mve_shap_alignment.json"


@pytest.fixture(scope="module")
def alignment_report():
    if not ARTIFACT.exists():
        pytest.skip(f"{ARTIFACT} missing — run tools/rq2_compute_faithfulness.py")
    with open(ARTIFACT) as f:
        return json.load(f)


def test_alignment_artifact_valid(alignment_report):
    assert "_meta" in alignment_report
    assert "mode_a_llm_narrative" in alignment_report
    assert "mode_b_rule_based" in alignment_report


def test_alignment_reports_both_modes(alignment_report):
    """Spec calls for Mode A (LLM) and Mode B (rule-based) side-by-side."""
    for mode_key in ("mode_a_llm_narrative", "mode_b_rule_based"):
        m = alignment_report[mode_key]
        for k in ("n_total", "contains_top1_pct", "contains_at_least_2_pct",
                  "contains_all_3_pct", "per_feature_hit_rate", "per_sample"):
            assert k in m, f"{mode_key} missing key: {k}"


def test_mode_b_top1_perfect(alignment_report):
    """Rule-based MVE injects the primary SHAP feature via the 'Primary
    signal: ({feat})' suffix — top-1 hit rate must be 100% by
    construction. A drop would mean src.mve_generator regressed.
    """
    mb = alignment_report["mode_b_rule_based"]
    assert mb["contains_top1_pct"] == 100.0, (
        f"Mode B top-1 rate dropped to {mb['contains_top1_pct']}% — "
        "src.mve_generator no longer injects the primary SHAP feature"
    )


def test_mode_a_top1_above_baseline(alignment_report):
    """Mode A (LLM narrative) abstracts features by design but should
    still surface the top-1 feature most of the time (~80% measured).
    Floor at 60% — anything lower indicates the narrative pipeline
    diverged from the SHAP source.
    """
    ma = alignment_report["mode_a_llm_narrative"]
    assert ma["contains_top1_pct"] >= 60.0, (
        f"Mode A top-1 rate = {ma['contains_top1_pct']}% — below 60% floor"
    )


def test_no_empty_layer1_excerpts(alignment_report):
    """Every sample must have non-empty Layer 1 text in both modes."""
    for mode_key in ("mode_a_llm_narrative", "mode_b_rule_based"):
        per = alignment_report[mode_key]["per_sample"]
        empty = [s for s in per if not s.get("layer1_excerpt")]
        assert not empty, (
            f"{mode_key} has {len(empty)} samples with empty layer1_excerpt"
        )


def test_per_feature_hit_rates_documented(alignment_report):
    """per_feature_hit_rate must report appearances + mentioned + hit_rate
    for every feature seen in the top-3 across samples.
    """
    for mode_key in ("mode_a_llm_narrative", "mode_b_rule_based"):
        rates = alignment_report[mode_key]["per_feature_hit_rate"]
        assert rates, f"{mode_key} per_feature_hit_rate empty"
        for feat, stats in rates.items():
            for k in ("appearances", "mentioned", "hit_rate"):
                assert k in stats, f"{mode_key}[{feat}] missing {k}"


def test_n_samples_consistent_across_modes(alignment_report):
    a = alignment_report["mode_a_llm_narrative"]["n_total"]
    b = alignment_report["mode_b_rule_based"]["n_total"]
    assert a == b, f"sample count mismatch: A={a} vs B={b}"


def test_meta_lists_alias_table(alignment_report):
    """Alias table size must be documented — the test relies on aliases
    to bridge raw feature names to natural-language references.
    """
    meta = alignment_report["_meta"]
    assert "alias_table_features" in meta
    assert len(meta["alias_table_features"]) >= 20, (
        "alias_table too small — cannot bridge feature names to text"
    )


# ── Post-fix targets (G1 + G2 closed) ───────────────────────────────


def test_mode_b_at_least_2_meets_target(alignment_report):
    """After the G1+G2 fix, Mode B rule-based MVE lists the top-3 SHAP
    features in Layer 1, so the ≥2 alignment rate should hit ≥95%.
    A drop below this means src.mve_generator regressed back to top-1.
    """
    mb = alignment_report["mode_b_rule_based"]
    assert mb["contains_at_least_2_pct"] >= 95.0, (
        f"Mode B ≥2 rate = {mb['contains_at_least_2_pct']}% — "
        "below 95% target. Did src.mve_generator stop injecting top-3?"
    )


def test_mode_b_all_3_meets_target(alignment_report):
    """After the G1+G2 fix, Mode B should also surface all 3 top SHAP
    features at ≥80% rate (the spec target).
    """
    mb = alignment_report["mode_b_rule_based"]
    assert mb["contains_all_3_pct"] >= 80.0, (
        f"Mode B all-3 rate = {mb['contains_all_3_pct']}% — "
        "below 80% target."
    )


# ── G6 large-N assertions ────────────────────────────────────────────


def test_mode_b_large_n_present(alignment_report):
    """G6 fix: the large-N Mode B measurement (n≥100) must be embedded
    in the alignment report so reviewers can verify the n=20 result
    isn't a small-sample artifact.
    """
    assert "mode_b_rule_based_large_n" in alignment_report, (
        "G6 large-N alignment missing — run tools/rq2_compute_faithfulness.py"
    )
    large = alignment_report["mode_b_rule_based_large_n"]
    assert large["metrics"]["n_total"] >= 100, (
        f"large-N sample size = {large['metrics']['n_total']} — must be ≥100 "
        "for the small-sample claim to be statistically defensible"
    )


def test_mode_b_large_n_meets_targets(alignment_report):
    """At n=200, both ≥2 (≥95%) and all-3 (≥80%) targets must hold."""
    m = alignment_report["mode_b_rule_based_large_n"]["metrics"]
    assert m["target_at_least_2_met"] is True, (
        f"large-N ≥2 rate = {m['contains_at_least_2_pct']}% — below 95% target"
    )
    assert m["target_all_3_met"] is True, (
        f"large-N all-3 rate = {m['contains_all_3_pct']}% — below 80% target"
    )


def test_mode_b_large_n_ci_lower_bound_above_target(alignment_report):
    """95% Wilson CI lower bound on ≥2 must clear the 95% spec target —
    this is the formal small-sample-fluke rebuttal.
    """
    m = alignment_report["mode_b_rule_based_large_n"]["metrics"]
    ci_low, ci_high = m["ci95_at_least_2_pct"]
    point = m["contains_at_least_2_pct"]
    assert ci_low >= 95.0, (
        f"CI lower bound {ci_low}% does not clear 95% target "
        f"(point estimate {point}%, CI=[{ci_low}, {ci_high}]). "
        f"Diagnostic checklist: "
        f"(1) did src.mve_generator stop injecting top-3 SHAP features? "
        f"(2) is the point estimate itself comfortably above 95% — "
        f"if at exactly 95%, CI lower bound *will* dip below by construction; "
        f"(3) raise n in tools/rq2_compute_faithfulness.py and re-run."
    )
