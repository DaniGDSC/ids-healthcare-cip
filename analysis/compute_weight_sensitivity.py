"""Stage 5B — Weight sensitivity analysis (Fix 1).

Reframes the four composite-risk weights as hospital-tunable policy
parameters. Verifies tier-assignment stability and safety-floor
preservation under +/- 10% and +/- 20% multiplicative joint perturbations.

Design decisions per docs/fix1_design_memo.md:
- D1: magnitudes 0.10 and 0.20 (joint coverage of legacy +/- 10% and YAML +/- 20%)
- D2: joint random sampling, L1 renormalize, N=30 per magnitude
- D3: exact tier match agreement metric (np.mean(tier == tier_base))
- D4: multiplicative R as named baseline comparator (not primary formula)
- R3: supersedes legacy via merge script precedence rule

Output: results/rq1_weight_sensitivity.json (RQ1_pipeline.md sections 6.1, 6.4).
"""

# ----- Imports -----
import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from module3_risk_scoring.module3_risk_scores import (  # noqa: E402
    WEIGHTS,
    compute_composite_risk,
)

# ----- Constants (per design memo D1, D2; per Session 11 Q-V5) -----
RANDOM_SEED = 42
N_PERTURBATIONS = 30
MAGNITUDES = (0.10, 0.20)
# Tier boundaries per configs/composite_risk_weights.yaml (Session 8 Q-W3)
TIER_BOUNDARIES = (0.80, 0.60, 0.40)  # critical_min, high_min, medium_min
TIER_CRITICAL = 3  # per _assign_tier integer labels
HIST_BINS = 10
HIST_RANGE = (0.0, 1.0)
OUTPUT_PATH = PROJECT_ROOT / "results" / "rq1_weight_sensitivity.json"
SCHEMA_VERSION = "2.0"  # legacy was 1.0
CODE_VERSION = "fix-1-weight-sensitivity"

# Canonical component source. Per Session 11 reads of analysis/compute_rq1.py:282,
# the legacy producer loads from results/reports/risk_scores.npz (a single npz
# whose underlying split is ambiguous per Session 11 section 4 contradiction).
# Inspection: file has keys c_detect, d_crit, s_data, d_clinical_tier, y_true,
# row_id, attack_category, device_class, ... shape (2448,) -- matches val_phase1
# row count from RQ1_pipeline.md:779 (val=2448).
CANONICAL_RISK_SCORES_NPZ = PROJECT_ROOT / "results" / "reports" / "risk_scores.npz"


# ----- Tier assignment (mirrors analysis/compute_rq1.py:430-437 per Session 11) -----
def _assign_tier(R: np.ndarray, boundaries: tuple) -> np.ndarray:
    """Integer tier labels: 0=LOW, 1=MEDIUM, 2=HIGH, 3=CRITICAL.

    Matches analysis/compute_rq1.py:_assign_tier semantics verbatim.
    """
    critical_min, high_min, medium_min = boundaries
    tiers = np.zeros(len(R), dtype=int)
    tiers[R >= medium_min] = 1
    tiers[R >= high_min] = 2
    tiers[R >= critical_min] = 3
    return tiers


# ----- Component loader (canonical npz; --split is provenance-only) -----
def _load_components(split: str):
    """Load (c_detect, d_crit, s_data, d_clinical_tier, y_true) and row count.

    `split` is a provenance label only. The four component arrays and y_true
    are loaded from results/reports/risk_scores.npz (the canonical M3 output
    per analysis/compute_rq1.py:282-294, Session 11 section 4). The npz's
    underlying split is ambiguous per Session 11's surfaced contradiction
    (inline comment says "test-split sourced"; row count = 2448 matches
    val_phase1). This script accepts the caller's split label and records
    it in provenance without claiming to independently verify it. Phase 0e
    is the resolution path for the split identity.
    """
    if split not in ("val_phase1", "test_phase1"):
        raise ValueError(f"split must be 'val_phase1' or 'test_phase1', got {split!r}")

    if not CANONICAL_RISK_SCORES_NPZ.exists():
        raise FileNotFoundError(
            f"Canonical component source missing: {CANONICAL_RISK_SCORES_NPZ}. "
            "Re-run module3_risk_scoring to regenerate before invoking this "
            "script."
        )

    data = np.load(CANONICAL_RISK_SCORES_NPZ, allow_pickle=True)
    needed = ("c_detect", "d_crit", "s_data", "d_clinical_tier", "y_true")
    missing = [k for k in needed if k not in data.files]
    if missing:
        raise KeyError(
            f"{CANONICAL_RISK_SCORES_NPZ.name} missing required keys: "
            f"{missing}. Present keys: {list(data.files)}"
        )

    c_detect = np.asarray(data["c_detect"], dtype=float)
    d_crit = np.asarray(data["d_crit"], dtype=float)
    s_data = np.asarray(data["s_data"], dtype=float)
    d_clinical_tier = np.asarray(data["d_clinical_tier"], dtype=float)
    y_true = np.asarray(data["y_true"]).astype(int)
    return c_detect, d_crit, s_data, d_clinical_tier, y_true, len(c_detect)


# ----- Perturbation generator (mirrors analysis/compute_rq1.py:343-349) -----
def _perturb_weights(rng, baseline_weights: dict, magnitude: float) -> dict:
    """Joint random multiplicative perturbation with L1 renormalize.

    Each weight w_i scaled by (1 + delta_i) where delta_i ~ Uniform(-mag, +mag),
    then the four-vector is renormalized to sum to 1.0.

    Returns a new dict {"w1": ..., "w2": ..., "w3": ..., "w4": ...} where the
    four float64 values sum to 1.0 at float64 precision (better than 1e-6).
    """
    delta = rng.uniform(-magnitude, magnitude, size=4)
    base_array = np.array([baseline_weights[k] for k in ("w1", "w2", "w3", "w4")])
    perturbed = base_array * (1.0 + delta)
    perturbed = perturbed / perturbed.sum()  # L1 renormalize
    return {f"w{i + 1}": float(perturbed[i]) for i in range(4)}


# ----- Agreement and FNR-critical (per D3) -----
def _agreement_exact_tier_match(tiers_a: np.ndarray, tiers_b: np.ndarray) -> float:
    """Exact tier match: np.mean(tier == tier_base). Per Session 11 Q-V5.3."""
    return float(np.mean(tiers_a == tiers_b))


def _fnr_critical_delta(tiers_baseline: np.ndarray, tiers_perturbed: np.ndarray) -> float:
    """Fraction of population that was CRITICAL under baseline and dropped below.

    Verbatim from legacy producer at analysis/compute_rq1.py:365-369
    (Session 11 Q-V5):

        crit_base = (tier_base == 3)
        if int(np.sum(crit_base)) > 0:
            fnr_delta = float(np.mean(crit_base & (tier < 3)))
        else:
            fnr_delta = 0.0

    This is a tier-to-tier metric (safety-floor breach rate over the
    population); it does NOT depend on y_true.
    """
    crit_base = tiers_baseline == TIER_CRITICAL
    if int(crit_base.sum()) == 0:
        return 0.0
    return float(np.mean(crit_base & (tiers_perturbed < TIER_CRITICAL)))


# ----- Three named baselines (per D4, Session 11 section 4) -----
def _baseline_equal_weights() -> dict:
    return {"w1": 0.25, "w2": 0.25, "w3": 0.25, "w4": 0.25}


def _baseline_c_detect_only() -> dict:
    return {"w1": 1.0, "w2": 0.0, "w3": 0.0, "w4": 0.0}


def _compute_multiplicative_R(c_detect, d_crit, s_data, d_clinical_tier):
    """Multiplicative baseline: c_detect * max(d_crit, s_data, d_clinical_tier).

    Verbatim from analysis/compute_rq1.py:357-359 (Session 11 section 4).
    """
    return c_detect * np.maximum.reduce([d_crit, s_data, d_clinical_tier])


# ----- Provenance -----
def _sha256_file(path: Path) -> str:
    if not path.exists():
        return "ABSENT"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_provenance(split_used: str) -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "random_seed": RANDOM_SEED,
        "schema_version": SCHEMA_VERSION,
        "code_version": CODE_VERSION,
        "python_version": platform.python_version(),
        "split_label": split_used,
        "split_label_note": (
            "Provenance label only. Component arrays were loaded from "
            "results/reports/risk_scores.npz; the npz's underlying split "
            "is ambiguous per Codebase_Investigation.html Session 11 "
            "section 4 (inline comment says 'test-split sourced'; row "
            "count = 2448 matches val_phase1). Pending Phase 0e."
        ),
        "input_files": {
            "composite_weights_yaml": _sha256_file(
                PROJECT_ROOT / "configs" / "composite_risk_weights.yaml"
            ),
            "risk_scores_npz": _sha256_file(CANONICAL_RISK_SCORES_NPZ),
            "compute_weight_sensitivity_py": _sha256_file(Path(__file__)),
        },
    }


# ----- Per-magnitude perturbation block -----
def _run_perturbations_for_magnitude(
    rng,
    magnitude,
    c_detect,
    d_crit,
    s_data,
    d_clinical_tier,
    y_true,
    baseline_weights,
    tiers_baseline,
):
    """Run N_PERTURBATIONS random perturbations at this magnitude.

    Returns a dict matching the legacy perturbation_results schema with
    the addition of fnr_critical_delta_max and fnr_critical_delta_mean.
    """
    agreements = []
    fnr_deltas = []

    for _ in range(N_PERTURBATIONS):
        perturbed = _perturb_weights(rng, baseline_weights, magnitude)
        # Sanity check sum-to-1.0 (matches production invariant at
        # module3_risk_scoring/module3_risk_scores.py:86-90).
        s = sum(perturbed.values())
        if abs(s - 1.0) > 1e-6:
            raise RuntimeError(f"Perturbation sum-to-1.0 violated: {s}; weights {perturbed}")
        R = compute_composite_risk(c_detect, d_crit, s_data, d_clinical_tier, perturbed)
        tiers_pert = _assign_tier(R, TIER_BOUNDARIES)
        agreements.append(_agreement_exact_tier_match(tiers_pert, tiers_baseline))
        fnr_deltas.append(_fnr_critical_delta(tiers_baseline, tiers_pert))

    agreements_arr = np.array(agreements)
    fnr_deltas_arr = np.array(fnr_deltas)
    hist_counts, hist_edges = np.histogram(agreements_arr, bins=HIST_BINS, range=HIST_RANGE)

    return {
        "n_perturbations": N_PERTURBATIONS,
        "agreement_mean": float(round(agreements_arr.mean(), 4)),
        "agreement_std": float(round(agreements_arr.std(), 4)),
        "agreement_min": float(round(agreements_arr.min(), 4)),
        "agreement_max": float(round(agreements_arr.max(), 4)),
        "agreement_p25": float(round(np.percentile(agreements_arr, 25), 4)),
        "agreement_p50": float(round(np.percentile(agreements_arr, 50), 4)),
        "agreement_p75": float(round(np.percentile(agreements_arr, 75), 4)),
        "histogram_counts": hist_counts.tolist(),
        "histogram_edges": [float(round(e, 2)) for e in hist_edges.tolist()],
        "fnr_critical_delta_max": float(round(fnr_deltas_arr.max(), 4)),
        "fnr_critical_delta_mean": float(round(fnr_deltas_arr.mean(), 4)),
    }


# ----- Three named baselines (Session 11 section 4 schema) -----
def _run_named_baselines(c_detect, d_crit, s_data, d_clinical_tier, tiers_baseline):
    """Run the three named baselines from legacy (Session 11 section 4).

    Note: `y_true` is intentionally not used. The legacy fnr_critical_delta
    metric is tier-to-tier (safety-floor breach over the population) and
    does not depend on labels.
    """
    results = {}

    for name, weights in (
        ("equal_weights", _baseline_equal_weights()),
        ("c_detect_only", _baseline_c_detect_only()),
    ):
        R = compute_composite_risk(c_detect, d_crit, s_data, d_clinical_tier, weights)
        tiers = _assign_tier(R, TIER_BOUNDARIES)
        results[name] = {
            "agreement": _agreement_exact_tier_match(tiers, tiers_baseline),
            "fnr_critical_delta": _fnr_critical_delta(tiers_baseline, tiers),
        }

    # Multiplicative baseline (different formula, not weight perturbation).
    R_mult = _compute_multiplicative_R(c_detect, d_crit, s_data, d_clinical_tier)
    tiers_mult = _assign_tier(R_mult, TIER_BOUNDARIES)
    results["multiplicative"] = {
        "agreement": _agreement_exact_tier_match(tiers_mult, tiers_baseline),
        "fnr_critical_delta": _fnr_critical_delta(tiers_baseline, tiers_mult),
    }

    # Round all values for stable JSON output.
    for key in results:
        results[key] = {k: float(round(v, 4)) for k, v in results[key].items()}
    return results


# ----- Entry point -----
def run_perturbation_analysis(split: str) -> dict:
    """Top-level analysis.

    Args:
        split: 'val_phase1' or 'test_phase1' (provenance label only;
            see _load_components docstring).

    Returns:
        Full output dict matching the JSON schema.
    """
    (c_detect, d_crit, s_data, d_clinical_tier, y_true, n_alerts) = _load_components(split)

    baseline_weights = dict(WEIGHTS)
    R_baseline = compute_composite_risk(c_detect, d_crit, s_data, d_clinical_tier, baseline_weights)
    tiers_baseline = _assign_tier(R_baseline, TIER_BOUNDARIES)

    rng = np.random.default_rng(RANDOM_SEED)

    by_magnitude = {}
    for mag in MAGNITUDES:
        key = f"{mag:.2f}"
        by_magnitude[key] = _run_perturbations_for_magnitude(
            rng,
            mag,
            c_detect,
            d_crit,
            s_data,
            d_clinical_tier,
            y_true,
            baseline_weights,
            tiers_baseline,
        )
        by_magnitude[key]["magnitude"] = mag

    named_baselines = _run_named_baselines(
        c_detect, d_crit, s_data, d_clinical_tier, tiers_baseline
    )

    return {
        "provenance": _build_provenance(split),
        "results": {
            "perturbation_method": (
                "multiplicative +/- magnitude then L1 renormalize to "
                "sum=1.0; magnitudes per by_magnitude keys"
            ),
            "perturbation_results": {"by_magnitude": by_magnitude},
            "baselines": named_baselines,
            "baseline_weights": {
                "detection_confidence": baseline_weights["w1"],
                "device_criticality": baseline_weights["w2"],
                "data_sensitivity": baseline_weights["w3"],
                "clinical_tier": baseline_weights["w4"],
            },
            "tier_boundaries": {
                "critical_min": TIER_BOUNDARIES[0],
                "high_min": TIER_BOUNDARIES[1],
                "medium_min": TIER_BOUNDARIES[2],
            },
            "n_alerts_evaluated": n_alerts,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Stage 5B weight sensitivity analysis")
    parser.add_argument(
        "--split",
        choices=("val_phase1", "test_phase1"),
        default="val_phase1",
        help=(
            "Provenance label for the run (default: val_phase1, per "
            "design memo R2 default). Component arrays come from "
            "results/reports/risk_scores.npz regardless; the label is "
            "recorded in provenance only."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"Output JSON path (default: {OUTPUT_PATH})",
    )
    args = parser.parse_args()

    result = run_perturbation_analysis(args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"Wrote {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
