#!/usr/bin/env python3
"""Compare single-shot SHAP stability vs robust (mean) SHAP variant.

Sprint 5 / Tầng 3.2 — measures whether the robust top-features
helper actually delivers more stable attributions than the
single-shot SHAP read the analyst report uses today.

For a sample of XGBoost-flagged alerts:

  1. Compute single-shot SHAP top-K (baseline) twice with different
     RNG seeds → measure Jaccard overlap.
  2. Compute robust SHAP top-K (mean over N perturbations) twice
     with different seeds → measure Jaccard overlap.
  3. Report the mean Jaccard for both.

A successful Sprint 5 / Tầng 3.2 win = robust mean Jaccard > single
mean Jaccard by ≥ 0.10 on the corpus. Anything less means the
mean isn't doing more than a single noisy attribution would.

Output: ``results/stability_variant_comparison.json``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _jaccard(a: set, b: set) -> float:
    u = a | b
    if not u:
        return 1.0
    return len(a & b) / len(u)


def _top_k(sv_row: np.ndarray, k: int) -> set:
    """Set of indices for the top-k features by |SHAP|."""
    if k <= 0 or sv_row.size == 0:
        return set()
    k = min(k, sv_row.size)
    return set(np.argsort(np.abs(sv_row))[-k:].tolist())


def main() -> int:
    import shap
    from common import loads_signed
    from module4_explanations.io import load_test_data
    from module4_explanations.compute import _normalise_shap_output
    from module4_explanations.stability import compute_robust_top_features

    X, _y, _ac, feat_names = load_test_data()
    pkl = PROJECT_ROOT / "results/models/xgboost_final_pipeline.pkl"
    obj = loads_signed(pkl)
    clf = obj.named_steps["classifier"] if hasattr(obj, "named_steps") else obj
    explainer = shap.TreeExplainer(clf)

    cached = np.load(PROJECT_ROOT / "results/reports/shap_values_xgboost.npz",
                     allow_pickle=True)
    flagged_mask = np.abs(cached["shap_values"]).sum(axis=1) > 0
    sampled = np.where(flagged_mask)[0][:80]  # cap at 80 for runtime

    top_k = 5
    n_perturbations = 20
    sigma = 0.01

    single_overlaps: list[float] = []
    robust_overlaps: list[float] = []

    for k, idx in enumerate(sampled):
        idx = int(idx)
        x_row = X[idx]

        # Single-shot SHAP twice with different perturbations
        rng_a = np.random.default_rng(seed=10_000 + idx)
        rng_b = np.random.default_rng(seed=20_000 + idx)
        noise_a = rng_a.normal(0.0, sigma, size=len(x_row)).astype(x_row.dtype)
        noise_b = rng_b.normal(0.0, sigma, size=len(x_row)).astype(x_row.dtype)
        sv_a = _normalise_shap_output(
            explainer.shap_values((x_row + noise_a).reshape(1, -1))
        )[0]
        sv_b = _normalise_shap_output(
            explainer.shap_values((x_row + noise_b).reshape(1, -1))
        )[0]
        single_overlaps.append(_jaccard(_top_k(sv_a, top_k), _top_k(sv_b, top_k)))

        # Robust SHAP twice with different perturbations
        rng_c = np.random.default_rng(seed=30_000 + idx)
        rng_d = np.random.default_rng(seed=40_000 + idx)
        robust_a = compute_robust_top_features(
            explainer, x_row, list(feat_names),
            n_perturbations=n_perturbations, sigma=sigma, top_k=top_k, rng=rng_c,
        )
        robust_b = compute_robust_top_features(
            explainer, x_row, list(feat_names),
            n_perturbations=n_perturbations, sigma=sigma, top_k=top_k, rng=rng_d,
        )
        set_a = {feat_names.index(f["feature"]) for f in robust_a}
        set_b = {feat_names.index(f["feature"]) for f in robust_b}
        robust_overlaps.append(_jaccard(set_a, set_b))

        if (k + 1) % 20 == 0:
            print(f"  …{k+1}/{len(sampled)} processed "
                  f"(single mean Jaccard so far: {np.mean(single_overlaps):.3f}, "
                  f"robust: {np.mean(robust_overlaps):.3f})")

    single_mean = float(np.mean(single_overlaps))
    robust_mean = float(np.mean(robust_overlaps))
    delta = robust_mean - single_mean

    print()
    print("=" * 76)
    print(" Stability variant comparison")
    print("=" * 76)
    print(f"  Single-shot mean Jaccard: {single_mean:.4f}")
    print(f"  Robust mean Jaccard:      {robust_mean:.4f}")
    print(f"  Δ:                        {delta:+.4f}")
    print(f"  Sprint 5 / Tầng 3.2 target: Δ ≥ +0.10")
    print(f"  Verdict: {'✓ MET' if delta >= 0.10 else '✗ NOT MET'}")
    print("=" * 76)

    report = {
        "n_samples":            len(sampled),
        "n_perturbations_each": n_perturbations,
        "sigma":                sigma,
        "top_k":                top_k,
        "single_shot_mean_jaccard": single_mean,
        "robust_mean_jaccard":      robust_mean,
        "delta":                    delta,
        "target_delta":             0.10,
        "met_target":               delta >= 0.10,
    }
    out = PROJECT_ROOT / "results" / "stability_variant_comparison.json"
    from common.artifact_versioning import embed_version_in_dict
    out.write_text(json.dumps(embed_version_in_dict(report, out.name), indent=2))
    print(f"\nWrote {out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
