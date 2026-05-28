#!/usr/bin/env python3
"""Offline regeneration of Module 4's clinician + analyst artifacts.

Used to verify Phase 1.1 (observation_phrase rendering) and Phase 2
(counterfactual injection) without re-executing TreeSHAP. Loads:

  - cached SHAP values per Track-A model from ``results/reports/shap_values_*.npz``
  - cached model predictions from ``results/models/*_test_predictions.npz``
  - the test split parquet (for ``X_test``)
  - cached risk_levels from ``results/reports/risk_scores.npz``

…and calls ``build_analyst_report`` + ``build_clinician_summaries``
with the new ``X_test=`` keyword so the Phase 1.1 observation phrase
gets rendered.

Phase 2 — when ``--counterfactual`` is passed (default ON), the XGBoost
classifier is loaded directly via ``joblib.load`` (bypassing the signed
pickle verifier — see note in main) and one counterfactual is computed
per XGBoost-flagged sample. The result is injected into both the
analyst entry (under ``counterfactual``) and the clinician summary
(as an appended clause + ``counterfactual`` field).

Does NOT regenerate admin_dashboard.json or the example_explanations
artefact — those are not the targets of Phase 0/1/2 metrics.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common import split_paths as sp  # noqa: E402
from module4_explanations.config import SHAP_MODELS, TRACK_A_MODELS  # noqa: E402
from module4_explanations.counterfactual import compute_counterfactual  # noqa: E402
from module4_explanations.stability import compute_stability  # noqa: E402
from module4_explanations.io import OUTPUT_DIR, load_test_data  # noqa: E402
from module4_explanations.stakeholder import (  # noqa: E402
    build_analyst_report,
    build_clinician_summaries,
)


def main(split: str = "test") -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    paths = {
        "parquet": sp.parquet(split),
        "xgboost_preds": sp.model_predictions("xgboost", split),
        "dae_preds": sp.dae_predictions(split),
        "risk_npz": sp.risk_scores(split),
        "suffix": sp.suffix(split),
    }

    X_test, y_test, attack_cats, feat_names = load_test_data(paths["parquet"])
    n_samples = len(y_test)
    print(
        f"[phase1-regen] split={split} n_samples={n_samples} n_features={len(feat_names)}"
    )

    # Predictions — same order as TRACK_A_MODELS dict iteration (XGBoost only after Phase 2)
    pred_paths = {
        "xgboost": paths["xgboost_preds"],
    }
    all_preds: dict = {}
    for name in TRACK_A_MODELS:
        data = np.load(pred_paths[name])
        all_preds[name] = {k: data[k] for k in data.files}
    dae_data = np.load(paths["dae_preds"])
    dae_preds = {k: dae_data[k] for k in dae_data.files}

    # Cached SHAP from npz — same shape contract as compute_tree_shap.
    # Per-split suffix matches the prediction npz scheme: the demo
    # cache file is ``shap_values_<model>_demo.npz`` (Sprint 2 1.3),
    # while test stays at ``shap_values_<model>.npz``.
    suffix_npz = "_demo" if split == "demo" else ""
    all_shap: dict = {}
    for name in SHAP_MODELS:
        npz_path = (
            PROJECT_ROOT / "results" / "reports" / f"shap_values_{name}{suffix_npz}.npz"
        )
        if not npz_path.exists():
            print(f"  WARN: missing {npz_path.name} — skipping {name}")
            continue
        data = np.load(npz_path, allow_pickle=True)
        all_shap[name] = data["shap_values"]
        cached_feats = list(data["feature_names"])
        if cached_feats != list(feat_names):
            raise RuntimeError(
                f"feature-name mismatch between cached SHAP ({len(cached_feats)} feats) "
                f"and parquet ({len(feat_names)} feats). Re-run Module 4 fully."
            )

    # DAE per-feature errors (needed by build_analyst_report).
    # For the demo split, compute on the fly because Module 4 in
    # explanations-only mode didn't cache them (and re-running full
    # mode is expensive). Falls back gracefully when the demo cache
    # exists too.
    dae_err_path = (
        PROJECT_ROOT / "results" / "reports" / f"dae_feature_errors{suffix_npz}.npz"
    )
    if dae_err_path.exists():
        dae_err = np.load(dae_err_path, allow_pickle=True)
        weighted_err = dae_err["weighted_per_feature_error"]
    else:
        from module4_explanations.compute import compute_dae_feature_errors

        print(
            f"[phase1-regen] DAE per-feature errors not cached for split={split} — "
            "computing on the fly"
        )
        _sq, weighted_err, _w = compute_dae_feature_errors(X_test, list(feat_names))

    # Risk levels (Module 3 canonical severity)
    if not paths["risk_npz"].exists():
        raise FileNotFoundError(
            f"{paths['risk_npz']} missing — Phase 1.1 verification needs Module 3 risk levels."
        )
    risk_levels = np.load(paths["risk_npz"], allow_pickle=True)["risk_levels"]

    # ── Phase 2 — counterfactual per XGBoost-flagged alert ──
    # Loads via the signed-pickle verifier. Sprint 1.1 added
    # ``tools/resign_models`` so the sidecar stays current; if this raises
    # ``SignedPickleError`` it means the operator needs to re-run
    # ``python -m tools.resign_models`` first.
    counterfactuals_by_idx: dict[int, dict] = {}
    stability_by_idx: dict[int, dict] = {}
    if "xgboost" in all_shap:
        import shap
        from common import loads_signed

        pkl_path = PROJECT_ROOT / "results/models/xgboost_final_pipeline.pkl"
        obj = loads_signed(pkl_path)
        clf = obj.named_steps["classifier"] if hasattr(obj, "named_steps") else obj

        try:
            from common.model_registry import get_track_a_thresholds

            threshold = float(get_track_a_thresholds()["xgboost"])
        except Exception:
            threshold = 0.5

        xgb_preds = all_preds["xgboost"]["y_pred"]
        flagged_idx = np.where(xgb_preds == 1)[0]
        print(
            f"[phase1-regen] computing counterfactuals for {len(flagged_idx)} "
            f"XGBoost-flagged samples (threshold={threshold:.4f})"
        )
        n_feasible = 0
        for k, idx in enumerate(flagged_idx):
            idx = int(idx)
            r = compute_counterfactual(
                clf,
                X_test[idx],
                all_shap["xgboost"][idx],
                list(feat_names),
                threshold,
            )
            counterfactuals_by_idx[idx] = r.to_dict()
            if r.feasible:
                n_feasible += 1
            if (k + 1) % 100 == 0:
                print(
                    f"  …{k + 1}/{len(flagged_idx)} processed "
                    f"({n_feasible} feasible so far)"
                )
        print(
            f"[phase1-regen] counterfactual coverage: "
            f"{n_feasible}/{len(flagged_idx)} feasible"
        )

        # ── Phase 4.1 — stability badge per XGBoost-flagged alert ──
        # Deterministic RNG seeded by alert index so per-alert
        # stability is reproducible across CI runs.
        explainer = shap.TreeExplainer(clf)
        print(f"[phase1-regen] computing stability for {len(flagged_idx)} samples")
        band_counts = {"STABLE": 0, "BORDERLINE": 0, "UNSTABLE": 0}
        for k, idx in enumerate(flagged_idx):
            idx = int(idx)
            rng = np.random.default_rng(seed=42 + idx)
            r = compute_stability(
                explainer,
                X_test[idx],
                list(feat_names),
                rng=rng,
                baseline_shap_row=all_shap["xgboost"][idx],
            )
            stability_by_idx[idx] = r.to_dict()
            band_counts[r.band] = band_counts.get(r.band, 0) + 1
            if (k + 1) % 100 == 0:
                print(
                    f"  …{k + 1}/{len(flagged_idx)} processed  "
                    f"(STABLE={band_counts['STABLE']}, "
                    f"BORDERLINE={band_counts['BORDERLINE']}, "
                    f"UNSTABLE={band_counts['UNSTABLE']})"
                )
        print(
            f"[phase1-regen] stability bands: "
            f"STABLE={band_counts['STABLE']}, "
            f"BORDERLINE={band_counts['BORDERLINE']}, "
            f"UNSTABLE={band_counts['UNSTABLE']}"
        )

        # ── Sprint 5 / Tầng 3.4 — RandomForest counterfactual augment ──
        # The XGBoost engine misses LOW-tier alerts flagged only by RF
        # (Phase 0 baseline showed ~6% LOW-tier CF coverage). Loop over
        # RF-flagged samples that XGBoost did NOT flag and run the same
        # counterfactual engine — SHAP for RF is computed on the fly
        # since only XGBoost SHAP is cached.
        try:
            rf_preds_data = np.load(sp.model_predictions("random_forest", split))
        except FileNotFoundError:
            rf_preds_data = None

        if rf_preds_data is not None:
            rf_y_pred = rf_preds_data["y_pred"]
            xgb_y_pred = all_preds["xgboost"]["y_pred"]
            rf_only = np.where((rf_y_pred == 1) & (xgb_y_pred == 0))[0]
            if len(rf_only):
                print(f"[phase1-regen] augmenting CF for {len(rf_only)} RF-only-flagged samples")
                from common import loads_signed
                rf_pkl = PROJECT_ROOT / "results/models/random_forest_final_pipeline.pkl"
                rf_obj = loads_signed(rf_pkl)
                rf_clf = rf_obj.named_steps["classifier"] if hasattr(rf_obj, "named_steps") else rf_obj
                rf_explainer = shap.TreeExplainer(rf_clf)
                try:
                    from common.model_registry import get_baseline_thresholds
                    rf_thr = float(get_baseline_thresholds()["random_forest"])
                except Exception:
                    rf_thr = 0.5
                rf_feasible = 0
                for k, idx in enumerate(rf_only):
                    idx = int(idx)
                    sv = rf_explainer.shap_values(X_test[idx].reshape(1, -1))
                    if isinstance(sv, list):
                        sv_row = sv[1][0] if len(sv) > 1 else sv[0][0]
                    elif sv.ndim == 3:
                        sv_row = sv[0, :, 1]
                    else:
                        sv_row = sv[0]
                    r = compute_counterfactual(
                        rf_clf, X_test[idx], sv_row,
                        list(feat_names), rf_thr,
                    )
                    counterfactuals_by_idx[idx] = r.to_dict()
                    if r.feasible:
                        rf_feasible += 1
                    if (k + 1) % 50 == 0:
                        print(f"  …{k+1}/{len(rf_only)} RF samples ({rf_feasible} feasible)")
                print(f"[phase1-regen] RF CF augment: {rf_feasible}/{len(rf_only)} feasible")

    # Build outputs — Phase 1.1: X_test is now passed through;
    # Phase 2: counterfactuals injected when present;
    # Phase 4.1: stability badges injected when present.
    build_analyst_report(
        all_shap,
        all_preds,
        weighted_err,
        dae_preds,
        feat_names,
        risk_levels,
        suffix=paths["suffix"],
        output_dir=OUTPUT_DIR,
        counterfactuals_by_idx=counterfactuals_by_idx or None,
        stability_by_idx=stability_by_idx or None,
    )
    build_clinician_summaries(
        all_shap,
        all_preds,
        dae_preds,
        feat_names,
        risk_levels,
        suffix=paths["suffix"],
        output_dir=OUTPUT_DIR,
        X_test=X_test,
        counterfactuals_by_idx=counterfactuals_by_idx or None,
        stability_by_idx=stability_by_idx or None,
    )

    print(f"[phase1-regen] wrote outputs under {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "test"))
