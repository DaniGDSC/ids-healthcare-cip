#!/usr/bin/env python3
"""Compute RQ1 detection metrics + ablation tables.

Writes 5 JSON artifacts to `results/`:
  * rq1_metrics.json            — headline metrics (FNR_crit, Sens, Spec, AUC, F2)
  * rq1_ablation_track_a.json   — XGBoost / RF / DT comparison
  * rq1_ablation_track_b.json   — DAE-raw vs cascade
  * rq1_track_b_per_class.json  — per-attack-category Track B breakdown
  * rq1_weight_sensitivity.json — composite risk weight sensitivity

Input artifacts:
  * results/reports/risk_scores.npz — composite + components for 2448 samples
  * results/reports/analyst_report.json — per-model predictions (668 samples)
  * results/reports/alert_responses.json — attack_category per sample
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Prepend project root so module3_risk_scoring + src.risk_scorer are
# importable when this script is executed directly (it is not a package
# member of any subdir).
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
RESULTS = PROJECT_ROOT / "results"
REPORTS = RESULTS / "reports"

# Conventions:
#   "Critical device" = d_crit >= 0.8 (life-critical class — Invariant 2 anchor).
#   Surfacing decision = composite risk R drives tier; LOW = suppressed,
#   MEDIUM+ = surfaced to operator.
#   Detection threshold for binary metrics: c_detect >= 0.5 (calibrated).
CRITICAL_DEVICE_THRESHOLD = 0.8
SURFACED_TIERS = ("MEDIUM", "HIGH", "CRITICAL")
DETECTION_THRESHOLD = 0.5


def _safe_div(num, den):
    return float(num / den) if den else 0.0


def _binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5):
    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    prec, rec, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0, labels=[0, 1]
    )
    sens = _safe_div(tp, tp + fn)        # recall = TPR = sensitivity
    spec = _safe_div(tn, tn + fp)        # TNR
    fnr = _safe_div(fn, fn + tp)         # 1 - sensitivity
    # F2 favors recall over precision (β=2)
    if prec + rec > 0:
        f2 = 5 * prec * rec / (4 * prec + rec)
    else:
        f2 = 0.0
    try:
        auc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auc = float("nan")
    return {
        "n": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "n_negative": int((y_true == 0).sum()),
        "threshold": float(threshold),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "sensitivity": round(sens, 6),
        "specificity": round(spec, 6),
        "precision": round(float(prec), 6),
        "recall": round(float(rec), 6),
        "fnr": round(fnr, 6),
        "fpr": round(_safe_div(fp, fp + tn), 6),
        "f1": round(float(2 * prec * rec / (prec + rec)) if prec + rec > 0 else 0.0, 6),
        "f2": round(f2, 6),
        "auc": round(auc, 6),
    }


# ──────────────────────────────────────────────────────────────────────
# R1: Headline metrics
# ──────────────────────────────────────────────────────────────────────
def compute_headline_metrics():
    d = np.load(REPORTS / "risk_scores.npz", allow_pickle=True)
    y_true = d["y_true"].astype(int)
    c_detect = d["c_detect"].astype(float)
    c_track_a = d["c_track_a"].astype(float)
    c_track_b = d["c_track_b"].astype(float)
    R = d["R"].astype(float)
    risk_levels = d["risk_levels"]
    d_crit = d["d_crit"].astype(float)

    surfaced_mask = np.isin(risk_levels, SURFACED_TIERS)
    surfacing_metrics = _binary_metrics(
        y_true, surfaced_mask.astype(float), threshold=0.5
    )

    crit_device_mask = d_crit >= CRITICAL_DEVICE_THRESHOLD
    crit_attack_mask = crit_device_mask & (y_true == 1)
    n_crit_attacks = int(crit_attack_mask.sum())
    n_crit_attacks_surfaced = int((crit_attack_mask & surfaced_mask).sum())
    fnr_critical = _safe_div(n_crit_attacks - n_crit_attacks_surfaced, n_crit_attacks)

    return {
        "_meta": {
            "description": "RQ1 headline detection metrics",
            "n_total_samples": int(len(y_true)),
            "n_attacks": int(y_true.sum()),
            "n_normal": int((y_true == 0).sum()),
            "attack_rate": round(float(y_true.mean()), 4),
            "critical_device_threshold_d_crit": CRITICAL_DEVICE_THRESHOLD,
            "surfaced_tiers": list(SURFACED_TIERS),
            "detection_threshold": DETECTION_THRESHOLD,
        },
        "primary_safety_metric": {
            "FNR_critical": round(fnr_critical, 6),
            "FNR_critical_target": 0.05,
            "target_met": bool(fnr_critical < 0.05),
            "n_critical_device_attacks": n_crit_attacks,
            "n_critical_attacks_surfaced": n_crit_attacks_surfaced,
            "definition": (
                "Among attacks on critical devices "
                f"(d_crit >= {CRITICAL_DEVICE_THRESHOLD}), the fraction NOT "
                "surfaced (tier == LOW)."
            ),
        },
        "track_a_detection": _binary_metrics(y_true, c_track_a, DETECTION_THRESHOLD),
        "track_b_detection": _binary_metrics(y_true, c_track_b, DETECTION_THRESHOLD),
        "fused_detection_c_detect": _binary_metrics(y_true, c_detect, DETECTION_THRESHOLD),
        "composite_R": {
            "auc_vs_y_true": round(float(roc_auc_score(y_true, R)), 6),
            "score_range": [float(R.min()), float(R.max())],
            "score_mean": float(R.mean()),
        },
        "surfacing_decision": surfacing_metrics,
        "tier_distribution": {
            "CRITICAL": int((risk_levels == "CRITICAL").sum()),
            "HIGH": int((risk_levels == "HIGH").sum()),
            "MEDIUM": int((risk_levels == "MEDIUM").sum()),
            "LOW": int((risk_levels == "LOW").sum()),
        },
    }


# ──────────────────────────────────────────────────────────────────────
# R2: Track A model comparison (XGBoost vs RF vs DT)
# ──────────────────────────────────────────────────────────────────────
def compute_track_a_ablation():
    """Compare XGBoost (runtime) vs RF / DT (baselines) on the full test set.

    After Phase 2 the analyst report no longer carries RF/DT prediction
    blocks (Module 4 only consults XGBoost + DAE), so this function reads
    each model's per-sample predictions directly from the npz files that
    Module 2's ``predict_split`` writes — those still cover all three
    models because M2 merges the runtime + baseline registries.
    """
    from common import split_paths as sp

    d = np.load(REPORTS / "risk_scores.npz", allow_pickle=True)
    y_true = d["y_true"].astype(int)

    models = ("xgboost", "random_forest", "decision_tree")
    per_model = {}
    for m in models:
        npz_path = sp.model_predictions(m, "test")
        if not npz_path.exists():
            per_model[m] = {"_skipped": f"missing {npz_path.name} — run module2"}
            continue
        npz = np.load(npz_path)
        confidence = npz["y_proba"].astype(float)
        prediction = npz["y_pred"].astype(int)
        # Length guard: should match risk_scores y_true row-for-row since
        # both come from the same test parquet.
        if len(confidence) != len(y_true):
            per_model[m] = {
                "_skipped": (
                    f"length mismatch: y_proba={len(confidence)} vs "
                    f"y_true={len(y_true)} (test parquet drifted?)"
                ),
            }
            continue
        metrics_at_05 = _binary_metrics(y_true, confidence, threshold=0.5)
        cm = confusion_matrix(y_true, prediction, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        prec, rec, _, _ = precision_recall_fscore_support(
            y_true, prediction, average="binary", zero_division=0, labels=[0, 1]
        )
        per_model[m] = {
            "metrics_at_threshold_0.5": metrics_at_05,
            "metrics_native_prediction": {
                "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
                "sensitivity": round(_safe_div(tp, tp + fn), 6),
                "specificity": round(_safe_div(tn, tn + fp), 6),
                "precision": round(float(prec), 6),
                "f1": round(
                    float(2 * prec * rec / (prec + rec)) if prec + rec > 0 else 0.0,
                    6,
                ),
            },
        }

    return {
        "_meta": {
            "description": (
                "Track A model comparison on full test set "
                "(reads per-model npz directly; analyst_report no longer "
                "carries RF/DT after Phase 2 runtime cleanup)."
            ),
            "n_samples": int(len(y_true)),
            "n_attacks": int(y_true.sum()),
        },
        "models": per_model,
    }


# ──────────────────────────────────────────────────────────────────────
# R3: Track B cascade ablation (DAE-raw vs cascade c_track_b)
# ──────────────────────────────────────────────────────────────────────
def compute_track_b_ablation():
    d = np.load(REPORTS / "risk_scores.npz", allow_pickle=True)
    y_true_full = d["y_true"].astype(int)
    c_track_b_full = d["c_track_b"].astype(float)

    with open(REPORTS / "analyst_report.json") as f:
        analyst = json.load(f)
    indices = [a["sample_index"] for a in analyst]
    y_true_subset = y_true_full[indices]

    # DAE raw — reconstruction error from analyst_report. Higher = more anomalous.
    # Normalize to [0,1] by min-max for AUC compatibility.
    dae_recon = np.array([
        (a.get("models", {}).get("dae") or {}).get("reconstruction_error", 0.0)
        for a in analyst
    ], dtype=float)
    # Robust normalization — clip outlier 99th percentile to avoid AUC distortion
    p99 = np.percentile(dae_recon, 99) if len(dae_recon) else 1.0
    dae_norm = np.clip(dae_recon, 0, p99) / max(p99, 1e-12)

    dae_pred = np.array([
        (a.get("models", {}).get("dae") or {}).get("prediction", 0) for a in analyst
    ], dtype=int)

    # Cascade (Track B post-cascade composite — same indices for fair comparison)
    c_track_b_subset = c_track_b_full[indices]

    return {
        "_meta": {
            "description": (
                "Track B cascade ablation — DAE-raw reconstruction error vs "
                "post-cascade composite c_track_b. Compared on the analyst "
                "report subset to ensure both have outputs for the same samples."
            ),
            "n_samples": int(len(y_true_subset)),
            "n_attacks_in_subset": int(y_true_subset.sum()),
            "dae_recon_99th_pct": float(p99),
        },
        "dae_raw_reconstruction_error": _binary_metrics(
            y_true_subset, dae_norm, threshold=0.5
        ),
        "dae_native_prediction": {
            "tp": int(((dae_pred == 1) & (y_true_subset == 1)).sum()),
            "fp": int(((dae_pred == 1) & (y_true_subset == 0)).sum()),
            "tn": int(((dae_pred == 0) & (y_true_subset == 0)).sum()),
            "fn": int(((dae_pred == 0) & (y_true_subset == 1)).sum()),
            "sensitivity": round(
                _safe_div(
                    ((dae_pred == 1) & (y_true_subset == 1)).sum(),
                    (y_true_subset == 1).sum(),
                ), 6,
            ),
        },
        "track_b_cascade_c_track_b": _binary_metrics(
            y_true_subset, c_track_b_subset, threshold=DETECTION_THRESHOLD
        ),
    }


# ──────────────────────────────────────────────────────────────────────
# R4: Track B per-class breakdown
# ──────────────────────────────────────────────────────────────────────
def compute_track_b_per_class():
    d = np.load(REPORTS / "risk_scores.npz", allow_pickle=True)
    y_true = d["y_true"].astype(int)
    c_track_b = d["c_track_b"].astype(float)
    n_total = int(len(y_true))

    with open(REPORTS / "alert_responses.json") as f:
        records = json.load(f)["records"]
    # alert_responses.json is a subset of test rows (surfaced alerts);
    # build a full-length (n_total,) categories array by joining on
    # sample_index. Rows that did not surface inherit "normal" so the
    # one-vs-rest AUC for "normal" still has the right negative class.
    categories = np.full(n_total, "normal", dtype=object)
    for r in records:
        idx = r.get("sample_index")
        if idx is None or not (0 <= idx < n_total):
            continue
        categories[int(idx)] = r.get("attack_category", "unknown")

    per_class = {}
    for cat in sorted(set(categories)):
        cat_mask = categories == cat
        is_attack_class = (cat != "normal")
        # For per-class AUC, compute one-vs-rest: this class vs everything else
        ovr_y = (cat_mask & (y_true == 1)).astype(int) if is_attack_class else (y_true == 0).astype(int)
        if 0 < ovr_y.sum() < len(ovr_y):
            try:
                auc = float(roc_auc_score(ovr_y, c_track_b))
            except ValueError:
                auc = float("nan")
        else:
            auc = float("nan")

        if is_attack_class:
            cat_score_subset = c_track_b[cat_mask]
            # Among samples of this attack class, what fraction surface at threshold?
            detected = int((cat_score_subset >= DETECTION_THRESHOLD).sum())
            n = int(cat_mask.sum())
            recall_for_class = _safe_div(detected, n)
        else:
            recall_for_class = None

        per_class[cat] = {
            "n_samples": int(cat_mask.sum()),
            "is_attack": is_attack_class,
            "auc_one_vs_rest": round(auc, 6) if not np.isnan(auc) else None,
            "recall_at_threshold_0.5": round(recall_for_class, 6) if recall_for_class is not None else None,
            "track_b_score_mean": round(float(c_track_b[cat_mask].mean()), 6),
            "track_b_score_std": round(float(c_track_b[cat_mask].std()), 6),
        }

    return {
        "_meta": {
            "description": "Track B per-attack-category breakdown",
            "threshold": DETECTION_THRESHOLD,
            "n_total": int(len(y_true)),
        },
        "per_class": per_class,
    }


# ──────────────────────────────────────────────────────────────────────
# R5: Composite risk weight sensitivity
# ──────────────────────────────────────────────────────────────────────
def compute_weight_sensitivity():
    """Vary the composite risk weights and recompute FNR_critical + AUC.

    Composite formula (canonical, from
    `module3_risk_scoring/module3_risk_scores.py::WEIGHTS`):

        R = w1·C_detect + w2·D_crit + w3·S_data + w4·D_clinical_tier

    Grid search perturbs each weight ±0.10 around the canonical anchor
    (w1=0.40, w2=0.25, w3=0.15, w4=0.20) and renormalizes to sum=1.

    The canonical baseline is guaranteed to appear as exactly one row in
    the grid so reviewers can read sensitivity around the actual
    operating point — not around an approximation.
    """
    # Canonical baseline (R3 fix — was approximated as 0.40/0.30/0.20/0.10
    # in the original pass; canonical is 0.40/0.25/0.15/0.20 with D_clinical
    # weighted more heavily than D_crit).
    try:
        from module3_risk_scoring.module3_risk_scores import (
            WEIGHTS as _MODULE3_WEIGHTS,
            RISK_THRESHOLDS as _MODULE3_TIER_THRESHOLDS,
        )
    except Exception as e:
        # Hard failure — sensitivity analysis without canonical anchor is
        # the bug we're trying to fix. Re-raise with context.
        raise ImportError(
            "Could not import canonical weights from "
            "module3_risk_scoring.module3_risk_scores — "
            f"re-run the pipeline or check the import path. Underlying error: {e}"
        )

    # Module 3 names them w1..w4; surface as (alpha, beta, gamma, delta) for
    # readability and to match the existing API of the audit JSON.
    canonical_baseline = {
        "alpha": float(_MODULE3_WEIGHTS["w1"]),   # C_detect
        "beta":  float(_MODULE3_WEIGHTS["w2"]),   # D_crit
        "gamma": float(_MODULE3_WEIGHTS["w3"]),   # S_data
        "delta": float(_MODULE3_WEIGHTS["w4"]),   # D_clinical_tier
    }

    # MEDIUM cutoff = surfacing threshold (anything below this lands in LOW
    # and is suppressed). Lookup by tier name — RISK_THRESHOLDS gained a
    # 4th NORMAL row in the formula-fix upgrade, so ``[-1]`` no longer
    # points at MEDIUM. Look up by name to stay stable across future
    # threshold additions (matches the assertion in
    # tests/test_rq1_weight_sensitivity.py::test_surfacing_threshold_canonical).
    surfacing_threshold = float(
        next(t for t, name in _MODULE3_TIER_THRESHOLDS if name == "MEDIUM")
    )

    d = np.load(REPORTS / "risk_scores.npz", allow_pickle=True)
    y_true = d["y_true"].astype(int)
    c_detect = d["c_detect"].astype(float)
    d_crit = d["d_crit"].astype(float)
    s_data = d["s_data"].astype(float)
    d_clinical = d["d_clinical_tier"].astype(float)

    crit_attack_mask = (d_crit >= CRITICAL_DEVICE_THRESHOLD) & (y_true == 1)

    # Grid: each weight perturbed at -0.10, 0, +0.10 around the canonical
    # anchor (clamped to [0.05, 1.0]). Configurations whose weights don't
    # sum to ~1 after the perturbation are skipped post-renormalization.
    # Note: set-dedupe collapses perturbations that land on the same clamped
    # value (e.g., center=0.05 → {0.05, 0.05, 0.15} → 2 entries). Acceptable
    # because the canonical anchor (0.40/0.25/0.15/0.20) never hits the
    # clamp; sanity-checked downstream via `n_canon == 1` assertion.
    def _perturbations(center: float) -> list[float]:
        return sorted({round(max(0.05, min(1.0, center + step)), 3)
                       for step in (-0.10, 0.0, 0.10)})

    grid = []
    canonical_row = None
    for alpha in _perturbations(canonical_baseline["alpha"]):
        for beta in _perturbations(canonical_baseline["beta"]):
            for gamma in _perturbations(canonical_baseline["gamma"]):
                for delta in _perturbations(canonical_baseline["delta"]):
                    s = alpha + beta + gamma + delta
                    if not (0.85 <= s <= 1.15):
                        continue
                    R_new = (
                        alpha * c_detect + beta * d_crit
                        + gamma * s_data + delta * d_clinical
                    ) / s
                    surfaced = R_new >= surfacing_threshold
                    missed_crit = (crit_attack_mask & ~surfaced).sum()
                    fnr_crit = _safe_div(missed_crit, crit_attack_mask.sum())
                    try:
                        auc = float(roc_auc_score(y_true, R_new))
                    except ValueError:
                        auc = float("nan")
                    is_canonical = (
                        alpha == canonical_baseline["alpha"]
                        and beta == canonical_baseline["beta"]
                        and gamma == canonical_baseline["gamma"]
                        and delta == canonical_baseline["delta"]
                    )
                    row = {
                        "alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta,
                        "weight_sum": round(s, 3),
                        "is_canonical": is_canonical,
                        "FNR_critical": round(fnr_crit, 6),
                        "AUC": round(auc, 6),
                        "n_surfaced": int(surfaced.sum()),
                    }
                    grid.append(row)
                    if is_canonical:
                        canonical_row = row

    # Sanity: canonical baseline must appear in the grid exactly once.
    n_canon = sum(1 for g in grid if g["is_canonical"])
    if n_canon != 1:
        raise RuntimeError(
            f"Canonical baseline must appear in grid exactly once (got "
            f"{n_canon}). Check perturbation logic — the baseline anchor "
            "must be one of the chosen perturbations."
        )

    fnr_crits = [g["FNR_critical"] for g in grid]
    aucs = [g["AUC"] for g in grid if not np.isnan(g["AUC"])]
    best = min(grid, key=lambda g: g["FNR_critical"])
    worst = max(grid, key=lambda g: g["FNR_critical"])

    return {
        "_meta": {
            "description": (
                "Composite risk weight sensitivity grid search. Each row "
                "shows FNR_critical and AUC at a particular (α, β, γ, δ) "
                "weighting after renormalizing to sum=1. Grid perturbs "
                "each weight by ±0.10 around the canonical anchor."
            ),
            "baseline_canonical": canonical_baseline,
            "baseline_source": (
                "module3_risk_scoring.module3_risk_scores.WEIGHTS "
                "(w1=C_detect, w2=D_crit, w3=S_data, w4=D_clinical_tier)"
            ),
            "surfacing_threshold_on_R": surfacing_threshold,
            "surfacing_threshold_source": (
                "module3_risk_scoring.module3_risk_scores.RISK_THRESHOLDS "
                "[MEDIUM cutoff — anything below = LOW = suppressed]"
            ),
            "n_critical_device_attacks": int(crit_attack_mask.sum()),
            "grid_size": len(grid),
        },
        "canonical_baseline_row": canonical_row,
        "summary": {
            "FNR_critical_min": min(fnr_crits),
            "FNR_critical_max": max(fnr_crits),
            "FNR_critical_range": round(max(fnr_crits) - min(fnr_crits), 6),
            "AUC_min": round(min(aucs), 6),
            "AUC_max": round(max(aucs), 6),
            "best_config": best,
            "worst_config": worst,
        },
        "grid": grid,
    }


def main():
    artifacts = {}

    print("[R1] Computing headline metrics...")
    artifacts["rq1_metrics.json"] = compute_headline_metrics()

    print("[R2] Computing Track A ablation...")
    artifacts["rq1_ablation_track_a.json"] = compute_track_a_ablation()

    print("[R3] Computing Track B cascade ablation...")
    artifacts["rq1_ablation_track_b.json"] = compute_track_b_ablation()

    print("[R4] Computing Track B per-class breakdown...")
    artifacts["rq1_track_b_per_class.json"] = compute_track_b_per_class()

    print("[R5] Computing weight sensitivity grid...")
    artifacts["rq1_weight_sensitivity.json"] = compute_weight_sensitivity()

    # Write all
    for name, payload in artifacts.items():
        out = RESULTS / name
        with open(out, "w") as f:
            json.dump(payload, f, indent=2, default=float)
        print(f"  wrote {out}")

    print()
    print("=== Quick summary ===")
    m = artifacts["rq1_metrics.json"]
    print(f"  FNR_critical: {m['primary_safety_metric']['FNR_critical']} (target <0.05)")
    print(f"  Sensitivity (Track A): {m['track_a_detection']['sensitivity']}")
    print(f"  Specificity (Track A): {m['track_a_detection']['specificity']}")
    print(f"  AUC (Track A):         {m['track_a_detection']['auc']}")
    print(f"  AUC (Track B):         {m['track_b_detection']['auc']}")
    print(f"  AUC (fused C_detect):  {m['fused_detection_c_detect']['auc']}")
    print(f"  AUC (composite R):     {m['composite_R']['auc_vs_y_true']}")
    print(f"  Target_met:            {m['primary_safety_metric']['target_met']}")


if __name__ == "__main__":
    main()
