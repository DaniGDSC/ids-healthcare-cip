"""Sanity checks for the surprising ``DAE-probas-only`` win on MedSec-25.

The ablation in ``analysis/dae_ablation_loo.py`` showed that a 1-D
``|z-score(P_xgb)|`` baseline beats both the raw and cascaded DAEs on
MedSec-25 LOO at AUC ≈ 0.997 across every held-out class. That number
is suspicious enough to deserve four independent gut-checks before
writing it into the thesis. This script runs them.

For each fold (one per attack class):

  Sanity 1  Distribution of ``P_xgb`` on test
            ─ overlap of benign-vs-novel ``P_xgb`` distributions, KS
              statistic, percentile summary. Histogram PNG saved under
              ``analysis/outputs/dae_ablation_sanity/``.

  Sanity 2  Train / test contamination
            ─ verify the val benigns used to fit the z-score reference
              distribution share no rows with the test benigns (the
              ones we then z-score-test). MedSec-25 preprocessing does
              a stratified 70/20/10 split; this check confirms the
              splits are disjoint at row level (defense against silent
              dedup or copy bugs).

  Sanity 3  XGB direct recall on held-out class
            ─ how often does the XGBoost classifier flag held-out-class
              rows as attack? If recall is high, the classifier itself
              is generalising (rather than the z-score finding hidden
              distribution shift); if recall is low but the z-score AUC
              is still high, the signal is genuinely distribution shift
              below the decision threshold.

  Sanity 4  OOF cross-check
            ─ re-run the z-score baseline using *out-of-fold*
              predictions on TRAIN benigns as the reference
              distribution. If the AUC drops substantially the val-set
              reference was biased.

Run::

    python -m analysis.dae_ablation_sanity --dataset medsec25 --sample-cap 30000
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from analysis.dae_ablation_loo import (  # noqa: E402
    _DATASETS,
    _features_and_labels,
    _load_split_path,
    _predict_p_attack,
    _stratified_subsample,
    _train_xgb,
)

logger = logging.getLogger(__name__)

OUTPUTS_DIR = PROJECT_ROOT / "analysis" / "outputs" / "dae_ablation_sanity"


def _hash_rows(X: np.ndarray) -> np.ndarray:
    """Per-row SHA-style hash. Used to compare row identity between
    val and test splits without depending on a row-id column."""
    # Round to suppress float-precision differences that aren't real.
    Xr = np.round(X.astype(np.float64), 6)
    # Deterministic hash via tobytes per row.
    return np.array([hash(row.tobytes()) for row in Xr])


def _save_histogram(
    p_benign: np.ndarray,
    p_novel: np.ndarray,
    holdout: str,
    dataset: str,
    out_dir: Path,
) -> Path:
    """Save a 2-population P_xgb histogram. Lazy-import matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    bins = np.linspace(0, 1, 51)
    ax.hist(p_benign, bins=bins, alpha=0.55, label=f"benign (test, n={len(p_benign)})", color="#2563EB")
    ax.hist(p_novel, bins=bins, alpha=0.55, label=f"novel '{holdout}' (n={len(p_novel)})", color="#DC2626")
    ax.set_xlabel("P_xgb (calibrated, P(attack))")
    ax.set_ylabel("count")
    ax.set_title(f"{dataset} — holdout='{holdout}' — P_xgb distributions")
    ax.legend(loc="best")
    plt.tight_layout()
    safe = "".join(c if c.isalnum() else "_" for c in holdout).lower()
    path = out_dir / f"{dataset.lower()}_holdout_{safe}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _percentile_summary(arr: np.ndarray) -> dict:
    if len(arr) == 0:
        return {"n": 0}
    return {
        "n": int(len(arr)),
        "mean":   float(np.mean(arr)),
        "std":    float(np.std(arr)),
        "p01":    float(np.percentile(arr, 1)),
        "p10":    float(np.percentile(arr, 10)),
        "p50":    float(np.percentile(arr, 50)),
        "p90":    float(np.percentile(arr, 90)),
        "p99":    float(np.percentile(arr, 99)),
    }


def _z_score_auc(
    p_xgb_test: np.ndarray,
    y_test: np.ndarray,
    *,
    mu_benign: float,
    sigma_benign: float,
) -> float:
    """Recompute the DAE-probas-only AUC given a benign reference (μ, σ)."""
    if sigma_benign <= 0:
        return float("nan")
    score = np.abs(p_xgb_test - mu_benign) / sigma_benign
    if len(np.unique(y_test)) < 2:
        return float("nan")
    return float(roc_auc_score(y_test, score))


def _oof_p_xgb_benign(
    X_train: np.ndarray, y_train: np.ndarray, *, seed: int, n_splits: int = 3,
) -> np.ndarray:
    """Compute out-of-fold P_xgb on TRAIN benigns by k-fold CV.

    Returns the OOF probabilities for the benign-only rows of
    ``X_train``. Used as an alternative reference distribution for
    the z-score baseline (Sanity 4)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.full(len(y_train), np.nan, dtype=np.float32)
    for fold_idx, (tr, te) in enumerate(skf.split(X_train, y_train)):
        clf = _train_xgb(X_train[tr], y_train[tr], seed=seed + fold_idx)
        oof[te] = _predict_p_attack(clf, X_train[te])
    benign_mask = y_train == 0
    return oof[benign_mask]


def _run_fold_sanity(
    holdout_class: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    seed: int,
    benign_category: str,
    sample_cap: int | None,
    dataset_label: str,
    out_dir: Path,
    skip_oof: bool = False,
) -> dict:
    logger.info("─" * 72)
    logger.info("Sanity fold: holdout=%s", holdout_class)
    logger.info("─" * 72)

    train_filtered = train_df[train_df["Attack Category"] != holdout_class].copy()
    val_filtered = val_df[val_df["Attack Category"] != holdout_class].copy()
    test_keep = test_df[test_df["Attack Category"].isin([benign_category, holdout_class])].copy()
    if sample_cap is not None:
        train_filtered = _stratified_subsample(
            train_filtered, sample_cap=sample_cap, random_state=seed,
        )

    X_train, y_train, _ = _features_and_labels(train_filtered)
    X_val, y_val, _ = _features_and_labels(val_filtered)
    X_test, y_test, _ = _features_and_labels(test_keep)

    val_benign_mask = y_val == 0
    test_benign_mask = y_test == 0
    test_novel_mask = y_test == 1

    # ── Train one fold's XGB ──
    t0 = time.perf_counter()
    xgb = _train_xgb(X_train, y_train, seed=seed)
    p_xgb_val_benign = _predict_p_attack(xgb, X_val[val_benign_mask])
    p_xgb_test = _predict_p_attack(xgb, X_test)
    p_xgb_test_benign = p_xgb_test[test_benign_mask]
    p_xgb_test_novel = p_xgb_test[test_novel_mask]
    logger.info("XGB trained + scored in %.1fs", time.perf_counter() - t0)

    # ── Sanity 1: distribution overlap ──
    ks_stat, ks_p = ks_2samp(p_xgb_test_benign, p_xgb_test_novel)
    histogram_path = _save_histogram(
        p_xgb_test_benign, p_xgb_test_novel, holdout_class,
        dataset_label, out_dir,
    )
    sanity1 = {
        "test_benign_summary": _percentile_summary(p_xgb_test_benign),
        "test_novel_summary":  _percentile_summary(p_xgb_test_novel),
        "ks_statistic": float(ks_stat),
        "ks_p_value":   float(ks_p),
        "histogram_png": str(histogram_path.relative_to(PROJECT_ROOT)),
    }
    logger.info(
        "  S1 KS=%.4f (p=%.2e) | benign p50=%.4f / novel p50=%.4f",
        ks_stat, ks_p, sanity1["test_benign_summary"]["p50"],
        sanity1["test_novel_summary"]["p50"],
    )

    # ── Sanity 2: row-level disjointness between val_benign and test_benign ──
    val_b_hashes = _hash_rows(X_val[val_benign_mask])
    test_b_hashes = _hash_rows(X_test[test_benign_mask])
    overlap = len(set(val_b_hashes) & set(test_b_hashes))
    sanity2 = {
        "val_benign_rows": int(val_benign_mask.sum()),
        "test_benign_rows": int(test_benign_mask.sum()),
        "row_overlap_count": overlap,
        "row_overlap_fraction": float(overlap / max(len(test_b_hashes), 1)),
        "ok": overlap == 0,
    }
    logger.info(
        "  S2 val_benign ∩ test_benign = %d rows (%.4f%% of test_benign)",
        overlap, sanity2["row_overlap_fraction"] * 100,
    )

    # ── Sanity 3: XGB direct recall on held-out + FPR on test benigns ──
    y_pred_novel = xgb.predict(X_test[test_novel_mask])
    y_pred_benign = xgb.predict(X_test[test_benign_mask])
    recall_novel = float((y_pred_novel == 1).mean())
    fpr_benign = float((y_pred_benign == 1).mean())
    sanity3 = {
        "n_test_novel": int(test_novel_mask.sum()),
        "n_test_benign": int(test_benign_mask.sum()),
        "xgb_recall_on_holdout_class": recall_novel,
        "xgb_fpr_on_test_benign": fpr_benign,
        "interpretation": (
            "XGB directly flags held-out class rows" if recall_novel > 0.5
            else "XGB does NOT directly flag held-out class rows"
        ),
    }
    logger.info(
        "  S3 XGB direct: recall(novel)=%.4f  fpr(benign)=%.4f",
        recall_novel, fpr_benign,
    )

    # ── Sanity 4: OOF reference distribution ──
    sanity4: dict = {"skipped": skip_oof}
    if not skip_oof:
        t_oof = time.perf_counter()
        oof_p_xgb_train_benign = _oof_p_xgb_benign(X_train, y_train, seed=seed)
        elapsed_oof = time.perf_counter() - t_oof

        # Original baseline: μ/σ from val benigns.
        mu_val, sd_val = float(np.mean(p_xgb_val_benign)), float(np.std(p_xgb_val_benign) or 1.0)
        auc_val_ref = _z_score_auc(p_xgb_test, y_test, mu_benign=mu_val, sigma_benign=sd_val)

        # OOF reference: μ/σ from train benigns via cross_val_predict.
        mu_oof = float(np.mean(oof_p_xgb_train_benign))
        sd_oof = float(np.std(oof_p_xgb_train_benign) or 1.0)
        auc_oof_ref = _z_score_auc(p_xgb_test, y_test, mu_benign=mu_oof, sigma_benign=sd_oof)

        delta = auc_oof_ref - auc_val_ref
        sanity4 = {
            "val_reference":      {"mu": mu_val, "sigma": sd_val, "auc_z_score": auc_val_ref},
            "oof_reference":      {"mu": mu_oof, "sigma": sd_oof, "auc_z_score": auc_oof_ref,
                                   "n_oof_benigns": int(len(oof_p_xgb_train_benign))},
            "delta_oof_minus_val": float(delta),
            "elapsed_seconds": round(elapsed_oof, 1),
        }
        logger.info(
            "  S4 z-score AUC: val-ref=%.4f → oof-ref=%.4f (Δ=%+.4f)",
            auc_val_ref, auc_oof_ref, delta,
        )

    return {
        "holdout_class": holdout_class,
        "sanity1_distribution_overlap": sanity1,
        "sanity2_split_disjointness":   sanity2,
        "sanity3_xgb_direct_performance": sanity3,
        "sanity4_oof_reference":         sanity4,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset", choices=sorted(_DATASETS.keys()), default="medsec25",
        help="Which dataset to sanity-check (default: medsec25 — the surprising result).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample-cap", type=int, default=30000,
        help="Stratified row-count cap on the per-fold XGB training set.",
    )
    parser.add_argument(
        "--skip-oof", action="store_true",
        help="Skip Sanity 4 (OOF cross-check) — saves ~3x XGB training time per fold.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    t0 = time.perf_counter()

    cfg = _DATASETS[args.dataset]
    logger.info("Sanity check on: %s", cfg.name)

    train_df = _load_split_path(cfg.train_parquet)
    val_df = _load_split_path(cfg.val_parquet)
    test_df = _load_split_path(cfg.test_parquet)

    attack_cats = sorted(
        c for c in train_df["Attack Category"].unique() if c != cfg.benign_category
    )
    logger.info("Attack categories: %s", attack_cats)

    out_dir = OUTPUTS_DIR / args.dataset
    fold_results = []
    for holdout in attack_cats:
        fold_results.append(
            _run_fold_sanity(
                holdout, train_df, val_df, test_df,
                seed=args.seed,
                benign_category=cfg.benign_category,
                sample_cap=args.sample_cap,
                dataset_label=cfg.name,
                out_dir=out_dir,
                skip_oof=args.skip_oof,
            )
        )

    # Cross-fold summary.
    s1_ks = [f["sanity1_distribution_overlap"]["ks_statistic"] for f in fold_results]
    s2_ok = all(f["sanity2_split_disjointness"]["ok"] for f in fold_results)
    s3_recalls = [f["sanity3_xgb_direct_performance"]["xgb_recall_on_holdout_class"] for f in fold_results]
    s3_fprs = [f["sanity3_xgb_direct_performance"]["xgb_fpr_on_test_benign"] for f in fold_results]
    s4_present = not args.skip_oof
    if s4_present:
        s4_deltas = [f["sanity4_oof_reference"]["delta_oof_minus_val"] for f in fold_results]
    else:
        s4_deltas = []

    summary = {
        "n_folds": len(fold_results),
        "sanity1_ks_mean": float(np.mean(s1_ks)),
        "sanity2_all_splits_disjoint": s2_ok,
        "sanity3_xgb_recall_mean": float(np.mean(s3_recalls)),
        "sanity3_xgb_fpr_mean":    float(np.mean(s3_fprs)),
        "sanity4_oof_vs_val_auc_delta_mean": float(np.mean(s4_deltas)) if s4_deltas else None,
        "verdict": _verdict(s1_ks, s2_ok, s3_recalls, s4_deltas),
    }

    payload = {
        "sanity_check": {
            "dataset": cfg.name,
            "seed": args.seed,
            "sample_cap": args.sample_cap,
            "wall_time_seconds": round(time.perf_counter() - t0, 1),
        },
        "results": fold_results,
        "summary": summary,
    }
    out_path = PROJECT_ROOT / "results" / "reports" / f"dae_ablation_sanity_{args.dataset}.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    print()
    print("─" * 72)
    print(f"DAE ablation sanity checks — {cfg.name}")
    print("─" * 72)
    print(_markdown_table(fold_results))
    print()
    for line in summary["verdict"]:
        print(f"  • {line}")
    print()
    print(f"Histograms: {out_dir.relative_to(PROJECT_ROOT)}/")
    print(f"Saved: {out_path.relative_to(PROJECT_ROOT)}")
    return 0


def _markdown_table(folds: list[dict]) -> str:
    lines = []
    lines.append("| Holdout | KS(benign,novel) | Splits disjoint? | XGB recall(novel) | XGB FPR(benign) | z-AUC val-ref | z-AUC oof-ref |")
    lines.append("|---|---:|---|---:|---:|---:|---:|")
    for f in folds:
        s1 = f["sanity1_distribution_overlap"]
        s2 = f["sanity2_split_disjointness"]
        s3 = f["sanity3_xgb_direct_performance"]
        s4 = f["sanity4_oof_reference"]
        oof_auc = s4.get("oof_reference", {}).get("auc_z_score") if not s4.get("skipped") else None
        val_auc = s4.get("val_reference", {}).get("auc_z_score") if not s4.get("skipped") else None
        lines.append(
            f"| {f['holdout_class']} "
            f"| {s1['ks_statistic']:.4f} "
            f"| {'✓' if s2['ok'] else '✗ overlap=' + str(s2['row_overlap_count'])} "
            f"| {s3['xgb_recall_on_holdout_class']:.4f} "
            f"| {s3['xgb_fpr_on_test_benign']:.4f} "
            f"| {(f'{val_auc:.4f}' if val_auc is not None else '—')} "
            f"| {(f'{oof_auc:.4f}' if oof_auc is not None else '—')} |"
        )
    return "\n".join(lines)


def _verdict(
    s1_ks: list[float],
    s2_ok: bool,
    s3_recalls: list[float],
    s4_deltas: list[float],
) -> list[str]:
    out: list[str] = []

    if not s2_ok:
        out.append(
            "❌ S2 (split disjointness) FAILED — there is row-level overlap between val "
            "benigns and test benigns. The z-score baseline result is not trustworthy "
            "until that contamination is fixed."
        )
        return out

    out.append("✓ S2: val/test splits are disjoint at the row level.")

    mean_ks = float(np.mean(s1_ks))
    if mean_ks > 0.6:
        out.append(
            f"✓ S1: P_xgb distributions are clearly separated (mean KS={mean_ks:.3f}). "
            "The z-score baseline has a real signal to pick up."
        )
    elif mean_ks > 0.3:
        out.append(
            f"⚠ S1: distributions overlap moderately (mean KS={mean_ks:.3f}). "
            "Z-score AUC at this KS is consistent with shape-of-tail effects."
        )
    else:
        out.append(
            f"❌ S1: distributions overlap heavily (mean KS={mean_ks:.3f}) — "
            "a z-score AUC ≈ 1.0 against this tiny shift is suspect."
        )

    mean_recall = float(np.mean(s3_recalls))
    if mean_recall > 0.5:
        out.append(
            f"✓ S3: XGB directly flags held-out rows as attack (mean recall={mean_recall:.3f}). "
            "The z-score AUC is a downstream consequence of ordinary supervised "
            "generalisation across attack classes — not unusual leakage."
        )
    elif mean_recall < 0.1:
        out.append(
            f"⚠ S3: XGB does NOT flag held-out rows as attack (mean recall={mean_recall:.3f}). "
            "If the z-score AUC is still high, the signal is genuine *distribution shift* "
            "below the decision threshold — a finding, not a leak."
        )
    else:
        out.append(
            f"~ S3: XGB partial-flags held-out rows (mean recall={mean_recall:.3f}). "
            "Distribution shift is the dominant z-score signal."
        )

    if s4_deltas:
        mean_delta = float(np.mean(s4_deltas))
        if abs(mean_delta) < 0.01:
            out.append(
                f"✓ S4: OOF reference vs val reference yields nearly identical z-score AUC "
                f"(Δ={mean_delta:+.4f}). The val-set reference distribution was not biased."
            )
        elif abs(mean_delta) < 0.05:
            out.append(
                f"~ S4: OOF reference shifts z-score AUC by Δ={mean_delta:+.4f}. "
                "Small but non-zero — val-set reference has a mild bias; result still stands."
            )
        else:
            out.append(
                f"❌ S4: OOF reference shifts z-score AUC by Δ={mean_delta:+.4f}. "
                "Val-set reference was materially biased; rerun the ablation with OOF probas."
            )
    else:
        out.append("⏭ S4 skipped (--skip-oof).")

    return out


if __name__ == "__main__":
    raise SystemExit(main())
