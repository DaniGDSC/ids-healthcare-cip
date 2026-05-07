"""DAE ablation study: cascade-input contribution under leave-one-class-out.

Datasets
--------
* ``ehms`` (default) — 2 attack classes (Spoofing, Data Alteration). 25
  raw features. N=2 folds.
* ``medsec25`` — 4 attack classes (Reconnaissance, Initial access,
  Exfiltration, Lateral movement). 69 raw features. N=4 folds.

Setup
-----
For each held-out class H we run **clean LOO** (Option A in the plan):

  1. Train an XGB-style classifier on ``benign + (attacks ∖ {H})`` so
     H is genuinely unseen by Track A.
  2. Compute calibrated ``P_xgb`` on val benigns + the test rows that
     remain after the held-out filter (benigns + H rows).
  3. For each of three DAE configs, train a fresh DAE on benign-only
     val rows under that config's input shape:

       - DAE-raw          25-dim raw features
       - DAE-cascade      26-dim ``[25 raw || P_xgb_val]`` (v5 contract)
       - DAE-probas-only  1-dim ``[P_xgb_val]``

  4. Score reconstruction error on the test set (benigns + H rows) and
     report AUC for the *novel-vs-benign* binary discrimination.

Output
------
``results/reports/dae_ablation_loo.yaml`` and a Markdown table to
stdout.

Caveats
-------
* **N=2 folds** because EHMS only has two attack classes. Generalisation
  claims from this study are weaker than they would be on MedSec-25
  (5 classes); we report this in the YAML.
* **DAE-probas-only is degenerate** — a 1-dim autoencoder can't compress
  below 1, so the DAEDetector class would refuse the architecture. We
  substitute a benign-mean z-score as the reconstruction-error proxy
  for that config and label it ``recon_error_proxy: z_score_abs`` in
  the output. It's a sanity baseline, not a serious model.
* **XGB hyperparameters are fixed defaults** (sklearn
  ``GradientBoostingClassifier`` with sigmoid Platt calibration). This
  is a deliberate simplification for ablation reproducibility — the
  per-fold XGB is re-trained from scratch but not RandomizedSearch'd,
  to keep total wall time under a minute.
* **DAE training is stochastic.** ``random_state=42`` everywhere.

Run::

    python -m analysis.dae_ablation_loo                 # EHMS-2020 (default)
    python -m analysis.dae_ablation_loo --dataset medsec25
    python -m analysis.dae_ablation_loo --dataset medsec25 --sample-cap 50000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Quiet TF before importing the DAE module.
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from module2_detection._features import drop_non_feature_cols  # noqa: E402
from module2_detection.models.DAE import DAEDetector  # noqa: E402

logger = logging.getLogger(__name__)

PROCESSED = PROJECT_ROOT / "data" / "processed"


# ── Dataset configuration ─────────────────────────────────────────────


class DatasetConfig:
    """Per-dataset paths + benign-class label and "no attack" sentinel."""

    def __init__(
        self,
        name: str,
        train_parquet: Path,
        val_parquet: Path,
        test_parquet: Path,
        n_raw_features_expected: int,
        benign_category: str,
        out_path: Path,
    ) -> None:
        self.name = name
        self.train_parquet = train_parquet
        self.val_parquet = val_parquet
        self.test_parquet = test_parquet
        self.n_raw_features_expected = n_raw_features_expected
        self.benign_category = benign_category
        self.out_path = out_path


_DATASETS: dict[str, DatasetConfig] = {
    "ehms": DatasetConfig(
        name="EHMS-2020",
        train_parquet=PROCESSED / "train_phase1.parquet",
        val_parquet=PROCESSED / "val_phase1.parquet",
        test_parquet=PROCESSED / "test_phase1.parquet",
        n_raw_features_expected=25,
        benign_category="normal",
        out_path=PROJECT_ROOT / "results" / "reports" / "dae_ablation_loo.yaml",
    ),
    "medsec25": DatasetConfig(
        name="MedSec-25",
        train_parquet=PROCESSED / "medsec25" / "train.parquet",
        val_parquet=PROCESSED / "medsec25" / "val.parquet",
        test_parquet=PROCESSED / "medsec25" / "test.parquet",
        n_raw_features_expected=69,
        benign_category="Benign",
        out_path=PROJECT_ROOT / "results" / "reports" / "dae_ablation_loo_medsec25.yaml",
    ),
}


# ── Helpers ────────────────────────────────────────────────────────────


def _load_split_path(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run Phase 1 preprocessing first.")
    return pd.read_parquet(path)


def _stratified_subsample(
    df: pd.DataFrame, *, sample_cap: int | None, random_state: int,
) -> pd.DataFrame:
    """Stratified subsample by ``Attack Category`` capped at ``sample_cap``.

    No-op when ``sample_cap`` is None or already >= row count. Used to
    keep MedSec-25 XGB training under a few minutes per fold without
    distorting the per-class proportions.
    """
    if sample_cap is None or len(df) <= sample_cap:
        return df
    rng = np.random.RandomState(random_state)
    grouped = df.groupby("Attack Category", group_keys=False)
    fraction = sample_cap / len(df)
    sampled = grouped.apply(
        lambda g: g.sample(
            n=max(1, int(round(len(g) * fraction))),
            random_state=int(rng.randint(0, 2**31 - 1)),
        )
    ).reset_index(drop=True)
    return sampled


def _features_and_labels(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, y_binary, attack_category)."""
    y = df["Label"].astype(int).values
    cat = df["Attack Category"].astype(str).values
    X = drop_non_feature_cols(df).values.astype(np.float32)
    return X, y, cat


def _train_xgb(X: np.ndarray, y: np.ndarray, *, seed: int) -> CalibratedClassifierCV:
    """Train a GradientBoosting + Platt-calibrated XGB-style classifier.

    Deliberately fixed-hyperparam (no RandomizedSearchCV) for ablation
    reproducibility and runtime — the production XGB tuning is in
    ``module2_detection/tuning/run_xgboost.py``.
    """
    base = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=seed,
    )
    cal = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
    cal.fit(X, y)
    return cal


def _predict_p_attack(model: CalibratedClassifierCV, X: np.ndarray) -> np.ndarray:
    """P(attack) from a binary calibrator. Pos label = 1."""
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    pos_idx = classes.index(1) if 1 in classes else 0
    return proba[:, pos_idx].astype(np.float32)


def _config_input(
    X_raw: np.ndarray,
    p_xgb: np.ndarray,
    config: str,
) -> np.ndarray:
    """Build the per-config DAE input vector."""
    if config == "DAE-raw":
        return X_raw
    if config == "DAE-cascade":
        return np.column_stack([X_raw, p_xgb.reshape(-1, 1)])
    if config == "DAE-probas-only":
        return p_xgb.reshape(-1, 1)
    raise ValueError(f"Unknown config: {config}")


def _train_and_score_one_config(
    config: str,
    X_train_benign: np.ndarray,
    p_xgb_train_benign: np.ndarray,
    X_test: np.ndarray,
    p_xgb_test: np.ndarray,
    y_test: np.ndarray,
    *,
    seed: int,
) -> dict:
    """Train one DAE config on benign train, score test, return per-cell metrics.

    The 1-dim ``DAE-probas-only`` configuration uses a benign-mean
    z-score in lieu of the autoencoder, because DAEDetector enforces
    bottleneck < n_features (impossible at n_features=1). Documented
    in the result dict as ``recon_error_proxy``.
    """
    train_inp = _config_input(X_train_benign, p_xgb_train_benign, config)
    test_inp = _config_input(X_test, p_xgb_test, config)

    if train_inp.shape[1] == 1:
        # 1-D fallback: |z-score| against benign mean. Same intent as a
        # 1-D autoencoder reconstruction error (which would degenerate
        # to a translation of the same quantity).
        mean_b = float(np.mean(train_inp))
        std_b = float(np.std(train_inp) or 1.0)
        recon_err = np.abs(test_inp.ravel() - mean_b) / std_b
        proxy = "z_score_abs"
    else:
        det = DAEDetector(
            encoding_dims=[24, 12, 24],
            noise_rate=0.1,
            learning_rate=1e-4,
            threshold_percentile=99.0,
            clip_percentile=1.0,
            epochs=100,
            batch_size=256,
            random_state=seed,
        )
        det.fit(train_inp)
        recon_err = det.reconstruction_error(test_inp)
        proxy = None

    # AUC: positive = novel attack (y=1), negative = benign (y=0).
    if len(np.unique(y_test)) < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(y_test, recon_err))

    benign_mask = y_test == 0
    novel_mask = y_test == 1
    benign_err = recon_err[benign_mask]
    novel_err = recon_err[novel_mask]
    return {
        "config": config,
        "input_dim": int(train_inp.shape[1]),
        "n_train_benign": int(train_inp.shape[0]),
        "n_test_benign": int(benign_mask.sum()),
        "n_test_novel": int(novel_mask.sum()),
        "recon_error_mean_benign":   float(np.mean(benign_err))   if benign_err.size else float("nan"),
        "recon_error_median_benign": float(np.median(benign_err)) if benign_err.size else float("nan"),
        "recon_error_mean_novel":    float(np.mean(novel_err))    if novel_err.size  else float("nan"),
        "recon_error_median_novel":  float(np.median(novel_err))  if novel_err.size  else float("nan"),
        "auc_benign_vs_novel": auc,
        "recon_error_proxy": proxy,
    }


def _run_fold(
    holdout_class: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    seed: int,
    benign_category: str,
    sample_cap: int | None = None,
) -> dict:
    """Run one LOO fold across all 3 DAE configs."""
    logger.info("─" * 72)
    logger.info("Fold: holdout=%s", holdout_class)
    logger.info("─" * 72)

    # 1. Split: drop the held-out attack class from XGB training and DAE training,
    #    keep it in the test set.
    train_filtered = train_df[train_df["Attack Category"] != holdout_class].copy()
    val_filtered = val_df[val_df["Attack Category"] != holdout_class].copy()
    test_keep = test_df[test_df["Attack Category"].isin([benign_category, holdout_class])].copy()
    if sample_cap is not None:
        before = len(train_filtered)
        train_filtered = _stratified_subsample(
            train_filtered, sample_cap=sample_cap, random_state=seed,
        )
        if len(train_filtered) < before:
            logger.info(
                "Subsampled training rows %d → %d (cap=%d)",
                before, len(train_filtered), sample_cap,
            )

    X_train, y_train, _ = _features_and_labels(train_filtered)
    X_val, y_val, _ = _features_and_labels(val_filtered)
    X_test, y_test, _ = _features_and_labels(test_keep)

    logger.info(
        "Train: %d rows (benign=%d, non-H attacks=%d)",
        len(y_train), int((y_train == 0).sum()), int((y_train == 1).sum()),
    )
    logger.info("Val (benign-only used for DAE training): %d benigns", int((y_val == 0).sum()))
    logger.info(
        "Test rows kept: %d (benign=%d, novel=%d)",
        len(y_test), int((y_test == 0).sum()), int((y_test == 1).sum()),
    )

    # 2. Train XGB on (benign + non-H attacks).
    t_xgb = time.perf_counter()
    xgb = _train_xgb(X_train, y_train, seed=seed)
    logger.info("XGB trained in %.1fs", time.perf_counter() - t_xgb)

    # 3. Compute calibrated P_xgb on val benigns + test rows.
    val_benign_mask = y_val == 0
    p_xgb_val_benign = _predict_p_attack(xgb, X_val[val_benign_mask])
    p_xgb_test = _predict_p_attack(xgb, X_test)
    logger.info(
        "P_xgb (val benigns): mean=%.4f, std=%.4f",
        float(np.mean(p_xgb_val_benign)), float(np.std(p_xgb_val_benign)),
    )

    # 4. For each DAE config, train and score.
    config_results = []
    for config in ("DAE-raw", "DAE-cascade", "DAE-probas-only"):
        t_cfg = time.perf_counter()
        out = _train_and_score_one_config(
            config=config,
            X_train_benign=X_val[val_benign_mask],
            p_xgb_train_benign=p_xgb_val_benign,
            X_test=X_test,
            p_xgb_test=p_xgb_test,
            y_test=y_test,
            seed=seed,
        )
        out["wall_time_seconds"] = round(time.perf_counter() - t_cfg, 1)
        logger.info(
            "  %-18s input=%d AUC=%.4f recon(benign)=%.5f recon(novel)=%.5f (%.1fs)",
            out["config"], out["input_dim"], out["auc_benign_vs_novel"],
            out["recon_error_mean_benign"], out["recon_error_mean_novel"],
            out["wall_time_seconds"],
        )
        config_results.append(out)

    return {
        "holdout_class": holdout_class,
        "n_train": int(len(y_train)),
        "n_val_benign_for_dae": int(val_benign_mask.sum()),
        "n_test_benign": int((y_test == 0).sum()),
        "n_test_novel": int((y_test == 1).sum()),
        "p_xgb_val_benign_mean": float(np.mean(p_xgb_val_benign)),
        "p_xgb_val_benign_std": float(np.std(p_xgb_val_benign)),
        "config_results": config_results,
    }


def _markdown_table(results: list[dict]) -> str:
    """Render results as a markdown table for stdout."""
    lines = []
    lines.append("| Fold (holdout) | Config | Input dim | AUC | Recon(benign) | Recon(novel) |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for fold in results:
        h = fold["holdout_class"]
        for cell in fold["config_results"]:
            lines.append(
                f"| {h} | {cell['config']} | {cell['input_dim']} | "
                f"{cell['auc_benign_vs_novel']:.4f} | "
                f"{cell['recon_error_mean_benign']:.5f} | "
                f"{cell['recon_error_mean_novel']:.5f} |"
            )
    return "\n".join(lines)


def _summary(results: list[dict]) -> dict:
    by_config: dict[str, list[float]] = {}
    by_fold_aucs: dict[str, dict[str, float]] = {}
    for fold in results:
        h = fold["holdout_class"]
        by_fold_aucs[h] = {}
        for cell in fold["config_results"]:
            by_config.setdefault(cell["config"], []).append(cell["auc_benign_vs_novel"])
            by_fold_aucs[h][cell["config"]] = cell["auc_benign_vs_novel"]
    mean_aucs = {k: float(np.mean(v)) for k, v in by_config.items()}
    ranking = sorted(
        ({"config": k, "mean_auc": v} for k, v in mean_aucs.items()),
        key=lambda r: r["mean_auc"], reverse=True,
    )
    raw_baseline = mean_aucs.get("DAE-raw", float("nan"))
    for entry in ranking:
        entry["delta_vs_raw"] = float(entry["mean_auc"] - raw_baseline)
    best_per_fold = {}
    for fold in results:
        best = max(fold["config_results"], key=lambda c: c["auc_benign_vs_novel"])
        best_per_fold[fold["holdout_class"]] = best["config"]

    # Honest interpretation of the result, generated from the numbers.
    cascade_delta = mean_aucs.get("DAE-cascade", 0.0) - raw_baseline
    interpretation_lines: list[str] = []

    # Per-fold reading.
    for fold in results:
        h = fold["holdout_class"]
        raw_auc = by_fold_aucs[h].get("DAE-raw", float("nan"))
        cas_auc = by_fold_aucs[h].get("DAE-cascade", float("nan"))
        if raw_auc >= 0.95:
            interpretation_lines.append(
                f"On {h!r} the raw 25-dim DAE already separates novel-from-benign "
                f"at AUC={raw_auc:.3f}; the cascade adds nothing measurable here."
            )
        elif raw_auc < 0.55:
            verdict_cascade = (
                f"the cascade lifts AUC to {cas_auc:.3f} (Δ={cas_auc - raw_auc:+.3f})"
                if cas_auc > raw_auc
                else "the cascade does not rescue it"
            )
            interpretation_lines.append(
                f"On {h!r} a benign-only DAE collapses to AUC={raw_auc:.3f} "
                f"(near random); {verdict_cascade}. The unsupervised path cannot "
                f"detect this attack class — Track A (a supervised classifier "
                f"that has seen this class) is required."
            )
        else:
            interpretation_lines.append(
                f"On {h!r} raw DAE is partial (AUC={raw_auc:.3f}); cascade "
                f"AUC={cas_auc:.3f} (Δ={cas_auc - raw_auc:+.3f})."
            )

    # Whole-experiment reading.
    if cascade_delta >= 0.05:
        verdict = "MEANINGFUL"
    elif cascade_delta >= 0.01:
        verdict = "MARGINAL"
    elif cascade_delta >= -0.01:
        verdict = "INDISTINGUISHABLE"
    else:
        verdict = "REGRESSION"

    return {
        "ranking_by_mean_auc": ranking,
        "best_per_fold": best_per_fold,
        "cascade_vs_raw_mean_auc_delta": float(cascade_delta),
        "cascade_vs_raw_verdict": verdict,
        "per_fold_interpretation": interpretation_lines,
        "conclusion": (
            f"Cascade input (raw + P_xgb) vs raw alone: mean Δ AUC = "
            f"{cascade_delta:+.4f} ({verdict}). "
            "DAE-probas-only is a 1-D z-score baseline and is reported only "
            "to show what falls out without raw features."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset", choices=sorted(_DATASETS.keys()), default="ehms",
        help="Which dataset to run LOO on (default: ehms).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample-cap", type=int, default=None,
        help="Stratified row-count cap on the per-fold XGB training set "
             "(default: no cap). Useful for MedSec-25 to bound wall time.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    t0 = time.perf_counter()

    cfg = _DATASETS[args.dataset]
    logger.info("Dataset: %s (%s)", args.dataset, cfg.name)
    train_df = _load_split_path(cfg.train_parquet)
    val_df = _load_split_path(cfg.val_parquet)
    test_df = _load_split_path(cfg.test_parquet)

    # Sanity-check feature dim — surfaces silent schema drift early.
    n_feat = len([c for c in test_df.columns if c not in ("Label", "Attack Category")])
    if n_feat != cfg.n_raw_features_expected:
        logger.warning(
            "%s expected %d raw features but parquet carries %d. "
            "Adapting on-the-fly.",
            cfg.name, cfg.n_raw_features_expected, n_feat,
        )

    attack_cats = sorted(
        c for c in train_df["Attack Category"].unique() if c != cfg.benign_category
    )
    logger.info("Attack categories detected: %s", attack_cats)
    if len(attack_cats) < 2:
        logger.error("Need ≥ 2 attack classes for LOO; got %d", len(attack_cats))
        return 1

    fold_results = []
    for holdout in attack_cats:
        fold_results.append(
            _run_fold(
                holdout, train_df, val_df, test_df,
                seed=args.seed,
                benign_category=cfg.benign_category,
                sample_cap=args.sample_cap,
            )
        )

    summary = _summary(fold_results)

    raw_dim = n_feat
    cascade_dim = n_feat + 1
    payload = {
        "ablation": {
            "dataset": cfg.name,
            "fold_protocol": "leave-one-class-out",
            "experiment_purity": "A",  # XGB retrained per fold
            "n_folds": len(fold_results),
            "seed": args.seed,
            "sample_cap": args.sample_cap,
            "wall_time_seconds": round(time.perf_counter() - t0, 1),
            "configs": [
                {"name": "DAE-raw", "input_dim": raw_dim,
                 "input_features": f"{raw_dim} raw network features"},
                {"name": "DAE-cascade", "input_dim": cascade_dim,
                 "input_features": f"{raw_dim} raw || P_xgb_val (v5 contract)"},
                {
                    "name": "DAE-probas-only",
                    "input_dim": 1,
                    "input_features": "P_xgb_val",
                    "note": "1-D DAE is degenerate (bottleneck must be < n_features); "
                            "we substitute |z-score| vs benign mean as recon-error proxy.",
                },
            ],
            "caveats": [
                f"{cfg.name} has {len(attack_cats)} attack classes; N={len(fold_results)} folds.",
                "XGB uses fixed default hyperparameters per fold (no RandomizedSearchCV) "
                "to keep ablation runtime bounded.",
                "DAE-probas-only is a 1-D z-score baseline, not a true autoencoder.",
            ],
        },
        "results": fold_results,
        "summary": summary,
    }

    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.out_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    print()
    print("─" * 72)
    print(f"DAE ablation — leave-one-class-out (Option A) — {cfg.name}")
    print("─" * 72)
    print(_markdown_table(fold_results))
    print()
    print("Mean AUC ranking:")
    for entry in summary["ranking_by_mean_auc"]:
        print(f"  {entry['config']:<18} mean_auc={entry['mean_auc']:.4f}  Δ vs raw={entry['delta_vs_raw']:+.4f}")
    print()
    print("Per-fold interpretation:")
    for line in summary["per_fold_interpretation"]:
        print(f"  - {line}")
    print()
    print(f"Best per fold: {summary['best_per_fold']}")
    print(f"Total wall time: {payload['ablation']['wall_time_seconds']}s")
    print(f"Saved: {cfg.out_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
