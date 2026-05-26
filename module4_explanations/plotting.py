"""Plot generators for Module 4.

All plot functions accept an explicit ``output_dir`` parameter so tests
can redirect outputs to ``tmp_path``. Production callers (the offline
CLI) pass ``CHARTS_DIR`` from ``io.py``.

Parallel rendering for waterfall + DAE breakdown plots uses
``ProcessPoolExecutor`` — matplotlib is not thread-safe but is
process-safe. Each worker imports its own matplotlib context so there
is no shared state.
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from .config import BIOMETRIC_FEATURES, TOP_K_FEATURES, TOP_N_WATERFALL  # noqa: E402

logger = logging.getLogger(__name__)


def _feat_color(name: str) -> str:
    return "#3274A1" if name in BIOMETRIC_FEATURES else "#C44E52"


# ── Subprocess worker functions for parallel rendering ──────────────


def _render_waterfall_worker(args: tuple) -> None:
    """Render one SHAP waterfall PNG — runs in a subprocess."""
    import shap as _shap
    import matplotlib as _mpl

    _mpl.use("Agg")
    import matplotlib.pyplot as _plt

    sv_row, expected, X_row, feat_names, out_path, model_name, idx, proba = args
    explanation = _shap.Explanation(
        values=sv_row, base_values=expected, data=X_row, feature_names=feat_names,
    )
    fig = _plt.figure(figsize=(10, 7))
    _shap.plots.waterfall(explanation, show=False)
    _plt.title(f"{model_name} — Sample {idx} (proba={proba:.3f})")
    _plt.tight_layout()
    _plt.savefig(out_path, dpi=150)
    _plt.close(fig)


def _render_dae_breakdown_worker(args: tuple) -> None:
    """Render one DAE breakdown PNG — runs in a subprocess."""
    from matplotlib.patches import Patch as _Patch
    import matplotlib as _mpl

    _mpl.use("Agg")
    import matplotlib.pyplot as _plt

    errs, feat_names, recon_error, out_path, idx, bio_features = args
    sorted_i = errs.argsort()[::-1][:TOP_K_FEATURES]
    names_plot = [feat_names[i] for i in sorted_i][::-1]
    values_plot = [float(errs[i]) for i in sorted_i][::-1]
    colors = ["#3274A1" if n in bio_features else "#C44E52" for n in names_plot]

    fig, ax = _plt.subplots(figsize=(10, 6))
    ax.barh(names_plot, values_plot, color=colors)
    ax.set_xlabel("Weighted Reconstruction Error")
    ax.set_title(f"DAE — Sample {idx} (error={recon_error:.6f})")
    ax.legend(
        handles=[
            _Patch(facecolor="#C44E52", label="Network"),
            _Patch(facecolor="#3274A1", label="Biometric"),
        ],
        loc="lower right",
    )
    _plt.tight_layout()
    _plt.savefig(out_path, dpi=150)
    _plt.close(fig)


# ── Offline plots ───────────────────────────────────────────────────


def plot_global_importance_bar(
    model_name: str,
    importance: list,
    *,
    output_dir: Path,
    title_suffix: str = "mean |SHAP|",
    value_key: str = "mean_abs_shap",
) -> Path:
    """Horizontal bar chart of top-K feature importance."""
    top = importance[:TOP_K_FEATURES]
    names = [f["feature"] for f in top][::-1]
    values = [f[value_key] for f in top][::-1]
    colors = [_feat_color(n) for n in names]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(names, values, color=colors)
    ax.set_xlabel(title_suffix)
    ax.set_title(f"{model_name} — Global Feature Importance ({title_suffix})")
    ax.legend(
        handles=[
            Patch(facecolor="#C44E52", label="Network"),
            Patch(facecolor="#3274A1", label="Biometric"),
        ],
        loc="lower right",
    )
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"global_importance_{model_name}.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("  Chart: %s", out.name)
    return out


def plot_waterfalls(
    model_name: str,
    sv: np.ndarray,
    expected: float,
    X_test: np.ndarray,
    feat_names: list,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    *,
    output_dir: Path,
) -> list[Path]:
    """Waterfall plots for top-N highest-confidence attack predictions."""
    attack_idx = np.where(y_pred == 1)[0]
    if len(attack_idx) == 0:
        logger.info("  No attacks predicted by %s, skipping waterfalls", model_name)
        return []

    top_idx = attack_idx[np.argsort(y_proba[attack_idx])[-TOP_N_WATERFALL:]][::-1]
    output_dir.mkdir(parents=True, exist_ok=True)

    out_paths = [
        output_dir / f"waterfall_{model_name}_sample_{idx:04d}.png"
        for idx in top_idx
    ]
    render_args = [
        (
            sv[idx], expected, X_test[idx], feat_names,
            str(out_paths[i]), model_name, int(idx), float(y_proba[idx]),
        )
        for i, idx in enumerate(top_idx)
    ]

    n_workers = min(len(render_args), 4)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        list(pool.map(_render_waterfall_worker, render_args))

    logger.info(
        "  Charts: %d waterfall plots for %s (parallel)", len(top_idx), model_name,
    )
    return out_paths


def plot_dae_breakdowns(
    weighted_err: np.ndarray,
    feat_names: list,
    y_pred: np.ndarray,
    recon_errors: np.ndarray,
    *,
    output_dir: Path,
) -> list[Path]:
    """Bar chart of per-feature error for top-N DAE anomalies."""
    anomaly_idx = np.where(y_pred == 1)[0]
    if len(anomaly_idx) == 0:
        logger.info("  No DAE anomalies, skipping breakdown plots")
        return []

    top_idx = anomaly_idx[
        np.argsort(recon_errors[anomaly_idx])[-TOP_N_WATERFALL:]
    ][::-1]
    output_dir.mkdir(parents=True, exist_ok=True)
    bio_set = set(BIOMETRIC_FEATURES)

    out_paths = [
        output_dir / f"dae_error_breakdown_sample_{idx:04d}.png"
        for idx in top_idx
    ]
    render_args = [
        (
            weighted_err[idx], feat_names, float(recon_errors[idx]),
            str(out_paths[i]), int(idx), bio_set,
        )
        for i, idx in enumerate(top_idx)
    ]

    n_workers = min(len(render_args), 4)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        list(pool.map(_render_dae_breakdown_worker, render_args))

    logger.info("  Charts: %d DAE breakdown plots (parallel)", len(top_idx))
    return out_paths


def plot_beeswarm(
    model_name: str,
    sv: np.ndarray,
    X_test: np.ndarray,
    feat_names: list,
    *,
    output_dir: Path,
) -> Path:
    """SHAP beeswarm (summary) plot."""
    import shap
    explanation = shap.Explanation(
        values=sv, data=X_test, feature_names=feat_names,
    )
    fig = plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(explanation, show=False, max_display=TOP_K_FEATURES)
    plt.title(f"{model_name} — SHAP Beeswarm (attack class)")
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"beeswarm_{model_name}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Chart: %s", out.name)
    return out


def plot_force(
    model_name: str,
    sv: np.ndarray,
    expected: float,
    X_test: np.ndarray,
    feat_names: list,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    *,
    output_dir: Path,
) -> list[Path]:
    """SHAP force plots for top-N highest-confidence attack predictions."""
    import shap
    attack_idx = np.where(y_pred == 1)[0]
    if len(attack_idx) == 0:
        return []

    top_idx = attack_idx[np.argsort(y_proba[attack_idx])[-TOP_N_WATERFALL:]][::-1]
    output_dir.mkdir(parents=True, exist_ok=True)
    out_paths = []

    for idx in top_idx:
        explanation = shap.Explanation(
            values=sv[idx], base_values=expected, data=X_test[idx],
            feature_names=feat_names,
        )
        fig = plt.figure(figsize=(14, 3))
        shap.plots.force(explanation, show=False, matplotlib=True)
        plt.title(
            f"{model_name} — Sample {idx} (proba={y_proba[idx]:.3f})", y=1.05,
        )
        plt.tight_layout()
        out_path = output_dir / f"force_{model_name}_sample_{idx:04d}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        out_paths.append(out_path)

    logger.info("  Charts: %d force plots for %s", len(top_idx), model_name)
    return out_paths


def plot_per_category_importance(
    model_name: str,
    sv: np.ndarray,
    y_test: np.ndarray,
    attack_cats: np.ndarray | None,
    feat_names: list,
    *,
    output_dir: Path,
    json_dir: Path | None = None,
) -> dict:
    """Per-attack-category SHAP importance + bar charts."""
    from .io import OUTPUT_DIR, write_json_strict
    if attack_cats is None:
        return {}

    categories: dict[str, list] = {}
    cats_str = attack_cats.astype(str)
    unique_cats = sorted(c for c in np.unique(cats_str) if c and c != "normal")

    output_dir.mkdir(parents=True, exist_ok=True)
    for cat in unique_cats:
        mask = (cats_str == cat) & (y_test == 1)
        if mask.sum() == 0:
            continue

        mean_abs = np.mean(np.abs(sv[mask]), axis=0)
        ranked = sorted(zip(feat_names, mean_abs), key=lambda x: -x[1])
        importance = [
            {"rank": i + 1, "feature": n, "mean_abs_shap": round(float(v), 6)}
            for i, (n, v) in enumerate(ranked)
        ]
        categories[cat] = importance

        top = importance[:TOP_K_FEATURES]
        names = [f["feature"] for f in top][::-1]
        values = [f["mean_abs_shap"] for f in top][::-1]
        colors = [_feat_color(n) for n in names]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(names, values, color=colors)
        ax.set_xlabel("mean |SHAP|")
        ax.set_title(
            f"{model_name} — {cat} (n={mask.sum()}) Feature Importance"
        )
        ax.legend(
            handles=[
                Patch(facecolor="#C44E52", label="Network"),
                Patch(facecolor="#3274A1", label="Biometric"),
            ],
            loc="lower right",
        )
        plt.tight_layout()
        safe_cat = cat.replace(" ", "_").lower()
        plt.savefig(
            output_dir / f"importance_{model_name}_{safe_cat}.png", dpi=150,
        )
        plt.close(fig)

    if categories:
        json_target_dir = json_dir or OUTPUT_DIR
        path = json_target_dir / f"per_category_importance_{model_name}.json"
        write_json_strict(path, {"model": model_name, "categories": categories})
        logger.info(
            "  Per-category importance: %d categories for %s",
            len(categories), model_name,
        )
    return categories


def plot_dae_global_weights(
    feat_weights: np.ndarray,
    feat_names: list,
    *,
    output_dir: Path,
) -> Path:
    """Bar chart of DAE inverse-variance feature weights."""
    importance = sorted(
        [
            {"feature": n, "weight": float(w)}
            for n, w in zip(feat_names, feat_weights)
        ],
        key=lambda x: -x["weight"],
    )
    top = importance[:TOP_K_FEATURES]
    names = [f["feature"] for f in top][::-1]
    values = [f["weight"] for f in top][::-1]
    colors = [_feat_color(n) for n in names]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(names, values, color=colors)
    ax.set_xlabel("Feature Weight (inverse variance)")
    ax.set_title("DAE — Feature Monitoring Weights")
    ax.legend(
        handles=[
            Patch(facecolor="#C44E52", label="Network"),
            Patch(facecolor="#3274A1", label="Biometric"),
        ],
        loc="lower right",
    )
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "global_importance_dae.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("  Chart: %s", out.name)
    return out


# ── Online (latency) plots ──────────────────────────────────────────


def plot_latency_distribution(
    all_timings: list,
    *,
    output_dir: Path,
) -> Path:
    """Histogram of per-alert total latency."""
    totals = [t["total_ms"] for t in all_timings]
    p50, p95 = np.percentile(totals, [50, 95])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(totals, bins=50, edgecolor="black", alpha=0.7, color="#3274A1")
    ax.axvline(p95, color="red", linestyle="--", label=f"p95 = {p95:.1f}ms")
    ax.axvline(p50, color="orange", linestyle="--", label=f"p50 = {p50:.1f}ms")
    ax.set_xlabel("Total Latency (ms)")
    ax.set_ylabel("Count")
    ax.set_title("Per-Alert Explanation Latency Distribution")
    ax.legend()
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "latency_distribution.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("  Chart: %s", out.name)
    return out


def plot_latency_cdf(
    all_timings: list,
    *,
    output_dir: Path,
) -> Path:
    """CDF showing % of alerts below each latency threshold."""
    totals = np.sort([t["total_ms"] for t in all_timings])
    cdf = np.arange(1, len(totals) + 1) / len(totals) * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(totals, cdf, linewidth=2, color="#3274A1")

    sla_vals = np.array([50, 100, 150])
    sla_pcts = (totals[:, np.newaxis] < sla_vals).mean(axis=0) * 100
    sla_colors = ["green", "orange", "red"]
    for sla, pct, color in zip(sla_vals, sla_pcts, sla_colors):
        ax.axvline(
            sla, color=color, linestyle="--", alpha=0.7,
            label=f"{sla}ms SLA: {pct:.1f}% pass",
        )

    ax.set_xlabel("Total Latency (ms)")
    ax.set_ylabel("Cumulative % of Alerts")
    ax.set_title("Per-Alert Explanation Latency CDF")
    ax.legend()
    ax.set_ylim(0, 105)
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "latency_cdf.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("  Chart: %s", out.name)
    return out


def plot_latency_component_breakdown(
    stats: dict,
    *,
    output_dir: Path,
) -> Path:
    """Stacked bar showing latency breakdown by component."""
    components = [
        "predict_ms", "treeshap_ms", "dae_decompose_ms",
        "nlg_ms", "risk_decompose_ms",
    ]
    labels = ["Predict", "TreeSHAP", "DAE Decompose", "NLG", "Risk Decompose"]
    colors = ["#C44E52", "#3274A1", "#55A868", "#CCB974", "#8172B2"]

    vals = [stats[c]["p50"] if c in stats else 0 for c in components]

    fig, ax = plt.subplots(figsize=(10, 4))
    left = 0
    for label, val, color in zip(labels, vals, colors):
        ax.barh("Per-Alert", val, left=left, color=color,
                label=f"{label} ({val:.1f}ms)")
        left += val

    ax.set_xlabel("Latency (ms)")
    ax.set_title(
        f"Per-Alert Explanation — Component Breakdown (p50 total={sum(vals):.1f}ms)"
    )
    ax.legend(loc="lower right")
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "latency_component_breakdown.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("  Chart: %s", out.name)
    return out


__all__ = [
    "plot_global_importance_bar",
    "plot_waterfalls",
    "plot_dae_breakdowns",
    "plot_beeswarm",
    "plot_force",
    "plot_per_category_importance",
    "plot_dae_global_weights",
    "plot_latency_distribution",
    "plot_latency_cdf",
    "plot_latency_component_breakdown",
]
