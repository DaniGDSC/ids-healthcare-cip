#!/usr/bin/env python3
"""Online-capable per-alert explanation pipeline — CLI entry point.

This file is now a thin CLI + backward-compatibility shim. The actual
implementation lives in:

  - ``module4_explanations.online_explainer`` — AlertExplainer class
  - ``module4_explanations.batch_sim``        — batch sim + latency stats
  - ``module4_explanations.plotting``         — latency plots
  - ``module4_explanations.config``           — CLINICIAN_TEMPLATES
  - ``module4_explanations.feature_groups``   — _FEATURE_GROUPS

Existing imports like ``from module4_explanations.module4_online_explainer
import AlertExplainer`` continue to work via the re-exports below.

Demonstrates that per-alert explanations can be generated within the
real-time SLA (<150ms) using TreeSHAP + DAE decomposition + NLG.

Usage:
    python -m module4_explanations.module4_online_explainer
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# ── Backward-compat re-exports ────────────────────────────────────────
from module4_explanations.batch_sim import (  # noqa: E402,F401
    compute_latency_stats,
    run_batch_simulation,
)
from module4_explanations.config import (  # noqa: E402,F401
    BIOMETRIC_FEATURES,
    CLINICIAN_TEMPLATES,
    TRACK_A_MODELS as TRACK_A,
)
from module4_explanations.feature_groups import (  # noqa: E402,F401
    _FEATURE_GROUPS,
    _feature_to_narrative,
)
from module4_explanations.io import (  # noqa: E402,F401
    CHARTS_DIR,
    OUTPUT_DIR,
    PROJECT_ROOT,
    NumpyJSONEncoder,
)
from module4_explanations.online_explainer import AlertExplainer  # noqa: E402,F401
from module4_explanations.plotting import (  # noqa: E402,F401
    plot_latency_cdf,
    plot_latency_component_breakdown as plot_component_breakdown,
    plot_latency_distribution,
)

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    sep = "=" * 72
    logger.info(sep)
    logger.info("ONLINE-CAPABLE EXPLANATION PIPELINE — LATENCY PROFILING")
    logger.info(sep)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load test data
    df = pd.read_parquet(PROJECT_ROOT / "data/processed/test_phase1.parquet")
    drop_cols = ["Label", "Attack Category", "row_id", "device_class"]
    feat_names = [c for c in df.columns if c not in drop_cols]
    X_test = df[feat_names].values.astype(np.float32)

    # Load XGBoost predictions to identify alert samples
    xgb_preds = np.load(PROJECT_ROOT / "results/models/xgboost_test_predictions.npz")
    y_pred_xgb = xgb_preds["y_pred"]
    n_alerts = (y_pred_xgb == 1).sum()
    logger.info(
        "Test set: %d samples, %d XGBoost alerts to explain",
        len(X_test), n_alerts,
    )

    # Initialize explainer with feat_names at construction (Y10 fix).
    logger.info("Loading AlertExplainer (one-time startup)...")
    explainer = AlertExplainer(feat_names)

    # Warmup: single call to trigger any lazy compilation
    _ = explainer.explain(X_test[0])
    logger.info("Warmup complete")

    # Batch simulation
    logger.info("")
    logger.info("── Per-Alert Simulation ──")
    all_timings, sample_explanations = run_batch_simulation(
        explainer, X_test, y_pred_xgb, feat_names,
    )

    stats = compute_latency_stats(all_timings)
    full_timings = [t for t in all_timings if "treeshap_ms" in t]
    minimal_timings = [t for t in all_timings if "treeshap_ms" not in t]

    profile = {
        "n_alerts_total": len(all_timings),
        "n_full_explanations": len(full_timings),
        "n_minimal_explanations": len(minimal_timings),
        "startup_ms": explainer._startup_ms,
        "all_alerts": stats,
        "full_only": compute_latency_stats(full_timings) if full_timings else {},
        "minimal_only": (
            compute_latency_stats(minimal_timings) if minimal_timings else {}
        ),
    }

    # Canonical filenames consumed by module6_app's simulation panel.
    profile_path = OUTPUT_DIR / "online_latency_profile.json"
    profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    logger.info("Saved: %s", profile_path)

    # Legacy alias paths kept for any external tooling.
    legacy_profile_path = CHARTS_DIR / "latency_profile.json"
    legacy_profile_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    logger.info("Saved: %s (legacy alias)", legacy_profile_path)

    # Sample explanations — NumpyJSONEncoder handles np types (Y6 fix).
    samples_path = OUTPUT_DIR / "online_sample_explanations.json"
    samples_path.write_text(
        json.dumps(sample_explanations, indent=2, cls=NumpyJSONEncoder),
        encoding="utf-8",
    )
    logger.info("Saved: %s (%d examples)", samples_path, len(sample_explanations))

    legacy_samples_path = OUTPUT_DIR / "sample_explanations.json"
    legacy_samples_path.write_text(
        json.dumps(sample_explanations, indent=2, cls=NumpyJSONEncoder),
        encoding="utf-8",
    )
    logger.info("Saved: %s (legacy alias)", legacy_samples_path)

    # Latency plots — output_dir param (Y7 / N8 fix)
    plot_latency_distribution(all_timings, output_dir=CHARTS_DIR)
    plot_latency_cdf(all_timings, output_dir=CHARTS_DIR)
    if full_timings:
        plot_component_breakdown(
            compute_latency_stats(full_timings), output_dir=CHARTS_DIR,
        )

    logger.info("")
    logger.info(sep)
    logger.info("LATENCY PROFILING COMPLETE")
    logger.info(sep)
    logger.info(
        "  Alerts profiled : %d (%d full, %d minimal)",
        len(all_timings), len(full_timings), len(minimal_timings),
    )
    if "total_ms" in stats:
        logger.info(
            "  Total latency   : p50=%.1fms, p95=%.1fms, p99=%.1fms",
            stats["total_ms"]["p50"], stats["total_ms"]["p95"],
            stats["total_ms"]["p99"],
        )
    if full_timings:
        fs = compute_latency_stats(full_timings)
        if "total_ms" in fs:
            logger.info(
                "  Full explain    : p50=%.1fms, p95=%.1fms, p99=%.1fms",
                fs["total_ms"]["p50"], fs["total_ms"]["p95"],
                fs["total_ms"]["p99"],
            )
        if "treeshap_ms" in fs:
            logger.info(
                "  TreeSHAP only   : p50=%.1fms, p95=%.1fms",
                fs["treeshap_ms"]["p50"], fs["treeshap_ms"]["p95"],
            )
    logger.info(
        "  SLA feasibility : <150ms per alert = %s",
        "PASS" if stats.get("total_ms", {}).get("p95", 999) < 150 else "FAIL",
    )
    logger.info("  Output          : %s", OUTPUT_DIR)
    logger.info(sep)


if __name__ == "__main__":
    main()
