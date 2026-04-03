"""Benchmark: GradientExplainer vs KernelSHAP.

Compares computation time and feature importance correlation
between the two SHAP methods on the same samples.

Usage::
    python -m src.phase5_explanation_engine.phase5.shap_benchmark
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def run_benchmark(n_samples: int = 10, n_background: int = 50) -> dict:
    """Run GradientExplainer vs KernelSHAP comparison.

    Args:
        n_samples: Number of test samples to explain.
        n_background: Background samples for both methods.

    Returns:
        Benchmark results dict.
    """
    import shap
    import tensorflow as tf
    import pandas as pd

    # Load model
    from src.production.inference_service import InferenceService
    svc = InferenceService(str(PROJECT_ROOT))
    svc.load()
    model = svc._model

    # Load test data
    test_df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "test_phase1.parquet")
    feature_names = [c for c in test_df.columns if c != "Label"]
    X_test = test_df[feature_names].values.astype(np.float32)

    # Prepare windowed samples
    from src.phase2_detection_engine.phase2.reshaper import DataReshaper
    reshaper = DataReshaper(timesteps=20, stride=5)
    X_w, y_w = reshaper.reshape(X_test, test_df["Label"].values)

    # Select samples + background
    background = X_w[:n_background]
    test_samples = X_w[n_background:n_background + n_samples]

    logger.info("Benchmark: %d samples, %d background, model=%d params",
                n_samples, n_background, model.count_params())

    results = {"n_samples": n_samples, "n_background": n_background}

    # ── GradientExplainer ──
    logger.info("Running GradientExplainer...")
    t0 = time.time()
    try:
        ge = shap.GradientExplainer(model, background)
        ge_values = ge.shap_values(test_samples)
        ge_time = time.time() - t0

        if isinstance(ge_values, list):
            ge_values = ge_values[0]
        ge_per_feature = np.mean(np.abs(ge_values), axis=(0, 1))

        results["gradient_explainer"] = {
            "total_time_s": round(ge_time, 2),
            "per_sample_ms": round(ge_time / n_samples * 1000, 1),
            "top_3_features": [
                feature_names[i] for i in np.argsort(ge_per_feature)[::-1][:3]
            ],
        }
        logger.info("  GradientExplainer: %.1fms/sample, top-3: %s",
                     ge_time / n_samples * 1000,
                     results["gradient_explainer"]["top_3_features"])
    except Exception as exc:
        results["gradient_explainer"] = {"error": str(exc)}
        ge_per_feature = None
        logger.error("  GradientExplainer failed: %s", exc)

    # ── KernelSHAP ──
    logger.info("Running KernelSHAP...")
    t0 = time.time()
    try:
        # KernelSHAP needs a predict function, not a model
        # Flatten windows for KernelSHAP (it expects 2D input)
        bg_flat = background.reshape(n_background, -1)
        test_flat = test_samples.reshape(n_samples, -1)

        def predict_flat(X_flat):
            X_3d = X_flat.reshape(-1, 20, len(feature_names))
            return model.predict(X_3d, verbose=0).ravel()

        ke = shap.KernelExplainer(predict_flat, bg_flat[:10])  # Small background for speed
        ke_values = ke.shap_values(test_flat[:3], nsamples=50)  # Fewer samples for speed

        ke_time = time.time() - t0

        if isinstance(ke_values, list):
            ke_values = ke_values[0] if len(ke_values) > 0 else ke_values
        ke_values = np.array(ke_values)
        ke_per_feature = np.mean(np.abs(ke_values.reshape(-1, 20, len(feature_names))), axis=(0, 1))

        results["kernel_shap"] = {
            "total_time_s": round(ke_time, 2),
            "per_sample_ms": round(ke_time / min(n_samples, 3) * 1000, 1),
            "top_3_features": [
                feature_names[i] for i in np.argsort(ke_per_feature)[::-1][:3]
            ],
        }
        logger.info("  KernelSHAP: %.1fms/sample, top-3: %s",
                     ke_time / min(n_samples, 3) * 1000,
                     results["kernel_shap"]["top_3_features"])
    except Exception as exc:
        results["kernel_shap"] = {"error": str(exc)}
        ke_per_feature = None
        logger.error("  KernelSHAP failed: %s", exc)

    # ── Correlation ──
    if ge_per_feature is not None and ke_per_feature is not None:
        correlation = float(np.corrcoef(ge_per_feature, ke_per_feature)[0, 1])
        results["correlation"] = round(correlation, 4)
        logger.info("  Correlation between methods: %.4f", correlation)

    # ── Speedup ──
    ge_ms = results.get("gradient_explainer", {}).get("per_sample_ms", 0)
    ke_ms = results.get("kernel_shap", {}).get("per_sample_ms", 0)
    if ge_ms > 0 and ke_ms > 0:
        results["speedup"] = round(ke_ms / ge_ms, 1)
        logger.info("  GradientExplainer is %.1fx faster than KernelSHAP", results["speedup"])

    # Save results
    output_path = PROJECT_ROOT / "data" / "phase5" / "shap_benchmark.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved: %s", output_path)

    return results


if __name__ == "__main__":
    run_benchmark()
