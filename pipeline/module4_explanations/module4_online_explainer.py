#!/usr/bin/env python3
"""Online-capable per-alert explanation pipeline with latency profiling.

Demonstrates that per-alert explanations can be generated within
real-time SLA (<150ms) using TreeSHAP + DAE decomposition + NLG.

Design: online-capable, validated in batch mode on the test set.
  - Global artifacts (importance rankings, templates) loaded once at startup
  - Per-alert: predict → TreeSHAP → DAE decompose → NLG fill → emit
  - Conditional: full explanations for MEDIUM+ only; LOW gets vote-only

Usage:
    python explain_alert.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import joblib
import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results/reports"

from pipeline.common.phi import BIOMETRIC_COLUMNS as BIOMETRIC_FEATURES  # noqa: E402

# ── Feature group mapping (Fix 3: SHAP stability) ─────────────────────
# Map individual SHAP features to clinically meaningful narrative
# categories. This absorbs within-category feature swaps (e.g.,
# DIntPkt↔Sport) that account for most SHAP instability, producing
# stable narratives even when the exact top-1 feature changes.
_FEATURE_GROUPS = {
    # Network timing anomalies
    "DIntPkt":    ("unusual network packet timing",     "network_timing"),
    "SIntPkt":    ("unusual network packet timing",     "network_timing"),
    "SIntPktAct": ("unusual network packet timing",     "network_timing"),
    "Dur":        ("abnormal connection duration",      "network_timing"),
    # Network protocol anomalies
    "Sport":      ("unexpected network port activity",  "network_protocol"),
    "Flgs":       ("abnormal protocol flags",           "network_protocol"),
    # Transfer volume anomalies
    "SrcBytes":   ("unusual data transfer volume",      "network_volume"),
    "DstBytes":   ("unusual data transfer volume",      "network_volume"),
    "TotBytes":   ("unusual data transfer volume",      "network_volume"),
    "SrcLoad":    ("abnormal network load",             "network_volume"),
    "DstLoad":    ("abnormal network load",             "network_volume"),
    "Load":       ("abnormal network load",             "network_volume"),
    # Packet structure anomalies
    "sMaxPktSz":  ("unusual packet structure",          "network_packet"),
    "dMaxPktSz":  ("unusual packet structure",          "network_packet"),
    "sMinPktSz":  ("unusual packet structure",          "network_packet"),
    "pSrcLoss":   ("abnormal packet loss",              "network_loss"),
    "pDstLoss":   ("abnormal packet loss",              "network_loss"),
    # Biometric anomalies
    "Temp":       ("abnormal temperature reading",      "biometric"),
    "SpO2":       ("abnormal oxygen saturation",        "biometric"),
    "Pulse_Rate": ("abnormal pulse rate",               "biometric"),
    "SYS":        ("abnormal blood pressure",           "biometric"),
    "DIA":        ("abnormal blood pressure",           "biometric"),
    "Heart_rate": ("abnormal heart rate",               "biometric"),
    "Resp_Rate":  ("abnormal respiratory rate",         "biometric"),
    "ST":         ("abnormal ST segment",               "biometric"),
}


def _feature_to_narrative(feature_name: str) -> tuple:
    """Map a SHAP feature name to (narrative_phrase, category).

    Returns:
        (narrative_phrase, category) — e.g., ("unusual network packet timing", "network_timing")
    """
    return _FEATURE_GROUPS.get(feature_name, (feature_name, "unknown"))


CLINICIAN_TEMPLATES = {
    "CRITICAL": (
        "CRITICAL ALERT: The system detected a likely intrusion "
        "affecting this patient's monitoring session. The primary indicator was "
        "{narrative}. "
        "{secondary_note}"
        "Recommend immediate review of device connectivity and patient vitals."
    ),
    "HIGH": (
        "HIGH ALERT: Suspicious activity detected. "
        "Key factor: {narrative}. "
        "{secondary_note}"
        "Consider verifying device integrity."
    ),
    "MEDIUM": (
        "MODERATE ALERT: Minor anomaly detected — "
        "{narrative}. "
        "{secondary_note}"
        "No immediate clinical action required, but flagged for review."
    ),
    "LOW": (
        "LOW ALERT: Borderline detection by one model. "
        "Likely benign; logged for audit purposes."
    ),
}

TRACK_A = {
    "xgboost": {
        "pipeline": "results/models/xgboost_final_pipeline.pkl",
        "report": "results/models/xgboost_final_report.json",
    },
    "random_forest": {
        "pipeline": "results/models/random_forest_final_pipeline.pkl",
        "report": "results/models/random_forest_final_report.json",
    },
    "decision_tree": {
        "pipeline": "results/models/decision_tree_final_pipeline.pkl",
        "report": "results/models/decision_tree_final_report.json",
    },
}


# ── AlertExplainer ─────────────────────────────────────────────────────

class AlertExplainer:
    """Per-alert explanation engine. Load once, call explain() per sample."""

    # Models that vote on detection but are NOT explained via TreeSHAP.
    # RandomForest stays in the predict vote (so n_flagged math is unchanged)
    # but its TreeExplainer is skipped because:
    #   1. It dominates startup_ms (~1.5 s of the 2.0 s baseline)
    #   2. It dominates treeshap_ms (~95 of the 108 ms baseline)
    #   3. Nothing downstream reads analyst.track_a.random_forest:
    #      the clinician summary uses XGBoost, the analyst panel renders
    #      waterfall_xgboost_*.png, and the offline batch explainer
    #      (module4_explanations.py) is the source of all RF charts.
    SKIP_SHAP = frozenset({"random_forest"})

    def __init__(self) -> None:
        t0 = time.perf_counter()

        # Track A: extract classifiers + create TreeExplainers (selectively)
        self.classifiers = {}
        self.explainers = {}
        self.thresholds = {}

        # Phase 2 final-training writes the bare classifier (NOT a
        # full Pipeline with the SMOTE wrapper) and signs it with the
        # Module 5 ECDSA key. ``loads_signed`` refuses to deserialise
        # without a valid signature, so a tampered or unsigned pickle
        # in results/models/ aborts startup rather than RCE-ing.
        # See finding #3a in the Phase 2 security review.
        from pipeline.common import loads_signed

        for name, cfg in TRACK_A.items():
            obj = loads_signed(PROJECT_ROOT / cfg["pipeline"])
            # Backwards compatibility: if the artefact is still a full
            # sklearn Pipeline (legacy file from before the strip), pull
            # the classifier out; otherwise the bare classifier is what
            # was loaded.
            self.classifiers[name] = (
                obj.named_steps["classifier"]
                if hasattr(obj, "named_steps") else obj
            )
            if name not in self.SKIP_SHAP:
                self.explainers[name] = shap.TreeExplainer(self.classifiers[name])
            with open(PROJECT_ROOT / cfg["report"]) as f:
                self.thresholds[name] = json.load(f)["optimal_threshold"]

        # Track B: DAE detector loaded via the pickle-free
        # DAEDetector.from_artefacts() API. Phase 2 finding #3b.
        from pipeline.module2_detection.models.DAE import DAEDetector
        self.dae = DAEDetector.from_artefacts(
            json_path=PROJECT_ROOT / "results/models/dae_detector.json",
            weights_path=PROJECT_ROOT / "results/models/dae_model.weights.h5",
        )

        # Feature names
        df = pd.read_parquet(PROJECT_ROOT / "data/processed/test_phase1.parquet",
                             columns=["Label"])
        self.feat_names = None  # set on first call from data shape

        self._startup_ms = round((time.perf_counter() - t0) * 1000, 1)
        logger.info("AlertExplainer loaded in %.1fms", self._startup_ms)

    def _severity(self, n_flagged: int) -> str:
        if n_flagged >= 4:
            return "CRITICAL"
        elif n_flagged == 3:
            return "HIGH"
        elif n_flagged == 2:
            return "MEDIUM"
        return "LOW"

    def _top_shap(self, sv_row: np.ndarray, k: int = 3) -> list:
        abs_vals = np.abs(sv_row)
        top_i = np.argsort(abs_vals)[-k:][::-1]
        return [
            {
                "feature": self.feat_names[i],
                "shap_value": round(float(sv_row[i]), 6),
                "direction": "increases_risk" if sv_row[i] > 0 else "decreases_risk",
            }
            for i in top_i
        ]

    def _top_dae(self, werr_row: np.ndarray, k: int = 3) -> list:
        total = werr_row.sum()
        top_i = np.argsort(werr_row)[-k:][::-1]
        return [
            {
                "feature": self.feat_names[i],
                "weighted_error": round(float(werr_row[i]), 8),
                "pct_contribution": round(float(werr_row[i] / total * 100), 1) if total > 0 else 0.0,
            }
            for i in top_i
        ]

    def _clinician_nlg(self, severity: str, top_features: list) -> str:
        """Generate clinician narrative using feature group categories.

        Fix 2 (confidence band): when the top-2 SHAP feature is ≥80% of
        top-1 magnitude, cite both in the narrative to be honest about
        explanation ambiguity.

        Fix 3 (feature groups): map individual features to clinically
        meaningful categories (e.g., DIntPkt → "unusual network packet
        timing") to absorb within-category feature swaps.
        """
        if severity == "LOW":
            return CLINICIAN_TEMPLATES["LOW"]

        top1_feat = top_features[0]["feature"]
        top1_val = abs(top_features[0]["shap_value"])
        narrative_1, category_1 = _feature_to_narrative(top1_feat)

        # Fix 2: confidence band — check if top-2 is close to top-1
        secondary_note = ""
        if len(top_features) >= 2:
            top2_feat = top_features[1]["feature"]
            top2_val = abs(top_features[1]["shap_value"])
            ambiguity_ratio = top2_val / top1_val if top1_val > 0 else 0

            if ambiguity_ratio > 0.8:
                # Ambiguous: cite both features
                narrative_2, category_2 = _feature_to_narrative(top2_feat)
                if category_1 != category_2:
                    secondary_note = (
                        f"A secondary indicator ({narrative_2}) also contributed. "
                    )
                # else: same category, single narrative covers both

            # Add biometric note if any top feature is biometric
            bio_feats = [f["feature"] for f in top_features
                         if f["feature"] in BIOMETRIC_FEATURES]
            if bio_feats and category_1 != "biometric":
                secondary_note += (
                    f"Note: Biometric data ({', '.join(bio_feats)}) "
                    "showed unusual values. "
                )

        return CLINICIAN_TEMPLATES[severity].format(
            narrative=narrative_1,
            secondary_note=secondary_note,
        )

    def explain(self, x_sample: np.ndarray, feat_names: list) -> dict:
        """Generate per-alert explanation with component-level timing.

        Args:
            x_sample: Single sample, shape (n_features,).
            feat_names: Feature name list.

        Returns:
            Dict with explanation + timing breakdown (ms).
        """
        self.feat_names = feat_names
        x_2d = x_sample.reshape(1, -1)
        timings = {}
        t_total = time.perf_counter()

        # ── Step 1: Model predictions ──
        t0 = time.perf_counter()
        votes = {}
        for name, clf in self.classifiers.items():
            proba = float(clf.predict_proba(x_2d)[0, 1])
            pred = int(proba >= self.thresholds[name])
            votes[name] = {"prediction": pred, "confidence": round(proba, 4)}

        # Cascaded DAE prediction: augment input with Track A probabilities.
        # The DAE was trained on [raw_features || Track_A_OOF_probas],
        # so inference must provide the same augmented input.
        track_a_probas = np.array([[
            votes[name]["confidence"]
            for name in self.classifiers
        ]])  # shape (1, 3)
        x_augmented = np.column_stack([x_2d, track_a_probas])

        dae_error_arr, dae_per_feature = self.dae.reconstruction_error_decomposed(x_augmented)
        dae_error = float(dae_error_arr[0])
        dae_pred = int(dae_error > self.dae.threshold)
        votes["dae"] = {"prediction": dae_pred, "reconstruction_error": round(dae_error, 8)}
        timings["predict_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        # ── Step 2: Determine severity ──
        n_flagged = sum(1 for v in votes.values() if v["prediction"] == 1)
        severity = self._severity(n_flagged)

        # Minimal explanation for LOW severity
        if severity == "LOW" and n_flagged <= 1:
            timings["total_ms"] = round((time.perf_counter() - t_total) * 1000, 3)
            return {
                "severity": severity,
                "n_models_flagged": n_flagged,
                "votes": votes,
                "explanation_level": "minimal",
                "clinician_summary": CLINICIAN_TEMPLATES["LOW"],
                "timings_ms": timings,
            }

        # ── Step 3: TreeSHAP (per model) ──
        t0 = time.perf_counter()
        shap_explanations = {}
        for name, explainer in self.explainers.items():
            sv = explainer.shap_values(x_2d)
            if isinstance(sv, list):
                sv = sv[1]
            elif isinstance(sv, np.ndarray) and sv.ndim == 3:
                sv = sv[:, :, 1]
            sv_row = sv[0]
            shap_explanations[name] = {
                "top_features": self._top_shap(sv_row),
                "shap_values": sv_row.tolist(),
            }
        timings["treeshap_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        # ── Step 4: DAE decomposition ──
        # Reuses the per-feature weighted error array computed during
        # Step 1's single forward pass — no second normalise + no second
        # Keras call. This stage is now pure-numpy slicing.
        t0 = time.perf_counter()
        w_err = dae_per_feature[0]
        dae_explanation = {"top_features": self._top_dae(w_err)}
        timings["dae_decompose_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        # ── Step 5: NLG generation ──
        t0 = time.perf_counter()
        # Use XGBoost SHAP as primary for clinician summary
        primary_top = shap_explanations["xgboost"]["top_features"]
        clinician_summary = self._clinician_nlg(severity, primary_top)
        timings["nlg_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        # ── Step 6: Risk decomposition ──
        t0 = time.perf_counter()
        flagging_models = [name for name, v in votes.items() if v["prediction"] == 1]
        confidences = [votes[m].get("confidence", 0) for m in flagging_models if "confidence" in votes[m]]
        risk_decomposition = {
            "flagging_models": flagging_models,
            "confidence_spread": {
                "min": round(min(confidences), 4) if confidences else 0,
                "max": round(max(confidences), 4) if confidences else 0,
                "mean": round(float(np.mean(confidences)), 4) if confidences else 0,
            },
            "dae_contributes": dae_pred == 1,
        }
        timings["risk_decompose_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        timings["total_ms"] = round((time.perf_counter() - t_total) * 1000, 3)

        return {
            "severity": severity,
            "n_models_flagged": n_flagged,
            "votes": votes,
            "explanation_level": "full",
            "analyst": {
                "track_a": shap_explanations,
                "track_b": dae_explanation,
            },
            "clinician_summary": clinician_summary,
            "risk_decomposition": risk_decomposition,
            "timings_ms": timings,
        }


# ── Batch simulation + latency profiling ───────────────────────────────

def run_batch_simulation(
    explainer: AlertExplainer,
    X_test: np.ndarray,
    y_pred_xgb: np.ndarray,
    feat_names: list,
) -> tuple:
    """Run per-alert explanations for all XGBoost-flagged samples."""
    alert_idx = np.where(y_pred_xgb == 1)[0]
    logger.info("Simulating %d per-alert explanations...", len(alert_idx))

    all_timings = []
    sample_explanations = []

    for i, idx in enumerate(alert_idx):
        result = explainer.explain(X_test[idx], feat_names)
        all_timings.append(result["timings_ms"])

        if len(sample_explanations) < 20:
            result["sample_index"] = int(idx)
            sample_explanations.append(result)

        if (i + 1) % 100 == 0:
            logger.info("  Processed %d/%d alerts", i + 1, len(alert_idx))

    return all_timings, sample_explanations


def compute_latency_stats(all_timings: list) -> dict:
    """Compute p50/p95/p99 for each timing component."""
    if not all_timings:
        return {}

    components = list(all_timings[0].keys())
    stats = {}

    for comp in components:
        values = [t[comp] for t in all_timings if comp in t]
        if not values:
            continue
        arr = np.array(values)
        stats[comp] = {
            "n_samples": len(arr),
            "mean": round(float(arr.mean()), 3),
            "p50": round(float(np.percentile(arr, 50)), 3),
            "p95": round(float(np.percentile(arr, 95)), 3),
            "p99": round(float(np.percentile(arr, 99)), 3),
            "min": round(float(arr.min()), 3),
            "max": round(float(arr.max()), 3),
        }

    return stats


def plot_latency_distribution(all_timings: list) -> None:
    """Histogram of per-alert total latency."""
    totals = [t["total_ms"] for t in all_timings]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(totals, bins=50, edgecolor="black", alpha=0.7, color="#3274A1")
    ax.axvline(np.percentile(totals, 95), color="red", linestyle="--",
               label=f"p95 = {np.percentile(totals, 95):.1f}ms")
    ax.axvline(np.percentile(totals, 50), color="orange", linestyle="--",
               label=f"p50 = {np.percentile(totals, 50):.1f}ms")
    ax.set_xlabel("Total Latency (ms)")
    ax.set_ylabel("Count")
    ax.set_title("Per-Alert Explanation Latency Distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "results/charts" / "latency_distribution.png", dpi=150)
    plt.close(fig)
    logger.info("  Chart: latency_distribution.png")


def plot_latency_cdf(all_timings: list) -> None:
    """CDF showing % of alerts below each latency threshold."""
    totals = np.sort([t["total_ms"] for t in all_timings])
    cdf = np.arange(1, len(totals) + 1) / len(totals) * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(totals, cdf, linewidth=2, color="#3274A1")

    # SLA markers
    for sla, color in [(50, "green"), (100, "orange"), (150, "red")]:
        pct = (totals < sla).sum() / len(totals) * 100
        ax.axvline(sla, color=color, linestyle="--", alpha=0.7,
                   label=f"{sla}ms SLA: {pct:.1f}% pass")

    ax.set_xlabel("Total Latency (ms)")
    ax.set_ylabel("Cumulative % of Alerts")
    ax.set_title("Per-Alert Explanation Latency CDF")
    ax.legend()
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "results/charts" / "latency_cdf.png", dpi=150)
    plt.close(fig)
    logger.info("  Chart: latency_cdf.png")


def plot_component_breakdown(stats: dict) -> None:
    """Stacked bar showing latency breakdown by component."""
    components = ["predict_ms", "treeshap_ms", "dae_decompose_ms", "nlg_ms", "risk_decompose_ms"]
    labels = ["Predict", "TreeSHAP", "DAE Decompose", "NLG", "Risk Decompose"]
    colors = ["#C44E52", "#3274A1", "#55A868", "#CCB974", "#8172B2"]

    # Use p50 values for full explanations
    vals = []
    for c in components:
        if c in stats:
            vals.append(stats[c]["p50"])
        else:
            vals.append(0)

    fig, ax = plt.subplots(figsize=(10, 4))
    left = 0
    for label, val, color in zip(labels, vals, colors):
        ax.barh("Per-Alert", val, left=left, color=color, label=f"{label} ({val:.1f}ms)")
        left += val

    ax.set_xlabel("Latency (ms)")
    ax.set_title(f"Per-Alert Explanation — Component Breakdown (p50 total={sum(vals):.1f}ms)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "results/charts" / "latency_component_breakdown.png", dpi=150)
    plt.close(fig)
    logger.info("  Chart: latency_component_breakdown.png")


# ── Main ────────────────────────────────────────────────────────────────

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
    drop_cols = ["Label", "Attack Category"]
    feat_names = [c for c in df.columns if c not in drop_cols]
    X_test = df[feat_names].values.astype(np.float32)

    # Load XGBoost predictions to identify alert samples
    xgb_preds = np.load(
        PROJECT_ROOT / "results/models/xgboost_test_predictions.npz"
    )
    y_pred_xgb = xgb_preds["y_pred"]
    n_alerts = (y_pred_xgb == 1).sum()
    logger.info("Test set: %d samples, %d XGBoost alerts to explain", len(X_test), n_alerts)

    # Initialize explainer (one-time startup cost)
    logger.info("Loading AlertExplainer (one-time startup)...")
    explainer = AlertExplainer()

    # Warmup: single call to trigger any lazy compilation
    _ = explainer.explain(X_test[0], feat_names)
    logger.info("Warmup complete")

    # Batch simulation
    logger.info("")
    logger.info("── Per-Alert Simulation ──")
    all_timings, sample_explanations = run_batch_simulation(
        explainer, X_test, y_pred_xgb, feat_names,
    )

    # Compute stats
    stats = compute_latency_stats(all_timings)

    # Separate stats for full vs minimal explanations
    full_timings = [t for t in all_timings if "treeshap_ms" in t]
    minimal_timings = [t for t in all_timings if "treeshap_ms" not in t]

    profile = {
        "n_alerts_total": len(all_timings),
        "n_full_explanations": len(full_timings),
        "n_minimal_explanations": len(minimal_timings),
        "startup_ms": explainer._startup_ms,
        "all_alerts": stats,
        "full_only": compute_latency_stats(full_timings) if full_timings else {},
        "minimal_only": compute_latency_stats(minimal_timings) if minimal_timings else {},
    }

    # Save — canonical filenames consumed by the eval app's simulation
    # panel (`module6_app.py:load_latency_profile` and the
    # online_sample_explanations baseline fixture). The legacy paths
    # under results/charts/latency_profile.json and
    # results/reports/sample_explanations.json are also written so any
    # external tooling that hard-coded the old names keeps working.
    profile_path = OUTPUT_DIR / "online_latency_profile.json"
    profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    logger.info("Saved: %s", profile_path)

    legacy_profile_path = PROJECT_ROOT / "results/charts" / "latency_profile.json"
    legacy_profile_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    logger.info("Saved: %s (legacy alias)", legacy_profile_path)

    # Make sample explanations JSON-serializable (convert numpy)
    def _clean(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        return obj

    cleaned_samples = _clean(sample_explanations)
    samples_path = OUTPUT_DIR / "online_sample_explanations.json"
    samples_path.write_text(
        json.dumps(cleaned_samples, indent=2), encoding="utf-8",
    )
    logger.info("Saved: %s (%d examples)", samples_path, len(sample_explanations))

    legacy_samples_path = OUTPUT_DIR / "sample_explanations.json"
    legacy_samples_path.write_text(
        json.dumps(cleaned_samples, indent=2), encoding="utf-8",
    )
    logger.info("Saved: %s (legacy alias)", legacy_samples_path)

    # Plots
    plot_latency_distribution(all_timings)
    plot_latency_cdf(all_timings)
    if full_timings:
        plot_component_breakdown(compute_latency_stats(full_timings))

    # Summary
    logger.info("")
    logger.info(sep)
    logger.info("LATENCY PROFILING COMPLETE")
    logger.info(sep)
    logger.info("  Alerts profiled : %d (%d full, %d minimal)",
                len(all_timings), len(full_timings), len(minimal_timings))
    if "total_ms" in stats:
        logger.info("  Total latency   : p50=%.1fms, p95=%.1fms, p99=%.1fms",
                    stats["total_ms"]["p50"], stats["total_ms"]["p95"], stats["total_ms"]["p99"])
    if full_timings:
        fs = compute_latency_stats(full_timings)
        if "total_ms" in fs:
            logger.info("  Full explain    : p50=%.1fms, p95=%.1fms, p99=%.1fms",
                        fs["total_ms"]["p50"], fs["total_ms"]["p95"], fs["total_ms"]["p99"])
        if "treeshap_ms" in fs:
            logger.info("  TreeSHAP only   : p50=%.1fms, p95=%.1fms",
                        fs["treeshap_ms"]["p50"], fs["treeshap_ms"]["p95"])
    logger.info("  SLA feasibility : <150ms per alert = %s",
                "PASS" if stats.get("total_ms", {}).get("p95", 999) < 150 else "FAIL")
    logger.info("  Output          : %s", OUTPUT_DIR)
    logger.info(sep)


if __name__ == "__main__":
    main()
