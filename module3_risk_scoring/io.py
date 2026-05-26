"""I/O for Module 3 — data loading + artefact saving + config exports.

Single responsibility: file-system reads and writes. All compute lives
in components/composition/feedback/analysis.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    BIOMETRIC_FEATURES,
    CIA_THREATS,
    DATA_SENSITIVITY,
    DEFAULT_DEVICE_TIER,
    DEVICE_TIERS,
    RESPONSE_MAPPING,
    RISK_THRESHOLDS,
    SIGMA_THRESHOLD,
    WEIGHTS,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results/reports"
CHARTS_DIR = PROJECT_ROOT / "results/charts"


# ── Path resolution ──────────────────────────────────────────────────


def _split_paths(split: str) -> dict:
    """Resolve per-split paths. Test = paper-clean; demo = operator-clean."""
    from common import split_paths as sp
    return {
        "parquet": sp.parquet(split),
        "out_npz": sp.risk_scores(split),
    }


# ── Loaders ─────────────────────────────────────────────────────────


def load_test_data(parquet_path: Path | None = None) -> tuple:
    """Load a split's parquet → X, y, attack_cats, feat_names."""
    path = parquet_path or (PROJECT_ROOT / "data/processed/test_phase1.parquet")
    df = pd.read_parquet(path)
    drop_cols = ["Label", "Attack Category", "row_id", "device_class"]
    feat_names = [c for c in df.columns if c not in drop_cols]
    X_test = df[feat_names].values.astype(np.float32)
    y_test = df["Label"].values
    attack_cats = (
        df["Attack Category"].values if "Attack Category" in df.columns else None
    )
    return X_test, y_test, attack_cats, feat_names


def load_xgboost_proba() -> tuple:
    """Load XGBoost predict_proba and optimal threshold."""
    preds = np.load(PROJECT_ROOT / "results/models/xgboost_test_predictions.npz")
    with open(PROJECT_ROOT / "results/models/xgboost_final_report.json") as f:
        threshold = json.load(f)["optimal_threshold"]
    return preds["y_proba"], threshold


# ── Config JSON exports (Tasks 3.1, 3.2, 3.8) ───────────────────────


def export_config_jsons(*, output_dir: Path | None = None) -> None:
    """Write the device-criticality / data-sensitivity / risk-config JSON files."""
    out = output_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    # 3.1 Device criticality
    device_crit = {
        "description": "Device criticality tiers mapped to D_crit scores",
        "tiers": {
            "1_life_sustaining": {"score": 1.0, "examples": ["infusion pump", "ventilator"]},
            "2_vital_monitoring": {"score": 0.8, "examples": ["ECG monitor", "pulse oximeter"]},
            "3_diagnostic": {"score": 0.5, "examples": ["blood pressure monitor", "thermometer"]},
            "4_auxiliary": {"score": 0.3, "examples": ["environmental sensor", "room monitor"]},
        },
        "default_tier": DEFAULT_DEVICE_TIER,
        "cia_threat_profiles": CIA_THREATS,
    }
    (out / "device_criticality.json").write_text(
        json.dumps(device_crit, indent=2), encoding="utf-8",
    )
    logger.info("  Saved: device_criticality.json")

    # 3.2 Data sensitivity
    data_sens = {
        "description": "Data sensitivity classification mapped to S_data scores",
        "tiers": {
            "phi_realtime": {"score": 1.0, "examples": ["real-time vital signs (SpO2, HR, BP)"]},
            "phi_stored": {"score": 0.7, "examples": ["stored patient records"]},
            "operational": {"score": 0.4, "examples": ["network flow metadata, device telemetry"]},
            "administrative": {"score": 0.1, "examples": ["timestamps, flags, non-clinical"]},
        },
    }
    (out / "data_sensitivity.json").write_text(
        json.dumps(data_sens, indent=2), encoding="utf-8",
    )
    logger.info("  Saved: data_sensitivity.json")

    # 3.8 Risk scoring config
    risk_cfg = {
        "formula": "R = w1*C_detect + w2*D_crit + w3*S_data + w4*D_clinical_tier",
        "fusion": (
            "C_detect = cascaded(Track_A → Track_B): DAE input = "
            "[raw_features || Track_A_probas]; DAE forward pass skipped where "
            "Track_A (XGBoost) proba >= 0.90 (compute optimisation; "
            "c_track_b=0 for those rows)"
        ),
        "weights": WEIGHTS,
        "thresholds": {label: thresh for thresh, label in RISK_THRESHOLDS},
        "alert_tiers": ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        "biometric_features": list(BIOMETRIC_FEATURES),
        "sigma_threshold": SIGMA_THRESHOLD,
        "device_tiers": DEVICE_TIERS,
        "data_sensitivity": DATA_SENSITIVITY,
        "cia_threats": CIA_THREATS,
        "response_mapping": RESPONSE_MAPPING,
    }
    (out / "risk_config.json").write_text(
        json.dumps(risk_cfg, indent=2), encoding="utf-8",
    )
    logger.info("  Saved: risk_config.json")


# ── save_outputs ────────────────────────────────────────────────────


def save_outputs(
    R: np.ndarray,
    c_detect: np.ndarray,
    d_crit: np.ndarray,
    s_data: np.ndarray,
    d_clinical_tier: np.ndarray,
    c_track_a: np.ndarray,
    c_track_b: np.ndarray,
    levels: np.ndarray,
    y_true: np.ndarray,
    attack_cats: np.ndarray,
    fusion: dict,
    contributions: dict,
    sensitivity: dict,
    worked_examples: list,
    *,
    out_npz: Path | None = None,
    output_dir: Path | None = None,
) -> None:
    """Save all risk score artifacts.

    Args:
        out_npz: defaults to ``risk_scores.npz`` (test); demo runs pass
            ``demo_scores.npz``. The auxiliary CSV/JSON outputs stay at
            canonical paths to preserve test as the source-of-truth for
            paper artifacts.
        output_dir: override for CSV/JSON outputs (used by tests).
    """
    out_dir = output_dir or OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_npz or (out_dir / "risk_scores.npz")
    np.savez(
        npz_path,
        R=R, c_detect=c_detect, d_crit=d_crit,
        s_data=s_data, d_clinical_tier=d_clinical_tier,
        c_track_a=c_track_a, c_track_b=c_track_b,
        risk_levels=levels, y_true=y_true,
    )
    logger.info("  Saved: %s", npz_path.name)

    # CSV detail
    df = pd.DataFrame({
        "R": R, "risk_level": levels, "y_true": y_true,
        "attack_category": attack_cats,
        "c_detect": c_detect, "c_track_a": c_track_a, "c_track_b": c_track_b,
        "d_crit": d_crit, "s_data": s_data, "d_clinical_tier": d_clinical_tier,
    })
    df.to_csv(out_dir / "risk_scores_detail.csv", index_label="sample_index")
    logger.info("  Saved: risk_scores_detail.csv")

    # JSON report — strict serialisation
    level_dist = {}
    for level in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        mask = levels == level
        level_dist[level] = {
            "count": int(mask.sum()),
            "pct": round(float(mask.mean() * 100), 1),
            "mean_R": round(float(R[mask].mean()), 4) if mask.any() else 0,
        }

    report = {
        "formula": "R = w1*C_detect + w2*D_crit + w3*S_data + w4*D_clinical_tier",
        "fusion": (
            "C_detect = cascaded(Track_A → Track_B): DAE input = "
            "[raw_features || Track_A_probas]; DAE forward pass skipped where "
            "Track_A (XGBoost) proba >= 0.90 (compute optimisation; c_track_b=0 "
            "for those rows)"
        ),
        "weights": WEIGHTS,
        "risk_thresholds": {label: thresh for thresh, label in RISK_THRESHOLDS},
        "total_samples": int(len(R)),
        "risk_level_distribution": level_dist,
        "response_mapping": RESPONSE_MAPPING,
        "overall_stats": {
            "mean_R": round(float(R.mean()), 4),
            "std_R": round(float(R.std()), 4),
            "median_R": round(float(np.median(R)), 4),
        },
        "per_category_stats": {},
        "dual_track_fusion": fusion,
        "component_contributions": contributions,
        "weight_sensitivity": sensitivity,
        "worked_examples": worked_examples,
        "limitations": [
            "Patient acuity proxy uses biometric deviation magnitude, not clinical diagnosis — a simplified surrogate for real patient acuity scoring (e.g., APACHE, NEWS2).",
            "Device criticality uses a static tier assignment for the WUSTL-EHMS-2020 testbed; production deployment requires integration with hospital asset management systems.",
            "Data sensitivity classification is feature-type-based, not content-aware — cannot distinguish encrypted vs plaintext PHI.",
            "Linear weighted sum assumes component independence; multiplicative or Bayesian formulations may better capture risk interactions.",
            "Weights are expert-calibrated defaults; institutional tuning via AHP or operational feedback loops is recommended for deployment.",
            "The WUSTL-EHMS-2020 dataset contains only 2 attack categories (Spoofing, Data Alteration); generalizability to broader IoMT threat landscapes requires validation on additional datasets.",
        ],
    }

    # Y2 fix: derive categories from data instead of hardcoded list. Still
    # report the canonical 3 (normal/Spoofing/Data Alteration) explicitly
    # when present so downstream paper figures stay stable.
    if attack_cats is not None:
        cats_str = attack_cats.astype(str)
        # Always include "normal" if any benign rows exist
        if (y_true == 0).any():
            mask = y_true == 0
            report["per_category_stats"]["normal"] = {
                "count": int(mask.sum()),
                "mean_R": round(float(R[mask].mean()), 4),
                "median_R": round(float(np.median(R[mask])), 4),
                "std_R": round(float(R[mask].std()), 4),
            }
        # Then every attack-class category that actually appears
        for cat in sorted(np.unique(cats_str[y_true == 1])):
            if cat in {"None", "nan", "normal"}:
                continue
            mask = (cats_str == cat) & (y_true == 1)
            if mask.any():
                report["per_category_stats"][cat] = {
                    "count": int(mask.sum()),
                    "mean_R": round(float(R[mask].mean()), 4),
                    "median_R": round(float(np.median(R[mask])), 4),
                    "std_R": round(float(R[mask].std()), 4),
                }

    report_path = out_dir / "risk_report.json"
    try:
        payload = json.dumps(report, indent=2)
    except TypeError as exc:
        raise TypeError(
            f"risk_report.json contains a non-JSON-serialisable value "
            f"(detail: {exc}). Fix the producer."
        ) from exc
    report_path.write_text(payload, encoding="utf-8")
    logger.info("  Saved: risk_report.json")


__all__ = [
    "OUTPUT_DIR",
    "CHARTS_DIR",
    "PROJECT_ROOT",
    "_split_paths",
    "load_test_data",
    "load_xgboost_proba",
    "export_config_jsons",
    "save_outputs",
]
