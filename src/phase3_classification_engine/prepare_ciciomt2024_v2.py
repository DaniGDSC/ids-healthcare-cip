# Cross-dataset alignment methodology:
# - 6 features mapped (High-confidence): Argus → CICFlowMeter
#   dur→duration, load→rate, srcload→srate, dstload→drate,
#   totbytes→tot_sum, packet_num→number
# - 23 features imputed: median of WUSTL Normal samples (label=0)
#   Includes: 8 biometric features + 15 unmapped network features
# - Scaler: WUSTL train set only — no refitting on CICIoMT2024
# - Zero-padding avoided: signals FDI attack pattern in WUSTL
# - Conservative approach: results represent lower bound of
#   true cross-dataset generalization performance

"""CICIoMT2024 cross-dataset alignment v2 — 6 high-confidence mappings.

Maps 6 semantically equivalent features between WUSTL-EHMS-2020
(Argus) and CICIoMT2024 (CICFlowMeter), imputes the remaining
23 with WUSTL Normal medians, validates alignment integrity,
applies the WUSTL-fitted scaler (no refit), and exports.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

from src.phase0_dataset_analysis.phase0.security import IntegrityVerifier

# ── Constants ──────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

WUSTL_TRAIN_PATH: Path = PROJECT_ROOT / "data" / "processed" / "train_phase1.parquet"
CICIOMT_PATH: Path = PROJECT_ROOT / "data" / "external" / "ciciomt2024_labeled.parquet"
SCALER_PATH: Path = PROJECT_ROOT / "models" / "scalers" / "robust_scaler.pkl"
OUTPUT_DIR: Path = PROJECT_ROOT / "data" / "external"
OUTPUT_PARQUET: str = "ciciomt2024_aligned.parquet"
OUTPUT_REPORT: str = "alignment_report.json"

LABEL_COLUMN: str = "label"
LABEL_NORMAL: int = 0
LABEL_ATTACK: int = 1

N_FEATURES: int = 29
N_MAPPED: int = 6
N_IMPUTED: int = 23

# Canonical WUSTL 29-feature order (post-normalization, excluding label)
WUSTL_FEATURE_ORDER: List[str] = [
    "srcbytes",
    "dstbytes",
    "srcload",
    "dstload",
    "srcgap",
    "dstgap",
    "sintpkt",
    "dintpkt",
    "sintpktact",
    "dintpktact",
    "smaxpktsz",
    "dmaxpktsz",
    "sminpktsz",
    "dminpktsz",
    "dur",
    "trans",
    "totbytes",
    "load",
    "psrcloss",
    "pdstloss",
    "packet_num",
    "temp",
    "spo2",
    "pulse_rate",
    "sys",
    "dia",
    "heart_rate",
    "resp_rate",
    "st",
]

# 6 High-confidence semantic mappings: WUSTL (Argus) → CICIoMT2024
# Keys = WUSTL feature name, Values = actual CICIoMT2024 column name
MAPPING: Dict[str, str] = {
    "dur": "duration",
    "load": "rate",
    "srcload": "srate",
    "dstload": "drate",
    "totbytes": "tot_sum",
    "packet_num": "number",
}

# ── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Functions ──────────────────────────────────────────────────────────


def normalize_columns(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Normalize column names: strip, lowercase, spaces to underscores.

    Args:
        df: DataFrame to normalize.
        name: Dataset name for logging.

    Returns:
        DataFrame with normalized column names.
    """
    raw_cols = list(df.columns)
    normalized = [c.strip().lower().replace(" ", "_") for c in raw_cols]
    df.columns = normalized
    logger.info("Normalized %s features: %s", name, normalized)
    return df


def load_artifacts() -> Tuple[pd.DataFrame, pd.DataFrame, Any]:
    """Load WUSTL train, CICIoMT2024, and scaler artifacts.

    Returns:
        Tuple of (wustl_df, ciciomt_df, scaler).

    Raises:
        FileNotFoundError: If any artifact is missing.
    """
    logger.info("── Loading artifacts ──")

    # WUSTL train
    if not WUSTL_TRAIN_PATH.exists():
        raise FileNotFoundError(f"WUSTL train not found: {WUSTL_TRAIN_PATH}")
    wustl_df = pd.read_parquet(WUSTL_TRAIN_PATH)
    wustl_features_raw = [c for c in wustl_df.columns if c != "Label"]
    logger.info("WUSTL features (%d): %s", len(wustl_features_raw), list(wustl_df.columns))
    wustl_df = normalize_columns(wustl_df, "WUSTL")

    # CICIoMT2024
    if not CICIOMT_PATH.exists():
        raise FileNotFoundError(f"CICIoMT2024 not found: {CICIOMT_PATH}")
    ciciomt_df = pd.read_parquet(CICIOMT_PATH)
    cic_features_raw = [c for c in ciciomt_df.columns if c != "Label"]
    logger.info("CICIoMT2024 features (%d): %s", len(cic_features_raw), list(ciciomt_df.columns))
    ciciomt_df = normalize_columns(ciciomt_df, "CICIoMT2024")

    logger.info("  CICIoMT2024 samples: %d", len(ciciomt_df))

    # Scaler
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")
    scaler = joblib.load(SCALER_PATH)
    logger.info("  Loaded scaler: %s", SCALER_PATH.name)

    return wustl_df, ciciomt_df, scaler


def compute_wustl_normal_medians(wustl_df: pd.DataFrame) -> Dict[str, float]:
    """Compute median values from WUSTL Normal samples (label=0).

    Args:
        wustl_df: Full WUSTL train DataFrame (normalized columns).

    Returns:
        Dict mapping feature name to median value.
    """
    normal_df = wustl_df[wustl_df[LABEL_COLUMN] == LABEL_NORMAL]
    logger.info("Computed medians from %d Normal samples", len(normal_df))

    feature_cols = [c for c in normal_df.columns if c != LABEL_COLUMN]
    medians: Dict[str, float] = {}
    for col in feature_cols:
        medians[col] = float(normal_df[col].median())

    logger.info(
        "  Sample medians: dur=%.6f, totbytes=%.6f, packet_num=%.6f",
        medians.get("dur", 0.0),
        medians.get("totbytes", 0.0),
        medians.get("packet_num", 0.0),
    )

    return medians


def build_aligned_dataframe(
    ciciomt_df: pd.DataFrame,
    medians: Dict[str, float],
) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    """Build 29-feature aligned DataFrame in WUSTL canonical order.

    Step A: Initialize all 29 features with WUSTL Normal medians.
    Step B: Override 6 mapped features with CICIoMT2024 values.
    Step C: Append label column from CICIoMT2024.

    Args:
        ciciomt_df: CICIoMT2024 DataFrame (normalized columns).
        medians: WUSTL Normal median values.

    Returns:
        Tuple of (aligned_df, successful_mappings, imputed_list).
    """
    logger.info("── Building aligned dataframe ──")

    n_samples = len(ciciomt_df)
    cic_cols = set(ciciomt_df.columns)

    # Step A: all 29 features = WUSTL Normal medians
    aligned: Dict[str, np.ndarray] = {}
    for col in WUSTL_FEATURE_ORDER:
        aligned[col] = np.full(n_samples, medians[col], dtype=np.float32)

    # Step B: override with 6 high-confidence mappings
    successful: Dict[str, str] = {}
    failed: List[str] = []

    for wustl_col, cic_col in MAPPING.items():
        if cic_col in cic_cols:
            aligned[wustl_col] = ciciomt_df[cic_col].values.astype(np.float32)
            successful[wustl_col] = cic_col
            logger.info("  Mapped: %s ← ciciomt[%s]", wustl_col, cic_col)
        else:
            failed.append(cic_col)
            logger.warning("  WARNING: %s not found — using median", cic_col)

    # Imputed features = all 29 minus the successfully mapped ones
    imputed_list = [c for c in WUSTL_FEATURE_ORDER if c not in successful]

    # Step C: label from CICIoMT2024
    aligned[LABEL_COLUMN] = ciciomt_df[LABEL_COLUMN].values.astype(int)

    aligned_df = pd.DataFrame(aligned)

    logger.info(
        "  Result: %d mapped, %d imputed, %d failed",
        len(successful),
        len(imputed_list),
        len(failed),
    )

    return aligned_df, successful, imputed_list


def validate_aligned(
    aligned_df: pd.DataFrame,
    wustl_feature_order: List[str],
    successful: Dict[str, str],
    imputed_list: List[str],
) -> Dict[str, bool]:
    """Run validation checks on the aligned dataset.

    Args:
        aligned_df: Aligned DataFrame (pre-scaler).
        wustl_feature_order: Canonical WUSTL feature order.
        successful: Successfully mapped features.
        imputed_list: Imputed feature names.

    Returns:
        Dict mapping assertion name to PASS/FAIL.
    """
    logger.info("── Validation checks ──")
    checks: Dict[str, bool] = {}
    feature_cols = [c for c in aligned_df.columns if c != LABEL_COLUMN]

    # A1: Exactly 29 feature columns + label
    n_feats = len(feature_cols)
    passed = n_feats == N_FEATURES
    checks["exactly_29_features"] = passed
    logger.info(
        "  [%s] Exactly 29 feature columns + label (%d found)",
        "PASS" if passed else "FAIL",
        n_feats,
    )

    # A2: Column names match WUSTL
    names_match = set(feature_cols) == set(wustl_feature_order)
    checks["column_names_match_wustl"] = names_match
    logger.info(
        "  [%s] Column names match WUSTL train.parquet",
        "PASS" if names_match else "FAIL",
    )

    # A3: Column order matches WUSTL
    order_match = feature_cols == wustl_feature_order
    checks["column_order_matches_wustl"] = order_match
    logger.info(
        "  [%s] Column order matches WUSTL train.parquet",
        "PASS" if order_match else "FAIL",
    )

    # A4: No NaN values
    n_nan = int(aligned_df[feature_cols].isna().sum().sum())
    passed = n_nan == 0
    checks["no_nan_values"] = passed
    logger.info(
        "  [%s] No NaN values (%d found)",
        "PASS" if passed else "FAIL",
        n_nan,
    )

    # A5: Label values only 0 or 1
    unique_labels = set(aligned_df[LABEL_COLUMN].unique())
    passed = unique_labels.issubset({LABEL_NORMAL, LABEL_ATTACK})
    checks["label_values_binary"] = passed
    logger.info(
        "  [%s] Label values only 0 or 1: %s",
        "PASS" if passed else "FAIL",
        unique_labels,
    )

    # A6: 6 mapped features have variance > 0
    mapped_with_variance: List[str] = []
    mapped_zero_var: List[str] = []
    for col in successful:
        var = float(aligned_df[col].var())
        if var > 0:
            mapped_with_variance.append(col)
        else:
            mapped_zero_var.append(col)

    passed = len(mapped_with_variance) == len(successful)
    checks["mapped_features_have_variance"] = passed
    logger.info(
        "  [%s] 6 mapped features have variance > 0 (%d/%d)",
        "PASS" if passed else "FAIL",
        len(mapped_with_variance),
        len(successful),
    )
    if mapped_zero_var:
        logger.info("    Zero-variance mapped: %s", mapped_zero_var)

    # A7: 23 imputed features have zero variance
    imputed_zero_var: List[str] = []
    imputed_nonzero_var: List[str] = []
    for col in imputed_list:
        var = float(aligned_df[col].var())
        if var == 0.0:
            imputed_zero_var.append(col)
        else:
            imputed_nonzero_var.append(col)

    passed = len(imputed_zero_var) == len(imputed_list)
    checks["imputed_features_zero_variance"] = passed
    logger.info(
        "  [%s] 23 imputed features have zero variance (%d/%d)",
        "PASS" if passed else "FAIL",
        len(imputed_zero_var),
        len(imputed_list),
    )
    if imputed_nonzero_var:
        logger.info("    Non-zero variance imputed: %s", imputed_nonzero_var)

    n_passed = sum(checks.values())
    logger.info("  Validation: %d/%d PASSED", n_passed, len(checks))

    return checks


def apply_scaler(
    aligned_df: pd.DataFrame,
    scaler: Any,
) -> pd.DataFrame:
    """Apply WUSTL-fitted RobustScaler — no refitting.

    Args:
        aligned_df: Aligned DataFrame (29 features + label).
        scaler: Pre-fitted RobustScaler from Phase 1.

    Returns:
        Scaled DataFrame with label preserved.
    """
    logger.info("── Applying scaler ──")

    feature_cols = [c for c in aligned_df.columns if c != LABEL_COLUMN]
    X = aligned_df[feature_cols].values.astype(np.float32)
    X_scaled = scaler.transform(X).astype(np.float32)

    scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
    scaled_df[LABEL_COLUMN] = aligned_df[LABEL_COLUMN].values

    logger.info("  Applied WUSTL RobustScaler — no refitting")
    return scaled_df


def export_artifacts(
    scaled_df: pd.DataFrame,
    successful: Dict[str, str],
    imputed_list: List[str],
    medians: Dict[str, float],
    checks: Dict[str, bool],
) -> None:
    """Export aligned Parquet and alignment report JSON.

    Args:
        scaled_df: Scaled, aligned DataFrame.
        successful: Successfully mapped features (wustl→cic).
        imputed_list: Imputed feature names.
        medians: WUSTL Normal median values.
        checks: Validation results.
    """
    logger.info("── Export ──")

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    verifier = IntegrityVerifier(output_dir)

    # 1. Parquet
    parquet_path = output_dir / OUTPUT_PARQUET
    scaled_df.to_parquet(parquet_path, index=False)
    parquet_hash = verifier.compute_hash(parquet_path)
    logger.info("Exported %s", parquet_path.name)
    logger.info("SHA-256: %s", parquet_hash)

    # 2. Alignment report
    labels = scaled_df[LABEL_COLUMN]
    n_normal = int((labels == LABEL_NORMAL).sum())
    n_attack = int((labels == LABEL_ATTACK).sum())

    report: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_samples": len(scaled_df),
        "normal_samples": n_normal,
        "attack_samples": n_attack,
        "mapped_features": successful,
        "imputed_features": sorted(imputed_list),
        "wustl_normal_medians": {k: round(v, 6) for k, v in sorted(medians.items())},
        "mapping_success": len(successful),
        "mapping_failed": len(MAPPING) - len(successful),
        "validation": {k: ("PASS" if v else "FAIL") for k, v in checks.items()},
    }

    report_path = output_dir / OUTPUT_REPORT
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    report_hash = verifier.compute_hash(report_path)
    logger.info("Exported %s — SHA-256: %s", report_path.name, report_hash)


def run() -> None:
    """Execute the CICIoMT2024 alignment pipeline v2."""
    logger.info("═══════════════════════════════════════════════════")
    logger.info("  CICIoMT2024 Cross-Dataset Alignment v2")
    logger.info("  (6 High-Confidence Mappings)")
    logger.info("═══════════════════════════════════════════════════")

    # 1. Load artifacts
    wustl_df, ciciomt_df, scaler = load_artifacts()

    # 3. Compute WUSTL Normal medians
    medians = compute_wustl_normal_medians(wustl_df)

    # 4-5. Build aligned dataframe
    aligned_df, successful, imputed_list = build_aligned_dataframe(ciciomt_df, medians)

    # 5. Validation
    checks = validate_aligned(aligned_df, WUSTL_FEATURE_ORDER, successful, imputed_list)

    # 6. Apply scaler
    scaled_df = apply_scaler(aligned_df, scaler)

    # 7. Export
    export_artifacts(scaled_df, successful, imputed_list, medians, checks)

    logger.info("═══════════════════════════════════════════════════")
    logger.info(
        "  Alignment v2 complete — %d samples, %d features " "(%d mapped, %d imputed)",
        len(scaled_df),
        N_FEATURES,
        len(successful),
        len(imputed_list),
    )
    logger.info("═══════════════════════════════════════════════════")


if __name__ == "__main__":
    run()
