# Cross-dataset alignment methodology (v2):
# - 5 features mapped (High-confidence, non-zero variance):
#   dur→duration, load→rate, srcload→srate,
#   totbytes→tot_sum, packet_num→number
# - 1 mapping excluded (drate=0.0 across all 1.6M samples):
#   dstload → imputed via WUSTL Normal median
#   Interpretation: IoMT devices operate with unidirectional traffic
# - 24 features imputed: median of WUSTL Normal samples (label=0)
#   Includes: 8 biometric features + 16 unmapped network features
# - Scaler: WUSTL train set only — no refitting on CICIoMT2024
# - Zero-padding avoided: would signal FDI attack pattern in WUSTL
# - Results represent conservative lower bound of generalization

"""CICIoMT2024 cross-dataset alignment v3 — 5 high-confidence mappings.

Maps 5 semantically equivalent features between WUSTL-EHMS-2020
(Argus) and CICIoMT2024 (CICFlowMeter), excludes dstload→drate
(zero variance across all 1.6M samples), imputes the remaining
24 with WUSTL Normal medians, validates alignment integrity,
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
OUTPUT_PARQUET: str = "ciciomt2024_aligned_v2.parquet"
OUTPUT_REPORT: str = "alignment_report_v2.json"

LABEL_COLUMN: str = "label"
LABEL_NORMAL: int = 0
LABEL_ATTACK: int = 1

N_FEATURES: int = 29
N_MAPPED: int = 5
N_IMPUTED: int = 24

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

# 5 High-confidence semantic mappings: WUSTL (Argus) → CICIoMT2024
MAPPING: Dict[str, str] = {
    "dur": "duration",
    "load": "rate",
    "srcload": "srate",
    "totbytes": "tot_sum",
    "packet_num": "number",
}

# Excluded mapping — zero variance in CICIoMT2024 source column
EXCLUDED_MAPPING: Dict[str, Dict[str, str]] = {
    "dstload": {
        "cic_col": "drate",
        "reason": "zero variance — drate=0.0 across all CICIoMT2024 samples",
        "interpretation": "IoMT unidirectional traffic pattern",
        "action": "imputed via WUSTL Normal median",
    },
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
    logger.info(
        "WUSTL features (%d): %s",
        len([c for c in wustl_df.columns if c != "Label"]),
        list(wustl_df.columns),
    )
    wustl_df = normalize_columns(wustl_df, "WUSTL")

    # CICIoMT2024
    if not CICIOMT_PATH.exists():
        raise FileNotFoundError(f"CICIoMT2024 not found: {CICIOMT_PATH}")
    ciciomt_df = pd.read_parquet(CICIOMT_PATH)
    logger.info(
        "CICIoMT2024 features (%d): %s",
        len([c for c in ciciomt_df.columns if c != "Label"]),
        list(ciciomt_df.columns),
    )
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


def validate_mapped_columns(
    ciciomt_df: pd.DataFrame,
) -> Tuple[Dict[str, Tuple[str, float]], List[str]]:
    """Validate CICIoMT2024 mapped columns have non-zero variance.

    For each mapping, checks the CICIoMT2024 source column exists
    and has variance > 0. Zero-variance columns are moved to imputed.

    Args:
        ciciomt_df: CICIoMT2024 DataFrame (normalized columns).

    Returns:
        Tuple of (validated dict {wustl_col: (cic_col, variance)},
        list of demoted wustl columns).
    """
    logger.info("── Validating mapped columns ──")

    cic_cols = set(ciciomt_df.columns)
    validated: Dict[str, Tuple[str, float]] = {}
    demoted: List[str] = []

    for wustl_col, cic_col in MAPPING.items():
        if cic_col not in cic_cols:
            logger.warning(
                "  %s: column '%s' not found in CICIoMT2024 → imputed",
                wustl_col,
                cic_col,
            )
            demoted.append(wustl_col)
            continue

        variance = float(ciciomt_df[cic_col].var())
        if variance > 0:
            validated[wustl_col] = (cic_col, variance)
            logger.info("  %s: variance=%.4f ✓", cic_col, variance)
        else:
            logger.warning(
                "  WARNING: %s zero variance → %s imputed",
                cic_col,
                wustl_col,
            )
            demoted.append(wustl_col)

    # Log excluded mapping
    for wustl_col, info in EXCLUDED_MAPPING.items():
        logger.info(
            "  %s: EXCLUDED — %s",
            info["cic_col"],
            info["reason"],
        )

    return validated, demoted


def build_aligned_dataframe(
    ciciomt_df: pd.DataFrame,
    medians: Dict[str, float],
    validated: Dict[str, Tuple[str, float]],
) -> Tuple[pd.DataFrame, List[str]]:
    """Build 29-feature aligned DataFrame in WUSTL canonical order.

    Step A: Initialize all 29 features with WUSTL Normal medians.
    Step B: Override validated mapped features with CICIoMT2024 values.
    Step C: Append label column from CICIoMT2024.

    Args:
        ciciomt_df: CICIoMT2024 DataFrame (normalized columns).
        medians: WUSTL Normal median values.
        validated: Validated mappings {wustl_col: (cic_col, variance)}.

    Returns:
        Tuple of (aligned_df, imputed_list).
    """
    logger.info("── Building aligned dataframe ──")

    n_samples = len(ciciomt_df)

    # Step A: all 29 features = WUSTL Normal medians
    aligned: Dict[str, np.ndarray] = {}
    for col in WUSTL_FEATURE_ORDER:
        aligned[col] = np.full(n_samples, medians[col], dtype=np.float32)

    # Step B: override with validated mapped features only
    for wustl_col, (cic_col, variance) in validated.items():
        aligned[wustl_col] = ciciomt_df[cic_col].values.astype(np.float32)
        logger.info(
            "  Mapped: %s ← ciciomt[%s] (var=%.4f)",
            wustl_col,
            cic_col,
            variance,
        )

    # Imputed features = all 29 minus the successfully mapped ones
    imputed_list = [c for c in WUSTL_FEATURE_ORDER if c not in validated]

    # Step C: label from CICIoMT2024
    aligned[LABEL_COLUMN] = ciciomt_df[LABEL_COLUMN].values.astype(int)

    aligned_df = pd.DataFrame(aligned)

    logger.info(
        "  Result: %d mapped, %d imputed",
        len(validated),
        len(imputed_list),
    )

    return aligned_df, imputed_list


def validate_aligned(
    aligned_df: pd.DataFrame,
    wustl_feature_order: List[str],
    validated: Dict[str, Tuple[str, float]],
    imputed_list: List[str],
    medians: Dict[str, float],
) -> Dict[str, bool]:
    """Run validation checks on the aligned dataset.

    Args:
        aligned_df: Aligned DataFrame (pre-scaler).
        wustl_feature_order: Canonical WUSTL feature order.
        validated: Validated mappings.
        imputed_list: Imputed feature names.
        medians: WUSTL Normal medians for dstload check.

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

    # A6: 5 mapped features have variance > 0
    mapped_with_variance: List[str] = []
    mapped_zero_var: List[str] = []
    for col in validated:
        var = float(aligned_df[col].var())
        if var > 0:
            mapped_with_variance.append(col)
        else:
            mapped_zero_var.append(col)

    passed = len(mapped_with_variance) == N_MAPPED
    checks["mapped_features_have_variance"] = passed
    logger.info(
        "  [%s] %d mapped features have variance > 0 (%d/%d)",
        "PASS" if passed else "FAIL",
        N_MAPPED,
        len(mapped_with_variance),
        N_MAPPED,
    )
    if mapped_zero_var:
        logger.info("    Zero-variance mapped: %s", mapped_zero_var)

    # A7: 24 imputed features have zero variance
    imputed_zero_var: List[str] = []
    imputed_nonzero_var: List[str] = []
    for col in imputed_list:
        var = float(aligned_df[col].var())
        if var == 0.0:
            imputed_zero_var.append(col)
        else:
            imputed_nonzero_var.append(col)

    passed = len(imputed_zero_var) == N_IMPUTED
    checks["imputed_features_zero_variance"] = passed
    logger.info(
        "  [%s] %d imputed features have zero variance (%d/%d)",
        "PASS" if passed else "FAIL",
        N_IMPUTED,
        len(imputed_zero_var),
        N_IMPUTED,
    )
    if imputed_nonzero_var:
        logger.info("    Non-zero variance imputed: %s", imputed_nonzero_var)

    # A8: dstload uses WUSTL Normal median (not drate)
    dstload_median = medians["dstload"]
    dstload_values = aligned_df["dstload"].unique()
    passed = len(dstload_values) == 1 and np.isclose(
        float(dstload_values[0]), dstload_median, rtol=1e-5
    )
    checks["dstload_uses_median"] = passed
    logger.info(
        "  [%s] dstload uses WUSTL Normal median (%.6f), not drate",
        "PASS" if passed else "FAIL",
        dstload_median,
    )

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
    validated: Dict[str, Tuple[str, float]],
    imputed_list: List[str],
    medians: Dict[str, float],
    checks: Dict[str, bool],
) -> None:
    """Export aligned Parquet and alignment report JSON.

    Args:
        scaled_df: Scaled, aligned DataFrame.
        validated: Validated mappings {wustl_col: (cic_col, variance)}.
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

    mapped_report: Dict[str, Dict[str, Any]] = {}
    for wustl_col, (cic_col, variance) in validated.items():
        mapped_report[wustl_col] = {
            "cic_col": cic_col,
            "variance": round(variance, 6),
        }

    report: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_samples": len(scaled_df),
        "normal_samples": n_normal,
        "attack_samples": n_attack,
        "mapped_features": mapped_report,
        "excluded_mapping": EXCLUDED_MAPPING,
        "imputed_features": sorted(imputed_list),
        "imputed_count": N_IMPUTED,
        "mapped_count": N_MAPPED,
        "wustl_normal_medians": {k: round(v, 6) for k, v in sorted(medians.items())},
        "validation": {k: ("PASS" if v else "FAIL") for k, v in checks.items()},
    }

    report_path = output_dir / OUTPUT_REPORT
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    report_hash = verifier.compute_hash(report_path)
    logger.info("Exported %s — SHA-256: %s", report_path.name, report_hash)


def run() -> None:
    """Execute the CICIoMT2024 alignment pipeline v3."""
    logger.info("═══════════════════════════════════════════════════")
    logger.info("  CICIoMT2024 Cross-Dataset Alignment v3")
    logger.info("  (5 High-Confidence Mappings)")
    logger.info("═══════════════════════════════════════════════════")

    # 1-2. Load artifacts (with normalization)
    wustl_df, ciciomt_df, scaler = load_artifacts()

    # 3. Compute WUSTL Normal medians
    medians = compute_wustl_normal_medians(wustl_df)

    # 4. Validate mapped columns before using
    validated, demoted = validate_mapped_columns(ciciomt_df)

    # 5. Build aligned dataframe
    aligned_df, imputed_list = build_aligned_dataframe(ciciomt_df, medians, validated)

    # 6. Validation
    checks = validate_aligned(aligned_df, WUSTL_FEATURE_ORDER, validated, imputed_list, medians)

    # 7. Apply scaler
    scaled_df = apply_scaler(aligned_df, scaler)

    # 8. Export
    export_artifacts(scaled_df, validated, imputed_list, medians, checks)

    logger.info("═══════════════════════════════════════════════════")
    logger.info(
        "  Alignment v3 complete — %d samples, %d features " "(%d mapped, %d imputed)",
        len(scaled_df),
        N_FEATURES,
        len(validated),
        len(imputed_list),
    )
    logger.info("═══════════════════════════════════════════════════")


if __name__ == "__main__":
    run()
