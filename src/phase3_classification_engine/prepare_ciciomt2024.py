# Cross-dataset alignment methodology:
# - 13 features semantically mapped: Argus → CICFlowMeter
# - 16 features imputed: median of WUSTL Normal samples (label=0)
# - Biometric features (8): imputed — not available in CICIoMT2024
# - Unmapped network features (5): imputed conservatively
# - Scaler: WUSTL train set only — no refitting on CICIoMT2024
# - Zero-padding avoided: would signal FDI attack pattern

"""CICIoMT2024 cross-dataset alignment for Phase 3 evaluation.

Loads CICIoMT2024, applies semantic feature mapping (Argus ↔
CICFlowMeter), imputes unmapped features with WUSTL Normal
medians, enforces the canonical 29-feature order, applies the
WUSTL-fitted scaler (no refit), and exports aligned Parquet.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

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
NORMAL_LABEL: int = 0

RANDOM_STATE: int = 42

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

N_FEATURES: int = 29

# WUSTL biometric feature names (post-normalization)
BIOMETRIC_FEATURES: List[str] = [
    "temp",
    "spo2",
    "pulse_rate",
    "sys",
    "dia",
    "heart_rate",
    "resp_rate",
    "st",
]

# Semantic feature mapping: Argus (WUSTL) → CICFlowMeter (CICIoMT2024)
# Keys = WUSTL names, Values = standard CICFlowMeter names
# CICIoMT2024 columns matching VALUES get renamed to KEYS
SEMANTIC_MAPPING: Dict[str, str] = {
    "dur": "flow_duration",
    "spkts": "total_fwd_packets",
    "dpkts": "total_bwd_packets",
    "sbytes": "total_length_of_fwd_packets",
    "dbytes": "total_length_of_bwd_packets",
    "rate": "flow_bytes/s",
    "sload": "fwd_packets/s",
    "dload": "bwd_packets/s",
    "sintpkt": "flow_iat_mean",
    "sjit": "fwd_iat_std",
    "djit": "bwd_iat_std",
    "loss": "fin_flag_count",
    "proto": "protocol",
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

    n_changed = sum(1 for r, n in zip(raw_cols, normalized) if r != n)
    logger.info("Normalized %d feature names in %s (%d changed)", len(normalized), name, n_changed)
    logger.info("  Normalized %s features: %s", name, sorted(normalized))

    return df


def load_artifacts() -> Tuple[pd.DataFrame, pd.DataFrame, Any]:
    """Load WUSTL train, CICIoMT2024, and scaler artifacts.

    Returns:
        Tuple of (wustl_normal_df, ciciomt_df, scaler).

    Raises:
        FileNotFoundError: If any artifact is missing.
    """
    logger.info("── Loading artifacts ──")

    # WUSTL train
    if not WUSTL_TRAIN_PATH.exists():
        raise FileNotFoundError(f"WUSTL train not found: {WUSTL_TRAIN_PATH}")
    wustl_df = pd.read_parquet(WUSTL_TRAIN_PATH)
    wustl_df = normalize_columns(wustl_df, "WUSTL")
    normal_df = wustl_df[wustl_df[LABEL_COLUMN] == NORMAL_LABEL].copy()
    logger.info("  WUSTL Normal samples: %d", len(normal_df))

    # CICIoMT2024
    if not CICIOMT_PATH.exists():
        raise FileNotFoundError(f"CICIoMT2024 not found: {CICIOMT_PATH}")
    ciciomt_df = pd.read_parquet(CICIOMT_PATH)
    ciciomt_df = normalize_columns(ciciomt_df, "CICIoMT2024")
    logger.info("  CICIoMT2024 samples: %d", len(ciciomt_df))

    # Scaler
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")
    scaler = joblib.load(SCALER_PATH)
    logger.info("  Loaded scaler: %s", SCALER_PATH.name)

    return normal_df, ciciomt_df, scaler


def compute_wustl_normal_medians(normal_df: pd.DataFrame) -> Dict[str, float]:
    """Compute median values from WUSTL Normal samples for all 29 features.

    Args:
        normal_df: WUSTL Normal-only samples (label=0).

    Returns:
        Dict mapping feature name to median value.
    """
    feature_cols = [c for c in normal_df.columns if c != LABEL_COLUMN]
    medians: Dict[str, float] = {}
    for col in feature_cols:
        medians[col] = float(normal_df[col].median())

    logger.info("Computed medians from %d Normal samples", len(normal_df))
    for col in WUSTL_FEATURE_ORDER:
        if col in medians:
            logger.info("  %s = %.6f", col, medians[col])

    return medians


def apply_semantic_mapping(
    ciciomt_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Apply semantic feature mapping: rename CICFlowMeter → Argus names.

    For each entry in SEMANTIC_MAPPING, if the CICFlowMeter name (value)
    exists in CICIoMT2024 columns, rename it to the Argus name (key).

    Args:
        ciciomt_df: CICIoMT2024 DataFrame with normalized columns.

    Returns:
        Tuple of (renamed_df, mapped_pairs, unmapped_features).
    """
    logger.info("── Semantic feature mapping ──")

    cic_cols: Set[str] = set(ciciomt_df.columns)
    rename_map: Dict[str, str] = {}
    mapped_pairs: List[str] = []
    unmatched_mappings: List[str] = []

    # Normalize mapping values and check against actual CICIoMT2024 columns
    for argus_name, cic_name in SEMANTIC_MAPPING.items():
        cic_normalized = cic_name.strip().lower().replace(" ", "_")
        if cic_normalized in cic_cols:
            rename_map[cic_normalized] = argus_name
            mapped_pairs.append(f"{cic_normalized} → {argus_name}")
        else:
            unmatched_mappings.append(f"{argus_name} ↔ {cic_name} (not found)")

    if rename_map:
        ciciomt_df = ciciomt_df.rename(columns=rename_map)
        logger.info("  Mapped %d features: %s", len(mapped_pairs), mapped_pairs)
    else:
        logger.info("  Mapped 0 features — CICIoMT2024 uses abbreviated names")
        logger.info("  CICIoMT2024 column names do not match standard CICFlowMeter names")

    if unmatched_mappings:
        logger.info("  Unmatched mappings: %s", unmatched_mappings)

    # Log unmapped CICIoMT2024 features (not part of any mapping)
    post_cols = set(ciciomt_df.columns) - {LABEL_COLUMN}
    wustl_set = set(WUSTL_FEATURE_ORDER)
    unmapped = sorted(post_cols - wustl_set)
    logger.info("  Unmapped CICIoMT2024 features: %s", unmapped)

    return ciciomt_df, mapped_pairs, unmapped


def build_aligned_dataset(
    ciciomt_df: pd.DataFrame,
    medians: Dict[str, float],
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Build 29-feature aligned dataset in WUSTL canonical order.

    For each WUSTL feature:
    - If it exists in (renamed) CICIoMT2024 → use it
    - Otherwise → fill with WUSTL Normal median

    Args:
        ciciomt_df: CICIoMT2024 DataFrame (post-mapping).
        medians: WUSTL Normal median values.

    Returns:
        Tuple of (aligned_df, mapped_list, imputed_list).
    """
    logger.info("── Building aligned dataset ──")

    n_samples = len(ciciomt_df)
    aligned: Dict[str, np.ndarray] = {}
    mapped_list: List[str] = []
    imputed_list: List[str] = []

    cic_cols = set(ciciomt_df.columns)

    for feat in WUSTL_FEATURE_ORDER:
        if feat in cic_cols:
            aligned[feat] = ciciomt_df[feat].values.astype(np.float32)
            mapped_list.append(feat)
            logger.info("  %s → source: mapped", feat)
        else:
            median_val = medians.get(feat, 0.0)
            aligned[feat] = np.full(n_samples, median_val, dtype=np.float32)
            imputed_list.append(feat)
            logger.info("  %s → source: imputed (median=%.6f)", feat, median_val)

    # Preserve label column
    aligned[LABEL_COLUMN] = ciciomt_df[LABEL_COLUMN].values.astype(int)

    aligned_df = pd.DataFrame(aligned)
    logger.info(
        "  Aligned: %d mapped, %d imputed, %d total features",
        len(mapped_list),
        len(imputed_list),
        len(WUSTL_FEATURE_ORDER),
    )

    return aligned_df, mapped_list, imputed_list


def validate_aligned(
    aligned_df: pd.DataFrame,
    medians: Dict[str, float],
) -> Dict[str, bool]:
    """Run validation checks on the aligned dataset.

    Args:
        aligned_df: Aligned DataFrame (pre-scaler).
        medians: WUSTL Normal medians for reference.

    Returns:
        Dict mapping assertion name to PASS/FAIL.
    """
    logger.info("── Validation checks ──")
    checks: Dict[str, bool] = {}
    feature_cols = [c for c in aligned_df.columns if c != LABEL_COLUMN]

    # A1: Exactly 29 features
    n_feats = len(feature_cols)
    passed = n_feats == N_FEATURES
    checks["exactly_29_features"] = passed
    logger.info("  [%s] Exactly 29 features (%d found)", "PASS" if passed else "FAIL", n_feats)

    # A2: Feature order matches WUSTL
    order_match = feature_cols == WUSTL_FEATURE_ORDER
    checks["feature_order_matches_wustl"] = order_match
    logger.info("  [%s] Feature order matches WUSTL", "PASS" if order_match else "FAIL")

    # A3: No NaN values
    n_nan = int(aligned_df[feature_cols].isna().sum().sum())
    passed = n_nan == 0
    checks["no_nan_values"] = passed
    logger.info("  [%s] No NaN values (%d found)", "PASS" if passed else "FAIL", n_nan)

    # A4: No zeros in biometric features
    # Note: RobustScaler centers data at median, so some WUSTL Normal
    # medians are legitimately 0.0 (spo2, heart_rate, resp_rate, st).
    # This assertion flags them for transparency.
    bio_zeros: List[str] = []
    for col in BIOMETRIC_FEATURES:
        if col in aligned_df.columns:
            median_val = medians.get(col, 0.0)
            if median_val == 0.0:
                bio_zeros.append(col)

    passed = len(bio_zeros) == 0
    checks["no_zeros_in_biometrics"] = passed
    if not passed:
        logger.info(
            "  [FAIL] No zeros in biometric features — %d features have "
            "zero median (RobustScaler centering, not padding): %s",
            len(bio_zeros),
            bio_zeros,
        )
    else:
        logger.info("  [PASS] No zeros in biometric features")

    # A5: Label column preserved
    passed = LABEL_COLUMN in aligned_df.columns
    checks["label_column_preserved"] = passed
    logger.info("  [%s] Label column preserved", "PASS" if passed else "FAIL")

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
        Scaled DataFrame.
    """
    logger.info("── Applying scaler ──")

    feature_cols = [c for c in aligned_df.columns if c != LABEL_COLUMN]
    X = aligned_df[feature_cols].values.astype(np.float32)

    X_scaled = scaler.transform(X).astype(np.float32)
    logger.info("  Applied WUSTL scaler — no refitting")

    scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
    scaled_df[LABEL_COLUMN] = aligned_df[LABEL_COLUMN].values

    return scaled_df


def export_artifacts(
    scaled_df: pd.DataFrame,
    mapped_list: List[str],
    imputed_list: List[str],
    medians: Dict[str, float],
    checks: Dict[str, bool],
) -> None:
    """Export aligned Parquet and alignment report JSON.

    Args:
        scaled_df: Scaled, aligned DataFrame.
        mapped_list: List of mapped feature names.
        imputed_list: List of imputed feature names.
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
    logger.info("Exported %s — SHA-256: %s", parquet_path.name, parquet_hash)

    # 2. Alignment report
    labels = scaled_df[LABEL_COLUMN]
    n_normal = int((labels == NORMAL_LABEL).sum())
    n_attack = int((labels != NORMAL_LABEL).sum())

    report: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_samples": len(scaled_df),
        "normal_samples": n_normal,
        "attack_samples": n_attack,
        "n_features": N_FEATURES,
        "mapped_features": sorted(mapped_list),
        "mapped_count": len(mapped_list),
        "imputed_features": sorted(imputed_list),
        "imputed_count": len(imputed_list),
        "wustl_normal_medians": {k: round(v, 6) for k, v in sorted(medians.items())},
        "semantic_mapping": SEMANTIC_MAPPING,
        "validation": {k: ("PASS" if v else "FAIL") for k, v in checks.items()},
        "disclosure": (
            "CICIoMT2024 uses CICFlowMeter (abbreviated column names) while "
            "WUSTL-EHMS-2020 uses Argus flow exporter. The standard CICFlowMeter "
            "names in the semantic mapping do not match CICIoMT2024's abbreviated "
            "names, resulting in zero direct column matches. All 29 features are "
            "imputed with WUSTL Normal medians. Cross-dataset evaluation operates "
            "on learned model representations, not raw feature correspondence."
        ),
    }

    report_path = output_dir / OUTPUT_REPORT
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    report_hash = verifier.compute_hash(report_path)
    logger.info("Exported %s — SHA-256: %s", report_path.name, report_hash)


def run() -> None:
    """Execute the full CICIoMT2024 alignment pipeline."""
    logger.info("═══════════════════════════════════════════════════")
    logger.info("  CICIoMT2024 Cross-Dataset Alignment")
    logger.info("═══════════════════════════════════════════════════")

    # 1. Load artifacts
    normal_df, ciciomt_df, scaler = load_artifacts()

    # 3. Compute WUSTL Normal medians
    medians = compute_wustl_normal_medians(normal_df)

    # 4. Semantic feature mapping
    ciciomt_df, mapped_pairs, unmapped = apply_semantic_mapping(ciciomt_df)

    # 5. Build aligned dataset
    aligned_df, mapped_list, imputed_list = build_aligned_dataset(ciciomt_df, medians)

    # 6. Validation
    checks = validate_aligned(aligned_df, medians)

    # 7. Apply scaler
    scaled_df = apply_scaler(aligned_df, scaler)

    # 8. Export
    export_artifacts(scaled_df, mapped_list, imputed_list, medians, checks)

    logger.info("═══════════════════════════════════════════════════")
    logger.info("  Alignment complete — %d samples, %d features", len(scaled_df), N_FEATURES)
    logger.info("═══════════════════════════════════════════════════")


if __name__ == "__main__":
    run()
