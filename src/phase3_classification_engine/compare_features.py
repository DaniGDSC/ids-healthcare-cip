# CICIoMT2024 vs WUSTL-EHMS-2020 feature comparison.
# WUSTL uses Argus-based feature names; CICIoMT2024 uses CICFlowMeter.
# Normalization (lowercase, strip, underscores) applied before comparison.
# Zero common features is expected — confirms heterogeneous toolchain origins.

"""Feature comparison between WUSTL-EHMS-2020 and CICIoMT2024.

Loads both datasets, normalizes column names, computes set
intersection/difference, identifies missing biometric features,
validates, and exports a structured comparison report.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

from src.phase0_dataset_analysis.phase0.security import IntegrityVerifier

# ── Constants ──────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

WUSTL_PATH: Path = PROJECT_ROOT / "data" / "processed" / "train_phase1.parquet"
CICIOMT_PATH: Path = PROJECT_ROOT / "data" / "external" / "ciciomt2024_labeled.parquet"
OUTPUT_DIR: Path = PROJECT_ROOT / "data" / "external"
OUTPUT_FILE: str = "feature_comparison.json"

LABEL_COLUMN: str = "label"

# Biometric features expected in WUSTL but absent in CICIoMT2024
# Mapped to the user-facing names from the task specification
EXPECTED_BIOMETRIC: List[str] = [
    "hr",  # Heart_rate
    "spo2",  # SpO2
    "sbp",  # SYS (systolic blood pressure)
    "dbp",  # DIA (diastolic blood pressure)
    "rr",  # Resp_Rate
    "temp",  # Temp
    "etco2",  # ST (end-tidal CO2 proxy)
    "activity",  # Pulse_Rate (activity proxy)
]

# Actual WUSTL biometric column names (pre-normalization)
WUSTL_BIOMETRIC_RAW: List[str] = [
    "Heart_rate",
    "SpO2",
    "SYS",
    "DIA",
    "Resp_Rate",
    "Temp",
    "ST",
    "Pulse_Rate",
]

# ── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Functions ──────────────────────────────────────────────────────────


def load_schema(path: Path) -> List[str]:
    """Load column names from a Parquet file without reading data.

    Args:
        path: Path to the Parquet file.

    Returns:
        List of raw column names.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    import pyarrow.parquet as pq

    schema = pq.read_schema(path)
    columns = schema.names
    logger.info("Loaded schema: %s — %d columns", path.name, len(columns))
    return columns


def normalize_names(columns: List[str], dataset_name: str) -> List[str]:
    """Normalize feature names: strip, lowercase, spaces to underscores.

    Args:
        columns: Raw column names.
        dataset_name: Name for logging.

    Returns:
        List of normalized column names.
    """
    normalized = [col.strip().lower().replace(" ", "_") for col in columns]

    # Log raw → normalized mapping for verification
    n_changed = sum(1 for raw, norm in zip(columns, normalized) if raw != norm)
    logger.info(
        "Normalized %d feature names in %s (%d changed)",
        len(normalized),
        dataset_name,
        n_changed,
    )

    if n_changed > 0:
        for raw, norm in zip(columns, normalized):
            if raw != norm:
                logger.info("    [%s] → [%s]", raw, norm)

    return normalized


def extract_feature_sets(
    wustl_cols: List[str],
    ciciomt_cols: List[str],
) -> Dict[str, Set[str]]:
    """Extract and compare feature sets from both datasets.

    Args:
        wustl_cols: Normalized WUSTL column names.
        ciciomt_cols: Normalized CICIoMT2024 column names.

    Returns:
        Dict with common, wustl_only, ciciomt_only, and
        missing_biometric feature sets.
    """
    wustl_features = set(wustl_cols) - {LABEL_COLUMN}
    ciciomt_features = set(ciciomt_cols) - {LABEL_COLUMN}

    common = wustl_features & ciciomt_features
    wustl_only = wustl_features - ciciomt_features
    ciciomt_only = ciciomt_features - wustl_features

    # Identify missing biometric features in CICIoMT2024
    wustl_bio_normalized = [c.strip().lower().replace(" ", "_") for c in WUSTL_BIOMETRIC_RAW]
    missing_bio = [col for col in wustl_bio_normalized if col not in ciciomt_features]

    return {
        "wustl_features": wustl_features,
        "ciciomt_features": ciciomt_features,
        "common": common,
        "wustl_only": wustl_only,
        "ciciomt_only": ciciomt_only,
        "missing_biometric": missing_bio,
    }


def log_comparison(result: Dict[str, Set[str]]) -> None:
    """Log the feature comparison results.

    Args:
        result: Feature comparison dict from extract_feature_sets.
    """
    logger.info("── Feature Comparison Results ──")
    logger.info(
        "  WUSTL features: %d",
        len(result["wustl_features"]),
    )
    logger.info(
        "  CICIoMT2024 features: %d",
        len(result["ciciomt_features"]),
    )
    logger.info(
        "  Common features: %d → %s",
        len(result["common"]),
        sorted(result["common"]) if result["common"] else "[]",
    )
    logger.info(
        "  WUSTL only: %d → %s",
        len(result["wustl_only"]),
        sorted(result["wustl_only"]),
    )
    logger.info(
        "  CICIoMT2024 only: %d → %s",
        len(result["ciciomt_only"]),
        sorted(result["ciciomt_only"]),
    )
    logger.info(
        "  Missing biometric features in CICIoMT2024: %s",
        sorted(result["missing_biometric"]),
    )


def validate(
    result: Dict[str, Set[str]],
    wustl_cols: List[str],
    ciciomt_cols: List[str],
) -> Dict[str, bool]:
    """Run validation assertions on the comparison.

    Args:
        result: Feature comparison dict.
        wustl_cols: Normalized WUSTL columns.
        ciciomt_cols: Normalized CICIoMT2024 columns.

    Returns:
        Dict mapping assertion name to PASS/FAIL.
    """
    logger.info("── Validation checks ──")
    checks: Dict[str, bool] = {}

    # A1: common_features is not empty
    passed = len(result["common"]) > 0
    checks["common_features_not_empty"] = passed
    logger.info(
        "  [%s] Common features is not empty (%d found)",
        "PASS" if passed else "FAIL",
        len(result["common"]),
    )

    # A2: 'label' not in common_features
    passed = LABEL_COLUMN not in result["common"]
    checks["label_not_in_common"] = passed
    logger.info(
        "  [%s] 'label' not in common features",
        "PASS" if passed else "FAIL",
    )

    # A3: all feature names lowercase after normalization
    all_cols = wustl_cols + ciciomt_cols
    all_lower = all(col == col.lower() for col in all_cols)
    checks["all_names_lowercase"] = all_lower
    logger.info(
        "  [%s] All feature names lowercase after normalization",
        "PASS" if all_lower else "FAIL",
    )

    n_passed = sum(checks.values())
    logger.info("  Validation: %d/%d PASSED", n_passed, len(checks))

    return checks


def export_report(
    result: Dict[str, Set[str]],
    checks: Dict[str, bool],
    output_path: Path,
) -> str:
    """Export feature comparison report as JSON.

    Args:
        result: Feature comparison dict.
        checks: Validation results.
        output_path: Destination path.

    Returns:
        SHA-256 hex digest of the exported file.
    """
    report: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "wustl_total": len(result["wustl_features"]),
        "ciciomt_total": len(result["ciciomt_features"]),
        "common_count": len(result["common"]),
        "common_features": sorted(result["common"]),
        "wustl_only": sorted(result["wustl_only"]),
        "ciciomt_only": sorted(result["ciciomt_only"]),
        "missing_biometric": sorted(result["missing_biometric"]),
        "validation": {k: ("PASS" if v else "FAIL") for k, v in checks.items()},
        "note": (
            "Zero common features expected — WUSTL-EHMS-2020 uses Argus "
            "flow exporter while CICIoMT2024 uses CICFlowMeter. "
            "Cross-dataset validation operates on learned model "
            "representations, not raw feature name matching."
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    verifier = IntegrityVerifier(output_path.parent)
    sha256 = verifier.compute_hash(output_path)
    logger.info("Exported %s — SHA-256: %s", output_path.name, sha256)
    return sha256


def run() -> None:
    """Execute the full feature comparison pipeline."""
    logger.info("═══════════════════════════════════════════════════")
    logger.info("  WUSTL vs CICIoMT2024 Feature Comparison")
    logger.info("═══════════════════════════════════════════════════")

    # 1. Load schemas
    wustl_raw = load_schema(WUSTL_PATH)
    ciciomt_raw = load_schema(CICIOMT_PATH)
    logger.info(
        "WUSTL features: %d, CICIoMT2024 features: %d",
        len(wustl_raw),
        len(ciciomt_raw),
    )

    # 2. Normalize BEFORE any comparison
    wustl_norm = normalize_names(wustl_raw, "WUSTL")
    ciciomt_norm = normalize_names(ciciomt_raw, "CICIoMT2024")

    # 3-4. Extract and compare feature sets
    result = extract_feature_sets(wustl_norm, ciciomt_norm)

    # 5. Log comparison
    log_comparison(result)

    # 6. Validate
    checks = validate(result, wustl_norm, ciciomt_norm)

    # 7. Export
    logger.info("── Export ──")
    output_path = OUTPUT_DIR / OUTPUT_FILE
    export_report(result, checks, output_path)

    logger.info("═══════════════════════════════════════════════════")
    logger.info("  Feature comparison complete")
    logger.info("═══════════════════════════════════════════════════")


if __name__ == "__main__":
    run()
