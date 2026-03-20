# Extract and display normalized feature names from WUSTL-EHMS-2020
# and CICIoMT2024 datasets for manual semantic mapping review.

"""Feature extraction for cross-dataset mapping review.

Loads both datasets, normalizes column names, prints indexed
feature lists, validates counts, and exports to JSON.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

from src.phase0_dataset_analysis.phase0.security import IntegrityVerifier

# ── Constants ──────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
WUSTL_PATH: Path = PROJECT_ROOT / "data" / "processed" / "train_phase1.parquet"
CICIOMT_PATH: Path = PROJECT_ROOT / "data" / "external" / "ciciomt2024_labeled.parquet"
OUTPUT_DIR: Path = PROJECT_ROOT / "data" / "external"
OUTPUT_FILE: str = "features_list.json"

LABEL_COLUMN: str = "label"
EXPECTED_WUSTL_COUNT: int = 29
EXPECTED_CICIOMT_COUNT: int = 45

# ── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Functions ──────────────────────────────────────────────────────────


def load_normalized_columns(path: Path, name: str) -> List[str]:
    """Load column names from Parquet, normalize, exclude label.

    Args:
        path: Path to the Parquet file.
        name: Dataset name for logging.

    Returns:
        Normalized feature names (excluding label).

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    import pyarrow.parquet as pq

    raw_cols = pq.read_schema(path).names
    normalized = [c.strip().lower().replace(" ", "_") for c in raw_cols]
    features = [c for c in normalized if c != LABEL_COLUMN]

    logger.info("Loaded %s: %d columns → %d features", name, len(raw_cols), len(features))
    return features


def print_feature_list(features: List[str], name: str) -> None:
    """Print indexed feature list to console.

    Args:
        features: List of normalized feature names.
        name: Dataset name for header.
    """
    print(f"\n{name} features ({len(features)}):")
    for i, feat in enumerate(features):
        print(f"  [{i:2d}]  {feat}")


def validate(
    wustl: List[str],
    ciciomt: List[str],
) -> Dict[str, bool]:
    """Run validation assertions on extracted feature lists.

    Args:
        wustl: WUSTL feature names.
        ciciomt: CICIoMT2024 feature names.

    Returns:
        Dict mapping assertion name to PASS/FAIL.
    """
    checks: Dict[str, bool] = {}

    passed = len(wustl) == EXPECTED_WUSTL_COUNT
    checks["wustl_count"] = passed
    logger.info(
        "  [%s] WUSTL count == %d (%d found)",
        "PASS" if passed else "FAIL",
        EXPECTED_WUSTL_COUNT,
        len(wustl),
    )

    passed = len(ciciomt) == EXPECTED_CICIOMT_COUNT
    checks["ciciomt_count"] = passed
    logger.info(
        "  [%s] CICIoMT2024 count == %d (%d found)",
        "PASS" if passed else "FAIL",
        EXPECTED_CICIOMT_COUNT,
        len(ciciomt),
    )

    all_lower = all(c == c.lower() for c in wustl + ciciomt)
    checks["all_lowercase"] = all_lower
    logger.info("  [%s] All names lowercase", "PASS" if all_lower else "FAIL")

    no_label = LABEL_COLUMN not in wustl and LABEL_COLUMN not in ciciomt
    checks["label_excluded"] = no_label
    logger.info("  [%s] 'label' not in either list", "PASS" if no_label else "FAIL")

    logger.info("  Validation: %d/%d PASSED", sum(checks.values()), len(checks))
    return checks


def export_json(
    wustl: List[str],
    ciciomt: List[str],
    output_path: Path,
) -> None:
    """Export feature lists to JSON.

    Args:
        wustl: WUSTL feature names.
        ciciomt: CICIoMT2024 feature names.
        output_path: Destination path.
    """
    report = {
        "wustl_features": wustl,
        "ciciomt_features": ciciomt,
        "wustl_count": len(wustl),
        "ciciomt_count": len(ciciomt),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    verifier = IntegrityVerifier(output_path.parent)
    sha256 = verifier.compute_hash(output_path)
    logger.info("Exported %s — SHA-256: %s", output_path.name, sha256)


def run() -> None:
    """Execute the feature extraction pipeline."""
    logger.info("═══════════════════════════════════════════════════")
    logger.info("  Feature Extraction for Mapping Review")
    logger.info("═══════════════════════════════════════════════════")

    wustl = load_normalized_columns(WUSTL_PATH, "WUSTL")
    ciciomt = load_normalized_columns(CICIOMT_PATH, "CICIoMT2024")

    print_feature_list(wustl, "WUSTL")
    print_feature_list(ciciomt, "CICIoMT2024")

    print()
    logger.info("── Validation ──")
    validate(wustl, ciciomt)

    logger.info("── Export ──")
    export_json(wustl, ciciomt, OUTPUT_DIR / OUTPUT_FILE)

    logger.info("═══════════════════════════════════════════════════")
    logger.info("  Extraction complete")
    logger.info("═══════════════════════════════════════════════════")


if __name__ == "__main__":
    run()
