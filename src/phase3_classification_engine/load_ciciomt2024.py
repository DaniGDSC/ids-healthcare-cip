# CICIoMT2024 labels derived from filenames.
# Files containing 'Benign_test' → label=0 (Normal).
# All other files → label=1 (Attack).
# Consistent with binary classification objective of RA-X-IoMT.

"""CICIoMT2024 dataset loader — scan, label, validate, export.

Loads the 21 CICIoMT2024 CSV files from ``data/external/``,
assigns binary labels based on filename convention, validates
integrity, and exports a single labelled Parquet file for
Phase 3 cross-dataset evaluation.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from src.phase0_dataset_analysis.phase0.security import IntegrityVerifier

# ── Constants ──────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data" / "external"
OUTPUT_DIR: Path = PROJECT_ROOT / "data" / "external"
OUTPUT_FILE: str = "ciciomt2024_labeled.parquet"

BENIGN_MARKER: str = "Benign_test"
LABEL_NORMAL: int = 0
LABEL_ATTACK: int = 1

EXPECTED_BENIGN_COUNT: int = 1
EXPECTED_ATTACK_COUNT: int = 20

RANDOM_STATE: int = 42

# ── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Functions ──────────────────────────────────────────────────────────


def discover_csv_files(directory: Path) -> List[Path]:
    """Scan directory for CSV files and return sorted list.

    Args:
        directory: Path to scan for .csv files.

    Returns:
        Sorted list of CSV file paths.

    Raises:
        FileNotFoundError: If directory does not exist or contains no CSVs.
    """
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    csv_files = sorted(directory.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {directory}")

    filenames = [f.name for f in csv_files]
    logger.info("Found %d files: %s", len(csv_files), filenames)
    return csv_files


def classify_files(
    csv_files: List[Path],
) -> Tuple[List[Path], List[Path]]:
    """Split files into benign and attack based on filename.

    Args:
        csv_files: List of CSV file paths.

    Returns:
        Tuple of (benign_files, attack_files).
    """
    benign: List[Path] = []
    attack: List[Path] = []

    for path in csv_files:
        if BENIGN_MARKER in path.name:
            benign.append(path)
        else:
            attack.append(path)

    return benign, attack


def load_and_label(csv_files: List[Path]) -> pd.DataFrame:
    """Load all CSVs, assign labels, concatenate, and shuffle.

    Args:
        csv_files: List of CSV file paths to load.

    Returns:
        Shuffled DataFrame with 'Label' column appended.
    """
    dfs: List[pd.DataFrame] = []

    for path in csv_files:
        df = pd.read_csv(path)
        label = LABEL_NORMAL if BENIGN_MARKER in path.name else LABEL_ATTACK
        label_name = "Normal" if label == LABEL_NORMAL else "Attack"
        df["Label"] = label
        logger.info(
            "  %s → Label %d (%s) — %d rows",
            path.name,
            label,
            label_name,
            len(df),
        )
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    # Deduplicate — raw PCAP captures may contain duplicate flows
    n_before = len(combined)
    combined = combined.drop_duplicates().reset_index(drop=True)
    n_dropped = n_before - len(combined)
    if n_dropped > 0:
        logger.warning(
            "Dropped %d duplicate rows (%.2f%% of %d)",
            n_dropped,
            n_dropped / n_before * 100,
            n_before,
        )

    combined = combined.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    n_normal = int((combined["Label"] == LABEL_NORMAL).sum())
    n_attack = int((combined["Label"] == LABEL_ATTACK).sum())
    logger.info(
        "Total samples: %d, Normal: %d, Attack: %d",
        len(combined),
        n_normal,
        n_attack,
    )

    return combined


def validate(
    df: pd.DataFrame,
    benign_files: List[Path],
    attack_files: List[Path],
) -> Dict[str, bool]:
    """Run validation assertions on the labelled dataset.

    Args:
        df: Combined labelled DataFrame.
        benign_files: List of benign file paths.
        attack_files: List of attack file paths.

    Returns:
        Dict mapping assertion name to PASS (True) / FAIL (False).
    """
    results: Dict[str, bool] = {}

    # A1: Exactly 1 benign file
    passed = len(benign_files) == EXPECTED_BENIGN_COUNT
    results["benign_file_count"] = passed
    logger.info(
        "  [%s] Benign file count: %d (expected %d)",
        "PASS" if passed else "FAIL",
        len(benign_files),
        EXPECTED_BENIGN_COUNT,
    )

    # A2: Exactly 20 attack files
    passed = len(attack_files) == EXPECTED_ATTACK_COUNT
    results["attack_file_count"] = passed
    logger.info(
        "  [%s] Attack file count: %d (expected %d)",
        "PASS" if passed else "FAIL",
        len(attack_files),
        EXPECTED_ATTACK_COUNT,
    )

    # A3: Label column exists
    passed = "Label" in df.columns
    results["label_column_exists"] = passed
    logger.info(
        "  [%s] 'Label' column exists",
        "PASS" if passed else "FAIL",
    )

    # A4: Labels only 0 or 1
    unique_labels = set(df["Label"].unique())
    passed = unique_labels.issubset({LABEL_NORMAL, LABEL_ATTACK})
    results["label_values_valid"] = passed
    logger.info(
        "  [%s] Label values ⊆ {0, 1}: %s",
        "PASS" if passed else "FAIL",
        unique_labels,
    )

    # A5: No duplicate rows
    n_dupes = int(df.duplicated().sum())
    passed = n_dupes == 0
    results["no_duplicate_rows"] = passed
    logger.info(
        "  [%s] No duplicate rows: %d duplicates found",
        "PASS" if passed else "FAIL",
        n_dupes,
    )

    return results


def export_parquet(df: pd.DataFrame, output_path: Path) -> str:
    """Export DataFrame to Parquet and compute SHA-256.

    Args:
        df: DataFrame to export.
        output_path: Destination path for the Parquet file.

    Returns:
        SHA-256 hex digest of the exported file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    verifier = IntegrityVerifier(output_path.parent)
    sha256 = verifier.compute_hash(output_path)
    logger.info(
        "Exported %s — SHA-256: %s",
        output_path.name,
        sha256,
    )
    return sha256


def run() -> None:
    """Execute the full CICIoMT2024 load-label-validate-export pipeline."""
    logger.info("═══════════════════════════════════════════════════")
    logger.info("  CICIoMT2024 Dataset Loader")
    logger.info("═══════════════════════════════════════════════════")

    # 1. Discover CSV files
    csv_files = discover_csv_files(DATA_DIR)

    # 2. Classify benign vs attack
    benign_files, attack_files = classify_files(csv_files)

    # 3. Load, label, concatenate, shuffle
    logger.info("── Loading and labelling ──")
    df = load_and_label(csv_files)

    # 4. Validate
    logger.info("── Validation checks ──")
    results = validate(df, benign_files, attack_files)

    all_passed = all(results.values())
    logger.info(
        "  Validation: %d/%d PASSED",
        sum(results.values()),
        len(results),
    )

    if not all_passed:
        failed = [k for k, v in results.items() if not v]
        logger.error("  FAILED assertions: %s", failed)
        sys.exit(1)

    # 5. Export
    logger.info("── Export ──")
    output_path = OUTPUT_DIR / OUTPUT_FILE
    export_parquet(df, output_path)

    logger.info("═══════════════════════════════════════════════════")
    logger.info("  CICIoMT2024 loader complete")
    logger.info("═══════════════════════════════════════════════════")


if __name__ == "__main__":
    run()
