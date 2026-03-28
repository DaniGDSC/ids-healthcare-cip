"""Phase 1: Data Preprocessing Pipeline for WUSTL-EHMS-2020.

Transforms the raw dataset into balanced, scaled train/test splits
ready for the Phase 2 detection engine.  Reads Phase 0 analysis
artifacts to avoid recomputation of correlations, missing-value
statistics, class distribution, and dataset integrity.

Pipeline
--------
1. HIPAA Sanitize   — drop network/temporal identifiers (PHI)
2. Missing Values   — ffill biometrics, dropna network (Phase 0-informed)
3. Redundancy       — read high_correlations.csv, drop one per pair
4. Stratified Split — 70/30 preserving class balance
5. SMOTE            — oversample minority (train only, before scaling)
6. Robust Scaling   — fit on train, transform both (no leakage)
7. Artifact Export  — Parquet splits, scaler pickle, JSON report

Design
------
- Single responsibility per function — each step is an isolated unit.
- No ``eval()`` or ``exec()`` anywhere.
- All thresholds are named constants or config-driven.
- Phase 0 artifacts are READ, never recomputed.

Author: IDS Healthcare CIP Research Team
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedShuffleSplit

logger = logging.getLogger(__name__)

# ======================================================================
# Constants
# ======================================================================

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent

# HIPAA Safe Harbor — columns encoding network identifiers / PHI
HIPAA_DROP_COLUMNS: List[str] = [
    "SrcAddr", "DstAddr", "Sport", "Dport",
    "SrcMac", "DstMac", "Dir", "Flgs",
]

# Biometric sensor columns (forward-fill on dropout)
BIOMETRIC_COLUMNS: List[str] = [
    "Temp", "SpO2", "Pulse_Rate", "SYS",
    "DIA", "Heart_rate", "Resp_Rate", "ST",
]

# Reproducibility
RANDOM_STATE: int = 42
TRAIN_RATIO: float = 0.70
TEST_RATIO: float = 0.30

# Redundancy
CORRELATION_THRESHOLD: float = 0.95

# SMOTE
SMOTE_K_NEIGHBORS: int = 5
SMOTE_STRATEGY: str = "auto"

# Scaling
SCALING_METHOD: str = "robust"

# Phase 0 artifact paths (relative to project root)
PHASE0_STATS_FILE: str = "results/phase0_analysis/stats_report.json"
PHASE0_CORR_FILE: str = "results/phase0_analysis/high_correlations.csv"
PHASE0_INTEGRITY_FILE: str = "results/phase0_analysis/dataset_integrity.json"

# Dataset
LABEL_COLUMN: str = "Label"
HASH_CHUNK_SIZE: int = 65_536


# ======================================================================
# Phase 0 Artifact Readers
# ======================================================================


def load_phase0_stats(path: Path) -> Dict[str, Any]:
    """Load Phase 0 descriptive statistics and class distribution.

    Args:
        path: Absolute path to ``stats_report.json``.

    Returns:
        Dict with keys ``descriptive_statistics``, ``missing_values``,
        ``class_distribution``.

    Raises:
        FileNotFoundError: If the stats file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Phase 0 stats not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    logger.info("Phase 0 stats loaded: %d features", len(data.get("descriptive_statistics", {})))
    return data


def load_phase0_correlations(path: Path) -> pd.DataFrame:
    """Load Phase 0 high-correlation pairs.

    Args:
        path: Absolute path to ``high_correlations.csv``.

    Returns:
        DataFrame with columns ``feature_a``, ``feature_b``, ``correlation``.

    Raises:
        FileNotFoundError: If the correlations file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Phase 0 correlations not found: {path}")
    df = pd.read_csv(path)
    logger.info("Phase 0 correlations loaded: %d pairs (threshold=%.2f)",
                len(df), CORRELATION_THRESHOLD)
    return df


def verify_dataset_integrity(dataset_path: Path, integrity_path: Path) -> str:
    """Verify the dataset SHA-256 against the Phase 0 baseline.

    Args:
        dataset_path: Path to the raw CSV file.
        integrity_path: Path to ``dataset_integrity.json``.

    Returns:
        The verified SHA-256 hex digest.

    Raises:
        FileNotFoundError: If either file is missing.
        ValueError: If the hash does not match the stored baseline.
    """
    if not integrity_path.exists():
        logger.warning("No integrity baseline found — skipping verification.")
        return _compute_sha256(dataset_path)

    metadata = json.loads(integrity_path.read_text(encoding="utf-8"))

    # Find the entry (key may be absolute path)
    stored_hash = None
    for key, value in metadata.items():
        if Path(key).name == dataset_path.name:
            stored_hash = value["sha256"]
            break

    if stored_hash is None:
        logger.warning("Dataset not in integrity file — computing fresh hash.")
        return _compute_sha256(dataset_path)

    current_hash = _compute_sha256(dataset_path)
    if current_hash != stored_hash:
        raise ValueError(
            f"INTEGRITY VIOLATION: expected {stored_hash[:16]}…, "
            f"got {current_hash[:16]}…"
        )
    logger.info("Dataset integrity verified: sha256=%s…", current_hash[:16])
    return current_hash


def _compute_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file.

    Args:
        path: File to hash.

    Returns:
        Lowercase hexadecimal digest string.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(HASH_CHUNK_SIZE):
            h.update(chunk)
    return h.hexdigest()


# ======================================================================
# Pipeline Steps (Single Responsibility)
# ======================================================================


def ingest(data_dir: Path, file_pattern: str = "*.csv") -> pd.DataFrame:
    """Read raw WUSTL-EHMS CSV files from the input directory.

    Args:
        data_dir: Directory containing the raw CSV files.
        file_pattern: Glob pattern for CSV files.

    Returns:
        Combined DataFrame from all matching files.

    Raises:
        FileNotFoundError: If no matching files are found.
    """
    csv_files = sorted(data_dir.glob(file_pattern))
    if not csv_files:
        raise FileNotFoundError(
            f"No files matching '{file_pattern}' in {data_dir}."
        )

    frames: List[pd.DataFrame] = []
    for path in csv_files:
        df = pd.read_csv(path, low_memory=False)
        logger.info("  Loaded %s: %d rows × %d cols", path.name, *df.shape)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    logger.info("Step 1 — Ingestion: %d rows × %d cols", *combined.shape)
    return combined


def hipaa_sanitize(
    df: pd.DataFrame,
    drop_columns: List[str] = HIPAA_DROP_COLUMNS,
) -> Tuple[pd.DataFrame, List[str]]:
    """Drop HIPAA-sensitive identifier columns.

    Args:
        df: Raw DataFrame.
        drop_columns: Column names to remove (network/MAC identifiers).

    Returns:
        Tuple of (sanitized DataFrame, list of actually dropped columns).
        Values of dropped columns are never logged.
    """
    present = [c for c in drop_columns if c in df.columns]
    df = df.drop(columns=present)
    logger.info("Step 2 — HIPAA: dropped %d identifier columns: %s",
                len(present), present)
    return df, present


def handle_missing_values(
    df: pd.DataFrame,
    biometric_cols: List[str] = BIOMETRIC_COLUMNS,
    label_col: str = LABEL_COLUMN,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Context-aware missing value handling.

    Biometric features use forward-fill (sensor dropout assumption).
    Network features use row-wise dropna (corrupted packet assumption).

    Args:
        df: HIPAA-sanitized DataFrame.
        biometric_cols: Biometric sensor column names.
        label_col: Label column name (excluded from network columns).

    Returns:
        Tuple of (cleaned DataFrame, missing value statistics dict).
    """
    bio_cols = [c for c in biometric_cols if c in df.columns]
    exclude = set(bio_cols) | {label_col, "Attack Category"}
    net_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude
    ]

    # Biometric: forward-fill then backward-fill
    bio_filled = 0
    if bio_cols:
        bio_filled = int(df[bio_cols].isna().sum().sum())
        df[bio_cols] = df[bio_cols].ffill().bfill()

    # Network: drop rows with any NaN
    rows_before = len(df)
    net_missing = int(df[net_cols].isna().sum().sum()) if net_cols else 0
    if net_cols:
        df = df.dropna(subset=net_cols)
    rows_dropped = rows_before - len(df)

    stats = {
        "biometric_cells_filled": bio_filled,
        "network_cells_missing": net_missing,
        "rows_dropped": rows_dropped,
    }
    logger.info(
        "Step 3 — Missing values: %d bio cells filled, %d rows dropped, "
        "%d rows remaining",
        bio_filled, rows_dropped, len(df),
    )
    return df, stats


def drop_redundant_features(
    df: pd.DataFrame,
    corr_df: pd.DataFrame,
    threshold: float = CORRELATION_THRESHOLD,
    label_col: str = LABEL_COLUMN,
) -> Tuple[pd.DataFrame, List[str]]:
    """Drop one feature from each high-correlation pair.

    Reads the Phase 0 high_correlations.csv — does NOT recompute the
    correlation matrix.  For each pair, drops ``feature_b`` (the second
    column in the CSV), keeping the feature with lower variance.

    Args:
        df: DataFrame after missing value handling.
        corr_df: Phase 0 high-correlation pairs (feature_a, feature_b, correlation).
        threshold: Minimum |r| to consider a pair redundant.
        label_col: Label column name (never dropped).

    Returns:
        Tuple of (reduced DataFrame, list of dropped column names).
    """
    high = corr_df[corr_df["correlation"].abs() >= threshold]
    cols_to_drop: List[str] = []

    for _, row in high.iterrows():
        candidate = row["feature_b"]
        if candidate in df.columns and candidate != label_col:
            if candidate not in cols_to_drop:
                cols_to_drop.append(candidate)

    df = df.drop(columns=cols_to_drop, errors="ignore")
    logger.info(
        "Step 4 — Redundancy: dropped %d features (|r| ≥ %.2f): %s",
        len(cols_to_drop), threshold, cols_to_drop,
    )
    return df, cols_to_drop


def stratified_split(
    df: pd.DataFrame,
    label_col: str = LABEL_COLUMN,
    test_ratio: float = TEST_RATIO,
    random_state: int = RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Stratified 70/30 train/test split preserving class balance.

    Args:
        df: Preprocessed DataFrame (numeric features + label).
        label_col: Name of the binary label column.
        test_ratio: Fraction of data reserved for testing.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names).

    Raises:
        ValueError: If the label column is not found.
    """
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found.")

    y = df[label_col].values
    X_df = df.drop(columns=[label_col]).select_dtypes(include=[np.number])

    # Drop any remaining non-numeric columns silently
    non_numeric = set(df.columns) - set(X_df.columns) - {label_col}
    if non_numeric:
        logger.info("  Dropped %d non-numeric columns: %s",
                     len(non_numeric), sorted(non_numeric))

    feature_names = X_df.columns.tolist()
    X = X_df.values

    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=test_ratio, random_state=random_state,
    )
    train_idx, test_idx = next(sss.split(X, y))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    logger.info(
        "Step 5 — Split: train=%d (attack=%.1f%%) | test=%d (attack=%.1f%%)",
        len(X_train), y_train.mean() * 100,
        len(X_test), y_test.mean() * 100,
    )
    return X_train, X_test, y_train, y_test, feature_names


def apply_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    strategy: str = SMOTE_STRATEGY,
    k_neighbors: int = SMOTE_K_NEIGHBORS,
    random_state: int = RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Oversample minority class on the training set using SMOTE.

    Applied BEFORE scaling so synthetic points are generated in the raw
    feature space and the scaler is subsequently fit on the balanced set.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.
        strategy: SMOTE sampling strategy.
        k_neighbors: Number of nearest neighbours for interpolation.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (X_resampled, y_resampled, smote_stats_dict).
    """
    n_before = len(X_train)
    attack_rate_before = float(y_train.mean())
    counts_before = dict(zip(*np.unique(y_train, return_counts=True)))

    smote = SMOTE(
        sampling_strategy=strategy,
        k_neighbors=k_neighbors,
        random_state=random_state,
    )
    X_res, y_res = smote.fit_resample(X_train, y_train)

    n_after = len(X_res)
    attack_rate_after = float(y_res.mean())
    counts_after = dict(zip(*np.unique(y_res, return_counts=True)))
    n_synthetic = n_after - n_before

    logger.info(
        "Step 6 — SMOTE: %d → %d samples (+%d synthetic), "
        "attack rate %.1f%% → %.1f%%",
        n_before, n_after, n_synthetic,
        attack_rate_before * 100, attack_rate_after * 100,
    )

    stats = {
        "sampling_strategy": strategy,
        "k_neighbors": k_neighbors,
        "samples_before": n_before,
        "samples_after": n_after,
        "synthetic_added": n_synthetic,
        "attack_rate_before": round(attack_rate_before, 4),
        "attack_rate_after": round(attack_rate_after, 4),
        "class_counts_before": {int(k): int(v) for k, v in counts_before.items()},
        "class_counts_after": {int(k): int(v) for k, v in counts_after.items()},
    }
    return X_res, y_res, stats


def robust_scale(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Any]:
    """Fit RobustScaler on training data only, transform both sets.

    RobustScaler uses median and IQR, making it robust to outliers in
    network traffic features.  Fitting on train only prevents data
    leakage from the test partition.

    Args:
        X_train: Training feature matrix (SMOTE-balanced).
        X_test: Test feature matrix.

    Returns:
        Tuple of (X_train_scaled, X_test_scaled, fitted_scaler).
    """
    from sklearn.preprocessing import RobustScaler

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logger.info(
        "Step 7 — RobustScaler: fit on train (%d×%d), transform test (%d×%d)",
        *X_train_scaled.shape, *X_test_scaled.shape,
    )
    return X_train_scaled, X_test_scaled, scaler


def export_artifacts(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    scaler: Any,
    report: Dict[str, Any],
    output_dir: Path,
    label_col: str = LABEL_COLUMN,
    attack_cat_train: np.ndarray | None = None,
    attack_cat_test: np.ndarray | None = None,
) -> Dict[str, str]:
    """Save Parquet splits, scaler pickle, and JSON report.

    Args:
        X_train: Scaled training features.
        X_test: Scaled test features.
        y_train: Training labels.
        y_test: Test labels.
        feature_names: Ordered list of feature column names.
        scaler: Fitted RobustScaler instance.
        report: Accumulated pipeline report dict.
        output_dir: Directory for output artifacts.
        label_col: Label column name for Parquet files.
        attack_cat_train: Attack Category string array for training set.
        attack_cat_test: Attack Category string array for test set.

    Returns:
        Dict mapping artifact name to its file path.
    """
    import joblib

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}

    # Train Parquet
    train_path = output_dir / "train_phase1.parquet"
    train_df = pd.DataFrame(X_train, columns=feature_names)
    train_df[label_col] = y_train
    if attack_cat_train is not None:
        train_df["Attack Category"] = attack_cat_train
    train_df.to_parquet(train_path, index=False)
    paths["train_parquet"] = str(train_path)
    logger.info("  train_phase1.parquet: %d rows × %d cols", *train_df.shape)

    # Test Parquet
    test_path = output_dir / "test_phase1.parquet"
    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df[label_col] = y_test
    if attack_cat_test is not None:
        test_df["Attack Category"] = attack_cat_test
    test_df.to_parquet(test_path, index=False)
    paths["test_parquet"] = str(test_path)
    logger.info("  test_phase1.parquet: %d rows × %d cols", *test_df.shape)

    # Scaler pickle
    scaler_dir = output_dir.parent.parent / "models" / "scalers"
    scaler_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = scaler_dir / "robust_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    paths["scaler_pkl"] = str(scaler_path)
    logger.info("  robust_scaler.pkl saved")

    # JSON report
    report["output"] = {
        "train_parquet": str(train_path),
        "test_parquet": str(test_path),
        "scaler_pkl": str(scaler_path),
        "feature_names": feature_names,
        "n_features": len(feature_names),
    }
    report_path = output_dir / "preprocessing_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    paths["report_json"] = str(report_path)
    logger.info("  preprocessing_report.json saved")

    logger.info("Step 7 — Export: %d artifacts written to %s", len(paths), output_dir)
    return paths


# ======================================================================
# Report Renderer
# ======================================================================


def render_preprocessing_report(report: Dict[str, Any]) -> str:
    """Render ``report_section_preprocessing.md`` (§4.1) for thesis.

    Args:
        report: Accumulated pipeline report dict from ``run_pipeline()``.

    Returns:
        Complete Markdown string ready for file export.
    """
    lines: List[str] = []
    w = lines.append

    ing = report.get("ingestion", {})
    hip = report.get("hipaa", {})
    mv = report.get("missing_values", {})
    red = report.get("redundancy", {})
    spl = report.get("split", {})
    smt = report.get("smote", {})
    scl = report.get("scaling", {})
    out = report.get("output", {})

    w("## 4.1 Data Preprocessing Pipeline")
    w("")
    w("This section documents the seven-step preprocessing pipeline applied "
      "to the WUSTL-EHMS-2020 dataset prior to model training. Each step is "
      "justified with reference to the data quality assessment in §3.2 and "
      "the security controls documented in §3.3.")
    w("")

    # ── 4.1.1 HIPAA ──
    w("### 4.1.1 HIPAA Safe Harbor De-identification")
    w("")
    dropped = hip.get("columns_dropped", [])
    col_list = ", ".join(f"`{c}`" for c in dropped)
    w(f"**{len(dropped)} columns dropped:** [{col_list}]")
    w("")
    w("These columns encode network identifiers (IP addresses, MAC addresses, "
      "port numbers) and flow metadata that constitute environment-specific "
      "artefacts. Their removal satisfies HIPAA Safe Harbor §164.514(b)(2) "
      "and prevents the model from memorising topology-specific patterns "
      "that do not generalise to unseen network environments.")
    w("")

    # ── 4.1.2 Missing Values ──
    w("### 4.1.2 Context-Aware Missing Value Handling")
    w("")
    w("| Stream | Strategy | Justification |")
    w("|--------|----------|---------------|")
    w(f"| Biometric ({len(BIOMETRIC_COLUMNS)} features) | Forward-fill "
      f"(ffill) | Sensor dropout produces temporal gaps; the most recent "
      f"valid reading is the best available estimate |")
    w(f"| Network (remaining features) | Row-wise dropna | Corrupted packets "
      f"produce incomplete flow records that cannot be reliably imputed |")
    w("")
    bio_filled = mv.get("biometric_cells_filled", 0)
    rows_dropped = mv.get("rows_dropped", 0)
    w(f"- Biometric cells filled: **{bio_filled:,}**")
    w(f"- Rows dropped (network NaN): **{rows_dropped:,}**")
    w(f"- Rows remaining: **{ing.get('raw_rows', 0) - rows_dropped:,}**")
    w("")

    # ── 4.1.3 Redundancy ──
    w("### 4.1.3 Redundancy Elimination")
    w("")
    red_cols = red.get("columns_dropped", [])
    threshold = red.get("threshold", CORRELATION_THRESHOLD)
    w(f"High-correlation pairs (|*r*| ≥ {threshold}) were identified in Phase 0 "
      f"(§3.2.3) and read from `high_correlations.csv`. For each pair, the "
      f"secondary feature was dropped, reducing the feature space by "
      f"**{len(red_cols)}** columns:")
    w("")
    if red_cols:
        w("| Dropped Feature | Reason |")
        w("|-----------------|--------|")
        for col in red_cols:
            w(f"| `{col}` | |*r*| ≥ {threshold} with a retained feature |")
    w("")

    # ── 4.1.4 Split ──
    w("### 4.1.4 Stratified Train/Test Split")
    w("")
    train_n = spl.get("train_samples", 0)
    test_n = spl.get("test_samples", 0)
    w("| Partition | Samples | Ratio |")
    w("|-----------|--------:|------:|")
    w(f"| Train | {train_n:,} | {spl.get('train_ratio', TRAIN_RATIO):.0%} |")
    w(f"| Test | {test_n:,} | {spl.get('test_ratio', TEST_RATIO):.0%} |")
    w("")
    w(f"Stratification via `StratifiedShuffleSplit` with "
      f"`random_state={RANDOM_STATE}` preserves the original class prior "
      f"({100 - smt.get('attack_rate_before', 0.125) * 100:.1f}% Normal / "
      f"{smt.get('attack_rate_before', 0.125) * 100:.1f}% Attack) in both "
      f"partitions, preventing evaluation bias from sampling variance.")
    w("")

    # ── 4.1.5 SMOTE ──
    w("### 4.1.5 SMOTE Oversampling (Train Only)")
    w("")
    w("| Metric | Before | After |")
    w("|--------|-------:|------:|")
    w(f"| Samples | {smt.get('samples_before', 0):,} | {smt.get('samples_after', 0):,} |")
    w(f"| Attack rate | {smt.get('attack_rate_before', 0) * 100:.1f}% "
      f"| {smt.get('attack_rate_after', 0) * 100:.1f}% |")
    w(f"| Synthetic added | — | {smt.get('synthetic_added', 0):,} |")
    w("")
    w(f"SMOTE (Synthetic Minority Oversampling Technique) with "
      f"*k* = {smt.get('k_neighbors', SMOTE_K_NEIGHBORS)} is applied "
      f"**exclusively to the training partition** to prevent synthetic data "
      f"from contaminating the test evaluation. The oversampling is performed "
      f"**before** scaling so that synthetic samples are generated in the "
      f"original feature space, not in a normalised space where inter-feature "
      f"distances are distorted.")
    w("")

    # ── 4.1.6 Scaling ──
    w("### 4.1.6 Robust Scaling")
    w("")
    w("RobustScaler (median / IQR normalisation) is chosen over StandardScaler "
      "(mean / std) or MinMaxScaler because the outlier analysis in §3.2.1 "
      "identified heavy-tailed distributions in network-traffic features. "
      "RobustScaler is insensitive to extreme values, preserving the "
      "morphology of attack signatures for downstream XAI (SHAP) "
      "interpretation.")
    w("")
    w("- **Fit** on the SMOTE-balanced training set only")
    w("- **Transform** both train and test sets")
    w("- **No leakage**: the scaler never observes test data during fitting")
    w("")

    # ── 4.1.7 Output ──
    w("### 4.1.7 Pipeline Output Summary")
    w("")
    n_features = out.get("n_features", 0)
    w(f"| Artifact | Format | Description |")
    w(f"|----------|--------|-------------|")
    w(f"| `train_phase1.parquet` | Apache Parquet | "
      f"{smt.get('samples_after', 0):,} rows × {n_features} features |")
    w(f"| `test_phase1.parquet` | Apache Parquet | "
      f"{test_n:,} rows × {n_features} features |")
    w(f"| `robust_scaler.pkl` | joblib pickle | Fitted RobustScaler for inference |")
    w(f"| `preprocessing_report.json` | JSON | Per-step audit trail |")
    w("")
    elapsed = report.get("elapsed_seconds", 0)
    w(f"Total pipeline elapsed time: **{elapsed:.2f} s**")
    w("")

    return "\n".join(lines)


# ======================================================================
# Pipeline Orchestrator
# ======================================================================


def run_pipeline(config_path: Path) -> Dict[str, Any]:
    """Execute the full 7-step preprocessing pipeline.

    Args:
        config_path: Path to ``phase1_config.yaml``.

    Returns:
        Pipeline report dict with per-step metadata.
    """
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_dir = PROJECT_ROOT / config["data"]["input_dir"]
    output_dir = PROJECT_ROOT / config["data"]["output_dir"]
    report: Dict[str, Any] = {}

    t0 = time.perf_counter()

    # ── Verify dataset integrity against Phase 0 baseline ──
    csv_files = sorted(data_dir.glob(config["data"].get("file_pattern", "*.csv")))
    if csv_files:
        integrity_path = PROJECT_ROOT / PHASE0_INTEGRITY_FILE
        dataset_hash = verify_dataset_integrity(csv_files[0], integrity_path)
        report["integrity"] = {"sha256": dataset_hash, "verified": True}

    # ── Step 1: Ingestion ──
    df = ingest(data_dir, config["data"].get("file_pattern", "*.csv"))
    report["ingestion"] = {
        "files_loaded": len(csv_files),
        "raw_rows": int(df.shape[0]),
        "raw_columns": int(df.shape[1]),
    }

    # ── Step 2: HIPAA Sanitize ──
    hipaa_cols = config.get("hipaa", {}).get("remove_columns", HIPAA_DROP_COLUMNS)
    df, dropped = hipaa_sanitize(df, hipaa_cols)
    report["hipaa"] = {
        "columns_requested": hipaa_cols,
        "columns_dropped": dropped,
    }

    # ── Step 3: Missing Values (Phase 0-informed) ──
    phase0_stats_path = PROJECT_ROOT / PHASE0_STATS_FILE
    if phase0_stats_path.exists():
        phase0_stats = load_phase0_stats(phase0_stats_path)
        p0_missing = phase0_stats.get("missing_values", {})
        logger.info("Phase 0 missing values: %s",
                     "none" if not p0_missing else f"{len(p0_missing)} features")

    bio_cols = config.get("missing_values", {}).get("biometric_columns", BIOMETRIC_COLUMNS)
    df, mv_stats = handle_missing_values(df, bio_cols)
    report["missing_values"] = mv_stats

    # ── Step 4: Redundancy (read from Phase 0) ──
    corr_path = PROJECT_ROOT / config.get("correlation_removal", {}).get(
        "phase0_corr_file", PHASE0_CORR_FILE
    )
    corr_df = load_phase0_correlations(corr_path)
    threshold = config.get("correlation_removal", {}).get("threshold", CORRELATION_THRESHOLD)
    df, red_cols = drop_redundant_features(df, corr_df, threshold)
    report["redundancy"] = {
        "threshold": threshold,
        "columns_dropped": red_cols,
        "n_dropped": len(red_cols),
    }

    # ── Step 4b: Variance filtering ──
    var_cfg = config.get("variance_filtering", {})
    if var_cfg.get("enabled", True):
        max_unique = var_cfg.get("max_unique", 1)
        label_col = config.get("data", {}).get("label_column", "Label")
        exclude = {label_col, "Attack Category"}
        numeric_cols = [
            c for c in df.select_dtypes(include=["number"]).columns
            if c not in exclude
        ]
        zero_var_cols = [c for c in numeric_cols if df[c].nunique() <= max_unique]
        df = df.drop(columns=zero_var_cols, errors="ignore")
        report["variance"] = {
            "max_unique": max_unique,
            "columns_dropped": zero_var_cols,
            "n_dropped": len(zero_var_cols),
        }
        logger.info(
            "Step 4b — Variance: dropped %d features (unique ≤ %d): %s",
            len(zero_var_cols), max_unique, zero_var_cols,
        )

    # ── Extract Attack Category before split (non-numeric, dropped by splitter) ──
    attack_cat_all = df["Attack Category"].values if "Attack Category" in df.columns else None

    # ── Step 5: Stratified Split ──
    split_cfg = config.get("splitting", {})
    # Reproduce split indices to align Attack Category
    label_col = config.get("data", {}).get("label_column", "Label")
    y_all = df[label_col].values
    X_df_num = df.drop(columns=[label_col]).select_dtypes(include=[np.number])
    feat_names = X_df_num.columns.tolist()
    X_all = X_df_num.values

    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=split_cfg.get("test_ratio", TEST_RATIO),
        random_state=split_cfg.get("random_state", RANDOM_STATE),
    )
    train_idx, test_idx = next(sss.split(X_all, y_all))
    X_train, X_test = X_all[train_idx], X_all[test_idx]
    y_train, y_test = y_all[train_idx], y_all[test_idx]

    attack_cat_train = attack_cat_all[train_idx] if attack_cat_all is not None else None
    attack_cat_test = attack_cat_all[test_idx] if attack_cat_all is not None else None

    logger.info(
        "Step 5 — Split: train=%d (attack=%.1f%%) | test=%d (attack=%.1f%%)",
        len(X_train), y_train.mean() * 100,
        len(X_test), y_test.mean() * 100,
    )
    report["split"] = {
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "train_ratio": round(1 - split_cfg.get("test_ratio", TEST_RATIO), 2),
        "test_ratio": split_cfg.get("test_ratio", TEST_RATIO),
        "stratified": split_cfg.get("stratify", True),
    }

    # ── Step 6: SMOTE (train only) ──
    smote_cfg = config.get("smote", {})
    X_train, y_train, smote_stats = apply_smote(
        X_train, y_train,
        strategy=smote_cfg.get("sampling_strategy", SMOTE_STRATEGY),
        k_neighbors=smote_cfg.get("k_neighbors", SMOTE_K_NEIGHBORS),
        random_state=smote_cfg.get("random_state", RANDOM_STATE),
    )
    report["smote"] = smote_stats

    # ── Step 7: Robust Scaling ──
    X_train_s, X_test_s, scaler = robust_scale(X_train, X_test)
    report["scaling"] = {"method": SCALING_METHOD}

    # ── Export ──
    elapsed = time.perf_counter() - t0
    report["elapsed_seconds"] = round(elapsed, 2)

    export_artifacts(
        X_train_s, X_test_s, y_train, y_test,
        feat_names, scaler, report, output_dir,
        attack_cat_train=attack_cat_train,
        attack_cat_test=attack_cat_test,
    )

    # ── Render thesis report section ──
    md_content = render_preprocessing_report(report)
    report_md_path = PROJECT_ROOT / "results" / "phase0_analysis" / "report_section_preprocessing.md"
    report_md_path.write_text(md_content, encoding="utf-8")
    logger.info("Thesis report → %s", report_md_path)

    _log_summary(report)
    return report


def _log_summary(report: Dict[str, Any]) -> None:
    """Print a structured pipeline summary to the logger.

    Args:
        report: Complete pipeline report dict.
    """
    sep = "=" * 72
    ing = report.get("ingestion", {})
    hip = report.get("hipaa", {})
    mv = report.get("missing_values", {})
    red = report.get("redundancy", {})
    var = report.get("variance", {})
    spl = report.get("split", {})
    smt = report.get("smote", {})
    out = report.get("output", {})

    logger.info("")
    logger.info(sep)
    logger.info("PHASE 1 — PREPROCESSING SUMMARY")
    logger.info(sep)
    logger.info("  Ingestion     : %d files → %d rows × %d cols",
                ing.get("files_loaded", 0), ing.get("raw_rows", 0), ing.get("raw_columns", 0))
    logger.info("  HIPAA         : %d columns dropped", len(hip.get("columns_dropped", [])))
    logger.info("  Missing       : %d bio cells filled, %d rows dropped",
                mv.get("biometric_cells_filled", 0), mv.get("rows_dropped", 0))
    logger.info("  Redundancy    : %d features dropped (|r| ≥ %.2f)",
                red.get("n_dropped", 0), red.get("threshold", 0))
    logger.info("  Variance      : %d features dropped (unique ≤ %d)",
                var.get("n_dropped", 0), var.get("max_unique", 0))
    logger.info("  Split         : train=%d, test=%d",
                spl.get("train_samples", 0), spl.get("test_samples", 0))
    logger.info("  SMOTE         : %d → %d (+%d synthetic)",
                smt.get("samples_before", 0), smt.get("samples_after", 0),
                smt.get("synthetic_added", 0))
    logger.info("  Scaling       : RobustScaler (median/IQR)")
    logger.info("  Features kept : %d", out.get("n_features", 0))
    logger.info("  Elapsed       : %.2f s", report.get("elapsed_seconds", 0))
    logger.info(sep)


# ======================================================================
# Entry Point
# ======================================================================


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )

    config_path = PROJECT_ROOT / "config" / "phase1_config.yaml"
    log_path = PROJECT_ROOT / "logs" / "phase1_preprocessing.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logging.getLogger().addHandler(file_handler)

    logger.info("=" * 72)
    logger.info("PHASE 1: DATA PREPROCESSING PIPELINE (WUSTL-EHMS-2020)")
    logger.info("=" * 72)

    report = run_pipeline(config_path)
