"""Preprocessing report renderer for thesis defence / IEEE Q1.

Renders ``report_section_preprocessing.md`` (§4.1) from the
accumulated pipeline report dict.  No computation — pure presentation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Biometric column count (fixed for WUSTL-EHMS-2020)
_N_BIOMETRIC: int = 8


def render_preprocessing_report(report: Dict[str, Any]) -> str:
    """Render the preprocessing thesis report section.

    Args:
        report: Accumulated pipeline report from ``PreprocessingPipeline``.

    Returns:
        Complete Markdown string.
    """
    lines: List[str] = []
    w = lines.append

    ing = report.get("ingestion", {})
    hip = report.get("hipaa", {})
    mv = report.get("missing_values", {})
    red = report.get("redundancy", {})
    spl = report.get("split", {})
    smt = report.get("smote", {})
    out = report.get("output", {})

    raw_rows = ing.get("raw_rows", 0)
    raw_cols = ing.get("raw_columns", 0)

    w("## 4.1 Data Preprocessing Pipeline")
    w("")
    w("This section documents the seven-step preprocessing pipeline applied "
      "to the WUSTL-EHMS-2020 dataset prior to model training. Each step is "
      "justified with reference to the data quality assessment in §3.2 and "
      "the security controls documented in §3.3.")
    w("")

    # ── Pipeline steps table ──
    _steps_table(w, ing, hip, mv, red, spl, smt, out)

    # ── Feature reduction table ──
    _feature_reduction_table(w, raw_cols, hip, red)

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
    w(f"| Biometric ({_N_BIOMETRIC} features) | Forward-fill (ffill) "
      f"| Sensor dropout produces temporal gaps; the most recent valid "
      f"reading is the best available estimate |")
    w(f"| Network (remaining features) | Row-wise dropna "
      f"| Corrupted packets produce incomplete flow records that cannot "
      f"be reliably imputed |")
    w("")
    bio_filled = mv.get("biometric_cells_filled", 0)
    rows_dropped = mv.get("rows_dropped", 0)
    w(f"- Biometric cells filled: **{bio_filled:,}**")
    w(f"- Rows dropped (network NaN): **{rows_dropped:,}**")
    w(f"- Rows remaining: **{raw_rows - rows_dropped:,}**")
    w("")

    # ── 4.1.3 Redundancy ──
    w("### 4.1.3 Redundancy Elimination")
    w("")
    red_cols = red.get("columns_dropped", [])
    threshold = red.get("threshold", 0.95)
    w(f"High-correlation pairs (|*r*| ≥ {threshold}) were identified in "
      f"Phase 0 (§3.2.3) and read from `high_correlations.csv` — the "
      f"correlation matrix was **not** recomputed. For each pair, the "
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
    random_state = report.get("random_state", 42)
    w("| Partition | Samples | Ratio |")
    w("|-----------|--------:|------:|")
    w(f"| Train | {train_n:,} | {spl.get('train_ratio', 0.70):.0%} |")
    w(f"| Test | {test_n:,} | {spl.get('test_ratio', 0.30):.0%} |")
    w("")
    w(f"Stratification via `StratifiedShuffleSplit` with "
      f"`random_state={random_state}` preserves the original class prior "
      f"in both partitions, preventing evaluation bias from sampling variance.")
    w("")

    # ── 4.1.5 SMOTE ──
    _smote_section(w, smt)

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
    train_fitted = smt.get("samples_after", train_n)
    w(f"Scaler fitted exclusively on training set (n={train_fitted:,}). "
      f"Test set transformed without refitting — preventing information "
      f"leakage from test distribution.")
    w("")

    # ── 4.1.7 Output ──
    w("### 4.1.7 Pipeline Output Summary")
    w("")
    n_features = out.get("n_features", 0)
    w("| Artifact | Format | Description |")
    w("|----------|--------|-------------|")
    w(f"| `train_phase1.parquet` | Apache Parquet | "
      f"{smt.get('samples_after', 0):,} rows × {n_features} features |")
    w(f"| `test_phase1.parquet` | Apache Parquet | "
      f"{test_n:,} rows × {n_features} features |")
    w(f"| `robust_scaler.pkl` | joblib pickle | "
      f"Fitted RobustScaler for inference |")
    w(f"| `preprocessing_report.json` | JSON | Per-step audit trail |")
    w("")
    elapsed = report.get("elapsed_seconds", 0)
    w(f"Total pipeline elapsed time: **{elapsed:.2f} s**")
    w("")

    content = "\n".join(lines)
    logger.info("Preprocessing report rendered: %d lines", len(lines))
    return content


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _steps_table(w, ing, hip, mv, red, spl, smt, out) -> None:
    """Render the pipeline steps summary table."""
    raw_shape = f"{ing.get('raw_rows', 0):,} × {ing.get('raw_columns', 0)}"
    n_hip = hip.get("n_dropped", len(hip.get("columns_dropped", [])))
    after_hipaa = f"{ing.get('raw_rows', 0):,} × {ing.get('raw_columns', 0) - n_hip}"
    rows_after_mv = mv.get("rows_remaining", ing.get("raw_rows", 0))
    after_mv = f"{rows_after_mv:,} × {ing.get('raw_columns', 0) - n_hip}"
    n_red = red.get("n_dropped", 0)
    after_red_cols = ing.get("raw_columns", 0) - n_hip - n_red
    after_red = f"{rows_after_mv:,} × {after_red_cols}"
    train_n = spl.get("train_samples", 0)
    test_n = spl.get("test_samples", 0)
    n_feat = out.get("n_features", after_red_cols)

    w("### Pipeline Steps Overview")
    w("")
    w("| Step | Input Shape | Output Shape | Notes |")
    w("|------|-------------|--------------|-------|")
    w(f"| 1. Ingestion | — | {raw_shape} | Raw WUSTL-EHMS CSV |")
    w(f"| 2. HIPAA | {raw_shape} | {after_hipaa} | {n_hip} identifier cols dropped |")
    w(f"| 3. Missing | {after_hipaa} | {after_mv} | ffill bio, dropna net |")
    w(f"| 4. Redundancy | {after_mv} | {after_red} | {n_red} correlated features dropped |")
    w(f"| 5. Split | {after_red} | train {train_n:,} / test {test_n:,} | Stratified 70/30 |")
    w(f"| 6. SMOTE | {train_n:,} × {n_feat} | {smt.get('samples_after', 0):,} × {n_feat} | Train only |")
    w(f"| 7. Scale | {smt.get('samples_after', 0):,} × {n_feat} | {smt.get('samples_after', 0):,} × {n_feat} | RobustScaler (train fit) |")
    w("")


def _feature_reduction_table(w, raw_cols, hip, red) -> None:
    """Render the feature reduction summary table."""
    n_hip = hip.get("n_dropped", len(hip.get("columns_dropped", [])))
    n_red = red.get("n_dropped", 0)
    # Also subtract non-numeric columns (Attack Category) and label
    n_nonnumeric = 1  # Attack Category
    remaining = raw_cols - n_hip - n_red - n_nonnumeric - 1  # -1 for label

    w("### Feature Reduction Summary")
    w("")
    w("| Reason | Features Dropped | Remaining |")
    w("|--------|----------------:|----------:|")
    w(f"| HIPAA identifiers | {n_hip} | {raw_cols - n_hip} |")
    w(f"| Redundancy (|*r*| ≥ 0.95) | {n_red} | {raw_cols - n_hip - n_red} |")
    w(f"| Non-numeric / label | {n_nonnumeric + 1} | {remaining} |")
    w(f"| **Total reduction** | **{n_hip + n_red + n_nonnumeric + 1}** | **{remaining}** |")
    w("")


def _smote_section(w, smt) -> None:
    """Render the SMOTE before/after section."""
    w("### 4.1.5 SMOTE Oversampling (Train Only)")
    w("")
    counts_before = smt.get("class_counts_before", {})
    counts_after = smt.get("class_counts_after", {})

    w("| Class | Before SMOTE | After SMOTE |")
    w("|-------|------------:|------------:|")
    w(f"| Normal (0) | {counts_before.get(0, 0):,} | {counts_after.get(0, 0):,} |")
    w(f"| Attack (1) | {counts_before.get(1, 0):,} | {counts_after.get(1, 0):,} |")
    w(f"| **Total** | **{smt.get('samples_before', 0):,}** | **{smt.get('samples_after', 0):,}** |")
    w("")
    w(f"SMOTE (Synthetic Minority Oversampling Technique) with "
      f"*k* = {smt.get('k_neighbors', 5)} is applied **exclusively to "
      f"the training partition** to prevent synthetic data from "
      f"contaminating the test evaluation. The oversampling is performed "
      f"**before** scaling so that synthetic samples are generated in the "
      f"original feature space, not in a normalised space where "
      f"inter-feature distances are distorted.")
    w("")
