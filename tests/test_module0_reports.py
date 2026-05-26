"""Quality + Reproducibility report rendering smoke tests."""
from __future__ import annotations

from pathlib import Path


from module0_analysis import (
    Phase0Config,
    render_quality_report,
    render_reproducibility_report,
)


def _config() -> Phase0Config:
    return Phase0Config(
        data_path=Path("data.csv"),
        output_dir=Path("out"),
        label_column="Label",
        required_columns=["Label"],
        leakage_columns=["SrcAddr", "DstAddr"],
        network_feature_count=35,
        biometric_feature_count=8,
        correlation_threshold=0.95,
        missing_value_warn_pct=5.0,
        outlier_iqr_multiplier=1.5,
        top_variance_k=5,
        random_state=42,
        train_ratio=0.7,
        test_ratio=0.3,
        stats_report_file="s.json",
        high_correlations_file="c.csv",
        correlation_matrix_file="m.parquet",
        quality_report_file="r.md",
    )


# ── Quality report ────────────────────────────────────────────────────


def test_quality_report_renders_all_sections():
    cfg = _config()
    md = render_quality_report(
        config=cfg,
        n_rows=1000,
        n_cols=43,
        class_dist={
            "Normal": {"count": 800, "percentage": 80.0},
            "Attack": {"count": 200, "percentage": 20.0},
            "imbalance_ratio": 4.0,
        },
        outlier_report=[
            {"feature": "Dur", "outlier_count": 12, "outlier_pct": 1.2,
             "q1": 0.1, "q3": 0.9, "iqr": 0.8,
             "lower_bound": -1.1, "upper_bound": 2.1, "total": 1000},
        ],
        high_pairs=[("Dur", "TotPkts", 0.97)],
        missing={},
        top_variance=[("Dur", 0.5)],
    )
    # Section headers
    assert "## 3.2 Data Quality Assessment" in md
    assert "### 3.2.1 Outlier Analysis" in md
    assert "### 3.2.2 Class Imbalance" in md
    assert "### 3.2.3 Feature Correlation" in md
    assert "### 3.2.4 Missing Value" in md
    assert "### 3.2.5 Data Leakage" in md
    assert "### 3.2.6 Reproducibility" in md


def test_quality_report_omits_missing_when_none_present():
    cfg = _config()
    md = render_quality_report(
        config=cfg, n_rows=100, n_cols=10,
        class_dist={"Normal": {"count": 50, "percentage": 50.0},
                    "Attack": {"count": 50, "percentage": 50.0},
                    "imbalance_ratio": 1.0},
        outlier_report=[], high_pairs=[], missing={}, top_variance=[],
    )
    assert "zero missing values" in md


def test_quality_report_lists_leakage_columns():
    cfg = _config()
    md = render_quality_report(
        config=cfg, n_rows=100, n_cols=10,
        class_dist={"Normal": {"count": 50, "percentage": 50.0},
                    "Attack": {"count": 50, "percentage": 50.0},
                    "imbalance_ratio": 1.0},
        outlier_report=[], high_pairs=[], missing={}, top_variance=[],
    )
    assert "`SrcAddr`" in md
    assert "`DstAddr`" in md


# ── Reproducibility report ────────────────────────────────────────────


def test_reproducibility_report_renders_all_sections():
    cfg = _config()
    md = render_reproducibility_report(
        config=cfg,
        dataset_hash="abc123def456" * 5,
        test_count=234,
        coverage_pct=87.5,
        installed_packages={"pandas": "2.0.0", "numpy": "1.25.0"},
        security_findings=0,
    )
    assert "## 3.4 Reproducibility" in md
    assert "### 3.4.1 Environment" in md
    assert "### 3.4.2 Experiment Reproducibility" in md
    assert "### 3.4.3 CI/CD Pipeline Summary" in md
    assert "### 3.4.4 Dataset Versioning" in md
    assert "### 3.4.5 Peer Review" in md
    assert "234 tests passing" in md
    assert "87.5%" in md


def test_reproducibility_report_handles_hyphenated_package_keys():
    """`scikit-learn` is hyphenated; loader normalises hyphens to underscores."""
    cfg = _config()
    md = render_reproducibility_report(
        config=cfg,
        dataset_hash="x" * 64,
        test_count=10,
        coverage_pct=80.0,
        installed_packages={"scikit-learn": "1.3.0", "imbalanced-learn": "0.11.0"},
    )
    assert "1.3.0" in md
    assert "0.11.0" in md


def test_reproducibility_report_displays_em_dash_for_missing_package():
    cfg = _config()
    md = render_reproducibility_report(
        config=cfg, dataset_hash="x" * 64,
        test_count=10, coverage_pct=80.0,
        installed_packages={},  # nothing installed
    )
    # Key packages from _KEY_PACKAGES should all show "—"
    assert "| pandas | — |" in md
    assert "| numpy | — |" in md
