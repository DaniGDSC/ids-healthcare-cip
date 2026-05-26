"""DataLoader tests — load + validate + overview.

Covers:
  - load() round-trips through IntegrityVerifier and returns a DataFrame
  - validate() raises on missing required columns
  - validate() enforces feature_count fields when set
  - validate() skips feature_count check when both are 0 (backwards-compat)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from module0_analysis import DataLoader, IntegrityVerifier, Phase0Config


# ── Fixtures ──────────────────────────────────────────────────────────


def _make_workspace(tmp_path: Path) -> tuple[Path, Path]:
    """Create a workspace + a small synthetic CSV with biometric + network columns."""
    ws = tmp_path / "ws"
    ws.mkdir()
    data_dir = ws / "data" / "raw"
    data_dir.mkdir(parents=True)
    csv = data_dir / "mini.csv"
    df = pd.DataFrame({
        "Label": [0, 1, 0, 1, 0],
        "Attack Category": ["normal", "recon", "normal", "exfil", "normal"],
        "Pulse_Rate": [72, 88, 75, 110, 70],          # biometric
        "Temp": [36.8, 37.2, 36.5, 38.1, 36.7],       # biometric
        "Dur": [0.5, 1.2, 0.3, 5.0, 0.4],             # network
        "TotPkts": [10, 50, 5, 200, 8],               # network
        "SrcAddr": ["10.0.0.1"] * 5,                  # leakage
    })
    df.to_csv(csv, index=False)
    return ws, csv


@pytest.fixture
def workspace_with_data(tmp_path):
    ws, csv = _make_workspace(tmp_path)
    return ws, csv


def _bootstrap_baseline(ws: Path, csv_path: Path) -> None:
    """Bootstrap the integrity baseline for the synthetic CSV."""
    verifier = IntegrityVerifier(ws / "module0_analysis")
    verifier.bootstrap(csv_path)


def _make_config(
    ws: Path,
    csv_path: Path,
    *,
    required: list[str],
    network: int = 0,
    biometric: int = 0,
    leakage: list[str] | None = None,
) -> Phase0Config:
    """Construct a Phase0Config dataclass directly (skip from_yaml ceremony)."""
    return Phase0Config(
        data_path=csv_path.relative_to(ws),
        output_dir=Path("results"),
        label_column="Label",
        required_columns=required,
        leakage_columns=leakage or [],
        network_feature_count=network,
        biometric_feature_count=biometric,
        correlation_threshold=0.95,
        missing_value_warn_pct=5.0,
        outlier_iqr_multiplier=1.5,
        top_variance_k=5,
        random_state=42,
        train_ratio=0.7,
        test_ratio=0.3,
        stats_report_file="stats.json",
        high_correlations_file="corr.csv",
        correlation_matrix_file="matrix.parquet",
        quality_report_file="report.md",
    )


# ── Tests ─────────────────────────────────────────────────────────────


def test_load_round_trips_through_integrity(workspace_with_data):
    ws, csv = workspace_with_data
    _bootstrap_baseline(ws, csv)
    cfg = _make_config(ws, csv, required=["Label", "Attack Category"])
    loader = DataLoader(cfg, workspace_root=ws)
    df = loader.load()
    assert len(df) == 5
    assert "Pulse_Rate" in df.columns


def test_validate_passes_with_all_required_columns(workspace_with_data):
    ws, csv = workspace_with_data
    _bootstrap_baseline(ws, csv)
    cfg = _make_config(ws, csv, required=["Label", "Attack Category"])
    loader = DataLoader(cfg, workspace_root=ws)
    df = loader.load()
    loader.validate(df)  # no raise


def test_validate_raises_on_missing_required(workspace_with_data):
    ws, csv = workspace_with_data
    _bootstrap_baseline(ws, csv)
    cfg = _make_config(ws, csv, required=["Label", "NonExistent"])
    loader = DataLoader(cfg, workspace_root=ws)
    df = loader.load()
    with pytest.raises(ValueError, match="unknown columns:.*NonExistent"):
        loader.validate(df)


def test_feature_count_enforced_biometric_mismatch(workspace_with_data):
    """Config says 5 biometric features, actual is 2 → fail."""
    ws, csv = workspace_with_data
    _bootstrap_baseline(ws, csv)
    cfg = _make_config(
        ws, csv,
        required=["Label"],
        biometric=5,
        leakage=["SrcAddr"],
    )
    loader = DataLoader(cfg, workspace_root=ws)
    df = loader.load()
    with pytest.raises(ValueError, match="Biometric feature count mismatch.*expects 5.*has 2"):
        loader.validate(df)


def test_feature_count_enforced_network_mismatch(workspace_with_data):
    """Config says 5 network features, actual is 2 → fail."""
    ws, csv = workspace_with_data
    _bootstrap_baseline(ws, csv)
    cfg = _make_config(
        ws, csv,
        required=["Label"],
        network=5,
        leakage=["SrcAddr"],
    )
    loader = DataLoader(cfg, workspace_root=ws)
    df = loader.load()
    with pytest.raises(ValueError, match="Network feature count mismatch.*expects 5.*has 2"):
        loader.validate(df)


def test_feature_count_check_skipped_when_zero(workspace_with_data):
    """Both counts = 0 → no enforcement (backwards-compat)."""
    ws, csv = workspace_with_data
    _bootstrap_baseline(ws, csv)
    cfg = _make_config(ws, csv, required=["Label"], network=0, biometric=0)
    loader = DataLoader(cfg, workspace_root=ws)
    df = loader.load()
    loader.validate(df)  # no raise even though counts don't match anything


def test_feature_count_correct_passes(workspace_with_data):
    """Correct counts in config pass the assertion."""
    ws, csv = workspace_with_data
    _bootstrap_baseline(ws, csv)
    cfg = _make_config(
        ws, csv,
        required=["Label"],
        network=2,         # Dur, TotPkts
        biometric=2,       # Pulse_Rate, Temp
        leakage=["SrcAddr"],
    )
    loader = DataLoader(cfg, workspace_root=ws)
    df = loader.load()
    loader.validate(df)


def test_overview_does_not_leak_biometric_values(workspace_with_data, caplog):
    import logging
    ws, csv = workspace_with_data
    _bootstrap_baseline(ws, csv)
    cfg = _make_config(ws, csv, required=["Label"])
    loader = DataLoader(cfg, workspace_root=ws)
    df = loader.load()
    caplog.set_level(logging.INFO)
    loader.overview(df)
    full_log = " ".join(r.message for r in caplog.records)
    # Schema info (dtypes) OK, but no biometric VALUES
    assert "72" not in full_log  # Pulse_Rate sample
    assert "36.8" not in full_log  # Temp sample
    assert "rows" in full_log.lower()
