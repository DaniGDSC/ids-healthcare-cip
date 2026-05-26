"""PreprocessingPipeline integration tests.

Smoke-runs the full pipeline on a tiny synthetic CSV inside a tmp
workspace with a bootstrapped Phase 0 integrity baseline. Verifies:
  - End-to-end run completes without error
  - source_dataset_sha256 is populated in split_metadata.yaml (C2 fix)
  - split_artifact_manifest.txt is produced (Phase 4)
  - The 4 parquet outputs are written and re-loadable
  - Refusing a non-baselined CSV (multi-file bypass guard)
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── Fixture: synthetic dataset + bootstrap integrity baseline ──────────


def _make_mini_csv(out_path: Path, n: int = 200) -> None:
    """Synthetic WUSTL-EHMS-ish CSV: 8 biometric + some network + labels."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        # biometrics
        "Temp":   rng.normal(36.8, 0.5, n),
        "SpO2":   rng.normal(98, 2, n),
        "Pulse_Rate": rng.normal(75, 10, n),
        "SYS":    rng.normal(120, 15, n),
        "DIA":    rng.normal(80, 10, n),
        "Heart_rate": rng.normal(72, 10, n),
        "Resp_Rate": rng.normal(16, 3, n),
        "ST":     rng.normal(0.0, 0.1, n),
        # network features
        "Dur":    rng.uniform(0, 5, n),
        "TotPkts": rng.integers(1, 100, n),
        "TotBytes": rng.integers(50, 5000, n),
        "SrcBytes": rng.integers(20, 2500, n),
        "Rate":   rng.uniform(0, 1000, n),
        "Load":   rng.uniform(0, 10000, n),
        # leakage columns (will be dropped by HIPAA step)
        "SrcAddr": ["10.0.0.1"] * n,
        "DstAddr": ["10.0.0.2"] * n,
        "SrcMac":  ["aa:bb:cc:dd:ee:ff"] * n,
        "DstMac":  ["00:11:22:33:44:55"] * n,
        "Packet_num": np.arange(n),
        # for label encoding
        "Dir":  rng.choice(["->", "<-", "<?>"], n),
        "Flgs": rng.choice(["A", "B", "C"], n),
        "Sport": rng.choice(["443", "80", "8080", "bogus"], n),
        # labels (stratify needs ≥2 samples per class)
        "Attack Category": rng.choice(["normal", "recon", "exfil"], n, p=[0.7, 0.2, 0.1]),
    })
    df["Label"] = (df["Attack Category"] != "normal").astype(int)
    df.to_csv(out_path, index=False)


def _make_phase0_corr_csv(out_path: Path) -> None:
    """Empty correlations CSV — no redundancy in synthetic data."""
    pd.DataFrame(columns=["feature_a", "feature_b", "correlation"]).to_csv(
        out_path, index=False
    )


def _setup_workspace(tmp_path: Path) -> Path:
    """Build a workspace mirroring the production layout."""
    ws = tmp_path / "ws"
    ws.mkdir()
    # Copy module0_analysis so IntegrityVerifier writes baseline next to it
    shutil.copytree(
        PROJECT_ROOT / "module0_analysis",
        ws / "module0_analysis",
        ignore=shutil.ignore_patterns("__pycache__", "dataset_integrity.json*"),
    )
    # Layout
    (ws / "data" / "raw" / "WUSTL-EHMS").mkdir(parents=True)
    (ws / "data" / "processed").mkdir(parents=True)
    (ws / "results" / "phase0_analysis").mkdir(parents=True)
    (ws / "models" / "scalers").mkdir(parents=True)

    csv = ws / "data" / "raw" / "WUSTL-EHMS" / "mini.csv"
    _make_mini_csv(csv)
    _make_phase0_corr_csv(ws / "results" / "phase0_analysis" / "high_correlations.csv")

    # Bootstrap baseline
    from module0_analysis.security import IntegrityVerifier
    verifier = IntegrityVerifier(ws / "module0_analysis")
    verifier.bootstrap(csv)
    return ws


def _make_config(ws: Path):
    from module1_preprocessing.phase1.config import Phase1Config
    return Phase1Config(
        input_dir=ws / "data" / "raw" / "WUSTL-EHMS",
        output_dir=ws / "data" / "processed",
        file_pattern="*.csv",
        label_column="Label",
        multi_label_column="Attack Category",
        id_removal_columns=["SrcAddr", "DstAddr", "SrcMac", "DstMac", "Packet_num"],
        label_encode_columns=["Dir", "Flgs"],
        parse_numeric_columns=["Sport"],
        biometric_columns=["Temp", "SpO2", "Pulse_Rate", "SYS", "DIA",
                           "Heart_rate", "Resp_Rate", "ST"],
        biometric_strategy="median",
        network_strategy="dropna",
        variance_enabled=True,
        correlation_enabled=True,
        correlation_threshold=0.95,
        phase0_corr_file=ws / "results" / "phase0_analysis" / "high_correlations.csv",
        train_ratio=0.60, val_ratio=0.15, test_ratio=0.15, demo_ratio=0.10,
        random_state=42, stratify=True,
        scaling_method="robust",
        smote_enabled=True, smote_strategy="auto", smote_k_neighbors=5,
        track_b_enabled=True,
        phase0_stats_file=ws / "results" / "phase0_analysis" / "stats_report.json",
    )


# ── Integration tests ─────────────────────────────────────────────────


def test_pipeline_runs_end_to_end(tmp_path):
    from module1_preprocessing.phase1.artifact_reader import Phase0ArtifactReader
    from module1_preprocessing.phase1.pipeline import PreprocessingPipeline

    ws = _setup_workspace(tmp_path)
    cfg = _make_config(ws)
    reader = Phase0ArtifactReader(
        project_root=ws,
        stats_file=cfg.phase0_stats_file.relative_to(ws),
        corr_file=cfg.phase0_corr_file.relative_to(ws),
    )
    pipeline = PreprocessingPipeline(cfg, reader, ws)
    report = pipeline.run()

    # Report shape
    assert report["integrity"]["verified"] is True
    assert report["integrity"]["n_files_verified"] == 1
    # 4 parquets exist
    for name in ("train_phase1.parquet", "val_phase1.parquet",
                 "test_phase1.parquet", "demo_phase1.parquet"):
        assert (ws / "data" / "processed" / name).exists(), f"missing {name}"


def test_pipeline_writes_source_sha256_in_split_metadata(tmp_path):
    """C2 fix: source_dataset_sha256 must be populated (was always empty)."""
    import yaml
    from module1_preprocessing.phase1.artifact_reader import Phase0ArtifactReader
    from module1_preprocessing.phase1.pipeline import PreprocessingPipeline

    ws = _setup_workspace(tmp_path)
    cfg = _make_config(ws)
    reader = Phase0ArtifactReader(
        project_root=ws,
        stats_file=cfg.phase0_stats_file.relative_to(ws),
        corr_file=cfg.phase0_corr_file.relative_to(ws),
    )
    PreprocessingPipeline(cfg, reader, ws).run()

    meta = yaml.safe_load(
        (ws / "data" / "processed" / "split_metadata.yaml").read_text()
    )
    sha = meta.get("source_dataset_sha256", "")
    assert sha and len(sha) == 64, (
        f"source_dataset_sha256 must be populated (got {sha!r}). "
        f"C2 regression: dict-shape lookup in _export_split_metadata broke."
    )


def test_pipeline_writes_split_artifact_manifest(tmp_path):
    """Phase 4: manifest with SHA-256 per parquet."""
    from module1_preprocessing.phase1.artifact_reader import Phase0ArtifactReader
    from module1_preprocessing.phase1.pipeline import PreprocessingPipeline

    ws = _setup_workspace(tmp_path)
    cfg = _make_config(ws)
    reader = Phase0ArtifactReader(
        project_root=ws,
        stats_file=cfg.phase0_stats_file.relative_to(ws),
        corr_file=cfg.phase0_corr_file.relative_to(ws),
    )
    PreprocessingPipeline(cfg, reader, ws).run()

    manifest = ws / "data" / "processed" / "split_artifact_manifest.txt"
    assert manifest.exists(), "Phase 4: split_artifact_manifest.txt missing"
    lines = [l for l in manifest.read_text().splitlines() if l.strip()]
    assert len(lines) >= 4, f"Expected ≥4 lines (train/val/test/demo), got {lines}"
    # Each line: <filename>  <64-hex-digest>
    for line in lines:
        parts = line.split()
        assert len(parts) == 2, f"Bad manifest line: {line!r}"
        assert parts[0].endswith(".parquet")
        assert len(parts[1]) == 64
        assert all(c in "0123456789abcdef" for c in parts[1])


def test_pipeline_manifest_hashes_match_parquet_files(tmp_path):
    """Manifest digests must match a fresh SHA-256 of each parquet."""
    from module1_preprocessing.phase1.artifact_reader import Phase0ArtifactReader
    from module1_preprocessing.phase1.pipeline import PreprocessingPipeline

    ws = _setup_workspace(tmp_path)
    cfg = _make_config(ws)
    reader = Phase0ArtifactReader(
        project_root=ws,
        stats_file=cfg.phase0_stats_file.relative_to(ws),
        corr_file=cfg.phase0_corr_file.relative_to(ws),
    )
    PreprocessingPipeline(cfg, reader, ws).run()

    out = ws / "data" / "processed"
    manifest_text = (out / "split_artifact_manifest.txt").read_text()
    manifest = {ln.split()[0]: ln.split()[1] for ln in manifest_text.splitlines() if ln.strip()}
    for name, claimed_sha in manifest.items():
        actual = hashlib.sha256((out / name).read_bytes()).hexdigest()
        assert actual == claimed_sha, f"{name}: manifest claims {claimed_sha[:16]}, actual {actual[:16]}"


def test_pipeline_refuses_unbaselined_csv(tmp_path):
    """Multi-file bypass guard: a CSV not in the baseline must abort."""
    from module0_analysis.security import IntegrityError
    from module1_preprocessing.phase1.artifact_reader import Phase0ArtifactReader
    from module1_preprocessing.phase1.pipeline import PreprocessingPipeline

    ws = _setup_workspace(tmp_path)
    # Drop a SECOND CSV that no one baselined
    extra = ws / "data" / "raw" / "WUSTL-EHMS" / "extra.csv"
    _make_mini_csv(extra, n=10)
    # Different bytes (n=10 vs n=200) → not in baseline
    cfg = _make_config(ws)
    reader = Phase0ArtifactReader(
        project_root=ws,
        stats_file=cfg.phase0_stats_file.relative_to(ws),
        corr_file=cfg.phase0_corr_file.relative_to(ws),
    )
    with pytest.raises(IntegrityError):
        PreprocessingPipeline(cfg, reader, ws).run()


def test_pipeline_refuses_path_pattern_with_separator():
    """file_pattern must be a basename glob — no path traversal."""
    from module1_preprocessing.phase1.artifact_reader import Phase0ArtifactReader
    from module1_preprocessing.phase1.config import Phase1Config
    from module1_preprocessing.phase1.pipeline import PreprocessingPipeline
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ws = _setup_workspace(tmp)
        cfg = _make_config(ws)
        # Tamper with file_pattern post-validation to escape the safety belt
        object.__setattr__(cfg, "file_pattern", "../etc/*.csv")
        reader = Phase0ArtifactReader(
            project_root=ws,
            stats_file=cfg.phase0_stats_file.relative_to(ws),
            corr_file=cfg.phase0_corr_file.relative_to(ws),
        )
        with pytest.raises(ValueError, match="file_pattern must be a basename"):
            PreprocessingPipeline(cfg, reader, ws).run()
