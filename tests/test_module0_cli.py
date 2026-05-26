"""Smoke tests for the two Phase-0 CLIs.

  - module0_analysis.bootstrap_integrity   (initial baseline)
  - module0_analysis.migrate_v2_to_v3      (one-time schema migration)

Round-trip both via subprocess + tmp workspace so the actual main()
entry point is exercised, not just the library API.
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _make_yaml(workspace: Path, csv_path: Path) -> Path:
    cfg_path = workspace / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "dataset": {
            "data_path": str(csv_path.relative_to(workspace)),
            "label_column": "Label",
            "required_columns": ["Label"],
        },
        "analysis": {"correlation_threshold": 0.9},
        "output": {
            "output_dir": "out",
            "stats_report_file": "stats.json",
            "high_correlations_file": "corr.csv",
            "correlation_matrix_file": "matrix.parquet",
        },
    }))
    return cfg_path


def _setup_workspace(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Copy module0_analysis into tmp_path so the CLI's auto-detected
    workspace root resolves there (matters because the integrity baseline
    is written next to the module file).
    """
    workspace = tmp_path / "ws"
    workspace.mkdir()
    shutil.copytree(
        PROJECT_ROOT / "module0_analysis",
        workspace / "module0_analysis",
        ignore=shutil.ignore_patterns("__pycache__", "dataset_integrity.json*"),
    )
    data_dir = workspace / "data" / "raw"
    data_dir.mkdir(parents=True)
    csv = data_dir / "sample.csv"
    csv.write_bytes(b"Label,Attack Category\n0,benign\n1,recon\n")
    cfg = _make_yaml(workspace, csv)
    return workspace, csv, cfg


def _run_cli(module: str, *args: str, cwd: Path) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT)}
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        cwd=cwd, env=env, capture_output=True, text=True,
    )


# ── bootstrap_integrity ────────────────────────────────────────────


def test_bootstrap_cli_creates_v3_baseline(tmp_path):
    workspace, csv, cfg = _setup_workspace(tmp_path)
    res = _run_cli(
        "module0_analysis.bootstrap_integrity",
        "--config", str(cfg),
        cwd=workspace,
    )
    assert res.returncode == 0, f"stdout={res.stdout!r} stderr={res.stderr!r}"
    assert "OK: baselined" in res.stdout
    meta = json.loads((workspace / "module0_analysis" / "dataset_integrity.json").read_text())
    assert meta["version"] == 3


def test_bootstrap_cli_idempotent(tmp_path):
    """Second invocation with the same content must not error."""
    workspace, csv, cfg = _setup_workspace(tmp_path)
    _run_cli("module0_analysis.bootstrap_integrity", "--config", str(cfg), cwd=workspace)
    res = _run_cli("module0_analysis.bootstrap_integrity", "--config", str(cfg), cwd=workspace)
    assert res.returncode == 0, f"stderr={res.stderr!r}"


# ── migrate_v2_to_v3 ───────────────────────────────────────────────


def _write_v2_baseline(workspace: Path, csv_path: Path) -> Path:
    """Build a properly-signed v2 baseline file (path-keyed schema)."""
    from module5_responses.signing import canonical_json, load_signing_key
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec
    import hashlib

    data = csv_path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    body = {
        "version": 2,
        "entries": {
            str(csv_path.resolve()): {
                "sha256": digest,
                "size_bytes": len(data),
                "bootstrapped_at": "2026-01-01T00:00:00+00:00",
            },
        },
    }
    private_key, _, key_id = load_signing_key()
    sig = private_key.sign(canonical_json(body), ec.ECDSA(hashes.SHA256()))
    body["signature"] = base64.b64encode(sig).decode("ascii")
    body["signing_key_id"] = key_id
    body["signature_alg"] = "ECDSA_P256_SHA256"
    p = workspace / "module0_analysis" / "dataset_integrity.json"
    p.write_text(json.dumps(body, indent=2))
    return p


def test_migrate_v2_to_v3_collapses_paths(tmp_path):
    workspace, csv, cfg = _setup_workspace(tmp_path)
    _write_v2_baseline(workspace, csv)

    res = _run_cli(
        "module0_analysis.migrate_v2_to_v3",
        "--in", str(workspace / "module0_analysis" / "dataset_integrity.json"),
        "--out", str(workspace / "module0_analysis" / "dataset_integrity.json"),
        cwd=workspace,
    )
    assert res.returncode == 0, f"stdout={res.stdout!r} stderr={res.stderr!r}"
    assert "OK: migrated" in res.stdout

    meta = json.loads((workspace / "module0_analysis" / "dataset_integrity.json").read_text())
    assert meta["version"] == 3
    assert len(meta["entries"]) == 1


def test_migrate_refuses_already_v3(tmp_path):
    """Idempotency: running migration on a v3 file is a no-op (exit 0)."""
    workspace, csv, cfg = _setup_workspace(tmp_path)
    # Bootstrap a v3 first
    _run_cli("module0_analysis.bootstrap_integrity", "--config", str(cfg), cwd=workspace)
    # Then try migrating
    res = _run_cli(
        "module0_analysis.migrate_v2_to_v3",
        "--in", str(workspace / "module0_analysis" / "dataset_integrity.json"),
        cwd=workspace,
    )
    assert res.returncode == 0
    assert "Already v3" in res.stderr


def test_migrate_refuses_forged_v2_signature(tmp_path):
    workspace, csv, cfg = _setup_workspace(tmp_path)
    p = _write_v2_baseline(workspace, csv)
    # Tamper with v2 entries WITHOUT re-signing
    meta = json.loads(p.read_text())
    next(iter(meta["entries"].values()))["size_bytes"] = 999999
    p.write_text(json.dumps(meta, indent=2))
    res = _run_cli(
        "module0_analysis.migrate_v2_to_v3",
        "--in", str(p),
        cwd=workspace,
    )
    assert res.returncode == 3
    assert "signature is invalid" in res.stderr
