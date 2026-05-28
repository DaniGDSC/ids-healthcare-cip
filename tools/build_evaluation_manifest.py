#!/usr/bin/env python3
"""Build the reproducibility manifest for the frozen evaluation pool.

Phase-1 wrote ``data/processed/split_metadata.yaml`` with per-split row
counts + the splitter ``random_state``, but the source-dataset SHA-256
field was historically left empty (see audit §4.5). This manifest is the
single artefact that pins:

  * the raw WUSTL-EHMS-2020 CSV (source of every split)
  * ``data/processed/demo_phase1.parquet`` (the operator-clean split)
  * ``tests/fixtures/user_study_alert_scenarios.yaml`` (the 20-alert
    user-study pool with its presentation order + seed)
  * ``results/reports/evaluation_alerts.json`` (the dashboard's frozen
    alert payload)

…by SHA-256 + mtime + size, with the splitter seed copied in.

Run::

    python3 -m tools.build_evaluation_manifest

Output: ``results/reports/evaluation_manifest.json``. The companion test
``tests/test_evaluation_manifest_integrity.py`` rehashes each file and
asserts the manifest matches.

When invoked with ``--patch-split-metadata``, this tool ALSO backfills
the ``source_dataset_sha256`` field in ``data/processed/split_metadata.yaml``
if it is currently empty. Future Phase-1 runs populate that field
automatically via the splitter fallback in
``module1_preprocessing/pipeline.py``; this one-shot fixup
closes the gap on the existing artefact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_CSV          = PROJECT_ROOT / "data/raw/WUSTL-EHMS/wustl-ehms-2020_with_attacks_categories.csv"
DEMO_PARQUET     = PROJECT_ROOT / "data/processed/demo_phase1.parquet"
USER_STUDY_YAML  = PROJECT_ROOT / "tests/fixtures/user_study_alert_scenarios.yaml"
EVAL_ALERTS_JSON = PROJECT_ROOT / "results/reports/evaluation_alerts.json"
SPLIT_METADATA   = PROJECT_ROOT / "data/processed/split_metadata.yaml"

MANIFEST_PATH    = PROJECT_ROOT / "results/reports/evaluation_manifest.json"
MANIFEST_VERSION = "evaluation_manifest.v1"

# Each entry maps the logical artefact name to the on-disk path. The
# manifest preserves this ordering so a diff is human-readable.
ARTIFACTS: list[tuple[str, Path]] = [
    ("source_dataset",       RAW_CSV),
    ("demo_split",           DEMO_PARQUET),
    ("user_study_scenarios", USER_STUDY_YAML),
    ("evaluation_alerts",    EVAL_ALERTS_JSON),
]


def _file_record(path: Path) -> dict:
    """Return SHA-256 + mtime + size for a single file."""
    if not path.exists():
        return {
            "path":   str(path.relative_to(PROJECT_ROOT)),
            "exists": False,
            "sha256": None,
            "size":   None,
            "mtime":  None,
        }
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    st = path.stat()
    return {
        "path":   str(path.relative_to(PROJECT_ROOT)),
        "exists": True,
        "sha256": digest,
        "size":   int(st.st_size),
        "mtime":  datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
    }


def _read_splitter_seed() -> int | None:
    """Read the splitter seed from ``split_metadata.yaml``.

    Returns ``None`` if the file is absent or malformed — the manifest
    still builds; the test asserts presence separately.
    """
    if not SPLIT_METADATA.exists():
        return None
    try:
        body = yaml.safe_load(SPLIT_METADATA.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        return None
    if not isinstance(body, dict):
        return None
    seed = body.get("random_state")
    return int(seed) if isinstance(seed, int) else None


def build_manifest() -> dict:
    """Compose the manifest payload."""
    return {
        "format":     MANIFEST_VERSION,
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "splitter_seed": _read_splitter_seed(),
        "artifacts": {
            name: _file_record(path) for name, path in ARTIFACTS
        },
    }


def patch_split_metadata(manifest: dict) -> bool:
    """Backfill ``source_dataset_sha256`` in ``split_metadata.yaml``.

    Returns True if the file was patched, False otherwise (already had a
    SHA, file missing, or source artefact absent).
    """
    if not SPLIT_METADATA.exists():
        return False
    body = yaml.safe_load(SPLIT_METADATA.read_text(encoding="utf-8"))
    if not isinstance(body, dict):
        return False
    current = body.get("source_dataset_sha256")
    if isinstance(current, str) and current:
        return False  # already populated
    new_sha = (manifest["artifacts"]["source_dataset"] or {}).get("sha256")
    if not new_sha:
        return False
    body["source_dataset_sha256"] = new_sha
    SPLIT_METADATA.write_text(
        yaml.safe_dump(body, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--patch-split-metadata",
        action="store_true",
        help="Also backfill source_dataset_sha256 in split_metadata.yaml.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=MANIFEST_PATH,
        help="Override the manifest output path (default: %(default)s).",
    )
    args = parser.parse_args(argv)

    manifest = build_manifest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {args.out.relative_to(PROJECT_ROOT)}", flush=True)
    for name, record in manifest["artifacts"].items():
        if record["exists"]:
            print(f"  {name:<22} sha256={record['sha256'][:16]}…  size={record['size']}")
        else:
            print(f"  {name:<22} MISSING ({record['path']})")

    if args.patch_split_metadata:
        patched = patch_split_metadata(manifest)
        if patched:
            print(f"  + patched split_metadata.yaml with source_dataset_sha256")
        else:
            print(f"  · split_metadata.yaml left as-is")

    return 0


if __name__ == "__main__":
    sys.exit(main())
