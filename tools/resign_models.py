#!/usr/bin/env python3
"""Re-sign Track-A model pickles that have stale ECDSA sidecars.

Background — Sprint 1.1 of the upgrade-plan remediation work:

  The Track-A classifier pickles in ``results/models/*_final_pipeline.pkl``
  were re-trained at some point after the upgrade work began, but their
  ``.pkl.sig`` sidecars were not regenerated. The integrity check in
  ``common.signed_pickle.loads_signed`` correctly refuses to deserialise
  the stale-sha files, which forced Phase 2 + Phase 4 offline tooling
  to bypass via direct ``joblib.load`` (a security regression as far as
  the production read path is concerned, even if the offline tools
  themselves are safe).

  This tool re-signs each pickle in place:

    1. Load the pickle via raw ``joblib.load`` (we explicitly bypass
       ``loads_signed`` here — the pickle bytes ARE the source of truth
       for what's currently on disk, and we trust the local filesystem
       inside the dev container).
    2. Re-save via ``dumps_signed`` using the private key at
       ``~/.iomt-ids/audit_signing_key.pem`` (created by the audit
       bootstrap on first run of Module 5).
    3. Verify the new sidecar by calling ``loads_signed`` and checking
       the object round-trips.

  After this runs, all production code paths that go through
  ``common.model_registry.get_track_a_classifiers`` succeed again, and
  the Phase 2 / Phase 4 regen tools can stop using direct joblib.

Usage:
    python -m tools.resign_models           # re-sign all 3 Track-A pickles
    python -m tools.resign_models --dry-run # show what would be re-signed
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common import dumps_signed, loads_signed  # noqa: E402
from common.signed_pickle import SignedPickleError  # noqa: E402


MODEL_PATHS = [
    PROJECT_ROOT / "results/models/xgboost_final_pipeline.pkl",
    PROJECT_ROOT / "results/models/random_forest_final_pipeline.pkl",
    PROJECT_ROOT / "results/models/decision_tree_final_pipeline.pkl",
]


def _check_integrity(path: Path) -> tuple[bool, str]:
    """Return ``(ok, reason)`` — does loads_signed currently succeed?"""
    try:
        loads_signed(path)
        return True, "sidecar valid"
    except SignedPickleError as exc:
        return False, str(exc).split(":")[0]
    except FileNotFoundError as exc:
        return False, f"missing: {exc}"


def _resign_one(path: Path, dry_run: bool) -> dict:
    if not path.exists():
        return {"path": path.name, "status": "MISSING", "before": None, "after": None}

    before_ok, before_reason = _check_integrity(path)
    if before_ok:
        return {
            "path": path.name, "status": "SKIPPED",
            "before": "valid", "after": "valid",
            "reason": "sidecar already current",
        }

    if dry_run:
        return {
            "path": path.name, "status": "WOULD_RESIGN",
            "before": before_reason, "after": "n/a (dry-run)",
        }

    # Load via raw joblib — we trust the bytes on disk locally.
    try:
        obj = joblib.load(path)
    except Exception as exc:
        return {
            "path": path.name, "status": "LOAD_FAILED",
            "before": before_reason, "error": str(exc),
        }

    # Re-sign in place.
    try:
        dumps_signed(obj, path)
    except Exception as exc:
        return {
            "path": path.name, "status": "SIGN_FAILED",
            "before": before_reason, "error": str(exc),
        }

    after_ok, after_reason = _check_integrity(path)
    return {
        "path": path.name,
        "status": "RESIGNED" if after_ok else "RESIGN_VERIFY_FAILED",
        "before": before_reason,
        "after": after_reason,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be re-signed; don't touch disk.")
    args = p.parse_args()

    results = [_resign_one(path, args.dry_run) for path in MODEL_PATHS]

    print()
    print("=" * 76)
    print("RE-SIGN MODEL PICKLES" + (" (dry run)" if args.dry_run else ""))
    print("=" * 76)
    for r in results:
        status = r["status"]
        marker = {
            "SKIPPED": "✓",
            "RESIGNED": "✓",
            "WOULD_RESIGN": "·",
            "RESIGN_VERIFY_FAILED": "✗",
            "LOAD_FAILED": "✗",
            "SIGN_FAILED": "✗",
            "MISSING": "?",
        }.get(status, "?")
        print(f"  [{marker}] {r['path']:<35s} {status}")
        if r.get("before") and r.get("before") != "valid":
            print(f"          before: {r['before']}")
        if r.get("error"):
            print(f"          error:  {r['error']}")
    print("=" * 76)

    n_failed = sum(1 for r in results if r["status"] in
                   {"RESIGN_VERIFY_FAILED", "LOAD_FAILED", "SIGN_FAILED"})
    return 1 if n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
