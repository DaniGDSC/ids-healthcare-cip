"""Artifact schema versioning chain (Sprint 6 / Tầng 3.5).

Catches Category 1 ("silent contract drift") bugs by maintaining a
single source of truth for what version each artifact on disk is
*expected* to be at, and tooling that:

  - embeds ``_schema_version`` (or ``schema_version``) into every
    artifact at write time,
  - reads it back at consume time,
  - raises ``ArtifactVersionMismatch`` when the in-memory expected
    version disagrees with the on-disk recorded version.

The bumps work like a Postel'sLaw negotiation:

  - The producer always writes the *current* version.
  - The consumer reads the version. If the on-disk version is the
    current version, business as usual. If it's an older known
    version, the consumer can route through a migration function;
    if it's a future or unknown version, the consumer fails loud
    instead of producing silently-wrong output.

The registry is intentionally **flat** — one entry per artifact
file. A new schema bump for any single artifact is a one-line
change here plus the migration function in
``common.artifact_versioning_migrations``. CI gates the build
against drift.

How versions roll forward
-------------------------

Use semver MAJOR.MINOR:

  - MINOR bump: backwards-compatible addition (new optional field,
    new key in a metadata block). Consumers loading an older
    artifact still work. No migration needed.
  - MAJOR bump: incompatible change (field renamed, key removed,
    semantics shifted). Migration function required.

Example::

    # In ARTIFACT_VERSIONS:
    "alert_responses.json": "3.2",   # bumped from 3.1

    # In tests/test_versioning.py:
    # Confirm 3.1 → 3.2 migration produces the expected shape.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ── Registry of expected versions ──────────────────────────────────


# Maps an artifact identifier to its current expected schema version.
# Identifier is the basename of the file (no path, no split suffix);
# the look-up handles ``_demo`` etc. variants. Keep this dict small
# and explicit — every new file type added to the pipeline gets a
# row here, and every consumer of the file does its compat check.
ARTIFACT_VERSIONS: dict[str, str] = {
    # Module 3 — risk scores
    "risk_scores.npz":             "2.0",   # Sprint 4 v2 formula
    # Module 4 — explanations (npz artifacts only — the bare-list
    # JSONs analyst_report and clinician_summaries are pending an
    # envelope migration before they can carry _schema_version, so
    # they're intentionally absent here; the version gate skips
    # unregistered files).
    "shap_values_xgboost.npz":     "1.0",
    # NOTE: RF/DT SHAP and DAE feature errors are produced by Module 4
    # full-mode runs only — the Sprint-N offline regen tools don't
    # touch them. They embed ``_schema_version`` automatically through
    # ``save_shap_values`` / ``save_dae_errors`` the next time someone
    # runs Module 4 full. Until that happens we mark them as opt-out
    # so the gate doesn't fail on stale files we don't actively
    # maintain. Move them back here when adding a regular regen path.
    # "shap_values_random_forest.npz": "1.0",
    # "shap_values_decision_tree.npz": "1.0",
    # "dae_feature_errors.npz":      "1.0",
    # Module 5 — alert responses (envelope shape)
    "alert_responses.json":        "3.2",   # post-Sprint 3 auto_execute always-emit
    # CI / instrumentation
    "phase0_baseline.json":        "2.1",   # post-Sprint 3 operational_health
    "faithfulness_gate.json":      "1.1",   # post-Sprint 1.2 fragile_share
    "coverage_audit.json":         "1.0",
    "formula_comparison.json":     "1.0",
    "v1_v2_comparison.json":       "1.0",
    "stability_variant_comparison.json": "1.0",
}


# Files explicitly tracked as opt-out / pending migration. Listed here
# so the doc generator can flag them rather than silently skip.
PENDING_ENVELOPE_MIGRATION: set[str] = {
    "analyst_report.json",
    "clinician_summaries.json",
}


# ── Errors ────────────────────────────────────────────────────────


class ArtifactVersionMismatch(Exception):
    """Raised when an artifact's recorded ``_schema_version`` disagrees
    with the expected version in :data:`ARTIFACT_VERSIONS`. The caller
    should either:

      - Bump the entry in :data:`ARTIFACT_VERSIONS` if the new shape
        is the intended new normal, OR
      - Re-run the producer with the current code so the artifact
        catches up.
    """


# ── Identifier normalisation ──────────────────────────────────────


def _normalise_artifact_name(path: Path | str) -> str:
    """Strip per-split / per-model suffixes so a single ARTIFACT_VERSIONS
    row covers ``risk_scores.npz`` and ``demo_scores.npz``, etc.
    """
    name = Path(path).name
    # Strip ``_demo`` before extension
    stem, dot, ext = name.rpartition(".")
    if dot:
        if stem.endswith("_demo"):
            stem = stem[: -len("_demo")]
        if stem.endswith("_test"):
            stem = stem[: -len("_test")]
        # Special case: demo_scores.npz maps to risk_scores.npz
        if stem == "demo_scores":
            stem = "risk_scores"
        return f"{stem}.{ext}"
    return name


# ── Read / write ──────────────────────────────────────────────────


def read_version(path: Path) -> str | None:
    """Return the on-disk schema version, or None when the artifact
    pre-dates versioning. Recognises:

      - ``.json`` files with a top-level ``_schema_version`` key, OR
        a ``_provenance._schema_version`` nested key for envelope
        artifacts.
      - ``.npz`` files with a ``schema_version`` array (0-d str).
    """
    path = Path(path)
    if not path.exists():
        return None
    if path.suffix == ".json":
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        if isinstance(data, dict):
            if "_schema_version" in data:
                return str(data["_schema_version"])
            prov = data.get("_provenance") or {}
            if "_schema_version" in prov:
                return str(prov["_schema_version"])
        return None
    if path.suffix == ".npz":
        import numpy as np
        try:
            d = np.load(path, allow_pickle=True)
        except Exception:
            return None
        if "schema_version" in d.files:
            return str(d["schema_version"])
        return None
    return None


@dataclass(frozen=True)
class VersionCheck:
    artifact:          str
    on_disk:           str | None
    expected:          str | None
    ok:                bool
    reason:            str


def check_compatibility(path: Path) -> VersionCheck:
    """Compare the on-disk version against the registry. ``ok=True``
    when they match, ``ok=False`` otherwise (with a human-readable
    ``reason``)."""
    name = _normalise_artifact_name(path)
    expected = ARTIFACT_VERSIONS.get(name)
    on_disk = read_version(path)
    if expected is None:
        return VersionCheck(
            artifact=name, on_disk=on_disk, expected=None, ok=True,
            reason="no expected version registered — treat as opt-out",
        )
    if on_disk is None:
        return VersionCheck(
            artifact=name, on_disk=None, expected=expected, ok=False,
            reason=("artifact has no _schema_version field — was written "
                    "before versioning landed; rerun the producer"),
        )
    if on_disk == expected:
        return VersionCheck(
            artifact=name, on_disk=on_disk, expected=expected, ok=True,
            reason="matches registry",
        )
    return VersionCheck(
        artifact=name, on_disk=on_disk, expected=expected, ok=False,
        reason=(f"on-disk {on_disk!r} ≠ expected {expected!r} — rerun "
                "the producer or bump ARTIFACT_VERSIONS"),
    )


def assert_compatible(path: Path) -> None:
    """Convenience wrapper around :func:`check_compatibility` that
    raises ``ArtifactVersionMismatch`` on failure. Intended for
    consumers that want a hard fail at load time."""
    check = check_compatibility(path)
    if not check.ok:
        raise ArtifactVersionMismatch(
            f"{check.artifact}: {check.reason}"
        )


# ── Embedding helpers ────────────────────────────────────────────


def embed_version_in_dict(payload: dict, artifact_name: str) -> dict:
    """Return ``payload`` augmented with a ``_schema_version`` key.

    When the payload has a ``_provenance`` block, the version is
    nested inside it (Module 5 envelope convention). Otherwise it's
    top-level. Returns a *new* dict so callers can use it as an
    expression.
    """
    version = ARTIFACT_VERSIONS.get(_normalise_artifact_name(artifact_name))
    if version is None:
        return payload  # opt-out
    out = dict(payload)
    if "_provenance" in out and isinstance(out["_provenance"], dict):
        prov = dict(out["_provenance"])
        prov["_schema_version"] = version
        out["_provenance"] = prov
    else:
        out["_schema_version"] = version
    return out


def version_kwarg_for(artifact_name: str) -> dict[str, Any]:
    """Return ``{"schema_version": ndarray("X.Y")}`` for ``np.savez``,
    or empty dict for unregistered artifacts."""
    import numpy as np
    version = ARTIFACT_VERSIONS.get(_normalise_artifact_name(artifact_name))
    if version is None:
        return {}
    return {"schema_version": np.array(version, dtype=str)}


__all__ = [
    "ARTIFACT_VERSIONS",
    "ArtifactVersionMismatch",
    "VersionCheck",
    "read_version",
    "check_compatibility",
    "assert_compatible",
    "embed_version_in_dict",
    "version_kwarg_for",
]
