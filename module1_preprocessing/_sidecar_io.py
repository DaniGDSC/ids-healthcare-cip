"""Shared JSON-sidecar I/O helpers for Phase 1 transformers.

Centralises three concerns that were previously open-coded in every
transformer that persists state (``CategoricalEncoder``,
``RobustScalerTransformer``):

1. ``migrate_legacy_pkl`` — rewrite ``.pkl`` paths to ``.json`` and
   delete any leftover legacy pickle so a downstream consumer cannot
   silently load a stale, executable byte stream.
2. ``atomic_write_json`` — write JSON via ``tmp + os.replace`` so a
   crash mid-write cannot leave a half-written file.
3. ``load_sidecar`` — open + parse + format-tag check, raising
   consistent errors across all transformers.

Why one module rather than copy-paste: the ``.pkl`` removal is the
security-critical part of the encoder/scaler persistence model.  A
single audit point makes it impossible for a future fix on one site
to silently miss the other.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


def migrate_legacy_pkl(path: Path, artefact_label: str) -> Path:
    """Rewrite a ``.pkl`` destination to ``.json`` and remove any legacy file.

    No-op if *path* does not have a ``.pkl`` suffix.

    Pickle removal is security-relevant (a leftover ``.pkl`` is an RCE
    sink at every load site), so we route the event through Module 5's
    hash-chained audit log via ``log_phase0_event`` — same discipline
    Module 0 uses for integrity events.

    Args:
        path: Destination path requested by the caller.  May still
            carry the historical ``.pkl`` extension.
        artefact_label: Short human-readable label (e.g. ``"encoder"``,
            ``"scaler"``) used in the warning log line.

    Returns:
        The canonical ``.json`` path the caller should actually write to.
    """
    if path.suffix != ".pkl":
        return path

    # Lazy import — `_sidecar_io` is allowed to be loaded in test contexts
    # where the Module 0 + Module 5 chain may not be wired up yet.
    try:
        from module0_analysis.security import log_phase0_event as _audit
    except (ImportError, ModuleNotFoundError):
        _audit = None  # type: ignore[assignment]

    legacy = path
    json_path = path.with_suffix(".json")
    if legacy.exists():
        try:
            legacy.unlink()
            logger.warning(
                "Removed legacy pickle %s at %s; sidecar at %s is now "
                "the canonical artefact.",
                artefact_label,
                legacy,
                json_path,
            )
            if _audit is not None:
                _audit(
                    "PICKLE_ARTEFACT_REMOVED",
                    {"artefact": artefact_label, "path": str(legacy)},
                    level=logging.WARNING,
                )
        except OSError as exc:
            logger.warning(
                "Could not remove legacy pickle %s at %s: %s "
                "(downstream consumers must be updated to load the "
                "JSON sidecar)",
                artefact_label,
                legacy,
                exc,
            )
            if _audit is not None:
                _audit(
                    "PICKLE_ARTEFACT_REMOVAL_FAILED",
                    {"artefact": artefact_label, "path": str(legacy), "error": str(exc)},
                    level=logging.ERROR,
                )
    return json_path


def atomic_write_json(
    path: Path,
    body: Dict[str, Any],
    *,
    indent: int = 2,
    sort_keys: bool = False,
) -> None:
    """Serialise *body* to *path* atomically.

    Writes to ``path + .tmp`` then ``os.replace`` so a crash mid-write
    cannot leave a half-written file that another tool would mistake
    for a complete sidecar.  ``os.replace`` is atomic on POSIX and
    Windows for same-filesystem moves.  Parent directories are created
    if missing.

    Args:
        path: Destination ``.json`` file.
        body: Serialisable dict.
        indent: Passed through to ``json.dumps``.
        sort_keys: Passed through to ``json.dumps``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(body, indent=indent, sort_keys=sort_keys))
    os.replace(tmp, path)


def load_sidecar(
    path: Path,
    expected_format: str,
    artefact_label: str,
) -> Dict[str, Any]:
    """Read + parse + format-validate a sidecar JSON file.

    Args:
        path: Sidecar to load.
        expected_format: Required value of the ``"format"`` key.
        artefact_label: Short human-readable label used in error messages.

    Returns:
        Parsed JSON body.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file is not a recognised sidecar (missing or
            mismatched ``"format"`` key).
    """
    if not path.exists():
        raise FileNotFoundError(f"{artefact_label.capitalize()} sidecar not found: {path}")

    body = json.loads(path.read_text())
    actual = body.get("format")
    if actual != expected_format:
        raise ValueError(
            f"{path} is not a {expected_format} sidecar (got format={actual!r})"
        )
    return body
