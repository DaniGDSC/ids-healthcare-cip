"""Safe loader for the Module 3 risk-score artefact pair.

Tier 2 F1 closed the previous setup where ``risk_scores.npz`` carried
object-dtype string fields (``risk_levels``, ``formula_version``,
``schema_version``) and every consumer had to ``np.load(allow_pickle=True)``
— five independent pickle-deserialisation sinks across module 4,
module 6, the dashboard, and example tooling.

The new shape:

  * ``risk_scores.npz``       — purely numeric arrays (no allow_pickle).
  * ``risk_scores.meta.json`` — string-typed companion fields.
  * ``risk_scores.meta.json.sig`` — ECDSA signature over the pair.

This module is the single entry point. Consumers call
``load_risk_scores(npz_path)`` and get back a dict-like view of every
field they used to read directly from the .npz, with the string fields
sourced from the JSON sidecar. The pair is verified before any bytes
are exposed to the caller.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RiskScoresArtefact:
    """In-memory view of a signed risk-scores pair.

    Numeric arrays come from the .npz; string-typed fields come from
    the JSON sidecar. ``risk_levels`` is reconstructed from the small
    ``risk_level_codes`` integer array + the JSON's decode table, so the
    field is available without allow_pickle.
    """

    R: np.ndarray
    c_detect: np.ndarray
    c_track_a: np.ndarray
    c_track_b: np.ndarray
    d_crit: np.ndarray
    s_data: np.ndarray
    d_clinical_tier: np.ndarray
    y_true: np.ndarray
    risk_level_codes: np.ndarray
    risk_levels: np.ndarray  # decoded str array; reconstructed from codes
    schema_version: str | None
    formula_version: str | None

    def __getitem__(self, key: str) -> Any:
        """Drop-in shim for code that reads `data["R"]` etc."""
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


def load_risk_scores(npz_path: Path) -> RiskScoresArtefact:
    """Load + verify the (npz, meta_json) pair and return a structured view.

    Raises:
        FileNotFoundError: when either half of the pair is missing.
        signed_sidecar.SignedSidecarError: when the pair is unsigned,
            tampered, or signed by a key whose id does not match the
            local pin.
        KeyError: when expected fields are absent.
    """
    npz_path = Path(npz_path)
    meta_path = npz_path.with_suffix(".meta.json")

    if not npz_path.exists():
        raise FileNotFoundError(f"risk_scores npz not found: {npz_path}")
    if not meta_path.exists():
        raise FileNotFoundError(
            f"risk_scores meta sidecar not found at {meta_path}. "
            "This artefact was produced by a pre-Sprint-2 writer; re-run "
            "Module 3 to emit the new schema."
        )

    # Verify the signed pair before reading any field. SignedSidecarError
    # propagates up; the dict is never partially exposed.
    from common.signed_sidecar import verify_signed_pair
    verify_signed_pair(meta_path, npz_path)

    # Now-safe: bytes have already been verified end-to-end.
    meta: Mapping[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("format") != "risk_scores.meta.v1":
        raise ValueError(
            f"{meta_path}: unexpected format {meta.get('format')!r}."
        )

    # The .npz now contains ONLY numeric arrays — load with the
    # default allow_pickle=False so the deserialization sink is closed.
    npz = np.load(npz_path)

    # Decode risk_levels from codes + table. The table is a small dict
    # in the JSON; reverse it once and apply.
    code_to_label = {int(v): str(k) for k, v in (meta.get("risk_level_codes") or {}).items()}
    codes = np.asarray(npz["risk_level_codes"])
    risk_levels = np.array(
        [code_to_label.get(int(c), "NORMAL") for c in codes], dtype="<U8",
    )

    return RiskScoresArtefact(
        R=np.asarray(npz["R"]),
        c_detect=np.asarray(npz["c_detect"]),
        c_track_a=np.asarray(npz["c_track_a"]),
        c_track_b=np.asarray(npz["c_track_b"]),
        d_crit=np.asarray(npz["d_crit"]),
        s_data=np.asarray(npz["s_data"]),
        d_clinical_tier=np.asarray(npz["d_clinical_tier"]),
        y_true=np.asarray(npz["y_true"]),
        risk_level_codes=codes,
        risk_levels=risk_levels,
        schema_version=meta.get("schema_version"),
        formula_version=meta.get("formula_version"),
    )


__all__ = ["RiskScoresArtefact", "load_risk_scores"]
