"""Pure (non-Streamlit) loaders for Module 6 dashboard inputs.

The Streamlit ``@st.cache_data``-wrapped variants in ``module6_app.py``
delegate to the inner functions defined here so unit tests can exercise the
parse/validate path without bootstrapping session state.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from .constants import _SPLIT_FILES, resolve_suffix

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "results/reports"
CHARTS_DIR = PROJECT_ROOT / "results/charts"

ENRICH_KEYS = (
    "device_class", "device_criticality", "affected_system",
    "patient_care_impact", "active_device", "correct_action",
)


def enrich_with_device_context(responses: list, split: str | None = None) -> list:
    """Join responses with evaluation_alerts{suffix}.json for device-context fields.

    Mutates ``responses`` in place AND returns it. Tolerates a missing
    evaluation_alerts file silently (degrades to no enrichment).
    """
    suffix = resolve_suffix(split)
    eval_path = EVAL_DIR / f"evaluation_alerts{suffix}.json"
    if not eval_path.exists():
        return responses
    with open(eval_path) as f:
        eval_alerts = {a["sample_index"]: a for a in json.load(f)}
    for r in responses:
        ea = eval_alerts.get(r.get("sample_index"))
        if ea:
            for k in ENRICH_KEYS:
                if k in ea and k not in r:
                    r[k] = ea[k]
    return responses


class LoaderError(RuntimeError):
    """Raised when responses JSON has an unexpected shape."""


def load_responses_inner(split: str | None) -> list:
    """Pure variant of ``load_responses_for`` without Streamlit error sinks.

    Raises :class:`LoaderError` on schema mismatch / unknown shape; the
    Streamlit-wrapped caller maps it to ``st.error`` + ``st.stop``.
    """
    from pydantic import ValidationError

    from common.alert_response_schema import AlertResponsesEnvelope

    if split is None:
        return []
    if split not in _SPLIT_FILES:
        raise RuntimeError(
            f"Refusing to load alert_responses for split={split!r}: must be "
            f"one of {sorted(_SPLIT_FILES)} or None."
        )
    suffix = _SPLIT_FILES[split]
    path = EVAL_DIR / f"alert_responses{suffix}.json"
    if not path.exists():
        return []
    with open(path) as f:
        raw = json.load(f)

    if isinstance(raw, list):
        responses = raw
    elif isinstance(raw, dict) and "records" in raw:
        try:
            envelope = AlertResponsesEnvelope.model_validate(raw)
        except ValidationError as exc:
            raise LoaderError(
                f"Schema mismatch in {path.name}: {exc}"
            ) from exc
        responses = [r.model_dump() for r in envelope.records]
    else:
        raise LoaderError(
            f"{path.name} is neither a bare list nor an envelope "
            f"({{'_provenance': ..., 'records': ...}})."
        )

    return enrich_with_device_context(responses, split=split)


def load_provenance_inner(split: str | None) -> dict | None:
    """Return the ``_provenance`` block from an envelope-formatted file."""
    if split is None or split not in _SPLIT_FILES:
        return None
    suffix = _SPLIT_FILES[split]
    path = EVAL_DIR / f"alert_responses{suffix}.json"
    if not path.exists():
        return None
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "_provenance" in raw:
        return raw["_provenance"]
    return None


def load_simulation_stream_inner(split: str | None) -> list:
    """Pure variant of the Online Simulation "Full stream" loader.

    Returns the ``stream`` array from ``simulation_stream_<split>.json`` —
    a per-arrival list of 1632 (demo) / 2448 (test) entries. NORMAL rows
    carry ``alert=None`` and serve as clock ticks; LOW+ rows embed the
    M5 alert payload under ``alert``. Empty list when the file is
    missing — caller (UI) can fall back to alerts-only mode.

    See :mod:`tools.build_simulation_stream` for the producer.
    """
    if split is None or split not in _SPLIT_FILES:
        return []
    suffix = _SPLIT_FILES[split]
    path = EVAL_DIR / f"simulation_stream{suffix}.json"
    if not path.exists():
        return []
    with open(path) as f:
        raw = json.load(f)
    stream = raw.get("stream") if isinstance(raw, dict) else None
    if not isinstance(stream, list):
        raise LoaderError(
            f"{path.name} missing 'stream' array — rebuild via "
            f"tools.build_simulation_stream --split {split}."
        )
    return stream


def load_simulation_stream_meta_inner(split: str | None) -> dict | None:
    """Return the ``_meta`` block of simulation_stream_<split>.json
    (split label, counts, anchor timestamp). None when file is missing."""
    if split is None or split not in _SPLIT_FILES:
        return None
    suffix = _SPLIT_FILES[split]
    path = EVAL_DIR / f"simulation_stream{suffix}.json"
    if not path.exists():
        return None
    with open(path) as f:
        raw = json.load(f)
    return raw.get("_meta") if isinstance(raw, dict) else None


__all__ = [
    "EVAL_DIR", "CHARTS_DIR", "PROJECT_ROOT", "ENRICH_KEYS",
    "enrich_with_device_context",
    "load_responses_inner",
    "load_provenance_inner",
    "load_simulation_stream_inner",
    "load_simulation_stream_meta_inner",
    "LoaderError",
]
