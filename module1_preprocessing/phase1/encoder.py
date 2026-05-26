"""Categorical encoder — deterministic mappings persisted as a JSON sidecar.

Applied after HIPAA sanitization, before missing value handling.

Persistence model
-----------------
For every column listed in ``label_encode``, the encoder builds a
**sorted-alphabetical** integer mapping (``"->"`` → 0, ``"<-"`` → 1, …)
and writes it to a JSON sidecar at ``save(path)``. The previous
implementation used ``sklearn.preprocessing.LabelEncoder`` which
assigned codes in *observation order* — that introduces a label
correlation if the dataset is sorted by class. Sorted-alphabetical
codes are independent of row order by construction, which closes that
leakage.

The fitted mappings are written to disk so downstream inference
modules can reproduce the exact same integer codes for unseen samples
without re-fitting an encoder. Loading is via ``from_json``; there is
no pickle path.

Unknown categories at inference time are mapped to ``unknown_value``
(default ``-1``, matching the ``parse_numeric`` sentinel) rather than
crashing — and the unknown count is logged.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from ._sidecar_io import atomic_write_json, load_sidecar, migrate_legacy_pkl
from .base import BaseTransformer

logger = logging.getLogger(__name__)

_SIDECAR_FORMAT = "phase1.encoder.v1"


class CategoricalEncoder(BaseTransformer):
    """Encode remaining categorical columns to numeric (deterministic).

    - Columns listed in ``label_encode`` are mapped to integers via a
      sorted-alphabetical lookup table (NOT ``LabelEncoder``).
    - Columns listed in ``parse_numeric`` are coerced via
      ``pd.to_numeric``; non-parseable values become ``sentinel``.

    Args:
        label_encode: Column names to label-encode.
        parse_numeric: Column names to coerce from string to numeric.
        sentinel: Value for non-parseable strings (default ``-1``).
        unknown_value: Value emitted for categorical strings not seen
            during fit (default ``-1``). Inference paths that load via
            ``from_json`` use this for novel labels.
    """

    def __init__(
        self,
        label_encode: List[str] | None = None,
        parse_numeric: List[str] | None = None,
        sentinel: int = -1,
        unknown_value: int = -1,
    ) -> None:
        self._label_encode = label_encode or []
        self._parse_numeric = parse_numeric or []
        self._sentinel = sentinel
        self._unknown_value = unknown_value
        # Per-column mapping: column → {category_string: int_code}
        self._mappings: Dict[str, Dict[str, int]] = {}
        self._report_data: Dict[str, Any] = {}

    # ── public API ─────────────────────────────────────────────────

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        label_encoded: Dict[str, int] = {}
        parsed: Dict[str, int] = {}

        # ── Label-encode categorical columns deterministically ──
        for col in self._label_encode:
            if col not in df.columns:
                continue
            # Stringify and build a sorted-alphabetical lookup. Sort
            # order is the only thing that determines integer codes,
            # so re-running on the same set of categories produces the
            # same mapping regardless of row order.
            string_col = df[col].astype(str)
            categories = sorted(string_col.unique().tolist())
            mapping = {cat: i for i, cat in enumerate(categories)}
            self._mappings[col] = mapping
            df[col] = string_col.map(mapping)
            # No NaNs are possible here because every category came
            # from string_col itself, but we defensively backfill any
            # `unknown_value` to keep numeric dtypes consistent.
            df[col] = df[col].fillna(self._unknown_value).astype(int)
            label_encoded[col] = len(categories)
            logger.info(
                "CategoricalEncoder: label-encoded '%s' (%d classes, "
                "deterministic alphabetical: %s)",
                col,
                len(categories),
                categories[:8],
            )

        # ── Parse string columns to numeric with sentinel for failures ──
        for col in self._parse_numeric:
            if col not in df.columns:
                continue
            n_before = df[col].notna().sum()
            df[col] = pd.to_numeric(df[col], errors="coerce")
            n_coerced = int(n_before - df[col].notna().sum())
            df[col] = df[col].fillna(self._sentinel)
            parsed[col] = n_coerced
            logger.info(
                "CategoricalEncoder: parsed '%s' to numeric " "(%d non-parseable → sentinel=%d)",
                col,
                n_coerced,
                self._sentinel,
            )

        self._report_data = {
            "label_encoded": label_encoded,
            "parsed_numeric": parsed,
            "sentinel": self._sentinel,
            "unknown_value": self._unknown_value,
            "mapping_classes": {col: len(m) for col, m in self._mappings.items()},
        }
        return df

    def get_report(self) -> Dict[str, Any]:
        return dict(self._report_data)

    # ── persistence ────────────────────────────────────────────────

    def save(self, path: Path) -> Path:
        """Persist all label-encoded mappings as a JSON sidecar.

        Atomic-write via ``tmp + os.replace`` so a crash mid-write
        cannot leave a half-written sidecar that would silently load
        as a different mapping.

        Args:
            path: Destination ``.json`` file. ``.pkl`` is rewritten to
                ``.json`` and the legacy file is removed.

        Returns:
            The absolute path actually written.
        """
        path = migrate_legacy_pkl(Path(path), "encoder")

        body = {
            "format": _SIDECAR_FORMAT,
            "format_version": 1,
            "sentinel": self._sentinel,
            "unknown_value": self._unknown_value,
            "mappings": self._mappings,
            "parse_numeric": self._parse_numeric,
        }
        atomic_write_json(path, body, sort_keys=True)
        logger.info(
            "CategoricalEncoder sidecar saved: %s (%d mapped columns)",
            path,
            len(self._mappings),
        )
        return path

    @classmethod
    def from_json(cls, path: Path) -> "CategoricalEncoder":
        """Reconstruct an encoder from a JSON sidecar.

        Loading executes no Python code; the sidecar is plain JSON.

        Raises:
            FileNotFoundError: if *path* does not exist.
            ValueError: if the sidecar is not a recognised format.
        """
        path = Path(path)
        body = load_sidecar(path, _SIDECAR_FORMAT, "encoder")
        instance = cls(
            label_encode=list(body.get("mappings", {}).keys()),
            parse_numeric=list(body.get("parse_numeric", [])),
            sentinel=int(body.get("sentinel", -1)),
            unknown_value=int(body.get("unknown_value", -1)),
        )
        instance._mappings = {
            col: {str(k): int(v) for k, v in m.items()}
            for col, m in body.get("mappings", {}).items()
        }
        logger.info(
            "CategoricalEncoder sidecar loaded: %s (%d mapped columns)",
            path,
            len(instance._mappings),
        )
        return instance
