"""Feature sanitization (ARCHITECTURE.md Step [5]).

Replaces NaN/Inf in incoming 25-feature flow vectors with per-feature medians
computed from the benign training subset, so a single bad sample cannot crash
downstream classifiers and an attacker cannot use NaN injection to mask
anomalies (EA-06).

Design notes
------------
- Medians are computed once from `data/processed/benign_only_train.parquet`
  and persisted to `data/processed/benign_medians.json`. The lookup is loaded
  lazily on first call to `sanitize_features()`.
- The flag taxonomy here (OK / DEGRADED / FAILED) is the operator-facing
  *severity* of the imputation event:
    OK       — nan_rate <= 5% (rare, isolated NaN)
    DEGRADED — nan_rate > 5%  (likely sensor or capture issue)
    FAILED   — nan_rate >= 50% (input is essentially garbage)
  This is more granular than the per-row IMPUTED_NAN flag emitted by the
  Module-3 batch path; that flag remains the row-level marker for downstream
  scoring, while this module returns the alert-level severity for the
  data-quality field on `ScoredAlert`.
- BENIGN_MEDIANS replacement (rather than 0.0) prevents NaN-injection
  attacks: zero-replacement creates an artificial outlier in the joint
  feature-prediction space, which an adversary could exploit; replacing with
  the benign-median makes a NaN-injected sample look ordinary in the raw
  features but still routes through the data_quality_flag for elevated
  scrutiny.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np

from src.data_models import DataQuality

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENIGN_MEDIANS_PATH = PROJECT_ROOT / "data/processed/benign_medians.json"

# Per-spec thresholds on nan_rate (fraction of NaN/Inf cells in a flow vector).
NAN_RATE_DEGRADED: float = 0.05
NAN_RATE_FAILED: float = 0.50

FEATURE_NAMES_25: Sequence[str] = (
    "Flgs", "Sport", "SrcBytes", "DstBytes", "SrcLoad", "DstLoad",
    "SIntPkt", "DIntPkt", "SIntPktAct", "sMaxPktSz", "dMaxPktSz",
    "sMinPktSz", "Dur", "TotBytes", "Load", "pSrcLoss", "pDstLoss",
    "Temp", "SpO2", "Pulse_Rate", "SYS", "DIA", "Heart_rate",
    "Resp_Rate", "ST",
)

# Lazy-loaded median lookup; populated on first call to sanitize_features().
_BENIGN_MEDIANS: dict[str, float] | None = None


# ── Public API ──────────────────────────────────────────────────────────

def load_benign_medians(path: Path | str = BENIGN_MEDIANS_PATH) -> dict[str, float]:
    """Load (and cache) the per-feature benign-median lookup."""
    global _BENIGN_MEDIANS
    if _BENIGN_MEDIANS is None:
        payload = json.loads(Path(path).read_text())
        _BENIGN_MEDIANS = payload["medians"]
    return _BENIGN_MEDIANS


def sanitize_features(
    x: np.ndarray,
    feature_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, str, float]:
    """Sanitize a single 25-feature flow vector.

    Args:
        x: Shape (25,) or (1, 25). Numeric. May contain NaN or +/-Inf.
        feature_names: Names matching x columns. Defaults to FEATURE_NAMES_25.

    Returns:
        (x_clean, data_quality_flag, nan_rate):
          x_clean — same shape as x, all NaN/Inf replaced by per-feature
                    benign medians.
          data_quality_flag — "OK" | "DEGRADED" | "FAILED".
          nan_rate — float in [0.0, 1.0], fraction of cells that were NaN/Inf.
    """
    feats = list(feature_names) if feature_names is not None else list(FEATURE_NAMES_25)

    arr = np.asarray(x, dtype=np.float64)
    original_shape = arr.shape
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != len(feats):
        raise ValueError(
            f"sanitize_features: expected {len(feats)} features, got {arr.shape[1]}"
        )

    finite_mask = np.isfinite(arr)
    n_total = arr.size
    n_bad = int((~finite_mask).sum())
    nan_rate = n_bad / n_total if n_total else 0.0

    if n_bad > 0:
        medians = load_benign_medians()
        median_row = np.array(
            [medians.get(f, 0.0) for f in feats], dtype=np.float64
        )
        # Broadcast median row to all rows; replace only non-finite cells.
        broadcast = np.broadcast_to(median_row, arr.shape)
        arr = np.where(finite_mask, arr, broadcast)

    if nan_rate >= NAN_RATE_FAILED:
        flag = DataQuality.FAILED.value if hasattr(DataQuality, "FAILED") else "FAILED"
        logger.warning(
            "sanitize_features: nan_rate=%.3f >= %.2f (FAILED) — input largely garbage",
            nan_rate, NAN_RATE_FAILED,
        )
    elif nan_rate > NAN_RATE_DEGRADED:
        flag = "DEGRADED"
        logger.warning(
            "sanitize_features: nan_rate=%.3f > %.2f (DEGRADED) — possible sensor/capture issue or NaN-injection attack (EA-06)",
            nan_rate, NAN_RATE_DEGRADED,
        )
    else:
        flag = "OK"

    return arr.reshape(original_shape), flag, round(float(nan_rate), 6)
