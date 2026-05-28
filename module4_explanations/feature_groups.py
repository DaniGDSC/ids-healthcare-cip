"""Feature-group narrative mapping (SHAP stability defense).

Maps individual SHAP feature names to clinically-meaningful narrative
categories. This absorbs within-category feature swaps (e.g., DIntPkt ↔
Sport) that account for most SHAP instability, producing stable
narratives even when the exact top-1 feature changes.

Used by:
  - online ``AlertExplainer._clinician_nlg`` (per-alert NLG)
  - offline ``build_clinician_summaries`` (batch NLG)
  - both via ``module4_explanations.nlg``

Phase 1.1 of the upgrade plan adds ``observation_phrase`` — an
optional baseline-comparison clause that grounds the categorical
narrative in a numeric deviation (``"~2.8 IQR-widths above benign
baseline"``). Loaded from ``artifacts/feature_baselines.json``
written by ``tools/build_feature_baselines.py``.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path


_FEATURE_GROUPS: dict[str, tuple[str, str]] = {
    # Network timing anomalies
    "DIntPkt":    ("unusual network packet timing",     "network_timing"),
    "SIntPkt":    ("unusual network packet timing",     "network_timing"),
    "SIntPktAct": ("unusual network packet timing",     "network_timing"),
    "Dur":        ("abnormal connection duration",      "network_timing"),
    # Network protocol anomalies
    "Sport":      ("unexpected network port activity",  "network_protocol"),
    "Flgs":       ("abnormal protocol flags",           "network_protocol"),
    # Transfer volume anomalies
    "SrcBytes":   ("unusual data transfer volume",      "network_volume"),
    "DstBytes":   ("unusual data transfer volume",      "network_volume"),
    "TotBytes":   ("unusual data transfer volume",      "network_volume"),
    "SrcLoad":    ("abnormal network load",             "network_volume"),
    "DstLoad":    ("abnormal network load",             "network_volume"),
    "Load":       ("abnormal network load",             "network_volume"),
    # Packet structure anomalies
    "sMaxPktSz":  ("unusual packet structure",          "network_packet"),
    "dMaxPktSz":  ("unusual packet structure",          "network_packet"),
    "sMinPktSz":  ("unusual packet structure",          "network_packet"),
    "pSrcLoss":   ("abnormal packet loss",              "network_loss"),
    "pDstLoss":   ("abnormal packet loss",              "network_loss"),
    # Biometric anomalies
    "Temp":       ("abnormal temperature reading",      "biometric"),
    "SpO2":       ("abnormal oxygen saturation",        "biometric"),
    "Pulse_Rate": ("abnormal pulse rate",               "biometric"),
    "SYS":        ("abnormal blood pressure",           "biometric"),
    "DIA":        ("abnormal blood pressure",           "biometric"),
    "Heart_rate": ("abnormal heart rate",               "biometric"),
    "Resp_Rate":  ("abnormal respiratory rate",         "biometric"),
    "ST":         ("abnormal ST segment",               "biometric"),
}


def _feature_to_narrative(feature_name: str) -> tuple[str, str]:
    """Map a feature name to ``(narrative_phrase, category)``.

    Unknown features fall back to ``(feature_name, "unknown")`` so the
    caller's narrative still renders something (the raw feature name
    instead of a clinical phrase).
    """
    return _FEATURE_GROUPS.get(feature_name, (feature_name, "unknown"))


# ── Phase 1.1: observed-value baseline comparison ──────────────────


_BASELINES_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "feature_baselines.json"


@functools.lru_cache(maxsize=1)
def _load_feature_baselines() -> dict:
    """Read ``artifacts/feature_baselines.json`` (built by
    ``tools/build_feature_baselines.py``). Returns ``{}`` if the file
    is missing — narrative degrades gracefully to the pre-Phase-1.1
    category-only form rather than raising.
    """
    if not _BASELINES_PATH.exists():
        return {}
    try:
        return json.loads(_BASELINES_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _deviation_band(iqr_widths: float) -> str:
    """Map a normalised deviation to a plain-language magnitude band.

    Magnitude is ``|x - median| / IQR_width``. Thresholds 0.5 / 1.5 /
    3.0 come from the convention that the IQR contains the middle 50%
    of benign values, so 1 IQR-width above the median is already near
    the Q75, and 3 IQR-widths is roughly the Tukey-fence "far outlier"
    cutoff.
    """
    a = abs(iqr_widths)
    if a < 0.5:
        return "near baseline"
    if a < 1.5:
        return "slightly outside baseline"
    if a < 3.0:
        return "well outside baseline"
    return "extreme deviation from baseline"


_DEGENERATE_IQR_THRESHOLD = 0.05


def observation_phrase(
    feature_name: str,
    observed_value: float | None,
    *,
    baselines: dict | None = None,
) -> str:
    """Return a short clause grounding the narrative in a numeric deviation.

    Example:
        ``observation_phrase("SYS", 2.4)`` →
            ``"(observed +2.40 mmHg vs benign median 0, ~7.2 IQR-widths above; extreme deviation from baseline)"``

    Degenerate-distribution fallback: when the benign IQR width is
    below ``_DEGENERATE_IQR_THRESHOLD`` (e.g. binary-like features where
    50% of benign rows hold a single value), the IQR-widths magnitude
    is meaningless — we emit a qualitative clause instead, noting that
    benign values cluster tightly. This avoids the "4×10^9 IQR-widths"
    failure mode that arose on Flgs before the threshold was added.

    Returns ``""`` when the feature has no baseline entry or when
    ``observed_value`` is None.
    """
    if observed_value is None:
        return ""
    stats = (baselines if baselines is not None else _load_feature_baselines()).get(feature_name)
    if not stats:
        return ""

    med  = float(stats.get("median", 0.0))
    iqr_low, iqr_high = float(stats.get("iqr_low", 0.0)), float(stats.get("iqr_high", 0.0))
    width = iqr_high - iqr_low
    dec   = int(stats.get("decimal_places", 2))
    unit  = stats.get("unit") or ""
    unit_str = f" {unit}" if unit else ""

    delta = float(observed_value) - med
    direction = "above" if delta >= 0 else "below"

    if width < _DEGENERATE_IQR_THRESHOLD:
        # Benign distribution is too tight to compute meaningful IQR-widths.
        if abs(delta) < _DEGENERATE_IQR_THRESHOLD:
            band = "near baseline"
        else:
            band = "outside benign baseline"
        return (
            f"(observed {observed_value:+.{dec}f}{unit_str} vs benign median "
            f"{med:.{dec}f}{unit_str}; {band} — benign values cluster tightly)"
        )

    iqr_widths = delta / width
    band = _deviation_band(iqr_widths)
    return (
        f"(observed {observed_value:+.{dec}f}{unit_str} vs benign median "
        f"{med:.{dec}f}{unit_str}, ~{abs(iqr_widths):.1f} IQR-widths "
        f"{direction}; {band})"
    )


__all__ = [
    "_FEATURE_GROUPS",
    "_feature_to_narrative",
    "observation_phrase",
    "_load_feature_baselines",
    "_deviation_band",
]
