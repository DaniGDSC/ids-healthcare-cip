"""Two-stage fusion validation — six named test cases (closes Step [7]).

Per results/reports/two_stage_fusion_validation.yaml § test_cases. Uses
the spec-default thresholds a_high=0.85, a_low=0.40, b=0.70 so the
boundary-condition tests behave exactly as the deliverable describes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from module3_risk_scoring.module3_risk_scores import classify_fusion
from src.data_models import FusionClass

# Spec-default thresholds. The runtime calibration may pick different
# values, but these tests pin behaviour at the spec contract.
SPEC = dict(a_high=0.85, a_low=0.40, b=0.70)


def _classify_one(p_xgb: float, dae: float) -> str:
    arr = classify_fusion(
        c_track_a=np.array([p_xgb], dtype=np.float64),
        c_track_b=np.array([dae], dtype=np.float64),
        **SPEC,
    )
    return arr[0]


def test_1_known_attack_high_prob() -> None:
    """P_xgb=0.95, DAE=0.20 → KNOWN_ATTACK (Stage 1 boundary check)."""
    assert _classify_one(0.95, 0.20) == FusionClass.KNOWN_ATTACK.value


def test_2_novel_anomaly() -> None:
    """P_xgb=0.20, DAE=0.85 → NOVEL_ANOMALY (Stage 2)."""
    assert _classify_one(0.20, 0.85) == FusionClass.NOVEL_ANOMALY.value


def test_3_multi_signal() -> None:
    """P_xgb=0.60, DAE=0.75 → CONFIRMED_ANOMALY (Stage 3)."""
    assert _classify_one(0.60, 0.75) == FusionClass.CONFIRMED_ANOMALY.value


def test_4_benign() -> None:
    """P_xgb=0.30, DAE=0.50 → BENIGN (DAE below b=0.70 — Stage 4)."""
    assert _classify_one(0.30, 0.50) == FusionClass.BENIGN.value


def test_5_boundary_known() -> None:
    """P_xgb=0.85 (exact a_high), DAE=0.30 → KNOWN_ATTACK.

    `>=` on the boundary admits 0.85 into Stage 1 regardless of DAE score.
    """
    assert _classify_one(0.85, 0.30) == FusionClass.KNOWN_ATTACK.value


def test_6_boundary_novel() -> None:
    """P_xgb=0.39 (just below a_low=0.40), DAE=0.70 (exact b) → NOVEL_ANOMALY.

    P_xgb < a_low AND DAE >= b puts the row in Stage 2.
    """
    assert _classify_one(0.39, 0.70) == FusionClass.NOVEL_ANOMALY.value


# ── Additional invariant checks ──────────────────────────────────────────

def test_known_attack_overrides_dae() -> None:
    """KNOWN_ATTACK boundary trumps everything: high P_xgb + low DAE → KNOWN."""
    assert _classify_one(0.95, 0.0) == FusionClass.KNOWN_ATTACK.value
    assert _classify_one(0.95, 1.0) == FusionClass.KNOWN_ATTACK.value


def test_dae_flag_required_for_novel_and_confirmed() -> None:
    """DAE below b → no NOVEL or CONFIRMED, regardless of P_xgb (when below a_high)."""
    # a_low <= P_xgb < a_high but DAE < b → BENIGN, not CONFIRMED
    assert _classify_one(0.60, 0.50) == FusionClass.BENIGN.value
    # P_xgb < a_low and DAE < b → BENIGN, not NOVEL
    assert _classify_one(0.20, 0.50) == FusionClass.BENIGN.value


def test_back_compat_with_old_kwargs() -> None:
    """Old call sites using xgb_threshold/dae_threshold still work."""
    arr = classify_fusion(
        c_track_a=np.array([0.95]),
        c_track_b=np.array([0.0]),
        a_high=0.85,           # pin a_high to spec default for stability
        xgb_threshold=0.05,    # back-compat alias for a_low
        dae_threshold=0.5,     # back-compat alias for b
    )
    # 0.95 >= a_high (0.85) → KNOWN_ATTACK
    assert arr[0] == FusionClass.KNOWN_ATTACK.value
