"""Pre-redesign Task 3: unified fusion entry point — both modes covered.

Contract:
    ``module3_risk_scoring.fusion.fuse(...)`` selects between the binary
    and multiclass fusion paths via a ``fusion_mode`` flag, returning a
    uniform ``FusionResult`` shape from either. Default mode is
    "multiclass".

These tests pin the routing + return-shape contract so future refactors
that delete one of the two backends would fail loudly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from module3_risk_scoring.fusion import (
    DEFAULT_FUSION_MODE,
    FusionResult,
    fuse,
)
from src.data_models import FusionClass, MULTICLASS_LABEL_ORDER_EHMS


def test_default_mode_is_multiclass() -> None:
    """Per the cascade-contract redesign, default fuses through multiclass."""
    assert DEFAULT_FUSION_MODE == "multiclass"


def test_binary_routing_minimal() -> None:
    """Binary mode with just p_xgb produces a sensible FusionResult."""
    p_xgb = np.array([0.95, 0.10, 0.50, 0.20])
    dae = np.array([0, 0, 1, 1])
    result = fuse(fusion_mode="binary", p_xgb=p_xgb, dae_score=dae,
                  a_high=0.85, a_low=0.40, b=0.70)
    assert isinstance(result, FusionResult)
    assert result.fusion_mode == "binary"
    assert len(result) == 4
    assert result.predicted_attack_class_idx.tolist() == [-1, -1, -1, -1]
    assert result.diversity.shape == (4,)
    # KNOWN_ATTACK on row 0 (high P_xgb), CONFIRMED on row 2 (mid + DAE),
    # NOVEL on row 3 (low + DAE), BENIGN on row 1.
    assert result.fusion_class[0] == FusionClass.KNOWN_ATTACK.value
    assert result.fusion_class[1] == FusionClass.BENIGN.value
    assert result.fusion_class[2] == FusionClass.CONFIRMED_ANOMALY.value
    assert result.fusion_class[3] == FusionClass.NOVEL_ANOMALY.value


def test_binary_with_diversity() -> None:
    """Binary mode with all 3 model probas populates diversity."""
    p_xgb = np.array([0.10, 0.90])
    p_rf = np.array([0.50, 0.85])
    p_dt = np.array([0.05, 0.95])
    result = fuse(
        fusion_mode="binary",
        p_xgb=p_xgb, p_rf=p_rf, p_dt=p_dt,
        dae_score=np.array([0, 0]),
    )
    # Row 0 has wide spread; row 1 has tight spread.
    assert result.diversity[0] > result.diversity[1]


def test_multiclass_routing_minimal() -> None:
    """Multiclass mode produces predicted_attack_class_idx for KNOWN_ATTACK."""
    # 3 rows × K=3 (normal, Data Alteration, Spoofing) softmax matrices
    sm_xgb = np.array(
        [[0.05, 0.90, 0.05], [0.95, 0.03, 0.02], [0.30, 0.30, 0.40]],
        dtype=np.float32,
    )
    sm_rf = sm_xgb.copy()
    sm_dt = sm_xgb.copy()
    dae = np.array([0, 0, 1])
    result = fuse(
        fusion_mode="multiclass",
        softmax_per_model=(sm_xgb, sm_rf, sm_dt),
        dae_score=dae,
        a_high=0.85,
    )
    assert result.fusion_mode == "multiclass"
    assert result.label_order == MULTICLASS_LABEL_ORDER_EHMS
    # Row 0: confident "Data Alteration" (idx 1) → KNOWN_ATTACK with class 1
    assert result.fusion_class[0] == FusionClass.KNOWN_ATTACK.value
    assert result.predicted_attack_class_idx[0] == 1
    # Row 1: confident "normal" (idx 0) + DAE silent → BENIGN
    assert result.fusion_class[1] == FusionClass.BENIGN.value
    # Row 2: spread softmax + DAE flag → CONFIRMED_ANOMALY (argmax was attack)
    assert result.fusion_class[2] == FusionClass.CONFIRMED_ANOMALY.value


def test_multiclass_resolves_class_names() -> None:
    """`predicted_attack_class_name` resolves indices via label_order."""
    sm = np.array([[0.05, 0.90, 0.05]], dtype=np.float32)
    result = fuse(
        fusion_mode="multiclass",
        softmax_per_model=(sm, sm, sm),
        dae_score=np.array([0]),
        a_high=0.85,
    )
    names = result.predicted_attack_class_name
    assert names[0] == MULTICLASS_LABEL_ORDER_EHMS[1]


def test_uniform_return_shape_across_modes() -> None:
    """Same n in → same fusion_class shape regardless of mode."""
    n = 5
    p = np.array([0.10, 0.50, 0.90, 0.30, 0.95])
    dae = np.array([0, 1, 0, 1, 0])
    sm = np.zeros((n, 3), dtype=np.float32)
    sm[:, 0] = 1.0 - p   # P(normal) = 1 - p
    sm[:, 1] = p / 2
    sm[:, 2] = p / 2

    bin_r = fuse(fusion_mode="binary", p_xgb=p, dae_score=dae)
    mc_r = fuse(
        fusion_mode="multiclass",
        softmax_per_model=(sm, sm, sm), dae_score=dae,
    )
    assert bin_r.fusion_class.shape == (n,)
    assert mc_r.fusion_class.shape == (n,)
    assert bin_r.predicted_attack_class_idx.shape == (n,)
    assert mc_r.predicted_attack_class_idx.shape == (n,)


def test_binary_missing_p_xgb_raises() -> None:
    """fusion_mode='binary' without p_xgb → clear error."""
    import pytest
    with pytest.raises(ValueError, match="p_xgb"):
        fuse(fusion_mode="binary", dae_score=np.array([0]))


def test_multiclass_missing_softmax_raises() -> None:
    """fusion_mode='multiclass' without softmax_per_model → clear error."""
    import pytest
    with pytest.raises(ValueError, match="softmax_per_model"):
        fuse(fusion_mode="multiclass", dae_score=np.array([0]))


def test_unknown_mode_raises() -> None:
    import pytest
    with pytest.raises(ValueError, match="unknown fusion_mode"):
        fuse(  # type: ignore[arg-type]
            fusion_mode="bogus",
            p_xgb=np.array([0.5]), dae_score=np.array([0]),
        )
