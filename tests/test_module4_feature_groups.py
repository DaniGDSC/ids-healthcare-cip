"""Module 4 feature_groups — narrative mapping."""
from __future__ import annotations

from module4_explanations.feature_groups import (
    _FEATURE_GROUPS,
    _feature_to_narrative,
)


def test_known_feature_returns_narrative_and_category():
    narrative, category = _feature_to_narrative("DIntPkt")
    assert narrative == "unusual network packet timing"
    assert category == "network_timing"


def test_unknown_feature_falls_back():
    narrative, category = _feature_to_narrative("BrandNewFeature")
    assert narrative == "BrandNewFeature"  # raw feature name as fallback
    assert category == "unknown"


def test_all_biometric_map_to_biometric_category():
    from common.phi import BIOMETRIC_COLUMNS
    for feat in BIOMETRIC_COLUMNS:
        _, category = _feature_to_narrative(feat)
        assert category == "biometric", f"{feat} → {category}"


def test_network_timing_features_grouped():
    """SHAP-stability defense: timing features all share one narrative."""
    for f in ["DIntPkt", "SIntPkt", "SIntPktAct"]:
        narrative, category = _feature_to_narrative(f)
        assert category == "network_timing"


def test_network_volume_features_grouped():
    for f in ["SrcBytes", "DstBytes", "TotBytes", "SrcLoad", "DstLoad", "Load"]:
        _, category = _feature_to_narrative(f)
        assert category == "network_volume", f"{f} → {category}"


def test_feature_groups_dict_is_dict_of_tuples():
    """Type invariant for callers that iterate the table."""
    for f, val in _FEATURE_GROUPS.items():
        assert isinstance(val, tuple)
        assert len(val) == 2
        assert isinstance(val[0], str)
        assert isinstance(val[1], str)
