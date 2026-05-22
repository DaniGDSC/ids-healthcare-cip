"""DAE cascaded-input configuration.

The Track B DAE is trained on ``[raw_features || Track_A_probas]`` (see
:mod:`module2_detection.dae_training`).  The list of Track A models
whose probabilities are appended is shared between training and
inference (:mod:`detection_engine`) — keep it in one place so they
cannot drift.

Changing ``TRACK_A_FOR_DAE`` requires retraining the DAE because the
input layer width is fixed at fit time.
"""

from __future__ import annotations

TRACK_A_FOR_DAE: tuple[str, ...] = ("xgboost",)


def augmented_feature_names(raw_feat_names: list[str]) -> list[str]:
    """Return raw feature names with the Track A proba columns appended."""
    return list(raw_feat_names) + [f"track_a_{n}" for n in TRACK_A_FOR_DAE]
