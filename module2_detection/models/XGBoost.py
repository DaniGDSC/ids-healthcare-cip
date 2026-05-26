"""Backwards-compatibility alias for ``GradientBoostingDetector``.

The class was renamed from ``XGBoostDetector`` to
``GradientBoostingDetector`` so the identity matches the underlying
sklearn ``GradientBoostingClassifier`` implementation. External callers
that still import ``XGBoostDetector`` keep working via this module.

New code should import from ``module2_detection.models.GradientBoosting``.
"""

from __future__ import annotations

from .GradientBoosting import GradientBoostingDetector
from .GradientBoosting import PARAM_SPACE  # noqa: F401 — preserved for legacy imports

# Deprecated alias — kept so existing artefact-loading code and any
# external import sites continue to work. Prefer the new name in new code.
XGBoostDetector = GradientBoostingDetector

__all__ = ["XGBoostDetector", "PARAM_SPACE"]
