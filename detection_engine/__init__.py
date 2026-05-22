"""Detection engine — inference-time fusion of Track A (supervised) and
Track B (DAE novelty) into a single detection confidence ``C_detect``.

Owns the cascaded-input construction (``[raw_features || Track_A_probas]``)
in one place so that Modules 3, 4, and 6 do not duplicate the logic.
The exact set of Track A models that feed the DAE is defined in
:mod:`common.dae_input` — change it there, retrain via
``module2_detection.dae_training``, and every consumer here picks it up
automatically.
"""

from .engine import DetectionEngine, DetectionResult

__all__ = ["DetectionEngine", "DetectionResult"]
