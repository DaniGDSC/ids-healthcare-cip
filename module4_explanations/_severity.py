"""Single source for the alert-severity threshold table.

Used by both the offline batch explainer (``module4_explanations.py``)
and the online streaming explainer (``module4_online_explainer.py``).
Lives in its own module so neither file has to import the other for
this primitive — the two files already participate in a one-way
import (offline → online) and adding the reverse edge would create a
cycle.
"""

from __future__ import annotations


def severity(n_models_flagged: int) -> str:
    """Map the number of detector votes to a severity label.

    Thresholds (kept in one place to prevent offline/online drift):

    ============  ========
    flagged_count severity
    ============  ========
    >= 4          CRITICAL
    == 3          HIGH
    == 2          MEDIUM
    <= 1          LOW
    ============  ========

    Args:
        n_models_flagged: Count of detectors that voted "attack" for
            this sample.

    Returns:
        One of ``"CRITICAL" / "HIGH" / "MEDIUM" / "LOW"``.
    """
    if n_models_flagged >= 4:
        return "CRITICAL"
    if n_models_flagged == 3:
        return "HIGH"
    if n_models_flagged == 2:
        return "MEDIUM"
    return "LOW"
