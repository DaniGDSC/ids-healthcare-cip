"""Shared threshold optimisation utility for all Track A detectors.

Replaces the O(T×N) Python loop that existed as four separate copies in
XGBoost.py, RandomForest.py, DecisionTree.py, and module2_train_models.py.

Algorithm
---------
``sklearn.metrics.precision_recall_curve`` computes precision/recall at
every unique predicted probability in a single O(N log N) sort pass.
We then apply the F-beta formula vectorised over all thresholds — zero
Python-level per-threshold iterations, zero intermediate ``y_pred``
allocations.

Previously (200-threshold Python loop):
  complexity  O(T × N) Python iterations, 200 × N-element array allocations
  T = 200, N = training set size (~50 000 for WUSTL-EHMS)

Now (precision_recall_curve):
  complexity  O(N log N) sort + O(U) vectorised arithmetic, U = unique probas
  allocations one precision array, one recall array, one threshold array
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    beta: float = 2.0,
) -> float:
    """Find the decision threshold that maximises F-beta on the attack class.

    Args:
        y_true:  Binary ground-truth labels (0 = benign, 1 = attack).
        y_proba: Predicted probability of attack for each sample.
        beta:    F-beta weight (default 2.0 — recall weighted 2× over precision,
                 appropriate for security where false negatives are costlier).

    Returns:
        Optimal threshold in [0, 1].  Falls back to 0.5 on degenerate inputs
        (single class, all-zero probabilities, etc.).
    """
    # precision_recall_curve requires at least two classes.
    if len(np.unique(y_true)) < 2:
        return 0.5

    precision, recall, thresholds = precision_recall_curve(y_true, y_proba, pos_label=1)

    # The arrays returned by precision_recall_curve have len(thresholds) ==
    # len(precision) - 1 == len(recall) - 1 (the last point is the trivial
    # precision=1, recall=0 sentinel).  Slice to the threshold-aligned region.
    p = precision[:-1]
    r = recall[:-1]

    b2 = beta ** 2
    denom = b2 * p + r
    # Avoid division by zero on degenerate (p=0, r=0) segments.
    fbeta = np.where(denom > 0, (1.0 + b2) * p * r / denom, 0.0)

    if fbeta.size == 0 or fbeta.max() == 0.0:
        return 0.5

    return float(thresholds[int(np.argmax(fbeta))])
