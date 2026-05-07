"""Multi-class cascade fusion (cascade-contract refactor).

Implements the user-stated cascade contract:

    Trees memorize specific known-attack patterns. Anything they cannot
    confidently match to a known pattern (or to "normal") falls through
    to the DAE. The DAE then decides novel-attack vs normal.

The binary fusion in ``module3_risk_scores.py::classify_fusion`` cannot
realise this contract — a binary classifier learns a boundary, so even
an unseen attack category lands on the attack side with high P_xgb and
gets routed as KNOWN_ATTACK. With multi-class softmax, an unseen
category produces *spread* softmax (no class confidently selected),
which this fusion treats as "uncertain → fall through to DAE".

Returns
-------
``classify_fusion_multiclass`` returns two parallel arrays of length n:

  - ``fusion_class``: one of ``FusionClass.KNOWN_ATTACK``,
    ``FusionClass.NOVEL_ANOMALY``, ``FusionClass.CONFIRMED_ANOMALY``,
    ``FusionClass.BENIGN``.
  - ``predicted_attack_class``: the *specific* known attack class index
    (within ``label_order``) that the trees matched, when
    fusion_class == KNOWN_ATTACK; -1 otherwise.

The CONFIRMED_ANOMALY band is preserved for back-compat with the
binary fusion's downstream consumers, but its semantic shifts: it now
means "trees uncertain AND DAE flags AND argmax was an attack class
(not normal)" — i.e. the trees' best guess was an attack but the
confidence wasn't high enough to call KNOWN_ATTACK without DAE
corroboration.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from src.data_models import FusionClass, P_XGB_HIGH_CONF


def classify_fusion_multiclass(
    softmax_a: np.ndarray,
    dae_score: np.ndarray,
    *,
    label_order: Sequence[str],
    a_high: float | None = None,
    b: float = 0.70,
    normal_idx: int = 0,
    gate_normal_through_dae: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample multi-class cascade classifier (vectorised).

    Args:
        softmax_a: ``(n, K)`` softmax from Track A — typically the
            ensemble mean across XGBoost / RF / DT, or a single model.
            Each row should sum to 1.
        dae_score: ``(n,)`` DAE flag {0, 1} or anomaly score in [0, 1].
            When a continuous score is passed it is thresholded at ``b``.
            For the standard DAE that produces binary predict() output,
            pass that array directly and ignore ``b`` (will treat any
            nonzero as a flag).
        label_order: tuple of class names matching softmax columns.
            ``label_order[normal_idx]`` is the benign class.
        a_high: confidence threshold for the *top-1* softmax class to be
            accepted without DAE corroboration. Defaults to
            ``P_XGB_HIGH_CONF`` (0.85). Per-sample contract:
              top_p >= a_high AND top_class == normal      → BENIGN
              top_p >= a_high AND top_class is an attack   → KNOWN_ATTACK
              top_p <  a_high                              → uncertain → DAE
        b: DAE flag threshold. Used iff ``dae_score`` is a continuous
            score. With ``dae_score`` already binary, this is unused.
        normal_idx: index of the benign class in ``label_order``.
        gate_normal_through_dae: when True (default — matches the literal
            cascade-contract intent), the trees' confident-normal
            predictions are NOT trusted on their own; they go through
            the DAE for verification. When False, confident-normal
            predictions short-circuit to BENIGN. The True path is more
            recall-biased; the False path is more precision-biased.

    Returns:
        ``(fusion_class, predicted_attack_class_idx)``
        - fusion_class: ndarray of strings (FusionClass values), shape (n,)
        - predicted_attack_class_idx: ndarray of int, shape (n,);
          equals the argmax class index when fusion_class == KNOWN_ATTACK,
          -1 otherwise.
    """
    if softmax_a.ndim != 2:
        raise ValueError(f"softmax_a must be (n, K), got shape {softmax_a.shape}")
    if softmax_a.shape[1] != len(label_order):
        raise ValueError(
            f"softmax_a has {softmax_a.shape[1]} classes but label_order "
            f"has {len(label_order)}"
        )
    if not (0 <= normal_idx < len(label_order)):
        raise ValueError(
            f"normal_idx {normal_idx} out of range for {len(label_order)} classes"
        )
    if len(softmax_a) != len(dae_score):
        raise ValueError(
            f"length mismatch: softmax_a={len(softmax_a)} vs "
            f"dae_score={len(dae_score)}"
        )

    a_high_v = float(a_high if a_high is not None else P_XGB_HIGH_CONF)

    top_p = softmax_a.max(axis=1)
    top_class = softmax_a.argmax(axis=1)

    # DAE flag — accept either a binary array or a continuous score.
    dae_arr = np.asarray(dae_score)
    if dae_arr.dtype == bool or set(np.unique(dae_arr)).issubset({0, 1}):
        dae_flag = dae_arr.astype(bool)
    else:
        dae_flag = dae_arr >= b

    n = len(softmax_a)
    fusion = np.full(n, FusionClass.BENIGN.value, dtype=object)
    predicted_attack = np.full(n, -1, dtype=np.int64)

    confident = top_p >= a_high_v
    is_normal = top_class == normal_idx

    # ── Confident on a specific attack class → KNOWN_ATTACK ──
    # This is the only path where the DAE is not consulted. The trees'
    # job is to *filter* known attacks out of the stream — once a tree
    # has confidently matched a specific known-attack signature, that's
    # the answer.
    confident_attack = confident & ~is_normal
    fusion[confident_attack] = FusionClass.KNOWN_ATTACK.value
    predicted_attack[confident_attack] = top_class[confident_attack]

    # ── Everything else goes through the DAE ──
    # When ``gate_normal_through_dae=True`` (the literal cascade contract),
    # confident-normal predictions are NOT trusted on their own — the
    # DAE has to corroborate that the row is actually on the benign
    # manifold. This catches attacks that fool the trees into a confident
    # "normal" prediction.
    if gate_normal_through_dae:
        rest = ~confident_attack
    else:
        # Legacy / precision-biased path: confident-normal → BENIGN
        # without DAE check. Equivalent to old binary fusion in spirit.
        rest = ~confident & ~confident_attack  # i.e. uncertain
    rest_flagged = rest & dae_flag

    # Within DAE-flagged "rest", distinguish two sub-bands by what the
    # tree's argmax was leaning toward:
    #   - argmax was an attack class → CONFIRMED_ANOMALY
    #     (trees leaned attack but with low-mid confidence; DAE agrees)
    #   - argmax was normal → NOVEL_ANOMALY
    #     (trees said benign, DAE disagrees — the canonical novelty case)
    confirmed = rest_flagged & ~is_normal
    novel = rest_flagged & is_normal

    fusion[confirmed] = FusionClass.CONFIRMED_ANOMALY.value
    predicted_attack[confirmed] = top_class[confirmed]
    fusion[novel] = FusionClass.NOVEL_ANOMALY.value
    # predicted_attack stays -1 for NOVEL — no specific known class matched

    # rest & ~dae_flag → BENIGN (fusion already initialised to BENIGN)

    return fusion, predicted_attack


def diversity_score(
    *p_attack_per_model: np.ndarray,
    metric: str = "std",
) -> np.ndarray:
    """Per-row ensemble disagreement score (Enhancement 4).

    Args:
        *p_attack_per_model: each ``(n,)`` array of P(attack) values from
            one Track A model. Pass binary `predict_proba(X)[:, 1]` or
            multi-class `1 - softmax[:, normal_idx]`.
        metric: "std" (default — standard deviation across models) or
            "range" (max - min). Std is the recommended default; range
            is more sensitive to a single dissenting model.

    Returns:
        ``(n,)`` non-negative scores. Higher = greater disagreement.
        With 3 models on a [0, 1] proba scale, std is bounded by ~0.47;
        thresholds in the literature use 0.15–0.25 for "high disagreement".
    """
    if not p_attack_per_model:
        raise ValueError("at least one model required")
    stack = np.stack(p_attack_per_model, axis=0)  # (n_models, n)
    if metric == "std":
        return stack.std(axis=0).astype(np.float32)
    if metric == "range":
        return (stack.max(axis=0) - stack.min(axis=0)).astype(np.float32)
    raise ValueError(f"unknown metric {metric!r} (use 'std' or 'range')")


def classify_fusion_with_diversity(
    softmax_a: np.ndarray,
    dae_score: np.ndarray,
    p_attack_per_model: tuple[np.ndarray, ...],
    *,
    label_order: Sequence[str],
    a_high: float | None = None,
    b: float = 0.70,
    b_diversity: float = 0.20,
    normal_idx: int = 0,
    gate_normal_through_dae: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Multi-class cascade fusion + Enhancement 4 (diversity gating).

    Wraps ``classify_fusion_multiclass`` with a post-step that demotes
    KNOWN_ATTACK outputs to DISAGREEMENT_ANOMALY when the three models'
    P(attack) disagree above ``b_diversity``. The intent: even when the
    ensemble's *averaged* softmax is sharp on a known attack class, a
    high disagreement score among the constituent models is a hedge
    signal — they don't actually all agree. Routing such rows to
    DISAGREEMENT_ANOMALY surfaces them to an operator with a "models
    disagree" tag instead of the high-confidence KNOWN_ATTACK tag.

    Args:
        softmax_a: ``(n, K)`` ensemble softmax (typically the mean).
        dae_score: ``(n,)`` DAE flag or score.
        p_attack_per_model: tuple of ``(n,)`` P(attack) arrays — one per
            constituent Track A model. Used solely for diversity.
        label_order, a_high, b, normal_idx, gate_normal_through_dae:
            see ``classify_fusion_multiclass``.
        b_diversity: disagreement threshold; rows with
            ``diversity_score(...) >= b_diversity`` are demoted from
            KNOWN_ATTACK to DISAGREEMENT_ANOMALY. Default 0.20 per the
            user's enhancement plan.

    Returns:
        ``(fusion_class, predicted_attack_class, diversity)``
        - fusion_class: ndarray of FusionClass values, shape (n,)
        - predicted_attack_class: ndarray of int, shape (n,);
          equals argmax for KNOWN/CONFIRMED, -1 otherwise.
          For DISAGREEMENT_ANOMALY rows demoted from KNOWN_ATTACK, this
          retains the would-have-been-predicted attack class so the
          operator still gets a hint.
        - diversity: ndarray of float, shape (n,). Per-row diversity
          score (raw, not thresholded).
    """
    fusion, pred_attack = classify_fusion_multiclass(
        softmax_a=softmax_a,
        dae_score=dae_score,
        label_order=label_order,
        a_high=a_high,
        b=b,
        normal_idx=normal_idx,
        gate_normal_through_dae=gate_normal_through_dae,
    )

    diversity = diversity_score(*p_attack_per_model, metric="std")
    high_diversity = diversity >= b_diversity

    # Diversity overrides for the *non-DAE-arbitrated* outcomes only:
    #   - KNOWN_ATTACK  : sharp ensemble + split models → demote to
    #                     DISAGREEMENT_ANOMALY (rare, but a hedge against
    #                     a single model dominating the average)
    #   - BENIGN        : ensemble said benign + DAE silent + split models
    #                     → promote to DISAGREEMENT_ANOMALY so the operator
    #                     reviews the row even though both gates passed
    # NOVEL_ANOMALY and CONFIRMED_ANOMALY already passed through DAE
    # arbitration; their classification is not reversed by ensemble
    # disagreement (would re-route a real anomaly back into the queue).
    overridable = (
        (fusion == FusionClass.KNOWN_ATTACK.value)
        | (fusion == FusionClass.BENIGN.value)
    )
    override = overridable & high_diversity
    fusion[override] = FusionClass.DISAGREEMENT_ANOMALY.value

    return fusion, pred_attack, diversity


def ensemble_softmax(
    *softmax_per_model: np.ndarray,
    method: str = "mean",
) -> np.ndarray:
    """Combine per-model softmax matrices into a single (n, K) softmax.

    Args:
        *softmax_per_model: each ``(n, K)`` and rows summing to 1.
        method: "mean" (default — simple average) or "geometric"
            (numerically stabilised geometric mean, then renormalise).

    Returns:
        ``(n, K)`` ensemble softmax with rows summing to 1.
    """
    if not softmax_per_model:
        raise ValueError("at least one softmax matrix required")
    shapes = {s.shape for s in softmax_per_model}
    if len(shapes) > 1:
        raise ValueError(f"shape mismatch across models: {shapes}")
    if method == "mean":
        out = np.stack(softmax_per_model, axis=0).mean(axis=0)
    elif method == "geometric":
        # log-mean then exp avoids underflow on near-zero probabilities.
        eps = 1e-12
        log_p = np.stack([np.log(s + eps) for s in softmax_per_model], axis=0)
        out = np.exp(log_p.mean(axis=0))
        out = out / out.sum(axis=1, keepdims=True)
    else:
        raise ValueError(f"unknown method {method!r} (use 'mean' or 'geometric')")
    return out.astype(np.float32)
