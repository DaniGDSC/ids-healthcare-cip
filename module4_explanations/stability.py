"""Explanation-stability score (Phase 4.1).

Each alert gets a single stability number measuring how robust the
top-K SHAP feature set is under tiny input perturbations. The intuition:

  "If a 1% wiggle in the input dramatically reshuffles which feature
   the model says is most important, the explanation is *unstable* —
   the operator should not treat the SHAP top-1 as ground truth.
   When the top-K stays put, the explanation is *stable* and the
   operator can rely on the prescribed action."

Procedure (per alert):

  1. Take the original row ``x``.
  2. For ``n_perturbations`` independent draws, sample Gaussian noise
     ``ε ~ 𝒩(0, σ²)`` and form ``x' = x + ε`` (σ defaults to 0.01,
     calibrated to the normalised feature space).
  3. Re-compute TreeSHAP for each ``x'`` and extract the top-K feature
     set by ``|SHAP|``.
  4. The stability score is the mean Jaccard overlap between the
     baseline top-K and each perturbed top-K. Range: ``[0, 1]``.

  5. Band:
       score ≥ 0.90 → STABLE       (display 🟢, trust the top features)
       0.70 ≤ score < 0.90 → BORDERLINE (display 🟡, double-check)
       score < 0.70 → UNSTABLE     (display 🔴, do not auto-execute)

The band is what the clinician/admin views consume. The raw score is
reserved for the analyst view + the faithfulness CI gate.

Reused by:
  - ``tools/phase1_regen_module4.py`` to attach a ``stability`` dict to
    each analyst entry / clinician summary
  - ``module5_responses.adaptive`` (via the pipeline) to demote
    ``auto_execute`` when the top-K is UNSTABLE
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np

logger = logging.getLogger(__name__)


# ── Tunables ────────────────────────────────────────────────────────


N_PERTURBATIONS_DEFAULT = 20
SIGMA_DEFAULT           = 0.01
TOP_K_DEFAULT           = 5

# Band thresholds — matched to the upgrade-plan acceptance text.
THRESHOLD_STABLE     = 0.90
THRESHOLD_BORDERLINE = 0.70


# ── Result ──────────────────────────────────────────────────────────


@dataclass
class StabilityResult:
    score: float            # mean Jaccard overlap ∈ [0, 1]
    band:  str              # STABLE / BORDERLINE / UNSTABLE
    n_perturbations: int
    sigma: float
    top_k: int
    baseline_top_features: list[str]
    # Worst-case overlap across all perturbations — useful for
    # debugging an UNSTABLE alert (which perturbation flipped the top?).
    min_overlap: float

    def to_dict(self) -> dict:
        d = asdict(self)
        d["score"]       = round(float(d["score"]),       4)
        d["min_overlap"] = round(float(d["min_overlap"]), 4)
        return d


# ── Banding ─────────────────────────────────────────────────────────


def stability_band(score: float) -> str:
    """Map a stability score to its band.

    Boundary convention: the thresholds are inclusive at the top of
    each band, so a score of exactly 0.90 is STABLE and a score of
    exactly 0.70 is BORDERLINE. This matches the upgrade-plan wording
    ("≥0.9 → STABLE, 0.7-0.9 → BORDERLINE").
    """
    if score >= THRESHOLD_STABLE:
        return "STABLE"
    if score >= THRESHOLD_BORDERLINE:
        return "BORDERLINE"
    return "UNSTABLE"


# ── Helpers ─────────────────────────────────────────────────────────


def _top_k_set(shap_row: np.ndarray, feat_names: list[str], k: int) -> set[str]:
    """Return the set of top-K feature names by ``|SHAP|`` for one row."""
    k = min(k, len(shap_row))
    if k <= 0:
        return set()
    idx = np.argpartition(np.abs(shap_row), -k)[-k:]
    return {feat_names[i] for i in idx}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


# ── Core compute ────────────────────────────────────────────────────


def compute_stability(
    explainer,
    x_row: np.ndarray,
    feat_names: list[str],
    *,
    n_perturbations: int = N_PERTURBATIONS_DEFAULT,
    sigma: float = SIGMA_DEFAULT,
    top_k: int = TOP_K_DEFAULT,
    rng: np.random.Generator | None = None,
    baseline_shap_row: np.ndarray | None = None,
) -> StabilityResult:
    """Bootstrap-perturb ``x_row`` and measure top-K stability.

    Args:
        explainer: a SHAP TreeExplainer-compatible object that exposes
            ``.shap_values(X)`` returning either a 2-D ``(n, F)`` array
            or the legacy list-of-two-classes shape (the attack-class
            slice will be extracted automatically).
        x_row: feature vector for the alert, shape ``(F,)``.
        feat_names: column names aligned with ``x_row``.
        n_perturbations: number of noisy draws. 20 is the default —
            empirically enough to stabilise the mean overlap to ±0.02.
        sigma: Gaussian noise std-dev in the *normalised* feature
            space. 0.01 is calibrated so the perturbation is a tiny
            fraction of the typical feature scale.
        top_k: how many features to compare. 5 matches what the
            analyst view shows, so the score reflects what an operator
            would actually see flip.
        rng: optional ``np.random.Generator`` for deterministic tests.
        baseline_shap_row: optional pre-computed baseline SHAP for
            ``x_row``. When provided we skip the unperturbed
            ``explainer.shap_values`` call — useful when the caller
            already has the cached SHAP vector.

    Returns:
        ``StabilityResult`` with score + band + diagnostics.
    """
    from .compute import _normalise_shap_output

    rng = rng if rng is not None else np.random.default_rng()
    F = len(x_row)

    if baseline_shap_row is None:
        baseline_sv = _normalise_shap_output(
            explainer.shap_values(x_row.reshape(1, -1))
        )
        baseline_shap_row = baseline_sv[0]

    baseline_set = _top_k_set(baseline_shap_row, feat_names, top_k)

    overlaps: list[float] = []
    for _ in range(n_perturbations):
        noise   = rng.normal(0.0, sigma, size=F).astype(x_row.dtype)
        perturb = (x_row + noise).reshape(1, -1)
        sv      = _normalise_shap_output(explainer.shap_values(perturb))
        top_set = _top_k_set(sv[0], feat_names, top_k)
        overlaps.append(_jaccard(baseline_set, top_set))

    overlaps_arr = np.asarray(overlaps)
    score = float(overlaps_arr.mean()) if overlaps else 0.0
    return StabilityResult(
        score=score,
        band=stability_band(score),
        n_perturbations=n_perturbations,
        sigma=sigma,
        top_k=top_k,
        baseline_top_features=sorted(baseline_set),
        min_overlap=float(overlaps_arr.min()) if overlaps else 0.0,
    )


# ── Badge string (clinician / admin views) ──────────────────────────


_BADGE_BY_BAND = {
    "STABLE":     "🟢 Explanation: STABLE — top features are robust under input noise.",
    "BORDERLINE": "🟡 Explanation: BORDERLINE — top features may shift; double-check before acting.",
    "UNSTABLE":   "🔴 Explanation: UNSTABLE — top features change under tiny input noise. Do NOT auto-execute; manual review required.",
}


def stability_badge(band: str) -> str:
    """Return the user-facing badge string for a stability band."""
    return _BADGE_BY_BAND.get(band, "")


# ── Sprint 5 / Tầng 3.2 — robust top features ──────────────────────


def compute_robust_top_features(
    explainer,
    x_row: np.ndarray,
    feat_names: list[str],
    *,
    n_perturbations: int = N_PERTURBATIONS_DEFAULT,
    sigma: float = SIGMA_DEFAULT,
    top_k: int = TOP_K_DEFAULT,
    rng: np.random.Generator | None = None,
) -> list[dict]:
    """Return the top-K features by *mean* |SHAP| over N perturbations.

    Sprint 5 / Tầng 3.2 — single-shot TreeSHAP attribution on the
    current corpus is empirically unstable (54% UNSTABLE band on the
    test split). Retraining XGBoost with stability regularisation is
    a large architectural lift deferred to a follow-up sprint.

    A cheaper mitigation is to *report* the top-K by the **mean
    SHAP** across the same perturbation ensemble the stability check
    already runs. The mean is by construction more stable than any
    individual draw because the perturbation noise averages out;
    features that genuinely drive the prediction remain at the top,
    while features that win the top-1 spot only on a single noise
    realisation drop down.

    This does NOT fix the underlying model fragility — it just gives
    the analyst a feature set they can rely on more confidently when
    the per-shot band is UNSTABLE. The Phase 4.1 badge still warns
    them; the robust attribution is the recommended view to use
    when investigating an UNSTABLE alert.

    Args:
        explainer: SHAP TreeExplainer-compatible.
        x_row: feature vector for the alert, shape ``(F,)``.
        feat_names: column names aligned with ``x_row``.
        n_perturbations: number of noisy draws averaged. Default 20.
        sigma: Gaussian noise std-dev in normalised feature space.
        top_k: how many features to return.
        rng: optional ``np.random.Generator`` for deterministic tests.

    Returns:
        List of ``{feature, mean_shap, std_shap, direction}`` dicts,
        ordered by |mean_shap| descending. ``std_shap`` exposes the
        per-feature variance so the analyst can see how confident the
        attribution is (high std → the feature pops in/out depending
        on noise).
    """
    from .compute import _normalise_shap_output

    rng = rng if rng is not None else np.random.default_rng()
    F = len(x_row)

    accum = np.zeros((n_perturbations, F), dtype=np.float64)
    for i in range(n_perturbations):
        noise   = rng.normal(0.0, sigma, size=F).astype(x_row.dtype)
        perturb = (x_row + noise).reshape(1, -1)
        sv      = _normalise_shap_output(explainer.shap_values(perturb))
        accum[i] = sv[0]

    mean_shap = accum.mean(axis=0)
    std_shap  = accum.std(axis=0)
    abs_mean  = np.abs(mean_shap)
    order     = np.argsort(abs_mean)[::-1][:top_k]
    return [
        {
            "feature":    feat_names[i],
            "mean_shap":  round(float(mean_shap[i]), 6),
            "std_shap":   round(float(std_shap[i]), 6),
            "direction":  "increases_risk" if mean_shap[i] > 0 else "decreases_risk",
        }
        for i in order
    ]


__all__ = [
    "StabilityResult",
    "compute_robust_top_features",
    "compute_stability",
    "stability_band",
    "stability_badge",
    "THRESHOLD_STABLE",
    "THRESHOLD_BORDERLINE",
    "N_PERTURBATIONS_DEFAULT",
    "SIGMA_DEFAULT",
    "TOP_K_DEFAULT",
]
