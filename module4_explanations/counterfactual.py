"""Counterfactual explanations for Track-A (XGBoost) alerts.

Phase 2 of the faithfulness/actionability upgrade. Produces, for each
flagged alert, a *minimal* feature-space perturbation that would have
flipped the model from "attack" to "benign". This gives non-ML users a
verifiable handle on the alert: "if `Sport` had been near 0, the
system would NOT have alerted."

Design choices:

* **Greedy coordinate descent over SHAP top-K candidates.**
  We don't search the full 25-feature space; instead we restrict to the
  ``CF_CANDIDATE_K`` features with the largest ``|SHAP|`` for the
  sample. This is both faster (≤K binary searches per attempt) and
  more faithful — it perturbs the features the model actually relied
  on, not arbitrary uncorrelated columns.

* **Sparsity preference.** We first try sparsity 1 (one feature),
  then 2, then 3. We stop at the first valid counterfactual. Most
  XGBoost alerts on this corpus flip with sparsity 1 or 2.

* **Plausibility clip to [p05, p95].** Candidate values are constrained
  to the benign training distribution percentiles read from
  ``artifacts/feature_baselines.json``. A counterfactual that requires
  the feature to be 100× outside training range is useless to the
  operator — they can't realistically apply it.

* **Immutable biometric features.** Biometric columns (``BIOMETRIC_FEATURES``)
  represent the patient, not the network. Suggesting "if SpO2 had been
  different, the alert would clear" is wrong both ethically and
  operationally — the system doesn't control the patient. We exclude
  them from the candidate set.

* **Single-call ``predict_proba`` per probe.** XGBoost is fast enough
  that a 10×8-step binary search (~80 calls per sample) over a few
  hundred candidate samples is sub-second. We do NOT do gradient-based
  optimisation (Wachter / DiCE) because XGBoost doesn't expose
  gradients cleanly and the sample-cost win is marginal at this scale.

The output is a ``CounterfactualResult`` (TypedDict-ish dataclass) with:

  - ``sparsity``: number of features changed (1/2/3 or 0 if infeasible)
  - ``changes``: ordered list of ``{feature, original, new, unit, abs_delta}``
  - ``flips_prediction``: True if applying ``changes`` to the original
        sample yields ``predict_proba < threshold``
  - ``new_proba``: model probability after applying the counterfactual
  - ``original_proba``: model probability on the original sample
  - ``remediation_hint``: short operational phrase derivable from the
        top-1 change ("Restrict outbound bandwidth", "Block source port"
        etc.) used by Phase 2.4's "try first" action.
  - ``feasible``: False iff no plausibility-respecting perturbation up
        to ``max_sparsity`` flipped the prediction.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Iterable

import numpy as np

from .config import BIOMETRIC_FEATURES
from .feature_groups import _load_feature_baselines

logger = logging.getLogger(__name__)


# ── Tunables ─────────────────────────────────────────────────────────


CF_CANDIDATE_K  = 5    # top-K SHAP features considered as candidates
CF_BINARY_STEPS = 12   # binary-search iterations per (feature, direction)
CF_MAX_SPARSITY = 3    # never propose more than this many changes
CF_PROBA_MARGIN = 1e-3  # require predict_proba < threshold - margin to count as flipped


# ── Remediation hints (Phase 2.4 — try-first action) ────────────────


_REMEDIATION_HINTS: dict[str, str] = {
    # Network volume → throttle / cap
    "SrcBytes":   "Throttle outbound bandwidth on the source until volume drops below the benign band.",
    "DstBytes":   "Throttle inbound bandwidth on the destination until volume drops below the benign band.",
    "TotBytes":   "Apply a per-flow byte cap; current flow exceeds the benign baseline.",
    "SrcLoad":    "Rate-limit the source port until load drops to the benign band.",
    "DstLoad":    "Rate-limit the destination port until load drops to the benign band.",
    "Load":       "Apply per-flow rate limit; current load exceeds the benign band.",
    # Network protocol
    "Sport":      "Block the anomalous source port at the segment firewall.",
    "Flgs":       "Drop packets with the anomalous TCP flag pattern at the segment firewall.",
    # Network timing
    "DIntPkt":    "Inspect the destination for packet-timing anomalies (possible covert channel).",
    "SIntPkt":    "Inspect the source for packet-timing anomalies (possible covert channel).",
    "SIntPktAct": "Inspect the active session for timing anomalies (possible covert channel).",
    "Dur":        "Force re-handshake; current connection duration is outside the benign band.",
    # Packet structure
    "sMaxPktSz":  "Inspect the source for anomalous packet sizes (possible MTU manipulation).",
    "dMaxPktSz":  "Inspect the destination for anomalous packet sizes (possible MTU manipulation).",
    "sMinPktSz":  "Inspect the source for anomalous packet sizes (possible MTU manipulation).",
    "pSrcLoss":   "Investigate source-side packet loss; possible link congestion or interference.",
    "pDstLoss":   "Investigate destination-side packet loss; possible link congestion or interference.",
}


def _remediation_hint(feature: str) -> str:
    """Return a short operational phrase for a feature, or a generic
    fallback when the feature isn't catalogued."""
    return _REMEDIATION_HINTS.get(
        feature,
        f"Inspect {feature} on the source/destination — the model would clear if this returned to the benign band.",
    )


# ── Result ──────────────────────────────────────────────────────────


@dataclass
class CounterfactualResult:
    sparsity: int = 0
    changes: list[dict] = field(default_factory=list)
    flips_prediction: bool = False
    new_proba: float = 0.0
    original_proba: float = 0.0
    remediation_hint: str = ""
    feasible: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["new_proba"]      = round(float(d["new_proba"]),      6)
        d["original_proba"] = round(float(d["original_proba"]), 6)
        return d


# ── Core search ─────────────────────────────────────────────────────


def _candidate_indices(
    sv_row: np.ndarray,
    feat_names: list[str],
    *,
    k: int = CF_CANDIDATE_K,
    immutable: Iterable[str] = BIOMETRIC_FEATURES,
) -> list[int]:
    """Return up to ``k`` feature indices ordered by descending |SHAP|,
    excluding biometric features."""
    immutable_set = set(immutable)
    abs_vals = np.abs(sv_row)
    order = np.argsort(abs_vals)[::-1]
    out: list[int] = []
    for i in order:
        if feat_names[i] in immutable_set:
            continue
        out.append(int(i))
        if len(out) == k:
            break
    return out


def _proba(clf, x_row: np.ndarray) -> float:
    """``predict_proba`` for one row; handles 1-D / 2-D quirks."""
    return float(clf.predict_proba(x_row.reshape(1, -1))[0, 1])


def _binary_search_flip(
    clf,
    x_row: np.ndarray,
    col: int,
    target_value: float,
    threshold: float,
    *,
    steps: int = CF_BINARY_STEPS,
) -> tuple[float, float] | None:
    """Binary search the smallest mid ∈ [original, target_value] for which
    ``predict_proba(x_with_mid) < threshold``.

    Returns ``(mid, new_proba)`` or None when no mid in the interval
    achieves the flip (i.e., even moving fully to ``target_value`` doesn't
    drop the prediction below threshold).
    """
    original = float(x_row[col])
    probe = x_row.copy()

    # Quick check: does the extreme already flip?
    probe[col] = target_value
    p_extreme = _proba(clf, probe)
    if p_extreme >= threshold - CF_PROBA_MARGIN:
        return None

    lo, hi = original, target_value
    best_mid = target_value
    best_p   = p_extreme

    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        probe[col] = mid
        p = _proba(clf, probe)
        if p < threshold - CF_PROBA_MARGIN:
            # mid is enough — try smaller change
            best_mid = mid
            best_p = p
            hi = mid
        else:
            # need larger change
            lo = mid
    return best_mid, best_p


# ── Public API ──────────────────────────────────────────────────────


def compute_counterfactual(
    clf,
    x_row: np.ndarray,
    sv_row: np.ndarray,
    feat_names: list[str],
    threshold: float,
    *,
    max_sparsity: int = CF_MAX_SPARSITY,
    baselines: dict | None = None,
) -> CounterfactualResult:
    """Find the minimal-sparsity counterfactual for ``x_row``.

    Args:
        clf: XGBoost classifier with ``predict_proba`` (attack class at col 1).
        x_row: feature vector for the alert (``(n_features,)``).
        sv_row: SHAP values for the alert (``(n_features,)``) — used to
            order candidate features by importance.
        feat_names: column names aligned with ``x_row`` and ``sv_row``.
        threshold: decision threshold above which the model emits "attack".
            Counterfactual flip requires ``new_proba < threshold - margin``.
        max_sparsity: never propose more than this many feature changes
            (default 3). When ``max_sparsity`` is exhausted we return an
            infeasible result.
        baselines: optional pre-loaded baselines dict (for test injection).
            When None, loaded from ``artifacts/feature_baselines.json``.

    Returns:
        A ``CounterfactualResult``. When infeasible, ``feasible=False``,
        ``sparsity=0``, ``changes=[]`` and ``remediation_hint=""``.
    """
    bl = baselines if baselines is not None else _load_feature_baselines()
    original_proba = _proba(clf, x_row)

    if original_proba < threshold:
        # Already a "benign" prediction — no counterfactual to find.
        return CounterfactualResult(
            original_proba=original_proba,
            new_proba=original_proba,
            feasible=False,
        )

    candidates = _candidate_indices(sv_row, feat_names)
    if not candidates or max_sparsity < 1:
        return CounterfactualResult(
            original_proba=original_proba, new_proba=original_proba,
            feasible=False,
        )

    # ── sparsity-1 sweep ──
    best_single: tuple[float, float, int, float] | None = None  # (abs_delta, new_val, col, new_p)
    for col in candidates:
        feat = feat_names[col]
        stats = bl.get(feat)
        if not stats:
            continue
        # Try both directions: clip toward p05 (below) and p95 (above).
        for target in (float(stats.get("p05", stats.get("iqr_low", 0.0))),
                       float(stats.get("p95", stats.get("iqr_high", 0.0)))):
            res = _binary_search_flip(clf, x_row, col, target, threshold)
            if res is None:
                continue
            new_val, new_p = res
            abs_delta = abs(new_val - float(x_row[col]))
            if best_single is None or abs_delta < best_single[0]:
                best_single = (abs_delta, new_val, col, new_p)

    if best_single is not None:
        _, new_val, col, new_p = best_single
        return _build_result(
            x_row, [(col, new_val)], new_p, original_proba,
            feat_names, bl,
        )

    if max_sparsity < 2:
        return CounterfactualResult(
            original_proba=original_proba, new_proba=original_proba,
            feasible=False,
        )

    # ── sparsity-2 sweep ──
    # For each pair from candidates, take the joint p95-direction clip
    # (or p05 if SHAP says feature decreases risk) and binary-search a
    # scalar λ ∈ [0, 1] that scales the joint move. This is cheaper than
    # 2-D search and works well for tabular XGBoost.
    pair_best: tuple[float, list[tuple[int, float]], float] | None = None
    for i, ci in enumerate(candidates):
        for cj in candidates[i + 1:]:
            res = _joint_binary_search(
                clf, x_row, [ci, cj], feat_names, threshold, bl, sv_row,
            )
            if res is None:
                continue
            new_vals, new_p = res
            abs_delta = sum(abs(v - float(x_row[c])) for c, v in zip([ci, cj], new_vals))
            if pair_best is None or abs_delta < pair_best[0]:
                pair_best = (abs_delta, list(zip([ci, cj], new_vals)), new_p)

    if pair_best is not None:
        _, changes_list, new_p = pair_best
        return _build_result(x_row, changes_list, new_p, original_proba, feat_names, bl)

    if max_sparsity < 3:
        return CounterfactualResult(
            original_proba=original_proba, new_proba=original_proba,
            feasible=False,
        )

    # ── sparsity-3 sweep — joint over top-3 candidates ──
    if len(candidates) >= 3:
        triple = candidates[:3]
        res = _joint_binary_search(
            clf, x_row, triple, feat_names, threshold, bl, sv_row,
        )
        if res is not None:
            new_vals, new_p = res
            changes_list = list(zip(triple, new_vals))
            return _build_result(x_row, changes_list, new_p, original_proba, feat_names, bl)

    return CounterfactualResult(
        original_proba=original_proba, new_proba=original_proba,
        feasible=False,
    )


def _joint_binary_search(
    clf,
    x_row: np.ndarray,
    cols: list[int],
    feat_names: list[str],
    threshold: float,
    baselines: dict,
    sv_row: np.ndarray,
    *,
    steps: int = CF_BINARY_STEPS,
) -> tuple[list[float], float] | None:
    """Binary search on a *joint* scalar λ ∈ [0, 1] that interpolates
    each column from its original value toward its plausibility-clipped
    target.

    The per-column target is the percentile that opposes the SHAP sign:
    if SHAP[col] > 0 (feature increases attack risk), the target is the
    benign-side extreme (p05 if x>median else still p05 — the safer
    floor); symmetrically for SHAP[col] < 0.

    Returns ``(new_values, new_proba)`` or None when even λ=1 doesn't
    flip the prediction below threshold.
    """
    targets: list[float] = []
    for col in cols:
        feat = feat_names[col]
        stats = baselines.get(feat, {})
        p05 = float(stats.get("p05", stats.get("iqr_low", 0.0)))
        p95 = float(stats.get("p95", stats.get("iqr_high", 0.0)))
        med = float(stats.get("median", 0.5 * (p05 + p95)))
        orig = float(x_row[col])
        shap_pushes_attack = float(sv_row[col]) > 0
        if shap_pushes_attack:
            # Push back toward benign — pick the side that crosses median
            target = p05 if orig > med else p95
        else:
            target = p95 if orig > med else p05
        targets.append(target)

    probe = x_row.copy()

    def _apply_lambda(lam: float) -> float:
        for col, tgt in zip(cols, targets):
            probe[col] = (1.0 - lam) * float(x_row[col]) + lam * tgt
        return _proba(clf, probe)

    p_full = _apply_lambda(1.0)
    if p_full >= threshold - CF_PROBA_MARGIN:
        return None

    lo, hi = 0.0, 1.0
    best_lam = 1.0
    best_p   = p_full
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        p = _apply_lambda(mid)
        if p < threshold - CF_PROBA_MARGIN:
            best_lam = mid
            best_p = p
            hi = mid
        else:
            lo = mid

    new_values = [
        (1.0 - best_lam) * float(x_row[col]) + best_lam * tgt
        for col, tgt in zip(cols, targets)
    ]
    return new_values, best_p


def _build_result(
    x_row: np.ndarray,
    col_value_pairs: list[tuple[int, float]],
    new_proba: float,
    original_proba: float,
    feat_names: list[str],
    baselines: dict,
) -> CounterfactualResult:
    """Materialise a CounterfactualResult from the search output."""
    changes: list[dict] = []
    for col, new_val in col_value_pairs:
        feat = feat_names[col]
        stats = baselines.get(feat, {})
        dec = int(stats.get("decimal_places", 2))
        unit = stats.get("unit") or ""
        orig = float(x_row[col])
        changes.append({
            "feature":  feat,
            "original": round(orig, dec + 2),
            "new":      round(float(new_val), dec + 2),
            "abs_delta": round(abs(float(new_val) - orig), dec + 2),
            "unit":     unit,
        })
    # Order by |delta| descending so the operator sees the most impactful
    # change first.
    changes.sort(key=lambda c: -c["abs_delta"])

    top_feat = changes[0]["feature"] if changes else ""
    return CounterfactualResult(
        sparsity=len(changes),
        changes=changes,
        flips_prediction=True,
        new_proba=new_proba,
        original_proba=original_proba,
        remediation_hint=_remediation_hint(top_feat),
        feasible=True,
    )


# ── Narrative formatter (clinician view) ────────────────────────────


def counterfactual_narrative(result: CounterfactualResult) -> str:
    """Convert a CounterfactualResult into a one-sentence clinician clause.

    Returns ``""`` for infeasible results so callers can ``if cf_narrative:``
    without guarding.
    """
    if not result.feasible or not result.changes:
        return ""
    if result.sparsity == 1:
        c = result.changes[0]
        unit_str = f" {c['unit']}" if c['unit'] else ""
        return (
            f"This alert would clear if {c['feature']} dropped from "
            f"{c['original']}{unit_str} to ~{c['new']}{unit_str} — "
            f"{result.remediation_hint}"
        )
    feats = ", ".join(c["feature"] for c in result.changes)
    return (
        f"This alert would clear if {feats} returned to the benign band — "
        f"{result.remediation_hint}"
    )


__all__ = [
    "CounterfactualResult",
    "compute_counterfactual",
    "counterfactual_narrative",
    "CF_CANDIDATE_K",
    "CF_MAX_SPARSITY",
]
