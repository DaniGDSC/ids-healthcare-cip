"""Multi-view example explanations for thesis figures.

Picks 5 representative alerts (top-2 confidence + 1 spoofing + 1 data
alteration + 1 borderline) and routes each through all 3 stakeholder
views (clinician / analyst / administrator).

Y3 fix: the risk-scores file path now resolves through
``common.split_paths.risk_scores(split)`` instead of the hardcoded
``data/phase2/risk_scores/risk_scores.npz``. The original path didn't
exist on disk so worked examples carried zeros for every risk field —
breaking the manuscript's R-decomposition figure. With the correct
path, ``example_explanations.json`` now contains real risk scores
when Module 3 has produced ``results/reports/risk_scores.npz``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .compute import _top_features_dae
from .io import OUTPUT_DIR, write_json_strict
from .nlg import route_explanation

logger = logging.getLogger(__name__)


def _load_risk_scores(split: str = "test") -> dict:
    """Load risk scores via the canonical split_paths resolver.

    Returns an empty dict (with a WARNING log) if Module 3 hasn't been
    run yet — previously this caught FileNotFoundError silently, which
    meant worked examples always lacked risk_context.
    """
    from common import split_paths as sp
    path = sp.risk_scores(split)
    if not path.exists():
        logger.warning(
            "Risk scores not found at %s — worked examples will lack "
            "risk_context. Run Module 3 first.",
            path,
        )
        return {}
    rd = np.load(path, allow_pickle=True)
    return {k: rd[k] for k in rd.files}


def generate_example_explanations(
    all_shap: dict,
    all_preds: dict,
    dae_preds: dict,
    weighted_err: np.ndarray,
    feat_names: list,
    y_test: np.ndarray,
    attack_cats: np.ndarray | None,
    risk_levels: np.ndarray,
    *,
    split: str = "test",
    output_dir: Path | None = None,
) -> list:
    """Generate multi-view examples for 5 alerts across all 3 stakeholders.

    Severity per example is the Module 3 canonical ``risk_level`` for the
    picked sample.
    """
    logger.info("Generating example explanations for thesis figures...")
    out_dir = output_dir or OUTPUT_DIR

    xgb_sv = all_shap["xgboost"]
    xgb_preds = all_preds["xgboost"]

    attack_idx = np.where(xgb_preds["y_pred"] == 1)[0]
    if len(attack_idx) == 0:
        return []

    # Pick 5 diverse alerts.
    sorted_by_conf = attack_idx[np.argsort(xgb_preds["y_proba"][attack_idx])[::-1]]
    picks = list(sorted_by_conf[:2])

    if attack_cats is not None:
        for cat in ["Spoofing", "Data Alteration"]:
            cat_idx = [
                i for i in attack_idx
                if str(attack_cats[i]) == cat and i not in picks
            ]
            if cat_idx:
                picks.append(cat_idx[0])

    borderline = sorted_by_conf[-1]
    if borderline not in picks:
        picks.append(borderline)

    picks = picks[:5]

    # Y3 fix: canonical risk-scores path via common.split_paths.
    risk_data = _load_risk_scores(split)

    risk_levels = np.asarray(risk_levels).astype(str)

    examples = []
    for idx in picks:
        sv_row = xgb_sv[idx]
        confidence = float(xgb_preds["y_proba"][idx])
        n_flagged = sum(
            1 for name in all_preds if all_preds[name]["y_pred"][idx] == 1
        )
        n_flagged += 1 if dae_preds["y_pred"][idx] == 1 else 0
        n_detectors = len(all_preds) + 1  # Track A models + DAE
        severity = str(risk_levels[idx])
        consensus = f"{n_flagged}/{n_detectors} detectors flagged"

        dae_top = _top_features_dae(weighted_err[idx], feat_names, k=3)

        risk_score = float(risk_data["R"][idx]) if "R" in risk_data else 0.0
        risk_comps = {}
        if "c_detect" in risk_data:
            risk_comps = {
                "c_detect":        float(risk_data["c_detect"][idx]),
                "d_crit":          float(risk_data["d_crit"][idx]),
                "s_data":          float(risk_data["s_data"][idx]),
                "d_clinical_tier": float(risk_data["d_clinical_tier"][idx]),
            }
        a_pat = (
            float(risk_data["d_clinical_tier"][idx])
            if "d_clinical_tier" in risk_data
            else 0.0
        )

        example = {
            "sample_index": int(idx),
            "ground_truth": "attack" if y_test[idx] == 1 else "benign",
            "attack_category": (
                str(attack_cats[idx]) if attack_cats is not None else "unknown"
            ),
            "views": {},
        }
        for role in ["clinician", "analyst", "administrator"]:
            example["views"][role] = route_explanation(
                int(idx), role, sv_row, feat_names,
                severity, confidence, consensus,
                risk_score, risk_comps, a_pat, dae_top,
            )

        examples.append(example)
        logger.info(
            "  Example: sample %d (%s, %s) — %s",
            idx, example["attack_category"], severity, consensus,
        )

    path = out_dir / "example_explanations.json"
    # Strict JSON (Y1): no `default=str` silent coercion. NumpyJSONEncoder
    # handles the np.float32 risk-score fields explicitly.
    from .io import NumpyJSONEncoder
    import json
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(examples, indent=2, cls=NumpyJSONEncoder))
    tmp.replace(path)
    logger.info("  Saved: example_explanations.json (%d examples)", len(examples))
    return examples


__all__ = ["generate_example_explanations", "_load_risk_scores"]
