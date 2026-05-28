"""Module 5 batch pipeline: build_all_records + provenance + drift guard."""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from common.alert_response_schema import (
    AlertResponsesEnvelope,
    InputFile,
    Provenance,
)
from common.phi import BIOMETRIC_COLUMNS

from .adaptive import build_audit_record, select_adaptive_response
from .config import MVE_LLM_FAIL_STREAK_MAX
from .loaders import PROJECT_ROOT

logger = logging.getLogger(__name__)


def build_all_records(
    risk_data: dict,
    attack_cats: np.ndarray,
    analyst_by_idx: dict,
    clinician_by_idx: dict,
    parquet_path: Path,
) -> tuple:
    """Build adaptive response records + audit trail for all non-NORMAL alerts.

    Generates a 3-layer MVE per record via ``src.mve_generator`` and
    attaches it under ``explanation.mve``. Provider chain is the default
    OpenAI → Anthropic → rule-based, with a tripwire that flips to
    force-rule-based after :data:`MVE_LLM_FAIL_STREAK_MAX` consecutive LLM
    failures so a quota outage doesn't waste a 1-2 second API attempt per
    record.
    """
    from common.device_class import (
        device_context_for_idx,
        synthesize_raw_alert,
    )
    from src.mve_generator import generate_mve

    R = risk_data["R"]
    levels = risk_data["risk_levels"]
    y_true = risk_data["y_true"]

    # M5-3: pre-cast numpy string arrays to Python lists once.
    levels_list = levels.tolist()
    cats_list = attack_cats.tolist() if attack_cats is not None else None
    active_indices = [i for i, lv in enumerate(levels_list) if lv != "NORMAL"]

    test_df = pd.read_parquet(parquet_path)

    records = []
    audit_trail = []

    llm_fail_streak = 0
    force_rule_based = False
    provider_counts: dict[str, int] = {"openai": 0, "anthropic": 0, "rule_based": 0}

    for idx in active_indices:
        level = levels_list[idx]
        cat = str(cats_list[idx]) if cats_list is not None else "unknown"
        gt = "attack" if y_true[idx] == 1 else "benign"

        bio_in_top = False
        if idx in analyst_by_idx:
            xgb_top = (
                analyst_by_idx[idx]
                .get("models", {})
                .get("xgboost", {})
                .get("top_features", [])
            )
            bio_in_top = any(f["feature"] in BIOMETRIC_COLUMNS for f in xgb_top)

        response = select_adaptive_response(
            risk_level=level,
            risk_score=float(R[idx]),
            attack_category=cat,
            biometric_in_top_features=bio_in_top,
        )

        clin_summary = ""
        if idx in clinician_by_idx:
            clin_summary = clinician_by_idx[idx]["summary"]

        device_ctx_full = device_context_for_idx(idx, test_df)
        mve_device_type = device_ctx_full["device_class"]
        if mve_device_type == "other":
            mve_device_type = "system"
        mve_device_ctx = {
            "device_type": mve_device_type,
            "criticality": device_ctx_full["device_criticality"],
            "clinical_function": device_ctx_full["affected_system"],
            "patchable": True,
        }
        raw_alert = synthesize_raw_alert(idx, cat, float(R[idx]))
        try:
            mve_out = generate_mve(
                raw_alert=raw_alert,
                device_context=mve_device_ctx,
                baseline={"baseline_days": 90},
                user_context=None,
                shap_context=None,
                event_context=None,
                force_rule_based=force_rule_based,
                risk_level=level,
            )
            mve_dict = mve_out.to_dict()
            mve_payload = {
                "layer_1": mve_dict["layer_1"],
                "layer_2": mve_dict["layer_2"],
                "layer_3": mve_dict["layer_3"],
                "why_anomalous": mve_dict["layer_1_why_anomalous"],
                "alert_involves_clinical_system": mve_dict[
                    "alert_involves_clinical_system"
                ],
                "total_word_count": mve_dict["total_word_count"],
                "provider": mve_out.provider,
            }
            provider_counts[mve_out.provider] = (
                provider_counts.get(mve_out.provider, 0) + 1
            )
            if not force_rule_based:
                if mve_out.provider == "rule_based":
                    llm_fail_streak += 1
                    if llm_fail_streak >= MVE_LLM_FAIL_STREAK_MAX:
                        force_rule_based = True
                        logger.warning(
                            "MVE LLM tripwire: %d consecutive rule-based "
                            "fallbacks — forcing rule-based for the rest of "
                            "the batch (idx=%d)",
                            llm_fail_streak, idx,
                        )
                else:
                    llm_fail_streak = 0
        except Exception as exc:
            logger.warning(
                "MVE generation failed for sample %d: %s — proceeding without MVE",
                idx, exc,
            )
            mve_payload = None

        record = {
            "sample_index": int(idx),
            "ground_truth": gt,
            "attack_category": cat,
            "risk_score": round(float(R[idx]), 4),
            "risk_level": level,
            "risk_components": {
                "C_detect": round(float(risk_data["c_detect"][idx]), 4),
                "C_track_a": round(float(risk_data["c_track_a"][idx]), 4),
                "C_track_b": round(float(risk_data["c_track_b"][idx]), 4),
                "D_crit": round(float(risk_data["d_crit"][idx]), 4),
                "S_data": round(float(risk_data["s_data"][idx]), 4),
                "D_clinical_tier": round(float(risk_data["d_clinical_tier"][idx]), 4),
            },
            "response": response,
            "explanation": {
                "clinician_summary": clin_summary,
                "analyst_available": idx in analyst_by_idx,
                "mve": mve_payload,
            },
        }
        records.append(record)

        audit = build_audit_record(
            idx, float(R[idx]), level, cat, gt, response, clin_summary,
        )
        audit_trail.append(audit)

    logger.info(
        "  MVE provider mix: openai=%d, anthropic=%d, rule_based=%d (tripwire=%s)",
        provider_counts.get("openai", 0),
        provider_counts.get("anthropic", 0),
        provider_counts.get("rule_based", 0),
        "fired" if force_rule_based else "not-fired",
    )

    return records, audit_trail


def _build_provenance(
    paths: dict,
    risk_data: dict,
    n_alerts: int,
    n_normal: int,
    filter_applied: str = "non_normal",
) -> Provenance:
    """Capture mtime/sha256 of every input + git rev + run timestamp."""
    def _stat(p: Path) -> InputFile | None:
        if not p.exists():
            return None
        b = p.read_bytes()
        return InputFile(
            path=str(p.relative_to(PROJECT_ROOT)),
            mtime_iso=datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(),
            sha256=hashlib.sha256(b).hexdigest(),
            size_bytes=len(b),
        )

    try:
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True, text=True, timeout=2, check=False,
        ).stdout.strip() or None
    except (subprocess.SubprocessError, FileNotFoundError):
        rev = None

    return Provenance(
        split=paths["split"],
        generated_at=datetime.now(timezone.utc).isoformat(),
        module5_git_rev=rev,
        n_input_samples=len(risk_data["R"]),
        n_alerts_emitted=n_alerts,
        n_normal_excluded=n_normal,
        filter_applied=filter_applied,
        inputs={
            "risk_scores_npz": _stat(paths["scores_npz"]),
            "parquet": _stat(paths["parquet"]),
            "analyst_json": _stat(paths["analyst_json"]),
            "clinician_json": _stat(paths["clinician_json"]),
        },
    )


def _assert_no_score_drift(
    records: list,
    risk_data: dict,
    tol: float = 1e-4,
) -> None:
    """Fail-loud if any record field diverges from the source npz."""
    component_map = [
        ("C_detect", "c_detect"),
        ("C_track_a", "c_track_a"),
        ("C_track_b", "c_track_b"),
        ("D_crit", "d_crit"),
        ("S_data", "s_data"),
        ("D_clinical_tier", "d_clinical_tier"),
    ]
    for rec in records:
        idx = rec["sample_index"]
        expected_R = round(float(risk_data["R"][idx]), 4)
        if abs(rec["risk_score"] - expected_R) > tol:
            raise ValueError(
                f"Score drift at sample_index={idx}: "
                f"record.risk_score={rec['risk_score']} vs "
                f"npz.R={expected_R}"
            )
        expected_level = str(risk_data["risk_levels"][idx])
        if rec["risk_level"] != expected_level:
            raise ValueError(
                f"Risk-level drift at sample_index={idx}: "
                f"record.risk_level={rec['risk_level']} vs "
                f"npz.risk_levels={expected_level}"
            )
        for rec_key, npz_key in component_map:
            expected = round(float(risk_data[npz_key][idx]), 4)
            actual = rec["risk_components"][rec_key]
            if abs(actual - expected) > tol:
                raise ValueError(
                    f"Component drift at sample_index={idx} {rec_key}: "
                    f"record={actual} vs npz={expected}"
                )


def run_one_split(split: str, sep: str = "=" * 72) -> None:
    """Execute the full pipeline for one frozen split."""
    from .effectiveness import compute_effectiveness, compute_response_stats
    from .loaders import (
        _paths,
        load_attack_categories,
        load_explanations,
        load_risk_scores,
    )
    from .plotting import (
        plot_effectiveness_by_action,
        plot_escalation_funnel,
        plot_precision_by_level,
        plot_response_distribution,
        plot_response_sankey,
    )

    paths = _paths(split)
    logger.info(sep)
    logger.info("MODULE 5 — CLOSED-LOOP RESPONSE ENGINE (RQ3/RO3) — split=%s", split)
    logger.info(sep)

    risk_data = load_risk_scores(paths["scores_npz"])
    analyst_by_idx, clinician_by_idx = load_explanations(
        paths["analyst_json"], paths["clinician_json"]
    )
    attack_cats = load_attack_categories(paths["parquet"])

    n_samples = len(risk_data["R"])
    logger.info(
        "Loaded: %d samples, %d analyst alerts, %d clinician summaries",
        n_samples, len(analyst_by_idx), len(clinician_by_idx),
    )

    logger.info("Building adaptive response records...")
    records, audit_trail = build_all_records(
        risk_data, attack_cats, analyst_by_idx, clinician_by_idx, paths["parquet"],
    )
    logger.info("  Generated %d alert-response records", len(records))

    stats = compute_response_stats(records)
    logger.info("")
    logger.info("── Response Statistics ──")
    for level in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        n = stats["alerts_by_level"].get(level, 0)
        prec = stats["precision_by_level"].get(level, 0)
        tp = stats["true_positives_by_level"].get(level, 0)
        fp = stats["false_positives_by_level"].get(level, 0)
        logger.info(
            "  %-10s %4d alerts (TP=%d, FP=%d, prec=%.2f)", level, n, tp, fp, prec
        )
    logger.info("  Actions: %s", stats["actions_triggered"])

    logger.info("")
    logger.info("── Effectiveness Analysis ──")
    effectiveness = compute_effectiveness(audit_trail)
    logger.info("  Outcomes: %s", effectiveness["outcome_distribution"])
    logger.info(
        "  Over-response (FP isolated): %d (%.1f%%)",
        effectiveness["over_response_count"],
        effectiveness["over_response_rate"] * 100,
    )
    logger.info(
        "  Under-response (attack only logged): %d (%.1f%%)",
        effectiveness["under_response_count"],
        effectiveness["under_response_rate"] * 100,
    )

    _assert_no_score_drift(records, risk_data)

    logger.info("")
    logger.info("Saving outputs...")

    n_normal = sum(
        1 for lv in risk_data["risk_levels"].tolist() if lv == "NORMAL"
    )
    provenance = _build_provenance(
        paths, risk_data, n_alerts=len(records),
        n_normal=n_normal, filter_applied="non_normal",
    )
    envelope = AlertResponsesEnvelope(_provenance=provenance, records=records)
    paths["out_alert_responses"].write_text(
        envelope.model_dump_json(by_alias=True, indent=2),
        encoding="utf-8",
    )
    logger.info(
        "  Saved: %s (%d records, envelope schema v1)",
        paths["out_alert_responses"].name, len(records),
    )

    paths["out_audit_trail"].write_text(
        json.dumps(audit_trail, indent=2), encoding="utf-8"
    )
    logger.info("  Saved: %s (%d records)",
                paths["out_audit_trail"].name, len(audit_trail))

    paths["out_effectiveness"].write_text(
        json.dumps(effectiveness, indent=2), encoding="utf-8"
    )
    logger.info("  Saved: %s", paths["out_effectiveness"].name)

    from .config import (
        ACTION_CATALOGUE,
        ATTACK_ROUTING,
        DEVICE_TIERS,
    )
    report = {
        "module": "Module 5 — Closed-Loop Response Engine (RQ3/RO3)",
        "total_samples": n_samples,
        "total_alerts": len(records),
        "statistics": stats,
        "effectiveness": effectiveness,
        "mitigation_catalogue": {
            k: v["description"] for k, v in ACTION_CATALOGUE.items()
        },
        "escalation_routing": {
            k: {kk: vv for kk, vv in v.items() if kk != "attack_specific_actions"}
            for k, v in ATTACK_ROUTING.items()
        },
        "device_constraints": DEVICE_TIERS,
    }
    paths["out_response_report"].write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    logger.info("  Saved: %s", paths["out_response_report"].name)

    rows = []
    for rec in records:
        rows.append(
            {
                "sample_index": rec["sample_index"],
                "ground_truth": rec["ground_truth"],
                "attack_category": rec["attack_category"],
                "risk_score": rec["risk_score"],
                "risk_level": rec["risk_level"],
                "actions": "|".join(rec["response"]["actions"]),
                "max_response_min": rec["response"]["max_response_min"],
                "escalation_primary": rec["response"]["escalation_chain"]["primary"],
                "device_constraint": rec["response"]["device_constraint_applied"],
                "rationale": rec["response"]["rationale"][:100],
            }
        )
    pd.DataFrame(rows).to_csv(paths["out_detail_csv"], index=False)
    logger.info("  Saved: %s", paths["out_detail_csv"].name)

    if split == "test":
        logger.info("Generating charts...")
        plot_response_distribution(records)
        plot_precision_by_level(stats)
        plot_escalation_funnel(stats)
        plot_effectiveness_by_action(effectiveness)
        plot_response_sankey(audit_trail)

    logger.info("")
    logger.info(sep)
    logger.info("SPLIT %s COMPLETE", split.upper())
    logger.info(sep)
    logger.info("  Alerts         : %d", len(records))
    logger.info("  Audit records  : %d", len(audit_trail))
    logger.info("  Over-response  : %.1f%%", effectiveness["over_response_rate"] * 100)
    logger.info("  Under-response : %.1f%%", effectiveness["under_response_rate"] * 100)
    logger.info("  Output         : %s", paths["out_alert_responses"])
    logger.info(sep)


__all__ = [
    "build_all_records",
    "_build_provenance",
    "_assert_no_score_drift",
    "run_one_split",
]
