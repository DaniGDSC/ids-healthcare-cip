"""Layer 7 v4.0 — 9-alert-types end-to-end validator.

The v4 enriched triage typology (``src.data_models.AlertType``) added
four alert types that did not exist in the legacy 5-class
``FusionClass`` vocabulary:

  KNOWN_ATTACK_UNCERTAIN, STRONG_NOVEL_ANOMALY,
  SUSPICIOUS_PATTERN, BENIGN_WATCH

This validator drives a representative input through the full v4
helper stack — Layer 3 triage classifier, Layer 4 adapter (legacy
template + adversarial flag + per-role MITRE), Layer 5 presentation
metadata, and Layer 6 tier routing — for every one of the nine alert
types and asserts:

  * every alert type is reachable from at least one synthetic input
  * the v4 helpers are total (every type produces a non-trivial output
    in every helper)
  * the cross-layer routing for ``DISAGREEMENT_ANOMALY`` is the only
    one flagged adversarial / routed to L2_SECURITY_SPECIALIST /
    coloured purple — operators key on this exclusivity

The output YAML report at
``results/reports/nine_alert_types_validation.yaml`` is the artifact
the thesis trace matrix points to for v4 typology coverage. Run via:

    python -m module6_evaluation.validate_nine_alert_types

Returns exit-code 0 on full PASS and 1 on any failure so the script
is CI-friendly.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from module3_risk_scoring.triage_v4 import classify_alert_v4  # noqa: E402
from module4_explanations.triage_v4_adapter import (  # noqa: E402
    alert_type_v4_to_legacy,
    format_mitre_for_alert_type,
    is_adversarial,
)
from module5_responses.tier_routing_v4 import (  # noqa: E402
    TierLevel,
    recommend_tier_v4,
)
from module6_evaluation.presentation_v4 import badge_for_alert_type  # noqa: E402
from src.data_models import AlertType, Confidence, OperatorRole  # noqa: E402

logger = logging.getLogger(__name__)


# ── Synthetic per-AlertType inputs ──────────────────────────────────────
#
# The triage classifier is the only stage that derives the AlertType
# from raw signals; downstream helpers consume the AlertType directly.
# We encode the prompt's prescribed predicate boundaries so each entry
# triggers exactly one stage of the 9-stage decision tree.

_SYNTHETIC_INPUTS: dict[AlertType, dict[str, float]] = {
    AlertType.KNOWN_ATTACK:           {"p_xgb": 0.95, "diversity": 0.05, "dae": 0.10},
    AlertType.KNOWN_ATTACK_UNCERTAIN: {"p_xgb": 0.95, "diversity": 0.20, "dae": 0.10},
    AlertType.DISAGREEMENT_ANOMALY:   {"p_xgb": 0.50, "diversity": 0.35, "dae": 0.80},
    AlertType.STRONG_NOVEL_ANOMALY:   {"p_xgb": 0.10, "diversity": 0.05, "dae": 0.97},
    AlertType.NOVEL_ANOMALY:          {"p_xgb": 0.10, "diversity": 0.05, "dae": 0.80},
    AlertType.CONFIRMED_ANOMALY:      {"p_xgb": 0.60, "diversity": 0.05, "dae": 0.80},
    AlertType.SUSPICIOUS_PATTERN:     {"p_xgb": 0.55, "diversity": 0.05, "dae": 0.30},
    AlertType.BENIGN_WATCH:           {"p_xgb": 0.10, "diversity": 0.05, "dae": 0.55},
    AlertType.BENIGN:                 {"p_xgb": 0.05, "diversity": 0.05, "dae": 0.10},
}


# ── Per-type validation record ──────────────────────────────────────────


@dataclass
class TypeValidation:
    """Validation record for one :class:`AlertType`."""
    alert_type: str
    triage_output: dict[str, Any] = field(default_factory=dict)
    layer4: dict[str, Any] = field(default_factory=dict)
    layer5: dict[str, Any] = field(default_factory=dict)
    layer6: dict[str, Any] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        return "FAIL" if self.failures else "PASS"


def _validate_one(alert_type: AlertType) -> TypeValidation:
    """Drive one synthetic alert through every v4 helper and record."""
    rec = TypeValidation(alert_type=alert_type.value)
    inp = _SYNTHETIC_INPUTS[alert_type]

    # Layer 3 — triage classifier must return the expected alert type.
    decision = classify_alert_v4(
        p_xgb=inp["p_xgb"], p_rf=inp["p_xgb"], p_dt=inp["p_xgb"],
        diversity_score=inp["diversity"], dae_score=inp["dae"],
    )
    rec.triage_output = {
        "input": inp,
        "alert_type": decision.alert_type.value,
        "confidence": decision.confidence.value,
        "template_id": decision.template_id,
        "c_detect": decision.c_detect,
    }
    if decision.alert_type != alert_type:
        rec.failures.append(
            f"triage classifier produced {decision.alert_type.value} "
            f"for synthetic input intended to trigger {alert_type.value}"
        )
    if decision.c_detect < inp["p_xgb"] - 1e-9:
        rec.failures.append(
            f"INVARIANT 1 violated: c_detect={decision.c_detect} < p_xgb={inp['p_xgb']}"
        )

    # Layer 4 — adapter routes to legacy template + flags adversarial.
    legacy = alert_type_v4_to_legacy(alert_type)
    if legacy not in {"T1", "T2", "T3", "T4", "T5"}:
        rec.failures.append(f"legacy template {legacy!r} is not in T1..T5")
    expected_adv = (alert_type == AlertType.DISAGREEMENT_ANOMALY)
    if is_adversarial(alert_type) is not expected_adv:
        rec.failures.append(
            f"is_adversarial returned {is_adversarial(alert_type)} for "
            f"{alert_type.value}; expected {expected_adv}"
        )
    rec.layer4 = {
        "legacy_template": legacy,
        "is_adversarial": is_adversarial(alert_type),
        "mitre_per_role": {
            OperatorRole.IT_GENERALIST.value:
                format_mitre_for_alert_type(alert_type, OperatorRole.IT_GENERALIST),
            OperatorRole.BIOMED_ENGINEER.value:
                format_mitre_for_alert_type(alert_type, OperatorRole.BIOMED_ENGINEER),
            OperatorRole.NURSE_MANAGER.value:
                format_mitre_for_alert_type(alert_type, OperatorRole.NURSE_MANAGER),
        },
    }
    # Each role's MITRE rendering must produce a non-empty string.
    for role_value, text in rec.layer4["mitre_per_role"].items():
        if not text:
            rec.failures.append(f"MITRE rendering empty for role {role_value}")

    # Layer 5 — badge metadata totality.
    badge = badge_for_alert_type(alert_type)
    rec.layer5 = {"badge": dict(badge)}
    if not badge.get("color") or not badge.get("label"):
        rec.failures.append("badge metadata missing color/label")
    # DISAGREEMENT_ANOMALY is the only purple badge.
    if alert_type == AlertType.DISAGREEMENT_ANOMALY and badge["color"] != "#9333EA":
        rec.failures.append(
            f"DISAGREEMENT_ANOMALY badge colour {badge['color']} != #9333EA"
        )
    if alert_type != AlertType.DISAGREEMENT_ANOMALY and badge["color"] == "#9333EA":
        rec.failures.append(
            f"non-disagreement type {alert_type.value} unexpectedly purple"
        )

    # Layer 6 — tier routing matches the prescribed table.
    tier_rec = recommend_tier_v4(alert_type, Confidence.MEDIUM)
    rec.layer6 = {
        "recommended_tier": tier_rec.recommended_tier.value,
        "rationale": tier_rec.rationale,
        "adversarial_flag": tier_rec.adversarial_flag,
        "requires_security_specialist": tier_rec.requires_security_specialist,
        "requires_immediate_attention": tier_rec.requires_immediate_attention,
        "fallback_options": list(tier_rec.fallback_options),
        "escalation_options": list(tier_rec.escalation_options),
    }
    expected_security = (alert_type == AlertType.DISAGREEMENT_ANOMALY)
    if tier_rec.requires_security_specialist is not expected_security:
        rec.failures.append(
            f"requires_security_specialist={tier_rec.requires_security_specialist} "
            f"for {alert_type.value}; expected {expected_security}"
        )
    if expected_security and tier_rec.recommended_tier != TierLevel.L2_SECURITY_SPECIALIST:
        rec.failures.append(
            f"DISAGREEMENT_ANOMALY routed to {tier_rec.recommended_tier.value} "
            f"instead of L2_SECURITY_SPECIALIST"
        )

    return rec


def run_validation() -> dict[str, Any]:
    """Validate every :class:`AlertType` and return a structured report."""
    per_type: list[TypeValidation] = [_validate_one(t) for t in AlertType]
    failures = [r for r in per_type if r.failures]
    pass_count = sum(1 for r in per_type if not r.failures)

    report: dict[str, Any] = {
        "format": "layer7_v4.nine_alert_types_validation",
        "format_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_alert_types": len(per_type),
            "passed": pass_count,
            "failed": len(failures),
            "pass_rate": pass_count / len(per_type) if per_type else 0.0,
            "overall_status": "PASS" if not failures else "FAIL",
        },
        "invariants_verified": [
            "Every AlertType reached by ≥1 synthetic input",
            "INVARIANT 1: c_detect ≥ p_xgb on every type",
            "Adversarial flag exclusive to DISAGREEMENT_ANOMALY",
            "L2_SECURITY_SPECIALIST routing exclusive to DISAGREEMENT_ANOMALY",
            "Purple badge (#9333EA) exclusive to DISAGREEMENT_ANOMALY",
            "Per-role MITRE rendering non-empty for all roles + types",
            "Layer 4 legacy template ∈ {T1..T5} for every type",
        ],
        "per_type": [
            {
                "alert_type": r.alert_type,
                "status": r.status,
                "failures": list(r.failures),
                "triage": r.triage_output,
                "layer4": r.layer4,
                "layer5": r.layer5,
                "layer6": r.layer6,
            }
            for r in per_type
        ],
    }
    return report


def write_report(report: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(report, sort_keys=False, default_flow_style=False))
    tmp.replace(out_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Layer 7 v4.0 — 9-alert-types end-to-end validator",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "results" / "reports" / "nine_alert_types_validation.yaml",
        help="Output YAML path "
             "(default: results/reports/nine_alert_types_validation.yaml).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    report = run_validation()
    write_report(report, args.out)

    summary = report["summary"]
    logger.info(
        "9-alert-types validation: %d / %d passed (status=%s)",
        summary["passed"], summary["total_alert_types"],
        summary["overall_status"],
    )
    logger.info("report written to %s", args.out.relative_to(PROJECT_ROOT))

    if summary["overall_status"] != "PASS":
        # Surface the failing types for CI logs.
        for entry in report["per_type"]:
            if entry["status"] == "FAIL":
                logger.error("FAIL %s: %s", entry["alert_type"], entry["failures"])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
