"""RQ3 Track 5 — RQ3-lens wrapper around RQ2.c per-role + escalation.

Wraps analysis/outputs/rq2c_per_role.json with the RQ3 framing
("distributed responsibility") and folds in the escalation Chi-square
block. Does NOT recompute the per-role Mann-Whitney; reads it.

Output: analysis/outputs/rq3_user_study.json
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RQ2C = REPO_ROOT / "analysis" / "outputs" / "rq2c_per_role.json"
ESC = REPO_ROOT / "analysis" / "outputs" / "rq3_escalation.json"
OUT = REPO_ROOT / "analysis" / "outputs" / "rq3_user_study.json"


def main() -> None:
    if not RQ2C.exists():
        raise SystemExit(
            f"{RQ2C.relative_to(REPO_ROOT)} missing — run "
            "`python -m analysis.compute_rq2c_per_role` first."
        )
    if not ESC.exists():
        raise SystemExit(
            f"{ESC.relative_to(REPO_ROOT)} missing — run "
            "`python -m analysis.compute_rq3_escalation` first."
        )

    rq2c = json.loads(RQ2C.read_text())
    esc = json.loads(ESC.read_text())

    out = {
        "_meta": {
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generated_by": "analysis/compute_rq3_per_role.py",
            "research_question": (
                "RQ3 — Does the system support distributed security "
                "responsibility across hospital roles while preserving "
                "clinical safety constraints?"
            ),
            "rq3_lens": (
                "Reframes RQ2.c per-role accuracy/confidence + adds the "
                "RQ3-specific 'appropriate escalation rate' Chi-square. "
                "Same data, different framings; RQ2.c asks about "
                "explanation quality, RQ3 asks about distributed action."
            ),
            "data_source": (
                "LLM-persona simulation (gpt-4o-mini); not human study"
            ),
            "inputs": {
                "rq2c_per_role_path": str(RQ2C.relative_to(REPO_ROOT)),
                "rq3_escalation_path": str(ESC.relative_to(REPO_ROOT)),
            },
        },
        "methodology_notes": [
            "Per-role accuracy + confidence: inherited from RQ2.c "
            "(Mann-Whitney U + Cliff's delta, persona-level aggregation).",
            "Per-role escalation: Chi-square 2x2 (Fisher's exact fallback "
            "when expected cell count < 5); Cramer's V effect size.",
            "Path C data source: LLM-persona simulation; behavioural "
            "fidelity to human operators not established.",
            "Multiple-comparisons policy: raw p-values, no correction. "
            "Disclosed in both inherited (RQ2.c) and new (RQ3 escalation) "
            "blocks.",
        ],
        "limitations": list(rq2c.get("limitations") or []) + [
            "Escalation collapsed to a binary outcome "
            "(escalated_appropriately) at persona level via a >=0.5 rule. "
            "Graded escalation severity is future work.",
            "Under the real correct_action vocabulary, only "
            "correct_action == 'isolate' satisfies the escalation-class "
            "set, so the appropriate-escalation rule effectively reduces "
            "to 'correct_action is isolate AND persona escalated/isolated'. "
            "Documented in configs/rq3_escalation_definition.yaml.",
        ],
        "per_role_accuracy_confidence": rq2c.get("per_role"),
        "overall_accuracy_confidence": rq2c.get("overall"),
        "per_role_escalation": esc.get("per_role"),
        "overall_escalation": esc.get("overall"),
        "rq2c_cell_diagnostics": rq2c.get("cell_diagnostics"),
        "rq3_cell_diagnostics": esc.get("cell_diagnostics"),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"Wrote {OUT.relative_to(REPO_ROOT)}")
    overall_esc = out["overall_escalation"]
    overall_acc = (out["overall_accuracy_confidence"] or {}).get("accuracy") or {}
    print(f"Overall escalation: rate_A={overall_esc.get('rate_A')} "
          f"rate_B={overall_esc.get('rate_B')} "
          f"p={overall_esc.get('p_value')}")
    print(f"Overall accuracy:   median_A={overall_acc.get('median_A')} "
          f"median_B={overall_acc.get('median_B')} "
          f"p={overall_acc.get('p_value')}")


if __name__ == "__main__":
    main()
