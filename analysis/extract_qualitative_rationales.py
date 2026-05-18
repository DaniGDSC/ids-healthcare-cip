"""Extract free-text rationales from LLM-persona responses for manual coding.

Bundles rationales by role|condition so a developer can read them as a unit and
identify recurring themes. Themes are written by hand into
survey/qualitative_themes.yaml.

Note: rationales here are LLM-generated text from the persona-simulation, not
clinician free-text. Themes therefore describe model behaviour, not human
reasoning — disclose this in the methodology field of the YAML manifest.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SURVEY_DIR = REPO_ROOT / "survey"
EXCL_PATH = SURVEY_DIR / "rq2c_exclusions.json"
OUT = SURVEY_DIR / "qualitative_rationales_for_review.json"


def _role_from_pid(pid: str) -> str:
    parts = pid.split("_")
    if parts and re.match(r"^P\d+$", parts[-1]):
        parts = parts[:-1]
    return "_".join(parts) if parts else pid


def main() -> None:
    excluded: set[str] = set()
    if EXCL_PATH.exists():
        doc = json.loads(EXCL_PATH.read_text())
        excluded = {str(e.get("persona_id")) for e in doc.get("exclusions", [])}

    by_role_cond: dict[str, list[dict]] = {}
    for path in sorted(SURVEY_DIR.glob("study_responses_*.json")):
        try:
            rec = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        pid = rec.get("persona_id", path.stem)
        if str(pid) in excluded:
            continue
        role = _role_from_pid(pid)
        for r in rec.get("rows", []):
            if r.get("error") is not None:
                continue
            resp = r.get("response")
            if not isinstance(resp, dict):
                continue
            rationale = (resp.get("rationale") or "").strip()
            if not rationale:
                continue
            key = f"{role}|{r.get('condition', 'UNKNOWN')}"
            by_role_cond.setdefault(key, []).append({
                "persona_id": pid,
                "alert_id": r.get("alert_id"),
                "action_taken": resp.get("action"),
                "correct_action": r.get("correct_action"),
                "rationale": rationale,
            })

    out = {
        "_meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generated_by": "analysis/extract_qualitative_rationales.py",
            "data_source": "LLM-persona rationales (gpt-4o-mini)",
            "instructions": (
                "Read each role|condition bucket. Identify recurring themes "
                "in the LLM rationales. Write themes into "
                "survey/qualitative_themes.yaml. Treat themes as descriptions "
                "of model behaviour, not human reasoning."
            ),
        },
        "by_role_condition": by_role_cond,
    }
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"Wrote {OUT.relative_to(REPO_ROOT)}")
    print(f"Role × Condition buckets: {len(by_role_cond)}")
    for k, v in by_role_cond.items():
        print(f"  {k}: {len(v)} rationales")


if __name__ == "__main__":
    main()
