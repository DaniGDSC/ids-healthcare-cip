"""Method 2 — Self-consistency variation rounds.

Two angles of self-consistency, both scoped to the existing Method 1
multi-stakeholder dataset (survey/m5_multi_role_raw.json):

A. Within-persona temporal consistency
   - Pick 9 Group-B personas (3 IT + 3 biomed + 3 nurse, deterministic).
   - Re-call all 20 alerts at temperature=0.
   - Compare to original Method-1 responses: action-match rate,
     severity-match rate.
   - Round-trip robustness check; also catches API-side variance even
     at temperature=0.

B. Cross-persona within-role consistency (no new API calls)
   - For each alert × condition (A/B), compute modal action and modal
     severity within each role.
   - Measure agreement-with-mode = fraction of personas matching the
     mode for that role.

C. Cross-role consistency on severity (no new API calls)
   - Already in m5_multi_role_result.yaml::cross_role_severity_consistency
   - Extend with action-level agreement (after mapping role-specific
     action vocabulary to a shared aggressiveness scale).

Output:
  survey/m5_self_consistency_raw.json — round-2 LLM responses
  survey/m5_self_consistency_result.yaml — final analysis
"""
from __future__ import annotations

import asyncio
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.run_llm_persona_simulation import (
    ROLE_PROMPTS, build_user_prompt, MODEL,
)


# ── API key load ─────────────────────────────────────────────────────────

def load_key() -> str:
    env_path = ROOT / ".env.local"
    for ln in env_path.read_text().splitlines():
        m = re.match(r"^OPENAI_API_KEY=(\S+)", ln.strip())
        if m: return m.group(1)
    raise SystemExit("ABORT: OPENAI_API_KEY not in .env.local")


# ── Persona selection (deterministic) ────────────────────────────────────

def select_round2_personas() -> list[tuple[str, str]]:
    """Pick 9 Group-B personas (3 per role) deterministically.

    Group B for IT generalist personas P26-P50 (after the 25 Group-A).
    Group B for biomed_engineer P16-P30 (after 15 Group-A).
    Group B for nurse_manager P11-P20 (after 10 Group-A).
    """
    return [
        # IT generalist — pick first 3 Group-B personas (P26, P27, P28)
        ("IT_generalist_P26", "IT_generalist"),
        ("IT_generalist_P27", "IT_generalist"),
        ("IT_generalist_P28", "IT_generalist"),
        # biomed_engineer — first 3 Group-B (P16, P17, P18)
        ("biomed_engineer_P16", "biomed_engineer"),
        ("biomed_engineer_P17", "biomed_engineer"),
        ("biomed_engineer_P18", "biomed_engineer"),
        # nurse_manager — first 3 Group-B (P11, P12, P13)
        ("nurse_manager_P11", "nurse_manager"),
        ("nurse_manager_P12", "nurse_manager"),
        ("nurse_manager_P13", "nurse_manager"),
    ]


# ── Async runner ─────────────────────────────────────────────────────────

async def call_one(client, sem, persona_id, role, alert):
    sys_prompt = ROLE_PROMPTS[role]
    user_prompt = build_user_prompt(alert, "B", role)
    async with sem:
        for attempt in range(8):
            try:
                resp = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=MODEL,
                        messages=[
                            {"role":"system","content":sys_prompt},
                            {"role":"user","content":user_prompt},
                        ],
                        temperature=0, max_tokens=200,
                        response_format={"type":"json_object"},
                    ), timeout=60)
                content = resp.choices[0].message.content
                parsed = json.loads(content)
                return dict(
                    persona_id=persona_id, role=role,
                    alert_id=alert["alert_id"],
                    response=parsed, raw_text=content,
                    prompt_tokens=resp.usage.prompt_tokens,
                    completion_tokens=resp.usage.completion_tokens,
                    error=None,
                )
            except Exception as e:
                if attempt < 7:
                    await asyncio.sleep(min(60, 2 ** attempt))
                else:
                    return dict(
                        persona_id=persona_id, role=role,
                        alert_id=alert["alert_id"],
                        response=None, raw_text="",
                        error=f"{type(e).__name__}: {str(e)[:120]}",
                    )


async def run_round2():
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=load_key())
    alerts = json.loads((ROOT / "results/reports/evaluation_alerts.json").read_text())
    personas = select_round2_personas()
    sem = asyncio.Semaphore(5)
    tasks = []
    for pid, role in personas:
        for alert in alerts:
            tasks.append(call_one(client, sem, pid, role, alert))
    print(f"Running {len(tasks)} round-2 calls @ concurrency=5...")
    t0 = time.perf_counter()
    results = []
    for fut in asyncio.as_completed(tasks):
        results.append(await fut)
    elapsed = time.perf_counter() - t0
    n_err = sum(1 for r in results if r.get("error"))
    print(f"Done in {elapsed:.1f}s.  errs={n_err}/{len(results)}")
    out_path = ROOT / "survey/m5_self_consistency_raw.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"Wrote {out_path}")


def main():
    asyncio.run(run_round2())


if __name__ == "__main__":
    main()
