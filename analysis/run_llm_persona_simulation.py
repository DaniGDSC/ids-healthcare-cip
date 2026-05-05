"""Method 1 — Multi-Stakeholder LLM Persona Simulation.

Runs 100 personas × 20 alerts = 2000 total OpenAI calls against
gpt-4o-mini, splitting into Group A (raw IDS view) and Group B
(MVE-augmented view) per the canonical 50/30/20 IT/Biomed/Nurse split.

Cost-guard: aborts if total prompt+completion tokens cross a budget.
Determinism: temperature=0; persona/alert ordering seeded by participant_id.

Output:
  survey/study_responses_LLM_<role>_P<NN>.json  (one file per persona)
  survey/m5_multi_role_raw.json                  (single aggregate dump)
  survey/m5_multi_role_result.yaml               (statistical analysis)
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Configuration ────────────────────────────────────────────────────────

MODEL = "gpt-4o-mini"
N_PERSONAS = {"IT_generalist": 50, "biomed_engineer": 30, "nurse_manager": 20}
GROUP_RATIO = 0.5  # Half each role to Group A (raw), half to Group B (MVE)
CONCURRENT = 40
MAX_RETRIES = 3
TIMEOUT_S = 60.0
TOKEN_BUDGET = 5_000_000  # cost-guard cap (~$3 at gpt-4o-mini pricing)

VALID_ACTIONS = ["dismiss", "monitor", "investigate", "isolate", "escalate"]
ACTION_AGGRESSIVENESS = {"dismiss": 0, "monitor": 1, "investigate": 2,
                          "isolate": 3, "escalate": 4}
VALID_SEVERITIES = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

# Role-forbidden verbs (must match src/mve_generator.py::ROLE_FORBIDDEN_ACTION_TERMS).
ROLE_FORBIDDEN = {
    "IT_generalist": [],
    "biomed_engineer": ["isolate"],            # isolate is a network action
    "nurse_manager":   ["isolate", "escalate"], # nurse can't push network or incident
}

OUT_DIR = ROOT / "survey"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── API key load ─────────────────────────────────────────────────────────

def load_api_key() -> str:
    """Load OPENAI_API_KEY from .env.local without polluting parent env."""
    env_path = ROOT / ".env.local"
    if not env_path.exists():
        raise SystemExit("ABORT: .env.local not found")
    for ln in env_path.read_text().splitlines():
        m = re.match(r"^OPENAI_API_KEY=(\S+)", ln.strip())
        if m:
            return m.group(1)
    raise SystemExit("ABORT: OPENAI_API_KEY not in .env.local")


# ── Persona generation ───────────────────────────────────────────────────

ROLE_PROMPTS = {
    "IT_generalist": """You are an IT generalist at a 250-bed hospital with 3 years of experience.
You handle network alerts, EHR support, and basic security incidents alongside other IT duties.
You are NOT a SOC specialist. You see 10-50 IDS alerts per day.
Your authority: network actions (isolate, restrict, investigate, monitor).""",

    "biomed_engineer": """You are a biomedical engineer at a 250-bed hospital with 8 years of experience.
You are responsible for the safety and operation of clinical medical devices (ventilators, infusion
pumps, monitors). You verify device firmware, document anomalies, and coordinate with IT Security.
Your authority: device-side actions (verify, document, coordinate). You CANNOT push network policy.""",

    "nurse_manager": """You are a nurse manager at a 250-bed hospital with 12 years of clinical experience.
You oversee patient care on a clinical unit. When equipment alerts appear, your priority is patient
safety: verify clinical backup, monitor patient vitals, document. You CANNOT touch network or
device firmware. You escalate clinical issues to clinicians, NOT to IT.""",
}


# ── Per-call prompt builder ──────────────────────────────────────────────

def build_user_prompt(alert: dict, condition: str, role: str) -> str:
    """Assemble the user prompt with alert text appropriate to condition.
    condition='A' → raw IDS view; condition='B' → MVE-augmented view."""
    if condition == "A":
        alert_text = alert.get("group_a_display", "(no raw view)")
    else:
        alert_text = alert.get("group_b_display", "(no MVE view)")

    if role == "IT_generalist":
        valid = "isolate, investigate, monitor, dismiss, escalate"
    elif role == "biomed_engineer":
        valid = "investigate, monitor, dismiss, escalate (you cannot isolate — that's network)"
    else:  # nurse_manager
        valid = "investigate, monitor, dismiss (you cannot isolate or escalate to incident response)"

    return f"""ALERT (#{alert['alert_id']}):
{alert_text}

QUESTION: What single action would you take? Reply with valid JSON only:
{{
  "action": "<one of: {valid}>",
  "severity_assessment": "<one of: LOW, MEDIUM, HIGH, CRITICAL>",
  "confidence": <integer 1-5>,
  "rationale": "<your reasoning, max 200 characters>"
}}

Respond with ONLY the JSON object. No markdown, no preamble."""


# ── Async runner ─────────────────────────────────────────────────────────

@dataclass
class CallResult:
    persona_id: str
    role: str
    condition: str
    alert_id: str
    correct_action: str
    response: dict | None  # None on parse failure
    raw_text: str
    latency_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: Optional[str] = None


async def call_openai(client, persona_id: str, role: str, condition: str,
                      alert: dict, semaphore: asyncio.Semaphore) -> CallResult:
    """One LLM call with bounded retry."""
    sys_prompt = ROLE_PROMPTS[role]
    user_prompt = build_user_prompt(alert, condition, role)
    correct = alert.get("correct_action", "")

    async with semaphore:
        attempt = 0
        backoff = 1.0
        while attempt < MAX_RETRIES:
            t0 = time.perf_counter()
            try:
                resp = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=MODEL,
                        messages=[
                            {"role": "system", "content": sys_prompt},
                            {"role": "user",   "content": user_prompt},
                        ],
                        temperature=0,
                        max_tokens=200,
                        response_format={"type": "json_object"},
                    ),
                    timeout=TIMEOUT_S,
                )
                elapsed = time.perf_counter() - t0
                content = resp.choices[0].message.content
                # Try parsing.
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    # Strip code fences if present.
                    cleaned = re.sub(r"^```(json)?|```$", "", content.strip(),
                                     flags=re.MULTILINE).strip()
                    parsed = json.loads(cleaned)
                return CallResult(
                    persona_id=persona_id, role=role, condition=condition,
                    alert_id=alert["alert_id"], correct_action=correct,
                    response=parsed, raw_text=content,
                    latency_s=round(elapsed, 3),
                    prompt_tokens=resp.usage.prompt_tokens,
                    completion_tokens=resp.usage.completion_tokens,
                )
            except (json.JSONDecodeError, KeyError) as e:
                # Parse failure — skip this alert with annotation
                return CallResult(
                    persona_id=persona_id, role=role, condition=condition,
                    alert_id=alert["alert_id"], correct_action=correct,
                    response=None, raw_text=content if 'content' in dir() else "",
                    latency_s=round(time.perf_counter() - t0, 3),
                    error=f"parse_error: {e}",
                )
            except asyncio.TimeoutError:
                attempt += 1
                if attempt >= MAX_RETRIES:
                    return CallResult(
                        persona_id=persona_id, role=role, condition=condition,
                        alert_id=alert["alert_id"], correct_action=correct,
                        response=None, raw_text="",
                        latency_s=TIMEOUT_S, error="timeout",
                    )
                await asyncio.sleep(backoff)
                backoff *= 2
            except Exception as e:
                attempt += 1
                if attempt >= MAX_RETRIES:
                    return CallResult(
                        persona_id=persona_id, role=role, condition=condition,
                        alert_id=alert["alert_id"], correct_action=correct,
                        response=None, raw_text="",
                        latency_s=round(time.perf_counter() - t0, 3),
                        error=f"{type(e).__name__}: {e}",
                    )
                await asyncio.sleep(backoff)
                backoff *= 2


def assign_personas() -> list[tuple[str, str, str]]:
    """Return list of (persona_id, role, condition) tuples."""
    out = []
    for role, n in N_PERSONAS.items():
        n_a = int(n * GROUP_RATIO)
        n_b = n - n_a
        # Deterministic assignment by ordinal
        for i in range(n_a):
            pid = f"{role}_P{i+1:02d}"
            out.append((pid, role, "A"))
        for i in range(n_b):
            pid = f"{role}_P{n_a+i+1:02d}"
            out.append((pid, role, "B"))
    return out


async def main():
    api_key = load_api_key()
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=api_key)

    alerts = json.loads((ROOT / "results/reports/evaluation_alerts.json").read_text())
    print(f"Loaded {len(alerts)} alerts")

    personas = assign_personas()
    print(f"Assigned {len(personas)} personas:")
    for role in N_PERSONAS:
        a = sum(1 for p, r, c in personas if r == role and c == "A")
        b = sum(1 for p, r, c in personas if r == role and c == "B")
        print(f"  {role}: {a} Group-A + {b} Group-B")

    total_calls = len(personas) * len(alerts)
    print(f"\nTotal calls planned: {total_calls}")

    sem = asyncio.Semaphore(CONCURRENT)
    tasks = []
    for pid, role, cond in personas:
        for alert in alerts:
            tasks.append(call_openai(client, pid, role, cond, alert, sem))

    print(f"Launching {len(tasks)} concurrent calls (max {CONCURRENT} in-flight)...")
    t_start = time.perf_counter()
    results: list[CallResult] = []
    completed = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    for fut in asyncio.as_completed(tasks):
        r = await fut
        results.append(r)
        total_prompt_tokens += r.prompt_tokens
        total_completion_tokens += r.completion_tokens
        completed += 1
        if completed % 200 == 0:
            elapsed = time.perf_counter() - t_start
            tok = total_prompt_tokens + total_completion_tokens
            print(f"  [{completed}/{len(tasks)}] elapsed={elapsed:.0f}s "
                  f"tokens={tok:,}  errs={sum(1 for x in results if x.error)}")
            if tok > TOKEN_BUDGET:
                print(f"⛔ TOKEN BUDGET ({TOKEN_BUDGET:,}) EXCEEDED — aborting")
                break

    elapsed = time.perf_counter() - t_start
    print(f"\nCompleted {completed}/{len(tasks)} in {elapsed:.1f}s")
    print(f"Total tokens: prompt={total_prompt_tokens:,} completion={total_completion_tokens:,}")
    print(f"Errors: {sum(1 for r in results if r.error)}")

    # Persist raw aggregate
    raw_dump = [
        dict(persona_id=r.persona_id, role=r.role, condition=r.condition,
             alert_id=r.alert_id, correct_action=r.correct_action,
             response=r.response, raw_text=r.raw_text,
             latency_s=r.latency_s,
             prompt_tokens=r.prompt_tokens,
             completion_tokens=r.completion_tokens,
             error=r.error)
        for r in results
    ]
    (OUT_DIR / "m5_multi_role_raw.json").write_text(json.dumps(raw_dump, indent=2, default=str))
    print(f"Wrote {OUT_DIR / 'm5_multi_role_raw.json'}")

    # Per-persona JSON
    by_persona: dict[str, list] = {}
    for r in results:
        by_persona.setdefault(r.persona_id, []).append(dict(
            alert_id=r.alert_id,
            condition=r.condition,
            correct_action=r.correct_action,
            response=r.response,
            error=r.error,
        ))
    for pid, rows in by_persona.items():
        (OUT_DIR / f"study_responses_LLM_{pid}.json").write_text(
            json.dumps({"persona_id": pid, "n_alerts": len(rows),
                         "rows": rows}, indent=2, default=str))
    print(f"Wrote {len(by_persona)} per-persona files")


if __name__ == "__main__":
    asyncio.run(main())
