# Demo Script — 10-min Defense Walkthrough

> Memorizable script for the thesis-defense demo, grounded in the
> actual code paths and data the dashboard surfaces today.
> Source-verified against `configs/demo_playlist.yaml`, `evaluation_alerts.json`,
> and `module6_evaluation/module6_app.py` on **2026-05-07**. Re-verify
> after any change to the playlist or to `derive_v4_fields`.
>
> Companion: [demo_rehearsal_log_day_8.md](demo_rehearsal_log_day_8.md) (template the user fills in across rehearsal runs).

---

## Pre-demo setup (5 min before audience)

| ☐ | Step |
|---|---|
| ☐ | Restart Streamlit server. Cold start avoids stale session state. |
| ☐ | Open `localhost:8501`. Verify Dashboard loads (top bar visible). |
| ☐ | Pre-load all 4 pages by clicking each in the sidebar (warms cache). |
| ☐ | Pre-click each playlist button in Browse. Verifies alerts resolve. |
| ☐ | Toggle Demo Mode ON, then OFF. Verifies state machine. |
| ☐ | In Study Mode (Demo OFF), confirm registration page renders. |
| ☐ | Toggle Demo Mode ON, click Skip Registration once. Verifies bypass. |
| ☐ | Reset to landing state: Dashboard / IT Generalist / Demo OFF. |
| ☐ | Browser at full screen (F11). Stopwatch ready. |

---

## What examiners will see (the 5-beat narrative)

The 5 alerts in `configs/demo_playlist.yaml` resolve as follows when
the heuristic runs against the current eval data:

| Beat | Alert ID | Heuristic alert_type | Confidence | Safety floor? |
|---|---|---|---:|---:|
| 1 Opener | `EVAL-3407` | KNOWN_ATTACK_UNCERTAIN (red) | HIGH | no |
| 2 Role Switching | `EVAL-1185` | KNOWN_ATTACK_UNCERTAIN (red) | HIGH | no |
| 3 Critical Safety | `EVAL-3544` | KNOWN_ATTACK (red) | VERY_HIGH | **yes** |
| 4 Adversarial | `SYNTHETIC_DEMO_001` | **DISAGREEMENT_ANOMALY (purple #9333EA)** | HIGH | no |
| 5 A/B Comparison | `EVAL-0227` | CONFIRMED_ANOMALY (yellow #EAB308) | MEDIUM | no |

These are the badge colors the examiner sees at each click. Memorize
them — when the badge color matches your narration, it lands.

---

## Demo walkthrough — 10:00 budget

### 0:00 — Opening (30 s)

> **Spoken:** *"This is the IoMT Security Dashboard for our intrusion-detection thesis. Three pillars — risk-adaptive detection, stakeholder-tailored explanation, distributed human-in-the-loop workflow. I'll walk through 5 alerts."*

**Action:** Toggle **Demo Mode** in the top bar.
**Visual confirmation:** `🎬 Demo Mode — showing 5-alert playlist` banner appears across Dashboard / Sim / Browse.

---

### 0:30 — Beat 1: Opener `EVAL-3407` (90 s)

> **Spoken:** *"Standard threat. Red `KNOWN ATTACK (Uncertain)` badge — the model thinks signature-matched but isn't certain. Confidence indicator HIGH. Mode A — LLM-generated explanation."*
>
> *"Three-layer Minimum Viable Explanation: WHY anomalous, CLINICAL impact, RECOMMENDED action. Threat-intelligence line shows the MITRE technique formatted for an IT operator — `T1071 (Application Layer Protocol)` with a clickable link."*

**Actions:**
1. In Dashboard, click the first row of the Alert Feed (`EVAL-3407`).
2. Point to severity badge → 9-class type badge → confidence dots (●●● HIGH) → Mode A indicator.
3. Briefly point to the three MVE expanders (already expanded by default).
4. Read the *Threat intelligence:* line aloud.

**If it fumbles:** the row-click drill-down has a selectbox fallback baked in. If `st.dataframe.on_select` errors, the page automatically renders the legacy selectbox below the table — point and select instead.

---

### 2:00 — Beat 2: Role Switching `EVAL-1185` (120 s)

> **Spoken:** *"Same alert, three roles. Watch Layer 1 adapt."*

**Actions:**
1. Sidebar → click **Browse Alerts**.
2. Sidebar → click `▶ 2. Role Switching`. Slider snaps to `EVAL-1185`.
3. Top bar → switch role to **🖥️ IT Generalist**. Layer 1 reads: *`T1071 (Application Layer Protocol)`*.
4. Top bar → **⚕️ Biomed**. Layer 1 reads: *"Network communication consistent with attacker remote control."*
5. Top bar → **👩‍⚕️ Nurse**. Layer 1 reads: *"Equipment may be communicating with an unauthorized external system."*
6. Top bar → back to **🖥️ IT Generalist**.

> **Spoken:** *"Same threat, three different framings. The DO-NOT box is invariant across all three — clinical safety doesn't depend on role. This is RQ2 made visible."*

---

### 4:00 — Beat 3: Critical Safety Floor `EVAL-3544` (120 s)

> **Spoken:** *"CRITICAL severity, ventilator, unpatchable. INVARIANT 2 — the safety floor — guarantees this surfaces. Maintenance window doesn't bypass. Suppressed-alert logic doesn't apply."*

**Actions:**
1. Sidebar → click `▶ 3. Critical Safety`. Slider snaps to `EVAL-3544`.
2. Read the red **CRITICAL** badge aloud.
3. Read the **DO-NOT red box** aloud (will say *"DO NOT power off ventilator. Blocking port 23 at switch is SAFE."* or similar).
4. **Optional segment** (30 s, skip if running long): sidebar → **Online Simulation** → Resume playback. When `EVAL-3544` comes up, watch the auto-pause banner: *"⚠ Safety Floor Invoked — Auto-paused on a CRITICAL + unpatchable device alert."* The Resume button highlights with `▶ Resume (Safety Floor)`.

> **Spoken:** *"INVARIANT 2 isn't a slogan — it's enforced at the playback layer too."*

---

### 6:00 — Beat 4: Adversarial `SYNTHETIC_DEMO_001` (120 s)

> **Spoken:** *"V4 enrichment showcase — 9-class taxonomy beyond severity tiers."*

**Actions:**
1. Sidebar → click `▶ 4. Adversarial`. Slider snaps to `SYNTHETIC_DEMO_001`.
2. Sidebar shows: *"⚠ Synthetic alert — for demo visualisation only; not part of the evaluation set."* Read aloud (honest disclosure).
3. Point to the **purple `🟣 ADVERSARIAL DETECTED`** badge.
4. Open MVE Layer 1: *"Track A models disagree (diversity score 0.34) and DAE flags 3 features as anomalous. Pattern consistent with adversarial-input perturbation rather than a recognised attack signature."*
5. Open Layer 3: *"Coordinate with security specialist (L2)."* Read aloud.

> **Spoken:** *"This row is synthetic. Real eval data has zero adversarial cases — by construction, none of the 4 benign rows in the eval set hit CRITICAL/HIGH risk. We mark it `is_synthetic_demo: true` and load it only when Demo Mode is on. The visual category is real; this row demonstrates it."*
>
> *"Routed to L2_security_specialist tier. The v4 difference: KNOWN_ATTACK can't carry adversarial context. The purple badge can."*

---

### 8:00 — Beat 5: A/B Comparison `EVAL-0227` (90 s)

> **Spoken:** *"The research question: does the MVE actually help operators?"*

**Actions:**
1. Sidebar → **Study Mode**.
2. Click **⏭ Skip Registration (Demo Only)**.
3. The bypass view loads with a red banner: *"⚠ DEMO ONLY — registration bypassed; no study data is collected."* Read aloud.
4. Click **Group A — Raw IDS (control)** tab. Show: severity badge + plain text in a code block.
5. Click **Group B — With MVE (treatment)** tab. Show: 9-class badge, threat-intelligence line, and the locked Phase-2 stimulus prose.

> **Spoken:** *"Same alert. Group A: raw output. Operator reasons from scratch. Group B: 9-class badge, MITRE per role, prominent DO-NOT, three-layer MVE."*
>
> *"Method-1 LLM-persona simulation: +60.8% composite-accuracy improvement for IT generalist. Wilcoxon p < 1e-6. Cohen's h = 0.43 — medium-large effect."*
>
> *"This is the value proposition, made visible."*

---

### 9:30 — Closing (30 s)

> **Spoken:** *"Every operator decision lands in an append-only audit trail. SHA-256 hash chain. INVARIANT 4 — and it's visible at the bottom of every page."*

**Actions:**
1. Click **← Exit Demo** to leave the bypass.
2. Sidebar → **Dashboard**. Scroll to bottom.
3. Click **📋 Last 5 Decisions (N)** expander.
4. Point to the **✓ Chain valid** badge.

> **Spoken:** *"Three pillars: risk-adaptive detection, stakeholder-tailored explanation, distributed HITL. Happy to take questions."*

---

## Transition crib sheet

The 6 transitions, ranked by failure risk:

| # | From → To | Click sequence | Risk |
|---|---|---|---|
| 1 | Opening → Beat 1 | Top bar Demo toggle → first row in feed | Low |
| 2 | Beat 1 → Beat 2 | Sidebar Browse → Beat 2 button | Medium (Browse mode warm-up) |
| 3 | Beat 2 → Beat 3 | Beat 3 button (already in Browse) | Low |
| 4 | Beat 3 → Beat 4 | Beat 4 button (already in Browse) | Low |
| 5 | Beat 4 → Beat 5 | Sidebar Study → Skip Registration | Medium (two clicks, button below fold possible) |
| 6 | Beat 5 → Closing | Exit Demo → sidebar Dashboard → scroll → expand panel | Low |

**Rule:** if a transition stalls > 3 s, skip the optional segment and continue with narrative. Time is more valuable than completeness.

---

## Q&A anticipation — defense-relevant questions

(Pre-loaded responses; tune for delivery during rehearsal.)

| # | Question | 30-second answer outline |
|---|---|---|
| Q1 | "How does this scale to 1000 alerts?" | Backend p95 = 217 ms (`results/reports/online_latency_profile.json`). Operator throughput is the bottleneck, not detection. Recommendation-only architecture (INVARIANT 3). Phase 3: production deployment. |
| Q2 | "What's the false-positive rate?" | F1 = 0.892, AUC = 0.994 on test set. Risk-adaptive scoring uses clinical context to suppress at Module 3 only — never at Track B. Visible in System Diagnostics → Thresholds. |
| Q3 | "Can the AI be wrong?" | Yes. INVARIANT 3: no auto-execution; recommendation-only. DISAGREEMENT_ANOMALY is the visual cue when models disagree. Mode B rule-based fallback when LLM unavailable. |
| Q4 | "Show me the audit trail." | Already in plan — Last 5 Decisions panel at bottom of any page. Append-only, SHA-256 chain. *"✓ Chain valid"* badge confirms no tampering. 7-year HIPAA retention. |
| Q5 | "How does the MVE change per role?" | Same Layer 1 *why*, different Layer 3 *actions*. MITRE technique reformatted: `T1071 (Application Layer Protocol)` for IT, threat-type prose for Biomed, plain-language sentence for Nurse. DO-NOT preserved across all three. Beat 2 demonstrates this live if asked again. |
| Q6 | "Why purple for adversarial?" | Distinct from the threat-tone reds and oranges. Test: `test_only_disagreement_anomaly_is_purple` locks `#9333EA` and refuses any other class to use it. Routes to L2_security_specialist tier. |
| Q7 | "How was this evaluated?" | Multi-method, no real users (IRB-free for thesis timeline). Method 1: 100 LLM personas × 20 alerts × A/B = 4000 simulated decisions. Heuristic eval (Nielsen + DARPA + NIST + HFMEA). Case study with realistic alerts. Formal compliance check (22 REQ-MVE). |
| Q8 | "Why no real users?" | IRB approval + thesis-defense timeline incompatible. Multi-method evaluation provides converging evidence. Phase 2: real-user study post-defense. M5 result: +60.8% in IT decisions, p < 1e-6. |
| Q9 | "What if MVE generation fails?" | Mode B fallback — rule-based templates with no LLM call. Triggered when `ANTHROPIC_API_KEY` unset OR LLM call times out. Top bar shows ⚠ Rule-based (orange) instead of ✓ AI Mode (green). All M1–M8 tests pass without LLM. |
| Q10 | "When does DISAGREEMENT_ANOMALY trigger?" | `diversity_score >= 0.30 AND DAE_score >= 0.70`. Three Track-A models disagree + Track-B autoencoder anomaly. Routes to L2_security_specialist. |
| Q11 | "What is the DAE doing?" | Denoising Autoencoder, 28-dim cascade input (25 raw features + 3 Track-A probas). Trained on benign-only. Reconstruction error = anomaly signal. Per-dim error visible in Dashboard's DAE expander. |
| Q12 | "Have clinicians validated this?" | MVE templates derived from clinical literature. DO-NOT statements from FDA medical-device guidance. Phase-2 clinical user study planned. Acknowledged limitation: no real clinician feedback yet. |
| Q13 | "Why ventilator as the safety-floor case?" | Highest-criticality medical device class in the IoMT threat model. Often unpatchable due to FDA recertification. Direct patient-safety impact. Beat 3's `EVAL-3544` is real eval data, not synthetic. |
| Q14 | "What if the projector washes out the yellow?" | Day-7 polish moved SUSPICIOUS_PATTERN from `#FACC15` → `#F59E0B` (amber) precisely for this. Test `test_suspicious_pattern_is_amber_not_yellow` locks the new value. CONFIRMED_ANOMALY stays `#EAB308`. |

**Delivery rules:**
- *Acknowledge the question* — don't launch into the answer with no preamble.
- *Brief answer first* — 1–2 sentences. Then offer to elaborate.
- *"Happy to elaborate after the demo"* is acceptable when the answer is long.
- *"I don't have that data point handy — I'll follow up"* is acceptable. Examiners respect honesty.

---

## Failure-mode backup paths

(Pre-loaded; do not invent recovery on the fly.)

| # | Failure | Detection | Recovery (in order) |
|---|---|---|---|
| F1 | Streamlit server crash | App unresponsive, terminal traceback | Apologize briefly. If you have the Day-10 screen recording, switch to it and continue narrating. |
| F2 | Alert doesn't load on click | Page shows blank or "no alerts" | Refresh (F5). Try the playlist button again. If still blank, click a different beat and adapt: *"Let me show another alert that demonstrates the same point."* |
| F3 | Role switch lands but page doesn't update | Top bar role changes; MVE doesn't | Wait 2 s for Streamlit rerun. If still stale, refresh and re-select. The session-state role survives a refresh. |
| F4 | Purple badge missing on Beat 4 | Badge shows red or yellow instead of purple | Don't draw attention. Continue with text narration: *"The 9-class taxonomy uses purple for DISAGREEMENT_ANOMALY — the adversarial category."* The MVE prose still describes the disagreement. |
| F5 | Demo Mode toggle inconsistent | Wrong number of alerts visible | Toggle off, then on. If still confused, refresh page. Demo Mode flag persists in session state. |
| F6 | Skip Registration button does nothing | Click button, registration page reappears | Verify Demo Mode banner is showing (top of page). If not, top bar → toggle Demo Mode again. Last-resort fallback: use Browse mode's XAI on/off toggle as a poor-man's A/B (mention it as such). |
| F7 | Browser freezes | Tab unresponsive for 10+ seconds | Close tab, reopen `localhost:8501`. Streamlit's session state (role, demo_mode) survives the tab close as long as the server keeps running. Resume from the beat you were on. |
| F8 | Projector goes blank | Display dies | Check cable. If you can't recover in 30 s, switch to laptop screen and apologize once: *"We'll continue on the laptop — the projector cut out."* Don't keep apologizing. |

---

## Final memorize-this list

Five claims, in order. If you forget everything else, hit these:

1. *"9-class taxonomy — KNOWN, UNCERTAIN, DISAGREEMENT, NOVEL, CONFIRMED, SUSPICIOUS, BENIGN_WATCH, BENIGN — beyond severity tiers."*
2. *"Same alert, three roles — IT, Biomed, Nurse — three different framings. Same DO-NOT."*
3. *"INVARIANT 2: CRITICAL + unpatchable always surfaces. Sim auto-pauses on it."*
4. *"Purple badge — DISAGREEMENT_ANOMALY — diversity ≥ 0.30 + DAE ≥ 0.70. Adversarial."*
5. *"+60.8% IT-operator improvement. p < 1e-6. h = 0.43. Wilcoxon."*
