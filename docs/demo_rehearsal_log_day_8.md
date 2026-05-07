# Demo Rehearsal Log — Day 8

> Template the user fills in across the 4 stopwatched rehearsals.
> Pair with [demo_script_day_8.md](demo_script_day_8.md) (the
> memorizable walkthrough this log measures).
>
> The timing, smoothness, and confidence rows are intentionally blank —
> they're observations from live runs that no static analysis can fake.
> Failure-mode catalog and Q&A anticipation rows are pre-filled because
> they derive from the system's actual code paths.

---

## Pre-rehearsal sanity check

(Run once before Run 1; everything in this section was ✅ as of
2026-05-07 12:58.)

| Check | Pass? | Source of truth |
|---|---|---|
| `pytest tests/` | 503 passed, 1 skipped | `tail -5 pytest output` |
| `pyright module6_evaluation/module6_app.py` | 0 errors | pyright report |
| `run_tests.py` | `SHIP_TO_USER_STUDY` | run_tests output |
| All 5 playlist alert IDs resolve | All 5 | `_playlist_alert_ids()` round-trip |
| Beat 3 (`EVAL-3544`) triggers safety floor | Yes | `_is_safety_floor_alert()` returns True |
| Beat 4 (`SYNTHETIC_DEMO_001`) classifies DISAGREEMENT_ANOMALY | Yes | `derive_v4_fields(syn)[0]` |
| Demo bypass view target alert resolves | Yes | `_study_alert_dict_for("EVAL-0227")` non-None |

---

## Per-run log

### Run 1 — first end-to-end

**Date/time:** _____________________________
**Goal:** Just complete it. Note timing.
**Total time:** _____ : _____

| Beat | Target | Actual | Δ vs target |
|---|---:|---:|---:|
| Opening | 30 s | _____ | _____ |
| Beat 1 — Opener (`EVAL-3407`) | 90 s | _____ | _____ |
| Beat 2 — Role Switching (`EVAL-1185`) | 120 s | _____ | _____ |
| Beat 3 — Critical Safety (`EVAL-3544`) | 120 s | _____ | _____ |
| Beat 4 — Adversarial (`SYNTHETIC_DEMO_001`) | 120 s | _____ | _____ |
| Beat 5 — A/B Comparison (`EVAL-0227`) | 90 s | _____ | _____ |
| Closing | 30 s | _____ | _____ |
| **Total** | **600 s** | **_____** | **_____** |

**Rough spots observed:**
- _____________________________________________________________
- _____________________________________________________________
- _____________________________________________________________

**Smooth spots (keep doing this):**
- _____________________________________________________________
- _____________________________________________________________

**Action items for Run 2:**
- _____________________________________________________________

---

### Run 2 — after Run 1 refinements

**Date/time:** _____________________________
**Goal:** Hit 10:30. Apply Run 1 lessons.
**Total time:** _____ : _____

| Beat | Run 1 | Run 2 | Δ |
|---|---:|---:|---:|
| Opening | _____ | _____ | _____ |
| Beat 1 | _____ | _____ | _____ |
| Beat 2 | _____ | _____ | _____ |
| Beat 3 | _____ | _____ | _____ |
| Beat 4 | _____ | _____ | _____ |
| Beat 5 | _____ | _____ | _____ |
| Closing | _____ | _____ | _____ |
| **Total** | **_____** | **_____** | **_____** |

**What changed from Run 1:**
- _____________________________________________________________

---

### Run 3 — focus on smoothness

**Date/time:** _____________________________
**Goal:** 10:00. Memorized claims.
**Total time:** _____ : _____

**Transitions feel natural?** ☐ Yes ☐ No
**Comfortable without notes?** ☐ Yes ☐ No
**Confidence:** Low / Medium / High (circle one)

**Last polish needed:**
- _____________________________________________________________

---

### Run 4 — pressure simulation

**Date/time:** _____________________________
**Goal:** Run as if defense is now. Stand. Speak aloud.
**Total time:** _____ : _____

**Posture / delivery:**
- Standing? ☐ Yes ☐ No
- Speaking aloud (not whisper)? ☐ Yes ☐ No
- Imaginary examiner eye contact? ☐ Yes ☐ No

**Did you handle a simulated interruption?** ☐ Yes ☐ No
**Confidence assessment:** ☐ Ready ☐ Need one more run

---

## Rehearsal progression summary

| Run | Total | Smoothness | Confidence | Key change |
|---|---|---|---|---|
| 1 | _____ | Rough | Low | Baseline |
| 2 | _____ | _____ | _____ | _____ |
| 3 | _____ | _____ | _____ | _____ |
| 4 | _____ | _____ | _____ | _____ |

---

## Failure-mode catalog (pre-filled, do not invent recovery on the fly)

These are the eight failures the system is mostly likely to throw and
the recovery sequence for each. The pre-demo mitigation column is what
the *5-minutes-before-audience checklist* in the script covers.

| # | Failure | Detection cue | Recovery (in order) | Pre-demo mitigation |
|---|---|---|---|---|
| F1 | Streamlit server crash | Terminal stack trace; browser shows connection refused | Apologize briefly. Switch to Day-10 screen recording (if prepared). Continue narrating over video. Resume live if the server recovers. | Restart Streamlit ≤ 5 min before. Test once before audience arrives. |
| F2 | Alert doesn't load on click | Blank section under header | Refresh (F5). Click playlist button again. If still blank, click a different beat: *"Let me show another alert that demonstrates the same point."* | Pre-click each playlist button before demo. Verify all 5 render. |
| F3 | Role switch doesn't update MVE | Top bar role changes; Layer 1 doesn't re-render | Wait 2 s for Streamlit rerun. If stale, refresh and re-select role (state persists across refresh). | Test role-switching end-to-end before demo. |
| F4 | Purple badge wrong color | Beat 4 shows red or yellow instead of `#9333EA` | Don't draw attention. Continue with text narration: *"The 9-class taxonomy uses purple for DISAGREEMENT_ANOMALY..."* MVE prose still describes the disagreement. | Test `test_disagreement_anomaly_purple_unchanged` before demo (it's locked in `tests/test_v4_render_helpers.py`). |
| F5 | Demo Mode state inconsistent | Wrong number of alerts visible | Toggle off → on. If still confused, refresh page. `demo_mode` flag persists in session state. | Test toggle on each page before demo. |
| F6 | Skip Registration does nothing | Click button → registration page reappears | Verify Demo Mode banner shows (top of page). If not, toggle Demo Mode in top bar. Last-resort fallback: use Browse mode's XAI on/off toggle as a degraded A/B comparison (mention it as such). | Click Skip Registration once before demo, click Exit Demo to reset. |
| F7 | Browser tab freezes | Tab unresponsive ≥ 10 s | Close tab, reopen `localhost:8501`. Server keeps running; session state survives. Resume from current beat. | Close all other tabs before demo. Disable autoplay videos. |
| F8 | Projector goes blank | Display dies mid-demo | Check cable / port. If no recovery in 30 s, switch to laptop screen and announce once: *"We'll continue on the laptop — projector cut out."* Don't keep apologizing. | Arrive 15 min early. Test projector before audience arrives. Carry HDMI/USB-C adapter. |

**During-rehearsal observations** (fill if you trigger any failure mode in practice):

- _____________________________________________________________
- _____________________________________________________________

---

## Q&A anticipation — pre-loaded responses

(See [demo_script_day_8.md](demo_script_day_8.md) §"Q&A anticipation"
for the full table. The boxes below are for *rehearsal observations*
about each response — wording that lands vs. wording that doesn't.)

| # | Question | Response lands? | Wording to refine |
|---|---|---|---|
| Q1 | Scaling to 1000 alerts | ☐ Y / ☐ N | _________________ |
| Q2 | False-positive rate | ☐ Y / ☐ N | _________________ |
| Q3 | Can the AI be wrong | ☐ Y / ☐ N | _________________ |
| Q4 | Show audit trail | ☐ Y / ☐ N | _________________ |
| Q5 | MVE per role | ☐ Y / ☐ N | _________________ |
| Q6 | Why purple | ☐ Y / ☐ N | _________________ |
| Q7 | How evaluated | ☐ Y / ☐ N | _________________ |
| Q8 | Why no real users | ☐ Y / ☐ N | _________________ |
| Q9 | MVE generation fails | ☐ Y / ☐ N | _________________ |
| Q10 | DISAGREEMENT trigger | ☐ Y / ☐ N | _________________ |
| Q11 | DAE explanation | ☐ Y / ☐ N | _________________ |
| Q12 | Clinical validation | ☐ Y / ☐ N | _________________ |
| Q13 | Why ventilator | ☐ Y / ☐ N | _________________ |
| Q14 | Yellow on projector | ☐ Y / ☐ N | _________________ |

**Q&A delivery rules** (memorize):
- *Acknowledge first* (*"Good question..."*). Don't dive in cold.
- *Brief answer first* — 1–2 sentences, then offer to elaborate.
- *"Happy to elaborate after the demo"* is acceptable.
- *"I don't have that data point handy — I'll follow up"* is acceptable.

---

## Day-of-defense pre-demo checklist

**5 min before:**

- [ ] Restart Streamlit server.
- [ ] Pre-load all 5 playlist alerts (click each button in Browse).
- [ ] Verify Demo Mode toggle works.
- [ ] Test role switching once (IT → Biomed → Nurse → IT).
- [ ] Test Skip Registration once (click → Exit Demo).
- [ ] Verify projector shows correct colors (purple, red CRITICAL, amber SUSPICIOUS).
- [ ] Have backup screen recording open in another tab.

**1 min before:**

- [ ] Browser at full screen (F11).
- [ ] Default state: Dashboard / IT Generalist / Demo OFF.
- [ ] Stopwatch ready (or visible clock).
- [ ] Notepad nearby.
- [ ] Water within reach.
