┌──────────────────────────────────────────────────────────────────┐
│              RQ-BY-RQ FOCUS ANALYSIS                               │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│  RQ1: Risk-Adaptive Detection                                     │
│  ─────────────────────────────                                    │
│   Question text: "Can risk-adaptive scoring + dual-track          │
│   detection provide effective threat detection in IoMT            │
│   environments while preserving clinical safety constraints?"     │
│                                                                    │
│   Keywords detected:                                              │
│   ├─ "risk-adaptive scoring" → security technique                 │
│   ├─ "dual-track detection" → security architecture               │
│   ├─ "threat detection" → security objective ★                   │
│   ├─ "IoMT environments" → domain (general)                       │
│   └─ "clinical safety constraints" → IoMT specific                │
│                                                                    │
│   Lean assessment:                                                │
│   ├─ Security focus: ★★★★★ (5/5)                                 │
│   └─ IoMT general: ★★ (2/5)                                       │
│                                                                    │
│   Verdict: HEAVILY SECURITY-FOCUSED                                │
│   ├─ Contribution = security defense mechanism                    │
│   ├─ Methods = detection, fusion, adversarial                     │
│   └─ Validation = security metrics (FNR, FPR, robustness)         │
│                                                                    │
┌──────────────────────────────────────────────────────────────────┐
│              RECOMMENDED RQ2 FINAL VERSION                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│  RQ2 :                                         │
│                                                                    │
│  "Can MVE provide role-tailored security explanations enabling    │
│   non-specialist hospital stakeholders to make informed threat    │
│   triage decisions?"                                               │
│                                                                    │
│  Sub-questions:                                                    │
│                                                                    │
│  RQ2.a: Does MVE satisfy formal explainability requirements       │
│         for security operations contexts?                          │
│                                                                    │
│  RQ2.b: Are MVE explanations faithful to underlying detection     │
│         decisions and threat intelligence?                         │
│                                                                    │
│  RQ2.c: Does MVE provide differentiated triage support across     │
│         stakeholder roles (IT Generalist, Biomedical Engineer,    │
│         Nurse Manager)?                                            │
│                                                                    │
│  RQ2.d: Does MVE iteration improve triage decision support on     │
│         identified failure modes?                                  │
│                                                                    │
│  RQ2.e: Does MVE ground explanations in established threat        │
│         intelligence frameworks (MITRE ATT&CK)?                    │
└──────────────────────────────────────────────────────────────────┘ │
│                                                                    │
│  RQ3: Distributed HITL Workflow                                   │
│  ──────────────────────────────                                  │
│   Question text: "Does the system support distributed security    │
│   responsibility across hospital roles while maintaining clinical │
│   safety?"                                                         │
│                                                                    │
│   Keywords detected:                                              │
│   ├─ "distributed security responsibility" → security ★          │
│   ├─ "hospital roles" → multi-stakeholder                         │
│   ├─ "clinical safety" → IoMT specific                            │
│   └─ "system support" → architecture                              │
│                                                                    │
│   Lean assessment:                                                │
│   ├─ Security focus: ★★★★ (4/5)                                  │
│   ├─ Hospital workflow: ★★★★ (4/5)                                │
│   └─ HCI/HITL: ★★★ (3/5)                                          │
│                                                                    │
│   Verdict: SECURITY + WORKFLOW                                     │
│   ├─ Contribution = distributed security workflow                 │
│   ├─ Method = role-based authority + HITL                         │
│   └─ Validation = no-auto-execution + decision support            │
└──────────────────────────────────────────────────────────────────┘