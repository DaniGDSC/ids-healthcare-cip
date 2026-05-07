# Module 6: Human Evaluation Protocol (RQ1/RO1, RQ3/RO3)

## 1. Study Design

**Type:** Within-subjects A/B study with Likert-scale questionnaire
**Condition A:** Alert with XAI explanation (SHAP + NLG + risk decomposition)
**Condition B:** Alert without XAI explanation (risk score + tier only)

## 2. Participants

| Role | Target N | Recruitment | Expertise Required |
|------|----------|-------------|-------------------|
| Security Analyst | 5 | SOC teams, cybersecurity researchers | IDS/SIEM experience |
| Clinician | 5 | Clinical IT, biomedical engineers | Healthcare device familiarity |
| Administrator | 5 | IT managers, hospital CISO | Risk management experience |

**Total:** 15 participants (minimum viable for Krippendorff's alpha)

## 3. Alert Set

- **20 curated alerts** spanning:
  - 4 CRITICAL (2 Spoofing, 2 Data Alteration)
  - 4 HIGH (2 Spoofing, 2 Data Alteration)
  - 4 MEDIUM (mixed)
  - 4 LOW (mixed)
  - 4 Benign calibration samples (false positives at various tiers)
- Each participant sees all 20 alerts
- 10 with XAI (Condition A), 10 without (Condition B)
- Assignment counterbalanced across participants

## 4. Evaluation Dimensions (Likert 1-5)

| Dimension | Question Stem |
|-----------|--------------|
| **Trust** | "I trust this alert classification is correct" |
| **Usefulness** | "The information provided helps me respond appropriately" |
| **Comprehensibility** | "I understand why this alert was triggered" |
| **Actionability** | "I know what action to take based on this alert" |

## 5. Task-Based Measures

| Measure | Method |
|---------|--------|
| **Decision accuracy** | Participant selects action; compare to ground-truth optimal response |
| **Decision time** | Seconds from alert display to action submission |
| **Confidence** | Self-rated 1-5 on "How confident are you in your decision?" |

## 6. Procedure

1. Briefing (5 min): System overview, IoMT context, evaluation task
2. Training (5 min): 2 practice alerts (1 with XAI, 1 without)
3. Main evaluation (30-40 min): 20 alerts in randomized order
4. Post-study questionnaire: SUS (System Usability Scale), free-text feedback
5. Debrief (5 min)

## 7. Analysis Plan

- **Inter-rater reliability:** Krippendorff's alpha on Likert scores
- **With vs Without XAI:** Paired Wilcoxon signed-rank test (non-parametric)
- **Effect size:** Cohen's d for decision accuracy and time
- **Significance threshold:** p < 0.05

## 8. Ethical Considerations

- No real patient data used (WUSTL-EHMS-2020 is synthetic/testbed)
- Participation is voluntary; informed consent required
- Data anonymized (participant IDs only)
- IRB exempt: no human subjects risk (evaluation of software interface)
