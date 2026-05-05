# Deploy User Study

## Local (testing)
```bash
pip install streamlit pyyaml scipy numpy
streamlit run module6_evaluation/module6_app.py
```

## Streamlit Cloud (participants)
1. Push to GitHub (private repo)
2. Go to share.streamlit.io
3. Connect repo → select module6_app.py
4. Share URL with participants

## After study completes
```bash
python module6_evaluation/study_analysis.py
# Output: survey/m5_result.yaml
```

## Run Full Batch Pipeline

```bash
python run_all_modules.py              # all modules
python run_all_modules.py --from 3     # resume from Module 3
python run_all_modules.py --only 6     # single module
```

Module sequence: 2 (Train) → 3 (Risk Scores) → 4 (Explanations) → 5 (Responses) → 5b (Pipeline) → 6 (Evaluation)

## Compute RQ2 Metrics

Run after Module 6 has produced `results/reports/evaluation_alerts.json`:

```bash
python module6_evaluation/compute_rq2_metrics.py
# Output: results/rq2_metrics.json
```

## Run RQ3 Analysis (post-collection)

```bash
python analysis/analyze_rq3.py
# Outputs: analysis/outputs/rq3_summary.json, analysis/outputs/rq3_group_stats.csv
#          analysis/plots/*.png
```

## Run Tests

```bash
python run_tests.py                       # full suite (M1–M8 + negative)
pytest tests/test_safe_failure.py         # 5 failure-mode tests
pytest tests/test_coverage_mve.py         # MVE branch coverage
```
