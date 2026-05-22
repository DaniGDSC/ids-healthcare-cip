# Deploy User Study

## Local (testing)
```bash
pip install streamlit pyyaml scipy numpy
streamlit run pipeline/module6_evaluation/module6_app.py
```

## Streamlit Cloud (participants)
1. Push to GitHub (private repo)
2. Go to share.streamlit.io
3. Connect repo → select module6_app.py
4. Share URL with participants

## After study completes
```bash
python pipeline/module6_evaluation/study_analysis.py
# Output: results/reports/m5_result.yaml
```
