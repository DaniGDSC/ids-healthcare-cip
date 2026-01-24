# IDS Healthcare CIP

**Intrusion Detection System for Healthcare using CIC-IDS-2018 Dataset**

## Overview

This project implements a comprehensive 5-phase intrusion detection system specifically designed for healthcare environments, ensuring HIPAA compliance and robust security monitoring.

## Phases

### Phase 1: Data Preprocessing
- Data loading and cleaning
- Train/validation/test splitting
- Normalization and scaling
- HIPAA compliance checks

### Phase 2: Feature Selection
- Information Gain analysis
- Random Forest feature importance
- Recursive Feature Elimination (RFE)
- Selection of top 35 features

### Phase 3: Autoencoder (Dimensionality Reduction)
- Deep autoencoder architecture
- Latent space compression to 8 dimensions
- Anomaly detection threshold optimization
- Model training and evaluation

### Phase 4: Clustering Analysis
- DBSCAN clustering on latent representations
- Cluster analysis and visualization
- Pattern discovery in network traffic

### Phase 5: Classification
- Ensemble classifier (SVM + Decision Tree)
- Multi-class attack classification
- Performance evaluation and reporting

## Installation

```bash
pip install -r requirements.txt
python setup.py install
```

## Usage

### Run Complete Pipeline
```bash
python scripts/run_full_pipeline.py
```

### Run Individual Phases
```bash
python src/phase1_preprocessing/run_phase1.py
python src/phase2_feature_selection/run_phase2.py
python src/phase3_autoencoder/run_phase3.py
python src/phase4_clustering/run_phase4.py
python src/phase5_classification/run_phase5.py
```

## Dataset

This project uses the **CIC-IDS-2018** dataset. Download it and place it in `data/raw/CSE-CIC-IDS2018/`.

```bash
bash scripts/download_dataset.sh
```

## Project Structure

See the complete directory structure in [docs/data_flow.md](docs/data_flow.md).

## Results

Results and reports are saved in the `results/` directory, organized by phase.

## Testing

```bash
pytest tests/
```

## License

MIT License - See [LICENSE](LICENSE) file for details.

## Author

Healthcare Security Research Team

## Citation

If you use this code in your research, please cite:
```
[Add citation information]
```
