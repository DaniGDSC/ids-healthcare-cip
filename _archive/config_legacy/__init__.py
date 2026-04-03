"""Global configuration settings for IDS Healthcare CIP."""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DATA_DIR = DATA_DIR / "features"
LATENT_DATA_DIR = DATA_DIR / "latent"
SPLITS_DATA_DIR = DATA_DIR / "splits"

# Model directories
MODELS_DIR = PROJECT_ROOT / "models"
SCALERS_DIR = MODELS_DIR / "scalers"
PHASE3_MODELS_DIR = MODELS_DIR / "phase3"
PHASE5_MODELS_DIR = MODELS_DIR / "phase5"

# Results directories
RESULTS_DIR = PROJECT_ROOT / "results"
PHASE1_RESULTS_DIR = RESULTS_DIR / "phase1"
PHASE2_RESULTS_DIR = RESULTS_DIR / "phase2"
PHASE3_RESULTS_DIR = RESULTS_DIR / "phase3"
PHASE4_RESULTS_DIR = RESULTS_DIR / "phase4"
PHASE5_RESULTS_DIR = RESULTS_DIR / "phase5"
REPORTS_DIR = RESULTS_DIR / "reports"

# Logs directory
LOGS_DIR = PROJECT_ROOT / "logs"

# Config directory
CONFIG_DIR = PROJECT_ROOT / "config"

# Global settings
RANDOM_SEED = 42
N_JOBS = -1  # Use all available cores

# Create directories if they don't exist
for directory in [
    DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, FEATURES_DATA_DIR, 
    LATENT_DATA_DIR, SPLITS_DATA_DIR, MODELS_DIR, SCALERS_DIR, 
    PHASE3_MODELS_DIR, PHASE5_MODELS_DIR, RESULTS_DIR, PHASE1_RESULTS_DIR,
    PHASE2_RESULTS_DIR, PHASE3_RESULTS_DIR, PHASE4_RESULTS_DIR, 
    PHASE5_RESULTS_DIR, REPORTS_DIR, LOGS_DIR, CONFIG_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)
