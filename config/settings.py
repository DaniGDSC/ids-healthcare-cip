"""
Global settings for Healthcare IDPS
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data paths
RAW_DATA_DIR = DATA_DIR / "raw" / "CSE-CIC-IDS2018"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DATA_DIR = DATA_DIR / "features"
LATENT_DATA_DIR = DATA_DIR / "latent"
SPLITS_DIR = DATA_DIR / "splits"

# Model paths
SCALER_DIR = MODELS_DIR / "scalers"
PHASE3_MODEL_DIR = MODELS_DIR / "phase3"
PHASE5_MODEL_DIR = MODELS_DIR / "phase5"

# Results paths
PHASE1_RESULTS = RESULTS_DIR / "phase1"
PHASE2_RESULTS = RESULTS_DIR / "phase2"
PHASE3_RESULTS = RESULTS_DIR / "phase3"
PHASE4_RESULTS = RESULTS_DIR / "phase4"
PHASE5_RESULTS = RESULTS_DIR / "phase5"
REPORTS_DIR = RESULTS_DIR / "reports"

# Global settings
RANDOM_STATE = 42
N_JOBS = -1  # Use all CPU cores

# Train/Val/Test split ratios
TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
TEST_RATIO = 0.20

# HIPAA Compliance
ENABLE_HIPAA_DEIDENTIFICATION = True
PSEUDONYMIZE_IPS = True
TRUNCATE_TIMESTAMPS = True

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Performance
USE_GPU = True
GPU_MEMORY_FRACTION = 0.8  # Use 80% of GPU memory

# Visualization
DPI = 300  # High resolution for thesis figures
FIGSIZE = (12, 8)

print(f"✅ Settings loaded. Project root: {PROJECT_ROOT}")
