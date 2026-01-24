"""Global settings for IDS Healthcare CIP."""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"

# Random seed for reproducibility
RANDOM_SEED = 42

# Multiprocessing
N_JOBS = -1  # Use all cores

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# HIPAA Compliance
HIPAA_ENABLED = True
ENCRYPTION_KEY_PATH = os.getenv("ENCRYPTION_KEY_PATH", None)

# Dataset configuration
DATASET_NAME = "CIC-IDS-2018"
ATTACK_CLASSES = [
    "Benign",
    "FTP-BruteForce",
    "SSH-BruteForce", 
    "DoS-GoldenEye",
    "DoS-Slowloris",
    "DoS-SlowHTTPTest",
    "DoS-Hulk",
    "DDoS-LOIC-UDP",
    "DDoS-HOIC",
    "Brute-Force",
    "SQL-Injection",
    "Infiltration",
    "Bot"
]

# Memory optimization
USE_REDUCED_PRECISION = True  # Use float32 instead of float64
CHUNK_SIZE = 10000  # For chunked data processing
