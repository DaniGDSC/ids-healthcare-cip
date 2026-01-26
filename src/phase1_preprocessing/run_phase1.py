"""Main script to run Phase 1: Data Preprocessing."""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import yaml
import numpy as np
import pandas as pd
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import RANDOM_STATE
from src.phase1_preprocessing import (
    DataLoader, DataCleaner, DataSplitter, 
    Normalizer, HIPAACompliance
)

# Default label mapping to reduce CIC-IDS-2018 labels to 7 categories
DEFAULT_LABEL_MAPPING: Dict[str, str] = {
    "Benign": "Benign",
    "Bot": "Bot",
    "Infiltration": "Infiltration",
    "FTP-BruteForce": "BruteForce",
    "SSH-Bruteforce": "BruteForce",
    "DDoS attacks-LOIC-HTTP": "DDoS",
    "DDoS attacks-LOIC-UDP": "DDoS",
    "DDoS attack-HOIC": "DDoS",
    "DoS attacks-SlowHTTPTest": "DoS",
    "DoS attacks-Hulk": "DoS",
    "DoS attacks-GoldenEye": "DoS",
    "DoS attacks-Slowloris": "DoS",
    "Brute Force -Web": "WebAttack",
    "Brute Force -XSS": "WebAttack",
    "SQL Injection": "WebAttack",
}

# Default PHI columns to drop for de-identification
DEFAULT_PHI_COLUMNS: List[str] = [
    "Src IP",
    "Dst IP",
    "Source IP",
    "Destination IP",
    "Timestamp",
    "Flow Timestamp",
    "Source Port",
    "Destination Port",
]


def setup_logging(log_file: str):
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class Phase1Pipeline:
    """Phase 1 preprocessing pipeline organized into testable steps."""

    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.data_dir = project_root / config['data']['input_dir']
        self.output_dir = project_root / config['data']['output_dir']
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.label_col = config['data']['stratify_column']
        self.corr_removed: List[str] = []
        self.dropped_counts: Dict[str, int] = {}
        self.mapping = config.get('label_normalization', {}).get('mapping', DEFAULT_LABEL_MAPPING)

    @staticmethod
    def load_config(config_path: Path) -> dict:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    @staticmethod
    def setup_logging(log_file: Path):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def _load_data(self) -> pd.DataFrame:
        self.logger.info("\n--- Step 1: Loading Data ---")
        loader = DataLoader(self.data_dir)
        df = loader.load_csv_files()
        self.logger.info(f"Loaded dataset shape: {df.shape}")
        return df

    def _hipaa_deid(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.config['hipaa']['enabled']:
            return df

        self.logger.info("\n--- Step 3: HIPAA De-identification ---")
        hipaa = HIPAACompliance(enabled=True)

        if self.config['hipaa'].get('anonymize_ips', True):
            ip_columns = ['Src IP', 'Dst IP', 'Source IP', 'Destination IP']
            df = hipaa.anonymize_ip_addresses(df, ip_columns)

        if self.config['hipaa'].get('remove_phi_columns', True):
            phi_columns = self.config['hipaa'].get('remove_columns', DEFAULT_PHI_COLUMNS)
            df = hipaa.remove_phi_columns(df, phi_columns)
            self.logger.info(f"Removed PHI columns: {phi_columns}")

        if self.config['hipaa'].get('log_access', True):
            hipaa.log_data_access(
                user='phase1_preprocessing',
                action='load',
                data_description='CIC-IDS-2018 dataset',
                record_count=len(df)
            )
        return df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("\n--- Step 2: Data Cleaning ---")
        cleaner = DataCleaner(
            drop_na_threshold=self.config['preprocessing']['drop_na_threshold'],
            fill_strategy=self.config['preprocessing']['fill_strategy'],
            replace_inf=self.config['preprocessing']['replace_inf'],
            inf_replacement_value=self.config['preprocessing'].get('inf_replacement_value', 0.0),
            remove_duplicates=self.config['preprocessing'].get('remove_duplicates', True),
            remove_constant_features=self.config['preprocessing'].get('remove_constant_features', True),
            constant_threshold=self.config['preprocessing'].get('constant_threshold', 0.01)
        )
        df = cleaner.clean(df)
        self.logger.info(f"Post-cleaning shape: {df.shape}")
        return df

    def _correlation_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        corr_cfg = self.config.get('correlation_filter', {})
        if not corr_cfg.get('enabled', True):
            return df

        self.logger.info("\n--- Step 4: Correlation Filter ---")
        feature_df = df.drop(columns=[self.label_col], errors='ignore').select_dtypes(include=[np.number])
        if feature_df.empty:
            return df

        corr_matrix = feature_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.corr_removed = [col for col in upper.columns if any(upper[col] > corr_cfg.get('threshold', 0.95))]

        if self.corr_removed:
            df = df.drop(columns=self.corr_removed)
            self.logger.info(f"Removed {len(self.corr_removed)} features with high correlation")
            self.logger.info(f"Correlated features dropped: {self.corr_removed}")
        return df

    def _normalize_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("\n--- Step 5: Label Normalization ---")
        if self.label_col not in df.columns:
            raise ValueError(f"Label column '{self.label_col}' not found")

        label_cfg = self.config.get('label_normalization', {})
        drop_unknown = label_cfg.get('drop_unknown', True)

        normalized = df[self.label_col].map(self.mapping)
        if drop_unknown:
            unknown_mask = normalized.isna()
            dropped = int(unknown_mask.sum())
            if dropped > 0:
                self.dropped_counts['unknown_labels_dropped'] = dropped
                df = df.loc[~unknown_mask].copy()
                normalized = normalized.loc[~unknown_mask]

        df[self.label_col] = normalized
        self.logger.info(f"Labels normalized to {len(set(self.mapping.values()))} categories")
        if self.dropped_counts:
            self.logger.warning(f"Dropped {self.dropped_counts.get('unknown_labels_dropped', 0)} rows with unknown labels")
        self.logger.info(f"Post-label-normalization shape: {df.shape}")
        return df

    def _separate_features_labels(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        self.logger.info("\n--- Step 6: Separating Features and Labels ---")
        y = df[self.label_col].values
        X_df = df.drop(columns=[self.label_col])
        X = X_df.values
        feature_names = X_df.columns.tolist()
        self.logger.info(f"Features shape: {X.shape}")
        self.logger.info(f"Labels shape: {y.shape}")
        return X, y, feature_names

    def _split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, ...]:
        self.logger.info("\n--- Step 7: Splitting Data ---")
        splitter = DataSplitter(
            train_ratio=self.config['data']['train_ratio'],
            val_ratio=self.config['data']['val_ratio'],
            test_ratio=self.config['data']['test_ratio'],
            random_state=RANDOM_STATE,
            stratify=self.config['data']['stratify']
        )
        return splitter.split(X, y)

    def _normalize_features(self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.logger.info("\n--- Step 7: Normalization ---")
        normalizer = Normalizer(method=self.config['preprocessing']['normalization_method'])
        X_train_norm = normalizer.fit_transform(X_train)
        X_val_norm = normalizer.transform(X_val)
        X_test_norm = normalizer.transform(X_test)

        scaler_path = project_root / "models" / "scalers" / "standard_scaler.pkl"
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        normalizer.save(scaler_path)
        return X_train_norm, X_val_norm, X_test_norm

    def _save_processed(self, X_train_norm: np.ndarray, y_train: np.ndarray,
                        X_val_norm: np.ndarray, y_val: np.ndarray,
                        X_test_norm: np.ndarray, y_test: np.ndarray):
        self.logger.info("\n--- Step 8: Save Outputs ---")
        if self.config['output']['save_format'] != 'npz':
            return
        np.savez_compressed(self.output_dir / 'train_normalized.npz', X=X_train_norm, y=y_train)
        np.savez_compressed(self.output_dir / 'val_normalized.npz', X=X_val_norm, y=y_val)
        np.savez_compressed(self.output_dir / 'test_normalized.npz', X=X_test_norm, y=y_test)

    def _save_metadata(self, feature_names: List[str], X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
                       y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray):
        if not self.config['output']['save_metadata']:
            return
        metadata = {
            'feature_names': feature_names,
            'n_features': len(feature_names),
            'n_train_samples': len(X_train),
            'n_val_samples': len(X_val),
            'n_test_samples': len(X_test),
            'normalization_method': self.config['preprocessing']['normalization_method'],
            'label_mapping': self.mapping,
            'correlated_features_removed': self.corr_removed,
            'dropped_counts': self.dropped_counts,
            'label_distribution': {
                'train': {str(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))},
                'val': {str(k): int(v) for k, v in zip(*np.unique(y_val, return_counts=True))},
                'test': {str(k): int(v) for k, v in zip(*np.unique(y_test, return_counts=True))}
            }
        }
        metadata_path = self.output_dir / 'phase1_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        self.logger.info(f"Metadata saved to {metadata_path}")

    def run(self):
        df = self._load_data()
        df = self._clean_data(df)
        df = self._hipaa_deid(df)
        df = self._correlation_filter(df)
        df = self._normalize_labels(df)
        X, y, feature_names = self._separate_features_labels(df)
        X_train, X_val, X_test, y_train, y_val, y_test = self._split_data(X, y)
        X_train_norm, X_val_norm, X_test_norm = self._normalize_features(X_train, X_val, X_test)
        self._save_processed(X_train_norm, y_train, X_val_norm, y_val, X_test_norm, y_test)
        self._save_metadata(feature_names, X_train, X_val, X_test, y_train, y_val, y_test)
        self.logger.info("\n" + "="*80)
        self.logger.info("PHASE 1 COMPLETE!")
        self.logger.info("="*80)
        self.logger.info(f"\nOutput files saved to: {self.output_dir}")


def main():
    config_path = project_root / "config" / "phase1_config.yaml"
    config = Phase1Pipeline.load_config(config_path)

    log_file = project_root / config['logging']['file']
    Phase1Pipeline.setup_logging(log_file)
    logger = logging.getLogger(__name__)

    logger.info("="*80)
    logger.info("PHASE 1: DATA PREPROCESSING")
    logger.info("="*80)

    try:
        pipeline = Phase1Pipeline(config, logger)
        pipeline.run()
    except Exception as e:
        logger.error(f"Error in Phase 1: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
