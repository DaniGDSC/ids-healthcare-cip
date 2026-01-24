"""Main script to run Phase 1: Data Preprocessing."""

import sys
import logging
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import RANDOM_SEED
from src.phase1_preprocessing import (
    DataLoader, DataCleaner, DataSplitter, 
    Normalizer, HIPAACompliance
)


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


def main():
    """Run Phase 1 preprocessing pipeline."""
    
    # Load configuration
    config_path = project_root / "config" / "phase1_config.yaml"
    config = load_config(config_path)
    
    # Setup logging
    log_file = project_root / config['logging']['file']
    setup_logging(log_file)
    logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info("PHASE 1: DATA PREPROCESSING")
    logger.info("="*80)
    
    try:
        # Initialize components
        data_dir = project_root / config['data']['input_dir']
        output_dir = project_root / config['data']['output_dir']
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Load data
        logger.info("\n--- Step 1: Loading Data ---")
        loader = DataLoader(data_dir)
        df = loader.load_csv_files()
        
        # Step 2: HIPAA Compliance
        if config['hipaa']['enabled']:
            logger.info("\n--- Step 2: HIPAA Compliance ---")
            hipaa = HIPAACompliance(enabled=True)
            
            # Anonymize IP addresses if present
            ip_columns = ['Src IP', 'Dst IP', 'Source IP', 'Destination IP']
            df = hipaa.anonymize_ip_addresses(df, ip_columns)
            
            hipaa.log_data_access(
                user='phase1_preprocessing',
                action='load',
                data_description='CIC-IDS-2018 dataset',
                record_count=len(df)
            )
        
        # Step 3: Data Cleaning
        logger.info("\n--- Step 3: Data Cleaning ---")
        cleaner = DataCleaner(
            drop_na_threshold=config['preprocessing']['drop_na_threshold'],
            fill_strategy=config['preprocessing']['fill_strategy'],
            replace_inf=config['preprocessing']['replace_inf']
        )
        df = cleaner.clean(df)
        
        # Step 4: Separate features and labels
        logger.info("\n--- Step 4: Separating Features and Labels ---")
        label_col = config['data']['stratify_column']
        
        if label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found")
        
        y = df[label_col].values
        X = df.drop(columns=[label_col]).values
        feature_names = df.drop(columns=[label_col]).columns.tolist()
        
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Labels shape: {y.shape}")
        
        # Step 5: Split data
        logger.info("\n--- Step 5: Splitting Data ---")
        splitter = DataSplitter(
            train_ratio=config['data']['train_ratio'],
            val_ratio=config['data']['val_ratio'],
            test_ratio=config['data']['test_ratio'],
            random_state=RANDOM_SEED,
            stratify=config['data']['stratify']
        )
        
        X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(X, y)
        
        # Step 6: Normalize data
        logger.info("\n--- Step 6: Normalizing Data ---")
        normalizer = Normalizer(method=config['preprocessing']['normalization_method'])
        
        X_train_norm = normalizer.fit_transform(X_train)
        X_val_norm = normalizer.transform(X_val)
        X_test_norm = normalizer.transform(X_test)
        
        # Save scaler
        scaler_path = project_root / "models" / "scalers" / "standard_scaler.pkl"
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        normalizer.save(scaler_path)
        
        # Step 7: Save processed data
        logger.info("\n--- Step 7: Saving Processed Data ---")
        
        if config['output']['save_format'] == 'npz':
            np.savez_compressed(
                output_dir / 'train_normalized.npz',
                X=X_train_norm,
                y=y_train
            )
            np.savez_compressed(
                output_dir / 'val_normalized.npz',
                X=X_val_norm,
                y=y_val
            )
            np.savez_compressed(
                output_dir / 'test_normalized.npz',
                X=X_test_norm,
                y=y_test
            )
        
        # Save metadata
        if config['output']['save_metadata']:
            metadata = {
                'feature_names': feature_names,
                'n_features': len(feature_names),
                'n_train_samples': len(X_train),
                'n_val_samples': len(X_val),
                'n_test_samples': len(X_test),
                'normalization_method': config['preprocessing']['normalization_method'],
                'label_distribution': {
                    'train': {str(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))},
                    'val': {str(k): int(v) for k, v in zip(*np.unique(y_val, return_counts=True))},
                    'test': {str(k): int(v) for k, v in zip(*np.unique(y_test, return_counts=True))}
                }
            }
            
            metadata_path = output_dir / 'phase1_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Metadata saved to {metadata_path}")
        
        logger.info("\n" + "="*80)
        logger.info("PHASE 1 COMPLETE!")
        logger.info("="*80)
        logger.info(f"\nOutput files saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Error in Phase 1: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
