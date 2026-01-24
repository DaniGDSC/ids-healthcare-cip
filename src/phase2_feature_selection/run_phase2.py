"""Main script to run Phase 2: Feature Selection."""

import sys
import logging
from pathlib import Path
import yaml
import numpy as np
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import RANDOM_SEED
from src.phase2_feature_selection import (
    InformationGainSelector,
    RandomForestSelector,
    RFESelector
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


def load_data(input_dir: Path):
    """Load preprocessed data from Phase 1."""
    logger = logging.getLogger(__name__)
    
    logger.info("Loading preprocessed data from Phase 1...")
    
    train_data = np.load(input_dir / 'train_normalized.npz')
    val_data = np.load(input_dir / 'val_normalized.npz')
    test_data = np.load(input_dir / 'test_normalized.npz')
    
    # Load metadata
    with open(input_dir / 'phase1_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    return (train_data['X'], train_data['y'],
            val_data['X'], val_data['y'],
            test_data['X'], test_data['y'],
            metadata['feature_names'])


def combine_feature_selections(selectors: dict, method: str = 'union', threshold: int = 2):
    """
    Combine feature selections from multiple methods.
    
    Args:
        selectors: Dictionary of selector objects
        method: Combination method (union/intersection/voting)
        threshold: Minimum number of votes for voting method
    """
    logger = logging.getLogger(__name__)
    
    all_features = set()
    feature_votes = {}
    
    for name, selector in selectors.items():
        selected = selector.get_selected_indices()
        all_features.update(selected)
        
        for idx in selected:
            feature_votes[idx] = feature_votes.get(idx, 0) + 1
    
    if method == 'union':
        final_features = sorted(all_features)
    elif method == 'intersection':
        final_features = sorted([f for f in all_features if feature_votes[f] == len(selectors)])
    elif method == 'voting':
        final_features = sorted([f for f in all_features if feature_votes[f] >= threshold])
    else:
        raise ValueError(f"Unknown combination method: {method}")
    
    logger.info(f"\nFeature selection combination ({method}):")
    logger.info(f"  Total unique features: {len(all_features)}")
    logger.info(f"  Final selected features: {len(final_features)}")
    
    return np.array(final_features)


def main():
    """Run Phase 2 feature selection pipeline."""
    
    # Load configuration
    config_path = project_root / "config" / "phase2_config.yaml"
    config = load_config(config_path)
    
    # Setup logging
    log_file = project_root / config['logging']['file']
    setup_logging(log_file)
    logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info("PHASE 2: FEATURE SELECTION")
    logger.info("="*80)
    
    try:
        # Load data
        input_dir = project_root / config['data']['input_dir']
        output_dir = project_root / config['data']['output_dir']
        output_dir.mkdir(parents=True, exist_ok=True)
        
        X_train, y_train, X_val, y_val, X_test, y_test, feature_names = load_data(input_dir)
        
        logger.info(f"\nData loaded:")
        logger.info(f"  Training: {X_train.shape}")
        logger.info(f"  Validation: {X_val.shape}")
        logger.info(f"  Test: {X_test.shape}")
        
        # Initialize selectors
        selectors = {}
        n_features = config['feature_selection']['n_features']
        
        # Information Gain
        if config['feature_selection']['information_gain']['enabled']:
            logger.info("\n--- Information Gain Selection ---")
            ig_selector = InformationGainSelector(
                n_features=n_features,
                discretization_bins=config['feature_selection']['information_gain']['discretization_bins']
            )
            ig_selector.fit(X_train, y_train, feature_names)
            selectors['information_gain'] = ig_selector
        
        # Random Forest
        if config['feature_selection']['random_forest']['enabled']:
            logger.info("\n--- Random Forest Selection ---")
            rf_selector = RandomForestSelector(
                n_features=n_features,
                n_estimators=config['feature_selection']['random_forest']['n_estimators'],
                max_depth=config['feature_selection']['random_forest']['max_depth'],
                min_samples_split=config['feature_selection']['random_forest']['min_samples_split'],
                random_state=config['feature_selection']['random_forest']['random_state']
            )
            rf_selector.fit(X_train, y_train, feature_names)
            selectors['random_forest'] = rf_selector
        
        # RFE
        if config['feature_selection']['rfe']['enabled']:
            logger.info("\n--- RFE Selection ---")
            rfe_selector = RFESelector(
                n_features=n_features,
                estimator=config['feature_selection']['rfe']['estimator'],
                step=config['feature_selection']['rfe']['step']
            )
            rfe_selector.fit(X_train, y_train, feature_names)
            selectors['rfe'] = rfe_selector
        
        # Combine selections
        logger.info("\n--- Combining Feature Selections ---")
        final_features = combine_feature_selections(
            selectors,
            method=config['feature_selection']['selection_strategy'],
            threshold=config['feature_selection'].get('voting_threshold', 2)
        )
        
        # Transform data
        logger.info("\n--- Transforming Data ---")
        X_train_selected = X_train[:, final_features]
        X_val_selected = X_val[:, final_features]
        X_test_selected = X_test[:, final_features]
        
        logger.info(f"Selected {X_train_selected.shape[1]} features")
        
        # Save transformed data
        logger.info("\n--- Saving Data ---")
        
        np.savez_compressed(
            output_dir / 'train_35features.npz',
            X=X_train_selected,
            y=y_train
        )
        np.savez_compressed(
            output_dir / 'val_35features.npz',
            X=X_val_selected,
            y=y_val
        )
        np.savez_compressed(
            output_dir / 'test_35features.npz',
            X=X_test_selected,
            y=y_test
        )
        
        # Save selected feature information
        selected_feature_names = [feature_names[i] for i in final_features]
        
        feature_info = {
            'selected_features': selected_feature_names,
            'selected_indices': final_features.tolist(),
            'n_features': len(final_features),
            'selection_methods': list(selectors.keys()),
            'combination_strategy': config['feature_selection']['selection_strategy']
        }
        
        with open(output_dir / 'selected_features.json', 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        logger.info(f"\nSelected features saved to {output_dir / 'selected_features.json'}")
        logger.info(f"\nTop 10 selected features:")
        for i, name in enumerate(selected_feature_names[:10], 1):
            logger.info(f"  {i}. {name}")
        
        logger.info("\n" + "="*80)
        logger.info("PHASE 2 COMPLETE!")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Error in Phase 2: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
