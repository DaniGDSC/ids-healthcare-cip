"""Main script to run Phase 2: Feature Selection."""

import sys
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple
import yaml
import numpy as np
import json
import csv

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import RANDOM_STATE
from src.phase2_feature_selection import (
    InformationGainSelector,
    RandomForestSelector,
    RFESelector
)


class Phase2Pipeline:
    """Phase 2 feature selection pipeline with 3 sequential stages."""

    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.input_dir = project_root / config['data']['input_dir']
        self.output_dir = project_root / config['data']['output_dir']
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.feature_names = None
        self.feature_history = {}

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

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self.logger.info("--- Loading Data from Phase 1 ---")
        train_data = np.load(self.input_dir / 'train_normalized.npz')
        val_data = np.load(self.input_dir / 'val_normalized.npz')
        test_data = np.load(self.input_dir / 'test_normalized.npz')

        with open(self.input_dir / 'phase1_metadata.json', 'r') as f:
            metadata = json.load(f)

        self.feature_names = metadata['feature_names']
        self.logger.info(f"Loaded {len(self.feature_names)} features")
        self.logger.info(f"  Train: {train_data['X'].shape}")
        self.logger.info(f"  Val:   {val_data['X'].shape}")
        self.logger.info(f"  Test:  {test_data['X'].shape}")

        return (train_data['X'], train_data['y'],
                val_data['X'], val_data['y'],
                test_data['X'], test_data['y'])

    def _stage1_information_gain(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Stage 1: Information Gain Filter (52 → 45 features)."""
        self.logger.info("\n" + "="*80)
        self.logger.info("STAGE 1: INFORMATION GAIN FILTER")
        self.logger.info("="*80)
        self.logger.info("Target: Top 45 features (~5 minutes)")

        start_time = time.time()
        n_input = X_train.shape[1]

        ig_config = self.config['feature_selection']['stage1_information_gain']
        ig_selector = InformationGainSelector(
            n_features=ig_config['n_features'],
            discretization_bins=ig_config.get('discretization_bins', 10)
        )
        ig_selector.fit(X_train, y_train, self.feature_names)
        selected_idx = ig_selector.get_selected_indices()

        elapsed = time.time() - start_time
        self.logger.info(f"✓ Stage 1 complete in {elapsed:.1f}s")
        self.logger.info(f"  {n_input} → {len(selected_idx)} features")

        self.feature_history['stage1_information_gain'] = {
            'selected_indices': selected_idx.tolist(),
            'n_features': len(selected_idx),
            'scores': ig_selector.get_feature_scores().tolist()
        }
        return selected_idx, X_train[:, selected_idx], (X_train.shape[1], len(selected_idx))

    def _stage2_random_forest(self, X_train: np.ndarray, y_train: np.ndarray, prev_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        """Stage 2: Random Forest Importance (45 → 40 features).
        
        Supports optional sampling for faster training on large datasets.
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("STAGE 2: RANDOM FOREST IMPORTANCE")
        self.logger.info("="*80)
        self.logger.info("Target: Top 40 features (~15 minutes with full data, ~3 min with sampling)")

        start_time = time.time()
        n_input = X_train.shape[1]

        rf_config = self.config['feature_selection']['stage2_random_forest']
        use_sampling = rf_config.get('use_sampling', False)
        sample_size = rf_config.get('sample_size', 0.15)
        
        rf_selector = RandomForestSelector(
            n_features=rf_config['n_features'],
            n_estimators=rf_config.get('n_estimators', 100),
            max_depth=rf_config.get('max_depth', 20),
            min_samples_split=rf_config.get('min_samples_split', 10),
            random_state=RANDOM_STATE,
            use_sampling=use_sampling,
            sample_size=sample_size
        )
        rf_selector.fit(X_train, y_train, [self.feature_names[i] for i in prev_indices])
        selected_stage2 = rf_selector.get_selected_indices()

        # Map back to original feature indices
        selected_idx = prev_indices[selected_stage2]

        elapsed = time.time() - start_time
        self.logger.info(f"✓ Stage 2 complete in {elapsed:.1f}s")
        self.logger.info(f"  {n_input} → {len(selected_idx)} features")

        self.feature_history['stage2_random_forest'] = {
            'selected_indices': selected_idx.tolist(),
            'n_features': len(selected_idx),
            'importance_scores': rf_selector.get_feature_importances().tolist(),
            'sampling_enabled': use_sampling,
            'sample_size': sample_size if use_sampling else None
        }
        return selected_idx, X_train[:, selected_stage2], (n_input, len(selected_idx))

    def _stage3_rfe(self, X_train: np.ndarray, y_train: np.ndarray, prev_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        """Stage 3: Recursive Feature Elimination (40 → 35 features).
        
        Supports optional greedy batch elimination for faster execution.
        Greedy mode removes ~5% of features per iteration instead of 1 feature.
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("STAGE 3: RECURSIVE FEATURE ELIMINATION")
        self.logger.info("="*80)
        self.logger.info("Target: 35 features (optimal) (~25 minutes standard, ~3-5 min greedy)")

        start_time = time.time()
        n_input = X_train.shape[1]

        rfe_config = self.config['feature_selection']['stage3_rfe']
        use_greedy = rfe_config.get('use_greedy', False)
        greedy_batch_pct = rfe_config.get('greedy_batch_pct', 0.05)
        
        rfe_selector = RFESelector(
            n_features=rfe_config['n_features'],
            estimator=rfe_config.get('estimator', 'linear_svc'),
            step=rfe_config.get('step', 1),
            cv=rfe_config.get('cv_folds', 5),
            greedy=use_greedy,
            greedy_batch_pct=greedy_batch_pct
        )
        rfe_selector.fit(X_train, y_train, [self.feature_names[i] for i in prev_indices])
        selected_stage3 = rfe_selector.get_selected_indices()

        # Map back to original feature indices
        selected_idx = prev_indices[selected_stage3]

        elapsed = time.time() - start_time
        self.logger.info(f"✓ Stage 3 complete in {elapsed:.1f}s")
        self.logger.info(f"  {n_input} → {len(selected_idx)} features")

        self.feature_history['stage3_rfe'] = {
            'selected_indices': selected_idx.tolist(),
            'n_features': len(selected_idx),
            'greedy_enabled': use_greedy,
            'greedy_batch_pct': greedy_batch_pct if use_greedy else None
        }
        return selected_idx, X_train[:, selected_stage3], (n_input, len(selected_idx))

    def _save_outputs(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray,
                      final_indices: np.ndarray):
        """Save feature-selected data and metadata."""
        self.logger.info("\n" + "="*80)
        self.logger.info("SAVING OUTPUTS")
        self.logger.info("="*80)

        # Select data
        X_train_selected = X_train[:, final_indices]
        X_val_selected = X_val[:, final_indices]
        X_test_selected = X_test[:, final_indices]

        # Save as compressed NPZ
        np.savez_compressed(self.output_dir / 'train_35features.npz', X=X_train_selected, y=y_train)
        np.savez_compressed(self.output_dir / 'val_35features.npz', X=X_val_selected, y=y_val)
        np.savez_compressed(self.output_dir / 'test_35features.npz', X=X_test_selected, y=y_test)
        self.logger.info(f"✓ Saved feature-selected data")

        # Save selected feature info
        selected_feature_names = [self.feature_names[i] for i in final_indices]
        feature_info = {
            'selected_features': selected_feature_names,
            'selected_indices': final_indices.tolist(),
            'n_features': len(final_indices),
            'original_n_features': len(self.feature_names)
        }
        with open(self.output_dir / 'selected_features.json', 'w') as f:
            json.dump(feature_info, f, indent=2)
        self.logger.info(f"✓ Saved selected features list")

        # Save feature importance CSV
        with open(self.output_dir / 'feature_importance.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Rank', 'Feature', 'Index'])
            for rank, (name, idx) in enumerate(zip(selected_feature_names, final_indices), 1):
                writer.writerow([rank, name, idx])
        self.logger.info(f"✓ Saved feature importance CSV")

        # Save phase2 metadata
        metadata = {
            'n_features_selected': len(final_indices),
            'selected_features': selected_feature_names,
            'feature_selection_history': self.feature_history,
            'output_files': {
                'train': 'train_35features.npz',
                'val': 'val_35features.npz',
                'test': 'test_35features.npz',
                'selected_features_list': 'selected_features.json',
                'importance_scores': 'feature_importance.csv'
            }
        }
        with open(self.output_dir / 'phase2_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        self.logger.info(f"✓ Saved phase2 metadata")

        self.logger.info(f"\nOutput summary:")
        self.logger.info(f"  Train: {X_train_selected.shape}")
        self.logger.info(f"  Val:   {X_val_selected.shape}")
        self.logger.info(f"  Test:  {X_test_selected.shape}")
        self.logger.info(f"\nTop 10 selected features:")
        for i, name in enumerate(selected_feature_names[:10], 1):
            self.logger.info(f"  {i}. {name}")

    def run(self):
        """Execute all 3 feature selection stages."""
        X_train, y_train, X_val, y_val, X_test, y_test = self._load_data()

        # Stage 1: Information Gain
        idx_stage1, X_train_s1, (n_in_s1, n_out_s1) = self._stage1_information_gain(X_train, y_train)

        # Stage 2: Random Forest
        idx_stage2, X_train_s2, (n_in_s2, n_out_s2) = self._stage2_random_forest(X_train_s1, y_train, idx_stage1)

        # Stage 3: RFE
        idx_final, X_train_s3, (n_in_s3, n_out_s3) = self._stage3_rfe(X_train_s2, y_train, idx_stage2)

        # Select corresponding data for val/test
        X_val_selected = X_val[:, idx_final]
        X_test_selected = X_test[:, idx_final]

        # Save outputs
        self._save_outputs(X_train, y_train, X_val, y_val, X_test, y_test, idx_final)

        self.logger.info("\n" + "="*80)
        self.logger.info("PHASE 2 COMPLETE!")
        self.logger.info("="*80)
        self.logger.info(f"Feature Selection Summary:")
        self.logger.info(f"  Stage 1 (IG):  {n_in_s1} → {n_out_s1} features")
        self.logger.info(f"  Stage 2 (RF):  {n_in_s2} → {n_out_s2} features")
        self.logger.info(f"  Stage 3 (RFE): {n_in_s3} → {n_out_s3} features")
        self.logger.info(f"\nOutput files saved to: {self.output_dir}")




def main():
    """Run Phase 2 feature selection pipeline."""
    config_path = project_root / "config" / "phase2_config.yaml"
    config = Phase2Pipeline.load_config(config_path)

    log_file = project_root / config['logging']['file']
    Phase2Pipeline.setup_logging(log_file)
    logger = logging.getLogger(__name__)

    logger.info("="*80)
    logger.info("PHASE 2: FEATURE SELECTION")
    logger.info("="*80)

    try:
        pipeline = Phase2Pipeline(config, logger)
        pipeline.run()
    except Exception as e:
        logger.error(f"Error in Phase 2: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

