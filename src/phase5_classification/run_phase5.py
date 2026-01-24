"""Run Phase 5: Classification."""

import sys
import logging
from pathlib import Path
import yaml
import numpy as np
import json

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.phase5_classification import EnsembleClassifier, ClassificationEvaluator


def setup_logging(log_file: str):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def load_config():
    with open(project_root / 'config' / 'phase5_config.yaml', 'r') as f:
        return yaml.safe_load(f)


def load_latent_and_labels():
    latent_dir = project_root / 'data' / 'latent'
    processed_dir = project_root / 'data' / 'processed'

    X_train = np.load(latent_dir / 'train_latent_8d.npz')['X']
    X_val = np.load(latent_dir / 'val_latent_8d.npz')['X']
    X_test = np.load(latent_dir / 'test_latent_8d.npz')['X']

    y_train = np.load(processed_dir / 'train_normalized.npz')['y']
    y_val = np.load(processed_dir / 'val_normalized.npz')['y']
    y_test = np.load(processed_dir / 'test_normalized.npz')['y']

    return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    config = load_config()
    log_file = project_root / config['logging']['file']
    setup_logging(log_file)
    logger = logging.getLogger(__name__)

    logger.info('=' * 80)
    logger.info('PHASE 5: CLASSIFICATION')
    logger.info('=' * 80)

    try:
        X_train, X_val, X_test, y_train, y_val, y_test = load_latent_and_labels()

        clf = EnsembleClassifier(config)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_proba = None
        try:
            y_proba = clf.predict_proba(X_test)
        except Exception:
            logger.warning("Probability predictions not available")

        evaluator = ClassificationEvaluator()
        metrics = evaluator.evaluate(y_test, y_pred, y_proba)

        # Save models
        model_dir = project_root / config['data']['model_dir']
        clf.save(model_dir)

        # Save predictions
        output_dir = project_root / config['data']['output_dir']
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / 'test_predictions.npy', y_pred)
        if y_proba is not None:
            np.save(output_dir / 'test_probabilities.npy', y_proba)

        # Save metrics
        with open(output_dir / 'classification_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info('\n' + '=' * 80)
        logger.info('PHASE 5 COMPLETE!')
        logger.info('=' * 80)

    except Exception as e:
        logger.error(f"Error in Phase 5: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()