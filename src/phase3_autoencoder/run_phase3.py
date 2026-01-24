"""Run Phase 3: Autoencoder training and evaluation."""

import sys
import logging
from pathlib import Path
import yaml
import numpy as np
import json

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import RANDOM_SEED
from src.phase3_autoencoder import (
    build_autoencoder,
    AutoencoderTrainer,
    AutoencoderEvaluator,
    ThresholdOptimizer
)


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
    config_path = project_root / 'config' / 'phase3_config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_features():
    input_dir = project_root / 'data' / 'features'
    train = np.load(input_dir / 'train_35features.npz')
    val = np.load(input_dir / 'val_35features.npz')
    test = np.load(input_dir / 'test_35features.npz')
    return train['X'], val['X'], test['X']


def save_models(autoencoder, encoder, decoder, model_dir: Path):
    model_dir.mkdir(parents=True, exist_ok=True)
    autoencoder.save(model_dir / 'autoencoder.keras')
    encoder.save(model_dir / 'encoder.keras')
    decoder.save(model_dir / 'decoder.keras')


def main():
    config = load_config()
    log_file = project_root / config['logging']['file']
    setup_logging(log_file)
    logger = logging.getLogger(__name__)

    logger.info('=' * 80)
    logger.info('PHASE 3: AUTOENCODER')
    logger.info('=' * 80)

    try:
        X_train, X_val, X_test = load_features()

        arch = config['architecture']
        model, encoder, decoder = build_autoencoder(
            input_dim=arch['input_dim'],
            encoder_layers=arch['encoder_layers'],
            decoder_layers=arch['decoder_layers'],
            latent_dim=arch['latent_dim'],
            activation=arch['activation'],
            output_activation=arch['output_activation'],
            dropout_rate=arch['dropout_rate'],
            use_batch_norm=arch['use_batch_normalization'],
            l2_reg=arch['l2_regularization']
        )

        trainer_cfg = config['training']
        trainer = AutoencoderTrainer(
            optimizer=trainer_cfg['optimizer'],
            learning_rate=trainer_cfg['learning_rate'],
            batch_size=trainer_cfg['batch_size'],
            epochs=trainer_cfg['epochs'],
            early_stopping=trainer_cfg['early_stopping'],
            reduce_lr=trainer_cfg['reduce_lr'],
            validation_split=trainer_cfg['validation_split'],
            shuffle=trainer_cfg['shuffle']
        )

        trainer.compile_model(model)

        tensorboard_dir = None
        if config['logging'].get('tensorboard', False):
            tensorboard_dir = project_root / config['logging']['tensorboard_dir']

        history = trainer.train(model, X_train, X_val, tensorboard_dir=tensorboard_dir)

        evaluator = AutoencoderEvaluator()
        metrics = evaluator.evaluate(model, X_val, metrics={m: True for m in config['evaluation']['metrics']})

        # Threshold optimization on validation set errors
        errors = evaluator.reconstruction_errors(model, X_val)
        threshold_cfg = config['anomaly_detection']
        threshold_opt = ThresholdOptimizer(
            method=threshold_cfg['threshold_method'],
            percentile=threshold_cfg.get('percentile', 95),
            n_std=threshold_cfg.get('n_std', 3)
        )
        threshold = threshold_opt.compute_threshold(errors)

        # Save models and outputs
        model_dir = project_root / config['data']['model_dir']
        save_models(model, encoder, decoder, model_dir)

        latent_dir = project_root / config['data']['output_dir']
        latent_dir.mkdir(parents=True, exist_ok=True)
        latent_train = encoder.predict(X_train, verbose=0)
        latent_val = encoder.predict(X_val, verbose=0)
        latent_test = encoder.predict(X_test, verbose=0)

        np.savez_compressed(latent_dir / 'train_latent_8d.npz', X=latent_train)
        np.savez_compressed(latent_dir / 'val_latent_8d.npz', X=latent_val)
        np.savez_compressed(latent_dir / 'test_latent_8d.npz', X=latent_test)

        # Save threshold config
        threshold_path = model_dir / 'threshold_config.json'
        with open(threshold_path, 'w') as f:
            json.dump({'method': threshold_cfg['threshold_method'], 'value': threshold}, f, indent=2)

        logger.info('\n' + '=' * 80)
        logger.info('PHASE 3 COMPLETE!')
        logger.info('=' * 80)

    except Exception as e:
        logger.error(f"Error in Phase 3: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()