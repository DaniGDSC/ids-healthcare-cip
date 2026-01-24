"""Run Phase 4: Clustering."""

import sys
import logging
from pathlib import Path
import yaml
import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.phase4_clustering import DBSCANClustering, ClusterAnalyzer


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
    config_path = project_root / 'config' / 'phase4_config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_latent():
    latent_dir = project_root / 'data' / 'latent'
    train = np.load(latent_dir / 'train_latent_8d.npz')
    val = np.load(latent_dir / 'val_latent_8d.npz')
    test = np.load(latent_dir / 'test_latent_8d.npz')
    return train['X'], val['X'], test['X']


def main():
    config = load_config()
    log_file = project_root / config['logging']['file']
    setup_logging(log_file)
    logger = logging.getLogger(__name__)

    logger.info('=' * 80)
    logger.info('PHASE 4: CLUSTERING')
    logger.info('=' * 80)

    try:
        X_train, X_val, X_test = load_latent()

        db_cfg = config['clustering']['dbscan']
        clusterer = DBSCANClustering(
            eps=db_cfg['eps'],
            min_samples=db_cfg['min_samples'],
            metric=db_cfg['metric'],
            algorithm=db_cfg['algorithm'],
            leaf_size=db_cfg['leaf_size']
        )

        labels_train = clusterer.fit_predict(X_train)
        labels_val = clusterer.fit_predict(X_val)
        labels_test = clusterer.fit_predict(X_test)

        analyzer = ClusterAnalyzer(
            compute_silhouette=config['analysis']['compute_silhouette'],
            compute_db=config['analysis']['compute_davies_bouldin'],
            compute_ch=config['analysis']['compute_calinski_harabasz']
        )

        metrics = analyzer.evaluate(X_val, labels_val)

        output_dir = project_root / config['data']['output_dir']
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / 'train_clusters.npy', labels_train)
        np.save(output_dir / 'val_clusters.npy', labels_val)
        np.save(output_dir / 'test_clusters.npy', labels_test)

        # Save metrics
        import json
        with open(output_dir / 'cluster_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info('\n' + '=' * 80)
        logger.info('PHASE 4 COMPLETE!')
        logger.info('=' * 80)

    except Exception as e:
        logger.error(f"Error in Phase 4: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()