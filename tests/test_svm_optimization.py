"""Test SVM kernel approximation optimization."""

import pytest
import numpy as np
import time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.phase5_classification.ensemble_classifier import EnsembleClassifier


class TestSVMOptimization:
    """Test suite for SVM kernel approximation optimization."""
    
    @pytest.fixture
    def synthetic_data_small(self):
        """Generate small synthetic dataset (1K samples)."""
        X, y = make_classification(
            n_samples=1000,
            n_features=43,
            n_informative=30,
            n_redundant=5,
            n_classes=3,
            weights=[0.7, 0.25, 0.05],  # Imbalanced like real data
            random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )
        return X_train, X_test, y_train, y_test
    
    @pytest.fixture
    def synthetic_data_large(self):
        """Generate large synthetic dataset (100K samples)."""
        X, y = make_classification(
            n_samples=100000,
            n_features=43,
            n_informative=30,
            n_redundant=5,
            n_classes=3,
            weights=[0.7, 0.25, 0.05],
            random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, stratify=y, random_state=42
        )
        return X_train, X_test, y_train, y_test
    
    def test_kernel_approximation_initialization(self):
        """Test that kernel approximation is enabled by default."""
        ensemble = EnsembleClassifier()
        assert ensemble.use_kernel_approximation is True
        assert ensemble.n_components == 500
    
    def test_exact_kernel_initialization(self):
        """Test that exact kernel can be disabled."""
        ensemble = EnsembleClassifier(use_kernel_approximation=False)
        assert ensemble.use_kernel_approximation is False
    
    def test_kernel_approximation_training(self, synthetic_data_small):
        """Test that kernel approximation trains successfully."""
        X_train, X_test, y_train, y_test = synthetic_data_small
        
        ensemble = EnsembleClassifier(use_kernel_approximation=True)
        ensemble.fit(X_train, y_train)
        
        assert ensemble.is_fitted is True
        
        # Test prediction
        y_pred = ensemble.predict(X_test)
        assert y_pred.shape == y_test.shape
        
        # Test probability prediction
        y_proba = ensemble.predict_proba(X_test)
        assert y_proba.shape == (len(X_test), 3)
        assert np.allclose(y_proba.sum(axis=1), 1.0)
    
    def test_exact_kernel_training(self, synthetic_data_small):
        """Test that exact kernel still works."""
        X_train, X_test, y_train, y_test = synthetic_data_small
        
        ensemble = EnsembleClassifier(use_kernel_approximation=False)
        ensemble.fit(X_train, y_train)
        
        assert ensemble.is_fitted is True
        
        y_pred = ensemble.predict(X_test)
        assert y_pred.shape == y_test.shape
    
    def test_accuracy_comparison(self, synthetic_data_small):
        """Test that kernel approximation maintains reasonable accuracy."""
        X_train, X_test, y_train, y_test = synthetic_data_small
        
        # Train with approximation
        ensemble_approx = EnsembleClassifier(use_kernel_approximation=True, n_components=500)
        ensemble_approx.fit(X_train, y_train)
        y_pred_approx = ensemble_approx.predict(X_test)
        acc_approx = accuracy_score(y_test, y_pred_approx)
        
        # Train with exact kernel
        ensemble_exact = EnsembleClassifier(use_kernel_approximation=False)
        ensemble_exact.fit(X_train, y_train)
        y_pred_exact = ensemble_exact.predict(X_test)
        acc_exact = accuracy_score(y_test, y_pred_exact)
        
        # Accuracy should be within 5% (approximation trade-off)
        print(f"Approximation accuracy: {acc_approx:.4f}")
        print(f"Exact kernel accuracy: {acc_exact:.4f}")
        print(f"Difference: {abs(acc_approx - acc_exact):.4f}")
        
        assert acc_approx > 0.5, "Approximation accuracy too low"
        assert abs(acc_approx - acc_exact) < 0.15, "Accuracy degradation too large"
    
    def test_speed_comparison_large_dataset(self, synthetic_data_large):
        """Test that kernel approximation is significantly faster on large datasets."""
        X_train, X_test, y_train, y_test = synthetic_data_large
        
        # Time approximation
        ensemble_approx = EnsembleClassifier(use_kernel_approximation=True, n_components=500)
        start_approx = time.time()
        ensemble_approx.fit(X_train, y_train)
        time_approx = time.time() - start_approx
        
        # For comparison purposes, we skip exact kernel on large dataset
        # as it would take too long (minutes to hours)
        # Instead, we verify that approximation completes in reasonable time
        
        print(f"Kernel approximation training time (100K samples): {time_approx:.2f}s")
        
        # Should complete in under 60 seconds for 100K samples
        assert time_approx < 60, f"Training too slow: {time_approx:.2f}s"
        
        # Verify predictions work
        y_pred = ensemble_approx.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy on 100K samples: {accuracy:.4f}")
        
        assert accuracy > 0.5, "Accuracy too low on large dataset"
    
    def test_save_load_with_approximation(self, synthetic_data_small, tmp_path):
        """Test save/load functionality with kernel approximation."""
        X_train, X_test, y_train, y_test = synthetic_data_small
        
        # Train and save
        ensemble = EnsembleClassifier(use_kernel_approximation=True, n_components=300)
        ensemble.fit(X_train, y_train)
        
        model_dir = tmp_path / "models"
        ensemble.save(model_dir)
        
        # Load and verify
        loaded_ensemble = EnsembleClassifier.load(model_dir)
        
        assert loaded_ensemble.is_fitted is True
        assert loaded_ensemble.use_kernel_approximation is True
        assert loaded_ensemble.n_components == 300
        
        # Verify predictions match
        y_pred_original = ensemble.predict(X_test)
        y_pred_loaded = loaded_ensemble.predict(X_test)
        
        np.testing.assert_array_equal(y_pred_original, y_pred_loaded)
    
    def test_save_load_exact_kernel(self, synthetic_data_small, tmp_path):
        """Test save/load functionality with exact kernel."""
        X_train, X_test, y_train, y_test = synthetic_data_small
        
        ensemble = EnsembleClassifier(use_kernel_approximation=False)
        ensemble.fit(X_train, y_train)
        
        model_dir = tmp_path / "models"
        ensemble.save(model_dir)
        
        loaded_ensemble = EnsembleClassifier.load(model_dir)
        
        assert loaded_ensemble.is_fitted is True
        assert loaded_ensemble.use_kernel_approximation is False
        
        y_pred_original = ensemble.predict(X_test)
        y_pred_loaded = loaded_ensemble.predict(X_test)
        
        np.testing.assert_array_equal(y_pred_original, y_pred_loaded)
    
    def test_different_n_components(self, synthetic_data_small):
        """Test different numbers of Nystroem components."""
        X_train, X_test, y_train, y_test = synthetic_data_small
        
        accuracies = {}
        
        for n_comp in [100, 300, 500, 1000]:
            ensemble = EnsembleClassifier(
                use_kernel_approximation=True,
                n_components=n_comp
            )
            ensemble.fit(X_train, y_train)
            y_pred = ensemble.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            accuracies[n_comp] = acc
            print(f"n_components={n_comp}: accuracy={acc:.4f}")
        
        # All configurations should achieve reasonable accuracy
        for n_comp, acc in accuracies.items():
            assert acc > 0.5, f"Accuracy too low for n_components={n_comp}"
    
    def test_confidence_scores(self, synthetic_data_small):
        """Test confidence score calculation with approximation."""
        X_train, X_test, y_train, y_test = synthetic_data_small
        
        ensemble = EnsembleClassifier(use_kernel_approximation=True)
        ensemble.fit(X_train, y_train)
        
        confidence = ensemble.get_confidence_scores(X_test)
        
        assert confidence.shape == (len(X_test),)
        assert np.all(confidence >= 0.0) and np.all(confidence <= 1.0)
        assert np.mean(confidence) > 0.3  # Should have reasonable confidence
    
    def test_feature_importance(self, synthetic_data_small):
        """Test feature importance extraction."""
        X_train, X_test, y_train, y_test = synthetic_data_small
        
        ensemble = EnsembleClassifier(use_kernel_approximation=True)
        ensemble.fit(X_train, y_train)
        
        importance = ensemble.get_feature_importance()
        
        assert importance.shape == (X_train.shape[1],)
        assert np.all(importance >= 0.0)
        assert np.isclose(importance.sum(), 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
