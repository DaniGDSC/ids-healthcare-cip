"""Autoencoder training utilities with optimizations."""

import logging
from typing import Tuple
import numpy as np
import tensorflow as tf
from tensorflow import keras

logger = logging.getLogger(__name__)


class AutoencoderTrainer:
    """Train autoencoder models with performance optimizations."""

    def __init__(self, optimizer: str = 'adam', learning_rate: float = 1e-3,
                 batch_size: int = 256, epochs: int = 100,
                 early_stopping: dict = None, reduce_lr: dict = None,
                 validation_split: float = 0.0, shuffle: bool = True,
                 mixed_precision: bool = False, use_xla: bool = False):
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping = early_stopping or {'enabled': True, 'patience': 15}
        self.reduce_lr = reduce_lr or {'enabled': True, 'patience': 5}
        self.validation_split = validation_split
        self.shuffle = shuffle
        self.mixed_precision = mixed_precision
        self.use_xla = use_xla
        
        # Enable mixed precision if requested
        if mixed_precision:
            try:
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                logger.info("Mixed precision (float16) enabled")
            except Exception as e:
                logger.warning(f"Could not enable mixed precision: {e}")
        
        # Enable XLA compilation if requested
        if use_xla:
            tf.config.optimizer.set_jit_compilation(True)
            logger.info("XLA compilation enabled")

    def _build_callbacks(self, tensorboard_dir=None):
        callbacks = []

        if self.early_stopping.get('enabled', True):
            callbacks.append(keras.callbacks.EarlyStopping(
                monitor=self.early_stopping.get('monitor', 'val_loss'),
                patience=self.early_stopping.get('patience', 15),
                restore_best_weights=self.early_stopping.get('restore_best_weights', True)
            ))

        if self.reduce_lr.get('enabled', True):
            callbacks.append(keras.callbacks.ReduceLROnPlateau(
                monitor=self.reduce_lr.get('monitor', 'val_loss'),
                factor=self.reduce_lr.get('factor', 0.5),
                patience=self.reduce_lr.get('patience', 5),
                min_lr=self.reduce_lr.get('min_lr', 1e-6)
            ))

        if tensorboard_dir:
            callbacks.append(keras.callbacks.TensorBoard(log_dir=tensorboard_dir))

        return callbacks

    def compile_model(self, model: keras.Model):
        if self.optimizer == 'adam':
            opt = keras.optimizers.Adam(learning_rate=self.learning_rate)
        elif self.optimizer == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")

        model.compile(optimizer=opt, loss='mse', metrics=['mae'])
        return model

    def train(self, model: keras.Model, X_train: np.ndarray, X_val: np.ndarray = None,
              tensorboard_dir=None):
        """
        Train autoencoder with optional tf.data pipelining.
        
        Args:
            model: Keras model to train.
            X_train: Training feature matrix.
            X_val: Validation feature matrix.
            tensorboard_dir: Optional TensorBoard log directory.
            
        Returns:
            Training history.
        """
        # Optional: use tf.data pipeline for better performance
        train_dataset = None
        val_dataset = None
        validation_data = None
        
        try:
            # Create tf.data pipeline with prefetch and batching
            train_dataset = tf.data.Dataset.from_tensor_slices((X_train, X_train))
            train_dataset = train_dataset.shuffle(buffer_size=min(10000, len(X_train)))
            train_dataset = train_dataset.batch(self.batch_size)
            train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
            
            if X_val is not None:
                val_dataset = tf.data.Dataset.from_tensor_slices((X_val, X_val))
                val_dataset = val_dataset.batch(self.batch_size)
                val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
                validation_data = val_dataset
            
            logger.info("Using tf.data pipeline with AUTOTUNE prefetch")
        except Exception as e:
            logger.warning(f"Could not create tf.data pipeline: {e}; falling back to numpy")
            train_dataset = None
        
        callbacks = self._build_callbacks(tensorboard_dir)
        
        # Use tf.data if available, else fall back to numpy arrays
        if train_dataset is not None:
            history = model.fit(
                train_dataset,
                validation_data=validation_data,
                epochs=self.epochs,
                callbacks=callbacks,
                verbose=2
            )
        else:
            history = model.fit(
                X_train,
                X_train,
                validation_data=(X_val, X_val) if X_val is not None else None,
                epochs=self.epochs,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                validation_split=self.validation_split if X_val is None else 0.0,
                callbacks=callbacks,
                verbose=2
            )
        
        return history