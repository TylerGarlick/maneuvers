"""Lightweight 1D-CNN model wrappers for time-series classification.

Provides _CNN1DClassifier: sklearn-like wrapper around a small Keras 1D-CNN.
This is optional and will raise an informative ImportError if TensorFlow is
not installed.
"""
from __future__ import annotations
import numpy as np

try:
    import tensorflow as tf
    from tensorflow.keras import models, layers, optimizers
    HAS_TF = True
except Exception:
    tf = None
    models = None
    layers = None
    optimizers = None
    HAS_TF = False


class _CNN1DClassifier:
    def __init__(self, seq_len: int, n_channels: int = 3, n_classes: int = None, epochs: int = 20, batch_size: int = 16):
        if not HAS_TF:
            raise ImportError("TensorFlow is required for _CNN1DClassifier. Install tensorflow to use this model.")
        self.seq_len = int(seq_len)
        self.n_channels = int(n_channels)
        self.n_classes = n_classes
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.model = None

    def _build(self, n_classes: int):
        m = models.Sequential()
        m.add(layers.Input(shape=(self.seq_len, self.n_channels)))
        m.add(layers.Conv1D(32, kernel_size=5, activation="relu", padding="same"))
        m.add(layers.MaxPool1D(pool_size=2))
        m.add(layers.Conv1D(64, kernel_size=3, activation="relu", padding="same"))
        m.add(layers.GlobalAveragePooling1D())
        m.add(layers.Dense(64, activation="relu"))
        m.add(layers.Dense(n_classes, activation="softmax"))
        m.compile(optimizer=optimizers.Adam(learning_rate=1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return m

    def fit(self, X, y, validation_data=None):
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim != 3:
            raise ValueError("X must be 3D: (n_samples, seq_len, n_channels)")
        n_classes = int(np.unique(y).shape[0]) if self.n_classes is None else int(self.n_classes)
        self.model = self._build(n_classes)
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0, validation_data=validation_data)
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        if self.model is None:
            raise RuntimeError("Model has not been trained")
        return self.model.predict(X)

    def predict(self, X):
        proba = self.predict_proba(X)
        return proba.argmax(axis=1)


class _CNN1DMultiBranchClassifier:
    """1D-CNN with separate convolutional branches for accel and gyro.

    Input: X shape (n_samples, seq_len, n_channels) where n_channels==6 (ax,ay,az,gx,gy,gz)
    Or provide accel and gyro separately as (n_samples, seq_len, n_accel) and (n_samples, seq_len, n_gyro).
    """

    def __init__(self, seq_len: int, n_accel: int = 3, n_gyro: int = 3, n_classes: int = None, epochs: int = 20, batch_size: int = 16):
        if not HAS_TF:
            raise ImportError("TensorFlow is required for _CNN1DMultiBranchClassifier. Install tensorflow to use this model.")
        self.seq_len = int(seq_len)
        self.n_accel = int(n_accel)
        self.n_gyro = int(n_gyro)
        self.n_classes = n_classes
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.model = None

    def _build(self, n_classes: int):
        # single input that will be split into two branches
        inp = layers.Input(shape=(self.seq_len, self.n_accel + self.n_gyro))
        # split accel and gyro channels
        accel_in = layers.Lambda(lambda x: x[:, :, : self.n_accel])(inp)
        gyro_in = layers.Lambda(lambda x: x[:, :, self.n_accel :])(inp)

        # accel branch
        a = layers.Conv1D(32, kernel_size=5, activation="relu", padding="same")(accel_in)
        a = layers.MaxPool1D(pool_size=2)(a)
        a = layers.Conv1D(64, kernel_size=3, activation="relu", padding="same")(a)
        a = layers.GlobalAveragePooling1D()(a)

        # gyro branch
        g = layers.Conv1D(32, kernel_size=5, activation="relu", padding="same")(gyro_in)
        g = layers.MaxPool1D(pool_size=2)(g)
        g = layers.Conv1D(64, kernel_size=3, activation="relu", padding="same")(g)
        g = layers.GlobalAveragePooling1D()(g)

        # combine
        x = layers.Concatenate()([a, g])
        x = layers.Dense(64, activation="relu")(x)
        out = layers.Dense(n_classes, activation="softmax")(x)

        model = models.Model(inputs=inp, outputs=out)
        model.compile(optimizer=optimizers.Adam(learning_rate=1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model

    def fit(self, X, y, validation_data=None):
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim != 3:
            raise ValueError("X must be 3D: (n_samples, seq_len, n_channels)")
        if X.shape[2] != (self.n_accel + self.n_gyro):
            raise ValueError(f"expected n_channels={self.n_accel + self.n_gyro}, got {X.shape[2]}")
        n_classes = int(np.unique(y).shape[0]) if self.n_classes is None else int(self.n_classes)
        self.model = self._build(n_classes)
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0, validation_data=validation_data)
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        if self.model is None:
            raise RuntimeError("Model has not been trained")
        return self.model.predict(X)

    def predict(self, X):
        proba = self.predict_proba(X)
        return proba.argmax(axis=1)
