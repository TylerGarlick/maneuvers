import pytest
import numpy as np
from maneuvers.data.loader import generate_synthetic_sequence
from maneuvers.classify import balance_classes, augment_windows


def test_balance_oversample():
    X = np.vstack([np.ones((5, 3)), np.zeros((10, 3))])
    y = np.array([0] * 5 + [1] * 10)
    Xb, yb = balance_classes(X, y, strategy="oversample")
    unique, counts = np.unique(yb, return_counts=True)
    assert counts.min() == counts.max()


def test_balance_undersample():
    X = np.vstack([np.ones((5, 3)), np.zeros((10, 3))])
    y = np.array([0] * 5 + [1] * 10)
    Xb, yb = balance_classes(X, y, strategy="undersample")
    unique, counts = np.unique(yb, return_counts=True)
    assert counts.min() == counts.max()


def test_augment_windows():
    X = np.ones((5, 10, 3))
    y = np.array([0, 0, 1, 1, 2])
    Xa, ya = augment_windows(X, y, n_aug=2, noise_std=0.01)
    assert Xa.shape[0] == 5 * (1 + 2)
    assert ya.shape[0] == 5 * (1 + 2)


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("tensorflow") is None,
    reason="tensorflow not installed",
)
def test_train_cnn_from_sequence():
    from maneuvers.classify import train_cnn_from_sequence

    seq = generate_synthetic_sequence(duration_s=4.0, fs=50, seed=0)
    out = train_cnn_from_sequence(
        seq,
        model_type="cnn_multi",
        window_s=0.8,
        hop_s=0.4,
        fs=50,
        balance="oversample",
        augment=1,
        epochs=2,
        batch_size=8,
    )
    assert "model" in out
    assert "encoder" in out
    assert out["model"] is not None
