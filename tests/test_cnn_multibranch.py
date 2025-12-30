import pytest
import numpy as np
from maneuvers.models.cnn import _CNN1DMultiBranchClassifier
from maneuvers.data.loader import generate_synthetic_sequence
from maneuvers.preprocessing import windowed_examples_from_sequence


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("tensorflow") is None,
    reason="tensorflow not installed",
)
def test_cnn_multibranch_fit_predict():
    # create windows with combined accel+gyro channels
    Xs = []
    ys = []
    for i in range(6):
        seq = generate_synthetic_sequence(duration_s=3.0, fs=50, seed=i)
        X, y, windows = windowed_examples_from_sequence(
            seq, window_s=0.8, hop_s=0.4, fs=50
        )
        for xx, lbl in zip(X, y):
            if lbl != "none":
                # X windows are (seq_len, 6)
                Xs.append(xx)
                ys.append(0 if "left" in lbl or "right" in lbl else 1)
                if len(Xs) >= 24:
                    break
        if len(Xs) >= 24:
            break

    Xs = np.asarray(Xs)
    ys = np.asarray(ys)
    assert Xs.ndim == 3

    clf = _CNN1DMultiBranchClassifier(
        seq_len=Xs.shape[1], n_accel=3, n_gyro=3, n_classes=2, epochs=3, batch_size=8
    )
    clf.fit(Xs, ys)
    p = clf.predict(Xs[:6])
    assert p.shape[0] == 6
    proba = clf.predict_proba(Xs[:6])
    assert proba.shape == (6, 2)
