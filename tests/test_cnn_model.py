import pytest
import numpy as np
from maneuvers.models.cnn import _CNN1DClassifier
from maneuvers.data.loader import generate_synthetic_sequence
from maneuvers.preprocessing import windowed_examples_from_sequence


@pytest.mark.skipif(__import__('importlib').util.find_spec('tensorflow') is None, reason='tensorflow not installed')
def test_cnn_fit_predict():
    # create a small set of sequences with labeled maneuvers
    Xs = []
    ys = []
    for i in range(6):
        seq = generate_synthetic_sequence(duration_s=3.0, fs=50, seed=i)
        X, y, windows = windowed_examples_from_sequence(seq, window_s=0.8, hop_s=0.4, fs=50)
        # pick windows whose label is not 'none'
        for xx, lbl in zip(X, y):
            if lbl != 'none':
                Xs.append(xx)
                ys.append(0 if 'left' in lbl or 'right' in lbl else 1)
                if len(Xs) >= 20:
                    break
        if len(Xs) >= 20:
            break

    Xs = np.asarray(Xs)
    ys = np.asarray(ys)
    assert Xs.ndim == 3

    clf = _CNN1DClassifier(seq_len=Xs.shape[1], n_channels=Xs.shape[2], n_classes=2, epochs=3, batch_size=8)
    clf.fit(Xs, ys)
    p = clf.predict(Xs[:4])
    assert p.shape[0] == 4
    proba = clf.predict_proba(Xs[:4])
    assert proba.shape == (4, 2)
