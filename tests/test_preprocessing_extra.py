import pytest
import numpy as np
from maneuvers.preprocessing import moving_average, windowed_examples_from_sequence
from maneuvers.data.loader import Sequence


def test_moving_average_short():
    x = np.array([1.0, 2.0])
    # window larger than len(x): return flat array of mean
    out = moving_average(x, window=5)
    assert out.shape == x.shape
    assert (out == np.full_like(x, x.mean())).all()


def test_moving_average_normal():
    x = np.arange(10.0)
    out = moving_average(x, window=3)
    assert out.shape == x.shape
    # simple numeric check at center
    assert out[5] == pytest.approx((4 + 5 + 6) / 3.0)


def test_windowed_examples_short_sequence():
    # create a short sequence (N=50) with fs=10, window 2s -> seq_len=20
    fs = 10
    N = 50
    t = np.linspace(0, 5, N, endpoint=False)
    accel = 0.01 * np.random.RandomState(0).standard_normal((N, 3))
    gyro = 0.01 * np.random.RandomState(1).standard_normal((N, 3))
    seq = Sequence(timestamps=t, accel=accel, gyro=gyro, segments=None)
    X, y, windows = windowed_examples_from_sequence(seq, window_s=2.0, hop_s=1.0, fs=fs)
    assert X.shape[1] == int(2.0 * fs)
    assert len(y) == len(windows)
