from maneuvers.data.loader import generate_synthetic_sequence


def test_generate_synthetic_sequence_shapes():
    seq = generate_synthetic_sequence(duration_s=5.0, fs=50, seed=1)
    assert seq.timestamps.shape[0] == 250
    assert seq.accel.shape == (250, 3)
    assert seq.gyro.shape == (250, 3)
    assert isinstance(seq.segments, list)
    assert len(seq.segments) >= 1
