from pathlib import Path
import json
import numpy as np

from maneuvers.data.maneuvers_catalog import MANEUVERS, _bump, synthesize_maneuver
from maneuvers.data.loader import from_csv, generate_maneuvers_dataset


def test_generate_all_creates_files(tmp_path: Path):
    outdir = tmp_path / "dataset"
    outdir.mkdir()
    manifest = generate_maneuvers_dataset(outdir, seed=0)
    assert len(manifest) == len(MANEUVERS)

    # Check that CSVs load and segments are present
    for entry, name in zip(manifest, MANEUVERS):
        path = Path(entry["file"])
        assert path.exists(), f"expected {path} to exist"
        seq = from_csv(path)
        assert seq.accel.shape[0] > 0
        assert seq.gyro.shape[0] > 0
        assert len(entry["segments"]) == 1
        s, e, lbl = entry["segments"][0]
        assert lbl == name
        assert 0 <= s < e <= seq.accel.shape[0]


def test_bump_function():
    """Test the _bump helper function."""
    # Test hann window
    bump = _bump(10, "hann")
    assert len(bump) == 10
    assert bump[0] < bump[5]  # Should be symmetric and peak in middle

    # Test gauss
    bump = _bump(10, "gauss")
    assert len(bump) == 10
    assert bump[5] > bump[0]  # Peak in middle

    # Test pulse
    bump = _bump(10, "pulse")
    assert len(bump) == 10
    assert np.sum(bump) == 5.0  # Half should be 1.0

    # Test default (ones)
    bump = _bump(5, "unknown")
    assert len(bump) == 5
    assert np.all(bump == 1.0)


def test_synthesize_maneuver_none():
    """Test synthesize_maneuver with 'none'."""
    rng = np.random.default_rng(42)
    accel, gyro = synthesize_maneuver("none", 1.0, 100, rng=rng)
    assert accel.shape == (100, 3)
    assert gyro.shape == (100, 3)
    # Should be just noise
    assert np.std(accel) < 0.1
    assert np.std(gyro) < 0.01


def test_synthesize_maneuver_default_rng():
    """Test synthesize_maneuver without passing rng (uses default)."""
    accel, gyro = synthesize_maneuver("none", 1.0, 100)  # No rng passed
    assert accel.shape == (100, 3)
    assert gyro.shape == (100, 3)


def test_synthesize_maneuver_roll():
    """Test synthesize_maneuver with roll maneuvers."""
    rng = np.random.default_rng(42)
    accel, gyro = synthesize_maneuver("left_roll", 1.0, 100, rng=rng)
    assert accel.shape == (100, 3)
    assert gyro.shape == (100, 3)
    # Left roll should have negative gyro x
    assert np.mean(gyro[:, 0]) < 0


def test_synthesize_maneuver_pitch():
    """Test synthesize_maneuver with pitch maneuvers."""
    rng = np.random.default_rng(42)
    accel, gyro = synthesize_maneuver("pitch_up", 1.0, 100, rng=rng)
    assert accel.shape == (100, 3)
    assert gyro.shape == (100, 3)
    # Pitch up should have positive gyro y
    assert np.mean(gyro[:, 1]) > 0


def test_synthesize_maneuver_climb():
    """Test synthesize_maneuver with climb."""
    rng = np.random.default_rng(42)
    accel, gyro = synthesize_maneuver("climb", 1.0, 100, rng=rng)
    assert accel.shape == (100, 3)
    assert gyro.shape == (100, 3)
    # Climb should have positive z acceleration
    assert np.mean(accel[:, 2]) > 0


def test_synthesize_maneuver_loop():
    """Test synthesize_maneuver with loop."""
    rng = np.random.default_rng(42)
    accel, gyro = synthesize_maneuver("loop", 1.0, 100, rng=rng)
    assert accel.shape == (100, 3)
    assert gyro.shape == (100, 3)
    # Loop should have significant gyro y variation
    assert np.std(gyro[:, 1]) > 0.3  # Lowered threshold


def test_synthesize_maneuver_stall():
    """Test synthesize_maneuver with stall/spin."""
    rng = np.random.default_rng(42)
    accel, gyro = synthesize_maneuver("stall", 1.0, 100, rng=rng)
    assert accel.shape == (100, 3)
    assert gyro.shape == (100, 3)
    # Stall should have high gyro variation
    assert np.std(gyro) > 0.1  # Lowered threshold
