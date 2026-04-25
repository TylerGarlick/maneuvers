"""Tests for external dataset loaders (Maneuver-ID TSV and Garmin G1000 CSV)."""

import pytest
import numpy as np
from pathlib import Path
from maneuvers.data.loader import from_garmin_g1000_csv, from_maneuver_id_tsv


def test_from_garmin_g1000_csv_sample():
    """Test loading the Garmin G1000 sample CSV file."""
    sample_path = Path("examples/data/maneuver_id_sample.csv")

    if not sample_path.exists():
        pytest.skip("Sample Garmin G1000 CSV not found")

    seq = from_garmin_g1000_csv(str(sample_path))

    # Check basic structure
    assert seq.timestamps is not None
    assert len(seq.timestamps) > 0
    assert seq.accel.shape == (len(seq.timestamps), 3)
    assert seq.gyro.shape == (len(seq.timestamps), 3)

    # Check optional fields are present
    assert seq.pos is not None
    assert seq.vel is not None
    assert seq.orient is not None

    # Check that gyro was converted from deg/s to rad/s (default)
    # Original sample has small values in deg/s, so converted should be even smaller
    assert np.all(np.abs(seq.gyro) < 1.0)  # Typical rad/s values


def test_from_garmin_g1000_csv_unit_conversion():
    """Test unit conversion options for Garmin G1000 loader."""
    sample_path = Path("examples/data/maneuver_id_sample.csv")

    if not sample_path.exists():
        pytest.skip("Sample Garmin G1000 CSV not found")

    # Load without gyro conversion
    seq_no_conv = from_garmin_g1000_csv(str(sample_path), convert_gyro_deg_to_rad=False)

    # Load with gyro conversion
    seq_conv = from_garmin_g1000_csv(str(sample_path), convert_gyro_deg_to_rad=True)

    # Converted values should be smaller (deg to rad)
    assert np.mean(np.abs(seq_conv.gyro)) < np.mean(np.abs(seq_no_conv.gyro))


def test_from_maneuver_id_tsv_basic(tmp_path):
    """Test basic Maneuver-ID TSV loader functionality."""
    # Create a minimal TSV file
    tsv_content = """time\txEast\tyNorth\tzUp\tvxEast\tvyNorth\tvzUp\theading\tpitch\troll
0.0\t0.0\t0.0\t100.0\t50.0\t0.0\t0.0\t0.0\t0.0\t0.0
0.1\t5.0\t0.0\t100.0\t50.0\t0.0\t0.0\t0.0\t0.0\t0.0
0.2\t10.0\t0.0\t100.0\t50.0\t0.0\t0.0\t0.0\t0.0\t0.0
0.3\t15.0\t0.0\t100.0\t50.0\t0.0\t0.0\t0.0\t0.0\t0.0
0.4\t20.0\t0.0\t100.0\t50.0\t0.0\t0.0\t0.0\t0.0\t0.0
"""

    tsv_file = tmp_path / "test_maneuver.tsv"
    tsv_file.write_text(tsv_content)

    seq = from_maneuver_id_tsv(str(tsv_file))

    # Check basic structure
    assert seq.timestamps is not None
    assert len(seq.timestamps) == 5
    assert seq.accel.shape == (5, 3)
    assert seq.gyro.shape == (5, 3)

    # Check optional fields
    assert seq.pos is not None
    assert seq.vel is not None
    assert seq.orient is not None

    # Check position values match input
    assert np.allclose(seq.pos[0], [0.0, 0.0, 100.0])
    assert np.allclose(seq.pos[1], [5.0, 0.0, 100.0])


def test_from_maneuver_id_tsv_acceleration_computation(tmp_path):
    """Test that acceleration is properly computed from velocity."""
    # Create TSV with constant velocity (should yield near-zero acceleration)
    tsv_content = """time\txEast\tyNorth\tzUp\tvxEast\tvyNorth\tvzUp\theading\tpitch\troll
0.0\t0.0\t0.0\t0.0\t10.0\t5.0\t2.0\t0.0\t0.0\t0.0
0.1\t1.0\t0.5\t0.2\t10.0\t5.0\t2.0\t0.0\t0.0\t0.0
0.2\t2.0\t1.0\t0.4\t10.0\t5.0\t2.0\t0.0\t0.0\t0.0
0.3\t3.0\t1.5\t0.6\t10.0\t5.0\t2.0\t0.0\t0.0\t0.0
"""

    tsv_file = tmp_path / "test_const_vel.tsv"
    tsv_file.write_text(tsv_content)

    seq = from_maneuver_id_tsv(str(tsv_file))

    # With constant velocity, acceleration should be near zero
    assert np.all(np.abs(seq.accel) < 1.0)  # Small values due to numerical diff


def test_loader_compatibility_with_pipeline():
    """Test that external loaders produce sequences compatible with the pipeline."""
    from maneuvers.preprocessing import compute_features_from_sequence
    from maneuvers.detection import detect_segments

    sample_path = Path("examples/data/maneuver_id_sample.csv")

    if not sample_path.exists():
        pytest.skip("Sample Garmin G1000 CSV not found")

    # Load sequence
    seq = from_garmin_g1000_csv(str(sample_path))

    # Compute features (should not raise)
    feats = compute_features_from_sequence(seq)

    # Check feature columns exist
    assert "accel_mag" in feats.columns
    assert "accel_smooth" in feats.columns
    assert "gyro_mag" in feats.columns

    # Run detection (should not raise)
    preds = detect_segments(feats, method="threshold", threshold=0.5, min_len=5)

    # Should return a list (possibly empty for this sample)
    assert isinstance(preds, list)
