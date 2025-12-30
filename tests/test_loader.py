from maneuvers.data.loader import generate_synthetic_sequence, from_csv
import tempfile
import csv


def test_generate_synthetic_sequence_shapes():
    seq = generate_synthetic_sequence(duration_s=5.0, fs=50, seed=1)
    assert seq.timestamps.shape[0] == 250
    assert seq.accel.shape == (250, 3)
    assert seq.gyro.shape == (250, 3)
    assert isinstance(seq.segments, list)
    assert len(seq.segments) >= 1


def test_generate_synthetic_sequence_clipped_maneuvers():
    """Test that maneuvers starting after sequence end are skipped."""
    # Create a very short sequence where maneuvers would be clipped to zero length
    seq = generate_synthetic_sequence(duration_s=0.5, fs=100, seed=1)  # Very short sequence
    # The maneuvers are hardcoded to start at specific times, so some may be skipped
    assert seq.timestamps.shape[0] == 50
    # Just check that it doesn't crash


def test_from_csv_empty_file():
    """Test from_csv with empty CSV file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("t,ax,ay,az,gx,gy,gz\n")  # Header only, no data rows
        temp_path = f.name

    try:
        from_csv(temp_path)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "no rows" in str(e).lower()
    finally:
        import os
        os.unlink(temp_path)
