from pathlib import Path
import json

from maneuvers.data.maneuvers_catalog import MANEUVERS
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
