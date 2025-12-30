import tempfile
import os
import json
import numpy as np
from maneuvers.data import loader


def _write_csv(path, header, rows):
    with open(path, "w") as fh:
        fh.write(header + "\n")
        for r in rows:
            fh.write(",".join(str(x) for x in r) + "\n")


def test_unit_conversion_deg_to_rad_explicit():
    header = "time,ax,ay,az,p,q,r"
    rows = [[0.0, 0.0, 0.0, 0.0, 180.0, 0.0, 0.0]]
    td = tempfile.mkdtemp()
    path = os.path.join(td, "deg.csv")
    _write_csv(path, header, rows)

    seq = loader.from_xplane_csv(path, convert_gyro_deg_to_rad=True)
    # p = 180 deg/s -> pi rad/s
    assert np.isclose(seq.gyro[0, 0], np.pi)


def test_unit_conversion_deg_to_rad_auto():
    header = "time,ax,ay,az,p,q,r"
    rows = [[0.0, 0.0, 0.0, 0.0, 90.0, 0.0, 0.0], [0.01, 0.0, 0.0, 0.0, 95.0, 0.0, 0.0]]
    td = tempfile.mkdtemp()
    path = os.path.join(td, "deg_auto.csv")
    _write_csv(path, header, rows)

    seq = loader.from_xplane_csv(path, convert_gyro_deg_to_rad=None)
    # mean ~92.5 deg/s -> converted to rad/s; check one value
    assert seq.gyro.shape[0] == 2
    assert seq.gyro[0, 0] > 1.0  # > 1 rad/s


def test_json_loader_per_axis():
    td = tempfile.mkdtemp()
    path = os.path.join(td, "data.json")
    data = {
        "time": [0.0, 0.01],
        "ax": [0.1, 0.2],
        "ay": [0.0, 0.0],
        "az": [-0.1, -0.05],
        "p": [0.01, 0.02],
        "q": [0.0, 0.0],
        "r": [0.0, 0.0],
    }
    with open(path, "w") as fh:
        json.dump(data, fh)
    seq = loader.from_json(path)
    assert seq.accel.shape == (2, 3)
    assert seq.gyro.shape == (2, 3)


def test_csv_filelike_loader_heuristic():
    # use existing example file
    with open("examples/simulators/simconnect_example.csv", "r") as fh:
        seq = loader.from_csv_filelike(fh)
        assert seq.accel.shape[1] == 3


def test_malformed_missing_columns_raises():
    header = "time,ax,ay"
    rows = [[0.0, 0.1, 0.2]]
    td = tempfile.mkdtemp()
    path = os.path.join(td, "bad.csv")
    _write_csv(path, header, rows)

    try:
        loader.from_xplane_csv(path)
        assert False, "Expected a KeyError for missing columns"
    except KeyError:
        pass


def test_large_file_loads():
    td = tempfile.mkdtemp()
    path = os.path.join(td, "large.csv")
    header = "time,ax,ay,az,p,q,r"
    rows = []
    N = 5000
    for i in range(N):
        rows.append([i * 0.01, 0.01, 0.0, -0.01, 0.001, 0.0, 0.0])
    _write_csv(path, header, rows)

    seq = loader.from_xplane_csv(path)
    assert seq.timestamps.shape[0] == N
    assert seq.accel.shape == (N, 3)
