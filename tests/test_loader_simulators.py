import tempfile
import os
import numpy as np
from maneuvers.data import loader


def _write_csv(path, header, rows):
    with open(path, "w") as fh:
        fh.write(header + "\n")
        for r in rows:
            fh.write(",".join(str(x) for x in r) + "\n")


def test_from_xplane_csv():
    header = "time,ax,ay,az,p,q,r"
    rows = [
        [0.0, 0.1, 0.2, -0.1, 0.01, 0.02, 0.03],
        [0.01, 0.11, 0.21, -0.11, 0.011, 0.021, 0.031],
    ]
    td = tempfile.mkdtemp()
    path = os.path.join(td, "xplane.csv")
    _write_csv(path, header, rows)

    seq = loader.from_xplane_csv(path)
    assert seq.timestamps.shape[0] == 2
    assert seq.accel.shape == (2, 3)
    assert seq.gyro.shape == (2, 3)
    assert np.allclose(seq.accel[0], [0.1, 0.2, -0.1])


def test_from_flightgear_csv():
    header = "time,udot,vdot,wdot,p,q,r"
    rows = [
        [0.0, 0.0, 0.1, -0.2, 0.001, 0.002, 0.003],
        [0.01, 0.01, 0.11, -0.21, 0.0011, 0.0021, 0.0031],
    ]
    td = tempfile.mkdtemp()
    path = os.path.join(td, "fg.csv")
    _write_csv(path, header, rows)

    seq = loader.from_flightgear_csv(path)
    assert seq.timestamps.shape[0] == 2
    assert seq.accel.shape == (2, 3)
    assert seq.gyro.shape == (2, 3)
    assert np.allclose(seq.accel[1], [0.01, 0.11, -0.21])


def test_from_simconnect_csv_case_insensitive():
    header = "Time,AccelerationX,AccelerationY,AccelerationZ,AngularVelocityX,AngularVelocityY,AngularVelocityZ"
    rows = [
        [0.0, 1.0, 2.0, -1.0, 0.1, 0.2, 0.3],
    ]
    td = tempfile.mkdtemp()
    path = os.path.join(td, "sim.csv")
    _write_csv(path, header, rows)

    seq = loader.from_simconnect_csv(path)
    assert seq.timestamps.shape[0] == 1
    assert seq.accel.shape == (1, 3)
    assert seq.gyro.shape == (1, 3)
    assert np.allclose(seq.accel[0], [1.0, 2.0, -1.0])


def test_from_jsbsim_csv():
    header = "time,ax,ay,az,p,q,r"
    rows = [
        [0.0, -0.5, 0.0, 0.5, -0.01, 0.0, 0.01],
    ]
    td = tempfile.mkdtemp()
    path = os.path.join(td, "jsb.csv")
    _write_csv(path, header, rows)

    seq = loader.from_jsbsim_csv(path)
    assert seq.timestamps.shape[0] == 1
    assert seq.accel.shape == (1, 3)
    assert seq.gyro.shape == (1, 3)
    assert np.allclose(seq.gyro[0], [-0.01, 0.0, 0.01])
