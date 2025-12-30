"""Generate a synthetic CSV and ground-truth JSON for examples."""

from maneuvers.data.loader import generate_synthetic_sequence
import csv
import json


def write_example(
    path_csv: str = "examples/synthetic.csv",
    path_gt: str = "examples/synthetic_gt.json",
):
    seq = generate_synthetic_sequence(duration_s=10.0, fs=100)
    with open(path_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["t", "ax", "ay", "az", "gx", "gy", "gz"])
        for i in range(len(seq.timestamps)):
            row = [
                f"{seq.timestamps[i]:.6f}",
                f"{seq.accel[i, 0]:.6f}",
                f"{seq.accel[i, 1]:.6f}",
                f"{seq.accel[i, 2]:.6f}",
                f"{seq.gyro[i, 0]:.6f}",
                f"{seq.gyro[i, 1]:.6f}",
                f"{seq.gyro[i, 2]:.6f}",
            ]
            writer.writerow(row)
    # write ground truth segments
    with open(path_gt, "w") as fh:
        json.dump({"segments": seq.segments}, fh)


if __name__ == "__main__":
    write_example()
