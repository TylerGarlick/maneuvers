"""Quick visualization script for the small maneuvers dataset.

Usage: python examples/visualize_maneuvers.py --outdir examples/datasets/maneuvers_small/visuals
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import csv

DATADIR = Path("examples/datasets/maneuvers_small")
OUTDIR = DATADIR / "visuals"
OUTDIR.mkdir(parents=True, exist_ok=True)

files = sorted([p for p in DATADIR.glob("*.csv")])[:6]

for f in files:
    t = []
    ax = []
    ay = []
    az = []
    gx = []
    gy = []
    gz = []
    with open(f, "r") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            t.append(float(r["t"]))
            ax.append(float(r["ax"]))
            ay.append(float(r["ay"]))
            az.append(float(r["az"]))
            gx.append(float(r["gx"]))
            gy.append(float(r["gy"]))
            gz.append(float(r["gz"]))

    t = np.array(t)
    accel_mag = np.sqrt(np.array(ax) ** 2 + np.array(ay) ** 2 + np.array(az) ** 2)
    gyro_mag = np.sqrt(np.array(gx) ** 2 + np.array(gy) ** 2 + np.array(gz) ** 2)

    fig, axs = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
    axs[0].plot(t, accel_mag, label="|a|")
    axs[0].set_ylabel("accel (m/s^2)")
    axs[1].plot(t, gyro_mag, label="|ω|", color="C1")
    axs[1].set_ylabel("gyro (rad/s)")
    axs[1].set_xlabel("time (s)")
    plt.suptitle(f.name)
    out = OUTDIR / (f.stem + ".png")
    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)

print(f"Wrote visuals to {OUTDIR}")
