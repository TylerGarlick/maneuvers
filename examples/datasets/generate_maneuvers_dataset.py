"""Generate a small example dataset with one sample per maneuver from MANEUVERS.

This script writes per-maneuver CSV files and a JSON manifest with ground-truth
segments. It's intended for demos and quick prototyping (not a large realistic
corpus).
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List
import numpy as np

from pathlib import Path
from maneuvers.data.loader import generate_maneuvers_dataset

OUTDIR = Path("examples/datasets/maneuvers_small")


def generate_all(outdir: Path = OUTDIR, seed: int | None = 0):
    manifest = generate_maneuvers_dataset(outdir, seed=seed)
    print(f"Wrote {len(manifest)} files to {outdir}")


if __name__ == "__main__":
    generate_all()
