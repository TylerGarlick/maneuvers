"""Generate a small example dataset with one sample per maneuver from MANEUVERS.

This script writes per-maneuver CSV files and a JSON manifest with ground-truth
segments. It's intended for demos and quick prototyping (not a large realistic
corpus).
"""

from __future__ import annotations
from pathlib import Path
import json

from maneuvers.data.loader import (
    generate_maneuvers_dataset,
    generate_synthetic_sequence,
    from_csv,
)

OUTDIR = Path("examples/datasets/maneuvers_small")


def generate_all(outdir: Path = OUTDIR, seed: int | None = 0):
    manifest = generate_maneuvers_dataset(outdir, seed=seed)
    print(f"Wrote {len(manifest)} files to {outdir}")


def generate_combined_training_data(
    num_synthetic: int = 10, outdir: Path = OUTDIR, seed: int | None = 0
):
    """Generate combined training data: synthetic sequences + maneuvers_small dataset.

    Returns list of Sequence objects with segments.
    """
    sequences = []

    # Generate synthetic sequences
    for i in range(num_synthetic):
        seq = generate_synthetic_sequence(
            duration_s=10.0, fs=100, seed=seed + i if seed else None
        )
        sequences.append(seq)

    # Load maneuvers_small dataset
    manifest_path = outdir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r") as fh:
            manifest = json.load(fh)
        for item in manifest:
            seq = from_csv(item["file"])
            seq.segments = item["segments"]
            sequences.append(seq)

    return sequences


if __name__ == "__main__":
    generate_all()
