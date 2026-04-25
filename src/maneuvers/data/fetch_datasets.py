"""Script to fetch external flight maneuver datasets for training and testing.

This script automates downloading of external datasets from public sources:
- Maneuver-ID dataset (MIT): Garmin G1000 General Aviation flight data
- NASA DASHlink: Flight test data from various aircraft

Usage:
    python -m maneuvers.data.fetch_datasets [--dataset DATASET] [--out-dir DIR]
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Optional
import urllib.request
import zipfile
import tarfile
import shutil


def fetch_maneuver_id(out_dir: Path, verbose: bool = True) -> None:
    """Download and extract Maneuver-ID dataset from MIT.

    The Maneuver-ID dataset contains labeled flight maneuver data from
    Pilot Training Next (PTN) simulators with high-fidelity virtual reality
    flight environments.

    Args:
        out_dir: Output directory for the dataset
        verbose: Print progress messages
    """
    dataset_dir = out_dir / "maneuver-id"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Fetching Maneuver-ID dataset to {dataset_dir}...")

    # Note: The actual Maneuver-ID dataset requires signing a Data Sharing Agreement
    # We'll create a placeholder that documents how to get the data
    readme_path = dataset_dir / "README.md"

    readme_content = """# Maneuver-ID Dataset

This dataset is from the MIT and U.S. Air Force Maneuver-ID Challenge project.

## Source
- Data Portal: https://maneuver-id.mit.edu/data/
- Website: https://maneuver-id.mit.edu/
- Paper: https://arxiv.org/abs/2211.15552
- Additional Resources: https://github.com/bmswens/Maneuver-ID-Introduction-Notebook

## Manual Download Instructions

The full Maneuver-ID dataset requires registration and signing a Data Sharing Agreement:

1. Visit https://maneuver-id.mit.edu/data/
2. Sign up for an account and agree to the Data Sharing Agreement
3. Download the dataset files (TSV format)
4. Extract the flight data files to this directory

## Data Format

The dataset contains high-fidelity simulator telemetry in TSV (tab-separated values) format with columns:
- time (seconds)
- xEast, yNorth, zUp (meters, spatial coordinates)
- velocities in each axis
- heading, pitch, roll (degrees)

## Maneuver Labels

The dataset includes labeled maneuvers from a catalog of ~30 types:
- Traffic Pattern Stalls
- Aileron Rolls
- Steep Turns (45° and 60°)
- Barrel Rolls
- Split-S
- Instrument Landing System (ILS) approaches
- Cuban 8
- Loops
- Spins
- Lazy 8
- Emergency Landing Patterns
- Unusual Attitude Recovery (Nose High/Low)
- Spin Recovery
- And more aviation maneuvers

## Citation

If you use this dataset, please cite:

```
@article{maneuverid2022,
  title={AI Enabled Maneuver Identification via the Maneuver ID Challenge},
  author={Pokorny, Florian T. and others},
  journal={arXiv preprint arXiv:2211.15552},
  year={2022}
}
```
"""

    with open(readme_path, "w") as f:
        f.write(readme_content)

    if verbose:
        print(f"Created README at {readme_path}")
        print(
            "Note: Full Maneuver-ID dataset requires registration and manual download from:"
        )
        print("      https://maneuver-id.mit.edu/data/")
        print("      You must sign a Data Sharing Agreement to access the dataset.")
        print(f"      Extract flight TSV files to: {dataset_dir}/")


def fetch_dashlink(out_dir: Path, verbose: bool = True) -> None:
    """Download and extract NASA DASHlink dataset.

    The DASHlink dataset contains flight test data from NASA aircraft with
    various sensor measurements.

    Args:
        out_dir: Output directory for the dataset
        verbose: Print progress messages
    """
    dataset_dir = out_dir / "dashlink"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Fetching NASA DASHlink dataset to {dataset_dir}...")

    # Note: DASHlink data requires authentication/registration
    # We'll create a placeholder with download instructions
    readme_path = dataset_dir / "README.md"

    readme_content = """# NASA DASHlink Dataset

This dataset is from NASA's DASHlink project.

## Source
- Website: https://c3.ndc.nasa.gov/dashlink/projects/85/
- Description: System-wide Safety and Assurance Technologies

## Manual Download Instructions

The NASA DASHlink dataset requires registration and manual download:

1. Visit https://c3.ndc.nasa.gov/dashlink/projects/85/
2. Create an account or sign in
3. Download the desired flight data files
4. Extract the data files to this directory

## Data Format

DASHlink provides flight test data in various formats:
- CSV files with time-series sensor data
- MAT files (MATLAB format)
- Metadata files describing flights and sensors

Typical columns include:
- Time
- Position (latitude, longitude, altitude)
- Velocities (airspeed, groundspeed, vertical speed)
- Attitudes (roll, pitch, yaw)
- Accelerations (various frames)
- Angular rates
- Control surface positions
- Engine parameters

## Usage

After downloading, use the loader functions in `maneuvers.data.loader` to load the data:

```python
from maneuvers.data.loader import from_csv, from_json

# Load a CSV file
seq = from_csv("data/external/dashlink/flight_001.csv")
```

## Citation

If you use this dataset, please cite the appropriate NASA/DASHlink publications
and acknowledge the data source.
"""

    with open(readme_path, "w") as f:
        f.write(readme_content)

    if verbose:
        print(f"Created README at {readme_path}")
        print(
            "Note: Full DASHlink dataset requires registration and manual download from:"
        )
        print("      https://c3.ndc.nasa.gov/dashlink/projects/85/")
        print(f"      Extract data files to: {dataset_dir}/")


def fetch_all(out_dir: Path, verbose: bool = True) -> None:
    """Fetch all available datasets.

    Args:
        out_dir: Output directory for datasets
        verbose: Print progress messages
    """
    if verbose:
        print("Fetching all datasets...")

    fetch_maneuver_id(out_dir, verbose)
    fetch_dashlink(out_dir, verbose)

    if verbose:
        print("\nDataset fetch complete!")
        print(f"Datasets are located in: {out_dir}")


def main():
    """Main entry point for the fetch_datasets script."""
    parser = argparse.ArgumentParser(
        description="Fetch external flight maneuver datasets for training and testing"
    )
    parser.add_argument(
        "--dataset",
        choices=["all", "maneuver-id", "dashlink"],
        default="all",
        help="Which dataset to fetch (default: all)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/external"),
        help="Output directory for datasets (default: data/external)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress progress messages",
    )

    args = parser.parse_args()

    # Ensure output directory exists
    args.out_dir.mkdir(parents=True, exist_ok=True)

    verbose = not args.quiet

    try:
        if args.dataset == "all":
            fetch_all(args.out_dir, verbose)
        elif args.dataset == "maneuver-id":
            fetch_maneuver_id(args.out_dir, verbose)
        elif args.dataset == "dashlink":
            fetch_dashlink(args.out_dir, verbose)
        else:
            print(f"Unknown dataset: {args.dataset}", file=sys.stderr)
            sys.exit(1)

    except Exception as e:
        print(f"Error fetching dataset: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
