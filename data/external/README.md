# External Flight Maneuver Datasets

This directory houses external datasets that can be downloaded for training and testing maneuver detection and classification models.

## Available Datasets

### 1. Maneuver-ID (MIT)
- **Source**: https://maneuver-id.mit.edu/data/
- **Website**: https://maneuver-id.mit.edu/
- **Paper**: https://arxiv.org/abs/2211.15552
- **Description**: Flight maneuver data from Pilot Training Next (PTN) simulators, including labeled maneuvers from Air Force pilot training with high-fidelity virtual reality simulators
- **Format**: TSV (tab-separated values) files with time-series positional and orientational data
- **License**: Requires signing a Data Sharing Agreement via the MIT website

### 2. NASA DASHlink
- **Source**: https://c3.ndc.nasa.gov/dashlink/projects/85/
- **Description**: Flight test data from various NASA aircraft with sensor telemetry
- **Format**: Various CSV/MAT formats
- **License**: Public domain (NASA data)

## Usage

### Downloading Datasets

Use the provided fetch script to automatically download datasets:

```bash
# Download all datasets
python -m maneuvers.data.fetch_datasets

# Download specific dataset
python -m maneuvers.data.fetch_datasets --dataset maneuver-id
python -m maneuvers.data.fetch_datasets --dataset dashlink
```

### Training with External Data

Once downloaded, you can use the datasets for training:

```bash
# Train using external Maneuver-ID data
maneuvers train --data-dir data/external/maneuver-id --out model_maneuver_id.joblib

# Detect and evaluate using trained model
maneuvers detect-real --data-path data/external/maneuver-id/flight_001.csv --model model_maneuver_id.joblib
```

## Sample Data

A small sample from the Maneuver-ID dataset is included in `examples/data/maneuver_id_sample.csv` for quick testing without downloading the full dataset.

## Directory Structure

After downloading, the directory structure will be:

```
data/external/
├── README.md (this file)
├── maneuver-id/
│   ├── flight_001.csv
│   ├── flight_002.csv
│   └── ...
└── dashlink/
    ├── sample_001.csv
    └── ...
```

## Notes

- Downloaded datasets are ignored by git (see `.gitignore`)
- The sample data in `examples/data/` is version controlled for quick demos
- Full datasets may be large (hundreds of MB to GB); ensure adequate disk space
