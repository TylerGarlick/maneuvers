"""Tests for export and plotting functionality."""

import pytest
import json
import csv
import tempfile
from pathlib import Path
import numpy as np

from maneuvers.data.loader import generate_synthetic_sequence
from maneuvers.preprocessing import compute_features_from_sequence
from maneuvers.detection import detect_segments
from maneuvers.export import export_segments, score_segment
from maneuvers.visualize import plot_3d_segments


def test_export_segments_json():
    seq = generate_synthetic_sequence(duration_s=5.0, fs=50)
    segments = [(10, 20), (30, 40)]
    labels = ["test1", "test2"]
    scores = [0.8, 0.9]
    
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        out_path = f.name
    
    try:
        export_segments(seq, segments, labels, scores, out_path)
        
        with open(out_path, 'r') as f:
            data = json.load(f)
        
        assert len(data) == 2
        assert data[0]['label'] == 'test1'
        assert data[0]['start_idx'] == 10
        assert data[0]['end_idx'] == 20
        assert data[0]['score'] == 0.8
        assert 'start_time' in data[0]
        assert 'end_time' in data[0]
        assert 'duration' in data[0]
    finally:
        Path(out_path).unlink()


def test_export_segments_csv():
    seq = generate_synthetic_sequence(duration_s=5.0, fs=50)
    segments = [(10, 20)]
    
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        out_path = f.name
    
    try:
        export_segments(seq, segments, out_path=out_path)
        
        with open(out_path, 'r') as f:
            reader = csv.DictReader(f)
            data = list(reader)
        
        assert len(data) == 1
        assert data[0]['label'] == 'unknown'
    finally:
        Path(out_path).unlink()


def test_score_segment():
    seq = generate_synthetic_sequence(duration_s=5.0, fs=50)
    segment = (10, 20)
    
    score = score_segment(seq, segment)
    assert 0 <= score <= 1


def test_plot_3d_segments():
    seq = generate_synthetic_sequence(duration_s=5.0, fs=50)
    segments = [(10, 20)]
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        out_path = f.name
    
    try:
        fig = plot_3d_segments(seq, segments, out_path=out_path)
        assert fig is not None
        assert Path(out_path).exists()
    finally:
        Path(out_path).unlink()