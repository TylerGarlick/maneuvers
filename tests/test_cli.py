from typer.testing import CliRunner
from maneuvers.cli import app
import tempfile
import os

runner = CliRunner()


def test_detect_synthetic_command():
    result = runner.invoke(
        app, ["detect-synthetic", "--duration", "3", "--fs", "50", "--threshold", "0.2"]
    )
    assert result.exit_code == 0
    assert "Detected" in result.output


def test_eval_synthetic_command():
    result = runner.invoke(
        app, ["eval-synthetic", "--duration", "3", "--fs", "50", "--threshold", "0.2"]
    )
    assert result.exit_code == 0
    assert "precision" in result.output or "tp" in result.output


def test_train_synthetic_command(tmp_path):
    """Test the train-synthetic command."""
    model_path = tmp_path / "test_model.joblib"
    result = runner.invoke(
        app,
        ["train-synthetic", "--duration", "5", "--fs", "50", "--out", str(model_path)],
    )
    assert result.exit_code == 0
    assert "Saved model to" in result.output
    assert model_path.exists()


def test_detect_with_model(tmp_path):
    """Test detect-synthetic with model loading."""
    # First train a model
    model_path = tmp_path / "test_model.joblib"
    train_result = runner.invoke(
        app,
        ["train-synthetic", "--duration", "5", "--fs", "50", "--out", str(model_path)],
    )
    assert train_result.exit_code == 0

    # Now test detection with model
    result = runner.invoke(
        app,
        [
            "detect-synthetic",
            "--duration",
            "3",
            "--fs",
            "50",
            "--threshold",
            "0.2",
            "--model",
            str(model_path),
        ],
    )
    assert result.exit_code == 0
    assert "Detected" in result.output


def test_detect_with_invalid_model():
    """Test detect-synthetic with invalid model path."""
    result = runner.invoke(
        app,
        [
            "detect-synthetic",
            "--duration",
            "3",
            "--fs",
            "50",
            "--threshold",
            "0.2",
            "--model",
            "nonexistent.joblib",
        ],
    )
    assert result.exit_code == 0
    assert "Failed to load/label" in result.output


def test_detect_with_output_file(tmp_path):
    """Test detect-synthetic with output file."""
    output_path = tmp_path / "detections.json"
    result = runner.invoke(
        app,
        [
            "detect-synthetic",
            "--duration",
            "3",
            "--fs",
            "50",
            "--threshold",
            "0.2",
            "--out",
            str(output_path),
        ],
    )
    assert result.exit_code == 0
    assert output_path.exists()
