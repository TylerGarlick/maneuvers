from typer.testing import CliRunner
from maneuvers.cli import app


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
