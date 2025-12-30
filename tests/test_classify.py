from typer.testing import CliRunner
from maneuvers.cli import app

runner = CliRunner()


def test_train_synthetic_creates_model(tmp_path):
    out = tmp_path / "model.joblib"
    result = runner.invoke(
        app, ["train-synthetic", "--duration", "3", "--fs", "50", "--out", str(out)]
    )
    assert result.exit_code == 0
    assert out.exists()


def test_detect_with_model(tmp_path):
    out = tmp_path / "model.joblib"
    # first create the model
    r1 = runner.invoke(app, ["train-synthetic", "--out", str(out)])
    assert r1.exit_code == 0
    # ensure detect-synthetic labels using model runs
    r2 = runner.invoke(
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
            str(out),
        ],
    )
    assert r2.exit_code == 0
    assert "Detected" in r2.output
