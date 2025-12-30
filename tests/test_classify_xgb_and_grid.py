import pytest
import numpy as np
from maneuvers.classify import build_pipeline, train_with_grid_search
from maneuvers.preprocessing import compute_features_from_sequence
from maneuvers.data.loader import generate_synthetic_sequence


def _make_small_dataset(n=20, seed=0):
    rng = np.random.default_rng(seed)
    X = []
    y = []
    for i in range(n):
        seq = generate_synthetic_sequence(
            duration_s=5.0, fs=50, seed=int(rng.integers(0, 1_000_000))
        )
        feats = compute_features_from_sequence(seq)
        Xv, yv = feats.iloc[0:10][
            ["accel_mag", "accel_smooth", "gyro_mag"]
        ].mean().values, ("A" if i % 2 == 0 else "B")
        X.append(Xv)
        y.append(yv)
    return np.vstack(X), y


@pytest.mark.skipif(
    not hasattr(__import__("importlib").import_module("importlib").util, "find_spec")
    or __import__("importlib").util.find_spec("xgboost") is None,
    reason="xgboost not installed",
)
def test_xgb_pipeline_and_training():
    # if xgboost is available, ensure we can build and train
    X, y = _make_small_dataset(20)
    pipe = build_pipeline("xgb")
    pipe.fit(X, y)
    preds = pipe.predict(X[:5])
    assert len(preds) == 5


@pytest.mark.parametrize("model_type", ["rf", "xgb"])
def test_train_with_grid_search_runs(model_type):
    # grid search should return best_params when possible; skip xgb if not installed
    if model_type == "xgb":
        spec = __import__("importlib").util.find_spec("xgboost")
        if spec is None:
            pytest.skip("xgboost not installed")
    X, y = _make_small_dataset(30)
    out = train_with_grid_search(X, y, model_type=model_type, param_grid=None, cv=3)
    assert "model" in out
    assert out["model"] is not None
