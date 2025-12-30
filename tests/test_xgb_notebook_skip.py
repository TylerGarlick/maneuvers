import importlib


def test_xgb_notebook_available_or_skip():
    spec = importlib.util.find_spec("xgboost")
    if spec is None:
        import pytest

        pytest.skip("xgboost not installed in this environment")
    # otherwise a smoke import check
    import xgboost as xgb

    assert hasattr(xgb, "XGBClassifier")
