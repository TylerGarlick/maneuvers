from maneuvers.data.loader import generate_synthetic_sequence
from maneuvers.preprocessing import compute_features_from_sequence
from maneuvers.classify import (
    build_training_data_from_sequence,
    train_classifier,
    save_model,
    load_model,
)
import tempfile


def test_train_classifier_cv():
    seq = generate_synthetic_sequence(duration_s=6.0, fs=50, seed=3)
    feats = compute_features_from_sequence(seq)
    X, y = build_training_data_from_sequence(seq, feats)
    res = train_classifier(X, y, model_type="rf", cv=3)
    assert "model" in res and "encoder" in res and "cv_scores" in res
    # model should be able to predict without error
    model = res["model"]
    preds = model.predict(X)
    assert len(preds) == X.shape[0]
    # save and reload
    tmp = tempfile.NamedTemporaryFile(suffix=".joblib", delete=False)
    save_model(res, tmp.name)
    loaded = load_model(tmp.name)
    assert "model" in loaded and "encoder" in loaded
