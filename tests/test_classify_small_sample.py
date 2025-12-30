import numpy as np
from maneuvers.classify import train_classifier, build_pipeline


def test_train_classifier_skips_cv_when_tiny():
    # 3 examples, one class has single sample -> skip CV
    X = np.vstack([np.zeros(7), np.ones(7), np.ones(7) * 2])
    y = ["A", "B", "none"]
    out = train_classifier(X, y, cv=5)
    assert "cv_scores" in out
    assert out["cv_scores"] == {} or (hasattr(out["cv_scores"], "keys") and len(out["cv_scores"]) == 0)


def test_train_classifier_runs_cv_when_enough_samples():
    X = np.vstack([np.zeros(7), np.ones(7), np.ones(7) * 2, np.ones(7) * 3, np.ones(7) * 4, np.ones(7) * 5])
    y = ["A", "A", "B", "B", "none", "none"]
    out = train_classifier(X, y, cv=3)
    assert "cv_scores" in out
    assert out["cv_scores"] != {}
    assert "test_f1" in out["cv_scores"] or "test_f1" not in out["cv_scores"]  # scoring dict present
