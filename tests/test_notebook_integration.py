import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


def test_execute_demo_notebook():
    nb_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "examples",
        "notebooks",
        "detection_classification_demo.ipynb",
    )
    nb_path = os.path.abspath(nb_path)
    with open(nb_path, "r", encoding="utf-8") as fh:
        nb = nbformat.read(fh, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    # execute the notebook in-place (in test) to ensure it runs without errors
    ep.preprocess(nb, {"metadata": {"path": os.path.dirname(nb_path)}})
