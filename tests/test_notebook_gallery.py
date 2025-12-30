import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from pathlib import Path


def test_execute_gallery_notebook(tmp_path: Path):
    nb_path = Path("examples/notebooks/maneuver_gallery.ipynb")
    nb = nbformat.read(str(nb_path), as_version=4)
    # ensure dataset exists at repo-relative examples/datasets/maneuvers_small
    from maneuvers.data.loader import generate_maneuvers_dataset
    data_dir = Path("examples/datasets/maneuvers_small")
    if not data_dir.exists():
        generate_maneuvers_dataset(data_dir, seed=0)

    ep = ExecutePreprocessor(timeout=240, kernel_name="python3")
    # run notebook with repo root as working-directory so the notebook's relative paths resolve
    ep.preprocess(nb, {"metadata": {"path": str(Path('.').resolve())}})
    # check that output images directory exists
    outdir = data_dir / "gallery"
    assert outdir.exists()
    imgs = list(outdir.glob("*.png"))
    assert len(imgs) >= 1
