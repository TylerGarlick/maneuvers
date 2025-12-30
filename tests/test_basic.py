from maneuvers import main
from maneuvers.preprocessing import moving_average
import numpy as np


def test_main_exit_code():
    # call main with a small argv to ensure it returns 0
    assert main(["--name", "Test"]) == 0


def test_moving_average_window_one():
    """Test moving_average with window <= 1 returns input unchanged."""
    x = np.array([1, 2, 3, 4, 5])
    result = moving_average(x, window=1)
    np.testing.assert_array_equal(result, x)
