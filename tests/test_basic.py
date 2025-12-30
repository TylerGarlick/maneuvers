from maneuvers import main


def test_main_exit_code():
    # call main with a small argv to ensure it returns 0
    assert main(["--name", "Test"]) == 0
