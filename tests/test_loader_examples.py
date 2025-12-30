from maneuvers.data import loader


def test_example_files_exist_and_load():
    # Sanity check: example files bundled with the repo can be loaded
    loader.from_xplane_csv("examples/simulators/xplane_example.csv")
    loader.from_flightgear_csv("examples/simulators/flightgear_example.csv")
    loader.from_simconnect_csv("examples/simulators/simconnect_example.csv")
    loader.from_jsbsim_csv("examples/simulators/jsbsim_example.csv")
