import importlib.util
import pytest


if importlib.util.find_spec("mpi4py") is None:
    pytest.skip("Skipping distributed tests because mpi4py is not installed.", allow_module_level=True)


def pytest_configure(config):
    config.addinivalue_line("markers", "need_4_procs: The test requires 4 processes")
