# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import pytest
import numpy as np


if importlib.util.find_spec("mpi4py") is None:
    pytest.skip("Skipping distributed tests because mpi4py is not installed.", allow_module_level=True)


def pytest_configure(config):
    config.addinivalue_line("markers", "need_4_procs: The test requires 4 processes")
    config.addinivalue_line("markers", "uncollect_if(*, func): function to unselect tests from parametrization")


def pytest_collection_modifyitems(config, items):
    removed = []
    kept = []
    for item in items:
        m = item.get_closest_marker("uncollect_if")
        if m:
            func = m.kwargs["func"]
            if func(**item.callspec.params):
                removed.append(item)
                continue
        kept.append(item)
    if removed:
        config.hook.pytest_deselected(items=removed)
        items[:] = kept


SYMMETRIC_MEMORY_LEAK_MESSAGE = "Symmetric heap memory needs to be deallocated explicitly"


@pytest.fixture
def check_symmetric_memory_leaks(caplog):
    """Check if an error message has been logged due to a NVSHMEM buffer being
    garbage-collected without the user having explicitly deleted it first, and
    raise an error to make test fail.

    NOTE: This is not a 100% reliable check since we depend on the garbage collector
    having collected all of a test's ndarrays/tensors by the time this check is done.
    We can make this reliable by running a full collection with `gc.collect()`, but
    this slows down testing and probably not worth it."""

    yield caplog

    error = False
    for record in caplog.get_records(when="call"):
        if SYMMETRIC_MEMORY_LEAK_MESSAGE in record.message:
            error = True
            break

    # Precaution in case of inconsistent garbage collector behavior across processes
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    error = np.array([error], dtype=np.bool_)
    comm.Allreduce(MPI.IN_PLACE, error, MPI.LOR)
    if error:
        raise MemoryError(SYMMETRIC_MEMORY_LEAK_MESSAGE)


@pytest.fixture(scope="session")
def symmetric_memory_leak_log_message():
    return SYMMETRIC_MEMORY_LEAK_MESSAGE
