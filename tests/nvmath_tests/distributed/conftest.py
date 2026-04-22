# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
import sys

import numpy as np
import pytest

if sys.platform != "linux":
    pytest.skip("Skipping distributed tests (require Linux).", allow_module_level=True)


if importlib.util.find_spec("mpi4py") is None and "TORCHELASTIC_RUN_ID" not in os.environ:
    pytest.skip(
        "Skipping distributed tests because mpi4py is not installed and not launched with torchrun.", allow_module_level=True
    )


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


@pytest.fixture(scope="session")
def process_group():
    try:
        from cuda.core import system
    except ImportError:
        from cuda.core.experimental import system

    try:
        num_devices = system.get_num_devices()
    except AttributeError:
        num_devices = system.num_devices

    from nvmath.distributed import MPIProcessGroup, TorchProcessGroup

    if "TORCHELASTIC_RUN_ID" in os.environ:
        # Launched with torchrun.
        assert "mpi4py" not in sys.modules
        import torch.distributed as dist

        rank = int(os.environ["RANK"])
        nranks = int(os.environ["WORLD_SIZE"])
        device_id = rank % num_devices
        backend = "nccl" if num_devices >= nranks else "gloo"
        dist.init_process_group(backend=backend, device_id=device_id if backend != "gloo" else None)
        process_group = TorchProcessGroup(device_id="cpu" if backend == "gloo" else device_id)
    else:
        # Assume launched with MPI.
        from mpi4py import MPI

        process_group = MPIProcessGroup(MPI.COMM_WORLD)

    yield process_group

    if "TORCHELASTIC_RUN_ID" in os.environ:
        dist.barrier()
        dist.destroy_process_group()


@pytest.fixture(scope="module", autouse=True)
def check_mpi4py_import():
    # This runs for every test module.
    # Check that, if tests were launched with torchrun and not MPI,
    # mpi4py is never imported during the test suite.
    if "TORCHELASTIC_RUN_ID" in os.environ:
        # Launched with torchrun.
        assert "mpi4py" not in sys.modules, "mpi4py was imported during tests that are running with torchrun"


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
    import nvmath.distributed
    from nvmath.distributed.process_group import ReductionOp

    # This fixture should only be used with nvmath.distributed initialized.
    ctx = nvmath.distributed.get_context()
    assert ctx is not None, "nvmath.distributed is not initialized"

    error = np.array([int(error)], dtype=np.int8)
    ctx.process_group.allreduce_buffer(error, op=ReductionOp.MAX)
    error = error[0] != 0
    if error:
        raise MemoryError(SYMMETRIC_MEMORY_LEAK_MESSAGE)


@pytest.fixture(scope="session")
def symmetric_memory_leak_log_message():
    return SYMMETRIC_MEMORY_LEAK_MESSAGE
