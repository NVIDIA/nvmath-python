# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Test NVSHMEM bindings and nvmath.distributed core functionality relating to NVSHMEM.
"""

import re
import numpy as np
import pytest

import nvmath.distributed
from nvmath.bindings import nvshmem
from nvmath.internal.utils import device_ctx

import cuda.core.experimental

try:
    import cupy
except ImportError:
    cupy = None

try:
    import torch
except ImportError:
    torch = None


SHAPE = (2, 5)
VALUE = 17


@pytest.fixture(scope="module")
def nvmath_distributed():
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    device_id = comm.Get_rank() % cuda.core.experimental.system.num_devices
    nvmath.distributed.initialize(device_id)

    yield

    nvmath.distributed.finalize()


def test_nvshmem_bootstrapped(nvmath_distributed):
    ctx = nvmath.distributed.get_context()
    assert ctx is not None

    rank = ctx.communicator.Get_rank()
    nranks = ctx.communicator.Get_size()
    assert ctx.device_id == rank % cuda.core.experimental.system.num_devices
    assert nvshmem.my_pe() == rank
    assert nvshmem.n_pes() == nranks


def test_nvshmem_malloc(nvmath_distributed):
    # Allocate some memory with NVSHMEM
    ctx = nvmath.distributed.get_context()
    with device_ctx(ctx.device_id):
        ptr = nvshmem.malloc(4)
        # nvshmem.ptr(ptr) != 0 means that the pointer points to NVSHMEM-allocated memory.
        assert nvshmem.ptr(ptr, pe=nvshmem.my_pe()) != 0
        nvshmem.free(ptr)


@pytest.mark.filterwarnings("error::pytest.PytestUnraisableExceptionWarning")
def test_torch_symmetric_memory(nvmath_distributed):
    if torch is None:
        pytest.skip("torch is not available")

    dtype = torch.int32

    device_id = nvmath.distributed.get_context().device_id
    with torch.cuda.device(device_id):
        expected = torch.full(SHAPE, VALUE, dtype=dtype, device=f"cuda:{device_id}")

        tensor_sheap = nvmath.distributed.allocate_symmetric_memory(SHAPE, torch, dtype=dtype)
        tensor_sheap.fill_(VALUE)

    assert torch.equal(tensor_sheap, expected)

    mype = nvshmem.my_pe()
    assert nvshmem.ptr(expected.data_ptr(), pe=mype) == 0
    assert nvshmem.ptr(tensor_sheap.data_ptr(), pe=mype) != 0

    nvmath.distributed.free_symmetric_memory(tensor_sheap)


@pytest.mark.filterwarnings("error::pytest.PytestUnraisableExceptionWarning")
def test_cupy_symmetric_memory(nvmath_distributed):
    if cupy is None:
        pytest.skip("cupy is not available")

    dtype = cupy.int32

    device_id = nvmath.distributed.get_context().device_id
    with cupy.cuda.Device(device_id):
        expected = cupy.full(SHAPE, VALUE, dtype=dtype)

        tensor_sheap = nvmath.distributed.allocate_symmetric_memory(SHAPE, cupy, dtype=dtype)
        tensor_sheap.fill(VALUE)

    cupy.testing.assert_array_equal(tensor_sheap, expected)

    mype = nvshmem.my_pe()
    assert nvshmem.ptr(expected.data.ptr, pe=mype) == 0
    assert nvshmem.ptr(tensor_sheap.data.ptr, pe=mype) != 0

    nvmath.distributed.free_symmetric_memory(tensor_sheap)


@pytest.mark.filterwarnings("error::pytest.PytestUnraisableExceptionWarning")
def test_nvshmem_communication(nvmath_distributed):
    if cupy is None:
        pytest.skip("cupy is not available")

    ctx = nvmath.distributed.get_context()
    rank = ctx.communicator.Get_rank()
    nranks = ctx.communicator.Get_size()

    from mpi4py import MPI

    a = nvmath.distributed.allocate_symmetric_memory(1, cupy, dtype=cupy.int32)
    try:
        stream = cupy.cuda.get_current_stream(ctx.device_id)

        mype = nvshmem.my_pe()
        npes = nvshmem.n_pes()
        peer = (mype + 1) % npes
        assert mype == rank
        assert npes == nranks

        with cupy.cuda.Device(ctx.device_id):
            nvshmem.int_p(a.data.ptr, mype, peer)

            nvshmem.barrier_all_on_stream(stream.ptr)
            stream.synchronize()

            peer = (rank - 1) % nranks
            good = ctx.communicator.allreduce(a[0] == peer, op=MPI.LAND)
            assert good
    finally:
        nvmath.distributed.free_symmetric_memory(a)


def test_allocate_wrong_package(nvmath_distributed):
    with pytest.raises(ValueError, match=re.escape("The package must be one of ('cupy', 'torch'). Got <module 'numpy'")):
        nvmath.distributed.allocate_symmetric_memory((3, 2), np)


def test_free_wrong_package(nvmath_distributed):
    a = np.array([1, 2, 3])
    with pytest.raises(
        ValueError,
        match=re.escape("The tensor package must be one of ('cupy', 'torch'). Got <class 'numpy.ndarray'> from package numpy."),
    ):
        nvmath.distributed.free_symmetric_memory(a)


def test_cupy_distributed_tensor_error(nvmath_distributed):
    if cupy is None:
        pytest.skip("cupy is not available")

    device_id = nvmath.distributed.get_context().device_id
    with cupy.cuda.Device(device_id):
        a = cupy.ones(SHAPE, dtype=cupy.int32)

    from nvmath.distributed._internal.tensor_ifc_cupy import CupyDistributedTensor

    with pytest.raises(TypeError, match="Operand must be on the symmetric heap"):
        CupyDistributedTensor(a)


def test_torch_distributed_tensor_error(nvmath_distributed):
    if torch is None:
        pytest.skip("torch is not available")

    device_id = nvmath.distributed.get_context().device_id
    a = torch.ones(SHAPE, dtype=torch.int32, device=f"cuda:{device_id}")

    from nvmath.distributed._internal.tensor_ifc_torch import TorchDistributedTensor

    with pytest.raises(TypeError, match="Operand must be on the symmetric heap"):
        TorchDistributedTensor(a)
