# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Test NVSHMEM bindings and nvmath.distributed core functionality relating to NVSHMEM.
"""

import importlib
import gc
import re
import numpy as np
import pytest

import nvmath.distributed
from nvmath.bindings import nvshmem
from nvmath.internal.utils import device_ctx, get_or_create_stream
from nvmath.distributed._internal.tensor_ifc import DistributedTensor
from nvmath.distributed._internal.tensor_wrapper import maybe_register_package
from .helpers import is_close

import cuda.core.experimental

SHAPE = (2, 5)
VALUE = 17


def cupy_installed():
    return importlib.util.find_spec("cupy") is not None


def torch_installed():
    return importlib.util.find_spec("torch") is not None


@pytest.fixture(scope="module")
def nvmath_distributed():
    from mpi4py import MPI

    if cupy_installed():
        maybe_register_package("cupy")

    if torch_installed():
        maybe_register_package("torch")

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
    # Allocate some memory with NVSHMEM, using nvshmem bindings
    ctx = nvmath.distributed.get_context()
    with device_ctx(ctx.device_id):
        ptr = nvshmem.malloc(4)
        # nvshmem.ptr(ptr) != 0 means that the pointer points to NVSHMEM-allocated memory.
        assert nvshmem.ptr(ptr, pe=nvshmem.my_pe()) != 0
        nvshmem.free(ptr)


@pytest.mark.parametrize(
    "package",
    [
        pytest.param("cupy", marks=[pytest.mark.skipif(not cupy_installed(), reason="cupy not found")]),
        pytest.param("torch", marks=[pytest.mark.skipif(not torch_installed(), reason="torch not found")]),
    ],
)
def test_allocate_symmetric(package, nvmath_distributed, check_symmetric_memory_leaks):
    if package == "torch":
        import torch as package
        from nvmath.distributed._internal.tensor_ifc_torch import TorchDistributedTensor as Tensor
    elif package == "cupy":
        import cupy as package
        from nvmath.distributed._internal.tensor_ifc_cupy import CupyDistributedTensor as Tensor
    dtype = package.int32

    device_id = nvmath.distributed.get_context().device_id
    with device_ctx(device_id):
        if package.__name__ == "torch":
            expected = package.full(SHAPE, VALUE, dtype=dtype, device=f"cuda:{device_id}")
        else:
            expected = package.full(SHAPE, VALUE, dtype=dtype)

        tensor_sheap = nvmath.distributed.allocate_symmetric_memory(SHAPE, package, dtype=dtype)
        assert Tensor(tensor_sheap).is_symmetric_memory
        tensor_sheap[:] = VALUE
        assert is_close(Tensor(tensor_sheap), Tensor(expected))

    mype = nvshmem.my_pe()
    assert nvshmem.ptr(Tensor(expected).data_ptr, pe=mype) == 0
    assert nvshmem.ptr(Tensor(tensor_sheap).data_ptr, pe=mype) != 0

    nvmath.distributed.free_symmetric_memory(tensor_sheap)


@pytest.mark.parametrize(
    "package",
    [
        pytest.param("cupy", marks=[pytest.mark.skipif(not cupy_installed(), reason="cupy not found")]),
        pytest.param("torch", marks=[pytest.mark.skipif(not torch_installed(), reason="torch not found")]),
    ],
)
@pytest.mark.parametrize("device_id", ["cpu", 0])
def test_allocate_non_symmetric(package, device_id, nvmath_distributed, check_symmetric_memory_leaks):
    if package == "torch":
        from nvmath.distributed._internal.tensor_ifc_torch import TorchDistributedTensor as Tensor
    elif package == "cupy":
        from nvmath.distributed._internal.tensor_ifc_cupy import CupyDistributedTensor as Tensor

        if device_id == "cpu":
            pytest.skip("cupy allocation not possible on host memory")

    stream = None
    if device_id != "cpu":
        stream = get_or_create_stream(device_id, stream=None, op_package=package)
    tensor = Tensor.empty(
        SHAPE,
        dtype="int32",
        device_id=device_id,
        stream_holder=stream,
        symmetric_memory=False,
    )
    assert tensor.device_id == device_id
    assert not tensor.is_symmetric_memory
    with pytest.raises(TypeError, match=re.escape("tensor is not on symmetric memory")):
        tensor.free_symmetric()
    assert nvshmem.ptr(tensor.data_ptr, nvshmem.my_pe()) == 0


@pytest.mark.parametrize(
    "package",
    [
        pytest.param("numpy"),
        pytest.param("cupy", marks=[pytest.mark.skipif(not cupy_installed(), reason="cupy not found")]),
        pytest.param("torch", marks=[pytest.mark.skipif(not torch_installed(), reason="torch not found")]),
    ],
)
@pytest.mark.parametrize("symmetric_memory", [False, True])
def test_tensor_to(package, symmetric_memory, nvmath_distributed, check_symmetric_memory_leaks):
    if package == "torch":
        from nvmath.distributed._internal.tensor_ifc_torch import (
            TorchDistributedTensor as CudaTensor,
        )

        HostTensor = CudaTensor
    elif package == "cupy":
        from nvmath.distributed._internal.tensor_ifc_cupy import (
            CupyDistributedTensor as CudaTensor,
            HostDistributedTensor as HostTensor,
        )
    elif package == "numpy":
        from nvmath.distributed._internal.tensor_ifc_numpy import (
            CudaDistributedTensor as CudaTensor,
            NumpyDistributedTensor as HostTensor,
        )

    tensor_cpu = HostTensor.empty(SHAPE, dtype="int64", device_id="cpu")
    assert isinstance(tensor_cpu, DistributedTensor)

    ctx = nvmath.distributed.get_context()
    device_id = ctx.device_id
    stream = get_or_create_stream(device_id, stream=None, op_package="cuda" if package == "numpy" else package)
    tensor_device = tensor_cpu.to(device_id, stream_holder=stream, symmetric_memory=symmetric_memory)
    assert isinstance(tensor_device, DistributedTensor)

    assert tensor_device.is_symmetric_memory == symmetric_memory
    if symmetric_memory:
        assert nvshmem.ptr(tensor_device.data_ptr, nvshmem.my_pe()) != 0
        with device_ctx(device_id):
            tensor_device.free_symmetric()
    else:
        assert nvshmem.ptr(tensor_device.data_ptr, nvshmem.my_pe()) == 0

    tensor_cpu_again = tensor_device.to("cpu", stream_holder=stream)
    assert isinstance(tensor_cpu_again, DistributedTensor)
    assert not tensor_cpu_again.is_symmetric_memory
    assert tensor_cpu.data_ptr != tensor_cpu_again.data_ptr
    assert is_close(tensor_cpu, tensor_cpu_again, allow_ndbuffer=package == "cupy")


@pytest.mark.skipif(not cupy_installed(), reason="cupy not found")
def test_nvshmem_communication(nvmath_distributed, check_symmetric_memory_leaks):
    import cupy

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
        TypeError,
        match=re.escape("free_symmetric_memory called on CPU array/tensor"),
    ):
        nvmath.distributed.free_symmetric_memory(a)


@pytest.mark.skipif(not cupy_installed(), reason="cupy not found")
def test_cupy_distributed_non_symmetric(nvmath_distributed):
    import cupy

    device_id = nvmath.distributed.get_context().device_id
    with cupy.cuda.Device(device_id):
        a = cupy.ones(SHAPE, dtype=cupy.int32)

    from nvmath.distributed._internal.tensor_ifc_cupy import CupyDistributedTensor

    assert not CupyDistributedTensor(a).is_symmetric_memory


@pytest.mark.skipif(not torch_installed(), reason="torch not found")
def test_torch_distributed_non_symmetric(nvmath_distributed):
    import torch

    device_id = nvmath.distributed.get_context().device_id
    a = torch.ones(SHAPE, dtype=torch.int32, device=f"cuda:{device_id}")

    from nvmath.distributed._internal.tensor_ifc_torch import TorchDistributedTensor

    assert not TorchDistributedTensor(a).is_symmetric_memory


@pytest.mark.skipif(not cupy_installed(), reason="cupy not found")
def test_mem_leak_reporting(nvmath_distributed, symmetric_memory_leak_log_message, caplog):
    import cupy

    a = nvmath.distributed.allocate_symmetric_memory(1, cupy, dtype=cupy.int32)
    # We don't free memory with nvmath.distributed.free_symmetric_memory(), which
    # means it leaks.
    del a
    gc.collect()
    try:
        # Error message must appear in logs.
        assert symmetric_memory_leak_log_message in caplog.text
    finally:
        # Internal resource registry was left in an inconsistent state.
        # Need to clear it to prevent subsequent tests from failing.
        nvmath.distributed._internal.nvshmem._resource_registry.clear()


@pytest.mark.parametrize(
    "tensor_type",
    [
        pytest.param("cuda"),
        pytest.param("cupy", marks=[pytest.mark.skipif(not cupy_installed(), reason="cupy not found")]),
        pytest.param("torch", marks=[pytest.mark.skipif(not torch_installed(), reason="torch not found")]),
    ],
)
def test_mem_leak_reporting_internal(tensor_type, nvmath_distributed, symmetric_memory_leak_log_message, caplog):
    Tensor = nvmath.distributed._internal.tensor_wrapper._TENSOR_TYPES[tensor_type]

    device_id = nvmath.distributed.get_context().device_id
    stream = cuda.core.experimental.Stream.from_handle(0)
    a = Tensor.empty((4,), stream_holder=stream, device_id=device_id, symmetric_memory=True)
    assert a.is_symmetric_memory
    # We don't free memory with a.free_symmetric(), which means it leaks.
    del a
    gc.collect()

    try:
        # Error message must appear in logs.
        assert symmetric_memory_leak_log_message in caplog.text
    finally:
        # Internal resource registry was left in an inconsistent state.
        # Need to clear it to prevent subsequent tests from failing.
        nvmath.distributed._internal.nvshmem._resource_registry.clear()
