# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import math
import numpy as np

import nvmath.distributed
from nvmath.internal.tensor_wrapper import wrap_operand
from nvmath.internal.utils import device_ctx
from nvmath.distributed._internal.tensor_wrapper import wrap_operand as dist_wrap_operand, _TENSOR_TYPES as _DIST_TENSOR_TYPES
from nvmath.distributed._internal.tensor_ifc import DistributedTensor
from nvmath.internal.tensor_ifc_ndbuffer import NDBufferTensor


def to_gpu(data_cpu, device_id, stream):
    """
    Move host tensor to GPU. For numpy tensor, we explicitly
    use cupy as a counterpart.
    """
    match data_cpu.name:
        case "numpy":
            return numpy2cupy(data_cpu, device_id, stream)
        case "torch":
            return data_cpu.to(device_id, stream, symmetric_memory=True)
        case _:
            raise AssertionError(f"Unsupported tensor type: {data_cpu.name}")


def numpy2cupy(data_cpu, device_id, stream):
    """
    Convert numpy tensor to cupy tensor. While we use cupy wrapper
    to allocate the nvshmem-based tensor, we use cupy to copy the
    data to the GPU (and not ndbuffer) to limit usage of
    internal utils that we tests in test data preparation.
    """
    cupy_wrapper = _DIST_TENSOR_TYPES["cupy"]
    import cupy as cp

    assert stream.package == "cupy", f"stream.package: {stream.package}"
    with cp.cuda.Device(device_id):
        tensor_device = cupy_wrapper.empty(
            data_cpu.shape,
            dtype=data_cpu.dtype,
            device_id=device_id,
            strides=data_cpu.strides,
            make_symmetric=True,
            symmetric_memory=True,
            stream_holder=stream,
        )
        with stream.ctx:
            tensor_device.tensor.set(data_cpu.tensor, stream=stream.external)
        stream.external.synchronize()
    return tensor_device


def to_host(data_gpu, device_id, stream):
    match data_gpu.name:
        case "cupy":
            return cupy2numpy(data_gpu, device_id, stream)
        case "torch":
            return data_gpu.to("cpu", stream)
        case _:
            raise AssertionError(f"Unsupported tensor type: {data_gpu.name}")


def cupy2numpy(data_gpu, device_id, stream):
    """
    Convert cupy tensor to numpy tensor. We explicitly use
    numpy/cupy to limit usage of internal utils that we test
    in test data preparation.
    """
    numpy_wrapper = _DIST_TENSOR_TYPES["numpy"]
    numpy_tensor = numpy_wrapper.empty(data_gpu.shape, dtype=data_gpu.dtype, strides=data_gpu.strides)
    import cupy as cp

    assert stream.package == "cupy", f"stream.package: {stream.package}"
    with cp.cuda.Device(device_id):
        with stream.ctx:
            data_gpu.tensor.get(stream=stream.external, out=numpy_tensor.tensor)
        stream.external.synchronize()
    return numpy_tensor


def ndbuffer_as_array(ndbuffer):
    if ndbuffer.device_id == "cpu":
        import ctypes

        buffer = (ctypes.c_char * ndbuffer.size_in_bytes).from_address(ndbuffer.data_ptr)
        return np.ndarray(
            shape=ndbuffer.shape,
            strides=ndbuffer.strides_in_bytes,
            dtype=ndbuffer.dtype_name,
            buffer=buffer,
        )
    else:
        import cupy as cp

        mem = cp.cuda.UnownedMemory(
            ndbuffer.data_ptr,
            ndbuffer.size_in_bytes,
            owner=ndbuffer.data,
            device_id=ndbuffer.device_id,
        )
        memptr = cp.cuda.MemoryPointer(mem, offset=0)
        return cp.ndarray(
            shape=ndbuffer.shape,
            strides=ndbuffer.strides_in_bytes,
            dtype=ndbuffer.dtype_name,
            memptr=memptr,
        )


def calculate_strides(shape, axis_order):
    """
    Calculate the strides for the provided shape and axis order.
    """
    strides = [0] * len(shape)

    stride = 1
    for axis in axis_order:
        strides[axis] = stride
        stride *= shape[axis]

    return strides


def generate_random_data(package, memory_space, shape, dtype, stream, memory_layout="C"):
    """Generate random data of the given shape and dtype.
    Returns instance of data on CPU, and a copy on the specified memory_space ("cpu", "gpu")
    wrapped around distributed TensorHolder.
    """
    if np.issubdtype(dtype, np.complexfloating):
        data_cpu = (np.random.rand(*shape) + 1j * np.random.rand(*shape)).astype(dtype)
    else:
        data_cpu = np.random.rand(*shape).astype(dtype)

    if memory_layout == "F":
        data_cpu = np.asfortranarray(data_cpu)

    if package.__name__ == "torch":
        data_cpu = package.from_numpy(data_cpu)
    else:
        assert package is np

    data_cpu = dist_wrap_operand(data_cpu)
    assert isinstance(data_cpu, DistributedTensor)
    if memory_space == "gpu":
        device_id = nvmath.distributed.get_context().device_id
        data_gpu = to_gpu(data_cpu, device_id, stream)
        assert isinstance(data_gpu, DistributedTensor)
        return data_cpu, data_gpu
    else:
        data_cpu_copy = data_cpu.__class__.empty(shape, dtype=data_cpu.dtype, strides=data_cpu.strides)
        data_cpu_copy.copy_(data_cpu, None)
        assert isinstance(data_cpu_copy, DistributedTensor)
        return data_cpu, data_cpu_copy


def is_close(a, b, rtol=1e-07, atol=0, allow_ndbuffer=False):
    # in principle, ndbuffer is internal opaque strided memory representation
    # that should never be returned to the user, the flag allow_ndbuffer is used
    # here to make sure that the test is expected to compare internal operands
    # and not user facing return values.
    assert a.module is b.module
    if a.shape != b.shape:
        return False
    assert a.device_id == b.device_id
    device_id = a.device_id
    module = a.module
    a_tensor = a.tensor
    b_tensor = b.tensor
    if allow_ndbuffer and isinstance(a, NDBufferTensor):
        a_tensor = ndbuffer_as_array(a_tensor)
        b_tensor = ndbuffer_as_array(b_tensor)
        if device_id == "cpu":
            module = np
        else:
            import cupy as cp

            module = cp
    if device_id != "cpu":
        with device_ctx(device_id):
            return module.allclose(a_tensor, b_tensor, rtol=rtol, atol=atol)
    else:
        return module.allclose(a_tensor, b_tensor, rtol=rtol, atol=atol)


def gather_array(arr, partition_dim, comm, rank):
    """Gather CPU array on rank 0. `partition_dim` is the dimension on which this array
    is partitioned across ranks"""

    assert isinstance(arr, DistributedTensor)
    assert arr.device == "cpu"
    package = arr.module
    assert package.__name__ in ("numpy", "torch"), f"package: {package}"

    if package.__name__ == "torch":
        import torch
    else:
        torch = None

    arr = arr.tensor

    # Convert to C-contiguous for gather
    if package is np and not arr.flags["C_CONTIGUOUS"]:
        arr = arr.copy()
    elif package is torch and not arr.is_contiguous():
        arr = arr.contiguous()

    def transpose(a, dim0, dim1, make_contiguous=False):
        if package is np:
            t = np.moveaxis(a, dim0, dim1)
            if make_contiguous:
                return t.copy()
            return t
        elif package is torch:
            t = torch.transpose(a, dim0, dim1)
            if make_contiguous:
                return t.contiguous()
            return t

    transposed = False
    if partition_dim == 1:
        # Transpose for gather and make contiguous for transport.
        arr = transpose(arr, 1, 0, make_contiguous=True)
        transposed = True

    from mpi4py import MPI

    # Note that after transposing, the partition dim is 0.
    partitioned_extent = comm.allreduce(arr.shape[0], MPI.SUM)
    global_shape = (partitioned_extent,) + arr.shape[1:]

    recv_counts = comm.gather(math.prod(arr.shape))
    if rank == 0:
        global_arr = package.empty(global_shape, dtype=arr.dtype)
        comm.Gatherv(sendbuf=arr, recvbuf=(global_arr, recv_counts), root=0)
        if transposed:
            # Undo the transpose.
            global_arr = transpose(global_arr, 1, 0, make_contiguous=True)
        # Note that this is not a distributed tensor any longer.
        return wrap_operand(global_arr)
    else:
        comm.Gatherv(arr, None)
