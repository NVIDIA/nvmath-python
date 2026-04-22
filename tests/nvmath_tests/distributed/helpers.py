# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np

import nvmath.distributed
from nvmath.distributed._internal.tensor_ifc import DistributedTensor
from nvmath.distributed._internal.tensor_wrapper import _TENSOR_TYPES as _DIST_TENSOR_TYPES
from nvmath.distributed._internal.tensor_wrapper import wrap_operand as dist_wrap_operand
from nvmath.distributed.process_group import MPIProcessGroup, ReductionOp, TorchProcessGroup
from nvmath.internal.tensor_ifc import TensorHolder
from nvmath.internal.tensor_ifc_ndbuffer import NDBufferTensor
from nvmath.internal.tensor_wrapper import wrap_operand
from nvmath.internal.typemaps import NAME_TO_DATA_WIDTH
from nvmath.internal.utils import device_ctx


def to_gpu(data_cpu, device_id, stream, symmetric_memory):
    """
    Move host tensor to GPU. For numpy tensor, we explicitly
    use cupy as a counterpart.
    """
    match data_cpu.name:
        case "numpy":
            return numpy2cupy(data_cpu, device_id, stream, symmetric_memory)
        case "torch":
            return data_cpu.to(device_id, stream, symmetric_memory=symmetric_memory)
        case _:
            raise AssertionError(f"Unsupported tensor type: {data_cpu.name}")


def numpy2cupy(data_cpu, device_id, stream, symmetric_memory):
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
            make_symmetric=symmetric_memory,
            symmetric_memory=symmetric_memory,
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


def generate_random_data(package, memory_space, shape, dtype, stream, memory_layout="C", symmetric_memory=True):
    """Generate random data of the given shape and dtype.
    Returns instance of data on CPU, and a copy on the specified memory_space ("cpu", "gpu")
    wrapped around distributed TensorHolder.

    Args:
        package: numpy or torch. For numpy package with memory_space="gpu", uses cupy.

        memory_space: "cpu" or "gpu"

        dtype: numpy dtype
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
        data_gpu = to_gpu(data_cpu, device_id, stream, symmetric_memory)
        assert isinstance(data_gpu, DistributedTensor)
        assert data_gpu.is_symmetric_memory == symmetric_memory
        return data_cpu, data_gpu
    else:
        data_cpu_copy = data_cpu.__class__.empty(shape, dtype=data_cpu.dtype, strides=data_cpu.strides)
        data_cpu_copy.copy_(data_cpu, None)
        assert isinstance(data_cpu_copy, DistributedTensor)
        return data_cpu, data_cpu_copy


def assert_close(actual, expected, rtol=1e-07, atol=0, allow_ndbuffer=False):
    # in principle, ndbuffer is internal opaque strided memory representation
    # that should never be returned to the user, the flag allow_ndbuffer is used
    # here to make sure that the test is expected to compare internal operands
    # and not user-facing return values.

    assert isinstance(actual, TensorHolder), "assert_close requires wrapped (TensorHolder) operands"
    assert isinstance(expected, TensorHolder), "assert_close requires wrapped (TensorHolder) operands"

    assert actual.module is expected.module, f"module of actual ({actual.module}) and expected ({expected.module}) differ"
    assert actual.shape == expected.shape, f"shape of actual ({actual.shape}) and expected ({expected.shape}) differ"
    assert actual.device_id == expected.device_id, (
        f"device_id of actual ({actual.device_id}) and expected ({expected.device_id}) differ"
    )
    device_id = actual.device_id
    module = actual.module
    actual_tensor = actual.tensor
    expected_tensor = expected.tensor

    if allow_ndbuffer and isinstance(actual, NDBufferTensor):
        actual_tensor = ndbuffer_as_array(actual_tensor)
        expected_tensor = ndbuffer_as_array(expected_tensor)
        if device_id == "cpu":
            module = np
        else:
            import cupy as module

    # Convert FP8 to FP32.
    if NAME_TO_DATA_WIDTH[actual.dtype] == 8 and "float" in actual.dtype:
        actual_tensor = actual_tensor.to(module.float32)
    if NAME_TO_DATA_WIDTH[expected.dtype] == 8 and "float" in expected.dtype:
        expected_tensor = expected_tensor.to(module.float32)

    # Call tensor package assert close function.
    test_func = module.testing.assert_close if module.__name__ == "torch" else module.testing.assert_allclose
    if device_id != "cpu":
        with device_ctx(device_id):
            test_func(actual_tensor, expected_tensor, rtol=rtol, atol=atol)
    else:
        test_func(actual_tensor, expected_tensor, rtol=rtol, atol=atol)


def process_group_broadcast(process_group, obj, root=0):
    if isinstance(process_group, MPIProcessGroup):
        return process_group._mpi_comm.bcast(obj, root=root)
    else:
        import torch
        import torch.distributed as dist

        result = [obj] if process_group.rank == root else [None]
        if process_group.device_id != "cpu":
            with torch.cuda.device(process_group.device_id):
                dist.broadcast_object_list(result, group=process_group._torch_process_group, group_src=root)
        else:
            dist.broadcast_object_list(result, group=process_group._torch_process_group, group_src=root)
        return result[0]


def gather_array(arr, partition_dim, process_group, rank):
    """Gather CPU array on rank 0. `partition_dim` is the dimension on which this array
    is partitioned across ranks"""

    assert isinstance(arr, DistributedTensor)
    assert arr.device == "cpu"
    dtype_name = arr.dtype
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

    # Note that after transposing, the partition dim is 0.
    partitioned_extent = np.array([arr.shape[0]])
    process_group.allreduce_buffer(partitioned_extent, op=ReductionOp.SUM)
    global_shape = (int(partitioned_extent[0]),) + arr.shape[1:]

    if isinstance(process_group, MPIProcessGroup):
        comm = process_group._mpi_comm
        recv_counts = comm.gather(math.prod(arr.shape))
        if rank == 0:
            global_arr = package.empty(global_shape, dtype=arr.dtype)

            sendbuf = arr
            recvbuf = (global_arr, recv_counts)
            if NAME_TO_DATA_WIDTH[dtype_name] <= 16:
                # WAR for MPI not having narrow-precision types.
                sendbuf = arr.view(dtype=package.int8)
                recv_counts = [x * (NAME_TO_DATA_WIDTH[dtype_name] // 8) for x in recv_counts]
                recvbuf = (global_arr.view(dtype=package.int8), recv_counts)

            comm.Gatherv(sendbuf=sendbuf, recvbuf=recvbuf, root=0)
            if transposed:
                # Undo the transpose.
                global_arr = transpose(global_arr, 1, 0, make_contiguous=True)
            # Note that this is not a distributed tensor any longer.
            return wrap_operand(global_arr)
        else:
            comm.Gatherv(arr if NAME_TO_DATA_WIDTH[dtype_name] > 16 else arr.view(dtype=package.int8), None)
    else:
        assert isinstance(process_group, TorchProcessGroup)

        import torch
        import torch.distributed as dist

        object_gather_list = None if rank != 0 else [None] * process_group.nranks

        def do_gather():
            dist.gather_object(
                arr.view(dtype=package.int8),
                object_gather_list,
                group=process_group._torch_process_group,
                group_dst=0,
            )

        if process_group.device_id != "cpu":
            with torch.cuda.device(process_group.device_id):
                do_gather()
        else:
            do_gather()

        if rank == 0:
            if package.__name__ == "torch":
                global_arr = torch.cat(object_gather_list)
            else:
                global_arr = np.concatenate(object_gather_list)
            global_arr = global_arr.view(dtype=arr.dtype).reshape(global_shape)
            if transposed:
                # Undo the transpose.
                global_arr = transpose(global_arr, 1, 0, make_contiguous=True)
            # Note that this is not a distributed tensor any longer.
            return wrap_operand(global_arr)
