# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import nvmath.distributed
from nvmath.internal.tensor_wrapper import wrap_operand
from nvmath.internal.utils import device_ctx
from nvmath.distributed._internal.tensor_wrapper import wrap_operand as dist_wrap_operand

try:
    import torch
except ImportError:
    torch = None


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


def generate_random_complex_data(package, memory_space, shape, dtype, stream, memory_layout="C"):
    """Generate random data of the given shape and dtype, where dtype must be a numpy
    complex dtype.
    Returns instance of data on CPU, and a copy on the specified memory_space ("cpu", "gpu")
    wrapped around distributed TensorHolder.
    """
    data_cpu = (np.random.rand(*shape) + 1j * np.random.rand(*shape)).astype(dtype)
    if memory_layout == "F":
        data_cpu = np.asfortranarray(data_cpu)
    assert np.iscomplexobj(data_cpu)
    if package is torch:
        data_cpu = torch.from_numpy(data_cpu)
    else:
        assert package is np
    data_cpu = dist_wrap_operand(data_cpu)
    if memory_space == "gpu":
        device_id = nvmath.distributed.get_context().device_id
        data_gpu = data_cpu.to(device_id, stream)
        return data_cpu, data_gpu
    else:
        data_cpu_copy = data_cpu.__class__.empty(shape, dtype=data_cpu.dtype, strides=data_cpu.strides)
        data_cpu_copy.copy_(data_cpu, None)
        return data_cpu, data_cpu_copy


def is_close(a, b, rtol=1e-07, atol=0):
    assert a.module is b.module
    if a.shape != b.shape:
        return False
    assert a.device_id == b.device_id
    if a.device != "cpu":
        with device_ctx(a.device_id):
            return a.module.allclose(a.tensor, b.tensor, rtol=rtol, atol=atol)
    else:
        return a.module.allclose(a.tensor, b.tensor, rtol=rtol, atol=atol)


def gather_array(arr, partition_dim, comm, rank):
    """Gather CPU array on rank 0. `partition_dim` is the dimension on which this array
    is partitioned across ranks"""

    assert arr.device == "cpu"

    package = arr.module
    assert package in (np, torch)

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

    if rank == 0:
        global_arr = package.empty(global_shape, dtype=arr.dtype)
        comm.Gather(arr, global_arr)
        if transposed:
            # Undo the transpose.
            global_arr = transpose(global_arr, 1, 0)
        # Note that this is not a distributed tensor any longer.
        return wrap_operand(global_arr)
    else:
        comm.Gather(arr, [])
