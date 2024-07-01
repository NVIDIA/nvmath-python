# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Utilities.
"""

__all__ = ['axis_order_in_memory', 'calculate_strides', 'check_batch_tileable', 'create_handle', 'destroy_handle', 'get_handle', 'pointer_aligned_to']

from nvmath.bindings import cublasLt as cublaslt
from nvmath._internal import utils

HANDLES = dict()

def create_handle(device_id):
    """
    Currently for internal use only.
    """
    with utils.device_ctx(device_id):
        handle = cublaslt.create()

    return handle

def destroy_handle(handle):
    """
    Currently for internal use only.
    """
    cublaslt.destroy(handle)

def get_handle(device_id):
    """
    Retrieve the BLAS library handle for the specified device. If one doesn't exist, create, cache, and return the handle.
    """
    return HANDLES.setdefault(device_id, create_handle(device_id))

def pointer_aligned_to(address):
    """
    Return the number of bytes the address is aligned to.
    """
    return address & ~(address - 1)

def axis_order_in_memory(strides):
    """
    Compute the order in which the axes appear in memory.
    """
    if len(strides) == 0:
        return tuple()

    _, axis_order = zip(*sorted(zip(strides, range(len(strides)))))

    return axis_order

def calculate_strides(shape, axis_order):
    """
    Calculate the strides for the provided shape and axis order.
    """
    strides = [None] * len(shape)

    stride = 1
    for axis in axis_order:
        strides[axis] = stride
        stride *= shape[axis]

    return strides

def _contiguous_layout(sorted_shape, sorted_strides):
    for s in range(1, len(sorted_strides)):
        if sorted_shape[s-1] * sorted_strides[s-1] != sorted_strides[s]:
            return False
    return True

def check_batch_tileable(batch_shape, batch_strides):
    """
    Check if the matrix layout is tileable across the specified batch layout.
    """
    sorted_batch_strides, sorted_batch_shape = zip(*sorted(((batch_strides[a], batch_shape[a]) for a in range(len(batch_shape)))))
    return _contiguous_layout(sorted_batch_shape, sorted_batch_strides)
