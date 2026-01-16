# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Utilities.
"""

__all__ = [
    "axis_order_in_memory",
    "calculate_strides",
    "check_batch_tileable",
    "create_handle",
    "destroy_handle",
    "get_handle",
    "pointer_aligned_to",
]

import atexit
import typing

from nvmath.bindings import cublas
from nvmath.bindings import cublasLt as cublaslt
from nvmath.internal import utils

HANDLES: dict[str, dict[int, int]] = {
    "cublas": {},
    "cublaslt": {},
}


def create_handle(device_id: int, binding="cublaslt") -> int:
    """
    Currently for internal use only.
    """
    with utils.device_ctx(device_id):
        match binding:
            case "cublas":
                handle = cublas.create()
            case "cublaslt" | _:
                handle = cublaslt.create()
    return handle


def destroy_handle(handle: int, binding="cublaslt"):
    """
    Currently for internal use only.
    """
    match binding:
        case "cublas":
            cublas.destroy(handle)
        case "cublaslt" | _:
            cublaslt.destroy(handle)


@atexit.register
def _cleanup_handles():
    """
    Cleanup all cached handles on program exit.
    """
    for binding, device_handles in HANDLES.items():
        for handle in device_handles.values():
            destroy_handle(handle, binding=binding)
        device_handles.clear()


def get_handle(device_id: int, binding="cublaslt") -> int:
    """
    Retrieve the cuBLAS[lt] library handle for the specified device. If one doesn't exist,
    create, cache, and return the handle.

    According to the docs for cublasLtHandle_t, any valid cublasHandle_t can be used in
    place of cublasLtHandle_t with a simple cast, so we use the same handle for both APIs.

    Handles are cached and automatically cleaned up on program exit via the atexit handler.
    We expect to have exactly one handle per device / thread.
    """
    if device_id in HANDLES[binding]:
        handle = HANDLES[binding][device_id]
    else:
        handle = create_handle(device_id, binding=binding)
        HANDLES[binding][device_id] = handle
    return handle


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
        return ()

    _, axis_order = zip(*sorted(zip(strides, range(len(strides)), strict=True)), strict=True)

    return axis_order


def calculate_strides(shape: typing.Sequence[int], axis_order: typing.Sequence[int], min_stride: int = 1):
    """
    Calculate the strides for the provided shape and axis order.
    """
    assert len(axis_order) == len(shape), f"axis_order length ({len(axis_order)}) must equal shape length ({len(shape)})"
    assert len(set(axis_order)) == len(axis_order), f"axis_order must not contain duplicates: {axis_order}"
    assert set(axis_order) == set(range(len(shape))), f"axis_order must be permutation of range({len(shape)}): {axis_order}"

    strides: list[None | int] = [None] * len(shape)

    stride = min_stride
    for axis in axis_order:
        strides[axis] = stride
        stride *= shape[axis]

    return strides


def _contiguous_layout(sorted_shape, sorted_strides):
    return all(sorted_shape[s - 1] * sorted_strides[s - 1] == sorted_strides[s] for s in range(1, len(sorted_strides)))


def check_batch_tileable(batch_shape, batch_strides):
    """
    Check if the matrix layout is tileable across the specified batch layout.
    """
    sorted_batch_strides, sorted_batch_shape = zip(
        *sorted((batch_strides[a], batch_shape[a]) for a in range(len(batch_shape))), strict=True
    )
    return _contiguous_layout(sorted_batch_shape, sorted_batch_strides)
