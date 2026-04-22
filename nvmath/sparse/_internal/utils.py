# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import sys
import typing

from nvmath.internal.tensor_wrapper import wrap_operand


def synchronize_dense_tensor(wrapped_tensor, *, stream_ptr=None):
    """
    This utility synchronizes the provided wrapped tensor for operations performed
    on it in the stream specified by ``stream_ptr``.

    The returned object is also a wrapped tensor.
    """
    if stream_ptr is None or wrapped_tensor.device == "cpu":
        return wrapped_tensor

    package = sys.modules[wrapped_tensor.name]
    capsule = wrapped_tensor.tensor.__dlpack__(stream=stream_ptr)
    tensor = package.from_dlpack(capsule)

    wrapped_tensor = wrap_operand(tensor)
    return wrapped_tensor


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
