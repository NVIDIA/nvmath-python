# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
A collection of (internal use) helper functions for shape/stride
validation and manipulation.
"""

from collections.abc import Sequence


def is_contiguous_layout(sorted_shape: Sequence[int], sorted_strides: Sequence[int]) -> bool:
    return all(sorted_shape[s - 1] * sorted_strides[s - 1] == sorted_strides[s] for s in range(1, len(sorted_strides)))


def is_contiguous_in_memory(shape: Sequence[int], strides: Sequence[int]) -> bool:
    """
    Check if the provided (shape, strides) result in a contiguous memory layout.
    """
    sorted_strides, sorted_shape = zip(*sorted(zip(strides, shape, strict=True)), strict=True)
    return is_contiguous_layout(sorted_shape, sorted_strides)


def is_contiguous_and_dense(shape: Sequence[int], strides: Sequence[int]) -> bool:
    """
    Check if the provided (shape, strides) result in a contiguous memory layout
    with no extra stride in least strided dimension.
    """
    sorted_strides, sorted_shape = zip(*sorted(zip(strides, shape, strict=True)), strict=True)
    if len(sorted_strides) > 0 and sorted_strides[0] != 1:
        return False
    return is_contiguous_layout(sorted_shape, sorted_strides)


def is_overlapping_layout(shape: Sequence[int], strides: Sequence[int]) -> bool:
    """
    For a tensor `t`, if `not is_overlapping_layout(t.shape, t.strides)`,
    there are no two different valid nd-indices `idxs` such that
    `t[idxs_0]` and `t[idxs_1]` map to the same offset in the memory.
    Checks that the strides:
        1. are positive
        2. any n - 1 extents maximal offset is smaller than the stride
           of the n-th extent.
    The check should return False for contiguous
    or contiguous and sliced tensors.
    """
    sorted_strides, sorted_shape = zip(*sorted(zip(strides, shape, strict=True)), strict=True)
    cur_max_offset = 0
    for s in range(1, len(sorted_strides)):
        stride = sorted_strides[s - 1]
        extent = sorted_shape[s - 1]
        if stride <= 0:
            return True
        cur_max_offset += stride * (extent - 1)
        if cur_max_offset >= sorted_strides[s]:
            return True
    return False
