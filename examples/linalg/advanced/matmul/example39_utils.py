# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Utilities for mxfp8 and nvfp4 quantization with microscaling examples.
"""

import torch


def split_extent(shape: tuple[int, ...], axis: int, block_size: int) -> tuple[int, ...]:
    """
    Split dimension ``axis`` of ``shape`` into ``(num_blocks, block_size)``.
    For example, ``(128, 128)`` with ``axis=-1`` and ``block_size=32``
    becomes ``(128, 4, 32)``.
    """
    if axis < 0:
        axis += len(shape)
    assert shape[axis] % block_size == 0
    return shape[:axis] + (shape[axis] // block_size, block_size) + shape[axis + 1 :]


def count_zeros(x: torch.Tensor) -> int:
    """
    Count how many elements ended up as zero.
    """
    return (x == 0).sum().item()


def count_fp8_maxed_out_values(x: torch.Tensor, fp8_dtype: torch.dtype) -> int:
    """
    Count how many elements ended up as maximal representable fp8 value
    (448/-448 for fp8_e4m3fn, 57344/-57344 for fp8_e5m2).
    """
    fp8_max = torch.finfo(fp8_dtype).max
    x_abs = x.abs()
    return (x_abs == fp8_max).sum().item()


def count_fp4_maxed_out_values(x: torch.Tensor) -> int:
    """
    Count how many elements ended up as maximal representable fp4 value
    (6/-6 for float4_e2m1fn_x2).
    """
    fp4_max = 6
    x_abs = x.abs()
    return (x_abs == fp4_max).sum().item()


def count_fp8_clamped_values(x: torch.Tensor, operand_name: str, when: str = "") -> int:
    """
    Count how many elements ended up maxing out at the fp8's range ends or being zero
    (possibly due to overflow or underflow respectively).
    """
    x_dtype = x.dtype
    assert x_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]
    x = x.to(torch.float32)
    x_maxed_out = count_fp8_maxed_out_values(x, x_dtype)
    x_zeros = count_zeros(x)
    print(f"Number of maxed out values in `{operand_name}` {when}: {x_maxed_out / x.numel():.2%}")
    print(f"Number of zeros in `{operand_name}` {when}: {x_zeros / x.numel():.2%}")
    return x_maxed_out + x_zeros


def count_fp4_clamped_values(x: torch.Tensor, operand_name: str, when: str = "") -> int:
    """
    Count how many elements ended up maxing out at the fp4's range ends or being zero
    (possibly due to overflow or underflow respectively).
    """
    x_maxed_out = count_fp4_maxed_out_values(x)
    x_zeros = count_zeros(x)
    print(f"Number of maxed out values in `{operand_name}` {when}: {x_maxed_out / x.numel():.2%}")
    print(f"Number of zeros in `{operand_name}` {when}: {x_zeros / x.numel():.2%}")
    return x_maxed_out + x_zeros
