# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates how to prepare NVFP4 inputs
from fp32 tensors, including quantization with microscaling.
"""

import torch
from example39_utils import count_fp4_clamped_values, split_extent

import nvmath
from nvmath.linalg.advanced.helpers.matmul import (
    expand_block_scale,
    quantize_to_fp4,
    to_block_scale,
    unpack_fp4,
)


def microscaling_quantization(x: torch.Tensor, axis: int, operand_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert an fp32 tensor to NVFP4 format: a packed fp4 tensor
    and fp8 block scales (one scale per 16 elements along ``axis``).

    * First, find quantization scales for each 16-element block, following
      https://docs.nvidia.com/cuda/cublas/#d-block-quantization
        * Group ``x`` into 16 element blocks along ``axis``.
        * Compute block scales from the maximum absolute value of each 16-element block
          and scale ``x`` accordingly.
    * Then, convert the scaled x and the block scales to required formats:
        * Use the ``quantize_to_fp4`` helper to convert the scaled ``x``
          to fp4 and pack every two elements into one byte.
        * Use ``to_block_scale`` to copy the block scales to the cuBLAS-compatible
          interleaved layout.
    """
    print(f"Quantize and microscale `{operand_name}` with shape {x.shape} along axis {axis}")
    assert x.dtype == torch.float32
    if axis >= 0:
        axis = axis - len(x.shape)
    block_size = 16
    fp4_max = 6.0
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    x_shape = x.shape
    # Group elements along the axis into blocks of size block_size
    x = x.reshape(split_extent(x_shape, axis, block_size))
    print(f"\tReshaped {operand_name} to group elements into blocks of size {block_size}: {x_shape} -> {x.shape}")
    # Block scale = (block max absolute value) / max fp4 representable value
    x_scale = x.abs().max(dim=axis).values / fp4_max
    # Scales are stored as float8_e4m3fn; clamp to fp8 max for consistent
    # overflow behavior in PyTorch.
    x_scale = x_scale.clamp(max=fp8_max).to(torch.float8_e4m3fn)
    # Reorder fp8_e4m3fn scales in memory for cuBLAS.
    quantization_scale = to_block_scale(
        x_scale,
        # Layout depends on the original shape and grouping axis.
        x_shape,
        axis=axis,
        block_scaling_format="NVFP4",
    )
    print(
        f"\tConverted block scales for {operand_name} to flat interleaved layout: {x_scale.shape} -> {quantization_scale.shape}"
    )
    # Quantization scales are ready; inversely scale x.
    x_scale = x_scale.to(torch.float32)
    # If we have an all-zeros block, the corresponding
    # scale doesn't matter, but we need to avoid division by zero.
    x_scale[x_scale == 0] = 1
    scale_reciprocal = torch.reciprocal(x_scale)  # 1 / x_scale
    # Add singleton dim to broadcast the scales to every 16 elements.
    scale_reciprocal = scale_reciprocal.unsqueeze(axis)
    scaled_x = x * scale_reciprocal
    scaled_x = scaled_x.reshape(x_shape)
    scaled_x = quantize_to_fp4(scaled_x, axis=axis)
    print(f"\tConverted scaled {operand_name} to fp4 packed format: {x.shape} -> {scaled_x.shape}")
    return scaled_x, quantization_scale


m, n, k = 128, 256, 64

# For illustration purposes, let's prepare fp32 tensors with
# similarly distributed values across columns (a) and rows (b).
a = torch.randn(m, k, device="cuda", dtype=torch.float32).abs()
b = torch.randn(n, k, device="cuda", dtype=torch.float32).abs().T
# Shift the distribution column-wise (a) and row-wise (b).
# This way, most of the elements will end up outside of the
# range of values representable in fp4, but locally they will remain
# similarly distributed.
a *= (torch.rand((m, 1), device="cuda", dtype=torch.float32) + 1) * 10
b *= (torch.rand((1, n), device="cuda", dtype=torch.float32) + 1) * 0.1
# Negate some rows (a) and columns (b) at random.
a *= torch.where(torch.randint(0, 2, (m, 1), device="cuda", dtype=torch.bool), 1, -1)
b *= torch.where(torch.randint(0, 2, (1, n), device="cuda", dtype=torch.bool), 1, -1)

print(f"a.min(), a.max(): {a.min()}, {a.max()}")
print(f"b.min(), b.max(): {b.min()}, {b.max()}")


# If we convert those tensors to fp4 without microscaling,
# most of the values in ``a`` will end up maxing out at 6/-6 or,
# for ``b``, being clamped to 0.
a_fp4 = quantize_to_fp4(a, axis=-1)
b_fp4 = quantize_to_fp4(b, axis=-2)
count_fp4_clamped_values(unpack_fp4(a_fp4, axis=-1), "a_fp4", "without microscaling")
count_fp4_clamped_values(unpack_fp4(b_fp4, axis=-2), "b_fp4", "without microscaling")

# So, let's try quantization with microscaling.
a_fp4, scale_a = microscaling_quantization(a, axis=-1, operand_name="a")
b_fp4, scale_b = microscaling_quantization(b, axis=-2, operand_name="b")

# With microscaling, fewer values sit at the extrema. Before, most ±6
# entries in a and zeros in b came from overflow or underflow; scaling
# makes better use of the narrow fp4 range.
# Note: each non-zero block still has at least one 6/-6 value,
# that corresponds to the block's amax value after scaling.
count_fp4_clamped_values(unpack_fp4(a_fp4, axis=-1), "a_fp4", "with microscaling")
count_fp4_clamped_values(unpack_fp4(b_fp4, axis=-2), "b_fp4", "with microscaling")

# Now, we can just use the a_fp4, b_fp4 and the quantization scales
# to run narrow-precision matmul.
result, aux = nvmath.linalg.advanced.matmul(
    a_fp4, b_fp4, quantization_scales={"a": scale_a, "b": scale_b}, options={"block_scaling": True}
)
d_out_scale = aux["d_out_scale"]


# Unpack the result and compare against a fp32 reference.
# The result is packed along the last axis (same as ``a``).
# Note, the result packing axis could differ, if we specified c operand or epilogue.
result_unpacked = unpack_fp4(result, axis=-1)
scale_expanded = expand_block_scale(d_out_scale, result, output_dtype=torch.float32, block_scaling_format="NVFP4")
result_unpacked *= scale_expanded

ref = a @ b
rel = torch.abs(ref - result_unpacked) / (torch.abs(ref) + 1e-6)
rms = torch.abs(ref - result_unpacked).norm() / ref.norm()
print(f"Matmul max relative error: {rel.max()}")
print(f"Mean square rel error: {rms}")


# Finally, let's compare our microscaling with how cuBLAS
# has microscaled the matmul's result.
# We received nvfp4 output as (result : fp4_e2m1fn_x2, d_out_scale : fp8_e4m3fn)
# and converted it to result_unpacked : fp32.
# Let's quantize the unpacked result back to nvfp4:
result_2, d_out_scale_2 = microscaling_quantization(result_unpacked, axis=-1, operand_name="result")
assert result_2.dtype == result.dtype == torch.float4_e2m1fn_x2
assert d_out_scale_2.dtype == d_out_scale.dtype == torch.float8_e4m3fn

# PyTorch doesn't support == on float4_e2m1fn_x2, so compare raw bytes.
assert torch.all(result.view(torch.uint8) == result_2.view(torch.uint8))
assert torch.all(d_out_scale == d_out_scale_2)
