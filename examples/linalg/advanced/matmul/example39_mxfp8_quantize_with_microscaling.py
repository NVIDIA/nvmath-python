# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates how to prepare MXFP8 inputs
from FP32 tensors, including quantization with microscaling.
"""

import torch
from example39_utils import count_fp8_clamped_values, split_extent

import nvmath
from nvmath.linalg.advanced.helpers.matmul import (
    apply_mxfp8_scale,
    invert_mxfp8_scale,
    to_block_scale,
)


def microscaling_quantization(
    x: torch.Tensor, axis: int, fp8_dtype: torch.dtype, operand_name: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a fp32 tensor to MXFP8 format: a fp8 tensor
    and ue8m0 block scales (one scale/exponent per 32 elements along ``axis``).

    * First, find quantization scales for each 32-element block, following
      https://docs.nvidia.com/cuda/cublas/#d-block-quantization
        * Group ``x`` into 32-element blocks along ``axis``.
        * Compute block scales from the maximum absolute value of each 32-element block
          and scale ``x`` accordingly.
        * Convert the scales to ue8m0 format by extracting and rounding the exponent.
    * Then, inversely scale x and convert the block scales to required formats:
        * Use ``invert_mxfp8_scale`` to get reciprocal of the scale.
        * Use ``to_block_scale`` to copy the block scales to the cuBLAS-compatible
          interleaved layout.
    """
    print(f"\tQuantize and microscale `{operand_name}` with shape {x.shape} along axis {axis}")
    assert x.dtype == torch.float32
    if axis >= 0:
        axis = axis - len(x.shape)
    block_size = 32
    x_shape = x.shape
    # Group elements along the axis into blocks of size block_size
    x = x.reshape(split_extent(x_shape, axis, block_size))
    print(f"\tReshaped `{operand_name}` to group elements into blocks of size {block_size}: {x_shape} -> {x.shape}")
    x_amax = x.abs().max(dim=axis).values
    # Block scale = (block max absolute value) / max fp8 representable value
    x_scale = x_amax / torch.finfo(fp8_dtype).max
    x_exponent = to_ue8m0(x_scale)
    quantization_scale = to_block_scale(
        x_exponent,
        # Layout depends on the original shape and grouping axis.
        x_shape,
        axis=axis,
        block_scaling_format="MXFP8",
    )
    print(
        f"\tConverted block scales for `{operand_name}` to "
        f"flat interleaved layout: {x_exponent.shape} -> {quantization_scale.shape}"
    )
    # Quantization scales are ready; inversely scale x.
    scale_reciprocal = invert_mxfp8_scale(x_exponent)
    # Convert exponent to fp32 scale.
    scale_reciprocal = 2 ** (scale_reciprocal.to(torch.float32) - 127)
    # Add singleton dimension along the grouping axis to broadcast
    # each scale to the corresponding elements along the grouping axis.
    scale_reciprocal = scale_reciprocal.unsqueeze(axis)
    x_scaled = x * scale_reciprocal
    print(
        f"\tMultiplied `{operand_name}` by inverse quantization scale: {x.shape} x {scale_reciprocal.shape} -> {x_scaled.shape}"
    )
    x_scaled = convert_to_fp8(x_scaled, fp8_dtype)
    print(f"\tReshaped scaled `{operand_name}` back to original shape: {x_scaled.shape} -> {x_shape}")
    x_scaled = x_scaled.reshape(x_shape)
    return x_scaled, quantization_scale


def to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    """
    Convert fp32 scale to UE8M0 format (an 8-bit exponent without any significand).
    Extract the exponent from fp32 bit representation and round it up
    depending on the fixed point part.

    Note, the UE8M0 format is biased by 127, i.e. a number x in UE8M0 format
    corresponds to a number 2^(x - 127), the same bias is used in fp32.
    """
    assert x.dtype == torch.float32
    bit_view = x.view(torch.int32)
    # Extract the mantissa (first 23 bits) that represent the
    # fixed point part of the number.
    mantissa = bit_view & 0x007FFFFF
    # Extract the exponent (next 8 bits) that represent
    # the exponent.
    # Note, the exponent is biased by 127
    exponent = (bit_view & 0x7F800000) >> 23
    is_subnormal = exponent == 0
    # Following recipe from https://docs.nvidia.com/cuda/cublas/#d-block-quantization,
    # we want to round up the exponent:
    # * for normal numbers, the number is
    #   2**(exponent - 127) * 1.mantissa,
    #   we round up if the fixed point part is greater than 1.0,
    #   i.e. there are any bits set in the mantissa,
    #   and the exponent can be safely increased by one
    #   (if it exceeded 254, the number could not be represented in fp32 anymore)
    mask = (~is_subnormal) & (mantissa > 0) & (exponent < 254)
    # * for subnormal numbers
    #   i.e. exponent == 0,
    #   unbiased exponent == -127,
    #   the number is 2**(-126) * 0.mantissa,
    #   we round up if the fixed point part is greater than 0.5.
    mask |= is_subnormal & (mantissa > 0x00400000)
    exponent[mask] += 1
    return exponent.to(torch.uint8)


def convert_to_fp8(x: torch.Tensor, fp8_dtype: torch.dtype) -> torch.Tensor:
    """
    Convert a fp32 tensor to fp8 tensor.
    """
    # Depending on torch version, conversion of large fp32 values
    # can either saturate or end up with nans/infs,
    # let's clamp explicitly to the fp8 range.
    assert x.dtype == torch.float32
    assert fp8_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]
    fp8_min = torch.finfo(fp8_dtype).min
    fp8_max = torch.finfo(fp8_dtype).max
    return x.clamp(min=fp8_min, max=fp8_max).to(fp8_dtype)


m, n, k = 128, 256, 128

a_fp8_dtype = torch.float8_e4m3fn  # torch.float8_e5m2 is accepted too
b_fp8_dtype = torch.float8_e4m3fn

# For illustration purposes, let's prepare fp32 tensors with
# similarly distributed values across columns (a) and rows (b).
a = torch.randn(m, k, device="cuda", dtype=torch.float32).abs()
b = torch.randn(n, k, device="cuda", dtype=torch.float32).abs().T
# Shift the distribution column-wise (a) and row-wise (b).
# This way, some of the elements will end up outside of the
# range of values representable in MXFP8, but locally they will remain
# similarly distributed.
a *= (torch.rand((m, 1), device="cuda", dtype=torch.float32) + 1) * 400
b *= (torch.rand((1, n), device="cuda", dtype=torch.float32) + 1) * 0.0025
# Negate some rows (a) and columns (b) at random.
a *= torch.where(torch.randint(0, 2, (m, 1), device="cuda", dtype=torch.bool), 1, -1)
b *= torch.where(torch.randint(0, 2, (1, n), device="cuda", dtype=torch.bool), 1, -1)

print(f"a.min(), a.max(): {a.min()}, {a.max()}")
print(f"b.min(), b.max(): {b.min()}, {b.max()}")

# Let's convert the fp32 tensors to fp8 format and see how many
# values ended up clamped to 0 or saturating at the
# end of the fp8 range (448/-448 for fp8_e4m3fn,
# 57344/-57344 for fp8_e5m2).
a_fp8 = convert_to_fp8(a, a_fp8_dtype)
b_fp8 = convert_to_fp8(b, b_fp8_dtype)
count_fp8_clamped_values(a_fp8, "a_fp8", "with no scaling")
count_fp8_clamped_values(b_fp8, "b_fp8", "with no scaling")

# We could try to scale the whole tensors to avoid overflows:
dtype_max = min(torch.finfo(a_fp8_dtype).max, torch.finfo(b_fp8_dtype).max)
# let's halve the max value to map max value to half of the range where
# the representable numbers are more densely distributed.
dtype_max /= 2
glob_max = max(a.abs().max(), b.abs().max())
global_scale = glob_max / dtype_max
print(f"global_scale: {global_scale} = max(a.abs().max(), b.abs().max()) / (dtype_max / 2) = {glob_max} / {dtype_max}")
a_scaled = a / global_scale
b_scaled = b / global_scale
print(f"a.min(), a.max(): {a_scaled.min()}, {a_scaled.max()}")
print(f"b.min(), b.max(): {b_scaled.min()}, {b_scaled.max()}")
a_fp8 = convert_to_fp8(a_scaled, a_fp8_dtype)
b_fp8 = convert_to_fp8(b_scaled, b_fp8_dtype)

# Now, we managed to eliminate overflows completely, but at the cost of
# a huge number of underflows (zeros) in b.
count_fp8_clamped_values(a_fp8, "a_fp8", "with a single scale")
count_fp8_clamped_values(b_fp8, "b_fp8", "with a single scale")

# Instead, let's use microscaling to have multiple local scales.
a_fp8, a_scale = microscaling_quantization(a, axis=-1, fp8_dtype=a_fp8_dtype, operand_name="a")
b_fp8, b_scale = microscaling_quantization(b, axis=-2, fp8_dtype=b_fp8_dtype, operand_name="b")

# Now, the fraction of clamped values should be insignificant.
count_fp8_clamped_values(a_fp8, "a_fp8", "with microscaling")
count_fp8_clamped_values(b_fp8, "b_fp8", "with microscaling")

# Now, we can just use the a_fp8, b_fp8 and the quantization scales
# to run narrow-precision matmul.
result, aux = nvmath.linalg.advanced.matmul(
    a_fp8, b_fp8, quantization_scales={"a": a_scale, "b": b_scale}, options={"block_scaling": True}
)
d_out_scale = aux["d_out_scale"]

# Convert the MXFP8 result to fp32.
result_fp32 = apply_mxfp8_scale(result, d_out_scale).to(torch.float32)

# Compute the reference result.
ref_fp32 = a @ b
rel = torch.abs(ref_fp32 - result_fp32) / (torch.abs(ref_fp32) + 1e-6)
rms = torch.abs(ref_fp32 - result_fp32).norm() / ref_fp32.norm()
print(f"Matmul max relative error: {rel.max()}")
print(f"Mean square rel error: {rms}")

# Finally, let's compare our microscaling with how cuBLAS
# has microscaled the matmul's result.

# Convert the MXFP8 inputs to fp32.
a_dequantized = apply_mxfp8_scale(a_fp8, a_scale).to(torch.float32)
b_dequantized = apply_mxfp8_scale(b_fp8, b_scale).to(torch.float32)
# Compute the result in fp32.
ref = a_dequantized @ b_dequantized
# Quantize the result back to MXFP8.
# Note, in this case (no c operand, no epilogue),
# the result is quantized along the last axis,
# which is the same as the grouping axis for a.
result_2, d_out_scale_2 = microscaling_quantization(ref, axis=-1, fp8_dtype=a_fp8_dtype, operand_name="result")

assert result_2.dtype == result.dtype == a_fp8_dtype
assert d_out_scale_2.dtype == d_out_scale.dtype == torch.uint8
assert torch.all(result.view(torch.uint8) == result_2.view(torch.uint8))
assert torch.all(d_out_scale == d_out_scale_2)
