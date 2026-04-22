# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates automatic output scaling in NVFP4 mode.
When using NVFP4 with an FP4 result type, the result of the matmul
is automatically scaled during the operation and the scale
used is returned as "d_out_scale". This scale can then be applied to
the result to recover the dequantized values.

To this end, we set up a contrived example where only a single output
element is non-zero, making the expected value analytically
calculable (similar to example34_nvfp4):

    - A: 128x64 matrix with 16 ones in row 5, K-group 2 (cols 32-47), scale = 1.0
    - B: 64x128 matrix with 16 ones in col 32, K-group 2 (rows 32-47), scale = 4.0
    - True result (before quantization): C[5, 32] = 16 * 1.0 * 4.0 = 64.0

We run the matmul operation with the following options
to enable automatic output scaling:
    - ``block_scaling`` option to ``True``
    - ``result_type`` to ``nvmath.CudaDataType.CUDA_R_4F_E2M1``

Since the max representable FP4 value is 6.0, the value 64.0 that we expect
cannot be stored directly in the output tensor. cuBLAS automatically picks
a per-block ``d_out_scale`` so that ``fp4_value * d_out_scale``
approximates the true result.
This example demonstrates the concept as well as how to use the
utilities provided in ``nvmath.linalg.advanced.helpers.matmul``
to dequantize the FP4 result using ``unpack_fp4``
and ``expand_block_scale``.

NVFP4 requires cuBLAS 12.8+, torch >= 2.9, and compute capability >= 10.0.
"""

import torch

import nvmath
from nvmath.linalg.advanced.helpers.matmul import (
    BlockScalingFormat,
    expand_block_scale,
    get_block_scale_offset,
    quantize_to_fp4,
    unpack_fp4,
)

# M and N must be multiples of 128, K must be a multiple of 64
m, k, n = 128, 64, 128
device = "cuda"
scaling_format = BlockScalingFormat.NVFP4

# A matrix
a_float = torch.zeros(m, k, device=device, dtype=torch.float32)
a_float[5, 32:48] = 1.0  # row 5, K-group 2 (cols 32-47)
a_fp4 = quantize_to_fp4(a_float, axis=-1)

# B matrix
b_float = torch.zeros(k, n, device=device, dtype=torch.float32)
b_float[32:48, 32] = 1.0  # col 32, K-group 2 (rows 32-47)
b_fp4 = quantize_to_fp4(b_float, axis=-2)

# Block scales
# cuBLASLt stores scales in a non-trivial tiled layout.
# Use get_block_scale_offset to compute the correct index.
a_scale = torch.zeros(m * (k // 16), dtype=torch.float8_e4m3fn, device=device)
# note, the same scale is applied to elements from (5, 32) to (5, 47)
# 32..47 // 16 = 2
a_scale[get_block_scale_offset((5, 2), a_fp4, scaling_format)] = 1.0
b_scale = torch.zeros(n * (k // 16), dtype=torch.float8_e4m3fn, device=device)
# note, the same scale is applied to elements from (32, 32) to (47, 32)
# 32..47 // 2 = 2
b_scale[get_block_scale_offset((2, 32), b_fp4, scaling_format)] = 4.0

# Setting result_type to CUDA_R_4F_E2M1 produces FP4-quantized output.
# The return value is a tuple (result, aux) where aux["d_out_scale"] contains
# the scales used for output quantization.
result, aux = nvmath.linalg.advanced.matmul(
    a_fp4,
    b_fp4,
    quantization_scales={"a": a_scale, "b": b_scale},
    options={
        "result_type": nvmath.CudaDataType.CUDA_R_4F_E2M1,
        "block_scaling": True,
    },
)

print(f"Result dtype: {result.dtype} (FP4, packed two values per byte)")
print(f"Result shape: {result.shape} (packed; logical shape is ({m}, {n}))")
print()

# --- Examine the d_out_scale ---
d_out_scale = aux["d_out_scale"]
print(f"d_out_scale shape: {d_out_scale.shape}, dtype: {d_out_scale.dtype}.")
print()

# --- Dequantize the FP4 result ---
# Step 1: Unpack FP4 packed bytes to float32.
result_unpacked = unpack_fp4(result, axis=-1)
print(f"Unpacked result shape: {result_unpacked.shape}, dtype: {result_unpacked.dtype}")

# Step 2: Expand the 1D tiled d_out_scale to a full (M, N) matrix.
# For the output matrix D (M x N): blocked in rows (axis=-1).
# We convert the float8_e4m3fn scales to float16,
# as torch does not support fp32 * fp8_e4m3fn elementwise multiplication directly.
scales_expanded = expand_block_scale(d_out_scale, result, scaling_format, output_dtype=torch.float16)
print(f"Expanded scales shape: {scales_expanded.shape}, dtype: {scales_expanded.dtype}")

# Step 3: Element-wise multiply to get the dequantized result in float32
dequantized = result_unpacked * scales_expanded
print()

# --- Check the result ---
# Note that due to the quantization error, the dequantized value
# may not match the true value exactly.
true_value = 16 * 1.0 * 4.0
fp4_raw = result_unpacked[5, 32].item()
scale_at_pos = scales_expanded[5, 32].item()
actual_value = dequantized[5, 32].item()
print(f"True (unquantized) value  = {true_value:.1f}")
print(f"FP4 raw value at C[5, 32] = {fp4_raw}")
print(f"d_out_scale at C[5, 32]   = {scale_at_pos}")
print(f"Dequantized C[5, 32]      = {fp4_raw} * {scale_at_pos} = {actual_value:.1f}")
