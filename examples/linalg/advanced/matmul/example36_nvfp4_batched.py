# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates batched NVFP4 matmul on 3D tensors
where the leading axis is a batch dimension. Each batch slice uses the same
"one nonzero block" pattern as example34, making every result analytically
predictable and easy to understand.

Setup (batch_size=2, M=128, K=64, N=128):

    Batch 0  (same as example34):
        - A[0]: 16 ones in row 5, K-group 2 (cols 32-47);  a_scale = 1.0
        - B[0]: 16 ones in col 32, K-group 2 (rows 32-47); b_scale = 4.0
        -> C[0, 5, 32]  = alpha * 16 * 1.0 * 4.0

    Batch 1:
        - A[1]: 16 ones in row 10, K-group 0 (cols 0-15);  a_scale = 2.0
        - B[1]: 16 ones in col 64, K-group 0 (rows 0-15);  b_scale = 3.0
        -> C[1, 10, 64] = alpha * 16 * 2.0 * 3.0

Scale addressing for batched tensors:

    The scale tensor is 1D with shape (batch_size * outer_dim * K // 16,).
    Scales for each batch element are contiguous.

NVFP4 requires cuBLAS 12.8+, torch >= 2.9, and compute capability >= 10.0.
"""

import torch

import nvmath
from nvmath.linalg.advanced.helpers.matmul import (
    expand_block_scale,
    get_block_scale_offset,
    quantize_to_fp4,
    unpack_fp4,
)

batch_size = 2
m, k, n = 128, 64, 128
device = "cuda"

# A matrix: logical shape (batch_size, m, k).
a_float = torch.zeros(batch_size, m, k, device=device, dtype=torch.float32)
a_float[0, 5, 32:48] = 1.0  # batch 0: row 5, K-group 2 (cols 32-47)
a_float[1, 10, 0:16] = 1.0  # batch 1: row 10, K-group 0 (cols 0-15)
# Quantize to FP4 using row-wise packing since for A this is the contracting dimension.
a_fp4 = quantize_to_fp4(a_float, axis=-1)  # (batch_size, m, k//2)

# B matrix: logical shape (batch_size, k, n).
b_float = torch.zeros(batch_size, k, n, device=device, dtype=torch.float32)
b_float[0, 32:48, 32] = 1.0  # batch 0: col 32, K-group 2 (rows 32-47)
b_float[1, 0:16, 64] = 1.0  # batch 1: col 64, K-group 0 (rows 0-15)
# Quantize to FP4 using column-wise packing since for B this is the contracting dimension.
b_fp4 = quantize_to_fp4(b_float, axis=-2)  # (batch_size, k//2, n)

# Build 1D scale tensors covering all batches.
# The full scale tensor concatenates all batch elements contiguously:
# b-th batch scales start at offset: b * per_matrix_scales.

# A scales: A is (M, K) blocked in rows (axis=-1) -> M * K // 16.
a_scales_per_matrix = m * k // 16
a_scale = torch.zeros(batch_size * a_scales_per_matrix, dtype=torch.float8_e4m3fn, device=device)
# For batched tensors, each batch element's scales occupy a contiguous
# chunk of a_scales_per_matrix entries in the 1D array.  The index for
# batch b is:  b * a_scales_per_matrix  +  (intra-matrix tiled offset).
# cuBLASLt stores scales in a non-trivial tiled layout.
# Use get_block_scale_offset to compute the correct index.
# Set scale for batch 0: row 5, cols 32-47
# Note, the elements in cols 32..47 are scaled with
# the same block scale factor (32..47 // 16 = 2)
a_scale[get_block_scale_offset((0, 5, 2), a_fp4, "NVFP4")] = 1.0
# get_block_scale_offset accepts operand or its shape and blocked dimension axis.
# batch 1: row 10, cols 0-15
a_scale[get_block_scale_offset((1, 10, 0), (batch_size, m, k), "NVFP4", axis=-1)] = 2.0

# B scales: unblocked dim = N, blocked dim = K -> per-matrix scales = N * K // 16.
# cuBLASLt stores scales in a non-trivial tiled layout.
# Use get_block_scale_offset to compute the correct index.
b_scales_per_matrix = (k // 16) * n
b_scale = torch.zeros(batch_size * b_scales_per_matrix, dtype=torch.float8_e4m3fn, device=device)
# batch 0: col 32, rows 32-47
b_scale[get_block_scale_offset((0, 2, 32), b_fp4, "NVFP4")] = 4.0
# batch 1: col 64, rows 0-15
b_scale[get_block_scale_offset((1, 0, 64), (batch_size, k, n), "NVFP4", axis=-2)] = 3.0

# Execute batched NVFP4 matmul.
alpha = 2.0
result = nvmath.linalg.advanced.matmul(
    a_fp4,
    b_fp4,
    alpha=alpha,
    quantization_scales={"a": a_scale, "b": b_scale},
    options={"result_type": nvmath.CudaDataType.CUDA_R_32F, "block_scaling": True},
)

# Verify results.
expected_0 = alpha * 16 * 1.0 * 4.0
expected_1 = alpha * 16 * 2.0 * 3.0
print(f"Batch 0: C[0, 5, 32]  = {result[0, 5, 32].item():.1f} (expected {expected_0:.1f})")
print(f"Batch 1: C[1, 10, 64] = {result[1, 10, 64].item():.1f} (expected {expected_1:.1f})")


# Now, let's run the same matmul, but request narrow-precision output.
# The return value is a tuple (result, aux) where aux["d_out_scale"] contains
# the scales used for output blocked scaling.
alpha = 2.0
result, aux = nvmath.linalg.advanced.matmul(
    a_fp4,
    b_fp4,
    alpha=alpha,
    quantization_scales={"a": a_scale, "b": b_scale},
    options={"result_type": nvmath.CudaDataType.CUDA_R_4F_E2M1, "block_scaling": True},
)

# Let's unpack the fp4 results.
result_unpacked = unpack_fp4(result, axis=-1)
print(f"Result unpacking: shape: {result.shape} -> {result_unpacked.shape}, dtype: {result.dtype} -> {result_unpacked.dtype}")
assert tuple(result_unpacked.shape) == (batch_size, m, n)

d_out_scale = aux["d_out_scale"]
# We get 1D tensor of scales
print(f"D_out scale shape: {d_out_scale.shape}, dtype: {d_out_scale.dtype}")

# We reshape and broadcast the scales to match result operand.
d_out_scale_expanded = expand_block_scale(d_out_scale, result, "NVFP4")
print(f"D_out scale expanded shape: {d_out_scale_expanded.shape}, dtype: {d_out_scale_expanded.dtype}")

# It's possible to pass all the meta-data
# (operand shape, blocked axis and mxfp type explicitly)
# instead of passing the result operand itself.
expanded_explicitly = expand_block_scale(d_out_scale, (batch_size, m, n), "NVFP4", axis=-1)
assert torch.equal(d_out_scale_expanded, expanded_explicitly)

# Verify results.
result_unpacked *= d_out_scale_expanded.type(torch.float32)
expected_0 = alpha * 16 * 1.0 * 4.0
expected_1 = alpha * 16 * 2.0 * 3.0
print(f"Batch 0: C[0, 5, 32]  = {result_unpacked[0, 5, 32].item():.1f} (expected {expected_0:.1f})")
print(f"Batch 1: C[1, 10, 64] = {result_unpacked[1, 10, 64].item():.1f} (expected {expected_1:.1f})")
