# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
NVFP4 matmul with sliced operands.

This is intentionally a contrived example intended to illustrate
how to slice FP4 matrices and pass the resulting views to
nvmath.linalg.advanced.matmul, with particular attention to how FP4 byte
packing affects the slice indices.

NVFP4 requires cuBLAS 12.8+, torch >= 2.9, and compute capability >= 10.0.
"""

import torch

import nvmath
from nvmath.linalg.advanced.helpers.matmul import quantize_to_fp4

device = "cuda"

# Setup: create two large FP4 matrices
#
# In practice FP4 tensors come from a quantized model or a prior computation.
# Here we create them from float32 ones using the helper function quantize_to_fp4
# so the result is easy to verify.
# For A, we use axis=-1 because the nvmath API requires A in row-major layout,
# so packing is done row-wise, yielding: float (M, K) --> FP4 (M, K // 2).
# For B, we use axis=-2 because B must be column-major, so packing is
# done column-wise, yielding: float (K, N) --> FP4 (K // 2, N).
M_big, K_big, N_big = 256, 128, 256
a_big = quantize_to_fp4(torch.ones(M_big, K_big, device=device, dtype=torch.float32), axis=-1)
b_big = quantize_to_fp4(torch.ones(K_big, N_big, device=device, dtype=torch.float32), axis=-2)

# Define the logical matmul we want: C(m, n) = A(m, k) @ B(k, n)
# Dimensions must satisfy nvmath requirements: multiples of 128 for unblocked
# scaling, 64 for block-scaled.  Offsets must preserve 16-byte alignment.
m, k, n = 128, 64, 128
m_offset, k_offset, n_offset = 16, 32, 48

# Derive packed indices for slicing
# FP4 packs 2 values per byte along the contiguous axis, so both A and B
# store the K dimension in packed form. k_off must be even in logical space
k_packed = k // 2  # 32
k_packed_offset = k_offset // 2  # 16

# Slice A: shape (M_big, K_big//2), row-major.
# A is packed along axis=-1, so only column indices need halving.
a = a_big[m_offset : m_offset + m, k_packed_offset : k_packed_offset + k_packed]

# Slice B: shape (K_big//2, N_big), col-major.
# B is packed along axis=-2, so only row indices need halving.
b = b_big[k_packed_offset : k_packed_offset + k_packed, n_offset : n_offset + n]

# Block scales: one scale per 16 logical FP4 values along k.
a_scale = torch.ones(m * (k // 16), dtype=torch.float8_e4m3fn, device=device)
b_scale = torch.ones(n * (k // 16), dtype=torch.float8_e4m3fn, device=device)

# Matmul on the sliced operands
alpha = 2.0
result = nvmath.linalg.advanced.matmul(
    a,
    b,
    alpha=alpha,
    quantization_scales={"a": a_scale, "b": b_scale},
    options={"result_type": nvmath.CudaDataType.CUDA_R_32F, "block_scaling": True},
)

# Every element is 1, every scale is 1, so each output element is
#   alpha * k = 2.0 * 64 = 128.0
expected_value = alpha * k
expected = torch.full((m, n), expected_value, device=device, dtype=result.dtype)
assert torch.equal(result, expected), (
    f"Result mismatch.\n  Expected all {expected_value}, got min={result.min().item()}, max={result.max().item()}"
)
print(f"All {m}x{n} result elements equal {expected_value} as expected.")
