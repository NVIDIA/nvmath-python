# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates NVFP4 matmul for a simple
scenario where we create two matrices filled with zeros where only
one 16-element scale group is non-zero, so the result is analytically predictable.

Setup:
    - A: 128x64 matrix with 16 ones in row 5, K-group 2 (cols 32-47)
    - B: 64x128 matrix with 16 ones in col 32, K-group 2 (rows 32-47)
    - A scale = 1.0 for the active block, B scale = 4.0 for the active block

Matrix A (M=128 x K=64):

          K-group 0    K-group 1    K-group 2    K-group 3
          cols 0-15    cols 16-31   cols 32-47   cols 48-63
        +------------+------------+------------+------------+
row 0   |     0      |     0      |     0      |     0      |
row 1   |     0      |     0      |     0      |     0      |
  ...   |     0      |     0      |     0      |     0      |
row 5   |     0      |     0      |  1 1...1   |     0      |
  ...   |     0      |     0      |     0      |     0      |
row 127 |     0      |     0      |     0      |     0      |
        +------------+------------+------------+------------+

Scales for matrix A:

a_scale: 1D tensor, shape (512,), dtype float8_e4m3fn.
*Logically* 128 rows x 4 K-groups = 512 block scales, all zero except:

         K-grp 0    K-grp 1    K-grp 2    K-grp 3
        +----------+----------+----------+----------+
row 0   |    0     |    0     |    0     |    0     |
  ...   |    0     |    0     |    0     |    0     |
row 5   |    0     |    0     |   1.0    |    0     |
  ...   |    0     |    0     |    0     |    0     |
row 127 |    0     |    0     |    0     |    0     |
        +----------+----------+----------+----------+

*Physically* stored as a 1D array with cuBLASLt's tiled indexing (see code below).

Matrix B (K=64 x N=128):

                    col 0      col 31     col 32     col 127
                  +----------+----------+----------+----------+
K-group 0 row 0   |    0     |    0     |    0     |    0     |
          row 1   |    0     |    0     |    0     |    0     |
            ...   |    0     |    0     |    0     |    0     |
          row 15  |    0     |    0     |    0     |    0     |
                  +----------+----------+----------+----------+
K-group 1 row 16  |    0     |    0     |    0     |    0     |
            ...   |    0     |    0     |    0     |    0     |
          row 31  |    0     |    0     |    0     |    0     |
                  +----------+----------+----------+----------+
K-group 2 row 32  |    0     |    0     |    1     |    0     |
          row 33  |    0     |    0     |    1     |    0     |
            ...   |    0     |    0     |    1     |    0     |
          row 47  |    0     |    0     |    1     |    0     |
                  +----------+----------+----------+----------+
K-group 3 row 48  |    0     |    0     |    0     |    0     |
            ...   |    0     |    0     |    0     |    0     |
          row 63  |    0     |    0     |    0     |    0     |
                  +----------+----------+----------+----------+

Scales for matrix B:

b_scale: 1D tensor, shape (512,), dtype float8_e4m3fn.
*Logically* 128 cols x 4 K-groups = 512 block scales, all zero except:

          K-grp 0    K-grp 1    K-grp 2    K-grp 3
         +----------+----------+----------+----------+
col 0    |    0     |    0     |    0     |    0     |
  ...    |    0     |    0     |    0     |    0     |
col 32   |    0     |    0     |   4.0    |    0     |
  ...    |    0     |    0     |    0     |    0     |
col 127  |    0     |    0     |    0     |    0     |
         +----------+----------+----------+----------+

*Physically* stored as a 1D array with cuBLASLt's tiled indexing (see code below).

With scales applied: C[5, 32] = alpha * 16 * a_scale * b_scale.

NVFP4 requires cuBLAS 12.8+, torch >= 2.9, and compute capability >= 10.0.
"""

import torch

import nvmath
from nvmath.linalg.advanced.helpers.matmul import (
    BlockScalingFormat,
    get_block_scale_offset,
    quantize_to_fp4,
)

# NVFP4 matmul dimensions: m, n are the outer dimensions and
# must be multiples of 128; k is the contracting dimension
# and must be a multiple of 64.
m, k, n = 128, 64, 128
device = "cuda"

# A matrix: logical shape (m, k). Must be packed row-wise because k is the
# contracting dimension; cuBLASLt for FP4 internally uses A^T, so packing along k
# yields physical shape (m, k//2). One scale per 16-element block along k for
# each row -> a_scale shape (m * k//16,).
a_float = torch.zeros(m, k, device=device, dtype=torch.float32)
a_float[5, 32:48] = 1.0  # row 5, K-group 2 (cols 32-47)
a_fp4 = quantize_to_fp4(a_float, axis=-1)

# B matrix: logical shape (k, n). Must be packed column-wise because k is
# the contracting dimension; cuBLASLt. One scale per 16-element block
# along k for each column -> b_scale shape (n * k//16,).
b_float = torch.zeros(k, n, device=device, dtype=torch.float32)
b_float[32:48, 32] = 1.0  # col 32, K-group 2 (rows 32-47)
b_fp4 = quantize_to_fp4(b_float, axis=-2)

# Block scales: shape (outer_dim * k//16,), dtype float8_e4m3fn.
# cuBLASLt stores scales in a non-trivial 128x4 tiled layout.
# Use get_block_scale_offset to compute the correct index.
a_scale = torch.zeros(m * (k // 16), dtype=torch.float8_e4m3fn, device=device)
# We're setting the scale for a[5, 32]..a[5, 47] elements.
# For the blocked axis we pass K-group index (2 = 32..47 // 16).
# block_scaling_format can be a BlockScalingFormat enum or a plain string like "NVFP4".
offset = get_block_scale_offset((5, 2), (m, k), BlockScalingFormat.NVFP4, axis=-1)
# Equivalent: get_block_scale_offset((5, 2), (m, k), "NVFP4", axis=-1)
a_scale[offset] = 1.0
# Instead of passing operand shape and axis to the helper,
# we can just pass the operand itself,
# and the helper will infer the shape and axis from it.
assert offset == get_block_scale_offset((5, 2), a_fp4, "NVFP4")

b_scale = torch.zeros(n * (k // 16), dtype=torch.float8_e4m3fn, device=device)
# We're setting the scale for b[32, 32]..b[47, 32] elements.
# For the blocked axis we pass K-group index (2 = 32..47 // 2).
offset = get_block_scale_offset((2, 32), (k, n), "NVFP4", axis=-2)
b_scale[offset] = 4.0
assert offset == get_block_scale_offset((2, 32), b_fp4, "NVFP4")

alpha = 2.0
result = nvmath.linalg.advanced.matmul(
    a_fp4,
    b_fp4,
    alpha=alpha,
    quantization_scales={"a": a_scale, "b": b_scale},
    options={"result_type": nvmath.CudaDataType.CUDA_R_32F, "block_scaling": True},
)

expected_value = 16 * alpha * 1.0 * 4.0
print(f"Result C[5, 32] = {result[5, 32].item():.1f} (expected {expected_value})")
