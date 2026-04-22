# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates how to use the get_block_scale_offset helper function
to modify individual scaling factors in MXFP8 matrix multiplication.

FP8 is only supported with cuBLAS 12.8 or newer and on devices with compute
capability 10.0 or higher.
"""

import torch

import nvmath
from nvmath.linalg.advanced.helpers.matmul import BlockScalingFormat

# Prepare sample input data
size = 256
a = torch.eye(size, device="cuda", dtype=torch.float8_e4m3fn)
b = torch.ones(size, size, device="cuda", dtype=torch.float8_e4m3fn).T

a_scale = nvmath.linalg.advanced.helpers.matmul.create_mxfp8_scale(a, 0)
b_scale = nvmath.linalg.advanced.helpers.matmul.create_mxfp8_scale(b, 0)

options = {"block_scaling": True, "result_type": nvmath.CudaDataType.CUDA_R_32F}

# Compute initial result with all scale factors set to 1
result = nvmath.linalg.advanced.matmul(a, b, quantization_scales={"a": a_scale, "b": b_scale}, options=options)
print("Initial result with all scale factors set to 1:")
print(result)

# Use get_block_scale_offset helper to modify the scale factor for the block
# for elements b[:32, 1] (first 32 elements in the second column).
# block_scaling_format can be a BlockScalingFormat enum or a plain string like "MXFP8".
offset = nvmath.linalg.advanced.helpers.matmul.get_block_scale_offset((0, 1), b, BlockScalingFormat.MXFP8)
# The same can also be called as a plain string: get_block_scale_offset((0, 1), b, "MXFP8")
b_scale[offset] += 4  # Increase the exponent by 4

# Compute result with modified scale factor
result2 = nvmath.linalg.advanced.matmul(a, b, quantization_scales={"a": a_scale, "b": b_scale}, options=options)
print("\nResult after modifying one scale factor:")
print(result2)
print(f"\nThe scale factor modification affected {(result2 != 1).sum().item()} elements in the block.")

# It is also possible to prepare unique scale factors as ND tensor
# and copy them to the required layout.

# Every consecutive 32 elements in B column have the same scale factor,
# so the scale matrix is 32x smaller in the first dimension.
b_scale_matrix = nvmath.linalg.advanced.helpers.matmul.create_mxfp8_scale(b, 0)
b_scale_matrix = b_scale_matrix.reshape(size // 32, size)
# Let's modify the same block as before.
# The [0, 1] block scale is applied to B[0:32, 1].
b_scale_matrix[0, 1] += 6
# And last block in the next column:
# The [-1, 2] block scale is applied to B[-32:, 2]
b_scale_matrix[-1, 2] += 5

# Copy the scale matrix to the required layout.
nvmath.linalg.advanced.helpers.matmul.to_block_scale(b_scale_matrix, b, "MXFP8", out=b_scale)

# Compute result with modified scale factor
result3 = nvmath.linalg.advanced.matmul(a, b, quantization_scales={"a": a_scale, "b": b_scale}, options=options)
print("\nResult after modifying two scale factors:")
print(result3)
print(f"\nThe scale factor modification affected {(result3 != 1).sum().item()} elements in two blocks.")
