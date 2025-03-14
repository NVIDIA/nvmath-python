# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates how to use the get_mxfp8_scale_offset helper function to modify
individual scaling factors in MXFP8 matrix multiplication.

FP8 is only supported with cuBLAS 12.8 or newer and on devices with compute
capability 10.0 or higher.
"""

import torch

import nvmath

# Prepare sample input data
size = 256
a = torch.eye(size, device="cuda", dtype=torch.float8_e4m3fn)
b = torch.ones(size, size, device="cuda", dtype=torch.float8_e4m3fn).T

a_scale = nvmath.linalg.advanced.helpers.matmul.create_mxfp8_scale(a, 0)
b_scale = nvmath.linalg.advanced.helpers.matmul.create_mxfp8_scale(a, 0)

options = {"block_scaling": True, "result_type": nvmath.CudaDataType.CUDA_R_32F}

# Compute initial result with all scale factors set to 1
result = nvmath.linalg.advanced.matmul(a, b, quantization_scales={"a": a_scale, "b": b_scale}, options=options)
print("Initial result with all scale factors set to 1:")
print(result)

# Use get_mxfp8_scale_offset helper to modify the scale factor for the block containing
# position (2, 1)
offset = nvmath.linalg.advanced.helpers.matmul.get_mxfp8_scale_offset(b, (2, 1))
b_scale[offset] += 4  # Increase the exponent by 4

# Compute result with modified scale factor
result2 = nvmath.linalg.advanced.matmul(a, b, quantization_scales={"a": a_scale, "b": b_scale}, options=options)
print("\nResult after modifying one scale factor:")
print(result2)
print(f"\nThe scale factor modification affected {(result2 != 1).sum().item()} elements in the block.")
