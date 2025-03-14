# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates basic matrix multiplication of FP8 tensors using MXFP8
(microscaled FP8) quantization scales.

Key differences from FP8:
- MXFP8 scales are applied to each 32-element block of the tensors, rather than using a
  single tensor-wide scaling factor. This allows more fine-grained control over scaling
  and improves the accuracy of MXFP8 operations.
- MXFP8 scales are uint8 numbers in exponent-only format, representing values of the form
  2^n, where n is an integer between -127 and 128.
- In MXFP8 mode, if D is FP8, it is scaled automatically during the matmul operation and
  the quantization scales used are returned as "d_out_scale". This is covered in the next
  example.

To use MXFP8, set the `block_scaling` option to True.

The layout of the quantization scales is relatively complex. To facilitate working with
MXFP8, we provide helper functions in `nvmath.linalg.advanced.helpers.matmul`.

FP8 is only supported with cuBLAS 12.8 or newer and on devices with compute
capability 10.0 or higher.
"""

import torch

import nvmath

# Prepare sample input data. Note that N, M and K must be divisible by 128 for MXFP8.
# cuBLAS requires B to be column-major, so we first create a row-major tensor and then
# transpose it.
a = torch.eye(256, device="cuda", dtype=torch.float8_e4m3fn)  # A is an identity matrix
b = torch.ones((256, 256), device="cuda", dtype=torch.float8_e4m3fn).T  # B is filled with ones

# Prepare quantization scales for A and B using the `create_mxfp8_scale` helper.
# While MXFP8 allows different scales for different blocks in A and B,
# this helper creates uniform scaling across all blocks.
# For more advanced scale configurations, see the cuBLAS documentation and
# the `get_mxfp8_scale_offset` helper.
scales = {
    "a": nvmath.linalg.advanced.helpers.matmul.create_mxfp8_scale(a, -1),  # 2^-1 = 0.5
    "b": nvmath.linalg.advanced.helpers.matmul.create_mxfp8_scale(b, 3),  # 2^3 = 8
}

# Enable block scaling by setting the `block_scaling` option to True. For simplicity, we
# request FP16 output. For FP8 output scaling, see the mxfp8_d_out_scale example.
options = {"block_scaling": True, "result_type": nvmath.CudaDataType.CUDA_R_16F}

# Perform the multiplication. The result is a tuple (result, aux), where aux
# contains the "d_out_scale" key with the scale used for the result.
result = nvmath.linalg.advanced.matmul(a, b, quantization_scales=scales, options=options)

# Compute reference result without scaling
reference = a.type(torch.float16) @ b.type(torch.float16)
print(f"Reference result (without scaling):\n{reference}")

# Print the result with scaling applied
print(f"Result with scaling (A scaled by 0.5, B scaled by 8):\n{result}")
