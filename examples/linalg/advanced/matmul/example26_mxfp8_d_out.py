# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates automatic output scaling in MXFP8 mode.
When using MXFP8, D is automatically scaled during the matmul operation and the scale used
is returned as "d_out_scale". This scale can be used as input for subsequent matrix
multiplications (see mxfp8_chaining example) or applied to the result using a helper
function.

To use MXFP8, set the `block_scaling` option to True.

The layout of MXFP8 scales is complex. To simplify working with them, we provide helper
functions in `nvmath.linalg.advanced.helpers.matmul`. For more advanced operations on
MXFP8 scales, please refer to the cuBLAS documentation.

FP8 is only supported with cuBLAS 12.8 or newer and on devices with compute
capability 10.0 or higher.
"""

import torch

import nvmath

# Prepare sample input data. Note that N, M and K must be divisible by 128 for MXFP8.
m, n, k = 256, 256, 512

# Create matrix A with values increasing by row to demonstrate scaling with different
# magnitudes
a = torch.zeros(m, k, device="cuda", dtype=torch.float8_e4m3fn)
a[:] = torch.arange(m)[:, None]  # Each row will have progressively larger values
print("Matrix A:")
print(a)
print()

# cuBLAS requires B to be column-major, so we first create a row-major tensor and then
# transpose it.
b = torch.rand(m, k, device="cuda").type(torch.float8_e4m3fn).T
print("Matrix B:")
print(b)
print()

# Prepare quantization scales for A and B using the create_mxfp8_scale helper.
# Note: We don't set a scale for D since MXFP8 automatically scales the result to fit
# within the output type's dynamic range.
scales = {
    "a": nvmath.linalg.advanced.helpers.matmul.create_mxfp8_scale(a, -6),  # 2^-6 = 0.015625
    "b": nvmath.linalg.advanced.helpers.matmul.create_mxfp8_scale(b, 0),  # 2^0 = 1
}

# Enable block scaling
options = {
    "block_scaling": True,
}

# Perform the multiplication. The result is a tuple containing (result, aux).
# The aux dictionary contains "d_out_scale" - the scale used for the result.
result, aux = nvmath.linalg.advanced.matmul(a, b, quantization_scales=scales, options=options)

# Display results
print("Result (each block scaled to fit within float8_e4m3fn range):")
print(result)
print()

# Examine the D_OUT quantization scales
print(f"Auxiliary output contains these keys: {list(aux.keys())}")
print(
    f"D scale tensor shape: {aux['d_out_scale'].shape}, type: {aux['d_out_scale'].dtype}. "
    f"Contains {len(aux['d_out_scale'].unique())} unique scale factors."
)

# Apply the scale to get the actual result. Note: This helper function is for demonstration
# purposes and may use significant memory. For production use, set result_type to a
# non-FP8 type instead.
actual_result = nvmath.linalg.advanced.helpers.matmul.apply_mxfp8_scale(result, aux["d_out_scale"])
print("Final result (with quantization scales applied):")
print(actual_result)
