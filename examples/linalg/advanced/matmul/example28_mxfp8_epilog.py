# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
In this example, we perform MXFP8 matrix multiplication with ReLU activation.

FP8 is only supported with cuBLAS 12.8 or newer and on devices with compute
capability 10.0 or higher.
"""

import torch

import nvmath

# Prepare sample input data with dimensions m=256, n=256, k=512
m, n, k = 256, 256, 512
a = torch.randn(m, k, device="cuda").type(torch.float8_e4m3fn)
b = torch.randn(n, k, device="cuda").type(torch.float8_e4m3fn).T

scales = {
    "a": nvmath.linalg.advanced.helpers.matmul.create_mxfp8_scale(a, 0),  # Scale factor 2^0 = 1
    "b": nvmath.linalg.advanced.helpers.matmul.create_mxfp8_scale(b, 0),  # Scale factor 2^0 = 1
}

options = {
    "block_scaling": True,
}

result, aux = nvmath.linalg.advanced.matmul(
    a, b, quantization_scales=scales, options=options, epilog=nvmath.linalg.advanced.MatmulEpilog.RELU
)

# Display the results
print("Result after applying D_OUT scales:")
print(nvmath.linalg.advanced.helpers.matmul.apply_mxfp8_scale(result, aux["d_out_scale"]))
print("All values are non-negative due to the ReLU activation.")
