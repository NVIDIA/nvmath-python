# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates using RELU epilog with FP8 matrix multiplication.

In FP8 operations, quantization scales must be provided for each tensor. These scales are
used to dequantize input operands and quantize the result. The RELU epilog is applied
after scaling but before final quantization.

FP8 is only supported with cuBLAS 12.8 or newer and on devices with compute
capability 8.9 or higher.
"""

import torch

import nvmath

# Prepare sample input data with some negative values
m, n, k = 64, 32, 48
a = (torch.rand(m, k, device="cuda") * 20 - 10).type(torch.float8_e4m3fn)
b = (torch.rand(n, k, device="cuda") * 20 - 10).type(torch.float8_e4m3fn).T

# Set quantization scales to keep values in range
scales = {"a": 1, "b": 1, "d": 0.1}

# First perform multiplication without RELU
result_no_relu = nvmath.linalg.advanced.matmul(a, b, quantization_scales=scales)

# Now perform multiplication with RELU epilog
result_with_relu = nvmath.linalg.advanced.matmul(
    a,
    b,
    epilog=nvmath.linalg.advanced.MatmulEpilog.RELU,
    quantization_scales=scales,
)

print("Result without RELU (notice negative values):")
print(result_no_relu)
print("\nResult with RELU (all values >= 0):")
print(result_with_relu)

# Verify that all values in the RELU result are non-negative
assert torch.all(result_with_relu.type(torch.float32) >= 0), "RELU result contains negative values!"
