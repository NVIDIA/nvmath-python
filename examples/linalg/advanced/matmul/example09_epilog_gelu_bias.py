# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates usage of epilogs.

Epilogs allow you to execute extra computations after the matrix multiplication in a single
fused kernel. In this example we'll use the GELU_BIAS epilog, which adds bias to the result
and applies the GELU function.
"""

import cupy as cp

import nvmath

# Prepare sample input data.
m, n, k = 64, 128, 256
a = cp.random.rand(m, k)
b = cp.random.rand(k, n)
bias = cp.random.rand(m, 1)

# Perform the multiplication with GELU_BIAS epilog.
epilog = nvmath.linalg.advanced.MatmulEpilog.GELU_BIAS
result = nvmath.linalg.advanced.matmul(a, b, epilog=epilog, epilog_inputs={"bias": bias})

# Synchronize the default stream, since by default the execution is non-blocking for GPU
# operands.
cp.cuda.get_current_stream().synchronize()
print(f"Inputs were of types {type(a)} and {type(b)}, the bias type is {type(bias)}, and the result is of type {type(result)}.")
