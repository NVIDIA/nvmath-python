# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates GEMM on CuPy ndarrays.

GEMM (General Matrix Multiply) is defined as:
alpha * A @ B + beta * C
where `@` denotes matrix multiplication.
"""

import cupy as cp

import nvmath

# Prepare sample input data.
m, n, k = 64, 128, 256
a = cp.random.rand(m, k)
b = cp.random.rand(k, n)
c = cp.random.rand(m, n)
alpha = 0.45
beta = 0.67

# Perform the GEMM.
result = nvmath.linalg.advanced.matmul(a, b, c=c, alpha=alpha, beta=beta)

# Synchronize the default stream, since by default the execution is non-blocking for GPU
# operands.
cp.cuda.get_current_stream().synchronize()
