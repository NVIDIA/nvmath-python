# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates basic matrix multiplication of CuPy arrays using the generic API,
using sliced operands as input. It should be noted that not all non-dense layouts are
supported.
"""

import cupy as cp

import nvmath

# Prepare sample input data.
m, n, k = 4, 6, 8
a = cp.random.rand(m, k)[::2]
b = cp.random.rand(k, n)

# The execution happens on the GPU by default since the operands are on the GPU.

# Perform the multiplication.
result = nvmath.linalg.matmul(a, b)

print(cp.allclose(a @ b, result))
