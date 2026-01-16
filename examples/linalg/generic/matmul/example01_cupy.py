# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates basic matrix multiplication of CuPy arrays using the generic API.

nvmath-python supports multiple frameworks. The result of each operation is a tensor of the
same framework that was used to pass the inputs. It is also located on the GPU like the
inputs.
"""

import cupy as cp

import nvmath

# Prepare sample input data.
m, n, k = 123, 456, 789
a = cp.random.rand(m, k)
b = cp.random.rand(k, n)

# The execution happens on the GPU by default since the operands are on the GPU.

# Perform the multiplication.
result = nvmath.linalg.matmul(a, b)

print(cp.allclose(a @ b, result))
